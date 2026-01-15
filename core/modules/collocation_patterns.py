"""
Collocation Pattern Matching Module

This module provides pattern-based clustering of collocates based on their
positional relationship to the node word in concordance lines.

Pattern Syntax:
- <> : the node word
- # : the collocate
- * : optional token (any word)
- + : required token (any word)
- direct_token : specific token (obligatory)
- (token) : optional specific token
- [lemma] : token must be from specified lemma
- _TAG : token must have specified POS tag
- (_TAG) : optional POS tag constraint
- ([lemma]) : optional lemma constraint
"""

import re
import duckdb
import pandas as pd
from typing import List, Dict, Tuple, Optional


def parse_pattern_definitions(pattern_text: str) -> Tuple[List[Dict], List[str]]:
    """
    Parse user input pattern definitions.
    
    Args:
        pattern_text: Multi-line string with format "label : pattern"
        
    Returns:
        Tuple of (parsed_patterns, errors)
        - parsed_patterns: List of dicts with 'label', 'pattern_str', 'parsed_tokens'
        - errors: List of error messages
    """
    patterns = []
    errors = []
    
    if not pattern_text or not pattern_text.strip():
        return patterns, errors
    
    lines = pattern_text.strip().split('\n')
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):  # Skip empty lines and comments
            continue
            
        # Parse "label : pattern" format
        if ':' not in line:
            errors.append(f"Line {line_num}: Missing ':' separator. Format should be 'label : pattern'")
            continue
            
        parts = line.split(':', 1)
        if len(parts) != 2:
            errors.append(f"Line {line_num}: Invalid format. Use 'label : pattern'")
            continue
            
        label = parts[0].strip()
        pattern_str = parts[1].strip()
        
        if not label:
            errors.append(f"Line {line_num}: Empty label")
            continue
        if not pattern_str:
            errors.append(f"Line {line_num}: Empty pattern")
            continue
            
        # Parse the pattern tokens
        parsed_tokens, parse_errors = parse_pattern_tokens(pattern_str)
        
        if parse_errors:
            for err in parse_errors:
                errors.append(f"Line {line_num} ({label}): {err}")
            continue
            
        patterns.append({
            'label': label,
            'pattern_str': pattern_str,
            'parsed_tokens': parsed_tokens
        })
    
    return patterns, errors


def parse_pattern_tokens(pattern_str: str) -> Tuple[List[Dict], List[str]]:
    """
    Parse a pattern string into a sequence of token specifications.
    
    Returns:
        Tuple of (tokens, errors)
        - tokens: List of dicts with 'type', 'value', 'optional', 'constraint_type', 'constraint_value'
        - errors: List of error messages
    """
    tokens = []
    errors = []
    
    # Split pattern by spaces
    parts = pattern_str.split()
    
    node_count = 0
    collocate_count = 0
    
    for i, part in enumerate(parts):
        token = {
            'type': None,
            'value': None,
            'optional': False,
            'constraint_type': None,  # 'lemma', 'pos', or None
            'constraint_value': None
        }
        
        # Check for node
        if part == '<>':
            token['type'] = 'node'
            node_count += 1
            tokens.append(token)
            continue
            
        # Check for collocate
        if part == '#':
            token['type'] = 'collocate'
            collocate_count += 1
            tokens.append(token)
            continue
            
        # Check for wildcards
        if part == '*':
            token['type'] = 'wildcard_optional'
            tokens.append(token)
            continue
            
        if part == '+':
            token['type'] = 'wildcard_required'
            tokens.append(token)
            continue
            
        # Check for optional constraints: (token), (_TAG), ([lemma])
        optional_match = re.match(r'^\((.+)\)$', part)
        if optional_match:
            inner = optional_match.group(1)
            token['optional'] = True
            
            # Check if it's a POS tag
            if inner.startswith('_'):
                token['type'] = 'constraint'
                token['constraint_type'] = 'pos'
                token['constraint_value'] = inner[1:]  # Remove underscore
            # Check if it's a lemma
            elif inner.startswith('[') and inner.endswith(']'):
                token['type'] = 'constraint'
                token['constraint_type'] = 'lemma'
                token['constraint_value'] = inner[1:-1].lower()
            # Direct token
            else:
                token['type'] = 'token'
                token['value'] = inner.lower()
                
            tokens.append(token)
            continue
            
        # Check for POS tag: _TAG
        if part.startswith('_'):
            token['type'] = 'constraint'
            token['constraint_type'] = 'pos'
            token['constraint_value'] = part[1:]
            tokens.append(token)
            continue
            
        # Check for lemma: [lemma]
        lemma_match = re.match(r'^\[(.+)\]$', part)
        if lemma_match:
            token['type'] = 'constraint'
            token['constraint_type'] = 'lemma'
            token['constraint_value'] = lemma_match.group(1).lower()
            tokens.append(token)
            continue
            
        # Otherwise, it's a direct token
        token['type'] = 'token'
        token['value'] = part.lower()
        tokens.append(token)
    
    # Validation
    if node_count == 0:
        errors.append("Pattern must contain exactly one node symbol '<>'")
    elif node_count > 1:
        errors.append(f"Pattern contains {node_count} node symbols, but only one '<>' is allowed")
        
    if collocate_count == 0:
        errors.append("Pattern must contain exactly one collocate symbol '#'")
    elif collocate_count > 1:
        errors.append(f"Pattern contains {collocate_count} collocate symbols, but only one '#' is allowed")
    
    return tokens, errors


def match_pattern_in_concordance(
    pattern_tokens: List[Dict],
    concordance_tokens: List[Dict],
    node_word: str,
    collocate_word: str
) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    Check if a concordance line matches a pattern and return matching indices.
    """
    # Helper to match node/collocate with potential constraints
    def _matches(t_dict, val):
        if not val: return False
        l_token = t_dict['token'].lower()
        l_val = val.lower()
        
        # Handle lemma syntax: [lemma]
        if l_val.startswith('[') and l_val.endswith(']'):
            lemma = l_val[1:-1]
            return t_dict.get('lemma', '').lower() == lemma
        # Handle POS syntax: _TAG
        if l_val.startswith('_'):
            tag = val[1:]
            return t_dict.get('pos', '') == tag
        # Regular token
        return l_token == l_val

    # Find node positions
    node_positions = [i for i, t in enumerate(concordance_tokens) if _matches(t, node_word)]
    
    if not node_positions:
        return False, None, None
    
    # Try each node position
    for node_pos in node_positions:
        # Find collocate positions
        collocate_positions = [i for i, t in enumerate(concordance_tokens) if _matches(t, collocate_word)]
        
        for coll_pos in collocate_positions:
            # Try to match pattern starting from node position
            if _match_pattern_from_position(pattern_tokens, concordance_tokens, 
                                           node_pos, coll_pos):
                return True, node_pos, coll_pos
    
    return False, None, None


def _match_pattern_from_position(
    pattern_tokens: List[Dict],
    concordance_tokens: List[Dict],
    node_pos: int,
    collocate_pos: int
) -> bool:
    """
    Helper function to match pattern from a specific node position.
    This validates that the concordance matches the pattern's positional structure.
    """
    # Find node and collocate indices in pattern
    node_pattern_idx = next((i for i, t in enumerate(pattern_tokens) if t['type'] == 'node'), None)
    collocate_pattern_idx = next((i for i, t in enumerate(pattern_tokens) if t['type'] == 'collocate'), None)
    
    if node_pattern_idx is None or collocate_pattern_idx is None:
        return False
        
    # Identifiers for anchors
    if collocate_pattern_idx < node_pattern_idx:
        first_p_idx, second_p_idx = collocate_pattern_idx, node_pattern_idx
        first_c_pos, second_c_pos = collocate_pos, node_pos
    else:
        first_p_idx, second_p_idx = node_pattern_idx, collocate_pattern_idx
        first_c_pos, second_c_pos = node_pos, collocate_pos

    # 1. Validate BEFORE first anchor
    before_p = pattern_tokens[:first_p_idx]
    if before_p:
        # We need to match before_p as a SUFFIX of the tokens preceding the first anchor
        # To do this easily: reverse both before_p and reversed(concordance[:first_c_pos])
        # then match as a prefix.
        rev_before_p = []
        for p in reversed(before_p):
            # Swap wildcard types for reverse matching if needed? 
            # Actually wildcard_optional and wildcard_required are symmetrical.
            rev_before_p.append(p)
            
        rev_conc_before = list(reversed(concordance_tokens[:first_c_pos]))
        if not _match_recursive(rev_before_p, rev_conc_before, 0, 0, require_full_match=False):
            return False

    # 2. Validate BETWEEN anchors
    between_p = pattern_tokens[first_p_idx + 1:second_p_idx]
    conc_between = concordance_tokens[first_c_pos + 1:second_c_pos]
    if not _match_recursive(between_p, conc_between, 0, 0, require_full_match=True):
        return False

    # 3. Validate AFTER second anchor
    after_p = pattern_tokens[second_p_idx + 1:]
    if after_p:
        conc_after = concordance_tokens[second_c_pos + 1:]
        if not _match_recursive(after_p, conc_after, 0, 0, require_full_match=False):
            return False

    return True


def _match_token_sequence(pattern_tokens: List[Dict], conc_tokens: List[Dict]) -> bool:
    """
    Match a sequence of pattern tokens against concordance tokens.
    Handles wildcards, optional tokens, and constraints.
    """
    if not pattern_tokens:
        # No pattern tokens means we expect no concordance tokens (immediate adjacency)
        return len(conc_tokens) == 0
    
    # Use dynamic programming / backtracking to match
    return _match_recursive(pattern_tokens, conc_tokens, 0, 0)


def _match_recursive(pattern_tokens: List[Dict], conc_tokens: List[Dict], p_idx: int, c_idx: int, require_full_match: bool = True) -> bool:
    """
    Recursively match pattern tokens against concordance tokens.
    """
    # Base cases
    if p_idx >= len(pattern_tokens):
        # All pattern tokens matched
        if require_full_match:
            # check if all concordance tokens consumed
            return c_idx >= len(conc_tokens)
        else:
            # Prefix matched, don't care about remaining concordance tokens
            return True
    
    p_token = pattern_tokens[p_idx]
    
    # If we've consumed all concordance tokens
    if c_idx >= len(conc_tokens):
        # Check if remaining pattern tokens are all optional
        remaining_optional = all(
            pt.get('optional', False) or pt['type'] == 'wildcard_optional'
            for pt in pattern_tokens[p_idx:]
        )
        return remaining_optional
    
    c_token = conc_tokens[c_idx]
    
    # Handle different pattern token types
    if p_token['type'] == 'wildcard_optional':
        # Try matching with 0 tokens (skip this wildcard)
        if _match_recursive(pattern_tokens, conc_tokens, p_idx + 1, c_idx, require_full_match):
            return True
        # Try matching with 1 token (consume one concordance token)
        if _match_recursive(pattern_tokens, conc_tokens, p_idx + 1, c_idx + 1, require_full_match):
            return True
        return False
    
    elif p_token['type'] == 'wildcard_required':
        # Must consume exactly 1 token
        return _match_recursive(pattern_tokens, conc_tokens, p_idx + 1, c_idx + 1, require_full_match)
    
    elif p_token['type'] == 'token':
        # Match specific token
        if c_token['token'].lower() == p_token['value']:
            return _match_recursive(pattern_tokens, conc_tokens, p_idx + 1, c_idx + 1, require_full_match)
        elif p_token.get('optional', False):
            # Optional token not matched, try skipping it
            return _match_recursive(pattern_tokens, conc_tokens, p_idx + 1, c_idx, require_full_match)
        else:
            return False
    
    elif p_token['type'] == 'constraint':
        # Match constraint (POS or lemma)
        matched = False
        if p_token['constraint_type'] == 'pos':
            matched = c_token.get('pos', '') == p_token['constraint_value']
        elif p_token['constraint_type'] == 'lemma':
            matched = c_token.get('lemma', '').lower() == p_token['constraint_value']
        
        if matched:
            return _match_recursive(pattern_tokens, conc_tokens, p_idx + 1, c_idx + 1, require_full_match)
        elif p_token.get('optional', False):
            # Optional constraint not matched, try skipping
            return _match_recursive(pattern_tokens, conc_tokens, p_idx + 1, c_idx, require_full_match)
        else:
            return False
    
    return False



def group_collocates_by_patterns(
    collocates_df: pd.DataFrame,
    patterns: List[Dict],
    corpus_db_path: str,
    node_word: str,
    window: int,
    max_collocates: int = 50,
    xml_where_clause: str = "",
    xml_params: List = []
) -> Dict[str, Dict]:
    """
    Group collocates by pattern matches using the "Concordance-First" approach.
    
    Returns:
        Dict mapping pattern labels to a dict with 'df' (DataFrame) and 'examples' (Dict collocate -> matching_line)
    """
    if collocates_df.empty or not patterns:
        return {}
    
    # 1. Fetch a significant sample of the node's concordance
    try:
        con = duckdb.connect(corpus_db_path, read_only=True)
        # Fetch up to 10,000 lines for comprehensive matching
        concordance_data = _fetch_node_concordance_sample(
            con, node_word, window, limit=10000, 
            xml_where_clause=xml_where_clause, xml_params=xml_params
        )
        con.close()
    except Exception as e:
        print(f"Error fetching node concordance: {e}")
        return {}
        
    if not concordance_data:
        return {}
        
    # 2. Limit to top N collocates for matching
    top_collocates = collocates_df.head(max_collocates).copy()
    collocates_list = top_collocates['Collocate'].tolist()
    
    # 3. Initialize result tracking
    # Mapping of label -> set of original indices
    pattern_matches = {p['label']: set() for p in patterns}
    # Mapping of label -> dict (collocate -> matching_line)
    pattern_examples = {p['label']: {} for p in patterns}
    
    # 4. Scan each concordance line
    for conc_line in concordance_data:
        # For each collocate in our top list
        for coll_idx, collocate in enumerate(collocates_list):
            original_idx = top_collocates.index[coll_idx]
            
            # Check each pattern
            for pattern in patterns:
                label = pattern['label']
                # Skip if this collocate already matched this pattern
                if original_idx in pattern_matches[label]:
                    continue
                    
                matched, n_idx, c_idx = match_pattern_in_concordance(
                    pattern['parsed_tokens'],
                    conc_line,
                    node_word,
                    collocate
                )
                if matched:
                    pattern_matches[label].add(original_idx)
                    pattern_examples[label][collocate] = (conc_line, n_idx, c_idx)
    
    # 5. Convert to final structure
    result = {}
    for label, matched_indices in pattern_matches.items():
        if matched_indices:
            sorted_indices = sorted(list(matched_indices))
            result[label] = {
                'df': collocates_df.loc[sorted_indices].copy(),
                'examples': pattern_examples[label]
            }
    
    return result


def _fetch_node_concordance_sample(
    con: duckdb.DuckDBPyConnection,
    node_query: str,
    window: int,
    limit: int = 1000,
    xml_where_clause: str = "",
    xml_params: List = []
) -> List[List[Dict]]:
    """
    Fetch a sample of concordance lines for the node.
    Handles [lemma] and _TAG syntax.
    """
    try:
        # Determine search criteria
        where_clause = ""
        param = ""
        ln_query = node_query.lower()
        
        if ln_query.startswith('[') and ln_query.endswith(']'):
            where_clause = "lower(lemma) = ?"
            param = ln_query[1:-1]
        elif ln_query.startswith('_'):
            where_clause = "pos = ?"
            param = node_query[1:]
        else:
            where_clause = "lower(token) = ?"
            param = ln_query
            
        search_query = f"SELECT id FROM corpus WHERE {where_clause} {xml_where_clause} LIMIT ?"
        node_ids = [r[0] for r in con.execute(search_query, [param] + xml_params + [limit]).fetchall()]
        
        if not node_ids:
            return []
            
        concordances = []
        for node_id in node_ids:
            window_query = """
            SELECT id, token, pos, lemma
            FROM corpus
            WHERE id BETWEEN ? AND ?
            ORDER BY id
            """
            window_tokens = con.execute(window_query, [node_id - window, node_id + window]).fetchall()
            
            token_list = [
                {'token': t[1], 'pos': t[2] if t[2] else '', 'lemma': t[3] if t[3] else ''}
                for t in window_tokens
            ]
            concordances.append(token_list)
            
        return concordances
    except Exception as e:
        print(f"Error in _fetch_node_concordance_sample: {e}")
        return []

