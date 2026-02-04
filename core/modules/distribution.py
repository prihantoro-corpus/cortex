import duckdb
import pandas as pd
import re

def calculate_distribution(corpus_db_path, raw_target_input, xml_where_clause="", xml_params=[]):
    """
    Calculates the frequency of a search term across 100 segments of the corpus.
    Reuses search logic from concordance module.
    """
    if not corpus_db_path:
        return pd.DataFrame(), {}

    try:
        con = duckdb.connect(corpus_db_path, read_only=True)
        
        # 1. Robust Query Tokenization
        query_pattern = r'<[^>]+>|[^\s]+'
        search_terms = re.findall(query_pattern, raw_target_input)
        
        # Check raw mode (copied from concordance.py)
        is_raw_mode = True
        try:
            cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
            if 'pos' in cols:
                raw_count = con.execute("SELECT count(*) FROM corpus WHERE pos LIKE '##%'").fetchone()[0]
                total_rows = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
                is_raw_mode = (raw_count / total_rows) > 0.99
        except: pass
            
        def parse_term(term):
            # XML Tag Check
            xml_tag_match = re.match(r'<(\w+)(?:\s+(.+?))?>', term, re.IGNORECASE)
            if xml_tag_match:
                tag_name = xml_tag_match.group(1).lower()
                attrs_str = xml_tag_match.group(2)
                attrs = {}
                if attrs_str:
                    attr_pattern = r'(\w+)=(["\'])([^"\']*)\2'
                    for match in re.finditer(attr_pattern, attrs_str):
                        attrs[match.group(1).lower()] = match.group(3)
                return {'type': 'xml_tag', 'tag': tag_name, 'attrs': attrs}
            
            lemma_match = re.search(r"\[(.*?)\]", term)
            if lemma_match: return {'type': 'lemma', 'val': lemma_match.group(1).strip().lower()}
            # Combined Token_POS Check (e.g. light_V*)
            if '_' in term and not term.startswith('_') and not lemma_match:
                parts = term.rsplit('_', 1)
                if len(parts) == 2 and parts[1]:
                    return {'type': 'token_pos', 'token': parts[0].lower(), 'pos': parts[1]}

            pos_match = re.search(r"\_([A-Za-z0-9\*|\\-]+)", term)
            if pos_match: return {'type': 'pos', 'val': pos_match.group(1).strip()}
            return {'type': 'word', 'val': term.lower()}

        search_components = [parse_term(term) for term in search_terms]

        # 2. Build Query with Dynamic Lengths
        token_concat_parts = []
        current_offset_exprs = []
        query_joins = "" # Initialize query_joins here
        
        for k, comp in enumerate(search_components):
            alias = f"c{k}"
            
            # For match_token display, we need to handle multi-token tags
            if comp['type'] == 'xml_tag':
                tag_name = comp['tag']
                token_expr = f"(SELECT string_agg(token, ' ') FROM corpus c_sub WHERE c_sub.id BETWEEN {alias}.id AND {alias}.id + COALESCE({alias}.{tag_name}_len, 1) - 1)"
                current_offset_exprs.append(f"COALESCE({alias}.{tag_name}_len, 1)")
            else:
                token_expr = f"{alias}.token"
                current_offset_exprs.append("1")
                
            token_concat_parts.append(token_expr)
            
            if k > 0:
                prev_alias = f"c{k-1}"
                prev_offset = current_offset_exprs[k-1]
                query_joins += f" JOIN corpus {alias} ON {alias}.id = {prev_alias}.id + {prev_offset} "
        
        match_token_expr = "(" + " || ' ' || ".join(token_concat_parts) + ")" if len(token_concat_parts) > 1 else token_concat_parts[0]
        
        query_select = f"SELECT c0.id, {match_token_expr} as match_token FROM corpus c0"
        query_where = []
        query_params = []
        
        for k, comp in enumerate(search_components):
            alias = f"c{k}"
            
            if comp['type'] == 'token_pos':
                 t_val = comp['token']
                 p_val = comp['pos']
                 
                 # Token SQL
                 if '*' in t_val:
                     query_where.append(f"regexp_matches({alias}._token_low, ?)")
                     query_params.append('^' + re.escape(t_val).replace(r'\*', '.*') + '$')
                 else:
                     query_where.append(f"{alias}._token_low = ?")
                     query_params.append(t_val)
                 
                 # POS SQL
                 if not is_raw_mode: 
                     if '|' in p_val or '*' in p_val:
                        pats = [p.strip() for p in p_val.split('|') if p.strip()]
                        regex = "^(" + "|".join([re.escape(p).replace(r'\*', '.*') for p in pats]) + ")$"
                        query_where.append(f"regexp_matches({alias}.pos, ?)")
                        query_params.append(regex)
                     else:
                        query_where.append(f"{alias}.pos = ?")
                        query_params.append(p_val)

            elif comp['type'] == 'xml_tag':
                tag_name = comp['tag']
                attrs = comp['attrs']
                
                # Check if it's the START of the tag instance
                tag_start_col = f"in_{tag_name}_start"
                query_where.append(f"{alias}.{tag_start_col} = TRUE")
                
                # Add attribute filters
                for attr_key, attr_val in attrs.items():
                    attr_col = f"{tag_name}_{attr_key}"
                    if '*' in attr_val:
                        regex_pat = '^' + re.escape(attr_val).replace(r'\*', '.*') + '$'
                        query_where.append(f"regexp_matches({alias}.{attr_col}, ?)")
                        query_params.append(regex_pat)
                    else:
                        query_where.append(f"{alias}.{attr_col} = ?")
                        query_params.append(attr_val)

            elif comp['type'] == 'word':
                val = comp['val']
                if '*' in val:
                    regex_pat = '^' + re.escape(val).replace(r'\*', '.*') + '$'
                    query_where.append(f"regexp_matches({alias}._token_low, ?)")
                    query_params.append(regex_pat)
                else:
                    query_where.append(f"{alias}._token_low = ?")
                    query_params.append(val)
            elif comp['type'] == 'lemma' and not is_raw_mode:
                val = comp['val']
                if '*' in val:
                    regex_pat = '^' + re.escape(val).replace(r'\*', '.*') + '$'
                    query_where.append(f"regexp_matches(lower({alias}.lemma), ?)")
                    query_params.append(regex_pat)
                else:
                    query_where.append(f"lower({alias}.lemma) = ?")
                    query_params.append(val)
            elif comp['type'] == 'pos' and not is_raw_mode:
                val = comp['val']
                if '|' in val or '*' in val:
                    pos_patterns = [p.strip() for p in val.split('|') if p.strip()]
                    full_regex = "^(" + "|".join([re.escape(p).replace(r'\*', '.*') for p in pos_patterns]) + ")$"
                    query_where.append(f"regexp_matches({alias}.pos, ?)")
                    query_params.append(full_regex)
                else:
                    query_where.append(f"{alias}.pos = ?")
                    query_params.append(val)

        # Assemble final WHERE
        final_query = query_select + query_joins
        if query_where:
            final_query += " WHERE " + " AND ".join(query_where)
            if xml_where_clause: final_query += xml_where_clause
        elif xml_where_clause:
            final_query += " WHERE " + xml_where_clause.strip()[4:]
        
        full_params = query_params + xml_params
        
        # Get all matching IDs
        df_matches = con.execute(final_query, full_params).fetch_df()
        
        if df_matches.empty:
            con.close()
            return pd.DataFrame(), {}

        match_ids = df_matches['id'].tolist()
        
        # Get total size of corpus (with filters)
        if xml_where_clause:
            total_tokens = con.execute(f"SELECT count(*) FROM corpus WHERE 1=1 {xml_where_clause}", xml_params).fetchone()[0]
        else:
            total_tokens = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
        
        if total_tokens == 0:
            con.close()
            return pd.DataFrame(), {}

        # 1. Divide into 100 segments
        num_segments = 100
        segment_size = total_tokens / num_segments
        
        # Initialize counts for each segment
        segment_counts = [0] * num_segments
        
        for mid in match_ids:
            idx = int(mid // segment_size)
            if idx >= num_segments:
                idx = num_segments - 1
            segment_counts[idx] += 1
            
        # 2. Metadata Attribute Distributions
        standard_cols = {'id', 'token', 'pos', 'lemma', '_token_low'} # Removed 'filename' and 'sent_id' from exclusion
        all_cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
        all_cols = [c[1] for c in all_cols_info]
        meta_cols = [c for c in all_cols if c not in standard_cols]
        # Exclude internal _len and _start columns from metadata display
        meta_cols = [c for c in meta_cols if not (c.endswith('_len') or c.endswith('_start') or c.endswith('_id'))]
        
        meta_dists = {}
        
        if meta_cols:
            # Re-fetch matches with metadata
            meta_select = ", ".join([f"c0.{c}" for c in meta_cols])
            final_query_with_meta = f"SELECT c0.id, {meta_select} FROM corpus c0"
            if query_joins: final_query_with_meta += query_joins
            if query_where:
                final_query_with_meta += " WHERE " + " AND ".join(query_where)
                if xml_where_clause: final_query_with_meta += xml_where_clause
            elif xml_where_clause:
                final_query_with_meta += " WHERE " + xml_where_clause.strip()[4:]
            
            df_matches_meta = con.execute(final_query_with_meta, full_params).fetch_df()
            
            for col in meta_cols:
                # 1. Count matches per value
                counts = df_matches_meta[col].value_counts().to_dict()
                
                # 2. Get sub-corpus size for each value (to normalize)
                baseline_query = f"SELECT {col}, count(*) as total FROM corpus WHERE 1=1"
                if xml_where_clause: baseline_query += xml_where_clause
                baseline_query += f" GROUP BY {col}"
                
                df_baseline = con.execute(baseline_query, xml_params).fetch_df()
                baseline_dict = dict(zip(df_baseline[col], df_baseline['total']))
                
                rows = []
                for val, count in counts.items():
                    if val is None or str(val).strip() == "": continue
                    total = baseline_dict.get(val, 0)
                    if total == 0: continue
                    pmw = (count / total) * 1_000_000
                    rows.append({'Value': val, 'Absolute': count, 'PMW': pmw})
                
                if rows:
                    df_attr = pd.DataFrame(rows)
                    max_pmw = df_attr['PMW'].max()
                    if max_pmw > 0:
                        df_attr['Relative (%)'] = (df_attr['PMW'] / max_pmw) * 100
                    else:
                        df_attr['Relative (%)'] = 0
                    
                    df_attr = df_attr.sort_values('Relative (%)', ascending=False)
                    meta_dists[col] = df_attr

        con.close()
        
        dist_df = pd.DataFrame({
            'Corpus Position (%)': [i for i in range(1, num_segments + 1)],
            'Absolute Frequency': segment_counts,
            'Relative Frequency (Density %)': [(c / segment_size) * 100 for c in segment_counts] if segment_size > 0 else [0]*num_segments
        })
        
        # Add Relative to Peak (%) normalization
        max_abs = dist_df['Absolute Frequency'].max()
        if max_abs > 0:
            dist_df['Relative to Peak (%)'] = (dist_df['Absolute Frequency'] / max_abs) * 100
        else:
            dist_df['Relative to Peak (%)'] = 0
            
        return dist_df, meta_dists

    except Exception as e:
        print(f"Error in calculate_distribution: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {}
