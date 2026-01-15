import duckdb
import pandas as pd
import re
import string

# Common punctuation set matching typical corpus cleanup needs
PUNCTUATION = set(['.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '"', "'", '“', '”', '‘', '’', '...', '-', '–', '—'])

def generate_n_grams_v2(corpus_db_path, n_size, n_gram_filters, is_raw_mode, corpus_name, xml_where_clause="", xml_params=[], skip_punctuation=False, basis='Token', skip_pos_tags=[], positional_bases=None, negative_filter=[]):
    """
    Generates N-grams, applies positional filters (token, POS, lemma) using DuckDB SQL.
    Returns DataFrame: N-Gram | Frequency | Relative Frequency (per M)
    """
    if not corpus_db_path or n_size < 1:
        return pd.DataFrame()

    try:
        con = duckdb.connect(corpus_db_path, read_only=True)
        if xml_where_clause:
            total_tokens = con.execute(f"SELECT count(*) FROM corpus WHERE 1=1 {xml_where_clause}", xml_params).fetchone()[0]
        else:
            total_tokens = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
        
        if total_tokens < n_size:
            con.close()
            return pd.DataFrame()
        
        # --- Build Dynamic SQL Query ---
        
        cols = ["_token_low as t1"]
        if not is_raw_mode: cols.extend(["pos as p1", "lemma as l1"])
        
        # Build LEAD clauses for 2..N
        for i in range(2, n_size + 1):
            offset = i - 1
            cols.append(f"LEAD(_token_low, {offset}) OVER (ORDER BY id) as t{i}")
            if not is_raw_mode:
                cols.append(f"LEAD(pos, {offset}) OVER (ORDER BY id) as p{i}")
                cols.append(f"LEAD(lemma, {offset}) OVER (ORDER BY id) as l{i}")
        
        subquery_select = ", ".join(cols)
        
        # Conditions
        conditions = []
        params = []
        
        # Process Filters
        # n_gram_filters is dict: {'1': 'pattern', '2': 'pattern'}
        
        def parse_filter(pattern_str, idx, is_raw, default_basis='Token'):
            if not pattern_str: return [], []
            
            def standardize_wildcards(p):
                # Standardize *, % and _ to regex equivalents
                res = re.escape(p)
                res = res.replace(r'\*', '.*')
                res = res.replace(r'\%', '.*').replace('%', '.*')
                res = res.replace(r'\_', '.').replace('_', '.')
                return res

            # 0. Handle Multiselect (List of POS Tags)
            if isinstance(pattern_str, list):
                if not pattern_str or is_raw: return [], []
                # Use case-insensitive matching for POS tags (?i)
                full_regex = "(?i)^(" + "|".join([standardize_wildcards(p) for p in pattern_str]) + ")$"
                return [f"regexp_matches(p{idx}, ?)"], [full_regex]

            # Split by ';' to handle multiple constraints (e.g., negative filters)
            raw_patterns = [p.strip() for p in pattern_str.split(';') if p.strip()]
            
            sub_clauses = []
            sub_params = []
            
            for pat in raw_patterns:
                is_negative = False
                if pat.startswith('-'):
                    is_negative = True
                    pat = pat[1:].strip()
                
                if not pat: continue

                target_col = f"t{idx}" # Default to token
                regex_pat = ""
                
                # 1. Lemma [lemma]
                m_lemma = re.search(r"\[(.*?)\]", pat)
                if m_lemma and not is_raw:
                    target_col = f"lower(l{idx})"
                    regex_pat = '^' + standardize_wildcards(m_lemma.group(1).lower()) + '$'
                
                # 2. POS _POS
                elif '_' in pat and not is_raw and pat.startswith('_'): 
                   pos_input = pat[1:].strip()
                   target_col = f"p{idx}"
                   patterns = [p.strip() for p in pos_input.split('|') if p.strip()]
                   # Use case-insensitive matching for POS tags (?i)
                   regex_pat = "(?i)^(" + "|".join([standardize_wildcards(p) for p in patterns]) + ")$"

                if not regex_pat:
                    # 3. Handle Default Basis (No Prefix)
                    pat_core = pat
                    regex_pat = '^' + standardize_wildcards(pat_core) + '$'
                    
                    if default_basis == 'POS Tag' and not is_raw:
                        target_col = f"p{idx}"
                        regex_pat = "(?i)" + regex_pat
                    elif default_basis == 'Lemma' and not is_raw:
                        target_col = f"lower(l{idx})"
                        regex_pat = regex_pat.lower()
                    else:
                        target_col = f"t{idx}"

                op = "NOT regexp_matches" if is_negative else "regexp_matches"
                sub_clauses.append(f"{op}({target_col}, ?)")
                sub_params.append(regex_pat)
                
            return sub_clauses, sub_params

        # Apply positional filters
        if n_gram_filters:
            for pos_idx, pat in n_gram_filters.items():
                idx = int(pos_idx)
                if 1 <= idx <= n_size:
                    pos_basis = positional_bases.get(pos_idx, basis) if positional_bases else basis
                    clauses, prms = parse_filter(pat, idx, is_raw_mode, default_basis=pos_basis)
                    if clauses:
                        conditions.extend(clauses)
                        params.extend(prms)
        
        # Apply Global Negative Filter (Exclude list)
        if negative_filter:
            neg_list_params = negative_filter
            placeholders = ','.join(['?'] * len(neg_list_params))
            
            # Exclude n-gram if ANY token matches the negative list
            # logic: AND (t1 NOT IN list AND t2 NOT IN list ...)
            for i in range(1, n_size + 1):
                conditions.append(f"t{i} NOT IN ({placeholders})")
                params.extend(neg_list_params)

        # --- Mapping Basis to DB Columns ---
        display_cols = []
        for i in range(1, n_size + 1):
            pos_basis = positional_bases.get(str(i), basis) if positional_bases else basis
            col_prefix = "t"
            if pos_basis == "POS Tag": col_prefix = "p"
            elif pos_basis == "Lemma": col_prefix = "l"
            display_cols.append(f"{col_prefix}{i}")
            
        tokens_grp = ', '.join(display_cols)
        
        where_parts = []
        if xml_where_clause:
            where_parts.append(xml_where_clause.strip()[4:])
        
        if skip_punctuation:
            # Punctuation list from the global set
            punc_list = list(PUNCTUATION)
            where_parts.append(f"_token_low NOT IN ({','.join(['?']*len(punc_list))})")
            xml_params = list(xml_params) + punc_list

        if skip_pos_tags and not is_raw_mode:
            where_parts.append(f"pos NOT IN ({','.join(['?']*len(skip_pos_tags))})")
            xml_params = list(xml_params) + list(skip_pos_tags)

        subquery_where = ""
        if where_parts:
            subquery_where = " WHERE " + " AND ".join(where_parts)

        query = f"""
        SELECT {', '.join(display_cols)}, count(*) as freq 
        FROM (
            SELECT {subquery_select}
            FROM corpus
            {subquery_where}
        ) sub
        """
        
        # Filter out rows where any column in the basis is NULL
        not_null_cond = " AND ".join([f"{c} IS NOT NULL" for c in display_cols])
        if conditions:
            query += f" WHERE ({' AND '.join(conditions)}) AND {not_null_cond}"
        else:
            query += f" WHERE {not_null_cond}"
            
        query += f" GROUP BY {tokens_grp} ORDER BY freq DESC"
        
        full_params = xml_params + params
        df_res = con.execute(query, full_params).fetch_df()
        con.close()
        
        if df_res.empty: return pd.DataFrame()
        
        # Post-Processing
        # Convert each row to a clean space-separated string
        def format_ngram_row(row):
            tokens = []
            for col in display_cols:
                val = row[col]
                # Handle both string and list types
                if isinstance(val, list):
                    tokens.extend([str(v) for v in val])
                else:
                    tokens.append(str(val))
            return ' '.join(tokens)
        
        df_res['N-Gram'] = df_res.apply(format_ngram_row, axis=1)
        df_res['Frequency'] = df_res['freq']
        df_res['Relative Frequency (per M)'] = (df_res['freq'] / total_tokens) * 1_000_000
        df_res['Relative Frequency (per M)'] = df_res['Relative Frequency (per M)'].round(4)
        
        final_df = df_res[['N-Gram', 'Frequency', 'Relative Frequency (per M)']].copy()
        
        # Filter Punctuation/Digits (Post-process logic aligned with app.py)
        puncts = set(string.punctuation)
        def has_content(text):
            # Returns True if the n-gram has at least one content word
            for t in text.split():
                 if t not in puncts and not t.isdigit():
                     return True
            return False
            
        final_df = final_df[final_df['N-Gram'].apply(has_content)]
        
        return final_df.reset_index(drop=True)

    except Exception as e:
        print(f"N-Gram Error: {e}")
        return pd.DataFrame()
