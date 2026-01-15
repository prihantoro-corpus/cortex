import duckdb
import pandas as pd
import re
from collections import Counter
from core.statistics.frequency import pmw_to_zipf, zipf_to_band

def generate_kwic(corpus_db_path, raw_target_input, kwic_left, kwic_right, corpus_name, pattern_collocate_input="", pattern_collocate_pos_input="", pattern_window=0, limit=100, do_random_sample=False, is_parallel_mode=False, show_pos=False, show_lemma=False, xml_where_clause="", xml_params=[]):
    """
    Generalized function to generate KWIC lines using DuckDB SQL queries.
    """
    if not corpus_db_path:
        return ([], 0, raw_target_input, 0, [], pd.DataFrame())

    try:
        con = duckdb.connect(corpus_db_path, read_only=True)
        
        search_terms = raw_target_input.split()
        primary_target_len = len(search_terms)
        
        # Check raw mode
        is_raw_mode = True
        try:
            cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
            if 'pos' in cols:
                raw_count = con.execute("SELECT count(*) FROM corpus WHERE pos LIKE '##%'").fetchone()[0]
                total_rows = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
                is_raw_mode = (raw_count / total_rows) > 0.99
        except: pass
            
        def parse_term(term):
            lemma_match = re.search(r"\[(.*?)\]", term)
            if lemma_match: return {'type': 'lemma', 'val': lemma_match.group(1).strip().lower()}
            # Enhanced POS regex to support dashes and various tag formats
            pos_match = re.search(r"\_([A-Za-z0-9\*|\\-]+)", term)
            if pos_match: return {'type': 'pos', 'val': pos_match.group(1).strip()}
            return {'type': 'word', 'val': term.lower()}

        search_components = [parse_term(term) for term in search_terms]

        # Dynamic SELECT construction to capture full multi-word match
        token_concat_parts = [f"c{i}.token" for i in range(len(search_components))]
        # Use DuckDB's concatenation operator ||
        if len(token_concat_parts) > 1:
            match_token_expr = "(" + " || ' ' || ".join(token_concat_parts) + ")"
        else:
            match_token_expr = "c0.token"
            
        query_select = f"SELECT c0.id, {match_token_expr} as match_token FROM corpus c0"
        query_joins = ""
        query_where = []
        query_params = []
        
        for k, comp in enumerate(search_components):
            alias = f"c{k}"
            if k > 0: query_joins += f" JOIN corpus {alias} ON {alias}.id = c0.id + {k} "
            
            val = comp['val']
            if comp['type'] == 'word':
                if '*' in val:
                    regex_pat = '^' + re.escape(val).replace(r'\*', '.*') + '$'
                    query_where.append(f"regexp_matches({alias}._token_low, ?)")
                    query_params.append(regex_pat)
                else:
                    query_where.append(f"{alias}._token_low = ?")
                    query_params.append(val)
            elif comp['type'] == 'lemma' and not is_raw_mode:
                if '*' in val:
                    regex_pat = '^' + re.escape(val).replace(r'\*', '.*') + '$'
                    query_where.append(f"regexp_matches(lower({alias}.lemma), ?)")
                    query_params.append(regex_pat)
                else:
                    query_where.append(f"lower({alias}.lemma) = ?")
                    query_params.append(val)
            elif comp['type'] == 'pos' and not is_raw_mode:
                if '|' in val or '*' in val:
                    pos_patterns = [p.strip() for p in val.split('|') if p.strip()]
                    full_regex = "^(" + "|".join([re.escape(p).replace(r'\*', '.*') for p in pos_patterns]) + ")$"
                    query_where.append(f"regexp_matches({alias}.pos, ?)")
                    query_params.append(full_regex)
                else:
                    query_where.append(f"{alias}.pos = ?")
                    query_params.append(val)

        # --- Primary Target Query Construction ---
        final_query = query_select + query_joins
        
        # Collocate Filter Logic (Integrated into primary WHERE via EXISTS)
        is_pattern_search_active = (pattern_collocate_input or pattern_collocate_pos_input) and pattern_window > 0
        coll_filter_params = []
        
        if is_pattern_search_active:
            coll_filter_parts = []
            
            # 1. Parse main collocate filter ([lemma], _TAG, or word)
            if pattern_collocate_input:
                cp = parse_term(pattern_collocate_input)
                val = cp['val']
                if cp['type'] == 'word':
                    if '*' in val:
                        regex = '(?i)^' + re.escape(val).replace(r'\*', '.*') + '$'
                        coll_filter_parts.append("regexp_matches(c_coll._token_low, ?)")
                        coll_filter_params.append(regex)
                    else:
                        coll_filter_parts.append("c_coll._token_low = ?")
                        coll_filter_params.append(val)
                elif cp['type'] == 'lemma':
                    # Use (?i) for lemmas to be safe
                    regex = '(?i)^' + re.escape(val).replace(r'\*', '.*') + '$'
                    coll_filter_parts.append("regexp_matches(c_coll.lemma, ?)")
                    coll_filter_params.append(regex)
                elif cp['type'] == 'pos':
                    # Always use case-insensitive matching for POS tags in ID-BPPT and others
                    pats = [p.strip() for p in val.split('|') if p.strip()]
                    regex = "(?i)^(" + "|".join([re.escape(p).replace(r'\*', '.*') for p in pats]) + ")$"
                    coll_filter_parts.append("regexp_matches(c_coll.pos, ?)")
                    coll_filter_params.append(regex)

            # 2. Parse secondary POS filter (legacy field support)
            if pattern_collocate_pos_input and not is_raw_mode:
                pos_patterns = [p.strip() for p in pattern_collocate_pos_input.split('|') if p.strip()]
                full_regex = "(?i)^(" + "|".join([re.escape(p).replace(r'\*', '.*') for p in pos_patterns]) + ")$"
                coll_filter_parts.append("regexp_matches(c_coll.pos, ?)")
                coll_filter_params.append(full_regex)

            if coll_filter_parts:
                coll_sql = " AND ".join(coll_filter_parts)
                node_exclusion = f"(c_coll.id < c0.id OR c_coll.id >= c0.id + {primary_target_len})"
                
                # Using explicit comparison instead of BETWEEN for potential speed/stability
                exists_clause = f"""
                EXISTS (
                    SELECT 1 FROM corpus c_coll 
                    WHERE c_coll.id >= (c0.id - ?) AND c_coll.id <= (c0.id + ? + {primary_target_len} - 1)
                    AND {node_exclusion}
                    AND {coll_sql}
                )
                """
                query_where.append(exists_clause)
                coll_filter_params = [pattern_window, pattern_window] + coll_filter_params

        # Assemble final WHERE
        if query_where:
            final_query += " WHERE " + " AND ".join(query_where)
            if xml_where_clause: final_query += xml_where_clause
        elif xml_where_clause:
            final_query += " WHERE " + xml_where_clause.strip()[4:]
        
        full_params = query_params + coll_filter_params + xml_params
        
        df_matches = con.execute(final_query, full_params).fetch_df()
        
        if df_matches.empty:
            con.close()
            return ([], 0, raw_target_input, 0, [], pd.DataFrame())

        all_match_ids = df_matches['id'].tolist()
        matching_tokens_at_node_one = df_matches['match_token'].tolist()
        literal_freq = len(all_match_ids)
        
        filtered_match_ids = all_match_ids

        total_matches = len(filtered_match_ids)
        if total_matches == 0:
            con.close()
            return ([], 0, raw_target_input, literal_freq, [], pd.DataFrame())

        display_ids = filtered_match_ids
        if do_random_sample and total_matches > limit:
            import random
            random.seed(42)
            display_ids = random.sample(filtered_match_ids, limit)
        else:
             display_ids = filtered_match_ids[:limit]

        display_params = ", ".join([f"({mid})" for mid in display_ids])
        
        current_kwic_left = pattern_window if is_pattern_search_active and pattern_window > 0 else kwic_left
        current_kwic_right = pattern_window if is_pattern_search_active and pattern_window > 0 else kwic_right
        
        # 1. Introspect columns to find metadata
        standard_cols = {'id', 'token', 'pos', 'lemma', 'sent_id', '_token_low'}
        all_cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
        all_cols = [c[1] for c in all_cols_info]
        meta_cols = [c for c in all_cols if c not in standard_cols]
        
        meta_select_part = ""
        if meta_cols:
            meta_select_part = ", " + ", ".join([f"c.{c}" for c in meta_cols])

        context_query = f"""
        SELECT m.match_id, c.token, c.pos, c.lemma, c.id, c.sent_id{meta_select_part}
        FROM (VALUES {display_params}) m(match_id)
        JOIN corpus c ON c.id BETWEEN m.match_id - {current_kwic_left} AND m.match_id + {primary_target_len} + {current_kwic_right} - 1
        ORDER BY m.match_id, c.id
        """
        
        df_context = con.execute(context_query).fetch_df()
        
        
        breakdown_data = Counter(matching_tokens_at_node_one)
        breakdown_list = []
        # Need total rows for relative freq.
        # We did it earlier.
        
        if xml_where_clause:
            total_rows_val = con.execute(f"SELECT count(*) FROM corpus WHERE 1=1 {xml_where_clause}", xml_params).fetchone()[0]
        else:
            total_rows_val = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
        con.close()
        
        total_tokens_float = float(total_rows_val)
        for token, freq in breakdown_data.most_common():
             rel_freq = (freq / total_tokens_float) * 1_000_000
             breakdown_list.append({
                 "Token Form": token, 
                 "Absolute Frequency": freq, 
                 "Relative Frequency (per M)": round(rel_freq, 4)
             })
        breakdown_df = pd.DataFrame(breakdown_list)
        if not breakdown_df.empty:
            breakdown_df['Zipf Score'] = breakdown_df['Relative Frequency (per M)'].apply(pmw_to_zipf).round(2)
            breakdown_df['Zipf Law Frequency Band'] = breakdown_df['Zipf Score'].apply(zipf_to_band)

        kwic_rows = []
        sent_ids = []
        
        coll_comp_hl = parse_term(pattern_collocate_input) if pattern_collocate_input else None
        coll_word_regex_hl = None
        coll_lemma_regex_hl = None
        coll_pos_regex_input_hl = None

        if coll_comp_hl:
            val = coll_comp_hl['val']
            if coll_comp_hl['type'] == 'word':
                pat = '^' + re.escape(val).replace(r'\*', '.*') + '$'
                coll_word_regex_hl = re.compile(pat, re.IGNORECASE)
            elif coll_comp_hl['type'] == 'lemma':
                pat = '^' + re.escape(val).replace(r'\*', '.*') + '$'
                coll_lemma_regex_hl = re.compile(pat, re.IGNORECASE)
            elif coll_comp_hl['type'] == 'pos':
                if '|' in val or '*' in val:
                    pos_patterns = [p.strip() for p in val.split('|') if p.strip()]
                    full_regex = "^(" + "|".join([re.escape(p).replace(r'\*', '.*') for p in pos_patterns]) + ")$"
                    coll_pos_regex_input_hl = re.compile(full_regex, re.IGNORECASE)
                else:
                    coll_pos_regex_input_hl = re.compile('^' + re.escape(val) + '$', re.IGNORECASE)

        collocate_pos_regex_highlight = None
        if pattern_collocate_pos_input and not is_raw_mode:
            pos_patterns = [p.strip() for p in pattern_collocate_pos_input.split('|') if p.strip()]
            if pos_patterns:
                 full_regex = "^(" + "|".join([re.escape(p).replace(r'\*', '.*') for p in pos_patterns]) + ")$"
                 collocate_pos_regex_highlight = re.compile(full_regex, re.IGNORECASE)

        grouped = df_context.groupby('match_id')
        
        for match_id, group in grouped:
            tokens = group['token'].tolist()
            tokens_low = group['token'].str.lower().tolist()
            poss = group['pos'].tolist()
            lemmas = group['lemma'].tolist()
            c_ids = group['id'].tolist()
            chunk_sent_ids = group['sent_id'].tolist()
            
            try:
                node_start_idx = c_ids.index(match_id)
            except ValueError: 
                continue 
                
            sent_ids.append(chunk_sent_ids[node_start_idx]) 
            
            # Extract metadata from the NODE row
            metadata = {}
            if meta_cols:
                # We can grab it from proper row index
                for mc in meta_cols:
                    if mc in group.columns:
                        # Grab value from the node's row
                        val = group.iloc[node_start_idx][mc]
                        if val is not None and str(val).strip() != "":
                            metadata[mc] = val
            
            formatted_line = []
            collocate_to_display = ""
            node_orig_tokens = []
            
            for k, token in enumerate(tokens):
                t_low = tokens_low[k]
                t_pos = poss[k]
                t_lemma = lemmas[k]
                
                is_node = (node_start_idx <= k < node_start_idx + primary_target_len)
                
                is_coll_match = False
                if is_pattern_search_active and not is_node:
                    wm = True
                    if coll_word_regex_hl: wm = coll_word_regex_hl.fullmatch(t_low)
                    elif coll_lemma_regex_hl: wm = coll_lemma_regex_hl.fullmatch(t_lemma.lower())
                    elif coll_pos_regex_input_hl: wm = coll_pos_regex_input_hl.fullmatch(t_pos)

                    pm = collocate_pos_regex_highlight is None or (collocate_pos_regex_highlight.fullmatch(t_pos) if not is_raw_mode else False)
                    if wm and pm:
                        is_coll_match = True
                        if not collocate_to_display: collocate_to_display = token
                
                token_html = token
                if is_coll_match: token_html = f"<span style='color: black; background-color: #FFEA00;'>{token}</span>"
                if is_node: token_html = f"<b>{token_html}</b>"
                
                output = [token_html]
                if show_pos and t_pos not in ('##', '###'):
                     output.append(f"/<span style='font-size: 0.8em; color: #33CC33;'>{t_pos}</span>")
                if show_lemma and t_lemma not in ('##', '###'):
                     output.append(f"{{<span style='font-size: 0.7em; color: #00AAAA;'>{t_lemma}</span>}}")
                
                final_html = "".join(output)
                
                if is_node:
                    node_orig_tokens.append(final_html)
                else:
                    formatted_line.append(final_html)
            
            left_part = formatted_line[:node_start_idx]
            right_part = formatted_line[node_start_idx:] 
            
            kwic_rows.append({
                "Left": " ".join(left_part),
                "Node": " ".join(node_orig_tokens),
                "Right": " ".join(right_part),
                "Collocate": collocate_to_display,
                "Metadata": metadata  # Add Metadata
            })

        return (kwic_rows, total_matches, raw_target_input, literal_freq, sent_ids, breakdown_df)

    except Exception as e:
        print(f"Error in generate_kwic: {e}")
        return ([], 0, raw_target_input, 0, [], pd.DataFrame())
