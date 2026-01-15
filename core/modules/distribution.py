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
        
        search_terms = raw_target_input.split()
        
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
            lemma_match = re.search(r"\[(.*?)\]", term)
            if lemma_match: return {'type': 'lemma', 'val': lemma_match.group(1).strip().lower()}
            pos_match = re.search(r"\_([A-Za-z0-9\*|\\-]+)", term)
            if pos_match: return {'type': 'pos', 'val': pos_match.group(1).strip()}
            return {'type': 'word', 'val': term.lower()}

        search_components = [parse_term(term) for term in search_terms]

        query_select = "SELECT c0.id FROM corpus c0"
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
        standard_cols = {'id', 'token', 'pos', 'lemma', 'sent_id', '_token_low'}
        all_cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
        all_cols = [c[1] for c in all_cols_info]
        meta_cols = [c for c in all_cols if c not in standard_cols]
        
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
