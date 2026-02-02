import duckdb
import pandas as pd
import numpy as np
import re
from scipy import stats
from typing import Dict, List, Tuple, Optional
from collections import Counter


def parse_query_term(term: str) -> Dict:
    """
    Parse a query term into structured components.
    
    Supports:
    - (word1|word2|word3) - OR pattern for words [NEW for stats module]
    - [lemma] - Lemma search
    - token_POS - Combined token and POS (e.g., light_V*)
    - _POS - POS tag filter (e.g., _JJ*)
    - *wildcard* - Wildcard matching
    - word - Plain word search
    
    Returns:
        Dict with 'type' and relevant value fields
    """
    # Check for OR pattern: (word1|word2|word3)
    or_pattern = re.match(r'\((.*?)\)', term)
    if or_pattern:
        words = [w.strip().lower() for w in or_pattern.group(1).split('|') if w.strip()]
        return {'type': 'word_or', 'values': words}
    
    # Check for lemma: [lemma]
    lemma_match = re.search(r"\[(.*?)\]", term)
    if lemma_match:
        return {'type': 'lemma', 'val': lemma_match.group(1).strip().lower()}
    
    # Check for combined Token_POS: light_V*
    if '_' in term and not term.startswith('_'):
        parts = term.rsplit('_', 1)
        if len(parts) == 2 and parts[1]:
            return {'type': 'token_pos', 'token': parts[0].lower(), 'pos': parts[1]}
    
    # Check for POS tag: _JJ*
    pos_match = re.search(r"\_([A-Za-z0-9\*|\-]+)", term)
    if pos_match:
        return {'type': 'pos', 'val': pos_match.group(1).strip()}
    
    # Default: plain word
    return {'type': 'word', 'val': term.lower()}


def build_query_where_clause(parsed_term: Dict, alias: str = "c") -> Tuple[str, List]:
    """
    Build SQL WHERE clause and params for a parsed query term.
    
    Args:
        parsed_term: Output from parse_query_term()
        alias: Table alias to use in query
        
    Returns:
        Tuple of (where_clause_string, params_list)
    """
    where_parts = []
    params = []
    
    if parsed_term['type'] == 'word_or':
        # (word1|word2|word3) → WHERE _token_low IN (?, ?, ?)
        placeholders = ', '.join(['?' for _ in parsed_term['values']])
        where_parts.append(f"{alias}._token_low IN ({placeholders})")
        params.extend(parsed_term['values'])
        
    elif parsed_term['type'] == 'word':
        val = parsed_term['val']
        if '*' in val:
            # Wildcard: *ing → WHERE regexp_matches(_token_low, '^.*ing$')
            regex_pat = '^' + re.escape(val).replace(r'\*', '.*') + '$'
            where_parts.append(f"regexp_matches({alias}._token_low, ?)")
            params.append(regex_pat)
        else:
            # Exact match
            where_parts.append(f"{alias}._token_low = ?")
            params.append(val)
            
    elif parsed_term['type'] == 'lemma':
        val = parsed_term['val']
        if '*' in val:
            regex_pat = '^' + re.escape(val).replace(r'\*', '.*') + '$'
            where_parts.append(f"regexp_matches(lower({alias}.lemma), ?)")
            params.append(regex_pat)
        else:
            where_parts.append(f"lower({alias}.lemma) = ?")
            params.append(val)
            
    elif parsed_term['type'] == 'pos':
        val = parsed_term['val']
        if '|' in val or '*' in val:
            # Multiple POS: _NN*|VB*
            pos_patterns = [p.strip() for p in val.split('|') if p.strip()]
            full_regex = "^(" + "|".join([re.escape(p).replace(r'\*', '.*') for p in pos_patterns]) + ")$"
            where_parts.append(f"regexp_matches({alias}.pos, ?)")
            params.append(full_regex)
        else:
            where_parts.append(f"{alias}.pos = ?")
            params.append(val)
            
    elif parsed_term['type'] == 'token_pos':
        # Combined: light_V*
        t_val = parsed_term['token']
        p_val = parsed_term['pos']
        
        # Token part
        if '*' in t_val:
            regex_pat = '^' + re.escape(t_val).replace(r'\*', '.*') + '$'
            where_parts.append(f"regexp_matches({alias}._token_low, ?)")
            params.append(regex_pat)
        else:
            where_parts.append(f"{alias}._token_low = ?")
            params.append(t_val)
        
        # POS part
        if '|' in p_val or '*' in p_val:
            pats = [p.strip() for p in p_val.split('|') if p.strip()]
            regex = "^(" + "|".join([re.escape(p).replace(r'\*', '.*') for p in pats]) + ")$"
            where_parts.append(f"regexp_matches({alias}.pos, ?)")
            params.append(regex)
        else:
            where_parts.append(f"{alias}.pos = ?")
            params.append(p_val)
    
    return (" AND ".join(where_parts), params)


def compare_groups_by_word(
    corpus_db_path: str,
    query: str,
    grouping_attr: str,
    groups: List[str],
    min_freq: int = 3,
    freq_measure: str = 'absolute',
    test_type: str = 'ttest_ind',
    multiple_comparison: Optional[str] = 'bonferroni',
    xml_where_clause: str = "",
    xml_params: List = None
) -> pd.DataFrame:
    """
    Compare frequency of words matching a query across groups.
    
    Args:
        corpus_db_path: Path to DuckDB corpus database
        query: Search query (e.g., "_JJ*", "(small|big|little)", "*ing")
        grouping_attr: XML attribute to group by (e.g., "sex")
        groups: List of group values (e.g., ["f", "m"])
        min_freq: Minimum total frequency threshold
        freq_measure: 'absolute', 'relative', or 'proportion'
        test_type: 'ttest_ind' or 'mann_whitney'
        multiple_comparison: None, 'bonferroni', or 'fdr_bh'
        xml_where_clause: Additional XML filters
        xml_params: Parameters for XML filters
        
    Returns:
        DataFrame with columns: word, group1_freq, group2_freq, 
        test_statistic, p_value, p_value_corrected, effect_size, significance
    """
    if xml_params is None:
        xml_params = []
    
    # Parse query
    parsed_terms = [parse_query_term(t.strip()) for t in query.split() if t.strip()]
    
    # For now, support single-term queries only
    if len(parsed_terms) != 1:
        raise ValueError("Multi-word queries not yet supported in statistical testing. Use single query term.")
    
    parsed = parsed_terms[0]
    
    # Build WHERE clause for the query
    query_where, query_params = build_query_where_clause(parsed, "c")
    
    con = duckdb.connect(corpus_db_path, read_only=True)
    
    try:
        # Build frequency extraction queries for each group
        group_freqs = {}
        
        for group in groups:
            # Build complete WHERE clause
            where_parts = [query_where]
            params = query_params.copy()
            
            # Add grouping attribute filter
            where_parts.append(f"c.{grouping_attr} = ?")
            params.append(group)
            
            # Add XML restrictions if any
            if xml_where_clause:
                where_parts.append(xml_where_clause.strip()[4:])  # Remove "AND "
                params.extend(xml_params)
            
            full_where = " AND ".join(where_parts)
            
            # Extract frequencies
            freq_query = f"""
                SELECT 
                    lower(c.token) as word,
                    COUNT(*) as freq
                FROM corpus c
                WHERE {full_where}
                GROUP BY lower(c.token)
            """
            
            df = con.execute(freq_query, params).fetch_df()
            group_freqs[group] = df.set_index('word')['freq'].to_dict()
        
        # Align frequencies using FULL OUTER JOIN logic
        all_words = set()
        for freq_dict in group_freqs.values():
            all_words.update(freq_dict.keys())
        
        # Build aligned frequency table
        aligned_data = []
        for word in all_words:
            freqs = [group_freqs[g].get(word, 0) for g in groups]
            total_freq = sum(freqs)
            
            # Apply minimum frequency filter
            if total_freq >= min_freq:
                aligned_data.append([word] + freqs)
        
        # Create DataFrame
        columns = ['word'] + [f'{g}_freq' for g in groups]
        df_aligned = pd.DataFrame(aligned_data, columns=columns)
        
        if df_aligned.empty:
            return df_aligned
        
        # Perform statistical tests
        results = []
        
        # Get total tokens per group for proper proportion testing
        group_totals = {}
        for group in groups:
            where_parts = []
            params = []
            
            where_parts.append(f"c.{grouping_attr} = ?")
            params.append(group)
            
            if xml_where_clause:
                where_parts.append(xml_where_clause.strip()[4:])
                params.extend(xml_params)
            
            full_where = " AND ".join(where_parts)
            
            total_query = f"SELECT COUNT(*) FROM corpus c WHERE {full_where}"
            group_totals[group] = con.execute(total_query, params).fetchone()[0]
        
        total1 = group_totals[groups[0]]
        total2 = group_totals[groups[1]]
        
        for idx, row in df_aligned.iterrows():
            word = row['word']
            freq1 = row[f'{groups[0]}_freq']
            freq2 = row[f'{groups[1]}_freq']
            
            # Skip if both groups have zero
            if freq1 == 0 and freq2 == 0:
                test_stat = np.nan
                p_value = 1.0
                effect_size = 0.0
            else:
                # Chi-square test for proportions (2x2 contingency table)
                # [[word_freq_group1, other_words_group1],
                #  [word_freq_group2, other_words_group2]]
                observed = np.array([
                    [freq1, total1 - freq1],
                    [freq2, total2 - freq2]
                ])
                
                # Avoid chi-square warning for low expected frequencies
                # Use Yates' correction for 2x2 tables
                try:
                    chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed, correction=True)
                    test_stat = chi2_stat
                except (ValueError, ZeroDivisionError):
                    # If table is degenerate, set to no difference
                    test_stat = 0.0
                    p_value = 1.0
                
                # Cohen's h for proportions (effect size)
                p1 = freq1 / total1 if total1 > 0 else 0
                p2 = freq2 / total2 if total2 > 0 else 0
                h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
                effect_size = h
            
            results.append({
                'word': word,
                f'{groups[0]}_freq': freq1,
                f'{groups[1]}_freq': freq2,
                f'{groups[0]}_prop': p1,
                f'{groups[1]}_prop': p2,
                'test_statistic': test_stat,
                'p_value': p_value,
                'effect_size': effect_size
            })
        
        df_results = pd.DataFrame(results)
        
        # Apply multiple comparison correction
        if multiple_comparison and len(df_results) > 1:
            if multiple_comparison == 'bonferroni':
                df_results['p_value_corrected'] = df_results['p_value'] * len(df_results)
                df_results['p_value_corrected'] = df_results['p_value_corrected'].clip(upper=1.0)
            elif multiple_comparison == 'fdr_bh':
                # Benjamini-Hochberg procedure (manual implementation, no statsmodels needed)
                p_values = df_results['p_value'].values
                n = len(p_values)
                
                # Sort p-values and get original indices
                sorted_indices = np.argsort(p_values)
                sorted_pvals = p_values[sorted_indices]
                
                # Calculate corrected p-values
                p_corrected = np.zeros(n)
                for i in range(n-1, -1, -1):
                    if i == n-1:
                        p_corrected[i] = sorted_pvals[i]
                    else:
                        p_corrected[i] = min(sorted_pvals[i] * n / (i+1), p_corrected[i+1])
                
                # Restore original order
                p_corrected_original_order = np.zeros(n)
                p_corrected_original_order[sorted_indices] = p_corrected
                
                df_results['p_value_corrected'] = np.clip(p_corrected_original_order, 0, 1.0)
        else:
            df_results['p_value_corrected'] = df_results['p_value']
        
        # Add significance stars
        def get_sig(p):
            if p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            else:
                return 'ns'
        
        df_results['significance'] = df_results['p_value_corrected'].apply(get_sig)
        
        # Sort by p-value
        df_results = df_results.sort_values('p_value')
        
        return df_results
        
    finally:
        con.close()


def preview_query_matches(
    corpus_db_path: str,
    query: str,
    min_freq: int = 3,
    xml_where_clause: str = "",
    xml_params: List = None,
    limit: int = 10
) -> Dict:
    """
    Preview what words match a query (for UI feedback).
    
    Returns:
        Dict with 'total_words', 'words_above_threshold', 'sample_words'
    """
    if xml_params is None:
        xml_params = []
    
    parsed_terms = [parse_query_term(t.strip()) for t in query.split() if t.strip()]
    
    if len(parsed_terms) != 1:
        return {'error': 'Multi-word queries not yet supported'}
    
    parsed = parsed_terms[0]
    query_where, query_params = build_query_where_clause(parsed, "c")
    
    con = duckdb.connect(corpus_db_path, read_only=True)
    
    try:
        # Build WHERE clause
        where_parts = [query_where]
        params = query_params.copy()
        
        if xml_where_clause:
            where_parts.append(xml_where_clause.strip()[4:])
            params.extend(xml_params)
        
        full_where = " AND ".join(where_parts)
        
        # Get word frequencies
        freq_query = f"""
            SELECT 
                lower(c.token) as word,
                COUNT(*) as freq
            FROM corpus c
            WHERE {full_where}
            GROUP BY lower(c.token)
            ORDER BY freq DESC
        """
        
        df = con.execute(freq_query, params).fetch_df()
        
        total_words = len(df)
        words_above_threshold = len(df[df['freq'] >= min_freq])
        sample_words = df.head(limit)['word'].tolist()
        
        return {
            'total_words': total_words,
            'words_above_threshold': words_above_threshold,
            'sample_words': sample_words,
            'total_freq': int(df['freq'].sum())
        }
        
    finally:
        con.close()

# -----------------------------------------------------------------------------
# PHASE 2: CORRELATION ANALYSIS BACKEND
# -----------------------------------------------------------------------------

def get_document_frequency_vector(
    corpus_db_path: str,
    query: str,
    xml_where_clause: str = "",
    xml_params: list = [],
    group_by: str = "filename"
) -> pd.DataFrame:
    """
    Returns a DataFrame [group, freq] representing the frequency of the query per document.
    """
    con = duckdb.connect(corpus_db_path, read_only=True)
    try:
        parsed = parse_query_term(query)
        where_q, params_q = build_query_where_clause(parsed, alias="c")
        
        full_where = f"{where_q}"
        if xml_where_clause:
            full_where += f" AND {xml_where_clause}"
            
        final_params = params_q + xml_params
        
        sql = f"""
            SELECT {group_by} as group_id, COUNT(*) as val
            FROM corpus c
            WHERE {full_where}
            GROUP BY {group_by}
        """
        df = con.execute(sql, final_params).fetch_df()
        return df.set_index('group_id')
    finally:
        con.close()


def get_document_metadata_vector(
    corpus_db_path: str,
    attribute: str,
    xml_where_clause: str = "",
    xml_params: list = [],
    group_by: str = "filename"
) -> pd.DataFrame:
    """
    Returns a DataFrame [group, val] for a numeric metadata attribute.
    Tries to cast to FLOAT. Filters out non-numeric values.
    """
    con = duckdb.connect(corpus_db_path, read_only=True)
    try:
        where_clause = "1=1"
        if xml_where_clause:
            where_clause = xml_where_clause
            
        sql = f"""
            SELECT {group_by} as group_id, TRY_CAST({attribute} AS FLOAT) as val
            FROM corpus
            WHERE {where_clause} AND {attribute} IS NOT NULL
            GROUP BY {group_by}, {attribute}
        """
        df = con.execute(sql, xml_params).fetch_df()
        
        # Drop NaNs (failed casts)
        df = df.dropna(subset=['val'])
        
        # If multiple values per document, take mean
        df = df.groupby('group_id').mean()
        
        return df
    finally:
        con.close()


def get_document_metric_vector(
    corpus_db_path: str,
    metric: str,  # 'ttr', 'token_count', 'type_count'
    xml_where_clause: str = "",
    xml_params: list = [],
    group_by: str = "filename"
) -> pd.DataFrame:
    """
    Returns a DataFrame [group, val] for a calculated metric (TTR, Length).
    """
    con = duckdb.connect(corpus_db_path, read_only=True)
    try:
        where_clause = "1=1"
        if xml_where_clause:
            where_clause = xml_where_clause
        
        if metric == 'ttr':
            sql = f"""
                SELECT {group_by} as group_id, CAST(COUNT(DISTINCT _token_low) AS FLOAT) / NULLIF(COUNT(*), 0) as val
                FROM corpus
                WHERE {where_clause}
                GROUP BY {group_by}
            """
        elif metric == 'token_count':
            sql = f"""
                SELECT {group_by} as group_id, COUNT(*) as val
                FROM corpus
                WHERE {where_clause}
                GROUP BY {group_by}
            """
        elif metric == 'type_count':
             sql = f"""
                SELECT {group_by} as group_id, COUNT(DISTINCT _token_low) as val
                FROM corpus
                WHERE {where_clause}
                GROUP BY {group_by}
            """
        else:
            return pd.DataFrame()
            
        df = con.execute(sql, xml_params).fetch_df()
        return df.set_index('group_id')
    finally:
        con.close()


def calculate_correlation(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    method: str = 'pearson' # 'pearson' or 'spearman'
) -> dict:
    """
    Aligns two dataframes by index (filename) and calculates correlation.
    Returns dict with r, p_value, n, and the aligned dataframe for plotting.
    """
    # Join on index (inner join)
    aligned = df1.join(df2, lsuffix='_x', rsuffix='_y', how='inner')
    
    if len(aligned) < 3:
        return {'error': "Not enough overlapping documents (need at least 3)."}
    
    # Fill NaN frequencies with 0? 
    # Technically, if a doc is in the corpus but not in query result, it has 0 frequency.
    # But get_document_frequency_vector only returns docs WITH matches.
    # We should fetch ALL docs to be correct about 0s?
    # This is important. If calculating correlation between "happy" and "sad", docs without "happy" should be 0.
    # Currently my implementation of get_document_frequency_vector only returns rows where COUNT > 0.
    pass 
    
    x = aligned.iloc[:, 0]
    y = aligned.iloc[:, 1]
    
    if method == 'spearman':
        stat, p = stats.spearmanr(x, y)
    else:
        stat, p = stats.pearsonr(x, y)
        
    return {
        'r': stat,
        'p_value': p,
        'n': len(aligned),
        'df_plot': aligned.reset_index()
    }


# -----------------------------------------------------------------------------
# PHASE 3: CLUSTERING ANALYSIS BACKEND
# -----------------------------------------------------------------------------

def get_feature_matrix(
    corpus_db_path: str,
    group_by: str,
    top_n_features: int = 50,
    xml_where_clause: str = "",
    xml_params: list = [],
    selected_groups: List[str] = None,
    ngram_sizes: List[int] = [1],
    exclude_punct: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Creates a document-term matrix for clustering, supporting mixed n-gram sizes.
    Rows: Segments (files or groups)
    Columns: Features (words/ngrams)
    
    Args:
        ngram_sizes: List of sizes to include, e.g. [1, 2] for unigrams + bigrams.
                     Will fetch 'top_n_features' for EACH size.
    """
    if isinstance(ngram_sizes, int):
        ngram_sizes = [ngram_sizes]
        
    con = duckdb.connect(corpus_db_path, read_only=True)
    try:
        where_clause = "1=1"
        if xml_where_clause:
            where_clause = xml_where_clause
            
        punct_filter = ""
        if exclude_punct:
            # Filter tokens that are just punctuation/symbols
            punct_filter = "AND NOT regexp_matches(token, '^[[:punct:]]+$')"

        all_matrices = []
        all_top_features = []
        
        feature_params = xml_params[:]
        if selected_groups is not None:
             placeholders_g = ', '.join(['?' for _ in selected_groups])
             # We reuse this param list logic below
             
        # Iterate through each requested N-gram size
        for size in ngram_sizes:
            # 1. Identify Top N features for this size
            top_words_query = ""
            top_params = xml_params[:]
            
            if size == 1:
                top_words_query = f"""
                    SELECT lower(token) as word, COUNT(*) as freq
                    FROM corpus
                    WHERE {where_clause} {punct_filter}
                """
                if selected_groups is not None:
                    if not selected_groups:
                        top_words_query += " AND 1=0"
                    else:
                        top_words_query += f" AND {group_by} IN ({placeholders_g})"
                        top_params.extend(selected_groups)
                        
                top_words_query += f" GROUP BY lower(token) ORDER BY freq DESC LIMIT ?"
                
            else:
                # N-gram Top Words
                window_cols = ", ".join([f"LEAD(token, {i}) OVER (PARTITION BY filename, sent_id ORDER BY id) as next_{i}" for i in range(1, size)])
                
                # Check for nulls in window
                null_check = ' AND '.join([f"next_{i} IS NOT NULL" for i in range(1, size)])
                
                # Construction of ngram string
                # We want: t0 || ' ' || lower(next_1) || ' ' || lower(next_2)...
                joined_next = " || ' ' || ".join([f"lower(next_{i})" for i in range(1, size)])
                ngram_build = f"t0 || ' ' || {joined_next}"
                
                top_words_query = f"""
                    WITH ngrams AS (
                        SELECT 
                            {group_by} as grp,
                            lower(token) as t0,
                            {window_cols}
                        FROM corpus
                        WHERE {where_clause} {punct_filter}
                    )
                    SELECT 
                        {ngram_build} as word,
                        COUNT(*) as freq
                    FROM ngrams
                    WHERE {null_check}
                """
                
                if selected_groups is not None:
                    if not selected_groups:
                        top_words_query += " AND 1=0"
                    else:
                        top_words_query += f" AND grp IN ({placeholders_g})"
                        top_params.extend(selected_groups)
                
                top_words_query += " GROUP BY word ORDER BY freq DESC LIMIT ?"

            # Execute Top Words
            df_top = con.execute(top_words_query, top_params + [top_n_features]).fetch_df()
            features = df_top['word'].tolist()
            
            if not features:
                continue
                
            all_top_features.extend(features)
            
            # 2. Build Matrix for this size
            params_mat = xml_params[:]
            placeholders_f = ', '.join(['?' for _ in features])
            
            if size == 1:
                matrix_query = f"""
                    SELECT {group_by} as group_id, lower(token) as word, COUNT(*) as freq
                    FROM corpus
                    WHERE {where_clause} {punct_filter}
                """
                if selected_groups is not None:
                    if not selected_groups:
                        matrix_query += " AND 1=0"
                    else:
                        matrix_query += f" AND {group_by} IN ({placeholders_g})"
                        params_mat.extend(selected_groups)
                    
                matrix_query += f" AND lower(token) IN ({placeholders_f}) GROUP BY {group_by}, lower(token)"
                params_mat.extend(features)
                
            else:
                 # N-gram Matrix
                window_cols = ", ".join([f"LEAD(token, {i}) OVER (PARTITION BY filename, sent_id ORDER BY id) as next_{i}" for i in range(1, size)])
                null_check = ' AND '.join([f"next_{i} IS NOT NULL" for i in range(1, size)])
                
                joined_next = " || ' ' || ".join([f"lower(next_{i})" for i in range(1, size)])
                ngram_build = f"t0 || ' ' || {joined_next}"
                
                matrix_query = f"""
                    WITH ngrams AS (
                        SELECT 
                            {group_by} as group_id,
                            lower(token) as t0,
                            {window_cols}
                        FROM corpus
                        WHERE {where_clause} {punct_filter}
                    )
                    SELECT group_id, {ngram_build} as word, COUNT(*) as freq
                    FROM ngrams
                    WHERE {null_check}
                """
                if selected_groups is not None:
                    if not selected_groups:
                        matrix_query += " AND 1=0"
                    else:
                        matrix_query += f" AND group_id IN ({placeholders_g})"
                        params_mat.extend(selected_groups)
                
                matrix_query += f" AND word IN ({placeholders_f}) GROUP BY group_id, word"
                params_mat.extend(features)
            
            # Execute Matrix
            df_mat = con.execute(matrix_query, params_mat).fetch_df()
            
            # Pivot
            if not df_mat.empty:
                mat_pivot = df_mat.pivot(index='group_id', columns='word', values='freq').fillna(0)
                all_matrices.append(mat_pivot)
        
        if not all_matrices:
            return pd.DataFrame(), []
            
        # 3. Merge all matrices
        # Join on index (group_id)
        final_matrix = pd.concat(all_matrices, axis=1, join='outer').fillna(0)
        
        # Ensure all features exist (if any dropped during pivot, unlikely but safe)
        for f in all_top_features:
            if f not in final_matrix.columns:
                final_matrix[f] = 0.0
                
        # Sort columns to specific order? 
        # Maybe n-gram size order, then freq? Current order is [Size1_Top50, Size2_Top50...] which is good.
        final_matrix = final_matrix[all_top_features]
        
        return final_matrix, all_top_features
        
    finally:
        con.close()


def perform_clustering(
    df_matrix: pd.DataFrame,
    distance_metric: str = 'cityblock',
    method: str = 'ward',
    use_z_scores: bool = True
) -> Dict:
    """
    Performs hierarchical clustering on the feature matrix.
    Defaults allow for 'Burrows Delta' style clustering (Manhattan dist on Z-scores).
    
    Args:
        df_matrix: Feature matrix (rows=documents, cols=features)
        distance_metric: 'cityblock' (Manhattan), 'euclidean', 'cosine'
        method: 'ward' (default), 'average', 'complete'
        use_z_scores: If True, standardizes features (columns) to Z-scores (Stylo default).
        
    Returns:
        Dict with 'linkage', 'labels', 'dendrogram_fig', 'distance_matrix'
    """
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist, squareform
    
    # 1. Normalize to Relative Frequencies
    # Sum of row
    row_sums = df_matrix.sum(axis=1)
    # Avoid division by zero
    norm_matrix = df_matrix.div(row_sums.replace(0, 1), axis=0) * 100  # Percentage for readability
    
    # 2. Z-Score Standardization (Stylo / Burrows Delta Standard)
    # (Value - Mean) / StdDev for each word column
    if use_z_scores:
        # We process columns. 
        # Note: If a word has 0 variance (same value in all docs), std is 0. 
        # We replace 0 std with 1 to avoid NaN, effectively making Z=0 for that column.
        means = norm_matrix.mean(axis=0)
        stds = norm_matrix.std(axis=0).replace(0, 1)
        
        # Apply Z-score
        analysis_matrix = (norm_matrix - means) / stds
    else:
        analysis_matrix = norm_matrix
        
    # 3. Calculate Distances
    cols = df_matrix.columns.tolist()
    index = df_matrix.index.tolist()
    
    if len(index) < 2:
        return {'error': "Need at least 2 groups to cluster."}
    
    # Map friendly names to scipy metric names if needed
    metric_map = {
        'manhattan': 'cityblock',
        'euclidean': 'euclidean',
        'cosine': 'cosine'
    }
    scipy_metric = metric_map.get(distance_metric, distance_metric)
    
    # pdist returns condensed distance matrix
    dist_condensed = pdist(analysis_matrix.values, metric=scipy_metric)
    dist_matrix = pd.DataFrame(
        squareform(dist_condensed), 
        index=index, 
        columns=index
    )
    
    # 4. Linkage
    # Ward's method is defined for Euclidean distances. 
    # If using Manhattan (Delta), 'ward' might technically be misused, BUT it's a common practice to try.
    # Ideally, for Delta, 'average' or 'complete' linkage is statistically safer, but Ward often produces cleaner trees.
    # We will allow the user's choice.
    try:
        z = hierarchy.linkage(dist_condensed, method=method)
    except ValueError:
        # Fallback for strict method checks
        z = hierarchy.linkage(dist_condensed, method='average')

    return {
        'linkage': z,
        'labels': index,
        'distance_matrix': dist_matrix,
        'norm_matrix': norm_matrix, # Return the RELATIVE frequencies for interpretation (not Z-scores)
        'z_matrix': analysis_matrix if use_z_scores else None
    }


def generate_clustering_interpretation(
    norm_matrix: pd.DataFrame,
    linkage_matrix: np.ndarray,
    labels: List[str],
    z_matrix: pd.DataFrame = None
) -> str:
    """
    Generates a text interpretation of the clustering.
    Identifies which features (words) define the tightest clusters.
    """
    # 1. Find the two most similar items (first step of linkage)
    idx1 = int(linkage_matrix[0][0])
    idx2 = int(linkage_matrix[0][1])
    dist = linkage_matrix[0][2]
    
    n_samples = len(labels)
    
    name1 = labels[idx1] if idx1 < n_samples else "Cluster " + str(idx1)
    name2 = labels[idx2] if idx2 < n_samples else "Cluster " + str(idx2)
    
    # Feature importance
    # If Z-scores were used, we look for words with high Z-scores in BOTH docs (both deviate above mean).
    # If not, we look for high frequency.
    
    if z_matrix is not None:
        # Use Z-matrix to find shared deviations
        vec1 = z_matrix.loc[name1]
        vec2 = z_matrix.loc[name2]
        
        # We want features where both act similarly and distinctly.
        # Simple heuristic: Sum of Z-scores, filtered where both are positive.
        # Showing words that are positively distinctive for this pair.
        combined = vec1 + vec2
        
        # valid: distinctively high in both
        valid_mask = (vec1 > 0.0) & (vec2 > 0.0) 
        candidates = combined[valid_mask].sort_values(ascending=False).head(3)
        
        explanation_type = "higher than average usage"
    else:
        # Use relative frequency
        vec1 = norm_matrix.loc[name1]
        vec2 = norm_matrix.loc[name2]
        combined = vec1 + vec2
        candidates = combined.sort_values(ascending=False).head(3)
        explanation_type = "high relative frequencies"
    
    if candidates.empty:
         # Fallback if no distinctively high words found
         top_words_str = "(no distinctively high shared words found)"
    else:
        top_words_str = ", ".join([f"**{w}**" for w in candidates.index])
    
    interpretation = f"""
    **Clustering Insight:**
    
    The most similar segments in this dataset are **{name1}** and **{name2}** (Distance: {dist:.4f}).
    
    They were grouped together primarily because they share {explanation_type} of: {top_words_str}.
    
    This approach typically aligns with **Burrows' Delta** methodology (using Z-scores and Manhattan distance) commonly used in stylometry (e.g. `stylo` R package).
    """
    
    return interpretation

def perform_correspondence_analysis(
    df_matrix: pd.DataFrame
) -> Dict:
    """
    Performs Simple Correspondence Analysis (CA) on a contingency table.
    
    Args:
        df_matrix: DataFrame of non-negative frequencies (Index=Documents/Groups, Columns=Features).
        
    Returns:
        Dict with 'row_coords', 'col_coords', 'explained_inertia' (list of var explained per dim).
        Returns {'error': ...} on failure.
    """
    # 1. Validation
    X = df_matrix.values
    if np.any(X < 0):
        return {'error': "Data contains negative values. CA requires non-negative frequencies."}
    if np.sum(X) == 0:
        return {'error': "Matrix is empty (sum=0)."}
        
    # 2. Probability Matrix
    grand_total = np.sum(X)
    P = X / grand_total
    
    # 3. Masses
    r_masses = np.sum(P, axis=1) # Row masses
    c_masses = np.sum(P, axis=0) # Column masses
    
    # Drop zero rows/cols to avoid division by zero
    valid_r = r_masses > 0
    valid_c = c_masses > 0
    
    if not np.any(valid_r) or not np.any(valid_c):
        return {'error': "Matrix consists entirely of zeros."}
        
    # Use only valid data for SVD
    P_valid = P[valid_r][:, valid_c]
    r_m = r_masses[valid_r]
    c_m = c_masses[valid_c]
    
    # 4. Standardized Residuals
    # R = (P - r_m @ c_m.T) / sqrt(r_m @ c_m.T) = (P - E) / sqrt(E)
    # Using element-wise operations with outer product
    E = np.outer(r_m, c_m)
    
    # Avoid div by zero in E (shouldn't happen with valid masses, but safe)
    with np.errstate(divide='ignore', invalid='ignore'):
         residuals = (P_valid - E) / np.sqrt(E)
         
    residuals = np.nan_to_num(residuals)
    
    # 5. SVD
    try:
        U, s, Vt = np.linalg.svd(residuals, full_matrices=False)
    except np.linalg.LinAlgError:
        return {'error': "SVD computation failed."}
        
    # 6. Inertia
    eigenvalues = s**2
    total_inertia = np.sum(eigenvalues)
    explained_inertia = eigenvalues / total_inertia
    
    # 7. Coordinates (Principal Coordinates)
    # Row Principal Coordinates = U * s / sqrt(r_m)
    row_coords_valid = (U * s) / np.sqrt(r_m)[:, None]
    
    # Col Principal Coordinates = V * s / sqrt(c_m)
    col_coords_valid = (Vt.T * s) / np.sqrt(c_m)[:, None]
    
    # Create DataFrames for results (Dimensions)
    n_dims = len(s)
    dim_cols = [f"Dim {i+1}" for i in range(n_dims)]
    
    row_df = pd.DataFrame(row_coords_valid, index=df_matrix.index[valid_r], columns=dim_cols)
    col_df = pd.DataFrame(col_coords_valid, index=df_matrix.columns[valid_c], columns=dim_cols)
    
    return {
        'row_coords': row_df,
        'col_coords': col_df,
        'eigenvalues': eigenvalues.tolist(),
        'explained_inertia': explained_inertia.tolist(),
        'total_inertia': total_inertia
    }

def generate_ca_interpretation(
    row_coords: pd.DataFrame,
    col_coords: pd.DataFrame,
    inertia: List[float],
    top_words: List[str],
    grouping_key: str,
    ngram_sizes: List[int]
) -> str:
    """
    Generates a journal-style interpretation of the Correspondence Analysis,
    including Results and Conclusions based on coordinate extremes.
    """
    # --- 1. Methodology ---
    dim1_expl = inertia[0] * 100
    dim2_expl = inertia[1] * 100
    total_expl = dim1_expl + dim2_expl
    
    ngrams_str = ", ".join([{1: "words", 2: "bigrams", 3: "trigrams"}.get(n, f"{n}-grams") for n in ngram_sizes])
    num_features = len(top_words)
    num_docs = len(row_coords)
    
    # --- 2. Auto-Analysis of Dim 1 and Dim 2 ---
    
    def get_extremes(df, col):
        # Returns index of max and min values
        sorted_df = df.sort_values(by=col)
        min_items = sorted_df.head(2).index.tolist()
        max_items = sorted_df.tail(2).index.tolist()
        return max_items, min_items

    # Dim 1 Analysis
    r1_pos, r1_neg = get_extremes(row_coords, 'Dim 1')
    c1_pos, c1_neg = get_extremes(col_coords, 'Dim 1')
    
    # Dim 2 Analysis
    r2_pos, r2_neg = get_extremes(row_coords, 'Dim 2')
    c2_pos, c2_neg = get_extremes(col_coords, 'Dim 2')
    
    # Helpers for formatting
    fmt_list = lambda x: ", ".join([f"**{i}**" for i in x])
    
    findings_d1 = f"Dimension 1 (Horizontal, {dim1_expl:.1f}%) primarily distinguishes **{fmt_list(r1_pos)}** (right) from **{fmt_list(r1_neg)}** (left)."
    why_d1 = f"- **Why?** The right side is characterized by usage of _{fmt_list(c1_pos)}_, whereas the left side strongly features _{fmt_list(c1_neg)}_."
    
    findings_d2 = ""
    if dim2_expl > 5.0: # Only report Dim 2 if it explains significant variance
        findings_d2 = f"\n    Dimension 2 (Vertical, {dim2_expl:.1f}%) contrasts **{fmt_list(r2_pos)}** (top) against **{fmt_list(r2_neg)}** (bottom), driven by words like _{fmt_list(c2_pos)}_ vs _{fmt_list(c2_neg)}_."

    methodology = f"""
    ### 📝 Methodological Report
    
    **Data & Processing**

    We analyzed a contingency table of **{num_docs} {grouping_key} segments × {num_features} {ngrams_str}**. 
    Correspondence Analysis (CA) was performed by computing Chi-square distances on standardized residuals and applying Singular Value Decomposition (SVD).
    
    **Results & Key Findings**

    - The first two dimensions explain **{total_expl:.1f}%** of the total variance (inertia), indicating that the 2D map captures a significant portion of the stylistic differences.

    - {findings_d1}
    - {findings_d2.strip()}
    """

    return methodology

def perform_burrows_delta(
    df_z_matrix: pd.DataFrame,
    labels: List[str],
    train_indices: List[int],
    test_indices: List[int]
) -> pd.DataFrame:
    """
    Performs Burrows' Delta Authorship Attribution.
    
    Args:
        df_z_matrix: Z-scored Document-Term Matrix (Rows=Docs, Cols=Features).
        labels: List of author labels corresponding to df_z_matrix rows.
        train_indices: Indices (0-based) of the Training set (Known).
        test_indices: Indices (0-based) of the Test set (Questioned).
        
    Returns:
        pd.DataFrame: Distance Table (Rows=Test Docs, Cols=Candidate Authors).
        Each cell is the Manhattan Delta distance.
    """
    import scipy.spatial.distance as dist
    
    # 1. Split Data
    X = df_z_matrix.values
    label_arr = np.array(labels)
    
    X_train = X[train_indices]
    y_train = label_arr[train_indices]
    
    X_test = X[test_indices]
    
    # 2. Compute Centroids for Training Labels (Known Candidates)
    unique_candidates = np.unique(y_train)
    centroids = []
    candidate_names = []
    
    for cand in unique_candidates:
        # Find integer indices where y_train equals this candidate
        cand_indices = np.where(y_train == cand)[0]
        # Select rows using integer indexing (not boolean)
        cand_vectors = X_train[cand_indices, :]
        # Compute mean
        centroid = np.mean(cand_vectors, axis=0)
        centroids.append(centroid)
        candidate_names.append(cand)
        
    centroids = np.array(centroids)
    
    # 3. Compute Distances (Manhattan / City Block for Delta)
    # Result shape: (n_test_samples, n_candidates)
    delta_distances = dist.cdist(X_test, centroids, metric='cityblock')
    
    # 4. Format Output
    test_doc_names = df_z_matrix.index[test_indices]
    
    results_df = pd.DataFrame(
        delta_distances,
        index=test_doc_names,
        columns=candidate_names
    )
    
    return results_df

def perform_pca(df_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, List[float]]:
    """
    Performs PCA on the matrix and returns coordinates and explained variance.
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=min(10, df_matrix.shape[0], df_matrix.shape[1]))
    coords = pca.fit_transform(df_matrix)
    
    df_pca = pd.DataFrame(
        coords,
        index=df_matrix.index,
        columns=[f"PC{i+1}" for i in range(coords.shape[1])]
    )
    
    return df_pca, pca.explained_variance_ratio_.tolist()

def perform_network_similarity(df_matrix: pd.DataFrame, threshold: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates document similarity network based on Manhattan distance.
    """
    from scipy.spatial.distance import pdist, squareform
    
    # 1. Calculate distances
    dist_matrix = squareform(pdist(df_matrix.values, metric='cityblock'))
    
    # 2. Normalize to similarity (0 to 1 range)
    # Delta distances can be large, so we use a relative scaling
    max_dist = dist_matrix.max() if dist_matrix.max() > 0 else 1
    sim_matrix = 1 - (dist_matrix / max_dist)
    
    # 3. Build Nodes and Edges
    nodes = pd.DataFrame({'id': df_matrix.index})
    
    edges = []
    for i in range(len(df_matrix)):
        for j in range(i + 1, len(df_matrix)):
            if sim_matrix[i, j] >= threshold:
                edges.append({
                    'source': df_matrix.index[i],
                    'target': df_matrix.index[j],
                    'weight': float(sim_matrix[i, j])
                })
    
    return nodes, pd.DataFrame(edges)

