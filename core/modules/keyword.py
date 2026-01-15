import duckdb
import pandas as pd
import numpy as np
from core.statistics.association import safe_ll_term, vec_sig

def generate_keyword_list(target_db_path, ref_db_path=None, target_xml_where="", target_xml_params=[], ref_xml_where="", ref_xml_params=[], min_freq=3, ref_freq_df=None, ref_total_tokens=0):
    """
    Generates a keyword list by comparing the target corpus against a reference corpus.
    Calculates Log Likelihood (LL) and Log Ratio.
    
    Args:
        target_db_path (str): Path to target corpus DB.
        ref_db_path (str, optional): Path to reference corpus DB.
        target_xml_where (str): XML filter clause for target.
        ref_xml_where (str): XML filter clause for reference.
        min_freq (int): Minimum frequency in target corpus to be considered.
        ref_freq_df (pd.DataFrame, optional): Pre-calculated frequency list for reference.
        ref_total_tokens (int, optional): Total token count for reference if ref_freq_df is used.
    """
    if not target_db_path or (not ref_db_path and ref_freq_df is None):
        return pd.DataFrame()

    try:
        # 1. Get Target Counts
        con_t = duckdb.connect(target_db_path, read_only=True)
        # Check standard columns to determine token column validity (e.g. exclude punctuation)
        # For simplicity, we assume _token_low or token exists.
        
        sql_t = f"""
        SELECT _token_low as token, count(*) as freq 
        FROM corpus 
        WHERE NOT regexp_matches(_token_low, '^[[:punct:]]+$') 
          AND NOT regexp_matches(_token_low, '^[0-9]+$')
          {target_xml_where}
        GROUP BY 1
        HAVING count(*) >= ?
        """
        df_target = con_t.execute(sql_t, target_xml_params + [min_freq]).fetch_df()
        
        if target_xml_where:
            total_target = con_t.execute(f"SELECT count(*) FROM corpus WHERE 1=1 {target_xml_where}", target_xml_params).fetchone()[0]
        else:
            total_target = con_t.execute("SELECT count(*) FROM corpus").fetchone()[0]
        
        con_t.close()
        
        if df_target.empty:
            return pd.DataFrame()
            
        # 2. Get Reference Counts
        if ref_freq_df is not None:
            df_ref = ref_freq_df.copy()
            total_ref = ref_total_tokens if ref_total_tokens > 0 else df_ref['freq'].sum()
        else:
            con_r = duckdb.connect(ref_db_path, read_only=True)
            sql_r = f"""
            SELECT _token_low as token, count(*) as freq 
            FROM corpus 
            WHERE NOT regexp_matches(_token_low, '^[[:punct:]]+$') 
              AND NOT regexp_matches(_token_low, '^[0-9]+$')
              {ref_xml_where}
            GROUP BY 1
            """
            df_ref = con_r.execute(sql_r, ref_xml_params).fetch_df()
            
            if ref_xml_where:
                total_ref = con_r.execute(f"SELECT count(*) FROM corpus WHERE 1=1 {ref_xml_where}", ref_xml_params).fetchone()[0]
            else:
                total_ref = con_r.execute("SELECT count(*) FROM corpus").fetchone()[0]
                
            con_r.close()
        
        # 3. Merge and Calculate
        # We need counts for ALL tokens in target, joined with ref.
        # Tokens in target but not in ref will have Ref freq = 0 (handled by merge how='left')
        
        merged = pd.merge(df_target, df_ref, on='token', how='left', suffixes=('_t', '_r'))
        merged['freq_r'] = merged['freq_r'].fillna(0)
        
        # Variables for calculation
        O1 = merged['freq_t']           # Obs in Target
        O2 = merged['freq_r']           # Obs in Ref
        N1 = total_target               # Total Target
        N2 = total_ref                  # Total Ref
        
        # --- Log Likelihood ---
        # Contingency Table:
        #        Target   Ref     Total
        # Word   O1       O2      O1+O2
        # Other  N1-O1    N2-O2   (N1+N2)-(O1+O2)
        # Total  N1       N2      N1+N2
        
        E1 = N1 * (O1 + O2) / (N1 + N2)
        E2 = N2 * (O1 + O2) / (N1 + N2)
        
        # LL = 2 * (O1 * ln(O1/E1) + O2 * ln(O2/E2))
        # We use safe calculation to handle zeros
        
        # Helper function for vectorized calc? We imported valid functions from association.py
        # But we need to handle arrays here.
        
        def safe_log(x):
            return np.log(np.where(x > 0, x, 1)) # log(1) = 0
            
        t1 = O1 * np.log(np.where(O1 > 0, O1/E1, 1)) # If O1=0, term is 0
        t2 = O2 * np.log(np.where(O2 > 0, O2/E2, 1)) # If O2=0, term is 0
        
        # Correction: O * log(O/E). If O=0, limit is 0. If E=0 (impossible here as O1+O2 >= min_freq > 0), undefined.
        # Wait, if O1=0?? We filtered min_freq >= 3, so O1 > 0 always.
        # But O2 can be 0. If O2=0, E2 > 0. O2/E2=0. log(0) undefined.
        # We need to handle O=0.
        
        term1 = np.where(O1 > 0, O1 * np.log(O1/E1), 0)
        term2 = np.where(O2 > 0, O2 * np.log(O2/E2), 0)
        
        merged['LL'] = 2 * (term1 + term2)
        
        # --- Log Ratio ---
        # log2( (O1/N1) / (O2/N2) )
        # Needs smoothing for O2=0
        smooth = 0.5
        norm_t = (O1 + smooth) / (N1 + smooth) # Optional smoothing
        # Common Log Ratio formula usually just handles zero by adding 0.5 to counts or similar.
        # Standard: Log2 ( (FreqA / N_A) / (FreqB / N_B) )
        # If FreqB is 0, it explodes. We use simple smoothing.
        
        rel_t = (O1) / N1
        rel_r = (O2) / N2
        
        # To avoid infinity, if freq is 0, treat as 0.5 (Kilgarriff smoothing)
        # Actually proper Log Ratio requires smoothing.
        # Binary Log Ratio: log2 ( (O1 / N1) / (O2 / N2) )
        
        rel_t_smooth = (O1 + 0.5) / (N1 + 0.5)
        rel_r_smooth = (O2 + 0.5) / (N2 + 0.5)
        
        merged['LogRatio'] = np.log2(rel_t_smooth / rel_r_smooth)
        
        # Significance
        merged['Significance'] = merged['LL'].apply(vec_sig)
        
        # Determine Positive vs Negative
        # Positive: More frequent (relatively) in Target (LogRatio > 0)
        # Negative: More frequent (relatively) in Ref (LogRatio < 0)
        # Actually, since we only query words present in Target (LEFT JOIN), we generally only find Positive keywords.
        # BUT, if a word is in Target but MUCH MORE frequent in Ref, it will have negative Log Ratio.
        # To find 'Negative' keywords (words ABSENT in Target but present in Ref), we would need a FULL OUTER JOIN or query Ref separately.
        # User Requirement: "divide into 2: positive and negative keywords".
        # Negative keywords are typically words that are unusually INFREQUENT in the target compared to ref.
        # So yes, words present in Target but with negative Log Ratio are "negative" keywords (underused).
        # Words NOT in Target (or < min_freq) but high in Ref are also Negative keywords.
        # To get proper Negative keywords list (missing words), we should ideally start with a list of ALL words from BOTH corpora.
        
        # Revised Strategy for Negative Keywords:
        # We need the union of vocabularies.
        
    except Exception as e:
        print(f"Keyword Gen Error: {e}")
        return pd.DataFrame()
        
    # Re-do with FULL Merge to capture Negative Keywords correctly
    try:
        # We need full outer join.
        # But we already fetched df_target and df_ref.
        # Let's merge them fully.
        
        merged = pd.merge(df_target, df_ref, on='token', how='outer', suffixes=('_t', '_r'))
        merged['freq_t'] = merged['freq_t'].fillna(0)
        merged['freq_r'] = merged['freq_r'].fillna(0)
        
        O1 = merged['freq_t']
        O2 = merged['freq_r']
        N1 = total_target
        N2 = total_ref
        
        E1 = N1 * (O1 + O2) / (N1 + N2)
        E2 = N2 * (O1 + O2) / (N1 + N2)
        
        term1 = np.where(O1 > 0, O1 * np.log(O1/E1), 0)
        term2 = np.where(O2 > 0, O2 * np.log(O2/E2), 0)
        merged['LL'] = 2 * (term1 + term2)
        
        rel_t_smooth = (O1 + 0.5) / (N1 + 0.5)
        rel_r_smooth = (O2 + 0.5) / (N2 + 0.5)
        merged['LogRatio'] = np.log2(rel_t_smooth / rel_r_smooth)
        merged['Significance'] = merged['LL'].apply(vec_sig)
        
        # Categorize
        merged['Type'] = np.where(merged['LogRatio'] > 0, 'Positive', 'Negative')
        
        # Sort by LL (absolute strength of keyness)
        merged = merged.sort_values("LL", ascending=False)
        
        return merged
        
    except Exception as e:
        print(f"Keyword calc error: {e}")
        return pd.DataFrame()

def generate_grouped_keyword_list(target_db_path, group_by_col, ref_db_path=None, target_xml_where="", target_xml_params=[], ref_xml_where="", ref_xml_params=[], min_freq=3, ref_freq_df=None, ref_total_tokens=0):
    """
    Generates a dictionary of keyword lists, grouped by a specific column (e.g., filename, author).
    Returns: { 'group_value': pd.DataFrame }
    """
    if not target_db_path or not group_by_col:
        return {}

    results = {}
    
    try:
        con_t = duckdb.connect(target_db_path, read_only=True)
        
        # 1. Get Distinct Groups
        # Verify column exists first
        try:
             con_t.execute(f"SELECT {group_by_col} FROM corpus LIMIT 1")
        except:
             print(f"Column {group_by_col} not found in corpus.")
             con_t.close()
             return {}

        groups = [r[0] for r in con_t.execute(f"SELECT DISTINCT {group_by_col} FROM corpus WHERE {group_by_col} IS NOT NULL").fetchall()]
    
        # 2. Get Global Reference Counts (Once)
        if ref_freq_df is not None:
            df_ref = ref_freq_df.copy()
            total_ref = ref_total_tokens if ref_total_tokens > 0 else df_ref['freq'].sum()
        else:
            con_r = duckdb.connect(ref_db_path, read_only=True)
            sql_r = f"""
            SELECT _token_low as token, count(*) as freq 
            FROM corpus 
            WHERE NOT regexp_matches(_token_low, '^[[:punct:]]+$') 
              AND NOT regexp_matches(_token_low, '^[0-9]+$')
              {ref_xml_where}
            GROUP BY 1
            """
            df_ref = con_r.execute(sql_r, ref_xml_params).fetch_df()
            
            if ref_xml_where:
                total_ref = con_r.execute(f"SELECT count(*) FROM corpus WHERE 1=1 {ref_xml_where}", ref_xml_params).fetchone()[0]
            else:
                total_ref = con_r.execute("SELECT count(*) FROM corpus").fetchone()[0]
            con_r.close()

        # 3. Iterate Groups
        for group_val in groups:
            # Safe parameterization for group value
            # We append the group condition to the existing XML where clause
            group_where = f" AND {group_by_col} = ?"
            full_where = target_xml_where + group_where
            full_params = target_xml_params + [group_val]
            
            sql_t = f"""
            SELECT _token_low as token, count(*) as freq 
            FROM corpus 
            WHERE NOT regexp_matches(_token_low, '^[[:punct:]]+$') 
              AND NOT regexp_matches(_token_low, '^[0-9]+$')
              {full_where}
            GROUP BY 1
            HAVING count(*) >= ?
            """
            
            df_target = con_t.execute(sql_t, full_params + [min_freq]).fetch_df()
            
            if df_target.empty:
                continue
                
            total_target = con_t.execute(f"SELECT count(*) FROM corpus WHERE 1=1 {full_where}", full_params).fetchone()[0]
            
            # Merge and Calculate
            merged = pd.merge(df_target, df_ref, on='token', how='outer', suffixes=('_t', '_r'))
            merged['freq_t'] = merged['freq_t'].fillna(0)
            merged['freq_r'] = merged['freq_r'].fillna(0)
            
            O1 = merged['freq_t']
            O2 = merged['freq_r']
            N1 = total_target
            N2 = total_ref
            
            E1 = N1 * (O1 + O2) / (N1 + N2)
            E2 = N2 * (O1 + O2) / (N1 + N2)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                term1 = np.where(O1 > 0, O1 * np.log(O1/E1), 0)
                term2 = np.where(O2 > 0, O2 * np.log(O2/E2), 0)
                merged['LL'] = 2 * (term1 + term2)
                
                rel_t_smooth = (O1 + 0.5) / (N1 + 0.5)
                rel_r_smooth = (O2 + 0.5) / (N2 + 0.5)
                merged['LogRatio'] = np.log2(rel_t_smooth / rel_r_smooth)
            
            merged['Significance'] = merged['LL'].apply(vec_sig)
            merged['Type'] = np.where(merged['LogRatio'] > 0, 'Positive', 'Negative')
            merged = merged.sort_values("LL", ascending=False)
            
            results[group_val] = merged
            
        con_t.close()
        return results
        
    except Exception as e:
        print(f"Grouped Keyword Gen Error: {e}")
        return {}
