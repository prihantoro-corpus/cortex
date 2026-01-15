import duckdb
import pandas as pd
import numpy as np

# Mock implementation of what will be in keyword.py
def mock_generate_grouped_keyword_list(db_path, ref_db_path, group_by_col):
    """
    Prototype logic for generating grouped keywords.
    """
    con = duckdb.connect(db_path, read_only=True)
    
    # 1. Get List of Groups
    groups = [r[0] for r in con.execute(f"SELECT DISTINCT {group_by_col} FROM corpus WHERE {group_by_col} IS NOT NULL").fetchall()]
    print(f"Groups found: {groups}")
    
    results = {}
    
    # 2. Get Global Reference Counts (Mocking Reference DB via a temp table in same DB for simplicity in this test)
    # in real app, we query ref_db_path. Here we assume ref data is in a separate table 'ref_corpus'
    con.execute("CREATE TEMP TABLE ref_corpus AS SELECT * FROM corpus") # Just cloning for test simplicity, usually it's different
    # Modify ref to make it distinct? 
    # Let's say Ref has "common" words.
    
    # For this test, let's assume Ref is provided as a Frequency DataFrame (e.g. BNC)
    # Ref Total: 1000. 'apple': 10, 'banana': 10, 'common': 500
    ref_freq_df = pd.DataFrame([
        {'token': 'apple', 'freq': 10},
        {'token': 'banana', 'freq': 10},
        {'token': 'common', 'freq': 500}
    ])
    total_ref = 1000
    
    for group in groups:
        print(f"Processing group: {group}")
        # 3. Get Target Counts for this group
        sql_t = f"""
        SELECT token, count(*) as freq 
        FROM corpus 
        WHERE {group_by_col} = ?
        GROUP BY 1
        """
        df_target = con.execute(sql_t, [group]).fetch_df()
        
        if df_target.empty:
            results[group] = pd.DataFrame()
            continue
            
        total_target = con.execute(f"SELECT count(*) FROM corpus WHERE {group_by_col} = ?", [group]).fetchone()[0]
        
        # 4. Keyword Calc (Simplified LL)
        merged = pd.merge(df_target, ref_freq_df, on='token', how='left', suffixes=('_t', '_r'))
        merged['freq_r'] = merged['freq_r'].fillna(0)
        
        O1 = merged['freq_t']
        O2 = merged['freq_r']
        N1 = total_target
        N2 = total_ref
        
        E1 = N1 * (O1 + O2) / (N1 + N2)
        E2 = N2 * (O1 + O2) / (N1 + N2)
        
        # Log Likelihood
        term1 = np.where(O1 > 0, O1 * np.log(O1/E1), 0)
        term2 = np.where(O2 > 0, O2 * np.log(O2/E2), 0)
        merged['LL'] = 2 * (term1 + term2)
        
        # Log Ratio
        rel_t_smooth = (O1 + 0.5) / (N1 + 0.5)
        rel_r_smooth = (O2 + 0.5) / (N2 + 0.5)
        merged['LogRatio'] = np.log2(rel_t_smooth / rel_r_smooth)
        
        results[group] = merged.sort_values('LL', ascending=False)
        
    con.close()
    return results

def setup_test_db(filename="test_grouped_kw.duckdb"):
    con = duckdb.connect(filename)
    con.execute("CREATE TABLE corpus (token VARCHAR, file_id VARCHAR, domain VARCHAR)")
    
    # File 1 (Fruit): apple x 20, common x 5
    data = []
    for _ in range(20): data.append(("apple", "file1.txt", "fruit"))
    for _ in range(5): data.append(("common", "file1.txt", "fruit"))
    
    # File 2 (Tech): python x 20, common x 5
    for _ in range(20): data.append(("python", "file2.txt", "tech"))
    for _ in range(5): data.append(("common", "file2.txt", "tech"))
        
    con.executemany("INSERT INTO corpus VALUES (?, ?, ?)", data)
    con.close()
    return filename

def test_grouped_keywords():
    db_path = setup_test_db()
    
    print("\n--- Testing Group By File ---")
    results_file = mock_generate_grouped_keyword_list(db_path, None, "file_id")
    
    # Check File 1
    # Expect 'apple' to be high keyness vs Ref (which has 'apple':10, 'common':500)
    # File 1 has 'apple':20.
    # Ref has 'apple':10.
    # Relative freq in File 1: 20/25 = 0.8
    # Relative freq in Ref: 10/1000 = 0.01
    # Should be HUGE positive keyness.
    
    df_f1 = results_file['file1.txt']
    top_kw_f1 = df_f1.iloc[0]['token']
    print(f"File1 Top KW: {top_kw_f1} (LL: {df_f1.iloc[0]['LL']:.2f})")
    
    assert top_kw_f1 == 'apple'
    assert 'python' not in df_f1['token'].values # Python is not in file1
    
    # Check File 2
    df_f2 = results_file['file2.txt']
    top_kw_f2 = df_f2.iloc[0]['token']
    print(f"File2 Top KW: {top_kw_f2} (LL: {df_f2.iloc[0]['LL']:.2f})")
    
    assert top_kw_f2 == 'python'
    
    print("\n--- Testing Group By Attribute (Domain) ---")
    results_domain = mock_generate_grouped_keyword_list(db_path, None, "domain")
    
    df_fruit = results_domain['fruit']
    print(f"Fruit Domain Top KW: {df_fruit.iloc[0]['token']}")
    assert df_fruit.iloc[0]['token'] == 'apple'
    
    print("[PASS] Grouped Keyword Logic Verified")
    
    import os
    try: os.remove(db_path)
    except: pass

if __name__ == "__main__":
    test_grouped_keywords()
