import duckdb
import pandas as pd
from core.modules.collocation import generate_collocation_results
from core.modules.ngram import generate_n_grams_v2

# Using the standard LIMOLA debug DB if it exists, or one we know works.
# The user's walkthrough says LIMOLA.xml works now.
# We need to find the DuckDB file for LIMOLA.xml.
# Usually it's in <CORPORA_DIR>/_db/LIMOLA.duckdb (or similar hash)
# For this test, I will mock the DB creation or find the existing one.
# To be safe and self-contained, I will Create a Temp DB with known data.

def create_test_db():
    con = duckdb.connect("test_restriction.duckdb")
    con.execute("CREATE TABLE corpus (id INTEGER, token VARCHAR, lemma VARCHAR, pos VARCHAR, speaker VARCHAR, year INTEGER, _token_low VARCHAR)")
    
    # Insert data:
    # Speaker A says "apple" 10 times. Speaker B says "apple" 0 times.
    # If we filter by Speaker A, "apple" count should be 10.
    # If we filter by Speaker B, "apple" count should be 0.
    
    data = []
    # Speaker A (Year 2000)
    for i in range(10):
        data.append((i, "apple", "apple", "NN", "A", 2000, "apple"))
    
    # Speaker B (Year 2005)
    for i in range(10, 20):
        data.append((i, "banana", "banana", "NN", "B", 2005, "banana"))
        
    con.executemany("INSERT INTO corpus VALUES (?, ?, ?, ?, ?, ?, ?)", data)
    con.close()
    return "test_restriction.duckdb"

def test_collocation_restriction(db_path):
    print("\n--- Testing Collocation Restriction ---")
    
    # 1. Unrestricted Search for "apple" (Should be 0 actually, collocations need window)
    # Let's add context. 
    # A: "eat apple" x 10
    # B: "eat banana" x 10
    
    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS corpus")
    con.execute("CREATE TABLE corpus (id INTEGER, token VARCHAR, lemma VARCHAR, pos VARCHAR, speaker VARCHAR, year INTEGER, _token_low VARCHAR)")
    
    data = []
    idx = 0
    # Speaker A: "I eat apple"
    for _ in range(10):
        data.append((idx, "I", "I", "PRP", "A", 2000, "i")); idx+=1
        data.append((idx, "eat", "eat", "VB", "A", 2000, "eat")); idx+=1
        data.append((idx, "apple", "apple", "NN", "A", 2000, "apple")); idx+=1
        
    # Speaker B: "I eat banana"
    for _ in range(10):
        data.append((idx, "I", "I", "PRP", "B", 2005, "i")); idx+=1
        data.append((idx, "eat", "eat", "VB", "B", 2005, "eat")); idx+=1
        data.append((idx, "banana", "banana", "NN", "B", 2005, "banana")); idx+=1
        
    con.executemany("INSERT INTO corpus VALUES (?, ?, ?, ?, ?, ?, ?)", data)
    con.close()
    
    # Query: Node "eat", Window 1. Should find "apple" (10) and "banana" (10) globally.
    print("Running Unrestricted...")
    df_unrestricted, freq, _ = generate_collocation_results(
        db_path, "eat", 1, 1, 100, False,
        corpus_stats={'token_counts': {'apple': 10, 'banana': 10}}
    )
    print(f"Unrestricted 'apple' obs: {df_unrestricted.loc[df_unrestricted['Collocate']=='apple', 'Observed'].values[0] if not df_unrestricted.empty else 0}")
    
    # Query: Node "eat", Filter Speaker='A'. Should find "apple" (10) and "banana" (0).
    print("Running Restricted (Speaker='A')...")
    xml_where = " AND speaker = ?"
    xml_params = ['A']
    
    df_restricted, freq, _ = generate_collocation_results(
        db_path, "eat", 1, 1, 100, False,
        corpus_stats={'token_counts': {'apple': 10, 'banana': 10}}, # Global stats
        xml_where_clause=xml_where,
        xml_params=xml_params
    )
    
    obs_apple = df_restricted.loc[df_restricted['Collocate']=='apple', 'Observed'].values[0] if not df_restricted.empty and 'apple' in df_restricted['Collocate'].values else 0
    obs_banana = df_restricted.loc[df_restricted['Collocate']=='banana', 'Observed'].values[0] if not df_restricted.empty and 'banana' in df_restricted['Collocate'].values else 0
    
    print(f"Restricted 'apple' obs: {obs_apple}")
    print(f"Restricted 'banana' obs: {obs_banana}")
    
    assert obs_apple == 10, "Apple should be found in Speaker A"
    assert obs_banana == 0, "Banana should NOT be found in Speaker A"
    print("[PASS] Collocation Restriction Logic Verified!")

def test_ngram_restriction(db_path):
    print("\n--- Testing N-Gram Restriction ---")
    # Same DB.
    # Unrestricted Bigrams "eat apple" -> 10, "eat banana" -> 10
    
    print("Running Unrestricted...")
    df_all = generate_n_grams_v2(db_path, 2, {}, False, "Test", skip_punctuation=False)
    freq_apple = df_all.loc[df_all['N-Gram']=='eat apple', 'Frequency'].values[0]
    print(f"Unrestricted 'eat apple': {freq_apple}")
    
    # Restricted Speaker='B'. "eat apple" -> 0, "eat banana" -> 10
    print("Running Restricted (Speaker='B')...")
    xml_where = " AND speaker = ?"
    xml_params = ['B']
    
    df_res = generate_n_grams_v2(db_path, 2, {}, False, "Test", xml_where_clause=xml_where, xml_params=xml_params, skip_punctuation=False)
    
    freq_apple_res = df_res.loc[df_res['N-Gram']=='eat apple', 'Frequency'].values[0] if not df_res.empty and 'eat apple' in df_res['N-Gram'].values else 0
    freq_banana_res = df_res.loc[df_res['N-Gram']=='eat banana', 'Frequency'].values[0] if not df_res.empty and 'eat banana' in df_res['N-Gram'].values else 0
    
    print(f"Restricted 'eat apple': {freq_apple_res}")
    print(f"Restricted 'eat banana': {freq_banana_res}")
    
    assert freq_apple_res == 0
    assert freq_banana_res == 10
    print("[PASS] N-Gram Restriction Logic Verified!")

if __name__ == "__main__":
    db = create_test_db()
    test_collocation_restriction(db)
    test_ngram_restriction(db)
    
    import os
    try: os.remove(db)
    except: pass
