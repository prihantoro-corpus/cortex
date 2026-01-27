import sys
import os
import duckdb
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.modules.keyword import generate_keyword_list

def setup_test_db(filename="test_target.duckdb"):
    con = duckdb.connect(filename)
    con.execute("CREATE TABLE corpus (token VARCHAR, _token_low VARCHAR)")
    
    # Target: 'apple' x 20, 'common' x 5
    data = []
    for _ in range(20): data.append(("apple", "apple"))
    for _ in range(5): data.append(("common", "common"))
    
    con.executemany("INSERT INTO corpus VALUES (?, ?)", data)
    con.close()
    return filename

def test_keyword_with_freq_list():
    db_path = setup_test_db()
    
    # Reference Freq List: 'apple': 10, 'common': 500, Total: 1000
    ref_df = pd.DataFrame([
        {'token': 'apple', 'freq': 10},
        {'token': 'common', 'freq': 500}
    ])
    total_ref = 1000
    
    print("Running generate_keyword_list with freq_list...")
    results = generate_keyword_list(
        target_db_path=db_path,
        ref_db_path=None,
        ref_freq_df=ref_df,
        ref_total_tokens=total_ref,
        min_freq=1
    )
    
    print(f"Results:\n{results}")
    
    assert not results.empty
    assert results.iloc[0]['token'] == 'apple'
    assert results.iloc[0]['Type'] == 'Positive'
    
    import os
    os.remove(db_path)
    print("Keyword Freq List Test PASSED!")

if __name__ == "__main__":
    test_keyword_with_freq_list()
