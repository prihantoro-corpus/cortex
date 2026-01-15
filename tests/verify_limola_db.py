import sys
import os
import duckdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.preprocessing.corpus_loader import load_monolingual_corpus_files
from core.config import CORPORA_DIR

def test_limola_db_schema():
    limola_path = os.path.join(CORPORA_DIR, "LIMOLA.xml")
    
    if not os.path.exists(limola_path):
        print(f"Skipping: {limola_path} not found.")
        return

    print(f"Loading {limola_path}...")
    with open(limola_path, 'rb') as f:
        # Mock file object with name
        # f is good but load_monolingual... expects a list of file-like objects with .name
        # File object 'f' has .name attribute.
        result = load_monolingual_corpus_files([f], 'id', 'XML (Tagged)')
        
    if result.get('error'):
        print(f"Error: {result['error']}")
        return
        
    db_path = result['db_path']
    print(f"DB created at: {db_path}")
    
    try:
        con = duckdb.connect(db_path, read_only=True)
        cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
        col_names = [c[1] for c in cols_info]
        print(f"DB Columns: {col_names}")
        
        expected = ['speaker', 'year', 'location', 'language'] 
        # Note: 'n' is excluded by parser logic now, so it shouldn't be here.
        
        missing = [c for c in expected if c not in col_names]
        
        if missing:
            print(f"FAILED. Missing columns in DB: {missing}")
        else:
            print("SUCCESS. All metadata columns present in DB.")
            
        con.close()
    except Exception as e:
        print(f"DB Inspection fail: {e}")

if __name__ == "__main__":
    test_limola_db_schema()
