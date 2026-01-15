import sys
import os
import duckdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.preprocessing.corpus_loader import load_monolingual_corpus_files
from core.config import CORPORA_DIR

def check_limola_values():
    limola_path = os.path.join(CORPORA_DIR, "LIMOLA.xml")
    
    with open(limola_path, 'rb') as f:
        # Re-load to get fresh DB
        result = load_monolingual_corpus_files([f], 'id', 'XML (Tagged)')
        
    db_path = result['db_path']
    con = duckdb.connect(db_path, read_only=True)
    
    cols = ['speaker', 'year', 'location', 'language']
    
    print(f"Checking values in DB: {db_path}")
    
    total_rows = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
    print(f"Total Rows (tokens): {total_rows}")
    
    for col in cols:
        try:
            # Check non-null count
            non_null = con.execute(f"SELECT count({col}) FROM corpus WHERE {col} IS NOT NULL AND {col} != 'NaN' AND {col} != ''").fetchone()[0]
            print(f"\nColumn '{col}': {non_null} non-null values")
            
            if non_null > 0:
                samples = con.execute(f"SELECT DISTINCT {col} FROM corpus WHERE {col} IS NOT NULL LIMIT 5").fetchall()
                print(f"  Samples: {[s[0] for s in samples]}")
            else:
                print(f"  [WARNING] Column is empty or all null!")
                
        except Exception as e:
            print(f"  Error checking {col}: {e}")

    con.close()

if __name__ == "__main__":
    check_limola_values()
