import sys
import os
import duckdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.preprocessing.xml_parser import get_xml_attribute_columns, is_integer_col
from core.preprocessing.corpus_loader import load_monolingual_corpus_files
from core.config import CORPORA_DIR

def simulate_filters_logic():
    limola_path = os.path.join(CORPORA_DIR, "LIMOLA.xml")
    with open(limola_path, 'rb') as f:
        result = load_monolingual_corpus_files([f], 'id', 'XML (Tagged)')
    
    db_path = result['db_path']
    con = duckdb.connect(db_path, read_only=True)
    
    print(f"DB Path: {db_path}")
    
    # 1. Test get_xml_attribute_columns
    attr_cols = get_xml_attribute_columns(con)
    print(f"Detected Attribute Columns: {attr_cols}")
    
    # Check what PRAGMA returns exactly
    cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
    all_cols = [c[1] for c in cols_info]
    print(f"ALL DB Columns: {all_cols}")
    
    exclusion = {'token', 'pos', 'lemma', 'sent_id', '_token_low', 'id', 'filename'}
    ignored = [c for c in all_cols if c in exclusion]
    print(f"Ignored columns (Standard): {ignored}")
    
    if not attr_cols:
        print("FAILED: get_xml_attribute_columns returned empty!")
        con.close()
        return

    # 2. Test Value Extraction for each col
    for attr in attr_cols:
        print(f"\nChecking attr: {attr}")
        if is_integer_col(con, attr):
            print("  Type: Integer")
            try:
                stats = con.execute(f"SELECT MIN(CAST({attr} AS BIGINT)), MAX(CAST({attr} AS BIGINT)) FROM corpus").fetchone()
                print(f"  Range: {stats}")
            except Exception as e:
                print(f"  Error checking int stats: {e}")
        else:
            print("  Type: String/Other")
            try:
                # The logic in filters.py
                rows = con.execute(f"SELECT DISTINCT {attr} FROM corpus WHERE {attr} IS NOT NULL ORDER BY {attr}").fetchall()
                unique_vals = [r[0] for r in rows]
                cleaned_vals = [str(v) for v in unique_vals if str(v).strip() and str(v).lower() != 'nan']
                print(f"  Raw Unique Count: {len(unique_vals)}")
                print(f"  Cleaned Unique Values: {cleaned_vals[:10]}")
                
                if not cleaned_vals:
                    print("  WARNING: No valid values found for filter!")
            except Exception as e:
                print(f"  Error checking string values: {e}")

    con.close()

if __name__ == "__main__":
    simulate_filters_logic()
