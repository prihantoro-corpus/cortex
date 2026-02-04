import duckdb
import os
import tempfile
import glob

def check_latest():
    temp_dir = tempfile.gettempdir()
    dbs = glob.glob(os.path.join(temp_dir, 'corpus_*.duckdb'))
    if not dbs:
        print("No corpus databases found in temp directory.")
        return
    
    dbs.sort(key=os.path.getmtime, reverse=True)
    latest_db = dbs[0]
    print(f"Inspecting latest DB: {latest_db}")
    
    try:
        con = duckdb.connect(latest_db)
        # Check tables
        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        print(f"Tables: {tables}")
        
        if 'corpus' not in tables:
            print("No 'corpus' table found.")
            con.close()
            return
            
        # Check columns
        cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
        print(f"Columns in 'corpus': {cols}")
        
        # Look for XML metadata
        pn_cols = [c for c in cols if 'pn' in c.lower()]
        print(f"PN-related columns: {pn_cols}")
        
        # Sample data
        limit = 5
        print(f"\nSample data from 'corpus' (first {limit} rows):")
        sample = con.execute(f"SELECT id, token, pos, lemma FROM corpus LIMIT {limit}").fetch_df()
        print(sample)
        
        if 'in_pn' in cols:
            print("\nRows where in_pn is True:")
            pn_rows = con.execute("SELECT id, token, in_pn, in_pn_start, pn_len FROM corpus WHERE in_pn = True LIMIT 10").fetch_df()
            print(pn_rows)
            
            total_pn = con.execute("SELECT count(*) FROM corpus WHERE in_pn = True").fetchone()[0]
            start_pn = con.execute("SELECT count(*) FROM corpus WHERE in_pn_start = True").fetchone()[0] if 'in_pn_start' in cols else "N/A"
            print(f"\nStats: Total tokens in PN={total_pn}, Start tokens={start_pn}")
        else:
            print("\n'in_pn' column not found! The corpus might not have been parsed as XML or the tag is different.")
            
        con.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_latest()
