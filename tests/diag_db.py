import duckdb
import os

corpus_dir = 'data/corpora'
if not os.path.exists(corpus_dir):
    print(f"Directory {corpus_dir} not found.")
else:
    db_files = [f for f in os.listdir(corpus_dir) if f.endswith('.db')]
    print(f"DB Files: {db_files}")
    for dbf in db_files:
        db_path = os.path.join(corpus_dir, dbf)
        try:
            con = duckdb.connect(db_path)
            cols = [c[1] for c in con.execute('PRAGMA table_info(corpus)').fetchall()]
            print(f"\n{dbf} columns: {cols}")
            
            # Check for PN-related columns
            pn_cols = [c for c in cols if 'pn' in c]
            if pn_cols:
                print(f"PN related columns in {dbf}: {pn_cols}")
                # Sample counts
                res = con.execute(f"SELECT count(*), count(in_pn), count(in_pn_start), count(pn_len) FROM corpus").fetchone()
                print(f"Counts (Total, in_PN, in_PN_start, PN_len): {res}")
                
                # Check for lowercase normalization
                sample = con.execute("SELECT token FROM corpus WHERE in_pn = TRUE LIMIT 3").fetchall()
                print(f"Sample PN tokens: {sample}")
            con.close()
        except Exception as e:
            print(f"Error checking {dbf}: {e}")
