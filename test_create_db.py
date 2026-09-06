import duckdb
import pandas as pd

# Create dummy db
con = duckdb.connect(r'C:\Users\priha\Documents\cortex\dummy.duckdb')

# Create corpus table
con.execute("""
    CREATE TABLE IF NOT EXISTS corpus (
        id INTEGER,
        _token_low VARCHAR,
        pos VARCHAR,
        lemma VARCHAR
    )
""")

# Insert some data
data = [
    (1, "run", "VB", "run"),
    (2, "the", "DT", "the"),
    (3, "fast", "JJ", "fast"),
    (4, "car", "NN", "car"),
    (5, "into", "IN", "into"),
    (6, "the", "DT", "the"),
    (7, "wall", "NN", "wall"),
]
con.executemany("INSERT INTO corpus VALUES (?, ?, ?, ?)", data)

# Close
con.close()
