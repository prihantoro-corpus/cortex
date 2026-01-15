
import sqlite3
import pandas as pd
import duckdb
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.modules.collocation import generate_collocation_results

def test_wildcard_collocation():
    # Setup dummy database
    db_path = "test_wildcard.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    con = duckdb.connect(db_path)
    con.execute("""
    CREATE TABLE corpus (
        id INTEGER,
        token TEXT,
        _token_low TEXT,
        pos TEXT,
        lemma TEXT
    )
    """)
    
    # Data: "berikan bantuan kepada masyarakat", "berikan dukungan kepada warga"
    data = [
        (0, "berikan", "berikan", "VB", "beri"),
        (1, "bantuan", "bantuan", "NN", "bantu"),
        (2, "kepada", "kepada", "IN", "kepada"),
        (3, "masyarakat", "masyarakat", "NN", "masyarakat"),
        (4, "berikan", "berikan", "VB", "beri"),
        (5, "dukungan", "dukungan", "NN", "dukung"),
        (6, "kepada", "kepada", "IN", "kepada"),
        (7, "warga", "warga", "NN", "warga")
    ]
    for row in data:
        con.execute("INSERT INTO corpus VALUES (?, ?, ?, ?, ?)", row)
    
    con.close()
    
    # Test 1: word wildcard: ber*
    print("Testing wildcard: ber*")
    stats_df, freq, node = generate_collocation_results(
        db_path, "ber*", coll_window=2, mi_min_freq=1, max_collocates=10, is_raw_mode=False
    )
    print(f"Node: {node}, Freq: {freq}")
    display_cols = ['Collocate', 'Observed', 'LL']
    existing_cols = [c for c in display_cols if c in stats_df.columns]
    print(stats_df[existing_cols].head(3))

    # Test 2: lemma wildcard: [bant]*
    print("\nTesting lemma wildcard: [bant]*")
    stats_df, freq, node = generate_collocation_results(
        db_path, "[bant]*", coll_window=2, mi_min_freq=1, max_collocates=10, is_raw_mode=False
    )
    print(f"Node: {node}, Freq: {freq}")
    print(stats_df[existing_cols].head(3))

    # Test 3: POS wildcard: _V*
    print("\nTesting POS wildcard: _V*")
    stats_df, freq, node = generate_collocation_results(
        db_path, "_V*", coll_window=2, mi_min_freq=1, max_collocates=10, is_raw_mode=False
    )
    print(f"Node: {node}, Freq: {freq}")
    print(stats_df[existing_cols].head(3))

    # Test 4: multiword wildcard: ber* bantuan
    print("\nTesting multiword wildcard: ber* bantuan")
    stats_df, freq, node = generate_collocation_results(
        db_path, "ber* bantuan", coll_window=2, mi_min_freq=1, max_collocates=10, is_raw_mode=False
    )
    print(f"Node: {node}, Freq: {freq}")
    print(stats_df[existing_cols].head(3))

    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    test_wildcard_collocation()
