import sys
sys.path.insert(0, 'c:\\Users\\priha\\Documents\\cortex')

# Force reload to pick up changes
import importlib
if 'core.preprocessing.xml_parser' in sys.modules:
    importlib.reload(sys.modules['core.preprocessing.xml_parser'])
if 'core.preprocessing.corpus_loader' in sys.modules:
    importlib.reload(sys.modules['core.preprocessing.corpus_loader'])

from core.preprocessing.corpus_loader import load_built_in_corpus
import duckdb

print("Testing built-in corpus loader...")
result = load_built_in_corpus("XML Tag Demo (EN)", "english/xml_tag_demo.xml")

if result.get('error'):
    print(f"ERROR: {result['error']}")
else:
    db_path = result['db_path']
    print(f"Success! DB Path: {db_path}")
    
    # Check token count
    con = duckdb.connect(db_path, read_only=True)
    count = con.execute("SELECT COUNT(*) FROM corpus").fetchone()[0]
    print(f"Total tokens: {count}")
    
    # Check for XML tag columns
    cols = con.execute("PRAGMA table_info(corpus)").fetchall()
    xml_cols = [c[1] for c in cols if c[1].startswith('in_') or ('_' in c[1] and not c[1].startswith('_'))]
    print(f"XML tag columns found: {len(xml_cols)}")
    print(f"Sample columns: {xml_cols[:10]}")
    
    # Test a query
    pn_tokens = con.execute("SELECT token FROM corpus WHERE in_PN = TRUE LIMIT 5").fetchall()
    print(f"\nSample <PN> tokens: {[t[0] for t in pn_tokens]}")
    
    con.close()
    print("\n✓ All checks passed!")
