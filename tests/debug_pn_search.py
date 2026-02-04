import duckdb
import os
import tempfile
import sys

# Simulation of the search logic
def parse_query_term(term):
    import re
    xml_tag_match = re.match(r'<(\w+)(?:\s+(.+?))?>', term)
    if xml_tag_match:
        tag_name = xml_tag_match.group(1)
        attrs_str = xml_tag_match.group(2)
        attrs = {}
        if attrs_str:
            attr_pattern = r'(\w+)=([\"\'\'])([^\"\'\']*)\2'
            for match in re.finditer(attr_pattern, attrs_str):
                attrs[match.group(1)] = match.group(3)
        return {'type': 'xml_tag', 'tag': tag_name, 'attrs': attrs}
    return None

# Path to the demo corpus (it might be in temp or CORPORA_DIR)
# I'll try to find any recently created duckdb files in temp, or look at the registered CORPORA_DIR
CORPORA_DIR = 'c:\\Users\\priha\\Documents\\cortex\\corpora'
demo_rel_path = 'english/xml_tag_demo.xml'

# Since it's loaded as a built-in, it lives in a temp DB after loading.
# But for debugging, I can just parse it again into a temporary DB.

sys.path.insert(0, 'c:\\Users\\priha\\Documents\\cortex')
from core.preprocessing.xml_parser import parse_xml_content_to_df
from core.preprocessing.cleaning import sanitize_xml_content

with open(os.path.join(CORPORA_DIR, demo_rel_path), 'r', encoding='utf-8') as f:
    content = f.read()

cleaned = sanitize_xml_content(content)
result = parse_xml_content_to_df(cleaned, preserve_inline_tags=True)

if 'error' in result:
    print(f"Error: {result['error']}")
    sys.exit(1)

import pandas as pd
df = pd.DataFrame(result['df_data'])
db_path = os.path.join(tempfile.gettempdir(), "debug_pn.duckdb")
if os.path.exists(db_path): os.remove(db_path)

con = duckdb.connect(db_path)
con.execute("CREATE TABLE corpus AS SELECT * FROM df")

print("Checking for in_PN column presence...")
cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
print(f"Columns: {cols}")

if 'in_PN' in cols:
    count = con.execute("SELECT count(*) FROM corpus WHERE in_PN = TRUE").fetchone()[0]
    print(f"Tokens where in_PN = TRUE: {count}")
    
    if count > 0:
        sample = con.execute("SELECT token, in_PN FROM corpus WHERE in_PN = TRUE LIMIT 5").fetch_df()
        print("Sample PN tokens:")
        print(sample)
    else:
        print("!! NO TOKENS FOUND WITH in_PN = TRUE")
else:
    print("!! COLUMN in_PN NOT FOUND")

# Test query parsing
q = "<PN>"
parsed = parse_query_term(q)
print(f"Parsed query '{q}': {parsed}")

if parsed and parsed['type'] == 'xml_tag':
    tag_name = parsed['tag']
    sql_cond = f"in_{tag_name} = TRUE"
    final_count = con.execute(f"SELECT count(*) FROM corpus WHERE {sql_cond}").fetchone()[0]
    print(f"Count with SQL '{sql_cond}': {final_count}")

con.close()
