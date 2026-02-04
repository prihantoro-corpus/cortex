import sys
sys.path.insert(0, 'c:\\Users\\priha\\Documents\\cortex')

from core.preprocessing.xml_parser import parse_xml_content_to_df
from core.preprocessing.corpus_loader import load_monolingual_corpus_files
import pandas as pd
import duckdb
import io
import tempfile
import os

# Create test XML
xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<corpus name="TestInlineTags" lang="EN">
    <text id="001">
        <s n="1"><PN type="human" sex="male">John</PN> works at <PN type="place">New Era Pizzeria</PN>.</s>
        <s n="2">He likes <FOOD category="italian">pizza</FOOD> and <FOOD category="beverage">coffee</FOOD>.</s>
    </text>
</corpus>"""

# Save to file
test_file = 'c:\\Users\\priha\\Documents\\cortex\\tests\\test_inline_tags.xml'
with open(test_file, 'w', encoding='utf-8') as f:
    f.write(xml_content)

# Load corpus
file_obj = io.BytesIO(xml_content.encode('utf-8'))
file_obj.name = 'test_inline_tags.xml'

result = load_monolingual_corpus_files([file_obj], 'EN', 'XML (Tagged)')

if result.get('error'):
    print(f"ERROR loading corpus: {result['error']}")
    sys.exit(1)

db_path = result['db_path']
print(f"Corpus loaded: {db_path}")

# Test queries
con = duckdb.connect(db_path, read_only=True)

print("\n=== Test 1: Query <PN> (all person/place names) ===")
query1 = "SELECT token FROM corpus WHERE in_PN = TRUE"
df1 = con.execute(query1).fetch_df()
print(df1['token'].tolist())
expected1 = ['John', 'New', 'Era', 'Pizzeria']
assert df1['token'].tolist() == expected1, f"Expected {expected1}, got {df1['token'].tolist()}"
print("PASS")

print("\n=== Test 2: Query <PN type=\"human\"> ===")
query2 = "SELECT token FROM corpus WHERE in_PN = TRUE AND PN_type = 'human'"
df2 = con.execute(query2).fetch_df()
print(df2['token'].tolist())
expected2 = ['John']
assert df2['token'].tolist() == expected2, f"Expected {expected2}, got {df2['token'].tolist()}"
print("PASS")

print("\n=== Test 3: Query <FOOD category=\"italian\"> ===")
query3 = "SELECT token FROM corpus WHERE in_FOOD = TRUE AND FOOD_category = 'italian'"
df3 = con.execute(query3).fetch_df()
print(df3['token'].tolist())
expected3 = ['pizza']
assert df3['token'].tolist() == expected3, f"Expected {expected3}, got {df3['token'].tolist()}"
print("PASS")

print("\n=== Test 4: Wildcard <PN type=\"pl*\"> ===")
query4 = "SELECT token FROM corpus WHERE in_PN = TRUE AND regexp_matches(PN_type, '^pl.*$')"
df4 = con.execute(query4).fetch_df()
print(df4['token'].tolist())
expected4 = ['New', 'Era', 'Pizzeria']
assert df4['token'].tolist() == expected4, f"Expected {expected4}, got {df4['token'].tolist()}"
print("PASS")

con.close()

print("\nAll tests passed!")
print(f"\nTest database: {db_path}")
