import sys
import os
import io
import pandas as pd
import duckdb
import re
import uuid
import tempfile

sys.path.insert(0, 'c:\\Users\\priha\\Documents\\cortex')

from core.preprocessing.cleaning import sanitize_xml_content
from core.preprocessing.xml_parser import parse_xml_content_to_df, extract_xml_structure

# 1. Read the demo XML
print("Reading demo XML...")
demo_path = 'c:\\Users\\priha\\Documents\\cortex\\corpora\\english\\xml_tag_demo.xml'
with open(demo_path, 'r', encoding='utf-8') as f:
    xml_content = f.read()

# 2. Sanitize
print("Sanitizing...")
cleaned_xml = sanitize_xml_content(xml_content)
# print(f"Cleaned XML Sample:\n{cleaned_xml[:200]}...")

# 3. Parse Structure
print("Parsing structure...")
file_structure, str_err = extract_xml_structure(cleaned_xml)
if str_err:
    print(f"Structure Error: {str_err}")
else:
    print(f"Structure Tags: {list(file_structure.keys())}")

# 4. Parse to DF
print("Parsing to DF (with preserve_inline_tags=True)...")
result = parse_xml_content_to_df(
    cleaned_xml, 
    stanza_processor=None, # Testing WITHOUT stanza first to check fallback
    lang_code='EN',
    preserve_inline_tags=True
)

if 'error' in result:
    print(f"Parse Error: {result['error']}")
else:
    df_data = result['df_data']
    print(f"Tokens extracted: {len(df_data)}")
    
    if len(df_data) > 0:
        df = pd.DataFrame(df_data)
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample data:\n{df.head(5)}")
    else:
        print("!! WARNING: ZERO TOKENS EXTRACTED")
        
        # INVESTIGATE WHY
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(cleaned_xml)
            print(f"Root tag: {root.tag}")
            pass1_tags = {'sent', 's', 'u', 'utterance'}
            
            elements = []
            def find_s(elem):
                if elem.tag in pass1_tags:
                    elements.append(elem)
                for child in elem:
                    find_s(child)
            find_s(root)
            print(f"Found {len(elements)} structural elements (<s> etc)")
            
            if elements:
                test_elem = elements[0]
                print(f"Testing first element: <{test_elem.tag}>")
                print(f"Content: {''.join(test_elem.itertext())}")
                
        except Exception as e:
            print(f"Simplified XML parse failed: {e}")

# 5. Check DuckDB Ingestion
if not 'error' in result and len(result['df_data']) > 0:
    print("\nSimulating DuckDB Ingestion...")
    db_path = os.path.join(tempfile.gettempdir(), f"test_{uuid.uuid4().hex}.duckdb")
    con = duckdb.connect(db_path)
    df_src = pd.DataFrame(result['df_data'])
    df_src["_token_low"] = df_src["token"].str.lower()
    con.execute("CREATE TABLE corpus AS SELECT * FROM df_src")
    total = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
    print(f"DuckDB table created. Total row count: {total}")
    con.close()
    os.remove(db_path)
