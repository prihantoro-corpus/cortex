import sys
sys.path.insert(0, 'c:\\Users\\priha\\Documents\\cortex')

from core.preprocessing.xml_parser import parse_xml_content_to_df
import pandas as pd

# Read demo XML
with open('c:\\Users\\priha\\Documents\\cortex\\corpora\\english\\xml_tag_demo.xml', 'r', encoding='utf-8') as f:
    xml_content = f.read()

print("=== Parsing with preserve_inline_tags=True ===")
result = parse_xml_content_to_df(xml_content, preserve_inline_tags=True)

if 'error' in result:
    print(f"ERROR: {result['error']}")
elif 'df_data' in result:
    df = pd.DataFrame(result['df_data'])
    print(f"Total tokens: {len(df)}")
    print(f"\nFirst 10 rows:")
    print(df.head(10).to_string())
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Check for XML tag columns
    xml_cols = [c for c in df.columns if c.startswith('in_') or '_' in c and not c.startswith('_')]
    print(f"\nXML tag columns: {xml_cols}")
else:
    print("Unexpected result:", result.keys())
