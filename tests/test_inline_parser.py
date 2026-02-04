import sys
sys.path.insert(0, 'c:\\Users\\priha\\Documents\\cortex')

from core.preprocessing.xml_parser import parse_xml_content_to_df
import pandas as pd

# Read test XML
with open('c:\\Users\\priha\\Documents\\cortex\\tests\\test_inline_tags.xml', 'r', encoding='utf-8') as f:
    xml_content = f.read()

# Parse with inline tags enabled
result = parse_xml_content_to_df(xml_content, preserve_inline_tags=True)

if 'error' in result:
    print(f"ERROR: {result['error']}")
else:
    df = pd.DataFrame(result['df_data'])
    print("=== Parsed DataFrame ===")
    print(df.to_string())
    print("\n=== Columns ===")
    print(df.columns.tolist())
    print("\n=== Sample: Tokens inside <PN> tags ===")
    if 'in_PN' in df.columns:
        pn_tokens = df[df['in_PN'] == True]
        print(pn_tokens[['token', 'PN_type', 'PN_sex', 'PN_domain']].to_string())
    else:
        print("WARNING: 'in_PN' column not found!")
