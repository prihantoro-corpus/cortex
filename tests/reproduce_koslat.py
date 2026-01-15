import xml.etree.ElementTree as ET
import pandas as pd
import sys
import os

# Adjust path to import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from core.preprocessing.xml_parser import parse_xml_content_to_df

# Mock KOSLAT XML structure based on user description
# Assuming standard <s> tags inside <text> since user implies sentences exist.
xml_content = """
<corpus>
    <root>
        <text id="RSHS0001" ownership="gov" sentiment="neg" sex="f" source="GoogleReview" time="Apr24" type="hos" who="p">
            This is a sentence. Hospital service was bad.
        </text>
        <text id="RSHS0002" ownership="priv" sentiment="pos" sex="m" source="GoogleReview" time="Apr24" type="hos" who="k">
            Great service here.
        </text>
    </root>
</corpus>
"""

def test_koslat_parsing():
    print("Parsing KOSLAT Mock XML...")
    result = parse_xml_content_to_df(xml_content)
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    df = pd.DataFrame(result['df_data'])
    
    print(f"\nExtracted DataFrame Columns: {df.columns.tolist()}")
    print(f"DataFrame Shape: {df.shape}")
    
    if df.empty:
        print("!! DataFrame is empty !!")
        return

    # Check for expected attributes
    expected_attrs = ['ownership', 'sentiment', 'sex', 'source', 'type', 'who']
    found_attrs = [col for col in expected_attrs if col in df.columns]
    missing_attrs = [col for col in expected_attrs if col not in df.columns]

    print("\nAttribute Check:")
    for attr in expected_attrs:
        if attr in df.columns:
            print(f"  [OK] Found '{attr}'. Unique values: {df[attr].unique().tolist()}")
        else:
            print(f"  [FAIL] Missing '{attr}'")
            
    if missing_attrs:
        print(f"\nFAILED: Missing attributes: {missing_attrs}")
    else:
        print("\nSUCCESS: All KOSLAT attributes found.")

if __name__ == "__main__":
    test_koslat_parsing()
