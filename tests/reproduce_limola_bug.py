import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.preprocessing.xml_parser import parse_xml_content_to_df

def test_limola_parsing():
    xml_path = r"C:\Users\priha\.gemini\antigravity\brain\69eecb35-40f3-4a64-8b57-c0263cd2f717\test_limola_snippet.xml"
    
    with open(xml_path, 'r', encoding='utf-8') as f:
        # We pass the file object directly as the parser supports it (or read string)
        # Using string for simplicity to match parser expectation if needed, but parser handles file-like object too if it has .read()
        # parse_xml_content_to_df checks isinstance(xml_input, str). 
        # But let's pass the string content to be safe and simple.
        content = f.read()
        
    result = parse_xml_content_to_df(content)
    
    if result.get('error'):
        print(f"Error parsing: {result['error']}")
        return

    attributes = result.get('attributes', {})
    print(f"Detected Attributes: {list(attributes.keys())}")
    
    expected_attrs = ['language', 'location', 'year', 'speaker']
    missing = [attr for attr in expected_attrs if attr not in attributes]
    
    if missing:
        print(f"FAILED: Missing attributes: {missing}")
    else:
        print("SUCCESS: All attributes found!")

if __name__ == "__main__":
    test_limola_parsing()
