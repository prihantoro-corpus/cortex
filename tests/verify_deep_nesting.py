import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.preprocessing.xml_parser import parse_xml_content_to_df

def test_deep_nesting():
    xml_content = """
    <root root_attr="root_val">
        <level1 l1_attr="val1">
            <ignore_me>
                <level2 l2_attr="val2">
                    <u sent_attr="sent_val">
                        Content here.
                    </u>
                </level2>
            </ignore_me>
        </level1>
    </root>
    """
    
    result = parse_xml_content_to_df(xml_content)
    
    if result.get('error'):
        print(f"Error: {result['error']}")
        return

    # Check the first token's attributes (only one sentence, so check its dict)
    # The 'attributes' key in result aggregates ALL found attributes across the corpus, 
    # but we want to check co-occurrence in a row basically.
    # But since there is only one sentence, 'attributes' summary should contain ALL of them.
    
    attrs = result.get('attributes', {})
    found_keys = list(attrs.keys())
    print(f"Found attributes: {found_keys}")
    
    expected = ['root_attr', 'l1_attr', 'l2_attr', 'sent_attr']
    missing = [k for k in expected if k not in found_keys]
    
    if missing:
        print(f"FAILED. Missing: {missing}")
    else:
        print("SUCCESS. All deep attributes found.")

if __name__ == "__main__":
    test_deep_nesting()
