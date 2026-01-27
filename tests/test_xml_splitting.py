import sys
import os
import io

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.preprocessing.xml_parser import parse_xml_content_to_df

def mock_stanza_proc(text, lang):
    # Simulate Stanza splitting into 2 sentences
    # Stanza records: [{'token': ..., 'pos': ..., 'lemma': ..., 'sent_id': 1}, ...]
    # We'll return 2 words, one per sentence
    return [
        {'token': 'Hello', 'pos': 'INTJ', 'lemma': 'hello', 'sent_id': 1},
        {'token': 'World', 'pos': 'NOUN', 'lemma': 'world', 'sent_id': 2}
    ], None

def test_xml_splitting():
    xml_content = "<text><s>Hello World. This is a test.</s></text>"
    
    print("Testing parse_xml_content_to_df with stanza split...")
    result = parse_xml_content_to_df(xml_content, stanza_processor=mock_stanza_proc)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
        
    df_data = result['df_data']
    print(f"Number of rows: {len(df_data)}")
    
    # Check sent IDs
    sent_ids = [row['sent_id'] for row in df_data]
    print(f"Sent IDs found: {sent_ids}")
    
    # We expect 2 unique sent IDs because our mock stanza split them
    unique_ids = set(sent_ids)
    print(f"Unique sent IDs: {unique_ids}")
    
    assert len(unique_ids) == 2
    assert 1 in unique_ids
    assert 2 in unique_ids
    
    print("XML Splitting Test PASSED!")

if __name__ == "__main__":
    test_xml_splitting()
