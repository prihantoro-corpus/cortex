import sys
import os
import io

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.preprocessing.xml_parser import parse_xml_content_to_df

def mock_stanza_proc(text, lang):
    return [
        {'token': 'Hello', 'pos': 'INTJ', 'lemma': 'hello', 'sent_id': 1},
        {'token': 'World', 'pos': 'NOUN', 'lemma': 'world', 'sent_id': 2}
    ], None

def test_xml_splitting_v2():
    xml_content = "<text><s>Hello World. This is a test.</s></text>"
    
    print("Testing parse_xml_content_to_df with stanza split and sent_map check...")
    result = parse_xml_content_to_df(xml_content, stanza_processor=mock_stanza_proc)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
        
    df_data = result['df_data']
    sent_map = result['sent_map']
    
    # Check sent IDs in token data
    token_sent_ids = sorted(list(set(row['sent_id'] for row in df_data)))
    print(f"Token Sent IDs: {token_sent_ids}")
    
    # Check sent IDs in sent_map
    map_sent_ids = sorted(list(sent_map.keys()))
    print(f"Map Sent IDs: {map_sent_ids}")
    
    assert len(token_sent_ids) == 2
    assert token_sent_ids == map_sent_ids
    
    # Check content of sent_map
    print(f"Sent 1: {sent_map[token_sent_ids[0]]}")
    print(f"Sent 2: {sent_map[token_sent_ids[1]]}")
    
    assert "Hello" in sent_map[token_sent_ids[0]]
    assert "World" in sent_map[token_sent_ids[1]]
    
    print("XML Splitting + SentMap Test PASSED!")

if __name__ == "__main__":
    test_xml_splitting_v2()
