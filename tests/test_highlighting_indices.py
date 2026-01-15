
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.modules.collocation_patterns import match_pattern_in_concordance, parse_pattern_tokens

def test_accurate_indices():
    # Sentence 1 from user results
    # Index 0: Perawat, 5: semua, 9: semua, 10: perawat
    conc_tokens = [
        {'token': 'Perawat', 'pos': 'NN', 'lemma': 'perawat'}, # 0
        {'token': 'nya', 'pos': 'PR', 'lemma': 'nya'},       # 1
        {'token': 'pada', 'pos': 'IN', 'lemma': 'pada'},     # 2
        {'token': 'ramah', 'pos': 'JJ', 'lemma': 'ramah'},   # 3
        {'token': 'ramah', 'pos': 'JJ', 'lemma': 'ramah'},   # 4
        {'token': 'semua', 'pos': 'CD', 'lemma': 'semua'},   # 5
        {'token': '.', 'pos': 'PUNCT', 'lemma': '.'},        # 6
        {'token': 'Terimakasih', 'pos': 'NN', 'lemma': 'terima kasih'}, # 7
        {'token': 'buat', 'pos': 'IN', 'lemma': 'buat'},     # 8
        {'token': 'semua', 'pos': 'CD', 'lemma': 'semua'},   # 9
        {'token': 'perawat', 'pos': 'NN', 'lemma': 'perawat'} # 10
    ]
    
    pattern_str = "<> #"
    parsed_tokens, _ = parse_pattern_tokens("dummy: " + pattern_str) # parse_pattern_tokens expects label: pattern
    # Wait, parse_pattern_tokens is internal? Let me check.
    # Ah, I see parse_pattern_definitions in my previous view.

    from core.modules.collocation_patterns import parse_pattern_definitions
    patterns, errors = parse_pattern_definitions("test: " + pattern_str)
    parsed_tokens = patterns[0]['parsed_tokens']
    
    # Test for collocate 'perawat'
    matched, n_idx, c_idx = match_pattern_in_concordance(parsed_tokens, conc_tokens, "semua", "perawat")
    print(f"Match result for 'semua perawat': {matched}, Node={n_idx}, Coll={c_idx}")
    # Expected: Node=9, Coll=10
    
    # Test for collocate 'ramah'
    matched, n_idx, c_idx = match_pattern_in_concordance(parsed_tokens, conc_tokens, "semua", "ramah")
    print(f"Match result for 'semua ramah': {matched}, Node={n_idx}, Coll={c_idx}")
    # Expected: False

if __name__ == "__main__":
    test_accurate_indices()
