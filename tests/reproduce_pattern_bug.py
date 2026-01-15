
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.modules.collocation_patterns import match_pattern_in_concordance, parse_pattern_definitions

def reproduce_kepada_bug():
    # Sentence: "berikan bantuan kepada masyarakat"
    # Pattern: <> # kepada
    # Node: bantuan, Collocate: kepada
    conc_tokens = [
        {'token': 'berikan', 'pos': 'VB', 'lemma': 'beri'},
        {'token': 'bantuan', 'pos': 'NN', 'lemma': 'bantu'}, # Node (idx 1)
        {'token': 'kepada', 'pos': 'IN', 'lemma': 'kepada'}, # Collocate (idx 2)
        {'token': 'masyarakat', 'pos': 'NN', 'lemma': 'masyarakat'} # Suffix (idx 3)
    ]
    
    # Test 1: Exact match with suffix
    pattern_str = "test: <> # masyarakat"
    patterns, _ = parse_pattern_definitions(pattern_str)
    parsed_tokens = patterns[0]['parsed_tokens']
    
    matched, n_idx, c_idx = match_pattern_in_concordance(parsed_tokens, conc_tokens, "bantuan", "kepada")
    print(f"Pattern '{pattern_str}': Matched={matched} (Expected: True)")

    # Test 2: Wrong suffix (should fail)
    pattern_str = "test: <> # kepada"
    patterns, _ = parse_pattern_definitions(pattern_str)
    parsed_tokens = patterns[0]['parsed_tokens']
    # This pattern says '#' is follow by nothing. But in Test 1 it's followed by masyarakat.
    # Wait, the user said: <> # kepada
    # If Node=bantuan, Collocate=kepada, then "kepada" matches #.
    # Pattern: <> # kepada -> bantuan kepada ??? 
    # That would mean ANOTHER "kepada" must follow.
    
    # User's case: node = semua, pattern = <> # kepada
    # If the sentence is "semua orang kepada", it should match.
    # If it's "semua orang ini", it should NOT match.
    
    conc_tokens_2 = [
        {'token': 'semua', 'pos': 'CD', 'lemma': 'semua'},
        {'token': 'orang', 'pos': 'NN', 'lemma': 'orang'},
        {'token': 'ini', 'pos': 'PR', 'lemma': 'ini'}
    ]
    pattern_str_2 = "test: <> # kepada"
    patterns_2, _ = parse_pattern_definitions(pattern_str_2)
    matched, _, _ = match_pattern_in_concordance(patterns_2[0]['parsed_tokens'], conc_tokens_2, "semua", "orang")
    print(f"Pattern '{pattern_str_2}' against 'semua orang ini': Matched={matched} (Expected: False)")

if __name__ == "__main__":
    reproduce_kepada_bug()
