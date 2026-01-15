import sys
import os

# Set paths
project_root = r"C:\Users\priha\OneDrive - Office's ID\Documents\Corpus Query System\CORTEX_ARCHITECTURE"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.modules.collocation_patterns import parse_pattern_tokens, match_pattern_in_concordance

def test_adjacency():
    # User's case: # <> should NOT match "dibongkar ini semua"
    pattern_str = "# <>"
    tokens, errors = parse_pattern_tokens(pattern_str)
    print(f"Pattern Tokens: {tokens}")
    
    # Concordance line: yg(0) udah(1) rapi(2) dibongkar(3) ini(4) semua(5)
    conc_line = [
        {'token': 'yg'}, # 0
        {'token': 'udah'}, # 1
        {'token': 'rapi'}, # 2
        {'token': 'dibongkar'}, # 3
        {'token': 'ini'}, # 4
        {'token': 'semua'}, # 5
    ]
    
    node = "semua"
    collocate = "dibongkar"
    
    result = match_pattern_in_concordance(tokens, conc_line, node, collocate)
    print(f"Match result for '# <>' on 'dibongkar ini semua': {result}")
    assert result == False, "FAILED: '# <>' should not match when a word is between them."

    # Test immediate adjacency
    # dibongkar(3) semua(4)
    conc_line_adj = [
        {'token': 'dibongkar'}, # 0
        {'token': 'semua'}, # 1
    ]
    result_adj = match_pattern_in_concordance(tokens, conc_line_adj, node, collocate)
    print(f"Match result for '# <>' on 'dibongkar semua': {result_adj}")
    assert result_adj == True, "FAILED: '# <>' should match when adjacent."

    # Test pattern with forced word: <> + #
    # yg(0) udah(1) rapi(2) dibongkar(3) ini(4) semua(5)
    # node: rapi(2), collocate: ini(4)
    # pattern: <> + # -> True (rapi + dibongkar + ini) -> Wait, rapi(2), dibongkar(3), ini(4). Yes.
    pattern_plus = "<> + #"
    tokens_plus, _ = parse_pattern_tokens(pattern_plus)
    node_rapi = "rapi"
    coll_ini = "ini"
    result_plus = match_pattern_in_concordance(tokens_plus, conc_line, node_rapi, coll_ini)
    print(f"Match result for '<> + #' on 'rapi dibongkar ini': {result_plus}")
    assert result_plus == True, "FAILED: '<> + #' should match with exactly one word between."

    # Test <> + # with WRONG distance (adjacent)
    # node: rapi(2), collocate: dibongkar(3)
    coll_dibongkar = "dibongkar"
    result_plus_wrong = match_pattern_in_concordance(tokens_plus, conc_line, node_rapi, coll_dibongkar)
    print(f"Match result for '<> + #' on 'rapi dibongkar' (adjacent): {result_plus_wrong}")
    assert result_plus_wrong == False, "FAILED: '<> + #' should NOT match when adjacent."


if __name__ == "__main__":
    try:
        test_adjacency()
        print("\nALL ADJACENCY TESTS PASSED!")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
