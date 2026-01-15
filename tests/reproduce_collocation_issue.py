import sys
import os

# Set paths
project_root = r"C:\Users\priha\OneDrive - Office's ID\Documents\Corpus Query System\CORTEX_ARCHITECTURE"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.modules.collocation_patterns import parse_pattern_tokens, match_pattern_in_concordance

def test_user_case():
    pattern_str = "<> + #"
    tokens, errors = parse_pattern_tokens(pattern_str)
    print(f"Pattern Tokens: {tokens}")
    
    # Example 1: untuk RSHS , sehat2 untuk semua dokter , perawat dan staff
    # Indices: 0:untuk 1:RSHS 2:, 3:sehat2 4:untuk 5:semua 6:dokter 7:, 8:perawat 9:dan 10:staff
    line1 = [
        {'token': 'untuk'}, # 0
        {'token': 'RSHS'},   # 1
        {'token': ','},      # 2
        {'token': 'sehat2'}, # 3
        {'token': 'untuk'}, # 4
        {'token': 'semua'}, # 5 (NODE)
        {'token': 'dokter'}, # 6 (+1)
        {'token': ','},      # 7 (+2)
        {'token': 'perawat'},# 8 (+3)
        {'token': 'dan'},    # 9 (+4, USER SAYS WRONG MATCH)
        {'token': 'staff'},  # 10
    ]
    
    node = "semua"
    
    print("\n--- Testing 'dan' in Line 1 ---")
    res_dan = match_pattern_in_concordance(tokens, line1, node, "dan")
    print(f"Match result for 'dan' at distance 4: {res_dan}")
    
    print("\n--- Testing 'perawat' in Line 1 ---")
    res_per = match_pattern_in_concordance(tokens, line1, node, "perawat")
    print(f"Match result for 'perawat' at distance 3: {res_per}")

    print("\n--- Testing ',' in Line 1 (Expected match at distance 2) ---")
    res_punct = match_pattern_in_concordance(tokens, line1, node, ",")
    print(f"Match result for ',' at distance 2: {res_punct}")

    # Example 2: baik lagi , perbaiki dari semua segi , dr , perawat
    # 0:baik 1:lagi 2:, 3:perbaiki 4:dari 5:semua 6:segi 7:, 8:dr 9:, 10:perawat
    line2 = [
        {'token': 'baik'},     # 0 (USER SAYS WRONG MATCH)
        {'token': 'lagi'},     # 1
        {'token': ','},        # 2
        {'token': 'perbaiki'}, # 3
        {'token': 'dari'},     # 4
        {'token': 'semua'},    # 5 (NODE)
        {'token': 'segi'},     # 6
        {'token': ','},        # 7
        {'token': 'dr'},       # 8
        {'token': ','},        # 9
        {'token': 'perawat'},  # 10
    ]
    print("\n--- Testing 'baik' in Line 2 ---")
    res_baik = match_pattern_in_concordance(tokens, line2, node, "baik")
    print(f"Match result for 'baik' (distance -5): {res_baik}")

if __name__ == "__main__":
    test_user_case()
