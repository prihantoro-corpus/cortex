
import sys
import os
sys.path.insert(0, os.path.abspath(r"c:\Users\priha\OneDrive - Office's ID\Documents\Corpus Query System\CORTEX_ARCHITECTURE"))

def verify_fixes():
    print("Verifying module updates...")
    try:
        from core.modules.overview import get_top_frequencies_v2
        print("[OK] get_top_frequencies_v2 exists")
        
        from core.modules.ngram import generate_n_grams_v2
        print("[OK] generate_n_grams_v2 exists")

        from core.ai_service import interpret_results_llm
        print("[OK] core.ai_service imported successfully (syntax check passed)")
        
        from ui_streamlit.views.dictionary_view import render_dictionary_view
        print("[OK] dictionary_view imports CEFRAnalyzer correctly (no import error at least)")
        
        from cefrpy import CEFRAnalyzer
        c = CEFRAnalyzer()
        if hasattr(c, 'get_average_word_level_CEFR'):
             print("[OK] CEFRAnalyzer has get_average_word_level_CEFR")
        else:
             print("[ERROR] CEFRAnalyzer missing method")
             
    except ImportError as e:
        print(f"[ERROR] Import Error: {e}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")

if __name__ == "__main__":
    verify_fixes()
