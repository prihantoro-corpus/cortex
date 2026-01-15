
import sys
import os
sys.path.insert(0, os.path.abspath(r"c:\Users\priha\OneDrive - Office's ID\Documents\Corpus Query System\CORTEX_ARCHITECTURE"))

def verify_modules():
    print("Verifying module imports...")
    try:
        from ui_streamlit.views.overview_view import render_overview
        print("[OK] overview_view imported successfully")
        
        from ui_streamlit.views.dictionary_view import render_dictionary_view
        print("[OK] dictionary_view imported successfully")
        
        from ui_streamlit.views.collocation_view import render_collocation_view
        print("[OK] collocation_view imported successfully")
        
        from ui_streamlit.components.result_display import render_kwic_table
        print("[OK] result_display component imported successfully")
        
        print("\nAll UI modules verified.")
        
    except ImportError as e:
        print(f"[ERROR] ImportError: {e}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")

if __name__ == "__main__":
    verify_modules()
