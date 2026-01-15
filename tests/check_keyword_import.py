import sys
import os

sys.path.append(os.getcwd())

try:
    from core.modules.keyword import generate_grouped_keyword_list
    print("Import Successful!")
except ImportError as e:
    print(f"Import Failed: {e}")
except Exception as e:
    print(f"Other Error: {e}")
