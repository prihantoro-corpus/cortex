import pandas as pd
from io import BytesIO
import requests

def df_to_excel_bytes(df):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    buf.seek(0)
    return buf.getvalue()

def download_file_to_bytesio(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 
        return BytesIO(response.content)
    except Exception as e:
        print(f"Failed to download built-in corpus from {url}. Error: {e}")
        return None

def dfs_to_zip_excel_bytes(df_dict):
    """
    Creates a ZIP file containing multiple Excel files from a dictionary of DataFrames.
    Args:
        df_dict (dict): { "filename_no_ext": pd.DataFrame }
    Returns:
        bytes: ZIP file content
    """
    import zipfile
    zip_buf = BytesIO()
    
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for name, df in df_dict.items():
            if df is None or df.empty:
                continue
            excel_bytes = df_to_excel_bytes(df)
            zip_file.writestr(f"{name}.xlsx", excel_bytes)
            
    zip_buf.seek(0)
    return zip_buf.getvalue()
