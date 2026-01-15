import streamlit as st
import pandas as pd

def render_kwic_table(kwic_data, is_parallel=False, target_lang="NA"):
    """
    Renders a list of KWIC examples in a styled HTML table, matching the original app.py look.
    
    Args:
        kwic_data (list of dict): Each dict should have keys:
            - 'Collocate' (optional, rendered in bold first col)
            - 'Source Corpus'
            - 'Left Context'
            - 'Node'
            - 'Right Context'
            - 'Translation' (optional)
        is_parallel (bool): Whether to show the Translation column.
        target_lang (str): Language code for translation header.
    """
    if not kwic_data:
        st.info("No examples to display.")
        return

    # CSS Styling (Directly ported from app.py)
    # CSS Styling (Directly ported from app.py)
    collocate_example_table_style = f"""
    <style>
    .collex-table-container-fixed {{
        max-height: 400px; 
        overflow-y: auto;
        margin-bottom: 1rem;
        width: 100%;
        border: 1px solid #444; 
        border-radius: 5px;
    }}
    .collex-table-inner table {{ 
        width: 100%; 
        table-layout: fixed; 
        font-family: monospace; 
        color: #ddd; 
        font-size: 0.9em;
        border-collapse: collapse;
    }}
    .collex-table-inner th {{ 
        font-weight: bold; 
        text-align: center; 
        background-color: #383838; 
        white-space: nowrap; 
        padding: 8px;
        position: sticky;
        top: 0;
        z-index: 1;
    }}
    .collex-table-inner td {{
        padding: 5px 10px;
        border-bottom: 1px solid #333;
        vertical-align: middle;
        word-wrap: break-word; /* Ensure wrapping */
    }}
    
    /* Column Widths (New Layout with Leftmost Metadata) */
    .collex-table-inner td:nth-child(1) {{ width: 12%; text-align: left; font-size: 0.8em; white-space: normal; border-right: 1px solid #444; }} /* Metadata */
    .collex-table-inner td:nth-child(2) {{ width: 6%; text-align: left; font-weight: bold; border-right: 1px solid #444; white-space: nowrap; }} /* Collocate/Num */
    .collex-table-inner td:nth-child(3) {{ width: 10%; text-align: center; font-size: 0.8em; white-space: normal; color: #aaa; }} /* Source */
    .collex-table-inner td:nth-child(4) {{ width: 24%; text-align: right; white-space: normal; }} /* Left */
    .collex-table-inner td:nth-child(5) {{ 
        width: 12%; 
        text-align: center; 
        font-weight: bold; 
        background-color: #eee; 
        color: black; 
        white-space: normal;
        border-radius: 4px;
    }} /* Node */
    .collex-table-inner td:nth-child(6) {{ width: 24%; text-align: left; white-space: normal; }} /* Right */
    
    /* Translation Column (if present - child 7) */
    .collex-table-inner td:nth-child(7) {{ 
        text-align: left; 
        color: #CCFFCC; 
        width: 12%; 
        font-family: sans-serif; 
        font-size: 0.85em; 
        white-space: normal; 
        border-left: 1px solid #444;
    }}
    </style>
    """
    st.markdown(collocate_example_table_style, unsafe_allow_html=True)
    
    df = pd.DataFrame(kwic_data)
    
    # Ensure columns exist
    required_cols = ['Collocate', 'Source Corpus', 'Left Context', 'Node', 'Right Context']
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
    
    # Process attributes/metadata content for the leftmost column
    def process_meta(row):
         if 'Metadata' not in row:
             return ""
         meta = row['Metadata']
         if not isinstance(meta, dict): meta = {}
         html = ""
         for k, v in meta.items():
              # Stacked badges
              html += f"<div style='margin-bottom:2px;'><span style='background-color: #334155; color: #e2e8f0; font-size: 0.85em; padding: 2px 4px; border-radius: 3px; border: 1px solid #475569; display:inline-block;' title='{k}'>{v}</span></div>"
         return html

    df['Metadata'] = df.apply(process_meta, axis=1)

    if is_parallel:
        if 'Translation' not in df.columns:
             df['Translation'] = ""
    else:
        if 'Translation' in df.columns:
            df = df.drop(columns=['Translation'])

    # Reorder columns: Metadata first
    cols = ['Metadata', 'Collocate', 'Source Corpus', 'Left Context', 'Node', 'Right Context']
    if is_parallel: cols.append('Translation')
    df = df[cols]

    # Build HTML Table
    header = "<tr><th>Meta</th><th>#</th><th>Source</th><th>Left Context</th><th>Node</th><th>Right Context</th>"
    if is_parallel:
        header += f"<th>Translation ({target_lang})</th>"
    header += "</tr>"

    html_table = df.to_html(header=False, escape=False, classes=['collex-table-inner'], index=False)
    
    if "<tbody>" in html_table:
        html_table = html_table.replace("<tbody>", f"<thead>{header}</thead><tbody>")
    else:
        # Fallback if tbody missing (rare with valid data)
        html_table = f"<table class='collex-table-inner'><thead>{header}</thead>" + html_table
    
    st.markdown(f"<div class='collex-table-container-fixed'>{html_table}</div>", unsafe_allow_html=True)
