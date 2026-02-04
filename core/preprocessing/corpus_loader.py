import pandas as pd
import duckdb
import os
import uuid
import tempfile
import re
import requests
import io
from io import StringIO
from .cleaning import sanitize_xml_content
from .xml_parser import extract_xml_structure, parse_xml_content_to_df
from core.config import CORPORA_DIR, TAGSET_DIR
from core.modules.overview import save_pos_definitions
from .tagging import tag_text_with_stanza, tag_text_simple_fallback

def load_monolingual_corpus_files(file_sources, explicit_lang_code, selected_format, progress_callback=None):
    """
    Loads one or more monolingual files into a DuckDB database.
    Returns: dict { 'db_path': str, 'stats': dict, 'structure': dict, 'lang_code': str, 'error': str }
    """
    if not file_sources:
        return {'error': "No files provided"}

    all_df_data = []
    
    # Defaults
    source_lang_code = explicit_lang_code
    is_tagged_format = 'Tagged' in selected_format
    use_stanza = 'Raw' in selected_format
    xml_detected_lang_code = None
    combined_structure = {}
    
    print(f"DEBUG: load_monolingual_corpus_files called. Lang: {explicit_lang_code}, Format: {selected_format}")

    num_files = len(file_sources)

    for idx, file_source in enumerate(file_sources):
        if progress_callback:
            progress_callback(idx / num_files, f"Processing {file_source.name}...")
            
        file_source.seek(0)
        filename = file_source.name
        
        # Read a sample to detect pseudo-XML
        sample_bytes = file_source.read(1024)
        file_source.seek(0)
        sample_str = sample_bytes.decode('utf-8', errors='ignore').strip()
        
        is_xml_ext = filename.lower().endswith('.xml')
        is_pseudo_xml = False
        if not is_xml_ext:
            if sample_str.startswith('<'):
                is_pseudo_xml = True
            elif any(tag in sample_str.lower() for tag in ['<text', '<corpus', '<p>', '<p ']):
                is_pseudo_xml = True

        # --- XML PROCESSING ---
        if is_xml_ext or is_pseudo_xml:
            try:
                xml_content = file_source.read().decode('utf-8', errors='ignore')
                cleaned_xml = sanitize_xml_content(xml_content)
                
                # 1. Structure Extraction
                file_structure, str_err = extract_xml_structure(cleaned_xml)
                if file_structure:
                    for tag, attributes in file_structure.items():
                        if tag not in combined_structure:
                            combined_structure[tag] = attributes
                        else:
                            for attr, vals in attributes.items():
                                if attr not in combined_structure[tag]:
                                    combined_structure[tag][attr] = vals
                                else:
                                    if len(combined_structure[tag][attr]) < 20:
                                        combined_structure[tag][attr].update(vals)
                
                # 2. Content Parsing
                stanza_proc = None
                if use_stanza or 'stanza' in selected_format.lower():
                    from .tagging import tag_text_with_stanza
                    stanza_proc = tag_text_with_stanza
                
                result = parse_xml_content_to_df(
                    cleaned_xml, 
                    stanza_processor=stanza_proc, 
                    lang_code=source_lang_code,
                    preserve_inline_tags=True
                )
                if 'df_data' in result:
                    if explicit_lang_code == 'OTHER' and result.get('lang_code') not in ('XML', 'OTHER'):
                        xml_detected_lang_code = result['lang_code'] 
                    
                    for record in result['df_data']:
                        record['filename'] = filename
                    
                    all_df_data.extend(result['df_data'])
                elif 'error' in result:
                    return {'error': f"XML Error ({filename}): {result['error']}"}

            except Exception as e:
                return {'error': f"Processing Error ({filename}): {str(e)}"}
        
        # --- TXT/CSV PROCESSING ---
        else: 
            try:
                file_bytes = file_source.read()
                file_content_str = file_bytes.decode('utf-8', errors='ignore')
                clean_lines = [line for line in file_content_str.splitlines() if line and not line.strip().startswith('#')]
                clean_content = "\n".join(clean_lines)
            except Exception as e:
                return {'error': f"Error reading raw file {filename}: {str(e)}"}

            current_is_tagged = is_tagged_format
            if current_is_tagged:
                file_buffer_for_pandas = StringIO(clean_content)
                df_attempt = None
                for sep_char in ['\t', r'\s+']: 
                    try:
                        file_buffer_for_pandas.seek(0)
                        df_attempt = pd.read_csv(file_buffer_for_pandas, sep=sep_char, header=None, engine="python", dtype=str, skipinitialspace=True, usecols=[0, 1, 2], names=['token', 'pos', 'lemma'])
                        if df_attempt is not None and df_attempt.shape[1] >= 3: break 
                        df_attempt = None 
                    except Exception: df_attempt = None 
                
                if df_attempt is not None and df_attempt.shape[1] >= 3:
                    df_file = df_attempt.copy()
                    df_file["token"] = df_file["token"].fillna("").astype(str).str.strip() 
                    df_file["pos"] = df_file["pos"].fillna("###").astype(str)
                    df_file["lemma"] = df_file["lemma"].fillna("###").astype(str)
                    df_file['sent_id'] = 0 
                    df_file['filename'] = filename
                    all_df_data.extend(df_file.to_dict('records'))
                else:
                    print(f"File {filename} could not be parsed as vertical format. Falling back to raw text.")
                    current_is_tagged = False 
            
            if not current_is_tagged or 'Raw' in selected_format: 
                raw_text = clean_content
                
                # Tagging Logic Integration
                # If explicit_lang_code is set, we try to use it.
                # If "OTHER" is selected, we perform fallback tagging.
                
                print(f"DEBUG: Processing raw text. explicit_lang_code='{explicit_lang_code}'")
                tagged_data = []
                
                if explicit_lang_code and explicit_lang_code != "OTHER" and (use_stanza or 'Raw' in selected_format):
                    try:
                        # Attempt Stanza Tagging
                        tagged_data, err = tag_text_with_stanza(raw_text, explicit_lang_code)
                    except Exception as e:
                        msg = f"Stanza tagging failed for '{explicit_lang_code}': {e}. Using simple fallback."
                        print(msg)
                        stanza_warning = msg
                        tagged_data, err = tag_text_simple_fallback(raw_text)
                elif explicit_lang_code == "OTHER" or (not use_stanza and 'Raw' not in selected_format):
                    tagged_data, err = tag_text_simple_fallback(raw_text)
                else:
                    # Legacy Fallback or Default Behavior for "auto" without specific language?
                    # The request says: "if they choose language, you will tokenise... Stanza"
                    # "When other is chosen... use 'TAG' instead"
                    # If user didn't specify language (e.g. from built-in?), corpus loader might default to 'en' or None.
                    # Current load_monolingual default is 'en' if not provided? -> No, argument is required.
                    # sidebar passes 'en' by default.
                    
                    # If we are here, likely 'en' or some coded lang.
                    # Maintain old behavior? Or upgrade everything? 
                    # "For no, only ENglish, Indonesia and Japan is available." -> "change to all languages supported by Stanza."
                    
                    if explicit_lang_code:
                         tagged_data = tag_text_with_stanza(raw_text, explicit_lang_code)
                    else:
                         # Very generic fallback if no lang code known
                         tagged_data = tag_text_simple_fallback(raw_text)

                # Add metadata
                for item in tagged_data:
                    item['filename'] = filename
                
                all_df_data.extend(tagged_data)

    if not all_df_data:
        return {'error': "No valid data extracted from files"}

    # --- DUCKDB DATA INGESTION ---
    unique_filename = f"corpus_{uuid.uuid4().hex}.duckdb"
    db_path = os.path.join(tempfile.gettempdir(), unique_filename)
    
    if os.path.exists(db_path):
        try: os.remove(db_path)
        except: pass

    try:
        con = duckdb.connect(db_path)
        df_src = pd.DataFrame(all_df_data)
        
        for col in ['token', 'pos', 'lemma', 'sent_id', 'filename']:
            if col not in df_src.columns: df_src[col] = "##" if col in ['pos', 'lemma'] else 0
            
        df_src["_token_low"] = df_src["token"].str.lower()
        con.execute("CREATE TABLE corpus AS SELECT * FROM df_src")
        con.execute("ALTER TABLE corpus ADD COLUMN id INTEGER")
        con.execute("CREATE SEQUENCE seq_id START 1")
        con.execute("UPDATE corpus SET id = nextval('seq_id')")
        con.execute("CREATE INDEX idx_token_low ON corpus(_token_low)")
        con.execute("CREATE INDEX idx_id ON corpus(id)")
        con.execute("CREATE INDEX idx_lemma ON corpus(lemma)")
        con.execute("CREATE INDEX idx_sent ON corpus(sent_id)")
        
        total_tokens = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
        token_freqs = con.execute("SELECT _token_low, count(*) FROM corpus GROUP BY _token_low").fetchall()
        token_counts = {row[0]: row[1] for row in token_freqs}
        corpus_stats = {'token_counts': token_counts, 'total_tokens': total_tokens}
        
        con.close()
        
    except Exception as e:
        return {'error': f"DuckDB Ingestion Failed: {e}"}

    final_lang_code = xml_detected_lang_code if xml_detected_lang_code else source_lang_code
    
    # Save Language to Metadata
    from core.modules.overview import set_corpus_language
    set_corpus_language(db_path, final_lang_code)
    
    # Auto-load local tagset definitions if available
    # Iterate through input files to find a matching tagset (taking the first match)
    for fs in file_sources:
        fname = fs.name
        _load_local_tagset(db_path, fname)
        # break # Maybe load all? Just one is probably safer to avoid mixing definitions blindly
    
    return {
        'db_path': db_path,
        'stats': corpus_stats,
        'structure': combined_structure,
        'lang_code': final_lang_code,
        'error': None,
        'warning': stanza_warning if 'stanza_warning' in locals() else None
    }

def load_xml_parallel_corpus(src_file, tgt_file, src_lang_code, tgt_lang_code, progress_callback=None):
    if src_file is None or tgt_file is None: return {'error': "Files missing"}

    try:
        if progress_callback: progress_callback(0.1, "Parsing source...")
        src_file.seek(0)
        src_content = src_file.read().decode('utf-8', errors='ignore')
        src_cleaned = sanitize_xml_content(src_content)
        src_result = parse_xml_content_to_df(src_cleaned, preserve_inline_tags=True)

        if progress_callback: progress_callback(0.5, "Parsing target...")
        tgt_file.seek(0)
        tgt_content = tgt_file.read().decode('utf-8', errors='ignore')
        tgt_cleaned = sanitize_xml_content(tgt_content)
        tgt_result = parse_xml_content_to_df(tgt_cleaned, preserve_inline_tags=True)
        
    except Exception as e:
        return {'error': f"Parsing failed: {e}"}
    
    if src_result.get('error'): return src_result
    if tgt_result.get('error'): return tgt_result
        
    df_src = pd.DataFrame(src_result['df_data'])
    df_tgt = pd.DataFrame(tgt_result['df_data'])

    src_sent_ids = set(df_src['sent_id'].unique())
    tgt_sent_ids = set(df_tgt['sent_id'].unique())
    
    if src_sent_ids != tgt_sent_ids:
        missing_in_tgt = src_sent_ids - tgt_sent_ids
        missing_in_src = tgt_sent_ids - src_sent_ids
        error_msg = f"Alignment Check Failed. ID mismatch."
        if missing_in_tgt: error_msg += f" Src has extras: {list(missing_in_tgt)[:3]}..."
        if missing_in_src: error_msg += f" Tgt has extras: {list(missing_in_src)[:3]}..."
        return {'error': error_msg}

    df_src["_token_low"] = df_src["token"].str.lower()
    
    # Structure
    src_structure, _ = extract_xml_structure(src_cleaned)
    tgt_structure, _ = extract_xml_structure(tgt_cleaned)
    combined_structure = {}
    if src_structure: combined_structure.update(src_structure)
    if tgt_structure:
        for tag, attrs in tgt_structure.items():
            if tag not in combined_structure: combined_structure[tag] = attrs
            else:
                for attr, values in attrs.items():
                    if attr not in combined_structure[tag]: combined_structure[tag][attr] = values
                    else: combined_structure[tag][attr] = set(list(combined_structure[tag][attr]) + list(values))[:20]

    # DuckDB
    unique_filename = f"corpus_{uuid.uuid4().hex}.duckdb"
    db_path = os.path.join(tempfile.gettempdir(), unique_filename)
    if os.path.exists(db_path):
        try: os.remove(db_path)
        except: pass

    try:
        con = duckdb.connect(db_path)
        if 'filename' not in df_src.columns: df_src['filename'] = src_file.name
        
        con.execute("CREATE TABLE corpus AS SELECT * FROM df_src")
        con.execute("ALTER TABLE corpus ADD COLUMN id INTEGER")
        con.execute("CREATE SEQUENCE seq_id START 1")
        con.execute("UPDATE corpus SET id = nextval('seq_id')")
        con.execute("CREATE INDEX idx_token_low ON corpus(_token_low)")
        con.execute("CREATE INDEX idx_id ON corpus(id)")
        con.execute("CREATE INDEX idx_lemma ON corpus(lemma)")
        con.execute("CREATE INDEX idx_sent ON corpus(sent_id)")
        
        total_tokens = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
        token_freqs = con.execute("SELECT _token_low, count(*) FROM corpus GROUP BY _token_low").fetchall()
        token_counts = {row[0]: row[1] for row in token_freqs}
        corpus_stats = {'token_counts': token_counts, 'total_tokens': total_tokens}
        
        con.close()
    except Exception as e:
        return {'error': f"DuckDB Ingestion Failed: {e}"}

    return {
        'db_path': db_path,
        'stats': corpus_stats,
        'structure': combined_structure,
        'target_df': df_tgt,
        'target_map': tgt_result['sent_map'],
        'error': None
    }

def load_excel_parallel_corpus_file(file_source, excel_format):
    if file_source is None: return {'error': "No file"}
    
    try:
        file_source.seek(0)
        df_raw = pd.read_excel(file_source, engine='openpyxl')
    except Exception as e:
        return {'error': f"Failed to read Excel: {e}"}

    if df_raw.shape[1] < 2:
        return {'error': "Excel must have 2+ columns"}
    
    src_lang = df_raw.columns[0]
    tgt_lang = df_raw.columns[1]
    
    data_src = []
    target_sent_map = {}
    sent_id_counter = 0
    
    for index, row in df_raw.iterrows():
        sent_id_counter += 1
        src_text = str(row.iloc[0]).strip()
        tgt_text = str(row.iloc[1]).strip()
        
        cleaned_text = re.sub(r'([^\w\s])', r' \1 ', src_text)
        src_tokens = [t.strip() for t in cleaned_text.split() if t.strip()]
        
        target_sent_map[sent_id_counter] = tgt_text 
        
        for token in src_tokens:
            data_src.append({
                "token": token,
                "pos": "##",
                "lemma": "##",
                "sent_id": sent_id_counter
            })
            
    if not data_src:
        return {'error': "No valid data"}

    df_src = pd.DataFrame(data_src)
    df_src["_token_low"] = df_src["token"].str.lower()

    unique_filename = f"corpus_{uuid.uuid4().hex}.duckdb"
    db_path = os.path.join(tempfile.gettempdir(), unique_filename)
    if os.path.exists(db_path):
        try: os.remove(db_path)
        except: pass

    try:
        con = duckdb.connect(db_path)
        if 'filename' not in df_src.columns: df_src['filename'] = file_source.name
        
        con.execute("CREATE TABLE corpus AS SELECT * FROM df_src")
        con.execute("ALTER TABLE corpus ADD COLUMN id INTEGER")
        con.execute("CREATE SEQUENCE seq_id START 1")
        con.execute("UPDATE corpus SET id = nextval('seq_id')")
        con.execute("CREATE INDEX idx_token_low ON corpus(_token_low)")
        con.execute("CREATE INDEX idx_id ON corpus(id)")
        con.execute("CREATE INDEX idx_lemma ON corpus(lemma)")
        con.execute("CREATE INDEX idx_sent ON corpus(sent_id)")
        
        total_tokens = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
        token_freqs = con.execute("SELECT _token_low, count(*) FROM corpus GROUP BY _token_low").fetchall()
        token_counts = {row[0]: row[1] for row in token_freqs}
        corpus_stats = {'token_counts': token_counts, 'total_tokens': total_tokens}
        
        con.close()
    except Exception as e:
        return {'error': f"DuckDB Ingestion Failed: {e}"}

    return {
        'db_path': db_path,
        'stats': corpus_stats,
        'target_map': target_sent_map,
        'src_lang': src_lang,
        'tgt_lang': tgt_lang,
        'error': None
    }

# Mapping from folder names to language codes
FOLDER_TO_LANG_MAP = {
    'indonesian': 'Indonesian',
    'english': 'English',
    'arabic': 'Arabic',
    'chinese': 'Chinese',
    'japanese': 'Japanese',
    'korean': 'Korean',
    'javanese': 'Javanese',
    'hindi': 'Hindi'
}

def load_built_in_corpus(name, url, progress_callback=None):
    """Downloads or loads one or more built-in corpora."""
    # Support both single and multiple corpora
    if isinstance(name, str):
        names = [name]
        urls = [url]
    else:
        names = name
        urls = url

    file_sources = []
    detected_lang = 'English'  # Default fallback
    
    try:
        for idx, (corpus_name, corpus_url) in enumerate(zip(names, urls)):
            filename = corpus_url
            local_path = os.path.join(CORPORA_DIR, filename)
            use_local = os.path.exists(local_path)
            
            # Detect language from folder name in the path
            # Extract first path component (folder name)
            path_parts = filename.replace('\\', '/').split('/')
            if len(path_parts) > 0:
                folder_name = path_parts[0].lower()
                if folder_name in FOLDER_TO_LANG_MAP:
                    detected_lang = FOLDER_TO_LANG_MAP[folder_name]
            
            if use_local:
                if progress_callback:
                    progress_callback(0.05 + (idx/len(names))*0.2, f"Loading local {corpus_name}...")
                with open(local_path, 'rb') as f:
                    file_bytes = f.read()
                    fs = io.BytesIO(file_bytes)
                    fs.name = filename
                    file_sources.append(fs)
            else:
                if filename.startswith("http"):
                    if progress_callback:
                        progress_callback(0.05 + (idx/len(names))*0.2, f"Downloading {corpus_name}...")
                    response = requests.get(filename, timeout=60)
                    response.raise_for_status()
                    file_bytes = response.content
                    fs = io.BytesIO(file_bytes)
                    fs.name = filename.split('/')[-1]
                    file_sources.append(fs)
                else:
                    return {'error': f"File not found locally in {CORPORA_DIR} and is not a URL: {filename}"}

        if not file_sources:
            return {'error': "No corpora files could be loaded."}

        # Determine format (use XML if any are XML)
        fmt = '.txt / auto'
        if any(fs.name.lower().endswith('.xml') for fs in file_sources):
            fmt = 'XML (Tagged)' 
        elif any('europarl' in n.lower() for n in names):
            fmt = 'verticalised (T/P/L)'

        # Pass detected language instead of hardcoded 'en'
        result = load_monolingual_corpus_files(file_sources, detected_lang, fmt, progress_callback=progress_callback)
        
        # Ensure detected language is saved if successfully loaded
        if result and not result.get('error'):
            from core.modules.overview import set_corpus_language
            set_corpus_language(result['db_path'], detected_lang)
            
        return result
        
    except Exception as e:
        return {'error': f"Failed to load built-in corpora: {e}"}

def _load_local_tagset(db_path, corpus_filename):
    """
    Looks for a corresponding .xlsx file in TAGSET_DIR and loads definitions.
    Filename matching:
       Corpus: 'MyCorpus.xml' -> Tagset: 'MyCorpus.xlsx'
    """
    if not TAGSET_DIR or not os.path.exists(TAGSET_DIR):
        return

    basename = os.path.splitext(corpus_filename)[0]
    # Check for .xlsx, .xls
    tagset_path = os.path.join(TAGSET_DIR, basename + ".xlsx")
    
    if not os.path.exists(tagset_path):
        # Try finding a file that *starts* with the basename?
        # User request: "searching file with the same name but with xlsx extension"
        return

    try:
        # Load Excel
        df = pd.read_excel(tagset_path)
        if df.shape[1] >= 2:
            # Assume Col 1 = Tag, Col 2 = Definition
            definitions = {}
            for _, row in df.iterrows():
                tag = str(row.iloc[0]).strip()
                defn = str(row.iloc[1]).strip()
                if tag and defn:
                    definitions[tag] = defn
            
            if definitions:
                save_pos_definitions(db_path, definitions)
                print(f"Loaded {len(definitions)} POS definitions from {tagset_path}")
    except Exception as e:
        print(f"Failed to load tagset from {tagset_path}: {e}")
