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
import core.preprocessing.tagging as tagging
import time
from core.utils.profiler import profile_func

@profile_func
def load_monolingual_corpus_files(file_sources, explicit_lang_code=None, selected_format="Raw", progress_callback=None, custom_tagger_config=None, eaf_main_tier=None, eaf_gloss_tier=None, eaf_trans_tier=None, whisper_lang=None):
    """
    Load and index a monolingual corpus from given file-like objects.
    Returns: dict { 'db_path': str, 'stats': dict, 'structure': dict, 'lang_code': str, 'error': str }
    """
    if progress_callback:
        progress_callback(0.0, "Initializing Corpus Loader...")

    if not file_sources:
        return {'error': "No files provided"}

    # --- SMART DUCKDB CACHING SYSTEM ---
    import hashlib
    try:
        cache_components = [str(selected_format), str(explicit_lang_code)]
        for fs in file_sources:
            fname = getattr(fs, 'name', 'file')
            fsize = 0
            if hasattr(fs, 'size'):
                fsize = fs.size
            elif hasattr(fs, 'seek') and hasattr(fs, 'tell'):
                try:
                    curr_pos = fs.tell()
                    fs.seek(0, 2)
                    fsize = fs.tell()
                    fs.seek(curr_pos)
                except Exception:
                    fsize = 0
            cache_components.append(f"{fname}_{fsize}")
            
        cache_key = hashlib.md5("_".join(cache_components).encode('utf-8')).hexdigest()
        cache_dir = os.path.join(tempfile.gettempdir(), "cortex_db_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cached_db_path = os.path.join(cache_dir, f"corpus_cached_{cache_key}.db")
        
        if os.path.exists(cached_db_path):
            if progress_callback:
                progress_callback(1.0, "⚡ Fast loading from cached database...")
            try:
                with duckdb.connect(cached_db_path, read_only=True) as con:
                    total_tokens = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
                    token_freqs = con.execute("SELECT _token_low, count(*) FROM corpus GROUP BY _token_low").fetchall()
                    token_counts = {row[0]: row[1] for row in token_freqs}
                    corpus_stats = {'token_counts': token_counts, 'total_tokens': total_tokens}
                    
                from core.modules.overview import get_corpus_language, get_xml_structure
                stored_lang = get_corpus_language(cached_db_path) or explicit_lang_code
                stored_struct = get_xml_structure(cached_db_path) or {}
                
                print(f"⚡ CACHE HIT! Instantly loaded {cached_db_path}")
                return {
                    'db_path': cached_db_path,
                    'stats': corpus_stats,
                    'structure': stored_struct,
                    'lang_code': stored_lang,
                    'error': None,
                    'warning': None
                }
            except Exception as ce:
                print(f"Cache check warning: {ce}")
    except Exception as e:
        print(f"Cache init warning: {e}")

    all_df_data = []
    
    # Defaults
    source_lang_code = explicit_lang_code
    is_tagged_format = 'Tagged' in selected_format
    use_stanza = 'Raw' in selected_format
    xml_detected_lang_code = None
    combined_structure = {}
    stanza_warning = None

    # Map language label to code for Stanza (e.g. "English" -> "en")
    from core.config import STANZA_LANG_MAP
    stanza_lang_code = explicit_lang_code
    # Search for label in keys
    if explicit_lang_code in STANZA_LANG_MAP:
        stanza_lang_code = STANZA_LANG_MAP[explicit_lang_code]
    elif explicit_lang_code.capitalize() in STANZA_LANG_MAP:
        stanza_lang_code = STANZA_LANG_MAP[explicit_lang_code.capitalize()]
    
    custom_tagger = None
    if custom_tagger_config:
        if 'pre_trained_tagger' in custom_tagger_config:
            custom_tagger = custom_tagger_config['pre_trained_tagger']
        elif custom_tagger_config.get('custom_type') == 'Rule-Based':
            from core.preprocessing.custom_tagger import CustomRuleBasedTagger
            custom_tagger = custom_tagger_config.get('rule_based_tagger')
            if custom_tagger is None:
                custom_tagger = CustomRuleBasedTagger()
        else:
            from core.preprocessing.custom_tagger import CustomDataDrivenTagger
            custom_tagger = CustomDataDrivenTagger(
                guesser_tag=custom_tagger_config.get('guesser_tag', 'NN'),
                algorithm=custom_tagger_config.get('algorithm', 'Averaged Perceptron'),
                context_window=custom_tagger_config.get('context_window', 2),
                prob_threshold=custom_tagger_config.get('prob_threshold', 0.1)
            )
            try:
                custom_tagger.train(
                    corpus_content=custom_tagger_config['corpus_content'],
                    lexicon_content=custom_tagger_config.get('lexicon_content')
                )
            except Exception as e:
                return {'error': f"Failed to train custom tagger: {e}"}
        # Initialize an empty buffer to collect annotated vertical text output
        custom_tagger.annotated_corpus_text = ""

    def make_custom_tagger_wrapper(tagger, s_lang):
        def custom_tagger_wrapper(text, lang_code=None, progress_callback=None, base_progress=0.0):
            if progress_callback: progress_callback(base_progress, f"Tagging with Custom Tagger...")
            try:
                if hasattr(tagger, 'tokenize_with_mwu'):
                    sentences = tagger.tokenize_with_mwu(text, s_lang)
                else:
                    sentences = tagging.tokenize_text_only(text, s_lang)
                tagged_results = []
                sent_id = 0
                
                # Build vertical representation
                annotated_lines = []
                
                for sent_tokens in sentences:
                    sent_id += 1
                    tagged_tokens = tagger.tag(sent_tokens)
                    for t_idx, token_info in enumerate(tagged_tokens):
                        word = sent_tokens[t_idx]
                        pos = token_info['pos']
                        lemma = token_info['lemma']
                        tagged_results.append({
                            'token': word,
                            'pos': pos,
                            'lemma': lemma,
                            'sent_id': sent_id,
                            'ent_type': ""
                        })
                        # Format: word <tab> tag <tab> lemma
                        annotated_lines.append(f"{word}\t{pos}\t{lemma}")
                    
                    # Separate sentences by an empty line
                    annotated_lines.append("")
                    
                # Append vertical output of this text block/file to the tagger buffer
                tagger.annotated_corpus_text += "\n".join(annotated_lines) + "\n"
                if progress_callback: progress_callback(base_progress, f"Tagged with Custom Tagger successfully!")
                return tagged_results, None
            except Exception as e:
                if progress_callback: progress_callback(base_progress, f"Tagging with Custom Tagger failed.")
                return None, str(e)
        return custom_tagger_wrapper
    
    print(f"DEBUG: load_monolingual_corpus_files called. Lang: {explicit_lang_code} (Stanza: {stanza_lang_code}), Format: {selected_format}")

    # Unpack ZIP files if present
    expanded_file_sources = []
    import zipfile
    for fs in file_sources:
        if fs.name.lower().endswith('.zip'):
            try:
                fs.seek(0)
                zip_bytes = fs.read()
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                    for member in zf.infolist():
                        if not member.is_dir() and not member.filename.startswith('__MACOSX') and not os.path.basename(member.filename).startswith('.'):
                            m_bytes = zf.read(member.filename)
                            m_fs = io.BytesIO(m_bytes)
                            m_fs.name = os.path.basename(member.filename)
                            expanded_file_sources.append(m_fs)
            except Exception as e:
                print(f"Warning: Failed to unpack ZIP file {fs.name}: {e}")
        else:
            expanded_file_sources.append(fs)
            
    file_sources = expanded_file_sources

    # Identify companion .wav files to skip them in the main loop (they are handled by TextGrid/EAF parsers)
    standalone_wav_sources = []
    companion_wav_sources = {}
    other_sources = []
    
    textgrid_basenames = {os.path.splitext(fs.name.lower())[0] for fs in file_sources if fs.name.lower().endswith(('.textgrid', '.eaf'))}
    
    for fs in file_sources:
        if fs.name.lower().endswith('.wav'):
            base = os.path.splitext(fs.name.lower())[0]
            if base not in textgrid_basenames:
                standalone_wav_sources.append(fs)
            else:
                companion_wav_sources[base] = fs
        else:
            other_sources.append(fs)
            
    file_sources = other_sources + standalone_wav_sources
    num_files = len(file_sources)

    for idx, file_source in enumerate(file_sources):
        if progress_callback:
            progress_callback(idx / num_files, f"Reading file: {file_source.name}...")
            
        file_source.seek(0)
        filename = file_source.name
        
        # Read a sample to detect pseudo-XML
        sample_bytes = file_source.read(1024)
        file_source.seek(0)
        sample_str = sample_bytes.decode('utf-8', errors='ignore').strip()
        
        is_xml_ext = filename.lower().endswith('.xml') or filename.lower().endswith('.eaf')
        is_eaf_ext = filename.lower().endswith('.eaf')
        is_conllu_ext = filename.lower().endswith('.conllu')
        is_docx_ext = filename.lower().endswith('.docx')
        is_pdf_ext = filename.lower().endswith('.pdf')
        is_textgrid_ext = filename.lower().endswith('.textgrid')
        is_wav_ext = filename.lower().endswith('.wav')
        is_pseudo_xml = False
        if not is_xml_ext and not is_conllu_ext and not is_docx_ext and not is_pdf_ext and not is_textgrid_ext and not is_wav_ext:
            if sample_str.startswith('<'):
                is_pseudo_xml = True
            elif any(tag in sample_str.lower() for tag in ['<text', '<corpus', '<p>', '<p ']):
                is_pseudo_xml = True

        # --- XML / EAF PROCESSING ---
        if is_xml_ext or is_pseudo_xml:
            try:
                xml_content = file_source.read().decode('utf-8', errors='ignore')
                cleaned_xml = sanitize_xml_content(xml_content)
                
                # Check if file is ELAN (.eaf or <ANNOTATION_DOCUMENT>)
                if is_eaf_ext or '<annotation_document' in cleaned_xml.lower():
                    from .xml_parser import parse_eaf_content_to_df_records
                    stanza_proc = None
                    if custom_tagger:
                        stanza_proc = make_custom_tagger_wrapper(custom_tagger, stanza_lang_code)
                    elif stanza_lang_code and stanza_lang_code != "OTHER":
                        stanza_proc = tagging.tag_text_with_stanza

                    eaf_records = parse_eaf_content_to_df_records(cleaned_xml, stanza_processor=stanza_proc, lang_code=stanza_lang_code, filename=filename, eaf_main_tier=eaf_main_tier, eaf_gloss_tier=eaf_gloss_tier, eaf_trans_tier=eaf_trans_tier)
                    all_df_data.extend(eaf_records)
                else:
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
                    if custom_tagger:
                        stanza_proc = make_custom_tagger_wrapper(custom_tagger, stanza_lang_code)
                    elif stanza_lang_code and stanza_lang_code != "OTHER":
                        stanza_proc = tagging.tag_text_with_stanza
                    
                    is_uam_format = 'uam' in str(selected_format).lower()
                    result = parse_xml_content_to_df(
                        cleaned_xml, 
                        stanza_processor=stanza_proc, 
                        lang_code=stanza_lang_code,
                        preserve_inline_tags=True,
                        is_uam_xml=is_uam_format
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
                
        # --- CONLL-U PROCESSING ---
        elif is_conllu_ext:
            try:
                file_bytes = file_source.read()
                file_content_str = file_bytes.decode('utf-8', errors='ignore')
                lines = file_content_str.splitlines()
                
                current_sent = []
                # Keep a global counter in case sent_id is missing or we just want sequential
                sent_id_counter = 0
                
                def flush_conllu_sent():
                    nonlocal sent_id_counter
                    if not current_sent: return
                    sent_id_counter += 1
                    
                    id_to_token = {r['_conllu_id']: r['token'] for r in current_sent}
                    id_to_token['0'] = 'ROOT'
                    
                    for r in current_sent:
                        head_id = r.get('dep_head_id', '')
                        if head_id in id_to_token:
                            r['dep_head_token'] = id_to_token[head_id]
                        else:
                            r['dep_head_token'] = ''
                            
                        # Clean up internal id
                        if '_conllu_id' in r:
                            del r['_conllu_id']
                            
                        r['sent_id'] = sent_id_counter
                        r['filename'] = filename
                        all_df_data.append(r)
                    current_sent.clear()
                    
                for line in lines:
                    line = line.strip()
                    if not line:
                        flush_conllu_sent()
                        continue
                    if line.startswith('#'):
                        continue
                        
                    parts = line.split('\t')
                    if len(parts) >= 8:
                        token_id = parts[0]
                        if progress_callback:
                            progress_callback(0.92, "Building DuckDB analytical database...")
                        if '-' in token_id or '.' in token_id:
                            continue # skip multi-word tokens or empty nodes
                            
                        token = parts[1]
                        lemma = parts[2] if parts[2] != '_' else token
                        pos = parts[3] if parts[3] != '_' else (parts[4] if len(parts) > 4 and parts[4] != '_' else 'TAG')
                        head = parts[6] if len(parts) > 6 and parts[6] != '_' else ''
                        deprel = parts[7] if len(parts) > 7 and parts[7] != '_' else ''
                        
                        ent_type = ""
                        misc = parts[9] if len(parts) > 9 else ""
                        if misc != '_':
                            ner_match = re.search(r'ner=([A-Za-z0-9_\-]+)', misc, re.IGNORECASE)
                            if ner_match:
                                ent_type = ner_match.group(1)
                                
                        row = {
                            'token': token,
                            'pos': pos,
                            'lemma': lemma,
                            'ent_type': ent_type,
                            'dep_rel': deprel,
                            'dep_head_id': head,
                            '_conllu_id': token_id
                        }
                        current_sent.append(row)
                        
                flush_conllu_sent()
                if explicit_lang_code != 'OTHER':
                    xml_detected_lang_code = stanza_lang_code
                    
            except Exception as e:
                return {'error': f"CoNLL-U Error ({filename}): {str(e)}"}
        
        # --- TEXTGRID PROCESSING ---
        elif is_textgrid_ext:
            try:
                from .textgrid_parser import textgrid_to_dataframe
                
                # Check for companion .wav file in the same directory, or from companion_wav_sources
                audio_path = None
                tmp_audio_path = None
                
                base = os.path.splitext(filename.lower())[0]
                if base in companion_wav_sources:
                    companion_fs = companion_wav_sources[base]
                    companion_fs.seek(0)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as taf:
                        taf.write(companion_fs.read())
                        audio_path = taf.name
                        tmp_audio_path = taf.name
                elif hasattr(file_source, 'name'):
                    possible_audio = file_source.name.replace('.TextGrid', '.wav').replace('.textgrid', '.wav')
                    if os.path.exists(possible_audio):
                        audio_path = possible_audio
                        
                if hasattr(file_source, 'seek'):
                    # Save temporary file because parsers need paths
                    file_source.seek(0)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".TextGrid") as tf:
                        tf.write(file_source.read())
                        tmp_tg_path = tf.name
                        
                    tg_records = textgrid_to_dataframe(tmp_tg_path, audio_path=audio_path, progress_callback=progress_callback, base_progress=idx/num_files)
                    
                    if stanza_lang_code and stanza_lang_code != "OTHER":
                        stanza_proc = None
                        if custom_tagger:
                            stanza_proc = make_custom_tagger_wrapper(custom_tagger, stanza_lang_code)
                        else:
                            stanza_proc = tagging.tag_text_with_stanza
                            
                        if progress_callback:
                            progress_callback(idx / num_files, f"Starting tagging pipeline for {filename}...")
                            
                        # Group by sent_id
                        sents = {}
                        for r in tg_records:
                            s = r['sent_id']
                            if s not in sents: sents[s] = []
                            sents[s].append(r)
                            
                        # Batch process ALL sentences at once to avoid launching TreeTagger for every sentence
                        sentence_texts = [" ".join([r['token'] for r in records]) for records in sents.values()]
                        full_text = "\n".join(sentence_texts)
                        
                        all_res, err = stanza_proc(full_text, stanza_lang_code, progress_callback=progress_callback, base_progress=idx/num_files)
                        
                        if all_res:
                            # Try to perfectly align back to tg_records
                            if len(all_res) == len(tg_records):
                                for i, r in enumerate(tg_records):
                                    r['pos'] = all_res[i]['pos']
                                    r['lemma'] = all_res[i]['lemma']
                            else:
                                # Desync occurred (e.g. tokenizer split "don't" into "do" "n't")
                                # Use difflib for robust sequence alignment instantly!
                                import difflib
                                tg_tokens = [r['token'].lower() for r in tg_records]
                                nlp_tokens = [t['token'].lower() for t in all_res]
                                seq = difflib.SequenceMatcher(None, tg_tokens, nlp_tokens)
                                
                                for tag, i1, i2, j1, j2 in seq.get_opcodes():
                                    if tag == 'equal':
                                        for i, j in zip(range(i1, i2), range(j1, j2)):
                                            tg_records[i]['pos'] = all_res[j]['pos']
                                            tg_records[i]['lemma'] = all_res[j]['lemma']
                                    else:
                                        # For mismatched chunks (e.g. "it's" vs "it", "'s")
                                        # Map the NLP tags as best as possible without launching new processes
                                        if i2 > i1 and j2 > j1:
                                            for k in range(i2 - i1):
                                                i = i1 + k
                                                j = j1 + min(k, (j2 - j1) - 1)
                                                tg_records[i]['pos'] = all_res[j]['pos']
                                                tg_records[i]['lemma'] = all_res[j]['lemma']
                                        
                    for r in tg_records:
                        r['filename'] = filename
                    all_df_data.extend(tg_records)
                    os.remove(tmp_tg_path)
                else:
                    tg_records = textgrid_to_dataframe(file_source.name, audio_path=audio_path, progress_callback=progress_callback, base_progress=idx/num_files)
                    
                    if stanza_lang_code and stanza_lang_code != "OTHER":
                        stanza_proc = None
                        if custom_tagger:
                            stanza_proc = make_custom_tagger_wrapper(custom_tagger, stanza_lang_code)
                        else:
                            stanza_proc = tagging.tag_text_with_stanza
                            
                        if progress_callback:
                            progress_callback(idx / num_files, f"Starting tagging pipeline for {filename}...")
                            
                        # Group by sent_id
                        sents = {}
                        for r in tg_records:
                            s = r['sent_id']
                            if s not in sents: sents[s] = []
                            sents[s].append(r)
                            
                        # Batch process ALL sentences at once to avoid launching TreeTagger for every sentence
                        sentence_texts = [" ".join([r['token'] for r in records]) for records in sents.values()]
                        full_text = "\n".join(sentence_texts)
                        
                        all_res, err = stanza_proc(full_text, stanza_lang_code, progress_callback=progress_callback, base_progress=idx/num_files)
                        
                        if all_res:
                            # Try to perfectly align back to tg_records
                            if len(all_res) == len(tg_records):
                                for i, r in enumerate(tg_records):
                                    r['pos'] = all_res[i]['pos']
                                    r['lemma'] = all_res[i]['lemma']
                            else:
                                # Desync occurred (e.g. tokenizer split "don't" into "do" "n't")
                                # Use difflib for robust sequence alignment instantly!
                                import difflib
                                tg_tokens = [r['token'].lower() for r in tg_records]
                                nlp_tokens = [t['token'].lower() for t in all_res]
                                seq = difflib.SequenceMatcher(None, tg_tokens, nlp_tokens)
                                
                                for tag, i1, i2, j1, j2 in seq.get_opcodes():
                                    if tag == 'equal':
                                        for i, j in zip(range(i1, i2), range(j1, j2)):
                                            tg_records[i]['pos'] = all_res[j]['pos']
                                            tg_records[i]['lemma'] = all_res[j]['lemma']
                                    else:
                                        # For mismatched chunks (e.g. "it's" vs "it", "'s")
                                        # Map the NLP tags as best as possible without launching new processes
                                        if i2 > i1 and j2 > j1:
                                            for k in range(i2 - i1):
                                                i = i1 + k
                                                j = j1 + min(k, (j2 - j1) - 1)
                                                tg_records[i]['pos'] = all_res[j]['pos']
                                                tg_records[i]['lemma'] = all_res[j]['lemma']
                                        
                    for r in tg_records:
                        r['filename'] = filename
                    all_df_data.extend(tg_records)
                    
                if tmp_audio_path and os.path.exists(tmp_audio_path):
                    os.remove(tmp_audio_path)
            except Exception as e:
                return {'error': f"TextGrid Error ({filename}): {str(e)}"}
                
        # --- RAW AUDIO PROCESSING (ASR) ---
        elif is_wav_ext:
            # Skip if there is a corresponding TextGrid file in the upload
            if any(os.path.splitext(f.name)[0] == base and f.name.lower().endswith('.textgrid') for f in file_sources if hasattr(f, 'name')):
                continue
                
            try:
                from .asr_extractor import transcribe_audio_to_words
                from .acoustic_extractor import AcousticExtractor
                
                # We need a path for librosa to load the audio
                audio_path = None
                if hasattr(file_source, 'seek'):
                    file_source.seek(0)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
                        tf.write(file_source.read())
                        audio_path = tf.name
                else:
                    audio_path = file_source.name
                    
                # 1. Run ASR
                asr_records = transcribe_audio_to_words(audio_path, model_size="base", language=whisper_lang)
                
                # 2. Run Acoustic Extraction on the generated boundaries
                if asr_records:
                    if progress_callback: progress_callback(idx/num_files + 0.1, f"Extracting acoustic features from audio...")
                    extractor = AcousticExtractor(audio_path)
                    
                    # Assign a dummy sent_id so it can be batched for tagging
                    sent_id_counter = 1
                    for r in asr_records:
                        r['sent_id'] = sent_id_counter
                        start = r.get('start_time')
                        end = r.get('end_time')
                        if start is not None and end is not None:
                            feats = extractor.get_features_for_interval(start, end)
                            if feats:
                                r.update(feats)
                        r['filename'] = filename
                        # Basic punctuation heuristic to increment sentence ID for batching
                        if r['token'].endswith(('.', '!', '?')):
                            sent_id_counter += 1
                            
                    # 3. NLP Tagging
                    if stanza_lang_code and stanza_lang_code != "OTHER":
                        stanza_proc = None
                        if custom_tagger:
                            stanza_proc = make_custom_tagger_wrapper(custom_tagger, stanza_lang_code)
                        else:
                            stanza_proc = tagging.tag_text_with_stanza
                            
                        if progress_callback: progress_callback(idx/num_files + 0.2, f"Starting tagging pipeline for {filename}...")
                        
                        sents = {}
                        for r in asr_records:
                            s = r['sent_id']
                            if s not in sents: sents[s] = []
                            sents[s].append(r)
                            
                        sentence_texts = [" ".join([r['token'] for r in records]) for records in sents.values()]
                        full_text = "\n".join(sentence_texts)
                        
                        all_res, err = stanza_proc(full_text, stanza_lang_code, progress_callback=progress_callback, base_progress=idx/num_files)
                        
                        if all_res:
                            # Try to perfectly align back to asr_records
                            if len(all_res) == len(asr_records):
                                for i, r in enumerate(asr_records):
                                    r['pos'] = all_res[i]['pos']
                                    r['lemma'] = all_res[i]['lemma']
                            else:
                                import difflib
                                tg_tokens = [r['token'].lower() for r in asr_records]
                                nlp_tokens = [t['token'].lower() for t in all_res]
                                seq = difflib.SequenceMatcher(None, tg_tokens, nlp_tokens)
                                
                                for tag, i1, i2, j1, j2 in seq.get_opcodes():
                                    if tag == 'equal':
                                        for i, j in zip(range(i1, i2), range(j1, j2)):
                                            asr_records[i]['pos'] = all_res[j]['pos']
                                            asr_records[i]['lemma'] = all_res[j]['lemma']
                                    else:
                                        if i2 > i1 and j2 > j1:
                                            for k in range(i2 - i1):
                                                i = i1 + k
                                                j = j1 + min(k, (j2 - j1) - 1)
                                                asr_records[i]['pos'] = all_res[j]['pos']
                                                asr_records[i]['lemma'] = all_res[j]['lemma']
                                                
                    # Fill missing tags with fallbacks if NLP failed or was skipped
                    for r in asr_records:
                        if 'pos' not in r: r['pos'] = '##TAG'
                        if 'lemma' not in r: r['lemma'] = r['token']
                        
                    all_df_data.extend(asr_records)
                else:
                    print(f"Warning: ASR returned no words for {filename}")
                
                # Cleanup if temporary
                if hasattr(file_source, 'seek') and audio_path:
                    os.remove(audio_path)
                    
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                return {'error': f"ASR Error ({filename}): {str(e)}"}
                
        # --- TXT/CSV/DOCX/PDF PROCESSING ---
        else: 
            try:
                if is_docx_ext:
                    import docx
                    doc = docx.Document(file_source)
                    file_content_str = "\n".join([p.text for p in doc.paragraphs])
                elif is_pdf_ext:
                    import pypdf
                    reader = pypdf.PdfReader(file_source)
                    pages_text = []
                    for page in reader.pages:
                        pages_text.append(page.extract_text() or "")
                    file_content_str = "\n".join(pages_text)
                    
                    # Quality Check
                    if len(file_content_str.strip()) < len(reader.pages) * 20:
                        return {'error': f"PDF Quality Error ({filename}): Extracted text is too short. This might be a scanned document without OCR. Process halted."}
                else:
                    file_bytes = file_source.read()
                    file_content_str = file_bytes.decode('utf-8', errors='ignore')
                
                clean_lines = [line for line in file_content_str.splitlines() if line and not line.strip().startswith('#')]
                clean_content = "\n".join(clean_lines)
            except Exception as e:
                return {'error': f"Error reading document {filename}: {str(e)}"}

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
                    log_file = f"ingestion_{int(time.time())}.log"
                    with open(log_file, "a") as f:
                        f.write(f"File {filename} could not be parsed as vertical format. Falling back to raw text.\n")
                    print(f"File {filename} could not be parsed as vertical format. Falling back to raw text.")
                    current_is_tagged = False 
            
            if not current_is_tagged or 'Raw' in selected_format: 
                raw_text = clean_content
                
                # Tagging Logic Integration
                # If explicit_lang_code is set, we try to use it.
                # If "OTHER" is selected, we perform fallback tagging.
                
                # Log execution start
                log_file = f"ingestion_{int(time.time())}.log"
                with open(log_file, "a") as f:
                    f.write(f"Processing raw text. lang='{explicit_lang_code}', stanza_lang='{stanza_lang_code}', format='{selected_format}'\n")
                
                tagged_data = []
                
                if custom_tagger:
                    stanza_proc = make_custom_tagger_wrapper(custom_tagger, stanza_lang_code)
                    tagged_data, err = stanza_proc(raw_text)
                elif stanza_lang_code and stanza_lang_code != "OTHER":
                    try:
                        # Attempt Stanza Tagging
                        tagged_data, err = tagging.tag_text_with_stanza(raw_text, stanza_lang_code)
                        if err:
                            stanza_warning = f"Stanza error: {err}. Using simple fallback."
                            with open(log_file, "a") as f:
                                f.write(f"Stanza Error: {err}\n")
                    except Exception as e:
                        import traceback
                        error_trace = traceback.format_exc()
                        msg = f"Stanza execution failed: {e}. Using simple fallback."
                        print(msg)
                        with open(log_file, "a") as f:
                            f.write(f"Stanza Exception: {e}\n{error_trace}\n")
                        stanza_warning = msg
                        tagged_data, err = tagging.tag_text_simple_fallback(raw_text)
                else:
                    with open(log_file, "a") as f:
                        f.write("Using simple fallback because lang is OTHER or None.\n")
                    # Fallback to simple tagging for "OTHER" or if no language code
                    tagged_data, err = tagging.tag_text_simple_fallback(raw_text)

                # Add metadata
                for item in tagged_data:
                    item['filename'] = filename
                
                all_df_data.extend(tagged_data)

    if progress_callback: progress_callback(0.9, "Structuring data for database indexing...")
    
    if not all_df_data:
        return {'error': "No valid data extracted from files"}

    # --- DUCKDB DATA INGESTION ---
    unique_filename = f"corpus_{uuid.uuid4().hex}.duckdb"
    db_path = os.path.join(tempfile.gettempdir(), unique_filename)
    
    if os.path.exists(db_path):
        try: os.remove(db_path)
        except: pass

    try:
        with duckdb.connect(db_path) as con:
            df_src = pd.DataFrame(all_df_data)
            
            for col in ['token', 'pos', 'lemma', 'sent_id', 'filename', 'ent_type']:
                if col not in df_src.columns:
                    df_src[col] = "" if col in ['pos', 'lemma', 'ent_type'] else 0
                
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
        
    except Exception as e:
        return {'error': f"DuckDB Ingestion Failed: {e}"}

    final_lang_code = xml_detected_lang_code if xml_detected_lang_code else source_lang_code
    
    # Save Language to Metadata
    from core.modules.overview import set_corpus_language, set_corpus_metadata
    set_corpus_language(db_path, final_lang_code)
    
    # Determine actual tagger used
    actual_tagger = "Pre-tagged"
    actual_tagset = "Unknown"
    
    if 'current_is_tagged' in locals() and current_is_tagged or 'Vertical' in selected_format:
        actual_tagger = "Pre-tagged (User Provided)"
    elif custom_tagger:
        if getattr(custom_tagger, 'tagger_type', None) == "Rule-Based":
            actual_tagger = "Custom Rule-Based Tagger"
        else:
            actual_tagger = "Custom Data-Driven Tagger"
    else:
        if 'stanza_warning' in locals() and stanza_warning and 'switching to SpaCy' in stanza_warning:
            actual_tagger = "SpaCy (Fallback from TreeTagger)"
        elif 'stanza_warning' in locals() and stanza_warning and 'switching to Stanza' in stanza_warning:
            actual_tagger = "Stanza (Fallback)"
        elif final_lang_code in ['id', 'mg', 'en']:
            actual_tagger = "TreeTagger"
            if final_lang_code == 'id':
                actual_tagset = "ID-BPPT Indonesian Tagset"
            elif final_lang_code == 'mg':
                actual_tagset = "TreeTagger (Malagasy)"
            elif final_lang_code == 'en':
                actual_tagset = "Penn Treebank (PTB)"
        elif 'stanza_warning' in locals() and stanza_warning:
            actual_tagger = "Simple Fallback Tagger"
        else:
            actual_tagger = "SpaCy/Stanza Pipeline"
            
    set_corpus_metadata(db_path, 'tagger', actual_tagger)
    if actual_tagset != "Unknown":
        set_corpus_metadata(db_path, 'tagset', actual_tagset)
    
    # Save XML structure to Metadata if present
    if combined_structure:
        from core.modules.overview import set_xml_structure
        set_xml_structure(db_path, combined_structure)
    
    # Auto-load local tagset definitions if available
    # Iterate through input files to find a matching tagset (taking the first match)
    for fs in file_sources:
        fname = getattr(fs, 'name', '')
        if fname:
            _load_local_tagset(db_path, fname)

    # Auto-apply Indonesian Semantic Tagging if corpus language is Indonesian
    if final_lang_code in ('id', 'ID', 'Indonesian'):
        try:
            _apply_indonesian_semantic_tagging_to_db(db_path)
        except Exception as _e_sem:
            print(f"Auto-semantic tagging error: {_e_sem}")

    # Save a copy to cache for instant future loading
    if 'cached_db_path' in locals() and cached_db_path:
        try:
            import shutil
            shutil.copyfile(db_path, cached_db_path)
            print(f"⚡ Saved compiled corpus to fast DB cache: {cached_db_path}")
        except Exception as e:
            print(f"Cache save warning: {e}")
    
    # Generate universal annotated corpus text for download
    annotated_lines = []
    current_sent_id = None
    for row in all_df_data:
        if current_sent_id is not None and current_sent_id != row.get('sent_id'):
            annotated_lines.append("")
        current_sent_id = row.get('sent_id')
        annotated_lines.append(f"{row.get('token', '')}\t{row.get('pos', '')}\t{row.get('lemma', '')}")
    annotated_text = "\n".join(annotated_lines) + "\n"
    
    return {
        'db_path': db_path,
        'stats': corpus_stats,
        'structure': combined_structure,
        'lang_code': final_lang_code,
        'error': None,
        'warning': stanza_warning if 'stanza_warning' in locals() else None,
        'trained_tagger': custom_tagger,
        'annotated_corpus_text': annotated_text
    }

@profile_func
def load_xml_parallel_corpus(src_file, tgt_file, src_lang_code, tgt_lang_code, progress_callback=None):
    if src_file is None or tgt_file is None: return {'error': "Files missing"}

    try:
        # 1. Parsing Source
        if progress_callback: progress_callback(0.1, "Parsing source...")
        src_file.seek(0)
        src_content = src_file.read().decode('utf-8', errors='ignore')
        src_cleaned = sanitize_xml_content(src_content)
        
        from core.config import STANZA_LANG_MAP
        
        # Source Stanza
        src_stanza_code = src_lang_code
        if src_lang_code in STANZA_LANG_MAP: src_stanza_code = STANZA_LANG_MAP[src_lang_code]
        elif src_lang_code.capitalize() in STANZA_LANG_MAP: src_stanza_code = STANZA_LANG_MAP[src_lang_code.capitalize()]
        
        src_proc = None
        if src_stanza_code and src_stanza_code != "OTHER": src_proc = tagging.tag_text_with_stanza
        
        src_result = parse_xml_content_to_df(src_cleaned, stanza_processor=src_proc, lang_code=src_stanza_code, preserve_inline_tags=True)

        # 2. Parsing Target
        if progress_callback: progress_callback(0.5, "Parsing target...")
        tgt_file.seek(0)
        tgt_content = tgt_file.read().decode('utf-8', errors='ignore')
        tgt_cleaned = sanitize_xml_content(tgt_content)
        
        # Target Stanza
        tgt_stanza_code = tgt_lang_code
        if tgt_lang_code in STANZA_LANG_MAP: tgt_stanza_code = STANZA_LANG_MAP[tgt_lang_code]
        elif tgt_lang_code.capitalize() in STANZA_LANG_MAP: tgt_stanza_code = STANZA_LANG_MAP[tgt_lang_code.capitalize()]
        
        tgt_proc = None
        if tgt_stanza_code and tgt_stanza_code != "OTHER": tgt_proc = tagging.tag_text_with_stanza
        
        tgt_result = parse_xml_content_to_df(tgt_cleaned, stanza_processor=tgt_proc, lang_code=tgt_stanza_code, preserve_inline_tags=True)
        
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
        with duckdb.connect(db_path) as con:
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
        with duckdb.connect(db_path) as con:
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
    'hindi': 'Hindi',
    'malagasy': 'Malagasy'
}

@profile_func
def load_built_in_corpus(name, url, progress_callback=None):
    """Downloads or loads one or more built-in corpora."""
    from core.config import DOWNLOADABLE_ASSETS_MAP
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
            
            # If file doesn't exist locally but is in our downloadable assets, download it first
            if not os.path.exists(local_path) and filename in DOWNLOADABLE_ASSETS_MAP:
                download_url = DOWNLOADABLE_ASSETS_MAP[filename]
                if progress_callback:
                    progress_callback(0.05, f"Downloading database for {corpus_name}...")
                download_file(download_url, local_path, progress_callback)
                
            # Check if there is a .zip file counterpart if local_path doesn't exist directly
            if not os.path.exists(local_path) and os.path.exists(local_path.rsplit('.', 1)[0] + '.zip'):
                zip_path = local_path.rsplit('.', 1)[0] + '.zip'
                import zipfile
                if progress_callback:
                    progress_callback(0.05 + (idx/len(names))*0.1, f"Extracting {corpus_name} archive...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(local_path))
                    
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
                
                # Special fast-path for pre-built DuckDB database files (.db, .duckdb)
                if filename.lower().endswith(('.db', '.duckdb')):
                    if progress_callback:
                        progress_callback(0.8, f"Configuring database {corpus_name}...")
                    
                    import uuid
                    import shutil
                    import json
                    
                    # For pre-built DuckDB database files, use local_path directly without copying to temp
                    temp_db_path = local_path
                    
                    # Read metadata directly from the pre-built database in read-only mode
                    con = duckdb.connect(temp_db_path, read_only=True)
                    try:
                        # 1. Get language
                        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
                        lang = detected_lang
                        if 'corpus_metadata' in tables:
                            res_lang = con.execute("SELECT value FROM corpus_metadata WHERE key='language'").fetchone()
                            if res_lang:
                                lang = res_lang[0]
                                
                        # 2. Get XML structure (if stored)
                        structure = {}
                        if 'corpus_metadata' in tables:
                            res_struct = con.execute("SELECT value FROM corpus_metadata WHERE key='xml_structure'").fetchone()
                            if res_struct:
                                try:
                                    serializable_struct = json.loads(res_struct[0])
                                    for tag in serializable_struct:
                                        structure[tag] = {}
                                        for attr in serializable_struct[tag]:
                                            structure[tag][attr] = set(serializable_struct[tag][attr])
                                except Exception as e:
                                    print(f"Error restoring xml_structure: {e}")
                                    
                        # 3. Get Stats (O(1) count instead of O(N) grouping 100M tokens into RAM)
                        total_tokens = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
                        stats = {'token_counts': {}, 'total_tokens': total_tokens}
                        
                    except Exception as e:
                        con.close()
                        if "IO Error" in str(e) or "read enough bytes" in str(e):
                            try:
                                os.remove(local_path)
                            except:
                                pass
                            return {'error': f"Failed to load corpus (corrupted file). The corrupted file has been deleted. Please try loading it again to trigger a fresh download. (Details: {e})"}
                        return {'error': f"Failed to read database metadata: {e}"}
                    
                    con.close()
                    
                    # Auto-load tagset definitions if available
                    _load_local_tagset(temp_db_path, filename)
                    
                    if progress_callback:
                        progress_callback(1.0, f"Successfully loaded {corpus_name}!")
                        
                    return {
                        'db_path': temp_db_path,
                        'stats': stats,
                        'structure': structure,
                        'lang_code': lang,
                        'error': None
                    }
                else:
                    with open(local_path, 'rb') as f:
                        file_bytes = f.read()
                    fs = io.BytesIO(file_bytes)
                    fs.name = local_path
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
                    # Try fetching from Hugging Face Dataset (prihantoro-corpus/cortex-data)
                    hf_raw_url = f"https://huggingface.co/datasets/prihantoro-corpus/cortex-data/raw/main/corpora/{filename}"
                    zip_filename = filename.rsplit('.', 1)[0] + '.zip'
                    hf_zip_url = f"https://huggingface.co/datasets/prihantoro-corpus/cortex-data/raw/main/corpora/{zip_filename}"
                    
                    if progress_callback:
                        progress_callback(0.05 + (idx/len(names))*0.2, f"Downloading {corpus_name} from dataset repository...")
                    try:
                        # Try direct file first, then zip version
                        response = requests.get(hf_raw_url, timeout=60)
                        if response.status_code != 200:
                            response = requests.get(hf_zip_url, timeout=120)
                            response.raise_for_status()
                            zip_target_path = os.path.join(CORPORA_DIR, zip_filename)
                            os.makedirs(os.path.dirname(zip_target_path), exist_ok=True)
                            with open(zip_target_path, 'wb') as out_f:
                                out_f.write(response.content)
                            import zipfile
                            with zipfile.ZipFile(zip_target_path, 'r') as zip_ref:
                                zip_ref.extractall(os.path.dirname(local_path))
                            with open(local_path, 'rb') as extracted_f:
                                file_bytes = extracted_f.read()
                        else:
                            os.makedirs(os.path.dirname(local_path), exist_ok=True)
                            with open(local_path, 'wb') as out_f:
                                out_f.write(response.content)
                            file_bytes = response.content
                            
                        fs = io.BytesIO(file_bytes)
                        fs.name = local_path
                        file_sources.append(fs)
                    except Exception as download_err:
                        return {'error': f"File not found locally in {CORPORA_DIR} and failed to download from dataset repository: {download_err}"}

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

def download_file(url, local_path, progress_callback=None):
    """Downloads a file from a URL to local_path with progress updates."""
    import requests
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB
    
    downloaded = 0
    with open(local_path, 'wb') as f:
        for data in response.iter_content(block_size):
            downloaded += len(data)
            f.write(data)
            if total_size > 0 and progress_callback:
                percent = downloaded / total_size
                progress_callback(0.05 + percent * 0.7, f"Downloading: {downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB")

def _apply_indonesian_semantic_tagging_to_db(db_path):
    """
    Applies Indonesian Semantic Lexicon tags to all tokens in the given DuckDB database.
    """
    excel_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'wordlist', 'indonesian', 'semantic', '_File Induk Semantic Lexicon_31_03_2023.xlsx')
    if not os.path.exists(excel_path):
        return
        
    import openpyxl
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    sheet = wb.active
    
    lexicon = {}
    for i, row in enumerate(sheet.iter_rows(values_only=True)):
        if i == 0: continue
        w_entry = row[10]
        if not w_entry: continue
        w = str(w_entry).strip()
        if not w or w.lower() == 'none': continue
        
        raw_tags = [str(c).strip() for c in row[11:16] if c is not None and str(c).strip() and str(c).strip().lower() != 'none']
        tag_str = 'Z99' if not raw_tags else '|'.join(raw_tags)
        
        if w not in lexicon: lexicon[w] = set()
        for t in tag_str.split('|'): lexicon[w].add(t)

    final_lexicon = {}
    for w, tset in lexicon.items():
        s_val = '|'.join(sorted(tset))
        final_lexicon[w] = s_val
        final_lexicon[w.lower()] = s_val

    with duckdb.connect(db_path, read_only=False) as con:
        cols_info = con.execute('PRAGMA table_info(corpus)').fetch_df()
        existing_cols = [c.lower() for c in cols_info['name'].tolist()]
        if 'semantic' not in existing_cols:
            con.execute('ALTER TABLE corpus ADD COLUMN semantic VARCHAR')
            
        tokens_df = con.execute('SELECT DISTINCT _token_low FROM corpus WHERE _token_low IS NOT NULL').fetch_df()
        updates = []
        for tok in tokens_df['_token_low']:
            stag = final_lexicon.get(tok, 'Z99')
            updates.append((stag, tok))
            
        con.execute('CREATE TEMP TABLE sem_map (tag VARCHAR, token_low VARCHAR)')
        con.executemany('INSERT INTO sem_map VALUES (?, ?)', updates)
        
        con.execute('UPDATE corpus SET semantic = sem_map.tag FROM sem_map WHERE corpus._token_low = sem_map.token_low')
        con.execute('DROP TABLE sem_map')
        print(f"Auto-annotated Indonesian corpus at {db_path} with USAS semantic tags.")

