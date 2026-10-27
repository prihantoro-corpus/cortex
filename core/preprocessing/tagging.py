import warnings
warnings.filterwarnings("ignore", message=".*Examining the path of torch.classes raised.*")
warnings.filterwarnings("ignore", category=UserWarning)

import stanza
import logging

# Configure logging
logging.getLogger('stanza').setLevel(logging.WARNING)

# Cache for Stanza pipelines to avoid reloading
_STANZA_PIPELINES = {}
_SPACY_PIPELINES = {}

# Map standard language codes to small/fast SpaCy models
SPACY_MODEL_MAP = {
    'en': 'en_core_web_sm',
    'id': 'id_core_news_sm',
    'zh': 'zh_core_web_sm',
    'ja': 'ja_core_news_sm',
    'ko': 'ko_core_news_sm',
    'de': 'de_core_news_sm',
    'fr': 'fr_core_news_sm',
    'es': 'es_core_news_sm',
    'it': 'it_core_news_sm',
    'ru': 'ru_core_news_sm',
    'pt': 'pt_core_news_sm',
    'nl': 'nl_core_news_sm',
    'el': 'el_core_news_sm',
    'pl': 'pl_core_news_sm',
    'uk': 'uk_core_news_sm',
    'ro': 'ro_core_news_sm',
    'sv': 'sv_core_news_sm',
    'da': 'da_core_news_sm',
    'nb': 'nb_core_news_sm',
    'fi': 'fi_core_news_sm',
    'ca': 'ca_core_news_sm',
    'hr': 'hr_core_news_sm',
    'lt': 'lt_core_news_sm',
    'mk': 'mk_core_news_sm',
    'sl': 'sl_core_news_sm',
}

def get_spacy_pipeline(lang_code):
    """
    Get or dynamically download a SpaCy pipeline for the specified language.
    """
    global _SPACY_PIPELINES
    if lang_code in _SPACY_PIPELINES:
        return _SPACY_PIPELINES[lang_code]
    
    if lang_code not in SPACY_MODEL_MAP:
        return None
        
    model_name = SPACY_MODEL_MAP[lang_code]
    try:
        import spacy
        try:
            nlp = spacy.load(model_name)
        except OSError as e:
            print(f"SpaCy model '{model_name}' not found or incompatible locally: {e}. Skipping dynamic download to prevent thread freeze.")
            return None
        _SPACY_PIPELINES[lang_code] = nlp
        return nlp
    except Exception as e:
        print(f"SpaCy load failed: {e}")
        return None

def tag_text_with_spacy(text, lang_code):
    """
    Process text with SpaCy.
    """
    try:
        nlp = get_spacy_pipeline(lang_code)
        if not nlp:
            return None, "SpaCy model mapping not available or download failed"
            
        # Handle large text inputs safely
        if len(text) > nlp.max_length:
            nlp.max_length = len(text) + 1000
            
        doc = nlp(text)
        results = []
        sent_id = 0
        for sent in doc.sents:
            sent_id += 1
            for token in sent:
                if token.is_space:
                    continue
                results.append({
                    'token': token.text,
                    'pos': token.pos_,  # Returns Universal Part of Speech tags (UPOS)
                    'lemma': token.lemma_ if token.lemma_ else token.text,
                    'sent_id': sent_id,
                    'ent_type': token.ent_type_ if token.ent_type_ else ""
                })
        return results, None
    except Exception as e:
        return None, str(e)

def split_sentences_custom(text):
    r"""
    Splits text into sentences using regex, preserving closing quotes/parentheses.
    """
    import re
    # Matches everything up to punctuation, followed by optional closing quotes/parens, 
    # and then space or end of string. If no punctuation is found, matches the rest (.+)
    pattern = r'.*?[\.\?\!](?:[\s]*[\'\"’”\)\]]+)*(?:\s+|$)|.+'
    sentences = re.findall(pattern, text, flags=re.DOTALL)
    
    return [s.strip() for s in sentences if s.strip()]

def get_stanza_pipeline(lang_code):
    """
    Get or create a Stanza pipeline for the specified language.
    Attempts local loading first to remain completely offline, falling back to download if missing.
    """
    global _STANZA_PIPELINES
    
    if lang_code in _STANZA_PIPELINES:
        return _STANZA_PIPELINES[lang_code]
    
    # 1. Attempt to load offline directly
    try:
        print(f"Initializing Stanza pipeline offline for '{lang_code}'...")
        try:
            nlp = stanza.Pipeline(lang=lang_code, processors='tokenize,mwt,pos,lemma', download_method=None)
        except Exception:
            nlp = stanza.Pipeline(lang=lang_code, processors='tokenize,pos,lemma', download_method=None)
        _STANZA_PIPELINES[lang_code] = nlp
        return nlp
    except Exception as local_err:
        print(f"Local Stanza load failed for '{lang_code}' ({local_err}). Attempting download...")
        
    # 2. Fallback to download if local load fails
    print(f"Skipping dynamic Stanza download on cloud for '{lang_code}'.")
    return None

import subprocess
import tempfile
import os
import platform

# ── MeCab (Japanese) ──────────────────────────────────────────────────────────
_MECAB_TAGGER = None

def _get_mecab():
    """Lazy-load MeCab tagger, returns None if not installed."""
    global _MECAB_TAGGER
    if _MECAB_TAGGER is not None:
        return _MECAB_TAGGER
    try:
        import MeCab
        import unidic_lite
        dicdir = unidic_lite.DICDIR
        _MECAB_TAGGER = MeCab.Tagger(f'-d {dicdir}')
    except Exception:
        try:
            import MeCab
            _MECAB_TAGGER = MeCab.Tagger('')
        except Exception:
            _MECAB_TAGGER = None
    return _MECAB_TAGGER

def tag_text_with_mecab(text):
    """
    Tokenize and POS-tag Japanese text with MeCab + UniDic.
    Returns (list of dicts, error_msg).
    Each dict: {token, pos, lemma, sent_id, ent_type}
    """
    tagger = _get_mecab()
    if tagger is None:
        return None, "MeCab not available"
    try:
        import re
        parsed = tagger.parse(text)
        results = []
        sent_id = 1
        sentence_enders = {'。', '！', '？', '!', '?', '.'}
        for line in parsed.split('\n'):
            if line == 'EOS' or not line:
                continue
            if '\t' in line:
                surface, features = line.split('\t', 1)
            elif ',' in line and line.count('\t') == 0:
                # fallback: mecab may use different delimiter
                continue
            else:
                continue
            if not surface or surface in ('EOS',):
                continue
            parts = features.split(',')
            pos = parts[0] if parts else 'TAG'
            # UniDic lemma is at index 10, IPA dict at index 6
            lemma = parts[10] if len(parts) > 10 and parts[10] not in ('*', '') \
                    else parts[6] if len(parts) > 6 and parts[6] not in ('*', '') \
                    else surface
            results.append({
                'token': surface,
                'pos': pos,
                'lemma': lemma,
                'sent_id': sent_id,
                'ent_type': ''
            })
            if surface in sentence_enders:
                sent_id += 1
        return results, None
    except Exception as e:
        return None, str(e)

def tokenize_japanese_with_mecab(text):
    """
    Tokenize Japanese text into sentences of tokens using MeCab.
    Returns list of list of str, or None on failure.
    """
    results, err = tag_text_with_mecab(text)
    if results is None:
        return None
    sentences = {}
    for r in results:
        sid = r['sent_id']
        sentences.setdefault(sid, []).append(r['token'])
    return list(sentences.values()) if sentences else None

# Map stanza lang codes to treetagger parameter files
TREETAGGER_LANG_MAP = {
    'id': 'indonesian/indonesian_v311225.par',
    'mg': 'malagasy/malagasy.par',
    'en': 'english/english.par',
    'zh': 'chinese/zh.par',
    'ko': 'korean/korean.par',
    'es': 'spanish/spanish.par',
    'de': 'german/german.par',
    'fr': 'french/french.par',
    'pt': 'portuguese/portuguese.par',
    'it': 'italian/italian.par',
    'nl': 'dutch/dutch.par',
    'el': 'greek/greek.par',
    'ru': 'russian/russian.par',
    'pl': 'polish/polish.par',
    'uk': 'ukrainian/ukrainian.par',
    'ro': 'romanian/romanian.par',
    'sv': 'swedish/swedish.par',
    'da': 'danish/danish.par',
    'nb': 'norwegian/norwegian.par',
    'fi': 'finnish/finnish.par',
    'ca': 'catalan/catalan.par',
    'sl': 'slovenian/slovenian.par',
    'sq': 'albanian/albanian.par',
    'be': 'belarusian/belarusian.par',
    'bg': 'bulgarian/bulgarian.par',
    'cs': 'czech/czech.par',
    'et': 'estonian/estonian.par',
    'gl': 'galician/galician.par',
    'hu': 'hungarian/hungarian.par',
    'la': 'latin/latin.par',
    'mn': 'mongolian/mongolian.par',
    'fa': 'persian/persian.par',
    'sk': 'slovak/slovak.par',
    'sw': 'swahili/swahili.par'
}

_TT_TEXT_CACHE = {}

def tag_text_with_treetagger(text, lang_code):
    """
    Process text with TreeTagger. Detects OS and uses correct binary.
    Returns (list of dicts, error_msg)
    """
    if lang_code not in TREETAGGER_LANG_MAP:
        return None, f"Language '{lang_code}' not supported by local TreeTagger."

    cache_key = (text.strip(), lang_code)
    if cache_key in _TT_TEXT_CACHE:
        return _TT_TEXT_CACHE[cache_key], None

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tt_dir = os.path.join(base_dir, 'treetagger')
    
    # OS Detection
    system = platform.system().lower()
    if system == 'windows':
        executable = os.path.join(tt_dir, '_Windows', 'tree-tagger.exe')
    else:
        executable = os.path.join(tt_dir, '_Linux', 'tree-tagger')
        
    param_file = os.path.join(tt_dir, 'language_model', TREETAGGER_LANG_MAP[lang_code].replace('/', os.sep))
    
    # Look for any .par file in the directory if the specific one is not found
    if not os.path.exists(param_file):
        lang_folder = os.path.join(tt_dir, 'language_model', TREETAGGER_LANG_MAP[lang_code].split('/')[0])
        if os.path.exists(lang_folder):
            for f in os.listdir(lang_folder):
                if f.endswith('.par'):
                    param_file = os.path.join(lang_folder, f)
                    break
    
    if not os.path.exists(executable):
        return None, f"TreeTagger executable not found at {executable}."
    if not os.path.exists(param_file):
        return None, f"TreeTagger parameter file not found at {param_file}."
        
    # Check if executable can be run (Linux might need chmod +x, or wrong arch)
    if system != 'windows' and not os.access(executable, os.X_OK):
        try:
            os.chmod(executable, 0o755)
        except Exception:
            pass

    try:
        # Tokenize text safely first
        sentences_tokens = tokenize_text_only(text, lang_code, fast=True)
        
        # Apply Indonesian Clitic Splitter if ID
        if lang_code == 'id':
            from core.preprocessing.indonesian_tokenizer import tokenize_indonesian_clitics
            processed_sentences = []
            for sent in sentences_tokens:
                processed_sentences.append(tokenize_indonesian_clitics(sent))
            sentences_tokens = processed_sentences
            
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f_in:
            for sent in sentences_tokens:
                for token in sent:
                    if token.strip():
                        f_in.write(token.strip() + "\n")
            input_path = f_in.name
            
        with tempfile.NamedTemporaryFile(mode='r', encoding='utf-8', delete=False) as f_out:
            output_path = f_out.name

        cmd = [executable, param_file, input_path, output_path, '-token', '-lemma']
        
        # Run subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Read output
        with open(output_path, 'r', encoding='utf-8') as f:
            output_text = f.read()
            
        try: os.remove(input_path)
        except Exception: pass
        try: os.remove(output_path)
        except Exception: pass
        
        # Parse vertical output: token \t pos \t lemma
        results = []
        sent_id = 1
        lines = [l.strip() for l in output_text.split('\n') if l.strip()]
        
        for line in lines:
            parts = line.split('\t')
            if not parts or not parts[0]:
                continue
            token = parts[0]
            pos = parts[1] if len(parts) > 1 else 'TAG'
            lemma = parts[2] if len(parts) > 2 else token
            
            results.append({
                'token': token,
                'pos': pos,
                'lemma': lemma,
                'sent_id': sent_id,
                'ent_type': ""
            })
            
            # Simple sentence splitting heuristic based on punctuation
            if token in ['.', '!', '?']:
                sent_id += 1
                
        _TT_TEXT_CACHE[cache_key] = results
        return results, None
        
    except subprocess.CalledProcessError as e:
        return None, f"TreeTagger execution failed: {e.stderr}"
    except Exception as e:
        return None, f"TreeTagger error: {str(e)}"

def tag_text_with_stanza(text, lang_code, progress_callback=None, base_progress=0.0):
    """
    Process text. Tries TreeTagger first, then SpaCy, then Stanza, followed by Custom HMM and simple fallback.
    Returns a tuple (list of dicts, error_msg)
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pkl_path = os.path.join(base_dir, 'model', f'{lang_code}-hmm.pkl')
    json_path = os.path.join(base_dir, 'model', f'{lang_code}-hmm.json')
    
    global _CUSTOM_TAGGERS
    if '_CUSTOM_TAGGERS' not in globals():
        _CUSTOM_TAGGERS = {}
        
    # 1a. MeCab — used exclusively for Japanese
    if lang_code == 'ja':
        if progress_callback: progress_callback(base_progress, "Tagging Japanese with MeCab...")
        mecab_results, mecab_err = tag_text_with_mecab(text)
        if mecab_results is not None:
            if progress_callback: progress_callback(base_progress, "Tagged with MeCab successfully!")
            return mecab_results, None
        if progress_callback: progress_callback(base_progress, f"MeCab failed: {mecab_err}. Trying SpaCy...")
        print(f"MeCab Warning: {mecab_err}", flush=True)

    # 1. Try TreeTagger (Prioritized first for all supported languages)
    if progress_callback: progress_callback(base_progress, f"Tagging with TreeTagger...")
    tt_results, tt_err = tag_text_with_treetagger(text, lang_code)
    if tt_err:
        if progress_callback: progress_callback(base_progress, f"Tagging with TreeTagger failed.")
        print(f"TreeTagger Warning: {tt_err}", flush=True)
    if tt_results is not None:
        if progress_callback: progress_callback(base_progress, f"Tagged with TreeTagger successfully!")
        return tt_results, None

    # 2. Try SpaCy (Fallback if TreeTagger not available)
    if progress_callback: progress_callback(base_progress, f"Tagging with SpaCy...")
    spacy_results, spacy_err = tag_text_with_spacy(text, lang_code)
    if spacy_err:
        if progress_callback: progress_callback(base_progress, f"Tagging with SpaCy failed.")
        print(f"SpaCy Warning: {spacy_err}", flush=True)
    if spacy_results is not None:
        if progress_callback: progress_callback(base_progress, f"Tagged with SpaCy successfully!")
        return spacy_results, f"treetagger fail, switching to SpaCy. {tt_err}" if tt_err else None

    # 3. Try Stanza if TreeTagger and SpaCy failed or weren't found
    if progress_callback: progress_callback(base_progress, f"Tagging with Stanza...")
    try:
        print("Falling back to Stanza...", flush=True)
        nlp = get_stanza_pipeline(lang_code)
        if nlp:
            # Use Stanza's native sentence splitting for better results
            doc = nlp(text)
            
            results = []
            for sent_id, stanza_sent in enumerate(doc.sentences, 1):
                for word in stanza_sent.words:
                    results.append({
                        'token': word.text,
                        'pos': word.upos, 
                        'lemma': word.lemma if word.lemma else word.text,
                        'sent_id': sent_id,
                        'ent_type': ""
                    })
            return results, f"treetagger/spacy fail, switching to Stanza. {tt_err}" if tt_err else None
    except Exception as e:
        print(f"Stanza error for {lang_code}: {str(e)}")

    # 4. Try Custom HMM Tagger fallback if TreeTagger/SpaCy/Stanza not available
    custom_tagger = _CUSTOM_TAGGERS.get(lang_code)
    if custom_tagger is None:
        if os.path.exists(json_path):
            print(f"Found custom {lang_code}-hmm.json model. Using it as fast fallback...")
            try:
                import json
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                from core.preprocessing.custom_tagger import CustomDataDrivenTagger
                custom_tagger = CustomDataDrivenTagger.from_json(data)
                _CUSTOM_TAGGERS[lang_code] = custom_tagger
            except Exception as e:
                print(f"Failed to load custom {lang_code}-hmm.json: {e}")
                
        if custom_tagger is None and os.path.exists(pkl_path):
            print(f"Found custom {lang_code}-hmm.pkl model. Using it as fast fallback...")
            try:
                import pickle
                with open(pkl_path, 'rb') as f:
                    custom_tagger = pickle.load(f)
                _CUSTOM_TAGGERS[lang_code] = custom_tagger
            except Exception as e:
                print(f"Failed to load custom {lang_code}-hmm.pkl: {e}")
                
    if custom_tagger is not None:
        if progress_callback: progress_callback(base_progress, f"Tagging with Custom Tagger...")
        try:
            # Use fast regex tokenizer to prevent invoking Stanza/SpaCy just for tokenization
            sentences_tokens = tokenize_text_only(text, lang_code, fast=True)
            results = []
            sent_id = 0
            for sent_tokens in sentences_tokens:
                sent_id += 1
                tagged_tokens = custom_tagger.tag(sent_tokens)
                for i, token_str in enumerate(sent_tokens):
                    item = tagged_tokens[i]
                    results.append({
                        'token': token_str,
                        'pos': item['pos'],
                        'lemma': item['lemma'],
                        'sent_id': sent_id,
                        'ent_type': ""
                    })
            if progress_callback: progress_callback(base_progress, f"Tagged with Custom Tagger successfully!")
            return results, f"treetagger/spacy/stanza fail, switching to Custom Tagger. {tt_err}" if tt_err else None
        except Exception as e:
            import traceback
            traceback.print_exc()
            if progress_callback: progress_callback(base_progress, f"Tagging with Custom Tagger failed.")
            print(f"Failed to tag with custom model: {e}", flush=True)

    # 5. Fallback to Simple Regex Tokenizer
    fallback_res, fallback_err = tag_text_simple_fallback(text)
    return fallback_res, f"treetagger/spacy/stanza fail, switching to fallback. {tt_err}" if tt_err else fallback_err

def tag_text_simple_fallback(text):
    """
    Fallback tagging. Returns (results, error)
    """
    import re
    
    sentences = split_sentences_custom(text)
    
    results = []
    sent_id = 0
    
    has_cjk = bool(re.search(r'[\u4e00-\u9fff]', text))
    use_jieba = False
    
    if has_cjk:
        try:
            import jieba
            use_jieba = True
        except ImportError:
            pass
    
    for sent_text in sentences:
        sent_id += 1
        
        if use_jieba:
            import jieba
            tokens = list(jieba.cut(sent_text))
            tokens = [t.strip() for t in tokens if t.strip()]
        else:
            # Simple tokenization: split but preserve punctuation
            cleaned_text = re.sub(r'([^\w\s])', r' \1 ', sent_text)
            
            # Insert space around CJK characters for fallback tokenization if jieba is missing
            if has_cjk:
                cleaned_text = re.sub(r'([\u4e00-\u9fff])', r' \1 ', cleaned_text)
                
            tokens = [t.strip() for t in cleaned_text.split() if t.strip()]
        
        for token in tokens:
            results.append({
                'token': token,
                'pos': 'TAG', 
                'lemma': token,
                'sent_id': sent_id,
                'ent_type': ""
            })
            
    return results, None

def get_stanza_tokenizer_only(lang_code):
    global _STANZA_PIPELINES
    key = f"{lang_code}_tokenize_only"
    if key in _STANZA_PIPELINES:
        return _STANZA_PIPELINES[key]
        
    try:
        nlp = stanza.Pipeline(lang=lang_code, processors='tokenize', download_method=None)
    except Exception:
        print(f"Skipping dynamic Stanza tokenizer download on cloud for '{lang_code}'.")
        return None
    _STANZA_PIPELINES[key] = nlp
    return nlp

def tokenize_text_with_spacy(text, lang_code):
    nlp = get_spacy_pipeline(lang_code)
    if not nlp:
        return None
    try:
        # We must add sentencizer if parser is disabled, otherwise doc.sents fails
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        
        # Handle large text inputs safely
        if len(text) > nlp.max_length:
            nlp.max_length = len(text) + 1000
            
        # Disable all parsing, tagging, lemmatization to run tokenizer only
        with nlp.select_pipes(disable=['tagger', 'parser', 'ner', 'lemmatizer']):
            doc = nlp(text)
            sentences = []
            for sent in doc.sents:
                sent_tokens = [token.text for token in sent if not token.is_space]
                if sent_tokens:
                    sentences.append(sent_tokens)
            return sentences
    except Exception:
        return None

def tokenize_text_with_stanza(text, lang_code):
    nlp = get_stanza_tokenizer_only(lang_code)
    if not nlp:
        return None
    try:
        doc = nlp(text)
        sentences = []
        for stanza_sent in doc.sentences:
            sent_tokens = [word.text for word in stanza_sent.words]
            if sent_tokens:
                sentences.append(sent_tokens)
        return sentences
    except Exception:
        return None

def tokenize_text_only(text, lang_code=None, fast=False):
    """
    Splits text into sentences and tokens.
    If fast=True or lang_code=="OTHER", uses high-speed regex tokenization (<0.001s).
    Returns: list of list of str (list of sentences, where each sentence is a list of tokens)
    """
    if lang_code in ('zh', 'CH'):
        from core.preprocessing.chinese_tokenizer import tokenize_chinese_text
        zh_tokens = tokenize_chinese_text(text)
        if zh_tokens is not None:
            return zh_tokens

    if lang_code == 'ja':
        ja_tokens = tokenize_japanese_with_mecab(text)
        if ja_tokens is not None:
            return ja_tokens

    if fast or not lang_code or lang_code == "OTHER":
        import re
        sentences = split_sentences_custom(text)
        results = []
        for sent_text in sentences:
            cleaned_text = re.sub(r'([^\w\s])', r' \1 ', sent_text)
            tokens = [t.strip() for t in cleaned_text.split() if t.strip()]
            if tokens:
                results.append(tokens)
        return results
        
    # 1. Try SpaCy
    spacy_res = tokenize_text_with_spacy(text, lang_code)
    if spacy_res is not None:
        return spacy_res
        
    # 2. Try Stanza
    stanza_res = tokenize_text_with_stanza(text, lang_code)
    if stanza_res is not None:
        return stanza_res
        
    # 3. Fallback
    import re
    sentences = split_sentences_custom(text)
    results = []
    for sent_text in sentences:
        cleaned_text = re.sub(r'([^\w\s])', r' \1 ', sent_text)
        tokens = [t.strip() for t in cleaned_text.split() if t.strip()]
        if tokens:
            results.append(tokens)
    return results

