import stanza
import logging

# Configure logging
logging.getLogger('stanza').setLevel(logging.WARNING)

# Cache for Stanza pipelines to avoid reloading
_STANZA_PIPELINES = {}

def split_sentences_custom(text):
    r"""
    Splits text into sentences using the regex: /.*?[\.\?\!]\s+/s
    Adapted for Python.
    """
    import re
    # Python equivalent of /.*?[\.\?\!]\s+/s
    # re.DOTALL (S) makes . match newlines.
    # We use findall to get all matches. 
    # The regex ensures it ends with punctuation followed by space or end of string.
    pattern = r'.*?[\.\?\!](?:\s+|$)'
    sentences = re.findall(pattern, text, flags=re.DOTALL)
    
    # If no matches (e.g. no punctuation), return the whole text as one sentence
    if not sentences:
        return [text.strip()] if text.strip() else []
    
    return [s.strip() for s in sentences if s.strip()]

def get_stanza_pipeline(lang_code):
    """
    Get or create a Stanza pipeline for the specified language.
    Downloads the model if not present.
    """
    global _STANZA_PIPELINES
    
    if lang_code in _STANZA_PIPELINES:
        return _STANZA_PIPELINES[lang_code]
    
    try:
        print(f"Initializing Stanza pipeline for '{lang_code}'...")
        stanza.download(lang_code, verbose=True)
        nlp = stanza.Pipeline(lang=lang_code, processors='tokenize,mwt,pos,lemma')
        _STANZA_PIPELINES[lang_code] = nlp
        return nlp
    except Exception as e:
        error_msg = f"Error initializing Stanza for {lang_code}: {e}"
        print(error_msg)
        raise RuntimeError(error_msg)

def tag_text_with_stanza(text, lang_code):
    """
    Process text with Stanza.
    Returns a tuple (list of dicts, error_msg)
    """
    try:
        nlp = get_stanza_pipeline(lang_code)
        if not nlp:
            return tag_text_simple_fallback(text)
            
        # Optional: Pre-split sentences if needed, but Stanza does its own.
        # However, user requested "sentence is split before tokenisation when auto tag is selected"
        # If we pre-split, we can pass sentences one by one or join with double newlines.
        custom_sentences = split_sentences_custom(text)
        
        results = []
        sent_id = 0
        
        for sent_text in custom_sentences:
            doc = nlp(sent_text)
            for stanza_sent in doc.sentences:
                sent_id += 1
                for word in stanza_sent.words:
                    results.append({
                        'token': word.text,
                        'pos': word.upos, 
                        'lemma': word.lemma if word.lemma else word.text,
                        'sent_id': sent_id
                    })
        return results, None
    except Exception as e:
        results, _ = tag_text_simple_fallback(text)
        return results, str(e)

def tag_text_simple_fallback(text):
    """
    Fallback tagging. Returns (results, error)
    """
    import re
    
    sentences = split_sentences_custom(text)
    
    results = []
    sent_id = 0
    
    for sent_text in sentences:
        sent_id += 1
        # Simple tokenization: split but preserve punctuation
        cleaned_text = re.sub(r'([^\w\s])', r' \1 ', sent_text)
        tokens = [t.strip() for t in cleaned_text.split() if t.strip()]
        
        for token in tokens:
            results.append({
                'token': token,
                'pos': 'TAG', 
                'lemma': token,
                'sent_id': sent_id
            })
            
    return results, None
