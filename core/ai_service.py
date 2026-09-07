import requests
import time
import json
import sys

def _resolve_ai_settings(ai_provider=None, gemini_api_key=None, gemini_model=None, ollama_url=None, ollama_model=None, openrouter_api_key=None, openrouter_model=None):
    # Retrieve from Streamlit session state if running in streamlit context
    import sys
    st_state = {}
    if 'streamlit' in sys.modules:
        import streamlit as st
        try:
            st_state = st.session_state
        except Exception:
            pass
            
    resolved_provider = ai_provider or st_state.get('ai_provider') or "Ollama"
    resolved_gemini_key = gemini_api_key or st_state.get('gemini_api_key') or ""
    resolved_gemini_model = gemini_model or st_state.get('gemini_model') or "gemini-2.5-flash"
    resolved_openrouter_key = openrouter_api_key or st_state.get('openrouter_api_key') or ""
    resolved_openrouter_model = openrouter_model or st_state.get('openrouter_model') or "google/gemini-2.5-flash"
    resolved_ollama_url = ollama_url or st_state.get('ollama_url') or "http://127.0.0.1:11434/api/generate"
    resolved_ollama_model = ollama_model or st_state.get('ai_model') or "phi3:latest"
    
    return {
        "ai_provider": resolved_provider,
        "gemini_api_key": resolved_gemini_key,
        "gemini_model": resolved_gemini_model,
        "openrouter_api_key": resolved_openrouter_key,
        "openrouter_model": resolved_openrouter_model,
        "ollama_url": resolved_ollama_url,
        "ollama_model": resolved_ollama_model
    }

def test_openrouter_connection(api_key, model=None):
    """
    Tests the connection to OpenRouter API by sending a simple prompt.
    Returns: (bool, message)
    """
    if not api_key:
        return False, "API Key is required."
    
    settings = _resolve_ai_settings(openrouter_api_key=api_key, openrouter_model=model)
    openrouter_model = settings["openrouter_model"]
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/prihantoro-corpus/cortex",
        "X-Title": "CORTEX",
        "Content-Type": "application/json"
    }
    payload = {
        "model": openrouter_model,
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        if response.status_code == 200:
            res_json = response.json()
            if 'error' in res_json:
                return False, f"OpenRouter Error: {res_json['error'].get('message', 'Unknown error')}"
            if 'choices' in res_json and len(res_json['choices']) > 0:
                return True, f"Successfully connected! OpenRouter model '{openrouter_model}' is ready."
            return False, "Unexpected response format from OpenRouter."
        else:
            try:
                res_json = response.json()
                if 'error' in res_json:
                    return False, f"API Error (Status {response.status_code}): {res_json['error'].get('message')}"
            except Exception:
                pass
            return False, f"Connection failed with status code {response.status_code}."
    except requests.exceptions.RequestException as e:
        return False, f"Request Failed: {e}"
    except Exception as e:
        return False, f"Unexpected Error: {e}"

def test_gemini_connection(api_key, model=None):
    """
    Tests the connection to Gemini API by sending a simple prompt.
    Returns: (bool, message)
    """
    if not api_key:
        return False, "API Key is required."
    
    settings = _resolve_ai_settings(gemini_api_key=api_key, gemini_model=model)
    gemini_model = settings["gemini_model"]
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": [{"text": "Hello"}]
        }]
    }
    
    try:
        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=15)
        if response.status_code == 200:
            res_json = response.json()
            if 'error' in res_json:
                return False, f"Gemini Error: {res_json['error'].get('message', 'Unknown error')}"
            if 'candidates' in res_json and len(res_json['candidates']) > 0:
                return True, f"Successfully connected! Model '{gemini_model}' is ready."
            return False, "Unexpected response format from Gemini."
        else:
            try:
                res_json = response.json()
                if 'error' in res_json:
                    return False, f"API Error (Status {response.status_code}): {res_json['error'].get('message')}"
            except Exception:
                pass
            return False, f"Connection failed with status code {response.status_code}."
    except requests.exceptions.RequestException as e:
        return False, f"Request Failed: {e}"
    except Exception as e:
        return False, f"Unexpected Error: {e}"

def interpret_results_openrouter(target_word, analysis_type, data_description, data, api_key, model=None):
    """
    Integrates with OpenRouter API via REST.
    """
    if not api_key:
        return None, "OpenRouter API Key missing."

    settings = _resolve_ai_settings(openrouter_api_key=api_key, openrouter_model=model)
    openrouter_model = settings["openrouter_model"]

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/prihantoro-corpus/cortex",
        "X-Title": "CORTEX",
        "Content-Type": "application/json"
    }
    
    # Prepare Data
    if hasattr(data, 'to_string'):
        data_text = data.to_string(index=False)
    else:
        data_text = str(data)

    prompt = f"""
    Role: Corpus Linguist.
    Task: Analyze the {analysis_type} data for "{target_word}".
    Context: {data_description}
    
    Data:
    {data_text}
    
    Instructions:
    1. Base your analysis STRICTLY on the actual provided data results.
    2. CRITICAL RULES FOR READING NUMERICAL & FREQUENCY DATA:
       - The actual total number of times a term/word appears in the corpus is given ONLY by "Absolute Frequency", "Total Occurrences", "Frequency", or "Counts" (e.g., 1,246 occurrences).
       - "Zipf Band" (e.g. 1, 2, 3, 4, 5, 6) is a LOGARITHMIC FREQUENCY CLASSIFICATION BAND (Zipf's Law scale from 1=rare to 6=extremely frequent), NOT an occurrence count! NEVER say a term appears "X times" using the Zipf Band number.
       - "Zipf Score" (e.g. 6.32) is a logarithmic score metric, NOT an occurrence count.
       - "Relative Frequency" / "PMW" is occurrences per million words, NOT the raw count.
       - "Sample Lines Shown" or KWIC lines rendered in a preview are a small subset sample, NOT the total occurrence count in the corpus.
       - NEVER confuse Zipf Band, Zipf Score, Relative Frequency, or Sample Line counts with Absolute Frequency / Total Occurrences.
       - If Absolute Frequency is 1246 and Zipf Band is 6, state: "The term appears 1,246 times in the corpus (Zipf Band 6, Relative Frequency ... PMW)."
       - Cite numbers EXACTLY as given under their specific column headers.
    3. Do NOT hallucinate, extrapolate, speculate, or invent any words, frequencies, metrics, categories, or patterns not explicitly present in the provided data.
    4. Do NOT treat this as a hypothetical example or add disclaimers about fictional data. The data is real empirical results from the corpus.
    5. Rely ONLY on the actual results supplied above. If details or specific words are missing from the data, do NOT guess or infer them.
    6. Output a concise, scholarly markdown summary of empirical linguistic patterns and coverage grounded strictly in the provided data.
    """

    payload = {
        "model": openrouter_model,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        res_json = response.json()
        
        if 'error' in res_json:
            return None, f"OpenRouter API Error: {res_json['error'].get('message', 'Unknown error')}"
            
        try:
            text = res_json['choices'][0]['message']['content']
            return text, None
        except (KeyError, IndexError):
            return None, "OpenRouter returned an unexpected response format."
            
    except Exception as e:
        return None, f"OpenRouter Connection Error: {e}"

def interpret_results_llm(target_word, analysis_type, data_description, data, 
                          ai_provider=None, gemini_api_key=None,
                          ollama_url=None, ollama_model=None, gemini_model=None,
                          openrouter_api_key=None, openrouter_model=None):
    """
    Router function to interpret results using Ollama, Google Gemini, or OpenRouter.
    """
    settings = _resolve_ai_settings(ai_provider, gemini_api_key, gemini_model, ollama_url, ollama_model, openrouter_api_key, openrouter_model)
    
    if settings["ai_provider"] == "OpenRouter" and settings["openrouter_api_key"]:
        return interpret_results_openrouter(target_word, analysis_type, data_description, data, settings["openrouter_api_key"], settings["openrouter_model"])
    
    if settings["ai_provider"] == "Gemini" and settings["gemini_api_key"]:
        return interpret_results_gemini(target_word, analysis_type, data_description, data, settings["gemini_api_key"], settings["gemini_model"])
    
    # 1. Prepare Content for AI
    if hasattr(data, 'to_string'):
        data_text = data.to_string(index=False)
    else:
        data_text = str(data)
        
    prompt = f"""
    Role: Corpus Linguist.
    Task: Analyze the {analysis_type} data for "{target_word}".
    Context: {data_description}
    
    Data:
    {data_text}
    
    Instructions:
    1. Base your analysis STRICTLY on the actual provided data results.
    2. CRITICAL RULES FOR READING NUMERICAL & FREQUENCY DATA:
       - The actual total number of times a term/word appears in the corpus is given ONLY by "Absolute Frequency", "Total Occurrences", "Frequency", or "Counts" (e.g., 1,246 occurrences).
       - "Zipf Band" (e.g. 1, 2, 3, 4, 5, 6) is a LOGARITHMIC FREQUENCY CLASSIFICATION BAND (Zipf's Law scale from 1=rare to 6=extremely frequent), NOT an occurrence count! NEVER say a term appears "X times" using the Zipf Band number.
       - "Zipf Score" (e.g. 6.32) is a logarithmic score metric, NOT an occurrence count.
       - "Relative Frequency" / "PMW" is occurrences per million words, NOT the raw count.
       - "Sample Lines Shown" or KWIC lines rendered in a preview are a small subset sample, NOT the total occurrence count in the corpus.
       - NEVER confuse Zipf Band, Zipf Score, Relative Frequency, or Sample Line counts with Absolute Frequency / Total Occurrences.
       - If Absolute Frequency is 1246 and Zipf Band is 6, state: "The term appears 1,246 times in the corpus (Zipf Band 6, Relative Frequency ... PMW)."
       - Cite numbers EXACTLY as given under their specific column headers.
    3. Do NOT hallucinate, extrapolate, speculate, or invent any words, frequencies, metrics, categories, or patterns not explicitly present in the provided data.
    4. Do NOT treat this as a hypothetical example or add disclaimers about fictional data. The data is real empirical results from the corpus.
    5. Rely ONLY on the actual results supplied above. If details or specific words are missing from the data, do NOT guess or infer them.
    6. Output a concise, scholarly markdown summary of empirical linguistic patterns and coverage grounded strictly in the provided data.
    """
    
    try:
        payload = {
            "model": settings["ollama_model"],
            "prompt": prompt,
            "stream": False
        }
        
        headers = {
            "User-Agent": "Mozilla/5.0 (CORTEX_AGENT)",
            "content-type": "application/json",
            "ngrok-skip-browser-warning": "true"
        }
        
        # Retry Logic
        max_retries = 3
        last_error = None
        
        print(f"[INFO] Starting Ollama Interpretation for '{target_word}' using {settings['ollama_model']}...")

        for attempt in range(max_retries):
            try:
                print(f"   PLEASE WAIT... Attempt {attempt + 1}/{max_retries}...")
                
                # Timeout: (connect, read)
                # Fail fast if server down (5s), but give model time to think (90s)
                response = requests.post(settings["ollama_url"], json=payload, headers=headers, timeout=(5, 180))
                response.raise_for_status()
                
                res_json = response.json()
                
                # Check for various response keys (Ollama API changes sometimes)
                full_response = res_json.get('response') or res_json.get('content') or str(res_json)
                
                if not full_response:
                     full_response = "Error: Empty response from model."

                print("   [SUCCESS] AI response received.")
                return full_response, None
                
            except requests.exceptions.Timeout:
                print(f"   [WARN] Timeout on attempt {attempt + 1}.")
                last_error = "[TIMEOUT] Ollama timed out."
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt) # 1s, 2s...
                    continue
                else:
                    return None, last_error
                    
            except requests.exceptions.ConnectionError:
                print("   [ERROR] Connection Error: Ensure Ollama is running.")
                return None, "[ERROR] Cannot connect to Ollama. Ensure Ollama is running at the specified URL."
            except Exception as e:
                print(f"   [ERROR] Error on attempt {attempt + 1}: {e}")
                last_error = f"Ollama Error: {e}"
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
        
        return None, last_error

    except Exception as e:
        return None, f"Unexpected Error: {e}"

def interpret_results_gemini(target_word, analysis_type, data_description, data, api_key, model=None):
    """
    Integrates with Google Gemini API via REST.
    """
    if not api_key:
        return None, "Gemini API Key missing."

    settings = _resolve_ai_settings(gemini_api_key=api_key, gemini_model=model)
    gemini_model = settings["gemini_model"]

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={api_key}"
    
    # Prepare Data
    if hasattr(data, 'to_string'):
        data_text = data.to_string(index=False)
    else:
        data_text = str(data)

    prompt = f"""
    Role: Corpus Linguist.
    Task: Analyze the {analysis_type} data for "{target_word}".
    Context: {data_description}
    
    Data:
    {data_text}
    
    Instructions:
    1. Base your analysis STRICTLY on the actual provided data results.
    2. CRITICAL RULES FOR READING NUMERICAL & FREQUENCY DATA:
       - The actual total number of times a term/word appears in the corpus is given ONLY by "Absolute Frequency", "Total Occurrences", "Frequency", or "Counts" (e.g., 1,246 occurrences).
       - "Zipf Band" (e.g. 1, 2, 3, 4, 5, 6) is a LOGARITHMIC FREQUENCY CLASSIFICATION BAND (Zipf's Law scale from 1=rare to 6=extremely frequent), NOT an occurrence count! NEVER say a term appears "X times" using the Zipf Band number.
       - "Zipf Score" (e.g. 6.32) is a logarithmic score metric, NOT an occurrence count.
       - "Relative Frequency" / "PMW" is occurrences per million words, NOT the raw count.
       - "Sample Lines Shown" or KWIC lines rendered in a preview are a small subset sample, NOT the total occurrence count in the corpus.
       - NEVER confuse Zipf Band, Zipf Score, Relative Frequency, or Sample Line counts with Absolute Frequency / Total Occurrences.
       - If Absolute Frequency is 1246 and Zipf Band is 6, state: "The term appears 1,246 times in the corpus (Zipf Band 6, Relative Frequency ... PMW)."
       - Cite numbers EXACTLY as given under their specific column headers.
    3. Do NOT hallucinate, extrapolate, speculate, or invent any words, frequencies, metrics, categories, or patterns not explicitly present in the provided data.
    4. Do NOT treat this as a hypothetical example or add disclaimers about fictional data. The data is real empirical results from the corpus.
    5. Rely ONLY on the actual results supplied above. If details or specific words are missing from the data, do NOT guess or infer them.
    6. Output a concise, scholarly markdown summary of empirical linguistic patterns and coverage grounded strictly in the provided data.
    """

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=60)
        res_json = response.json()
        
        if 'error' in res_json:
            return None, f"Gemini API Error: {res_json['error'].get('message', 'Unknown error')}"
            
        # Parse candidate
        try:
            text = res_json['candidates'][0]['content']['parts'][0]['text']
            return text, None
        except (KeyError, IndexError):
            return None, "Gemini returned an unexpected response format."
            
    except Exception as e:
        return None, f"Gemini Connection Error: {e}"

def get_ai_response(prompt, ai_provider=None, ollama_model=None, api_key=None, ollama_url=None, gemini_model=None, format=None):
    """
    Generic function to get a response from an AI provider for a given prompt.
    """
    settings = _resolve_ai_settings(ai_provider, api_key, gemini_model, ollama_url, ollama_model)
    
    if settings["ai_provider"] == "Gemini" and settings["gemini_api_key"]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings['gemini_model']}:generateContent?key={settings['gemini_api_key']}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(url, json=payload, timeout=60)
            res_json = response.json()
            if 'error' in res_json: return None, f"Gemini Error: {res_json['error'].get('message')}"
            return res_json['candidates'][0]['content']['parts'][0]['text'], None
        except Exception as e: return None, f"Gemini Error: {e}"
    else:
        # Ollama
        try:
            payload = {"model": settings["ollama_model"], "prompt": prompt, "stream": False}
            if format:
                payload["format"] = format
            response = requests.post(settings["ollama_url"], json=payload, timeout=180)
            res_json = response.json()
            return res_json.get('response') or res_json.get('content'), None
        except Exception as e: return None, f"Ollama Error: {e}"

def test_ollama_connection(ollama_url):
    """
    Tests if the Ollama server is reachable at the given URL.
    Returns: (bool, message)
    """
    # Try to reach the base URL (without /api/generate)
    base_url = ollama_url.split('/api/')[0]
    test_url = f"{base_url}/api/tags"
    
    try:
        response = requests.get(test_url, timeout=5)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            return True, f"Connected! Models: {', '.join(models[:5])}"
        else:
            return False, f"Server reachable but status {response.status_code}."
    except requests.exceptions.RequestException as e:
        return False, f"Connection Failed: {e}"

def get_available_models(ollama_url):
    """
    Fetches the list of available models from Ollama.
    Returns: List of model names (strings).
    """
    base_url = ollama_url.split('/api/')[0]
    enc_models_url = f"{base_url}/api/tags"
    
    try:
        response = requests.get(enc_models_url, timeout=2)
        if response.status_code == 200:
            return [m['name'] for m in response.json().get('models', [])]
    except:
        pass
    return []

def chat_with_llm(user_message, context, chat_history=[], 
                  ai_provider=None, gemini_api_key=None,
                  ollama_url=None, ollama_model=None,
                  gemini_model=None):
    """
    Sends a chat message to the selected AI provider.
    """
    settings = _resolve_ai_settings(ai_provider, gemini_api_key, gemini_model, ollama_url, ollama_model)
    
    history_text = "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in chat_history])
    
    prompt = f"""
    Context: {context}
    
    Previous Chat:
    {history_text}
    
    User: {user_message}
    AI (Instructions: Answer in natural language only.):"""

    if settings["ai_provider"] == "Gemini" and settings["gemini_api_key"]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings['gemini_model']}:generateContent?key={settings['gemini_api_key']}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(url, json=payload, timeout=60)
            res_json = response.json()
            if 'error' in res_json: return None, f"Gemini Error: {res_json['error'].get('message')}"
            return res_json['candidates'][0]['content']['parts'][0]['text'], None
        except Exception as e: return None, f"Gemini Error: {e}"
    else:
        # Ollama Chat
        try:
            payload = {"model": settings["ollama_model"], "prompt": prompt, "stream": False}
            response = requests.post(settings["ollama_url"], json=payload, timeout=180)
            res_json = response.json()
            return res_json.get('response') or res_json.get('content'), None
        except Exception as e: return None, f"Ollama Error: {e}"

def app_guide_chat(user_query, chat_history=[], api_key=None, gemini_model=None):
    """
    Specialized chat for app usage assistance.
    """
    
    system_context = """
    You are the CORTEX Assistant. CORTEX is a Corpus Query System.
    Core Modules:
    1. Overview: Basic stats (TTR, Tokens, Tags).
    2. Concordance: KWIC (Key Word In Context) search. Supports wildcards (*), POS (_TAG), and Lemma ([lemma]).
    3. Collocation: Relationship between words. Calculates Log-Likelihood and MI.
    4. Patterns: Groups collocates by positional patterns (e.g. <> * #).
    5. N-Gram: Frequency of sequences (2-5 words).
    6. Keywords: Compares two corpora (Target vs Reference).
    7. Dictionary: Comprehensive analysis of a single word including CEFR and IPA.
    8. Distribution: Shows how word frequency is distributed throughout the corpus.
    
    Inputs:
    - Supports XML files and plain text.
    - Uses DuckDB for storage.
    
    Features & How-Tos:
    - Restricting searches to sub-corpora: Enter the desired module (e.g., Concordance, Collocation), make sure you are NOT in simple mode, find the "Restricted Searches" menu under query settings, select the relevant sub-corpora, and perform the search as usual. (Do NOT suggest using the Dictionary module for this).
    """
    
    # Try to load extended knowledge base if available
    import os
    kb_path = os.path.join("doc", "cortex_knowledge_base.txt")
    if os.path.exists(kb_path):
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                kb_text = f.read()
                # Limit to 12000 chars to avoid overwhelming local models
                if len(kb_text) > 12000:
                    kb_text = kb_text[:12000] + "...(truncated)"
                system_context += "\n\n--- EXTENDED CORTEX KNOWLEDGE BASE ---\n" + kb_text
        except Exception:
            pass
            
    system_context += "\n\nInstructions: Help the user use the app based on the knowledge above. Be concise and accurate."
    
    settings = _resolve_ai_settings(gemini_api_key=api_key, gemini_model=gemini_model)
    return chat_with_llm(user_query, system_context, chat_history, ai_provider="Gemini" if settings["gemini_api_key"] else "Ollama", gemini_api_key=settings["gemini_api_key"], gemini_model=settings["gemini_model"])


def generate_pos_mappings_from_definitions(pos_definitions):
    """
    Generates POS mappings for the rule-based parser from database definitions.
    
    Args:
        pos_definitions: dict {tag: definition} e.g., {"NN": "Noun", "VB": "Verb"}
    
    Returns:
        dict: Regex patterns mapped to POS tags, e.g., {r'\bnoun\b': '_NN', r'\bverb\b': '_VB'}
    """
    if not pos_definitions:
        return {}
    
    import re
    mappings = {}
    
    for tag, definition in pos_definitions.items():
        if not definition or not definition.strip():
            continue
            
        # Normalize the definition to lowercase for matching
        def_lower = definition.lower().strip()
        
        # Remove common prefixes/suffixes to get the core term
        # e.g., "Common Noun" -> "noun", "Past Tense Verb" -> "verb"
        core_terms = []
        
        # Split by common separators
        words = re.split(r'[,/\-\s]+', def_lower)
        
        # Extract meaningful linguistic terms
        linguistic_keywords = {
            'noun', 'verb', 'adjective', 'adverb', 'preposition', 'pronoun',
            'determiner', 'conjunction', 'modal', 'particle', 'interjection',
            'article', 'auxiliary', 'numeral', 'quantifier'
        }
        
        for word in words:
            word = word.strip()
            if word in linguistic_keywords:
                core_terms.append(word)
        
        # If we found linguistic terms, create mappings
        if core_terms:
            for term in core_terms:
                # Create both singular and plural patterns
                singular_pattern = rf'\b{re.escape(term)}\b'
                plural_pattern = rf'\b{re.escape(term)}s\b'
                
                mappings[singular_pattern] = f'_{tag}'
                mappings[plural_pattern] = f'_{tag}'
        else:
            # Fallback: use the full definition as-is
            # This handles custom definitions like "kata kerja" (Indonesian for verb)
            pattern = rf'\b{re.escape(def_lower)}\b'
            mappings[pattern] = f'_{tag}'
    
    return mappings

def preprocess_query_with_rules(query):
    import re
    
    # Normalize the query
    normalized = query.lower().strip()
    
    # Helper pattern components
    # Matches words, words with wildcards (*), POS tags (_NN), or Lemmatized tags ([lemma])
    # simplified to avoid the bracket escaping hell: just match non-whitespace generally within quotes or specific chars
    # We use a permissive pattern: something that looks like a token
    token_pat = r"[\w\*\[\]_]+" 
    
    # helper for "find 'X'" or just "X"
    # match optionally quoted token
    # We use non-raw strings construction or careful escaping for the final pattern
    
    def make_pat(connector):
        # returns pattern for: (find...)? TERM1 connector TERM2
        # We manually build the regex string to ensure proper escaping
        return (
            r"(?:find|search|concordance of)?\s*"
            r"['\"]?(" + token_pat + r")['\"]?\s+" + 
            connector + 
            r"\s+(?:a |an |the )?['\"]?(" + token_pat + r")['\"]?"
        )

    # Pattern 0: chained "followed by"
    # "A followed by B followed by C"
    pat0 = (
        r"(?:find|search|concordance of)?\s*"
        r"['\"]?(" + token_pat + r")['\"]?\s+"
        r"followed by\s+(?:a |an |the )?['\"]?(" + token_pat + r")['\"]?\s+"
        r"followed by\s+(?:a |an |the )?['\"]?(" + token_pat + r")['\"]?"
    )
    chained_match = re.search(pat0, normalized, re.IGNORECASE)
    if chained_match:
        return f"{chained_match.group(1)} {chained_match.group(2)} {chained_match.group(3)}"
        
    # Pattern 0b: "followed by ... and ..."
    # "A followed by B and C"
    pat0_and = (
        r"(?:find|search|concordance of)?\s*"
        r"['\"]?(" + token_pat + r")['\"]?\s+"
        r"followed by\s+(?:a |an |the )?['\"]?(" + token_pat + r")['\"]?\s+"
        r"and\s+(?:a |an |the )?['\"]?(" + token_pat + r")['\"]?"
    )
    and_match = re.search(pat0_and, normalized, re.IGNORECASE)
    if and_match:
        return f"{and_match.group(1)} {and_match.group(2)} {and_match.group(3)}"
    
    
    # Pattern 1: "X followed by Y" -> "X Y"
    pat1 = make_pat(r"followed by")
    match = re.search(pat1, normalized, re.IGNORECASE)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    
    # Pattern 2: "X preceded by Y" -> "Y X"
    pat2 = make_pat(r"preceded by")
    match = re.search(pat2, normalized, re.IGNORECASE)
    if match:
        return f"{match.group(2)} {match.group(1)}"
    
    # Pattern 3: "X before Y" -> "X Y"
    pat3 = make_pat(r"before")
    match = re.search(pat3, normalized, re.IGNORECASE)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    
    # Pattern 4: "X after Y" -> "Y X"
    pat4 = make_pat(r"after")
    match = re.search(pat4, normalized, re.IGNORECASE)
    if match:
        return f"{match.group(2)} {match.group(1)}"
    
    # Pattern 5: "Y following X" -> "X Y"
    pat5 = make_pat(r"following")
    match = re.search(pat5, normalized, re.IGNORECASE)
    if match:
        return f"{match.group(2)} {match.group(1)}"
    
    # Pattern 6: "X and Y" -> "X Y" (Simple sequences)
    # This must come after specific "followed by... and..." patterns to avoid partial matches
    pat6 = make_pat(r"and")
    match = re.search(pat6, normalized, re.IGNORECASE)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    
    # Pattern 7: "Y preceding X" -> "X Y"
    pat7 = make_pat(r"preceding")
    match = re.search(pat7, normalized, re.IGNORECASE)
    if match:
        return f"{match.group(2)} {match.group(1)}"
    
    # No pattern matched, return original
    return query

def parse_nl_query_rules_only(user_query, module_selection="concordance", reverse_pos_map=None):
    """
    Pure rule-based natural language parser - NO AI required.
    Fast, deterministic, and works offline.
    
    Handles:
    - Word order: "A followed by B", "A preceded by B", etc.
    - POS mapping: "noun" -> "_NN", "verb" -> "_VB", etc.
    - Wildcard heuristics:
        * "beginning with X" / "starts with X" -> "X*"
        * "ending with X" / "ends with X" -> "*X"
        * "containing X" / "contains X" -> "*X*"
        * "any word" / "anything" -> "*"
    - Command stripping: "find", "search for", "concordance of"
    - Quotes removal
    
    Args:
        user_query: Natural language query string
        module_selection: Target module name
        reverse_pos_map: Optional dict {tag: definition} from database for dynamic mapping
    
    Returns: (dict, error_message)
    """
    import re
    
    # Step 0: Remove quotes first
    processed = user_query.replace('"', '').replace("'", '')
    
    # Step 1: Strip common command phrases early
    processed = re.sub(r'\b(find|search for|search|concordance of|query|look up|examples of|the word|the)\b', '', processed, flags=re.IGNORECASE)
    
    # Step 2: Apply wildcard heuristics BEFORE word order preprocessing
    # Pattern: "words beginning with X" or "word starting with X" -> "X*"
    match = re.search(r'\b(?:words?|)\s*(?:beginning|begins|starting|starts|start)\s+with\s+(\w+)', processed, re.IGNORECASE)
    if match:
        term = match.group(1)
        processed = re.sub(r'\b(?:words?|)\s*(?:beginning|begins|starting|starts|start)\s+with\s+\w+', f'{term}*', processed, flags=re.IGNORECASE)
    
    # Pattern: "words ending with X" or "word ends with X" -> "*X"
    match = re.search(r'\b(?:words?|)\s*(?:ending|ends|end)\s+with\s+(\w+)', processed, re.IGNORECASE)
    if match:
        term = match.group(1)
        processed = re.sub(r'\b(?:words?|)\s*(?:ending|ends|end)\s+with\s+\w+', f'*{term}', processed, flags=re.IGNORECASE)
    
    # Pattern: "words containing X" or "word contains X" -> "*X*"
    match = re.search(r'\b(?:words?|)\s*(?:containing|contains|contain)\s+(\w+)', processed, re.IGNORECASE)
    if match:
        term = match.group(1)
        processed = re.sub(r'\b(?:words?|)\s*(?:containing|contains|contain)\s+\w+', f'*{term}*', processed, flags=re.IGNORECASE)
    
    # Pattern: "any word" or "anything" -> "*"
    processed = re.sub(r'\b(?:any\s+word|anything|any)\b', '*', processed, flags=re.IGNORECASE)
    
    # Step 3: NOW apply word order preprocessing (wildcards are already in place)
    processed = preprocess_query_with_rules(processed)
    
    # Step 4: Map linguistic terms to POS tags
    # Use custom definitions if provided, otherwise fall back to defaults
    if reverse_pos_map:
        # Use ONLY custom mappings from database
        pos_mappings = generate_pos_mappings_from_definitions(reverse_pos_map)
    else:
        # Fall back to default mappings (Penn Treebank style) only if no custom definitions
        pos_mappings = {
            r'\bnoun\b': '_NN',
            r'\bnouns\b': '_NN',
            r'\bverb\b': '_VB',
            r'\bverbs\b': '_VB',
            r'\badjective\b': '_JJ',
            r'\badjectives\b': '_JJ',
            r'\badverb\b': '_RB',
            r'\badverbs\b': '_RB',
            r'\bpreposition\b': '_IN',
            r'\bprepositions\b': '_IN',
            r'\bdeterminer\b': '_DT',
            r'\bdeterminers\b': '_DT',
            r'\bpronoun\b': '_PRP',
            r'\bpronouns\b': '_PRP',
            r'\bconjunction\b': '_CC',
            r'\bconjunctions\b': '_CC',
            r'\bmodal\b': '_MD',
            r'\bmodals\b': '_MD',
            r'\bparticle\b': '_RP',
            r'\bparticles\b': '_RP',
            r'\binterjection\b': '_UH',
            r'\binterjections\b': '_UH',
        }
    
    for pattern, tag in pos_mappings.items():
        processed = re.sub(pattern, tag, processed, flags=re.IGNORECASE)
    
    # Step 5: Clean up extra whitespace
    processed = ' '.join(processed.split())
    processed = processed.strip()
    
    # Step 6: Build response based on module
    if module_selection.lower() == "concordance":
        return {
            "query": processed,
            "window": 5,
            "pos_filter": "",
            "sort_order": "Node"
        }, None
    elif module_selection.lower() == "collocation":
        return {
            "node_word": processed,
            "window": 5,
            "min_freq": 3,
            "pos_filter": "",
            "lemma_filter": "",
            "token_filter": ""
        }, None
    elif module_selection.lower() == "ngram":
        return {
            "n_size": 2,
            "min_freq": 3,
            "search_term": processed
        }, None
    elif module_selection.lower() == "dictionary":
        return {
            "word": processed
        }, None
    elif module_selection.lower() == "distribution":
        return {
            "search_term": processed,
            "metric": "Relative Frequency (Density %)"
        }, None
    else:
        return None, f"Unknown module: {module_selection}"



def parse_nl_query(user_query, module_selection, ai_provider=None, gemini_api_key=None,
                  ollama_url=None, 
                  ollama_model=None,
                  pos_definitions=None,
                  language="English",
                  gemini_model=None):
    """
    Parses a natural language query into specific parameters for the selected module.
    pos_definitions: Optional dict {tag: definition} to help AI understand corpus-specific tags.
    Returns: (dict, error_message)
    """
    
    # Apply rule-based preprocessing first
    preprocessed_query = preprocess_query_with_rules(user_query)
    
    # ... (schemas logic remains) ...
    schemas = {
        "collocation": """
        {
            "node_word": "string (the target word or pattern, e.g. 'run', '[lemma]*')",
            "window": "integer (1-10, default 5)",
            "min_freq": "integer (default 3)",
            "pos_filter": "string (comma joined tags e.g. 'JJ, NN' or exclusion '-VB')",
            "lemma_filter": "string (comma joined lemmas)",
            "token_filter": "string (comma joined specific tokens)"
        }
        """,
        "concordance": """
        {
            "query": "string",
            "window": 5,
            "pos_filter": "",
            "sort_order": "Node"
        }
        """,
        "ngram": """
        {
            "n_size": "integer (2-5, default 3)",
            "min_freq": "integer (default 3)",
            "search_term": "string (optional filter for the n-grams)"
        }
        """,
        "dictionary": """
        {
             "word": "string (target word to look up)"
         }
        """,
        "distribution": """
        {
            "search_term": "string (target word or pattern)",
            "metric": "string (optional preference: 'Absolute Frequency', 'Relative to Peak (%)', 'Relative Frequency (Density %)')"
        }
        """,
        "keyword": """
        {
            "min_freq": "integer (default 5)",
            "top_n": "integer (default 100)",
            "p_val_cutoff": "string (0.05, 0.01, 0.001, or None)"
        }
        """
    }

    schema = schemas.get(module_selection.lower())
    if not schema:
        return None, f"Unknown module: {module_selection}"

    # Construct POS context
    pos_context = ""
    if pos_definitions:
        # Check for language override in pos_definitions (safe-passing mechanism)
        if isinstance(pos_definitions, dict) and '__language_context__' in pos_definitions:
            if language == "English": # Only override if default
                 language = pos_definitions.pop('__language_context__')
        
        pos_list = "\n".join([f"            - '{v}' -> '_{k}'" for k, v in pos_definitions.items() if k != '__language_context__'])
        pos_context = f"""
        - User-Defined POS Tag Mappings (PRIORITIZE THESE):
{pos_list}
        """

    prompt = f"""
    Role: Corpus Query System Assistant.
    Language Context: The user is querying a corpus in {language}.
    Task: Convert the User's Natural Language Query into a JSON object for the '{module_selection}' module.
    
    Reference Schema:
    {schema}
    
    Instructions:
    - You must FILL the JSON structure above.
    - Extract constraints.
    - Tagging & Search Query Construction:
{pos_context}
        - Standard Mappings:
            - "adjective" -> "_JJ", "noun" -> "_NN", "verb" -> "_VB"
            - "adverb" -> "_RB", "preposition" -> "_IN", "pronoun" -> "_PRP"
        - Wildcards: "starting with A" -> "A*", "ending with B" -> "*B"
    - CRITICAL RULES:
        - IGNORE and STRIP command phrases: "find", "search for", "concordance of", "query", "look up", "examples of".
        - The 'query' MUST be the exact content requested.
        - NEVER guess tags. If user says "find cat", output "cat".
        - ONLY add tags if the user EXPLICITLY says "noun", "verb", "adjective", or a custom tag name.
        - IF NO TAGS ARE REQUESTED, DO NOT ADD THEM. "Find house" -> "house".
        - DEFAULT WINDOW IS 5. Do not use 50.
        - NO placeholders like "string" or "SEARCH_PATTERN".
 
    ### EXAMPLES (Do not copy these, use them as a guide) ###
    Input: "happy _NN"
    Output: {{ "query": "happy _NN", "window": 5, "pos_filter": "", "sort_order": "Node" }}
    
    Input: "run _RB"
    Output: {{ "query": "run _RB", "window": 5, "pos_filter": "", "sort_order": "Node" }}

    Input: "house"
    Output: {{ "query": "house", "window": 5, "pos_filter": "", "sort_order": "Node" }}

    Input: "makan"
    Output: {{ "query": "makan", "window": 5, "pos_filter": "", "sort_order": "Node" }}

    Input: "un* *ly"
    Output: {{ "query": "un* *ly", "window": 5, "pos_filter": "", "sort_order": "Node" }}

    ### YOUR TASK ###
    Input: "{preprocessed_query}"
    Output:
    """

    settings = _resolve_ai_settings(ai_provider, gemini_api_key, gemini_model, ollama_url, ollama_model)

    if settings["ai_provider"] == "Gemini" and settings["gemini_api_key"]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings['gemini_model']}:generateContent?key={settings['gemini_api_key']}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(url, json=payload, timeout=60)
            res_json = response.json()
            if 'error' in res_json: return None, f"Gemini Error: {res_json['error'].get('message')}"
            text = res_json['candidates'][0]['content']['parts'][0]['text']
            # Clean markdown code blocks if present
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text), None
        except Exception as e: return None, f"Gemini Error: {e}"
    else:
        # Ollama
        try:
            payload = {"model": settings["ollama_model"], "prompt": prompt, "stream": False, "format": "json"} 
            # Note: "format": "json" is supported by recent Ollama versions/models, valid for phi3 usually or it just helps
            
            response = requests.post(settings["ollama_url"], json=payload, timeout=60)
            res_json = response.json()
            raw_text = res_json.get('response') or res_json.get('content')
            
            if not raw_text: return None, "Empty response from AI."
            
            # Attempt to parse JSON
            return json.loads(raw_text), None
            
        except Exception as e: return None, f"Ollama Error: {e}"

def guess_pos_definitions(tags_list, ai_provider=None, gemini_api_key=None,
                          ollama_url=None,
                          ollama_model=None,
                          gemini_model=None):
    """
    Uses AI to guess definitions for a list of POS tags.
    Returns: (dict {tag: definition}, error)
    """
    settings = _resolve_ai_settings(ai_provider, gemini_api_key, gemini_model, ollama_url, ollama_model)
    
    tags_str = ", ".join(tags_list)
    prompt = f"""
    Role: Computational Linguist.
    Task: specific definitions for these Part-of-Speech (POS) tags.
    Tags: {tags_str}
    
    Instructions:
    - Provide a short, standard linguistic definition for each tag.
    - Common conventions: NN -> Noun, VB -> Verb, JJ -> Adjective, etc.
    - If unsure, provide a reasonable guess based on standard tagsets (e.g. Penn Treebank, CLAWS).
    - Return ONLY valid JSON: {{ "TAG": "Definition", ... }}
    """
    
    if settings["ai_provider"] == "Gemini" and settings["gemini_api_key"]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings['gemini_model']}:generateContent?key={settings['gemini_api_key']}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(url, json=payload, timeout=60)
            res_json = response.json()
            if 'error' in res_json: return None, f"Gemini Error: {res_json['error'].get('message')}"
            text = res_json['candidates'][0]['content']['parts'][0]['text']
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text), None
        except Exception as e: return None, f"Gemini Error: {e}"
        
    else:
        # Ollama
        try:
            payload = {"model": settings["ollama_model"], "prompt": prompt, "stream": False, "format": "json"}
            response = requests.post(settings["ollama_url"], json=payload, timeout=90)
            res_json = response.json()
            raw_text = res_json.get('response') or res_json.get('content')
            if not raw_text: return None, "Empty response from AI."
            return json.loads(raw_text), None
        except Exception as e: return None, f"Ollama Error: {e}"
