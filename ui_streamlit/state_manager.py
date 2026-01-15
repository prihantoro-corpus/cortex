import streamlit as st
import pandas as pd
from core.visualiser.styles import POS_COLOR_MAP

def init_session_state():
    """
    Initializes all necessary session state variables for the CORTEX application.
    """
    # 1. Corpus Data & Settings
    if 'current_corpus_path' not in st.session_state:
        st.session_state['current_corpus_path'] = None
    if 'current_corpus_name' not in st.session_state:
        st.session_state['current_corpus_name'] = "No Corpus Loaded" 
    if 'corpus_type' not in st.session_state:
        st.session_state['corpus_type'] = "Monolingual" # or Parallel
    if 'target_lang' not in st.session_state:
        st.session_state['target_lang'] = "en"
    if 'corpus_stats' not in st.session_state:
        st.session_state['corpus_stats'] = None
        
    # 2. Comparison Mode
    if 'comparison_mode' not in st.session_state:
        st.session_state['comparison_mode'] = False
    if 'comp_corpus_path' not in st.session_state:
        st.session_state['comp_corpus_path'] = None
    if 'comp_corpus_name' not in st.session_state:
        st.session_state['comp_corpus_name'] = "No Secondary Corpus"

    # 3. Tool State: Concordance
    if 'kwic_search_term' not in st.session_state:
        st.session_state['kwic_search_term'] = ""
    if 'kwic_results' not in st.session_state:
        st.session_state['kwic_results'] = None
        
    # 4. Tool State: N-Gram
    if 'ngram_results' not in st.session_state:
        st.session_state['ngram_results'] = None
        
    # 5. Tool State: Collocation
    if 'collocation_results' not in st.session_state:
        st.session_state['collocation_results'] = None
    if 'collocation_params' not in st.session_state:
        st.session_state['collocation_params'] = {}
        
    # 6. Tool State: Dictionary
    if 'dictionary_history' not in st.session_state:
        st.session_state['dictionary_history'] = []
    if 'current_dict_term' not in st.session_state:
        st.session_state['current_dict_term'] = ""
        
    # 7. AI Service
    if 'ai_provider' not in st.session_state:
        st.session_state['ai_provider'] = "Ollama"
    if 'gemini_api_key' not in st.session_state:
        st.session_state['gemini_api_key'] = ""
    if 'ollama_url' not in st.session_state:
        st.session_state['ollama_url'] = "http://127.0.0.1:11434/api/generate"
    if 'ai_model' not in st.session_state:
        st.session_state['ai_model'] = "phi3:latest" 
        
    # 8. XML Structure
    if 'xml_structure_data' not in st.session_state:
        st.session_state['xml_structure_data'] = None

    # 9. UI Triggers
    if 'trigger_kwic' not in st.session_state:
        st.session_state['trigger_kwic'] = False


def reset_tool_states():
    """
    Clears results when switching corpora or major modes.
    """
    st.session_state['kwic_results'] = None
    st.session_state['ngram_results'] = None
    st.session_state['collocation_results'] = None
    st.session_state['dictionary_history'] = []
    st.session_state['current_dict_term'] = ""

def get_state(key, default=None):
    return st.session_state.get(key, default)

def set_state(key, value):
    st.session_state[key] = value
