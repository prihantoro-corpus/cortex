import streamlit as st
import pandas as pd
from core.modules.dictionary_service import (
    get_all_lemma_forms_details, 
    get_detailed_contextual_ngrams, 
    get_dictionary_examples,
    get_random_examples,
    get_related_forms_by_regex,
    get_subcorpus_size
)
from core.modules.concordance import generate_kwic
from core.modules.ngram import generate_n_grams_v2
from core.modules.collocation import generate_collocation_results

@st.cache_data(show_spinner="Searching Dictionary...")
def cached_get_lemma_details(db_path, word, **kwargs):
    return get_all_lemma_forms_details(db_path, word, **kwargs)

@st.cache_data(show_spinner="Extracting Contexts...")
def cached_get_context_ngrams(db_path, word, **kwargs):
    return get_detailed_contextual_ngrams(db_path, word, **kwargs)

@st.cache_data(show_spinner="Formatting Examples...")
def cached_get_dict_examples(db_path, word, **kwargs):
    return get_dictionary_examples(db_path, word, **kwargs)

@st.cache_data(show_spinner="Fetching Random Examples...")
def cached_get_random_examples(db_path, word, **kwargs):
    return get_random_examples(db_path, word, **kwargs)

@st.cache_data(show_spinner="Calculating sub-corpus size...")
def cached_get_subcorpus_size(db_path, **kwargs):
    return get_subcorpus_size(db_path, **kwargs)

@st.cache_data(show_spinner="Searching related forms...")
def cached_get_related_forms(db_path, word, **kwargs):
    return get_related_forms_by_regex(db_path, word, **kwargs)

@st.cache_data(show_spinner="Generating KWIC...")
def cached_generate_kwic(db_path, query, left, right, corpus_name, **kwargs):
    return generate_kwic(db_path, query, left, right, corpus_name, **kwargs)

@st.cache_data(show_spinner="Generating Collocations...")
def cached_generate_collocation(db_path, word, window, min_freq, max_rows, is_raw, corpus_stats, 
                               token_filter="", pos_filter="", lemma_filter="", **kwargs):
    return generate_collocation_results(
        db_path, word, window, min_freq, max_rows, is_raw, 
        token_filter=token_filter, 
        pos_filter=pos_filter, 
        lemma_filter=lemma_filter, 
        corpus_stats=corpus_stats, 
        **kwargs
    )

@st.cache_data(show_spinner="Generating N-Grams...")
def cached_generate_ngrams(db_path, n, filters, is_raw, corpus_name, **kwargs):
    return generate_n_grams_v2(db_path, n, filters, is_raw, corpus_name, **kwargs)
