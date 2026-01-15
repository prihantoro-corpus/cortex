import streamlit as st
import pandas as pd
import textwrap
from ui_streamlit.state_manager import get_state, set_state
from ui_streamlit.caching import (
    cached_get_lemma_details, 
    cached_get_context_ngrams, 
    cached_get_dict_examples,
    cached_get_random_examples,
    cached_get_related_forms,
    cached_generate_kwic,
    cached_get_subcorpus_size
)
from core.preprocessing.xml_parser import apply_xml_restrictions
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.ai_service import interpret_results_llm
from core.statistics.frequency import pmw_to_zipf, zipf_to_band
from core.visualiser.charts import get_zipf_bar_html
import streamlit.components.v1 as components
import eng_to_ipa as ipa
from cefrpy import CEFRAnalyzer
from ui_streamlit.caching import cached_generate_collocation
from ui_streamlit.components.result_display import render_kwic_table
from core.modules.g2p_service import get_ipa_transcription
from core.ai_service import interpret_results_llm, chat_with_llm, parse_nl_query
import time

def render_dictionary_view():
    st.header("Dictionary & Lexicography")
    
    corpus_path = get_state('current_corpus_path')
    if not corpus_path:
        st.warning("Please load a corpus first.")
        return

    # 1. Search Bar
    comp_mode = get_state('comparison_mode', False)
    comp_path = get_state('comp_corpus_path')
    comp_name = get_state('comp_corpus_name')

    if not comp_mode:
        # Search Mode Toggle
        search_mode = st.radio("Input Mode", ["Standard", "Natural Language (AI)"], horizontal=True, key="dict_search_mode")
        
        if search_mode == "Natural Language (AI)":
             nl_query = st.text_input("Ask what you want to define", placeholder="e.g. What does 'artificial' mean?")
             if st.button("Ask AI", type="primary"):
                 with st.spinner("Parsing..."):
                     params, err = parse_nl_query(
                         nl_query, 
                         "dictionary",
                         ai_provider=get_state('ai_provider'),
                         gemini_api_key=get_state('gemini_api_key'),
                         ollama_url=get_state('ollama_url'),
                         ollama_model=get_state('ai_model')
                     )
                 
                 if params and params.get('word'):
                     term = params.get('word')
                     set_state('current_dict_term', term)
                     update_history(term)
                     st.rerun()
                 else:
                     st.error("Could not identify the word to define.")
        
        if search_mode == "Standard":
            col1, col2 = st.columns([3, 1])
            with col1:
                 search_term = st.text_input("Enter a word to analyze:", value=get_state('current_dict_term'), key="dict_search_input")
            with col2:
                 if st.button("Search", key="dict_search_btn"):
                     set_state('current_dict_term', search_term)
                     if search_term:
                         update_history(search_term)
                     st.rerun()
        current_term = get_state('current_dict_term')
    else:
        st.markdown("##### Comparison Search")
        c1, c2 = st.columns(2)
        with c1:
            term1 = st.text_input(f"Primary ({get_state('current_corpus_name', 'Corpus')})", value=get_state('current_dict_term'), key="dict_term_1")
        with c2:
            term2 = st.text_input(f"Comparison ({comp_name if comp_name else 'Secondary'})", value=get_state('current_dict_term_2', ''), key="dict_term_2")
            
        if st.button("Search Comparison", type="primary", key="dict_comp_btn"):
             set_state('current_dict_term', term1)
             set_state('current_dict_term_2', term2)
             if term1: update_history(term1)
             st.rerun()
             
        current_term = get_state('current_dict_term')
        current_term_2 = get_state('current_dict_term_2')

    
    # comp_mode/path/name var definition moved up


    # --- XML Restriction Filters ---
    if not comp_mode:
        xml_filters = render_xml_restriction_filters(corpus_path, "dictionary")
        xml_where, xml_params = apply_xml_restrictions(xml_filters)
        render_dictionary_result_column(corpus_path, get_state('current_corpus_name'), current_term, xml_where, xml_params)
    else:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            xml_filters_1 = render_xml_restriction_filters(corpus_path, "dictionary_c1")
            xml_where_1, xml_params_1 = apply_xml_restrictions(xml_filters_1)
        with col_f2:
            xml_filters_2 = render_xml_restriction_filters(comp_path, "dictionary_c2")
            xml_where_2, xml_params_2 = apply_xml_restrictions(xml_filters_2)
            
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.subheader(f"Primary: {get_state('current_corpus_name', 'Corpus')}")
            render_dictionary_result_column(corpus_path, get_state('current_corpus_name'), current_term, xml_where_1, xml_params_1, key_suffix="c1")
        with col_c2:
            st.subheader(f"Comparison: {comp_name}")
            if not comp_path:
                st.info("Load a comparison corpus in sidebar.")
            else:
                comp_stats = get_state('comp_corpus_stats')
                render_dictionary_result_column(comp_path, comp_name, current_term_2, xml_where_2, xml_params_2, key_suffix="c2", override_stats=comp_stats)

def render_dictionary_result_column(path, corpus_name, current_term, xml_where, xml_params, key_suffix="", override_stats=None):
    if not current_term:
        st.info("Enter a word to view dictionary data.")
        return

    # Data Fetching
    forms_df, unique_pos, unique_lemmas = cached_get_lemma_details(
        path, current_term, xml_where_clause=xml_where, xml_params=xml_params
    )
    context_ngrams = cached_get_context_ngrams(
        path, current_term, xml_where_clause=xml_where, xml_params=xml_params
    ) or {}
    examples = cached_get_dict_examples(
        path, current_term, xml_where_clause=xml_where, xml_params=xml_params
    )
    
    # New: Random Examples and Related Forms
    random_examples = cached_get_random_examples(path, current_term, limit=5, xml_where_clause=xml_where, xml_params=xml_params)
    related_forms = cached_get_related_forms(path, current_term, xml_where_clause=xml_where, xml_params=xml_params)
    
    # New: Collocates for examples
    coll_df, _, _ = cached_generate_collocation(
        db_path=path, 
        word=current_term, 
        window=5, 
        min_freq=3, 
        max_rows=20, 
        is_raw=False, 
        corpus_stats=override_stats if override_stats else get_state('corpus_stats', {}),
        xml_where_clause=xml_where, xml_params=xml_params
    )

    # 3. Display Results
    # --- Section A: Overview & Summary ---
    # Header Statistics: Freq, Rel, Band should reflect the SPECIFIC search word
    word_forms_df = forms_df[forms_df['Token'] == current_term.lower()] if not forms_df.empty else pd.DataFrame()
    total_freq = word_forms_df['freq'].sum() if not word_forms_df.empty else 0
    
    stats = override_stats if override_stats else get_state('corpus_stats', {})
    if xml_where:
        total_tokens = cached_get_subcorpus_size(path, xml_where_clause=xml_where, xml_params=xml_params)
    else:
        total_tokens = stats.get('total_tokens', 1)
    
    pmw = (total_freq / total_tokens) * 1000000 if total_tokens > 0 else 0
    zipf_score = pmw_to_zipf(pmw)
    band = zipf_to_band(zipf_score)
    
    # Generate Senses HTML (All detected senses)
    entry_examples_html = []
    if examples:
        for i, x in enumerate(examples): 
             # Handle both tuple (legacy) and dict (new) formats for backward compatibility
             if isinstance(x, dict):
                 pos_tag = x.get('pos', 'UNK')
                 ex_html = x.get('text', '')
                 meta = x.get('metadata', {})
             else:
                 pos_tag = x[0]
                 ex_html = x[1]
                 meta = {}
                 
             sense_df = forms_df[forms_df['POS'] == pos_tag] if not forms_df.empty else pd.DataFrame()
             # Fallback freq calculation if sense_df is empty or POS mismatch
             s_freq = sense_df['freq'].sum() if not sense_df.empty else 0
             s_pmw = (s_freq / total_tokens) * 1000000
             s_band = zipf_to_band(pmw_to_zipf(s_pmw))
             zipf_html = get_zipf_bar_html(s_band)
             
             # Metadata Badges
             meta_html = ""
             if meta:
                 for k, v in meta.items():
                     meta_html += f"""<span style="display: inline-block; background-color: #334155; color: #e2e8f0; font-size: 0.7em; padding: 1px 4px; border-radius: 4px; margin-left: 6px; border: 1px solid #475569; vertical-align: middle; cursor: help;" title="{k}: {v}">{v}</span>"""

             entry_examples_html.append(f"""
<div style="margin-bottom: 8px;">
    {i+1}. <span style="color: #33CC33; font-weight: bold;">[{pos_tag}]</span> {zipf_html} <i>{ex_html}</i>{meta_html}
</div>""")

    # Linguistic Enrichment
    ipa_text = ""
    cefr_level = ""
    target_lang = get_state('target_lang', 'en').lower()
    
    ipa_warning = ""
    
    if target_lang == 'en':
        ipa_text = ipa.convert(current_term)
        cefr_analyser = CEFRAnalyzer()
        try:
            cefr_res = cefr_analyser.get_average_word_level_CEFR(current_term)
            if cefr_res:
                cefr_level = f" (CEFR: {str(cefr_res).upper()})"
        except:
            cefr_level = ""
    elif target_lang in ['id', 'indonesian']:
        is_syllabified = get_state(f"syllabify_{key_suffix}", False)
        ipa_text = get_ipa_transcription(current_term, syllabify=is_syllabified)
        ipa_warning = " <span style='color: #FF5555; font-size: 0.7em;' title='Transcription is rule-based and may be inaccurate.'>⚠️</span>"

    # Key Patterns Logic
    # Bigrams: Top 1 Left, Top 1 Right (as per request "search_word word, word search_word")
    # Trigrams: Top 1 Left, Top 1 Center, Top 1 Right
    # Fourgrams: Top 1 Left, Top 1 Center-Left, Top 1 Center-Right, Top 1 Right
    key_patterns = []
    
    # helper to format tuple to string
    def fmt_ngram(x): return " ".join(x[0])
    
    # Bigrams: Target 2 total (Pooled from left and right)
    all_bg_l = [fmt_ngram(x) for x in context_ngrams.get('bigrams_left', [])]
    all_bg_r = [fmt_ngram(x) for x in context_ngrams.get('bigrams_right', [])]
    # Selection: Try to take one from each, otherwise take what's available
    tops_bg = []
    if all_bg_l: tops_bg.append(all_bg_l[0])
    if all_bg_r: tops_bg.append(all_bg_r[0])
    if len(tops_bg) < 2:
        for x in (all_bg_l[1:] + all_bg_r[1:]):
            if x not in tops_bg:
                tops_bg.append(x)
                if len(tops_bg) >= 2: break
    
    # Trigrams: Target 3 total
    all_tg_l = [fmt_ngram(x) for x in context_ngrams.get('trigrams_left', [])]
    all_tg_c = [fmt_ngram(x) for x in context_ngrams.get('trigrams_center', [])]
    all_tg_r = [fmt_ngram(x) for x in context_ngrams.get('trigrams_right', [])]
    tops_tg = []
    if all_tg_l: tops_tg.append(all_tg_l[0])
    if all_tg_c: tops_tg.append(all_tg_c[0])
    if all_tg_r: tops_tg.append(all_tg_r[0])
    if len(tops_tg) < 3:
        for x in (all_tg_l[1:] + all_tg_c[1:] + all_tg_r[1:]):
            if x not in tops_tg:
                tops_tg.append(x)
                if len(tops_tg) >= 3: break

    # Fourgrams: Target 4 total
    all_fg_l = [fmt_ngram(x) for x in context_ngrams.get('fourgrams_left', [])]
    all_fg_cl = [fmt_ngram(x) for x in context_ngrams.get('fourgrams_center_left', [])]
    all_fg_cr = [fmt_ngram(x) for x in context_ngrams.get('fourgrams_center_right', [])]
    all_fg_r = [fmt_ngram(x) for x in context_ngrams.get('fourgrams_right', [])]
    tops_fg = []
    if all_fg_l: tops_fg.append(all_fg_l[0])
    if all_fg_cl: tops_fg.append(all_fg_cl[0])
    if all_fg_cr: tops_fg.append(all_fg_cr[0])
    if all_fg_r: tops_fg.append(all_fg_r[0])
    if len(tops_fg) < 4:
        for x in (all_fg_l[1:] + all_fg_cl[1:] + all_fg_cr[1:] + all_fg_r[1:]):
            if x not in tops_fg:
                tops_fg.append(x)
                if len(tops_fg) >= 4: break
    
    bg_patterns = "; ".join(tops_bg) + (";" if tops_bg else "")
    tg_patterns = "; ".join(tops_tg) + (";" if tops_tg else "")
    fg_patterns = "; ".join(tops_fg) + (";" if tops_fg else "")
    
    # Actually populate key_patterns for AI usage
    key_patterns = [f"Bigram: {bg_patterns}", f"Trigram: {tg_patterns}", f"Fourgram: {fg_patterns}"]
    
    # Use continuous string for labeling to prevent Markdown from escaping
    key_patterns_labeled = f"<b>bigram:</b> {bg_patterns if bg_patterns else 'N/A'}<br><b>trigram:</b> {tg_patterns if tg_patterns else 'N/A'}<br><b>fourgram:</b> {fg_patterns if fg_patterns else 'N/A'}"
    
    # Helper for URLs (Ported from original app.py)
    def dictionary_url_helper(token, lang_code):
        l_upper = lang_code.upper()
        if l_upper in ('ID', 'INDONESIAN'):
             return f"https://kbbi.kemendikdasmen.go.id/entri/{token.lower()}", "Dictionary ↗", f"https://tesaurus.kemendikdasmen.go.id/tematis/lema/{token.lower()}", "Thesaurus ↗"
        elif l_upper in ('EN', 'ENG', 'ENGLISH'):
             return f"https://dictionary.cambridge.org/dictionary/english/{token.lower()}", "Dictionary ↗", f"https://www.collinsdictionary.com/dictionary/english-thesaurus/{token.lower()}", "Thesaurus ↗"
        else:
             return f"https://forvo.com/word/{token}/#{lang_code.lower()}", "Dictionary ↗", f"https://www.google.com/search?q={token.lower()}+thesaurus", "Thesaurus ↗"
             
    current_lang = get_state('target_lang', 'EN')
    dict_url, dict_lbl, thes_url, thes_lbl = dictionary_url_helper(current_term, current_lang)
    
    dict_link = dict_url
    thes_link = thes_url

    # Summary Data
    lemma_str = ", ".join(unique_lemmas) if unique_lemmas else "N/A"
    collocates_str = ", ".join(coll_df['Collocate'].tolist()[:20]) if not coll_df.empty else "N/A"
    related_str = ", ".join(related_forms[:5]) if related_forms else "N/A"

    lemma_forms = []
    if not forms_df.empty:
        if 'Token' in forms_df.columns:
            lemma_forms = sorted(list(set(forms_df['Token'].tolist())))
        elif 'Word' in forms_df.columns:
            lemma_forms = sorted(list(set(forms_df['Word'].tolist())))
        else:
            lemma_forms = []
    lemma_forms_str = ", ".join(lemma_forms) if lemma_forms else "N/A"
    
    base_font_size = "1.5em" if not get_state('comparison_mode') else "1.0em"

    # Harmonized Header Layout: Term /IPA/ (CEFR) [Warning]
    s_html = f'<div style="background-color: #1a1a1a; padding: 20px; border-radius: 10px; border: 1px solid #444; color: #eee; font-family: sans-serif; font-size: {base_font_size}; line-height: 1.5; margin-bottom: 20px;">'
    s_html += f'<h3 style="color: #FFEA00; margin: 0; display: inline-block;">{current_term}</h3>'
    if ipa_text:
        s_html += f'<span style="color: #888; margin-left:10px;">/{ipa_text}/</span>'
    if cefr_level:
        s_html += f'<span style="color: #888; margin-left:10px;">{cefr_level}</span>'
    if ipa_warning: 
        s_html += ipa_warning
    
    s_html += '<div style="margin-top: 5px; margin-bottom: 5px; display: flex; align-items: center; flex-wrap: wrap; gap: 15px;">'
    s_html += f'<span><b>Lemma:</b> <span style="color: #00ADB5;">{lemma_str}</span></span>'
    s_html += f'<span><b>POS:</b> {", ".join(unique_pos)}</span>'
    s_html += f'<span><b>Freq:</b> {total_freq:,}</span>'
    s_html += f'<span><b>Rel:</b> {pmw:.2f} pmw</span>'
    s_html += f'<span style="display: inline-flex; align-items: center; gap: 5px;"><b>Band:</b> {get_zipf_bar_html(band)}</span>'
    s_html += '</div>'
    
    s_html += '<div style="margin-bottom: 8px;">'
    s_html += f'<span style="font-size: 0.85em; color: #aaa;"><b>Words from the same lemma:</b> {lemma_forms_str}</span>'
    s_html += '</div>'
    
    s_html += '<div style="margin-bottom: 8px;">'
    if examples:
        for i, x in enumerate(examples):
             # Handle both tuple (legacy) and dict (new) formats for backward compatibility
             if isinstance(x, dict):
                 pos = x.get('pos', 'UNK')
                 text = x.get('text', '')
                 meta = x.get('metadata', {})
             else:
                 pos = x[0]
                 text = x[1]
                 meta = {}
             
             s_freq = forms_df[forms_df['POS']==pos]['freq'].sum() if not forms_df.empty else 0
             s_pmw = (s_freq / total_tokens) * 1000000
             s_zh = get_zipf_bar_html(zipf_to_band(pmw_to_zipf(s_pmw)))
             
             # Create Metadata Badges
             meta_html = ""
             if meta:
                 for k, v in meta.items():
                     # Generate a small distinct badge for each metadata item
                     # Using a soft background color
                     meta_html += f"""
                     <span style="
                        display: inline-block; 
                        background-color: #334155; 
                        color: #e2e8f0; 
                        font-size: 0.7em; 
                        padding: 1px 4px; 
                        border-radius: 4px; 
                        margin-left: 6px; 
                        border: 1px solid #475569;
                        vertical-align: middle;
                        cursor: help;" 
                        title="{k}: {v}">
                        {v}
                     </span>"""
             elif not meta and corpus_name:
                 # Fallback to Corpus Name if no specific metadata columns exist
                 meta_html += f"""
                 <span style="
                    display: inline-block; 
                    background-color: #334155; 
                    color: #e2e8f0; 
                    font-size: 0.7em; 
                    padding: 1px 4px; 
                    border-radius: 4px; 
                    margin-left: 6px; 
                    border: 1px solid #475569;
                    vertical-align: middle;
                    cursor: help;" 
                    title="Source Corpus">
                    {corpus_name}
                 </span>"""

             s_html += f"<div style='margin-bottom: 6px; display: flex; align-items: flex-start; gap: 8px;'><span style='min-width: 20px;'>{i+1}.</span> <div><span style='color: #33CC33; font-weight: bold;'>[{pos}]</span> {s_zh} <i>{text}</i>{meta_html}</div></div>"
    else:
        s_html += "<i style='color: #888;'>No sense examples available.</i>"
    s_html += '</div>'
    
    s_html += '<div style="background: #222; padding: 10px; border-radius: 5px; margin-bottom: 8px;">'
    s_html += '<b style="color: #FFEA00; text-transform: uppercase; font-size: 0.75em;">N-gram</b><br>'
    s_html += f'<span style="color: #aaa; font-size: 0.85em;">{key_patterns_labeled}</span>'
    if not bg_patterns:
        s_html += "<br><span style='color: #888; font-size: 0.75em;'>no bigram available</span>"
    s_html += '</div>'
    
    s_html += '<div style="margin-bottom: 8px; font-size: 0.85em;">'
    s_html += f'<b>Top Collocates:</b> {collocates_str}<br>'
    s_html += f'<b>Related:</b> {related_str}'
    s_html += '</div>'
    
    s_html += '<div style="font-size: 0.8em; margin-top: 10px; border-top: 1px solid #333; padding-top: 10px;">'
    s_html += f'<a href="{dict_link}" target="_blank" style="color: #00ADB5; text-decoration: none;">{dict_lbl} ↗</a> | '
    s_html += f'<a href="{thes_link}" target="_blank" style="color: #00ADB5; text-decoration: none;">{thes_lbl} ↗</a>'
    s_html += '</div></div>'
    
    # Use st.markdown for natural height growth, preventing clipping
    st.markdown(s_html, unsafe_allow_html=True)

    if not forms_df.empty:
        with st.expander("Word Form Distribution (Total Profile)", expanded=True):
            df_dist = forms_df.copy()
            df_dist['PMW'] = (df_dist['freq'] / total_tokens) * 1000000
            df_dist['Zipf'] = df_dist['PMW'].apply(pmw_to_zipf).round(2)
            df_dist['Band'] = df_dist['Zipf'].apply(zipf_to_band)
            st.dataframe(df_dist.sort_values('freq', ascending=False), use_container_width=True, hide_index=True)

    # --- Section B: Contextual Patterns ---
    with st.expander("Contextual Patterns", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### Preceding")
            if context_ngrams.get('bigrams_left'):
                st.table(pd.DataFrame(context_ngrams['bigrams_left'], columns=['Bigram', 'Freq']))
        with c2:
            st.markdown("##### Surrounding")
            if context_ngrams.get('trigrams_center'):
                st.table(pd.DataFrame(context_ngrams['trigrams_center'], columns=['Trigram', 'Freq']))
        with c3:
            st.markdown("##### Following")
            if context_ngrams.get('bigrams_right'):
                 st.table(pd.DataFrame(context_ngrams['bigrams_right'], columns=['Bigram', 'Freq']))

    # --- Section C: Examples ---
    with st.expander("Usage Examples (By POS Sense)", expanded=False):
        if examples:
            for x in examples[:10]:
                if isinstance(x, dict):
                    pos = x.get('pos', 'UNK')
                    sentence_html = x.get('text', '')
                    meta = x.get('metadata', {})
                else:
                    pos = x[0]
                    sentence_html = x[1]
                    meta = {}
                
                meta_html = ""
                if meta:
                     for k, v in meta.items():
                         meta_html += f"<span style='background-color: #334155; color: #e2e8f0; font-size: 0.6em; padding: 1px 3px; border-radius: 3px; margin-left: 4px; border: 1px solid #475569; vertical-align: middle; cursor: help;' title='{k}: {v}'>{v}</span>"

                st.markdown(f"**[{pos}]** {sentence_html}{meta_html}", unsafe_allow_html=True)
        else:
            st.caption("No examples found matching criteria.")
            
    with st.expander("Random Examples", expanded=False):
        if random_examples:
            for ex in random_examples:
                st.markdown(f"• {ex}", unsafe_allow_html=True)
        else:
            st.caption("No random examples.")

    with st.expander("Collocate Examples (Random Lines Containing Top Collocates)", expanded=False):
        if not coll_df.empty:
             kwic_table_data = []
             is_parallel = get_state('parallel_mode', False)
             target_map = get_state('target_sent_map', {})
             
             for i, coll in enumerate(coll_df['Collocate'].tolist()[:10]):
                 with st.spinner(f"Loading {coll}..."):
                      c_kwic, _, _, _, c_sent_ids, _ = cached_generate_kwic(
                          db_path=path, query=current_term, left=5, right=5, 
                          corpus_name=corpus_name, pattern_collocate_input=coll, pattern_window=5, limit=1,
                          xml_where_clause=xml_where, xml_params=xml_params
                       )
                      if c_kwic:
                           row_data = {
                               'Collocate': f"{i+1}. {coll}",
                               'Source Corpus': corpus_name,
                               'Left Context': c_kwic[0]['Left'],
                               'Node': c_kwic[0]['Node'],
                               'Right Context': c_kwic[0]['Right'],
                               'Metadata': c_kwic[0].get('Metadata', {})
                           }
                           if is_parallel:
                               sid = c_sent_ids[0] if c_sent_ids else None
                               trans = target_map.get(sid, "N/A") if sid is not None else ""
                               row_data['Translation'] = trans
                               
                           kwic_table_data.append(row_data)
             
             if kwic_table_data:
                  render_kwic_table(kwic_table_data, is_parallel=is_parallel, target_lang=get_state('tgt_lang_code', 'Target'))
             else:
                  st.caption("No examples found.")
        else:
            st.caption("No collocates available.")

    # --- Section D: Related Forms ---
    with st.expander("Related Forms (Regex Match)", expanded=False):
        if related_forms:
            st.write(", ".join(f"`{f}`" for f in related_forms[:50]))
        else:
            st.caption("No related forms.")

    # --- Section D: AI Interpretation ---
    st.warning("⚠️ **Note:** AI interpretation can be slow. For the fastest experience, use a smaller model like **`tinyllama`** in the Sidebar settings.")
    
    if st.button("Interpret with AI (Fast with tinyllama)", key=f"btn_dict_ai_{key_suffix}"):
        with st.spinner("Asking AI..."):
            # Slimmed Data for AI
            ai_data_slim = {
                "word": current_term,
                "pos_dist": unique_pos,
                "lemma": lemma_str,
                "top_collocates": collocates_str,
                "frequency": f"{total_freq} (Rel: {pmw:.2f} pmw)",
                "key_pattern_summary": key_patterns[:5] # Only send top 5 summary strings
            }
            
            # Format context string for follow-up chat
            context_summary = f"Dictionary Summary for '{current_term}':\n"
            context_summary += f"POS: {', '.join(unique_pos)}\n"
            context_summary += f"Patterns: {'; '.join(key_patterns)}\n"
            
            start_time = time.time()
            response, err = interpret_results_llm(
                target_word=current_term,
                analysis_type="Dictionary Analysis",
                data_description=f"Snapshot for '{current_term}' in '{corpus_name}'.",
                data=ai_data_slim,
                ai_provider=get_state('ai_provider'),
                gemini_api_key=get_state('gemini_api_key'),
                ollama_url=get_state('ollama_url'),
                ollama_model=get_state('ai_model')
            )
            elapsed_time = time.time() - start_time
            
            if response:
                set_state(f'llm_res_dict_{key_suffix}', response)
                set_state(f'llm_time_dict_{key_suffix}', elapsed_time)
                set_state(f'dict_chat_hist_{key_suffix}', [])
                set_state(f'dict_ai_context_{key_suffix}', context_summary)
            else:
                st.error(f"AI Error: {err}")
    
    ai_res = get_state(f'llm_res_dict_{key_suffix}')
    ai_time = get_state(f'llm_time_dict_{key_suffix}')
    
    if ai_res:
        # Time Notification
        if ai_time:
            mins, secs = divmod(int(ai_time), 60)
            st.caption(f"[Response time: {mins} minutes and {secs} seconds]")
            
        # Scrollable Box for Response
        st.markdown(f"""
        <div style="height: 400px; overflow-y: auto; background-color: #222; padding: 15px; border-radius: 10px; border: 1px solid #444;">
            {ai_res}
        </div>
        """, unsafe_allow_html=True)
        
        # --- Chat Interface ---
        st.markdown("### Follow-up Chat")
        
        # Chat History Container
        chat_container = st.container()
        chat_hist = get_state(f'dict_chat_hist_{key_suffix}', [])
        
        with chat_container:
            for chat in chat_hist:
                st.markdown(f"**You:** {chat['user']}")
                st.markdown(f"**AI:** {chat['ai']}")
                st.markdown("---")
        
        # Chat Input
        with st.form(key=f"dict_chat_form_{key_suffix}"):
            user_input = st.text_input("Type your comment/question here", key=f"dict_chat_in_{key_suffix}")
            send_btn = st.form_submit_button("Send")
            
        if send_btn and user_input:
            context_data = get_state(f'dict_ai_context_{key_suffix}', "No context available.")
            with st.spinner("AI is thinking..."):
                chat_resp, chat_err = chat_with_llm(
                    user_message=user_input,
                    context=context_data,
                    chat_history=chat_hist,
                    ai_provider=get_state('ai_provider'),
                    gemini_api_key=get_state('gemini_api_key'),
                    ollama_url=get_state('ollama_url'),
                    ollama_model=get_state('ai_model')
                )
                
                if chat_resp:
                    chat_hist.append({"user": user_input, "ai": chat_resp})
                    set_state(f'dict_chat_hist_{key_suffix}', chat_hist)
                    st.rerun()
                else:
                    st.error(f"Chat Error: {chat_err}")

def update_history(term):
    hist = get_state('dictionary_history')
    if term not in hist:
        hist.insert(0, term)
        set_state('dictionary_history', hist[:10])

def render_history():
    hist = get_state('dictionary_history')
    if hist:
        st.markdown("### Recent Searches")
        for term in hist:
            if st.button(term, key=f"hist_{term}"):
                set_state('current_dict_term', term)
                st.rerun()
