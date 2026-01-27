import streamlit as st
import pandas as pd
import altair as alt
from ui_streamlit.state_manager import get_state, set_state
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.preprocessing.xml_parser import apply_xml_restrictions
from core.modules.distribution import calculate_distribution
from core.ai_service import parse_nl_query, parse_nl_query_rules_only
from core.modules.overview import get_pos_definitions

def render_distribution_view():
    st.header("Distribution")
    
    corpus_path = get_state('current_corpus_path')
    corpus_name = get_state('current_corpus_name', 'Corpus')
    
    if not corpus_path:
        st.warning("Please load a corpus first.")
        return

    # --- Comparison Mode State ---
    comp_mode = get_state('comparison_mode', False)
    comp_path = get_state('comp_corpus_path')
    comp_name = get_state('comp_corpus_name')

    # 1. Controls
    search_mode = st.radio("Search Mode", ["Standard", "Natural Language (Rule)", "Natural Language (AI)"], horizontal=True, key="dist_search_mode")
    
    if search_mode == "Natural Language (AI)":
        st.markdown("### 🧠 Natural Language Search")
        nl_query = st.text_input("Describe distribution query", placeholder="e.g. Distribution of 'technology'")
        
        if st.button("Search with AI", type="primary", key="dist_ai_btn"):
             if not nl_query:
                 st.warning("Please enter a query.")
             else:
                 with st.spinner("Parsing query..."):
                     params, err = parse_nl_query(
                         nl_query, 
                         "distribution",
                         ai_provider=get_state('ai_provider'),
                         gemini_api_key=get_state('gemini_api_key'),
                         ollama_url=get_state('ollama_url'),
                         ollama_model=get_state('ai_model')
                     )
                 
                 if params and params.get('search_term'):
                     term = params.get('search_term')
                     set_state('dist_search_term', term)
                     st.success(f"✓ Found term: '{term}'")
                     
                     # Execute Logic directly
                     with st.spinner(f"Calculating distribution for '{term}'..."):
                         # Basic default restrictions
                         xml_filters = render_xml_restriction_filters(corpus_path, "distribution", corpus_name=corpus_name)
                         xml_where, xml_params = apply_xml_restrictions(xml_filters)
                         
                         dist_df, meta_dists = calculate_distribution(corpus_path, term, xml_where, xml_params)
                         if dist_df.empty:
                             st.warning("No matches found.")
                             set_state('last_dist_results_primary', None)
                         else:
                             set_state('last_dist_results_primary', {'df': dist_df, 'meta_dists': meta_dists, 'term': term, 'corpus': corpus_name})
                 else:
                     st.error("Could not determine search term from query.")

    if search_mode == "Natural Language (Rule)":
        st.markdown("### ⚡ Natural Language Search (Rule-Based)")
        st.caption("Fast, deterministic parsing. Finds distribution of terms, wildcards, or POS tags.")
        nl_query = st.text_input("Distribution Query (NL)", value=get_state('dist_nl_query_rule', ''), placeholder="e.g. distribution of 'noun'", key="dist_nl_input_rule")

    if search_mode == "Standard":
        with st.expander("Search Controls", expanded=True):
            if not comp_mode:
                search_term = st.text_input(
                    "Node Word(s)", 
                    value=get_state('dist_search_term', ''), 
                    key="dist_input", 
                    help="Use * for wildcards (e.g. run*), _TAG for POS (e.g. _NN), or [lemma] for lemma (e.g. [run])"
                )
            else:
                c1, c2 = st.columns(2)
                with c1:
                    search_term_1 = st.text_input(f"Primary ({corpus_name})", value=get_state('dist_search_term', ''), key="dist_input_c1")
                with c2:
                    search_term_2 = st.text_input(f"Comparison ({comp_name if comp_name else 'Secondary'})", value=get_state('dist_search_term_2', ''), key="dist_input_c2")

    # --- XML Restriction Filters ---
    if not comp_mode:
        xml_filters = render_xml_restriction_filters(corpus_path, "distribution", corpus_name=corpus_name)
        xml_where, xml_params = apply_xml_restrictions(xml_filters)
    else:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            xml_filters_1 = render_xml_restriction_filters(corpus_path, "distribution_c1", corpus_name=corpus_name)
            xml_where_1, xml_params_1 = apply_xml_restrictions(xml_filters_1)
        with col_f2:
            if comp_path:
                xml_filters_2 = render_xml_restriction_filters(comp_path, "distribution_c2", corpus_name=comp_name)
                xml_where_2, xml_params_2 = apply_xml_restrictions(xml_filters_2)
            else:
                st.info("Load a comparison corpus in sidebar.")
                xml_where_2, xml_params_2 = "", []
    
    # 3. Execution (Shared Logic)
    if st.button("Generate Distribution", type="primary"):
        # Helper to resolve term
        def resolve_term(mode, std_term, nl_q):
            if mode == "Standard": return std_term
            if mode == "Natural Language (Rule)":
                if not nl_q: return None
                set_state('dist_nl_query_rule', nl_q)
                
                pos_defs = get_pos_definitions(corpus_path) or {}
                reverse_pos_map = {v.lower(): k for k, v in pos_defs.items() if v}

                p, _ = parse_nl_query_rules_only(nl_q, "concordance", reverse_pos_map=reverse_pos_map)
                return p.get('query', '') if p else None
            return None

        if not comp_mode:
            # Single Mode
            term_to_use = resolve_term(search_mode, search_term if 'search_term' in locals() else '', nl_query if 'nl_query' in locals() else '')
            
            if not term_to_use:
                st.error("Please enter a search term.")
                return
                
            set_state('dist_search_term', term_to_use)
            with st.spinner(f"Calculating for '{term_to_use}'..."):
                dist_df, meta_dists = calculate_distribution(corpus_path, term_to_use, xml_where, xml_params)
                if dist_df.empty:
                    st.warning("No matches found.")
                    set_state('last_dist_results_primary', None)
                else:
                    set_state('last_dist_results_primary', {'df': dist_df, 'meta_dists': meta_dists, 'term': term_to_use, 'corpus': corpus_name})
        else:
            # Comparison Mode
            # For NL Rule in Comparison, we apply the NL query to BOTH? Or split?
            # Current design: NL Rule is single field. We'll use it for both for now, or fall back to Standard layout for comparison.
            # Actually, standard layout had 2 inputs. NL Rule has 1.
            # Let's apply NL Rule term to BOTH if active.
            
            term_1 = ""
            term_2 = ""
            
            if search_mode == "Standard":
                term_1 = search_term_1
                term_2 = search_term_2
            elif search_mode == "Natural Language (Rule)":
                # Use same term
                resolved = resolve_term("Natural Language (Rule)", "", nl_query if 'nl_query' in locals() else '')
                term_1 = resolved
                term_2 = resolved
            
            if not term_1 and not term_2:
                st.error("Please enter a search term.")
                return
            
            set_state('dist_search_term', term_1)
            set_state('dist_search_term_2', term_2)
            
            if term_1:
                with st.spinner(f"Primary: {term_1}..."):
                    d1, m1 = calculate_distribution(corpus_path, term_1, xml_where_1, xml_params_1)
                    set_state('last_dist_results_primary', {'df': d1, 'meta_dists': m1, 'term': term_1, 'corpus': corpus_name})
            if term_2 and comp_path:
                with st.spinner(f"Comparison: {term_2}..."):
                    d2, m2 = calculate_distribution(comp_path, term_2, xml_where_2, xml_params_2)
                    set_state('last_dist_results_secondary', {'df': d2, 'meta_dists': m2, 'term': term_2, 'corpus': comp_name})

    # 2. Results Display
    if not comp_mode:
        res = get_state('last_dist_results_primary')
        if res:
            render_dist_column(res)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"📊 Primary: {corpus_name}")
            res1 = get_state('last_dist_results_primary')
            if res1: render_dist_column(res1, key_suffix="c1")
        with col2:
            st.subheader(f"📊 Comparison: {comp_name}")
            if not comp_path:
                st.info("Load a comparison corpus in sidebar.")
            else:
                res2 = get_state('last_dist_results_secondary')
                if res2: render_dist_column(res2, key_suffix="c2")

def render_dist_column(results, key_suffix=""):
    dist_df = results['df']
    meta_dists = results.get('meta_dists', {})
    term = results['term']
    
    st.markdown(f"**Results for '{term}'**")
    
    # Linear Distribution
    total_freq = dist_df['Absolute Frequency'].sum()
    st.metric("Total Absolute Frequency", f"{total_freq:,}")
    
    chart_df = dist_df.copy()
    available_metrics = [m for m in ["Relative to Peak (%)", "Relative Frequency (Density %)", "Absolute Frequency"] if m in chart_df.columns]
    
    if not available_metrics:
        st.error("Data out of date. Regenerate.")
    else:
        metric_to_plot = st.radio("Metric", available_metrics, horizontal=True, key=f"dist_metric_{key_suffix}")
        
        chart = alt.Chart(chart_df).mark_bar(color='#00ADB5').encode(
            x=alt.X('Corpus Position (%):Q', title='Position (%)', scale=alt.Scale(domain=[1, 100])),
            y=alt.Y(f'{metric_to_plot}:Q', title=metric_to_plot, scale=alt.Scale(domainMin=0)),
            tooltip=['Corpus Position (%)', 'Absolute Frequency', 'Relative to Peak (%)']
        ).properties(height=300).configure_axis(labelAngle=0)
        
        st.altair_chart(chart, use_container_width=True)
    
    # Metadata & Filename
    if meta_dists:
        # Separate Filename if present for prominence
        df_file = meta_dists.pop('filename', None)
        if df_file is not None:
            st.markdown("#### 📁 Distribution by Filename")
            file_chart = alt.Chart(df_file).mark_bar(color='#00ADB5').encode(
                x=alt.X('Value:N', title='Filename', sort=None),
                y=alt.Y('Relative (%):Q', title='Relative (%)', scale=alt.Scale(domainMin=0, domainMax=100)),
                tooltip=['Value', 'Absolute', 'PMW', 'Relative (%)']
            ).properties(height=250).configure_axis(labelAngle=-45)
            st.altair_chart(file_chart, use_container_width=True)
            st.dataframe(df_file, use_container_width=True, hide_index=True)
            st.markdown("---")

        with st.expander("Metadata Distributions", expanded=False):
            for attr, df_attr in meta_dists.items():
                st.markdown(f"**{attr.title()}**")
                meta_chart = alt.Chart(df_attr).mark_bar(color='#00ADB5').encode(
                    x=alt.X('Value:N', title=attr.title(), sort=None),
                    y=alt.Y('Relative (%):Q', title='Relative (%)', scale=alt.Scale(domainMin=0, domainMax=100)),
                    tooltip=['Value', 'Absolute', 'PMW', 'Relative (%)']
                ).properties(height=200).configure_axis(labelAngle=0)
                st.altair_chart(meta_chart, use_container_width=True)
                st.dataframe(df_attr, use_container_width=True, hide_index=True)

    st.markdown("---")
    if st.button("🧠 Interpret Distribution (AI)", key=f"btn_dist_ai_{key_suffix}"):
        with st.spinner("AI is analyzing distribution..."):
            dist_summary = dist_df[['Corpus Position (%)', 'Absolute Frequency']].to_string(index=False)
            from core.ai_service import interpret_results_llm
            resp, err = interpret_results_llm(
                target_word=term,
                analysis_type="Lexical Distribution Analysis",
                data_description=f"Distribution of '{term}' in '{results.get('corpus', 'Corpus')}'.",
                data=dist_summary,
                ai_provider=get_state('ai_provider'),
                gemini_api_key=get_state('gemini_api_key'),
                ollama_url=get_state('ollama_url'),
                ollama_model=get_state('ai_model')
            )
            if resp: set_state(f'llm_res_dist_{key_suffix}', resp)
            else: st.error(err)
    
    ai_res = get_state(f'llm_res_dist_{key_suffix}')
    if ai_res:
        with st.expander("🤖 AI Distribution Interpretation", expanded=True):
            st.markdown(ai_res)
