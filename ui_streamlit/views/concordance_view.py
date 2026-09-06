import streamlit as st
import re
import pandas as pd
import os
import itertools
import math
from ui_streamlit.state_manager import get_state, set_state
from ui_streamlit.utils import notify_timing
from ui_streamlit.caching import cached_generate_kwic, cached_get_subcorpus_size
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.preprocessing.xml_parser import apply_xml_restrictions
from core.ai_service import interpret_results_llm, parse_nl_query, parse_nl_query_rules_only
from core.io_utils import df_to_excel_bytes
import core.modules.overview as ov

def render_concordance_view():
    st.header("Concordance (KWIC)")
    
    # Guidelines Layout using shared component
    from ui_streamlit.components.guidelines import render_guidelines
    col_main = render_guidelines("Concordance")

    with col_main:


            corpus_path = get_state('current_corpus_path')
            corpus_name = get_state('current_corpus_name', 'Corpus')

            if not corpus_path:
                st.warning("Please load a corpus first.")
                return

            # Initialize XML restriction variables to prevent NameError in NL search modes
            xml_where = ""
            xml_params = []
            xml_where_1 = ""
            xml_params_1 = []
            xml_where_2 = ""
            xml_params_2 = []

            # Deferred execution flag for NL modes (query runs AFTER XML filters are rendered)
            _deferred_nl_query = None

            search_term_1 = get_state('kwic_search_term', '')
            search_term_2 = None
            results = st.session_state.get('last_kwic_results_primary')
            cluster_results = st.session_state.get('last_kwic_results_cluster')

            tab_simple, tab_advanced = st.tabs(["Simple", "Advanced"])

            with tab_simple:
                c_s1, c_s2, c_s3 = st.columns([2, 1, 1])
                with c_s1:
                    search_term_simple = st.text_input("Node Word(s)", value=get_state('kwic_search_term', ''), key="kwic_input_simple", help="Search word or phrase")
                with c_s2:
                    limit_simple_input = st.number_input("Max Lines", 10, 50000, 500, step=50, key="kwic_limit_simple")
                with c_s3:
                    st.write("")
                    fetch_all_simple = st.checkbox("Retrieve All (Max 10k)", value=get_state('kwic_fetch_all_simple', False), key="kwic_all_simple_cb")
                
                limit_simple = 10000 if fetch_all_simple else limit_simple_input

                if st.button("Generate Concordance", type="primary", key="btn_generate_simple", use_container_width=True):
                    set_state('kwic_search_term', search_term_simple)
                    st.session_state['last_kwic_results_cluster'] = None
                    set_state('kwic_show_meta', False)
                    set_state('kwic_show_pos', False)
                    set_state('kwic_show_lemma', False)
                    set_state('kwic_hide_symbols', False)
                    set_state('kwic_focus_sentence', False)
                    run_concordance_query(
                        identifier='primary',
                        path=corpus_path,
                        name=corpus_name,
                        query=search_term_simple,
                        left=5,
                        right=5,
                        limit=limit_simple,
                        coll_filter="",
                        xml_where="",
                        xml_params=[],
                        show_pos=False,
                        show_lemma=False,
                        source='simple'
                    )
                    st.rerun()

            with tab_advanced:
                # 1. Controls
                valid_modes = ["Standard", "Natural Language (Rule)", "Natural Language (AI)"]
                if get_state('kwic_search_mode') not in valid_modes:
                    set_state('kwic_search_mode', 'Standard')
                search_mode = st.radio("Search Mode", valid_modes, horizontal=True, key="kwic_search_mode")
                search_term = get_state('kwic_search_term', '')

                if search_mode == "Natural Language (Rule)":
                    st.markdown("### ⚡ Natural Language Search (Rule-Based)")
                    st.caption("Fast, deterministic parsing without AI. Supports: 'followed by', 'preceded by', 'before', 'after', and POS terms like 'noun', 'verb', 'adjective'.")

                    with st.expander("Search Controls", expanded=True):
                        col1, col2, col3, col4 = st.columns([2.5, 1.5, 1.2, 1.2])
                        with col1:
                             nl_query = st.text_input("Natural Language Query", value=get_state('kwic_nl_query_rule', ''), key="kwic_nl_input_rule", help="e.g. any word followed by 'adjective'")
                        with col2:
                             window_size = st.slider("Context Window", 1, 20, 5, key="kwic_window_rule")
                        with col3:
                             limit_rule_input = st.number_input("Max Lines", 10, 50000, 500, step=50, key="kwic_limit_rule")
                        with col4:
                             st.write("")
                             fetch_all_rule = st.checkbox("Retrieve All (Max 10k)", value=get_state('kwic_fetch_all_rule', False), key="kwic_all_rule_cb")

                        limit = 10000 if fetch_all_rule else limit_rule_input

                        # Annotation Tag Guide Popover
                        from ui_streamlit.components.pos_help import render_annotation_help_button
                        render_annotation_help_button(corpus_path, "concordance_rule")

                        # Advanced Filters
                        c_adv1, c_adv2 = st.columns(2)
                        with c_adv1:
                            coll_filter_input = st.text_input("Filter by Collocate (NL/Regex)", help="e.g. 'noun' or 'very'", key="kwic_coll_rule")
                        with c_adv2:
                            sort_order = st.radio("Sort By", ["Random (Default)", "Node", "Left Context", "Right Context"], horizontal=True, key="kwic_sort_rule")
                            # Display Control Row 1
                            c_r1_1, c_r1_2, c_r1_3 = st.columns(3)
                            with c_r1_1:
                                show_pos = st.checkbox("Show POS", value=get_state('kwic_show_pos', False), key="kwic_show_pos_rule")
                            with c_r1_2:
                                show_lemma = st.checkbox("Show Lemma", value=get_state('kwic_show_lemma', False), key="kwic_show_lemma_rule")
                            with c_r1_3:
                                show_meta = st.checkbox("Show Metadata", value=get_state('kwic_show_meta', False), key="kwic_show_meta_rule")

                            # Display Control Row 2
                            c_r2_1, c_r2_2, c_r2_3 = st.columns(3)
                            with c_r2_1:
                                hide_symbols = st.checkbox("Hide symbol token match", value=get_state('kwic_hide_symbols', False), key="kwic_hide_symbols_rule")
                            with c_r2_2:
                                focus_sentence = st.checkbox("Focus sentence", value=get_state('kwic_focus_sentence', False), key="kwic_focus_sentence_rule", help="Only preserve the exact sentence containing the keyword in context")
                            with c_r2_3:
                                show_duplicates = st.checkbox("Show duplicate concordance lines", value=get_state('kwic_show_duplicates', False), key="kwic_show_duplicates_rule", help="Show all occurrences even if they are identical sentences")

                            set_state('kwic_show_pos', show_pos)
                            set_state('kwic_show_lemma', show_lemma)
                            set_state('kwic_show_meta', show_meta)
                            set_state('kwic_hide_symbols', hide_symbols)
                            set_state('kwic_focus_sentence', focus_sentence)
                            set_state('kwic_show_duplicates', show_duplicates)

                            wrap_mode = st.checkbox("Wrap Text", value=get_state('kwic_wrap_mode', True), key="kwic_wrap_mode_rule", help="Enable to prevent text overlap by wrapping content to multiple lines")
                            set_state('kwic_wrap_mode', wrap_mode)

                    col_r1, col_r2 = st.columns([1, 4])
                    with col_r1:
                        analyze_btn = st.button("Search (Rule-Based)", type="primary")

                    if analyze_btn:
                        if not nl_query:
                            st.warning("Please enter a query.")
                        else:
                            set_state('kwic_nl_query_rule', nl_query)

                            # 1. Parse Main Query
                            pos_defs = ov.get_pos_definitions(corpus_path) or {}
                            reverse_pos_map = {v.lower(): k for k, v in pos_defs.items() if v}

                            params, err = parse_nl_query_rules_only(nl_query, "concordance", reverse_pos_map=reverse_pos_map)

                            # 2. Parse Collocate Filter (treat as query fragment)
                            coll_filter_parsed = ""
                            if coll_filter_input:
                                 c_params, c_err = parse_nl_query_rules_only(coll_filter_input, "concordance", reverse_pos_map=reverse_pos_map)
                                 if c_params:
                                     coll_filter_parsed = c_params.get('query', '')

                            if params:
                                query = params.get('query', '')
                                set_state('kwic_search_term', query)

                                # Use UI slider for window, ignoring parser default for consistency
                                set_state('kwic_window', window_size)

                                st.success(f"✓ Executing search for '{query}'...")
                                if coll_filter_parsed:
                                    st.info(f"   + Collocate Filter: '{coll_filter_parsed}'")

                                # Defer execution until after XML filters are rendered below
                                _deferred_nl_query = {
                                    'query': query, 'window': window_size, 'limit': limit,
                                    'coll_filter': coll_filter_parsed, 'show_pos': show_pos, 'show_lemma': show_lemma
                                }
                            else:
                                st.error(f"Error parsing query: {err}")

                if search_mode == "Natural Language (AI)":
                    st.markdown("### 🧠 Natural Language Search")
                    nl_query = st.text_area("Describe your concordance query", height=70, placeholder="e.g. Find examples of 'make' followed by a noun")
                    
                    # Annotation Tag Guide Popover
                    from ui_streamlit.components.pos_help import render_annotation_help_button
                    render_annotation_help_button(corpus_path, "concordance_ai")

                    # Display & Search Options for AI Mode
                    with st.expander("Search & Display Options", expanded=True):
                        c_ai1, c_ai2, c_ai3 = st.columns([2, 1.2, 1.2])
                        with c_ai1:
                            window_size_ai = st.slider("Context Window", 1, 20, get_state('kwic_window', 5), key="kwic_window_ai")
                        with c_ai2:
                            limit_ai_input = st.number_input("Max Lines", 10, 50000, 500, step=50, key="kwic_limit_ai")
                        with c_ai3:
                            st.write("")
                            fetch_all_ai = st.checkbox("Retrieve All (Max 10k)", value=get_state('kwic_fetch_all_ai', False), key="kwic_all_ai_cb")

                        limit_ai = 10000 if fetch_all_ai else limit_ai_input
                            
                        c_ai3, c_ai4 = st.columns(2)
                        with c_ai3:
                            wrap_mode = st.checkbox("Wrap Text", value=get_state('kwic_wrap_mode', True), key="kwic_wrap_mode_ai")
                            set_state('kwic_wrap_mode', wrap_mode)
                        with c_ai4:
                            focus_sentence = st.checkbox("Focus sentence", value=get_state('kwic_focus_sentence', False), key="kwic_focus_sentence_ai", help="Only preserve the exact sentence containing the keyword in context")
                            set_state('kwic_focus_sentence', focus_sentence)

                    col_ai1, col_ai2 = st.columns([1, 4])
                    with col_ai1:
                        analyze_btn = st.button("Search with AI", type="primary")

                    if analyze_btn:
                        if not nl_query:
                            st.warning("Please enter a query.")
                        else:
                            with st.spinner("AI is determining search parameters..."):
                                # Fetch user definitions if available
                                pos_defs = ov.get_pos_definitions(corpus_path) or {}
                                lang = ov.get_corpus_language(corpus_path)

                                # Safe-pass language context via pos_defs to avoid stale-cache TypeErrors
                                if lang:
                                    pos_defs['__language_context__'] = lang

                                params, err = parse_nl_query(
                                    nl_query, 
                                    "concordance",
                                    ai_provider=get_state('ai_provider'),
                                    gemini_api_key=get_state('gemini_api_key'),
                                    ollama_url=get_state('ollama_url'),
                                    ollama_model=get_state('ai_model'),
                                    pos_definitions=pos_defs
                                )

                            if params:
                                # Map and update state
                                # Query
                                query = params.get('query', '')
                                set_state('kwic_search_term', query)

                                # Window/Limit defaults
                                try:
                                    win = int(params.get('window', window_size_ai))
                                except (ValueError, TypeError):
                                    win = window_size_ai
                                set_state('kwic_window', win)
                                set_state('kwic_limit', limit_ai)

                                # Sort
                                sort = params.get('sort_order', 'Node')
                                if 'left' in sort.lower(): set_state('kwic_sort_col_primary', 'Left') # We need to check exact key usage
                                elif 'right' in sort.lower(): set_state('kwic_sort_col_primary', 'Right')

                                st.success(f"✓ Executing search for '{query}'...")

                                # Defer execution until after XML filters are rendered below
                                _deferred_nl_query = {
                                    'query': query, 'window': win, 'limit': limit_ai,
                                    'coll_filter': '', 'show_pos': False, 'show_lemma': False
                                }
                            else:
                                st.error(f"Could not parse query: {err}")

                if search_mode == "Standard":
                    with st.expander("Search Controls", expanded=True):
                        col1, col2, col3, col4 = st.columns([2.5, 1.5, 1.2, 1.2])
                        with col1:
                             search_term = st.text_input("Node Word(s)", value=get_state('kwic_search_term'), key="kwic_input", help="Use * for wildcards (e.g. run*), _TAG for POS (e.g. _NN), [lemma] for lemma, token_POS (e.g. light_V*), or <TAG> for XML tags (e.g. <PN type=\"human\">)")
                        with col2:
                             window_size = st.slider("Context Window", 1, 20, 5, key="kwic_window")
                        with col3:
                             limit_standard_input = st.number_input("Max Lines", 10, 50000, 500, step=50, key="kwic_limit")
                        with col4:
                             st.write("")
                             fetch_all_standard = st.checkbox("Retrieve All (Max 10k)", value=get_state('kwic_fetch_all_standard', False), key="kwic_all_standard_cb")

                        limit = 10000 if fetch_all_standard else limit_standard_input

                        # Annotation Tag Guide Popover
                        from ui_streamlit.components.pos_help import render_annotation_help_button
                        render_annotation_help_button(corpus_path, "concordance_standard")

                        # Advanced Filters
                        c_adv1, c_adv2 = st.columns(2)
                        with c_adv1:
                            coll_filter = st.text_input("Filter by Collocate (Regex)", help="Show only lines containing this pattern")
                        with c_adv2:
                            sort_order = st.radio("Sort By", ["Random (Default)", "Node", "Left Context", "Right Context"], horizontal=True, key="kwic_sort_standard")
                            
                            # Display Control Row 1
                            c_st1, c_st2, c_st3 = st.columns(3)
                            with c_st1:
                                show_pos = st.checkbox("Show POS", value=get_state('kwic_show_pos', False), key="kwic_show_pos_cb")
                            with c_st2:
                                show_lemma = st.checkbox("Show Lemma", value=get_state('kwic_show_lemma', False), key="kwic_show_lemma_cb")
                            with c_st3:
                                show_meta = st.checkbox("Show Metadata", value=get_state('kwic_show_meta', False), key="kwic_show_meta_cb")

                            # Display Control Row 2
                            c_st4, c_st5, c_st6 = st.columns(3)
                            with c_st4:
                                hide_symbols = st.checkbox("Hide symbol token match", value=get_state('kwic_hide_symbols', False), key="kwic_hide_symbols_cb")
                            with c_st5:
                                focus_sentence = st.checkbox("Focus sentence", value=get_state('kwic_focus_sentence', False), key="kwic_focus_sentence_cb", help="Only preserve the exact sentence containing the keyword in context")
                            with c_st6:
                                show_duplicates = st.checkbox("Show duplicate concordance lines", value=get_state('kwic_show_duplicates', False), key="kwic_show_duplicates_cb", help="Show all occurrences even if they are identical sentences")

                            set_state('kwic_show_pos', show_pos)
                            set_state('kwic_show_lemma', show_lemma)
                            set_state('kwic_show_meta', show_meta)
                            set_state('kwic_hide_symbols', hide_symbols)
                            set_state('kwic_focus_sentence', focus_sentence)
                            set_state('kwic_show_duplicates', show_duplicates)

                            wrap_mode = st.checkbox("Wrap Text", value=get_state('kwic_wrap_mode', True), key="kwic_wrap_mode_cb", help="Enable to prevent text overlap by wrapping content to multiple lines")
                            set_state('kwic_wrap_mode', wrap_mode)

                # --- XML Restriction Filters ---
                comp_mode = get_state('comparison_mode', False)
                comp_path = get_state('comp_corpus_path')
                comp_name = get_state('comp_corpus_name')

                if not comp_mode:
                    xml_filters = render_xml_restriction_filters(corpus_path, "concordance", corpus_name=corpus_name)
                    xml_where, xml_params = apply_xml_restrictions(xml_filters)
                    
                    forced_where = get_state('concordance_forced_xml_where', '')
                    if forced_where:
                        xml_where += forced_where
                        # Clear it so it doesn't persist across arbitrary searches
                        set_state('concordance_forced_xml_where', '')
                else:
                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        xml_filters_1 = render_xml_restriction_filters(corpus_path, "concordance_c1", corpus_name=corpus_name)
                        xml_where_1, xml_params_1 = apply_xml_restrictions(xml_filters_1)
                    with col_f2:
                        if comp_path:
                            xml_filters_2 = render_xml_restriction_filters(comp_path, "concordance_c2", corpus_name=comp_name)
                            xml_where_2, xml_params_2 = apply_xml_restrictions(xml_filters_2)
                        else:
                            xml_where_2, xml_params_2 = "", []

                search_term_1 = ""
                search_term_2 = None

                if search_mode == "Standard":
                    if comp_mode:
                        st.markdown("##### Comparison Search Inputs")
                        c_search_1, c_search_2 = st.columns(2)
                        with c_search_1:
                            search_term_1 = st.text_input(f"Search {get_state('current_corpus_name', 'Primary')}", value=get_state('kwic_search_term'), key="kwic_input_1")
                        with c_search_2:
                            search_term_2 = st.text_input(f"Search {comp_name if comp_name else 'Secondary'}", value=get_state('kwic_search_term_2', ''), key="kwic_input_2")
                    else:
                        search_term_1 = search_term # Use the main input
                        search_term_2 = None

                    if not comp_mode:
                        has_search_term = bool(search_term_1 and search_term_1.strip())
                        _active_filters = xml_filters or {}
                        _has_categorical = any(
                            f.get('type') == 'list' and f.get('values') for f in _active_filters.values()
                        ) if _active_filters else False
                        
                        # Generate a descriptive name for the current restriction
                        _restriction_parts = []
                        for _k, _f in _active_filters.items():
                            if _f['type'] == 'list' and _f['values']:
                                _restriction_parts.append(", ".join([str(v) for v in _f['values']]))
                        _display_name = " | ".join(_restriction_parts) if _restriction_parts else corpus_name

                        cluster_examples_limit = st.radio(
                            "Examples per Cluster",
                            options=[5, 10, 15, 20, 25, "All"],
                            index=0,
                            horizontal=True,
                            key="cluster_examples_limit_select"
                        )
                        btn_col1, btn_col2 = st.columns([1, 1])
                        with btn_col1:
                            if st.button("Generate Concordance Lines", type="primary", use_container_width=True):
                                # Clear any previous cluster results when re-generating
                                st.session_state['last_kwic_results_cluster'] = None
                                set_state('kwic_search_term', search_term_1)
                                run_concordance_query('primary', corpus_path, _display_name, search_term_1, window_size, window_size, limit, coll_filter, xml_where, xml_params, show_pos, show_lemma)
                        with btn_col2:
                            cluster_btn_help = (
                                "Cluster concordance by selected metadata categories"
                                if has_search_term and _has_categorical
                                else "Enter a search term and select categorical metadata filters to enable clustering"
                            )
                            cluster_btn_disabled = not (has_search_term and _has_categorical)
                            if st.button(
                                "🧩 Cluster Mode",
                                type="secondary",
                                disabled=cluster_btn_disabled,
                                help=cluster_btn_help,
                                use_container_width=True,
                                key="btn_cluster_mode"
                            ):
                                _cluster_limit = st.session_state.get('kwic_limit', 100)
                                _active_search_term = get_state('kwic_search_term', search_term)
                                limit_val = 999999 if cluster_examples_limit == "All" else int(cluster_examples_limit)
                                run_cluster_concordance_query(
                                    corpus_path, corpus_name,
                                    _active_search_term,
                                    window_size,
                                    limit_val,  # samples per cluster
                                    _active_filters,
                                    coll_filter=coll_filter,
                                    show_pos=show_pos,
                                    show_lemma=show_lemma,
                                    hide_symbols=hide_symbols,
                                    show_duplicates=show_duplicates
                                )
                    else:
                        if st.button("Generate Comparison Concordance", type="primary"):
                            set_state('kwic_search_term', search_term_1)
                            set_state('kwic_search_term_2', search_term_2) # New state for query 2

                            run_concordance_query('primary', corpus_path, corpus_name, search_term_1, window_size, window_size, limit, coll_filter, xml_where_1, xml_params_1, show_pos, show_lemma)
                            if comp_path:
                                run_concordance_query('secondary', comp_path, comp_name, search_term_2, window_size, window_size, limit, coll_filter, xml_where_2, xml_params_2, show_pos, show_lemma)
                else:
                    # For NL mode
                    search_term_1 = get_state('kwic_search_term', '')
                    search_term_2 = None

                # --- Deferred NL Query Execution (runs AFTER xml_where/xml_params are set) ---
                if _deferred_nl_query is not None:
                    _dq = _deferred_nl_query
                    run_concordance_query(
                        'primary', corpus_path, corpus_name,
                        _dq['query'], _dq['window'], _dq['window'],
                        _dq['limit'], _dq['coll_filter'],
                        xml_where, xml_params,
                        _dq['show_pos'], _dq['show_lemma']
                    )

                # 2. Annotation Resume (High visibility at the top)
                col_res1, col_res2 = st.columns([1, 3])
                with col_res1:
                    if st.button("📁 Continue Annotation", help="Resume annotation by uploading a saved file", use_container_width=True):
                        set_state('show_ann_upload', True)

                if get_state('show_ann_upload'):
                    with st.container(border=True):
                        st.markdown("##### Resume Annotation Session")
                        uploaded_file = st.file_uploader("Upload Annotation JSON", type="json", key="ann_uploader_main")
                        if uploaded_file:
                            import json
                            try:
                                data = json.load(uploaded_file)
                                ann_path = data.get('corpus_path')
                                ann_term = data.get('search_term')

                                if ann_path and ann_term:
                                    # Migration: Ensure all annotations are lists
                                    raw_ann = data.get('annotations', {})
                                    processed_ann = {}
                                    for k, v in raw_ann.items():
                                        if isinstance(v, list):
                                            processed_ann[k] = v
                                        else:
                                            processed_ann[k] = [v] # Wrap old single pair in list

                                    st.session_state['kwic_annotations'] = processed_ann
                                    st.success(f"✅ Loaded annotations for '{ann_term}'")

                                    # Logic to determine if it's the SAME corpus logically, even if path changed
                                    raw_source_name = data.get('corpus_name', os.path.basename(ann_path))

                                    def clean_name(n, p=None):
                                        if not n: return "Unknown"
                                        n = n.replace('.duckdb', '')
                                        if n.startswith('corpus_') and len(n) > 20 and p:
                                            parts = p.replace('\\', '/').split('/')
                                            for part in reversed(parts[:-1]):
                                                if part.lower() not in ('temp', 'corpora', 'cortex', 'documents', 'users'):
                                                    return f"{part} (Uploaded)"
                                            return "Uploaded Corpus"
                                        return n

                                    source_display = clean_name(raw_source_name, ann_path)
                                    current_display = clean_name(corpus_name, corpus_path)

                                    # Auto-trigger search if it looks like the right corpus and query
                                    is_match = (ann_path == corpus_path) or (source_display == current_display)

                                    if is_match and ann_term == search_term_1:
                                        st.info("🔄 Re-generating concordance lines...")
                                        run_concordance_query('primary', corpus_path, corpus_name, ann_term, 5, 5, 100, "", "", [])
                                        set_state('show_ann_upload', False)
                                        st.rerun()
                                    else:
                                        # Show mismatch UI with Force Load option
                                        st.error("🚫 **Annotation Mismatch**")
                                        st.write(f"This annotation file is linked to a different corpus or search query.")

                                        col_war1, col_war2 = st.columns(2)
                                        with col_war1:
                                            st.markdown(f"**Required (from file):**\n- 📂 Corpus: `{source_display}`\n- 🔍 Query: `{ann_term}`")
                                        with col_war2:
                                            q_status = "✅ Match" if ann_term == search_term_1 else f"❌ `{search_term_1}`"
                                            st.markdown(f"**Current (Active):**\n- 📂 Corpus: `{current_display}`\n- 🔍 Query: {q_status}")

                                        st.info("💡 If you are sure this is the correct data, you can force the load below.")
                                        if st.button("⚠️ Force Load Annotations Anyway", type="secondary"):
                                            run_concordance_query('primary', corpus_path, corpus_name, ann_term, 5, 5, 100, "", "", [])
                                            set_state('show_ann_upload', False)
                                            st.rerun()
                                else:
                                    st.error("❌ Invalid annotation file format.")
                            except Exception as e:
                                st.error(f"Error loading file: {e}")
                        if st.button("Close"):
                            set_state('show_ann_upload', False)
                            st.rerun()

            # --- Results Display / Annotation Resume ---
            results = st.session_state.get('last_kwic_results_primary')
            cluster_results = st.session_state.get('last_kwic_results_cluster')

            # Initialize multi-annotation state if missing
            if 'kwic_annotations' not in st.session_state:
                st.session_state['kwic_annotations'] = {}
            kwic_annotations = st.session_state['kwic_annotations']

            # Annotation Mode Toggle
            col_ann1, col_ann2, col_ann3 = st.columns([1, 1, 2])
            with col_ann1:
                ann_mode = st.toggle("✍️ Annotation Mode", value=get_state('kwic_ann_mode', False), key="kwic_ann_mode_toggle")
                set_state('kwic_ann_mode', ann_mode)

            with col_ann2:
                if ann_mode:
                    if st.button("🏛️ Apply to Session", help="Add these annotations to the active working corpus for all tabs"):
                        set_state('show_db_save_confirm', True)

            if get_state('show_db_save_confirm'):
                with st.container(border=True):
                    st.info("ℹ️ **Apply to Active Session**")
                    st.write("This will add these labels to the current working corpus in this session. They will be visible in the Overview and Restricted Search tabs.")
                    st.write("⚠️ *Note: These changes are not saved to the source XML. If you re-upload the corpus, you will need to restore your annotations from a backup file.*")
                    st.checkbox("I understand and want to proceed", key="db_save_confirm_check")

                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("🚀 Apply Labels", type="primary", disabled=not st.session_state.get('db_save_confirm_check')):
                            import importlib
                            import core.modules.concordance as cm
                            importlib.reload(cm) 
                            if hasattr(cm, 'persist_annotations_to_db'):
                                success, msg = cm.persist_annotations_to_db(results['path'], st.session_state.get('kwic_annotations', {}))
                            else:
                                success, msg = False, "Internal Error: Persistence function not found in module after reload."
                            if success:
                                st.success(f"✅ {msg}")
                                set_state('show_db_save_confirm', False)
                                # Reset some caches to make sure other modules see the change
                                st.cache_data.clear() 
                            else:
                                st.error(f"❌ {msg}")
                    with c2:
                        if st.button("Cancel"):
                            set_state('show_db_save_confirm', False)
                            st.rerun()

            # Case A: Cluster Mode Results -- 3 TABS
            if cluster_results:
                _cluster_search_term = get_state('kwic_search_term', search_term_1 if 'search_term_1' in dir() else '')
                st.markdown(f"## 🧩 Cluster Concordance: *{_cluster_search_term}*")
                st.caption(f"**{len(cluster_results)} cluster(s)** generated from selected metadata filters.")

                _first_res = next(iter(cluster_results.values()))
                _has_coll = bool(_first_res.get('coll_filter'))

                _tab_lines, _tab_tables, _tab_viz = st.tabs(["📝 Concordance Lines", "📊 Tables", "📈 Visualisation"])

                with _tab_lines:
                    for cluster_name, res in cluster_results.items():
                        _n = len(res.get('rows', []))
                        _total = res.get('total', _n)
                        st.markdown(f"### 📦 **{cluster_name}** — {_n} sample(s) of {_total:,} total")
                        render_concordance_column(res, _cluster_search_term, key_suffix=f"cluster_{cluster_name}")
                        st.markdown("---")

                with _tab_tables:
                    st.markdown("### 📊 Clustered Result Analysis")
                    render_aggregate_cluster_summary(cluster_results)
                    if _has_coll:
                        render_collocate_filter_tables(cluster_results)

                with _tab_viz:
                    st.markdown("### 📈 Visualisation")
                    render_visualisation_tab(cluster_results, has_coll_filter=_has_coll)

            # Case B: Standard Results -- 3 TABS
            elif results:
                if not comp_mode:
                    _active_filters = xml_filters or {}
                    _active_keys = []
                    _active_values = []
                    for k, f in _active_filters.items():
                        if f.get('type') == 'list' and f.get('values'):
                            _active_keys.append(k)
                            _active_values.append(f['values'])

                    _has_multiple = _active_values and len(list(itertools.product(*_active_values))) > 1
                    _coll = results.get('coll_filter', "")

                    # Build comparative data (needed by both Tables and Viz tabs)
                    _comparative_data = {}
                    if _has_multiple:
                        with st.spinner("Analyzing restrictions..."):
                            _combinations = list(itertools.product(*_active_values))
                            for combo in _combinations:
                                combo_name = " | ".join([str(v) for v in combo])
                                combo_filters = {}
                                for j, val in enumerate(combo):
                                    combo_filters[_active_keys[j]] = {'type': 'list', 'values': [val]}
                                where, params = apply_xml_restrictions(combo_filters)
                                _, total, _, _, _, breakdown = cached_generate_kwic(
                                    db_path=results['path'],
                                    query=results['search_term'],
                                    left=results.get('left', 5),
                                    right=results.get('right', 5),
                                    corpus_name=results['name'],
                                    pattern_collocate_input=_coll,
                                    pattern_window=results.get('left', 5),
                                    limit=1,
                                    xml_where_clause=where,
                                    xml_params=tuple(params),
                                    show_pos=results.get('show_pos', False),
                                    show_lemma=results.get('show_lemma', False),
                                    hide_symbols=results.get('hide_symbols', False),
                                    focus_sentence=results.get('focus_sentence', False),
                                    show_duplicates=results.get('show_duplicates', False)
                                )
                                col_counts = {}
                                if _coll:
                                    import core.modules.concordance as cm
                                    col_counts = cm.get_collocate_frequency_list(
                                        db_path=results['path'],
                                        query=results['search_term'],
                                        collocate_filter=_coll,
                                        window=results.get('left', 5),
                                        xml_where=where,
                                        xml_params=tuple(params)
                                    )
                                _comparative_data[combo_name] = {
                                    'breakdown': breakdown,
                                    'total': total,
                                    'xml_where': where,
                                    'xml_params': params,
                                    'path': results['path'],
                                    'coll_filter': _coll,
                                    'collocate_counts': col_counts
                                }
                    else:
                        _res_name = results.get('name', 'Whole Corpus')
                        if _coll and 'collocate_counts' not in results:
                            import core.modules.concordance as cm
                            results['collocate_counts'] = cm.get_collocate_frequency_list(
                                db_path=results['path'],
                                query=results['search_term'],
                                collocate_filter=_coll,
                                window=results.get('left', 5),
                                xml_where=results.get('xml_where', ""),
                                xml_params=tuple(results.get('xml_params', []))
                            )
                        _comparative_data = {_res_name: results}

                    # ---- 3 TABS ----
                    _tab_lines, _tab_tables, _tab_viz = st.tabs(["📝 Concordance Lines", "📊 Tables", "📈 Visualisation"])

                    with _tab_lines:
                        render_concordance_column(results, search_term_1)

                    with _tab_tables:
                        if _has_multiple:
                            st.markdown("### 📊 Comparative Result Analysis (By Restrictions)")
                        else:
                            st.markdown("### 📊 Search Analysis")
                        render_aggregate_cluster_summary(_comparative_data)
                        if _coll:
                            render_collocate_filter_tables(_comparative_data)

                    with _tab_viz:
                        st.markdown("### 📈 Visualisation")
                        render_visualisation_tab(_comparative_data, has_coll_filter=bool(_coll))

                else:
                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        st.subheader(f"Primary: {corpus_name}")
                        render_concordance_column(results, search_term_1, key_suffix="c1")
                    with col_c2:
                        st.subheader(f"Comparison: {get_state('comp_corpus_name', 'Comparison')}")
                        comp_path = get_state('comp_corpus_path')
                        if not comp_path:
                            st.info("Load a comparison corpus in sidebar.")
                        else:
                            results_2 = st.session_state.get('last_kwic_results_secondary')
                            if results_2:
                                render_concordance_column(results_2, get_state('kwic_search_term_2', ''), key_suffix="c2")

@notify_timing("Cluster Concordance generated")
def run_cluster_concordance_query(path, name, query, window, limit, filters, coll_filter="", show_pos=False, show_lemma=False, hide_symbols=False, show_duplicates=False):
    # 1. Prepare Cartesian Product of list-based filters
    keys = []
    value_lists = []
    for k, f in filters.items():
        if f['type'] == 'list' and f['values']:
            keys.append(k)
            value_lists.append(f['values'])
        elif f['type'] == 'range':
            st.error(f"Attribute '{k}' is a numeric range and cannot be used for clustering. Please select categorical attributes (like domain or sentiment) instead.")
            continue
    
    if not value_lists:
        st.error("No categorical attributes selected for clustering. Please select at least one attribute and multiple values in the Restricted Search.")
        return

    combinations = list(itertools.product(*value_lists))
    
    cluster_results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, combo in enumerate(combinations):
        # Build filter for this specific combo
        current_filters = {}
        combo_name_parts = []
        for j, val in enumerate(combo):
            current_filters[keys[j]] = {'type': 'list', 'values': [val]}
            combo_name_parts.append(str(val))
        
        combo_name = " | ".join(combo_name_parts)
        status_text.text(f"Processing cluster {i+1}/{len(combinations)}: {combo_name}")
        
        # Convert to where clause
        where, params = apply_xml_restrictions(current_filters)
        
        # Run query with random sampling as requested
        rows, total, _, _, _, breakdown = cached_generate_kwic(
            db_path=path,
            query=query,
            left=window,
            right=window,
            corpus_name=name,
            pattern_collocate_input=coll_filter,
            pattern_window=window,
            limit=limit,
            do_random_sample=True,
            xml_where_clause=where,
            xml_params=tuple(params),
            show_pos=show_pos,
            show_lemma=show_lemma,
            hide_symbols=hide_symbols,
            focus_sentence=get_state('kwic_focus_sentence', False),
            show_duplicates=show_duplicates
        )
        
        if rows:
            col_counts = {}
            if coll_filter:
                import core.modules.concordance as cm
                col_counts = cm.get_collocate_frequency_list(
                    db_path=path,
                    query=query,
                    collocate_filter=coll_filter,
                    window=window,
                    xml_where=where,
                    xml_params=tuple(params)
                )
            cluster_results[combo_name] = {
                'rows': rows,
                'total': total,
                'path': path,
                'name': combo_name,
                'search_term': query,
                'window': window,
                'xml_where': where,
                'xml_params': params,
                'breakdown': breakdown, 
                'coll_filter': coll_filter,
                'show_pos': show_pos,
                'show_lemma': show_lemma,
                'hide_symbols': hide_symbols,
                'focus_sentence': get_state('kwic_focus_sentence', False),
                'show_duplicates': show_duplicates,
                'collocate_counts': col_counts
            }
        
        progress_bar.progress((i + 1) / len(combinations))
    
    progress_bar.empty()
    status_text.empty()
    st.session_state['last_kwic_results_cluster'] = cluster_results
    st.session_state['last_kwic_results_primary'] = None # Clear primary to focus on cluster
    if not cluster_results:
        st.warning("No results found for any of the generated clusters.")
    else:
        st.success(f"Generated {len(cluster_results)} clusters.")
    st.rerun()

def render_aggregate_cluster_summary(cluster_results):
    """Renders Absolute and Relative frequency tables for all clusters."""
    if not cluster_results:
        st.info("No cluster results available for summary.")
        return

    # 1. Collect all unique node forms across all clusters
    all_node_forms = set()
    for res in cluster_results.values():
        br = res.get('breakdown')
        if br is not None and not br.empty:
            # Check for column name variations
            target_col = 'Token Form' if 'Token Form' in br.columns else br.columns[0]
            all_node_forms.update(br[target_col].tolist())
    
    if not all_node_forms:
        st.warning("⚠️ No node word breakdown data found. Summary tables cannot be generated.")
        return

    node_forms = sorted(list(all_node_forms))
    cluster_names = list(cluster_results.keys())

    # 2. Build Absolute Frequency Data
    abs_data = []
    for form in node_forms:
        row = {"Query result": form}
        for name in cluster_names:
            br = cluster_results[name].get('breakdown')
            if br is not None and not br.empty:
                # Find the 'Token Form' column
                t_col = 'Token Form' if 'Token Form' in br.columns else br.columns[0]
                # Find the 'Absolute Frequency' column
                a_col = 'Absolute Frequency' if 'Absolute Frequency' in br.columns else (br.columns[1] if len(br.columns)>1 else None)
                
                match = br[br[t_col] == form]
                if not match.empty and a_col:
                    row[name] = int(match[a_col].iloc[0])
                else:
                    row[name] = 0
            else:
                row[name] = 0
        abs_data.append(row)
    
    df_abs = pd.DataFrame(abs_data)

    # 3. Build Relative Frequency Data
    rel_data = []
    for form in node_forms:
        row = {"Query result": form}
        for name in cluster_names:
            br = cluster_results[name].get('breakdown')
            if br is not None and not br.empty:
                # Find the 'Token Form' column
                t_col = 'Token Form' if 'Token Form' in br.columns else br.columns[0]
                # Find the 'Relative Frequency' column
                r_col = 'Relative Frequency (per M)' if 'Relative Frequency (per M)' in br.columns else (br.columns[2] if len(br.columns)>2 else None)
                
                match = br[br[t_col] == form]
                if not match.empty and r_col:
                    row[name] = float(match[r_col].iloc[0])
                else:
                    row[name] = 0.0
            else:
                row[name] = 0.0
        rel_data.append(row)
    
    df_rel = pd.DataFrame(rel_data)

    st.markdown("#### Absolute Frequency")
    st.dataframe(df_abs, use_container_width=True, hide_index=True)

    st.markdown("#### Relative Frequency (PMW)")
    st.dataframe(df_rel, use_container_width=True, hide_index=True)

    return df_abs, df_rel, cluster_names

def render_collocate_filter_tables(cluster_results):
    """Renders Collocate Filter tables (Absolute and Relative)."""
    st.markdown("---")
    st.markdown("### 🔍 Collocate Filter Analysis")
    
    # 1. Aggregate Collocate Counts per cluster
    # cluster_colls: { cluster_name: { collocate_word: count } }
    cluster_colls = {}
    all_collocates = set()
    
    for cluster_name, res in cluster_results.items():
        # Optimization: Use pre-calculated collocate_counts if available (from standard mode comparative analysis)
        if 'collocate_counts' in res:
            counts = res['collocate_counts']
            for c in counts: all_collocates.add(c)
            cluster_colls[cluster_name] = counts
            continue

        counts = {}
        for row in res.get('rows', []):
            coll = row.get('Collocate')
            if coll:
                counts[coll] = counts.get(coll, 0) + 1
                all_collocates.add(coll)
        cluster_colls[cluster_name] = counts

    if not all_collocates:
        st.info("No collocates found matching the filter.")
        return

    sorted_collocates = sorted(list(all_collocates))
    cluster_names = list(cluster_results.keys())

    # 2. Build Absolute Frequency Table
    abs_list = []
    for coll in sorted_collocates:
        row = {"Collocate Filter": coll}
        found_in = []
        for name in cluster_names:
            count = cluster_colls[name].get(coll, 0)
            row[name] = count
            if count > 0:
                found_in.append(name)
        
        # Shared / Unique Logic
        if len(found_in) == len(cluster_names):
            row["Shared"] = "all"
            row["Unique"] = ""
        elif len(found_in) == 1:
            row["Shared"] = ""
            row["Unique"] = found_in[0]
        else:
            row["Shared"] = ", ".join(found_in)
            row["Unique"] = ""
        
        abs_list.append(row)
    
    df_abs = pd.DataFrame(abs_list)
    
    # 3. Build Relative Frequency Table
    # Need subcorpus sizes for each cluster
    rel_list = []
    for coll in sorted_collocates:
        row = {"Collocate Filter": coll}
        found_in = []
        for name in cluster_names:
            res = cluster_results[name]
            count = cluster_colls[name].get(coll, 0)
            
            # Fetch subcorpus size
            total_tokens = cached_get_subcorpus_size(res['path'], xml_where_clause=res['xml_where'], xml_params=tuple(res['xml_params']))
            rel_freq = (count / total_tokens) * 1_000_000 if total_tokens > 0 else 0
            
            row[name] = round(rel_freq, 2)
            if count > 0:
                found_in.append(name)
        
        # Shared / Unique Logic
        if len(found_in) == len(cluster_names):
            row["Shared"] = "all"
            row["Unique"] = ""
        elif len(found_in) == 1:
            row["Shared"] = ""
            row["Unique"] = found_in[0]
        else:
            row["Shared"] = ", ".join(found_in)
            row["Unique"] = ""
            
        rel_list.append(row)

    df_rel = pd.DataFrame(rel_list)

    st.markdown("#### Collocate Filter Table (Absolute Frequency)")
    st.dataframe(df_abs, use_container_width=True, hide_index=True)

    st.markdown("#### Collocate Filter Table (Relative Frequency)")
    st.dataframe(df_rel, use_container_width=True, hide_index=True)

    return df_abs, df_rel, cluster_names, cluster_colls


def render_visualisation_tab(cluster_results, has_coll_filter=False):
    """Renders visualisations organized in separate tabs: Bar Charts, Network, and Overlap Size Overview."""
    import plotly.graph_objects as go
    from ui_streamlit.views.concordance_network import render_concordance_network, render_concordance_overlap_overview

    vtab_chart, vtab_net, vtab_overlap = st.tabs([
        "📊 Frequency Bar Charts",
        "🕸️ Concordance Network",
        "🔀 Overlap Size Overview"
    ])

    with vtab_chart:
        cluster_names = list(cluster_results.keys())

        # ---- 1. Node word frequency chart ----
        all_node_forms = set()
        for res in cluster_results.values():
            br = res.get('breakdown')
            if br is not None and not br.empty:
                target_col = 'Token Form' if 'Token Form' in br.columns else br.columns[0]
                all_node_forms.update(br[target_col].tolist())
                
        node_forms = sorted(list(all_node_forms))
        
        node_totals = {name: cluster_results[name].get('total', 0) for name in cluster_names}

        if any(v > 0 for v in node_totals.values()):
            freq_type = st.radio("Frequency Metric for Node Word", ["Absolute Frequency", "Relative Frequency (PMW)"], horizontal=True, key="viz_node_freq_type")
            is_rel = (freq_type == "Relative Frequency (PMW)")

            fig_node = go.Figure()
            if node_forms:
                if len(cluster_names) == 1:
                    # Single corpus / no restrictions case: plot variations on Y-axis
                    name = cluster_names[0]
                    br = cluster_results[name].get('breakdown')
                    frequencies = []
                    for form in node_forms:
                        val = 0.0
                        if br is not None and not br.empty:
                            t_col = 'Token Form' if 'Token Form' in br.columns else br.columns[0]
                            if is_rel:
                                col = 'Relative Frequency (per M)' if 'Relative Frequency (per M)' in br.columns else (br.columns[2] if len(br.columns)>2 else None)
                            else:
                                col = 'Absolute Frequency' if 'Absolute Frequency' in br.columns else (br.columns[1] if len(br.columns)>1 else None)
                            
                            match = br[br[t_col] == form]
                            if not match.empty and col:
                                val = float(match[col].iloc[0]) if is_rel else int(match[col].iloc[0])
                        frequencies.append(val)
                        
                    text_labels = [f"{f:.2f}" if is_rel else str(int(f)) for f in frequencies]
                    fig_node.add_trace(go.Bar(
                        y=node_forms,
                        x=frequencies,
                        orientation='h',
                        text=text_labels,
                        textposition='outside',
                        marker_color='#FFEA00' # Highlight color
                    ))
                    fig_node.update_layout(
                        title=f'Node Word Variation {freq_type} ({name})',
                        xaxis_title=freq_type,
                        yaxis_title='Token Form',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=13),
                        height=max(300, 45 * len(node_forms) + 100)
                    )
                else:
                    # Multiple restrictions case: plot grouped bar chart
                    for form in node_forms:
                        x_vals = []
                        y_vals = []
                        for name in cluster_names:
                            br = cluster_results[name].get('breakdown')
                            val = 0.0
                            if br is not None and not br.empty:
                                t_col = 'Token Form' if 'Token Form' in br.columns else br.columns[0]
                                if is_rel:
                                    col = 'Relative Frequency (per M)' if 'Relative Frequency (per M)' in br.columns else (br.columns[2] if len(br.columns)>2 else None)
                                else:
                                    col = 'Absolute Frequency' if 'Absolute Frequency' in br.columns else (br.columns[1] if len(br.columns)>1 else None)
                                
                                match = br[br[t_col] == form]
                                if not match.empty and col:
                                    val = float(match[col].iloc[0]) if is_rel else int(match[col].iloc[0])
                            y_vals.append(name)
                            x_vals.append(val)
                            
                        text_labels = [f"{x:.2f}" if is_rel and x > 0 else (str(int(x)) if x > 0 else "") for x in x_vals]
                        fig_node.add_trace(go.Bar(
                            name=form,
                            y=y_vals,
                            x=x_vals,
                            orientation='h',
                            text=text_labels,
                            textposition='inside'
                        ))
                    fig_node.update_layout(
                        barmode='group',
                        title=f'Node Word Variation {freq_type} by Restriction',
                        xaxis_title=freq_type,
                        yaxis_title='Restriction',
                        legend_title='Token Form',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=13),
                        height=max(350, 80 * len(cluster_names) + 120)
                    )
            else:
                # Fallback to aggregates if no breakdown exists
                fig_node = go.Figure()
                for name in cluster_names:
                    val = node_totals[name]
                    fig_node.add_trace(go.Bar(
                        name=name,
                        y=[name],
                        x=[val],
                        orientation='h',
                        text=[str(val)],
                        textposition='outside'
                    ))
                fig_node.update_layout(
                    barmode='group',
                    title=f'Node Word {freq_type} by Restriction (Aggregate)',
                    xaxis_title=freq_type,
                    yaxis_title='Restriction',
                    legend_title='Restriction',
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=13),
                    height=max(300, 60 * len(cluster_names) + 100)
                )
            st.plotly_chart(fig_node, use_container_width=True)
        else:
            st.info("No node word frequency data available for visualisation.")

        # ---- 2. Collocate frequency — stacked horizontal bar per collocate ----
        if has_coll_filter:
            cluster_colls = {}
            all_collocates = set()
            for cluster_name, res in cluster_results.items():
                if 'collocate_counts' in res:
                    counts = res['collocate_counts']
                else:
                    counts = {}
                    for row in res.get('rows', []):
                        coll = row.get('Collocate')
                        if coll:
                            counts[coll] = counts.get(coll, 0) + 1
                cluster_colls[cluster_name] = counts
                all_collocates.update(counts.keys())

            if all_collocates:
                # Sort by total frequency descending, take top 30
                sorted_colls = sorted(
                    list(all_collocates),
                    key=lambda c: -sum(cluster_colls[n].get(c, 0) for n in cluster_names)
                )
                top_colls = sorted_colls[:30]
                top_colls_display = list(reversed(top_colls))

                fig_coll = go.Figure()
                for name in cluster_names:
                    vals = [cluster_colls[name].get(c, 0) for c in top_colls_display]
                    fig_coll.add_trace(go.Bar(
                        name=name,
                        y=top_colls_display,
                        x=vals,
                        orientation='h'
                    ))
                fig_coll.update_layout(
                    barmode='stack',
                    title='Collocate Absolute Frequency by Restriction (Top 30)',
                    xaxis_title='Frequency',
                    yaxis_title='Collocate',
                    legend_title='Restriction',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    height=max(450, 22 * len(top_colls_display) + 120)
                )
                st.plotly_chart(fig_coll, use_container_width=True)
            else:
                st.info("No collocate data found for visualisation.")

    with vtab_net:
        render_concordance_network(cluster_results, has_coll_filter=has_coll_filter)

    with vtab_overlap:
        render_concordance_overlap_overview(cluster_results, has_coll_filter=has_coll_filter)



def run_concordance_query(identifier, path, name, query, left, right, limit, coll_filter, xml_where, xml_params, show_pos=False, show_lemma=False, source='advanced'):
    if not query or not query.strip():
        st.warning(f"Please enter a Node Word(s) to search for in {name}.")
        return
        
    hide_symbols = get_state('kwic_hide_symbols', False)
    focus_sentence = get_state('kwic_focus_sentence', False)
    show_duplicates = get_state('kwic_show_duplicates', False)
    with st.spinner(f"Searching {name}..."):
        kwic_rows, total, raw_q, lit_freq, sent_ids, breakdown_df = cached_generate_kwic(
            db_path=path,
            query=query,
            left=left,
            right=right,
            corpus_name=name,
            pattern_collocate_input=coll_filter,
            pattern_window=left,
            limit=limit,
            xml_where_clause=xml_where,
            xml_params=tuple(xml_params) if xml_params else (),
            show_pos=show_pos,
            show_lemma=show_lemma,
            hide_symbols=hide_symbols,
            focus_sentence=focus_sentence,
            show_duplicates=show_duplicates
        )
        st.session_state[f'last_kwic_results_{identifier}'] = {
            'rows': kwic_rows,
            'total': total,
            'breakdown': breakdown_df,
            'name': name,
            'search_term': query,
            'xml_where': xml_where,
            'xml_params': xml_params,
            'path': path,
            'left': left,
            'right': right,
            'coll_filter': coll_filter,
            'source': source,
            'show_pos': show_pos,
            'show_lemma': show_lemma,
            'hide_symbols': hide_symbols,
            'focus_sentence': focus_sentence,
            'show_duplicates': show_duplicates
        }

def render_concordance_column(results, search_term, key_suffix=""):
     kwic_rows = results['rows']
     total = results['total']
     breakdown = results['breakdown']
     name = results['name']
     is_simple = (results.get('source') == 'simple')
     show_meta = False if is_simple else get_state('kwic_show_meta', False)
     
     # metrics (Target Query Summary table replacement)
     stats_key = 'corpus_stats' if key_suffix != "c2" else 'comp_corpus_stats'
     stats = get_state(stats_key)
     path = results.get('path')
     xml_where = results.get('xml_where', "")
     xml_params = results.get('xml_params', [])
     
     if xml_where and path:
         total_tokens = cached_get_subcorpus_size(path, xml_where_clause=xml_where, xml_params=xml_params)
     else:
         total_tokens = stats.get('total_tokens', 1) if stats else 1
         
     rel_freq = (total / total_tokens) * 1_000_000 if total_tokens > 0 else 0
     
     st.markdown(f"### Target Query Summary: {name}")
     summary_df = pd.DataFrame([{
         "Metric": "Absolute Frequency",
         "Value": f"{total:,}"
     }, {
         "Metric": "Relative Frequency (PMW)",
         "Value": f"{rel_freq:.2f}"
     }, {
         "Metric": "Samples Shown",
         "Value": f"{len(kwic_rows)}"
     }])
     st.table(summary_df)

     if total > len(kwic_rows):
         st.warning(
             f"⚠️ **Results Truncated**: Showing **{len(kwic_rows):,}** sample lines out of **{total:,}** total matches in the corpus. "
             f"To retrieve all {total:,} matches, check **'Retrieve All (No Limit)'** in Search Controls."
         )

     if not breakdown.empty:
         with st.expander("Token Breakdown Stats", expanded=True):
             st.dataframe(breakdown.head(20), use_container_width=True, hide_index=True)
     
     # Results Table
     if kwic_rows:
         # Clickable Headers for Sorting
         st.markdown("---")
         st.markdown("##### Click headers to sort results")
         h_col0, h_col1, h_col2, h_col3 = st.columns([2.5, 2.5, 2.5, 2.5])
         
         sort_col = get_state(f'kwic_sort_col_{key_suffix}', 'Random')
         sort_dir = get_state(f'kwic_sort_dir_{key_suffix}', 'asc')

         def set_sort(col):
             current = get_state(f'kwic_sort_col_{key_suffix}')
             if current == col:
                 new_dir = 'desc' if get_state(f'kwic_sort_dir_{key_suffix}') == 'asc' else 'asc'
                 set_state(f'kwic_sort_dir_{key_suffix}', new_dir)
             else:
                 set_state(f'kwic_sort_col_{key_suffix}', col)
                 set_state(f'kwic_sort_dir_{key_suffix}', 'asc')

         with h_col0:
             if st.button("🎲 Random (Default)", key=f"btn_sort_rnd_{key_suffix}", use_container_width=True):
                 set_sort('Random')
                 st.rerun()
         with h_col1:
             if st.button("🎯 Node", key=f"btn_sort_n_{key_suffix}", use_container_width=True):
                 set_sort('Node')
                 st.rerun()
         with h_col2:
             if st.button("⬅ Left Context", key=f"btn_sort_l_{key_suffix}", use_container_width=True):
                 set_sort('Left')
                 st.rerun()
         with h_col3:
             if st.button("Right Context ➡", key=f"btn_sort_r_{key_suffix}", use_container_width=True):
                 set_sort('Right')
                 st.rerun()

         # Perform Sorting
         # Clean tags for sorting
         def clean_html(raw_html):
             return re.sub(r'<[^>]*>', '', str(raw_html))

         if sort_col == 'Random':
             sorted_rows = list(kwic_rows)
             if sort_dir == 'desc':
                 sorted_rows.reverse()
         else:
             sorted_rows = sorted(
                 kwic_rows, 
                 key=lambda x: clean_html(x.get(sort_col, '') if isinstance(x, dict) else str(x)).lower(), 
                 reverse=(sort_dir == 'desc')
             )

         # --- PAGINATION & VIEW CONTROLS ---
         st.markdown("---")
         c_pag1, c_pag2, c_pag3 = st.columns([2, 3, 2])
         
         page_size_options = ["All", 25, 50, 100, 250, 500]
         saved_ps = get_state(f'kwic_page_size_{key_suffix}', "All")
         try:
             default_idx = page_size_options.index(saved_ps)
         except ValueError:
             default_idx = 0
             
         with c_pag1:
             selected_ps = st.selectbox(
                 "Rows per page",
                 options=page_size_options,
                 index=default_idx,
                 key=f"kwic_ps_select_{key_suffix}"
             )
             set_state(f'kwic_page_size_{key_suffix}', selected_ps)
             
             if selected_ps == "All":
                 page_size = len(sorted_rows) if sorted_rows else 1
             else:
                 page_size = int(selected_ps)

         total_items = len(sorted_rows)
         total_pages = max(1, math.ceil(total_items / page_size)) if page_size > 0 else 1
         
         current_page = get_state(f'kwic_page_num_{key_suffix}', 1)
         if current_page > total_pages: current_page = total_pages
         if current_page < 1: current_page = 1

         start_idx = (current_page - 1) * page_size
         end_idx = min(current_page * page_size, total_items)
         display_start = start_idx + 1 if total_items > 0 else 0

         with c_pag2:
             st.markdown(
                 f"<div style='text-align: center; margin-top: 10px; font-weight: bold; color: #00FFF5;'>"
                 f"Page {current_page} of {total_pages} ({display_start}-{end_idx} of {total_items:,} matches shown)"
                 f"</div>",
                 unsafe_allow_html=True
             )

         with c_pag3:
             b_prev, b_next = st.columns(2)
             with b_prev:
                 if st.button("◀ Prev", key=f"btn_prev_top_{key_suffix}", disabled=(current_page <= 1), use_container_width=True):
                     set_state(f'kwic_page_num_{key_suffix}', current_page - 1)
                     st.rerun()
             with b_next:
                 if st.button("Next ▶", key=f"btn_next_top_{key_suffix}", disabled=(current_page >= total_pages), use_container_width=True):
                     set_state(f'kwic_page_num_{key_suffix}', current_page + 1)
                     st.rerun()

         # Slice rows for current page display
         page_rows = sorted_rows[start_idx:end_idx]

         wrap_style = "white-space: normal !important; overflow-wrap: break-word;" if get_state('kwic_wrap_mode', True) else "white-space: nowrap;"
         
         html = f"""
         <style>
         .kwic-table-wrapper {{
             max-height: 650px;
             overflow-y: auto;
             overflow-x: auto;
             border: 1px solid #334155;
             border-radius: 8px;
             margin-top: 10px;
             margin-bottom: 12px;
             background-color: #0f172a;
         }}
         .kwic-table {{
             width: 100%;
             min-width: 800px;
             font-family: 'Courier New', monospace;
             font-size: 0.9em;
             border-collapse: collapse;
             table-layout: auto;
         }}
         .kwic-table thead tr {{
             position: sticky;
             top: 0;
             background-color: #1e293b;
             z-index: 10;
             box-shadow: 0 2px 5px rgba(0,0,0,0.5);
         }}
         .kwic-table th {{
             padding: 10px;
             border-bottom: 2px solid #00ADB5;
             color: #00FFF5;
             font-weight: bold;
             text-align: center;
         }}
         .kwic-table td {{
             padding: 8px 10px;
             border-bottom: 1px solid #333;
             vertical-align: middle;
             line-height: 1.6;
         }}
         .meta-col {{ text-align: left; width: 15%; font-size: 0.8em; border-right: 1px solid #444; color: #e2e8f0; vertical-align: top; display: {'table-cell' if show_meta else 'none'}; }}
         .ctx-l {{ text-align: right; width: 35%; color: #bbb; {wrap_style} }}
         .node {{ text-align: center; width: auto; white-space: nowrap; font-weight: bold; background-color: #222; color: #FFEA00; border-left: 1px solid #444; border-right: 1px solid #444; padding: 8px 15px; }}
         .ctx-r {{ text-align: left; width: 35%; color: #bbb; {wrap_style} }}
         .ann-col {{ text-align: left; width: 15%; border-left: 1px solid #444; padding: 8px; }}
         .sort-info {{ font-size: 0.8em; color: #888; text-align: center; margin-bottom: 5px; }}
         .ann-input-container {{ display: flex; flex-direction: column; gap: 4px; }}
         .ann-input {{ background: #1e293b; color: white; border: 1px solid #334155; padding: 2px 4px; font-size: 11px; border-radius: 3px; }}
         </style>
         <div class='sort-info'>Sorted by <b>{sort_col}</b> ({'Ascending' if sort_dir == 'asc' else 'Descending'})</div>
         <div class="kwic-table-wrapper">
          <table class="kwic-table">
            <thead>
              <tr>
                <th style="display: {'table-cell' if show_meta else 'none'}; text-align: left;">Metadata</th>
                <th style="text-align: right;">Left Context</th>
                <th style="text-align: center;">Node</th>
                <th style="text-align: left;">Right Context</th>
              </tr>
            </thead>
            <tbody>
         """
         
         ann_mode = get_state('kwic_ann_mode', False)
         kwic_annotations = st.session_state.get('kwic_annotations', {})

         for i, row in enumerate(page_rows):
             l_text = row['Left']
             r_text = row['Right']
             m_id = str(row['match_id'])
             
             # Merge manual annotations into metadata for display
             display_meta = row.get('Metadata', {}).copy()
             if m_id in kwic_annotations:
                 anns = kwic_annotations[m_id]
                 if isinstance(anns, list):
                     for ann in anns:
                         if ann.get('attr') and ann.get('val'):
                             display_meta[ann['attr']] = ann['val']
                 elif isinstance(anns, dict): # Legacy support
                     if anns.get('attr') and anns.get('val'):
                         display_meta[anns['attr']] = anns['val']

             meta_html = ""
             if display_meta:
                 for k, v in display_meta.items():
                     meta_html += f"<div style='margin-bottom:2px;'><span style='background-color: #334155; color: #e2e8f0; font-size: 0.85em; padding: 2px 4px; border-radius: 3px; border: 1px solid #475569; display: inline-block;' title='{k}'>{v}</span></div>"
             
             ann_cell_html = ""
             if ann_mode:
                 pass
             
             html += f"<tr><td class='meta-col'>{meta_html}</td><td class='ctx-l'>{l_text}</td><td class='node'>{row['Node']}</td><td class='ctx-r'>{r_text}</td></tr>"
         html += "</tbody></table></div>"
         
         if not ann_mode:
            st.markdown(html, unsafe_allow_html=True)
         else:
            # INTERACTIVE ANNOTATION MODE
            st.markdown("##### ✍️ Annotation Mode Active")
            st.caption("Enter attribute (upper) and value (lower). No spaces, alphanumeric only.")
            
            # Save progress button at the top too
            if st.button("💾 Save Annotation Progress", key=f"save_ann_top_{key_suffix}"):
                save_annotations(results, kwic_annotations)

            show_meta_active = show_meta
            for i, row in enumerate(page_rows):
                m_id = str(row['match_id'])
                if show_meta_active:
                    col_m, col_l, col_n, col_r, col_a = st.columns([1, 3.5, 2, 3.5, 2])
                    with col_m:
                        m = row.get('Metadata', {})
                        for k, v in m.items():
                            st.caption(f"{v}")
                else:
                    col_l, col_n, col_r, col_a = st.columns([4, 2, 4, 2])
                
                with col_l:
                    st.markdown(f"<div style='text-align:right; color:#bbb;'>{row['Left']}</div>", unsafe_allow_html=True)
                with col_n:
                    st.markdown(f"<div style='text-align:center; font-weight:bold; color:#FFEA00;'>{row['Node']}</div>", unsafe_allow_html=True)
                with col_r:
                    st.markdown(f"<div style='text-align:left; color:#bbb;'>{row['Right']}</div>", unsafe_allow_html=True)
                
                with col_a:
                    current_list = kwic_annotations.get(m_id, [{"attr": "", "val": ""}])
                    if not isinstance(current_list, list): current_list = [current_list]
                    
                    updated_list = []
                    for idx, ann in enumerate(current_list):
                        c1, c2 = st.columns([6, 1])
                        with c1:
                            new_attr = st.text_input("Attr", value=ann['attr'], key=f"ann_attr_{m_id}_{idx}_{key_suffix}", label_visibility="collapsed", placeholder="attr")
                            new_val = st.text_input("Val", value=ann['val'], key=f"ann_val_{m_id}_{idx}_{key_suffix}", label_visibility="collapsed", placeholder="value")
                        with c2:
                            if st.button("🗑️", key=f"del_ann_{m_id}_{idx}", help="Remove this pair"):
                                continue # Skip adding to updated_list
                        
                        clean_attr = re.sub(r'[^a-zA-Z0-9]', '', new_attr)
                        clean_val = re.sub(r'[^a-zA-Z0-9]', '', new_val)
                        updated_list.append({"attr": clean_attr, "val": clean_val})
                    
                    if st.button("➕ Add Pair", key=f"add_pair_{m_id}"):
                        updated_list.append({"attr": "", "val": ""})
                        st.session_state['kwic_annotations'][m_id] = updated_list
                        st.rerun()
                    
                    st.session_state['kwic_annotations'][m_id] = updated_list

            st.markdown("---")
            if st.button("💾 Save Annotation Progress", key=f"save_ann_bottom_{key_suffix}", type="primary", use_container_width=True):
                save_annotations(results, st.session_state['kwic_annotations'])

         # Bottom Pagination Bar
         if total_pages > 1:
             c_bot1, c_bot2, c_bot3 = st.columns([2, 3, 2])
             with c_bot2:
                 st.markdown(
                     f"<div style='text-align: center; margin-top: 5px; font-weight: bold; color: #00FFF5;'>"
                     f"Page {current_page} of {total_pages}"
                     f"</div>",
                     unsafe_allow_html=True
                 )
             with c_bot3:
                 b_prev_b, b_next_b = st.columns(2)
                 with b_prev_b:
                     if st.button("◀ Prev", key=f"btn_prev_bot_{key_suffix}", disabled=(current_page <= 1), use_container_width=True):
                         set_state(f'kwic_page_num_{key_suffix}', current_page - 1)
                         st.rerun()
                 with b_next_b:
                     if st.button("Next ▶", key=f"btn_next_bot_{key_suffix}", disabled=(current_page >= total_pages), use_container_width=True):
                         set_state(f'kwic_page_num_{key_suffix}', current_page + 1)
                         st.rerun()
     else:
         st.info("No matches found.")
         
     # Download
     if kwic_rows:
         # Prepare DF for export (clean versions of context)
         export_rows = []
         for r in kwic_rows:
             # Remove HTML tags for Excel
             export_rows.append({
                 "Left": re.sub(r'<[^>]*>', '', r['Left']),
                 "Node": re.sub(r'<[^>]*>', '', r['Node']),
                 "Right": re.sub(r'<[^>]*>', '', r['Right']),
             })
         df_export = pd.DataFrame(export_rows)
         st.download_button(
             label=f"Download {name} results (Excel)",
             data=df_to_excel_bytes(df_export),
             file_name=f"concordance_{search_term}_{name}.xlsx",
             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
             key=f"dl_{key_suffix}"
         )
         
     
     st.markdown("---")
     if st.button("Interpret with AI", key=f"btn_kwic_ai_{key_suffix}"):
          with st.spinner("Analyzing patterns..."):
               sample_lines_text = "\n".join([f"{r['Left']} [{r['Node']}] {r['Right']}" for r in kwic_rows[:15]])
               breakdown_str = breakdown.head(10).to_string(index=False) if (breakdown is not None and not breakdown.empty) else "None"
               
               data_payload = f"""=== QUANTITATIVE FREQUENCY METRICS ===
Target Keyword: '{search_term}'
Corpus Dataset: '{name}'
Absolute Frequency (Total Occurrences in Corpus): {total:,} occurrences
Relative Frequency: {rel_freq:.2f} per million words (PMW)
Total Corpus Tokens: {total_tokens:,}
Sample Lines Provided Below: {len(kwic_rows[:15])} sample preview lines (out of {total:,} total occurrences in corpus)

=== TOKEN BREAKDOWN STATS (Top Variations) ===
{breakdown_str}

=== SAMPLE KWIC CONCORDANCE LINES (For Contextual Analysis) ===
{sample_lines_text}
"""
               resp, err = interpret_results_llm(
                   target_word=search_term,
                   analysis_type="Concordance Analysis",
                   data_description=f"Snapshot and quantitative frequency summary for '{search_term}' in '{name}'.",
                   data=data_payload,
                   ai_provider=get_state('ai_provider'),
                   gemini_api_key=get_state('gemini_api_key'),
                   ollama_url=get_state('ollama_url'),
                   ollama_model=get_state('ai_model')
               )
               if resp:
                   set_state(f'llm_res_kwic_{key_suffix}', resp)
               elif err:
                   st.error(err)
                   
     llm_res = get_state(f'llm_res_kwic_{key_suffix}')
     if llm_res:
         st.markdown(llm_res)

def save_annotations(results, annotations):
    import json
    # Create integrity key
    key = f"{results['path']}_{results['search_term']}"
    save_data = {
        "key": key,
        "corpus_path": results['path'],
        "corpus_name": get_state('current_corpus_name', 'Unknown Corpus'),
        "search_term": results['search_term'],
        "annotations": annotations
    }
    
    # We use a download button for the save action to allow user to pick location
    json_str = json.dumps(save_data, indent=2)
    st.download_button(
        "📥 Download Annotation File",
        data=json_str,
        file_name=f"annotations_{results['search_term']}.json",
        mime="application/json",
        key="download_ann_btn"
    )
    st.info("Click above to save your work to your local machine.")
