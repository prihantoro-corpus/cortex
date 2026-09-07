import streamlit as st
import pandas as pd
from ui_streamlit.state_manager import get_state, set_state
from ui_streamlit.caching import cached_generate_collocation
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.preprocessing.xml_parser import apply_xml_restrictions
from core.ai_service import interpret_results_llm, parse_nl_query, parse_nl_query_rules_only
from core.visualiser.network import create_pyvis_graph
import core.modules.collocation_patterns
from core.io_utils import df_to_excel_bytes
from ui_streamlit.caching import cached_generate_kwic, cached_get_subcorpus_size
from ui_streamlit.components.result_display import render_kwic_table
import core.modules.overview as ov

def render_collocation_view():
    st.header("Collocation Analysis")
    
    with st.expander("💡 **Method & Transparency: Collocation Analysis**", expanded=False):
        st.markdown("""
        **Goal:** Identify words that appear near your 'Node Word' more frequently than would be expected by chance.
        
        **Data Used:** 
        - Frequencies of the node word and potential collocates within a specific **Span (Window)**.
        - Global frequencies of these words across the entire (sub)corpus.
        
        **Statistical Measures:**
        - **Log-Likelihood (LL):** Measures **statistical significance**. High LL means the association is very unlikely to be a coincidence. (Recommended for identifying key associations).
        - **Mutual Information (MI):** Measures **association strength**. High MI indicates that the words are very "exclusive" to each other, even if they are infrequent.
        - **Observed (Obs):** The actual number of times the words appeared together.
        """)
    
    corpus_path = get_state('current_corpus_path')
    corpus_name = get_state('current_corpus_name', 'Corpus')
    corpus_stats = get_state('corpus_stats')
    
    if not corpus_path:
        st.warning("Please load a corpus first.")
        return

    # Guidelines Layout using shared component
    from ui_streamlit.components.guidelines import render_guidelines
    col_main = render_guidelines("Collocation")

    with col_main:


        # Initialize XML restriction variables to prevent NameError in NL search modes
        xml_where = ""
        xml_params = []
        xml_where_1 = ""
        xml_params_1 = []
        xml_where_2 = ""
        xml_params_2 = []

        # Deferred execution flag for NL AI mode (query runs AFTER XML filters are rendered)
        _deferred_coll_query = None

        comp_mode = get_state('comparison_mode', False)
        comp_path = get_state('comp_corpus_path')
        comp_name = get_state('comp_corpus_name')

        tab_simple, tab_advanced, tab_multi_compare = st.tabs(["Simple", "Advanced", "Multi-Node Comparison"])

        with tab_simple:
            node_word_simple = st.text_input("Node Word", value="", key="coll_node_simple", help="Search word or phrase")
            if st.button("Calculate Collocations", type="primary", key="btn_calculate_collocation_simple", use_container_width=True):
                 if node_word_simple:
                     run_collocation_query(
                         identifier='primary',
                         path=corpus_path,
                         word=node_word_simple,
                         window=5,
                         min_freq=3,
                         max_rows=100,
                         stats=corpus_stats,
                         xml_where="",
                         xml_params=[],
                         token_filter="",
                         pos_filter="",
                         lemma_filter="",
                         pattern_text="",
                         pattern_limit=50,
                         stat_measure="Log-Likelihood",
                         source='simple'
                     )
                     st.rerun()
                 else:
                     st.warning("Please enter a Node Word.")

        with tab_advanced:
            # 1. Inputs
            search_mode = st.radio("Search Mode", ["Standard", "Natural Language (Rule)", "Natural Language (AI)"], horizontal=True, key="coll_search_mode")

            if search_mode == "Natural Language (AI)":
                st.markdown("### 🧠 Natural Language Search")
                nl_query = st.text_area("Describe what you want to find", height=70, placeholder="e.g. Find adjectives that appear within 3 words of 'environment' appearing at least 5 times")

                col_ai1, col_ai2 = st.columns([1, 4])
                with col_ai1:
                    analyze_btn = st.button("Analyze & Search", type="primary")
                with col_ai2:
                    show_example_meta = st.checkbox("Show Metadata in Examples", value=get_state('coll_show_example_meta', False), key="coll_show_example_meta_ai")
                    set_state('coll_show_example_meta', show_example_meta)

                if analyze_btn:
                    if not nl_query:
                        st.warning("Please enter a query.")
                    else:
                        with st.spinner("AI is parsing your query..."):
                            params, err = parse_nl_query(
                                nl_query, 
                                "collocation",
                                ai_provider=get_state('ai_provider'),
                                gemini_api_key=get_state('gemini_api_key'),
                                ollama_url=get_state('ollama_url'),
                                ollama_model=get_state('ai_model')
                            )

                        if params:
                            # Safely parse params
                            try:
                                win = int(params.get('window', 5))
                            except (ValueError, TypeError): win = 5

                            try:
                                freq = int(params.get('min_freq', 3))
                            except (ValueError, TypeError): freq = 3

                            try:
                                mx = int(params.get('max_rows', 100))
                            except (ValueError, TypeError): mx = 100

                            # Update state with parsed parameters
                            set_state('coll_node', params.get('node_word', ''))
                            set_state('coll_window', win)
                            set_state('coll_min_freq', freq)
                            set_state('coll_token_filt', params.get('token_filter', ''))
                            set_state('coll_pos_filt', params.get('pos_filter', ''))
                            set_state('coll_lemma_filt', params.get('lemma_filter', ''))

                            st.success("✓ Query interpretation successful! Running search...")

                            # Defer execution until after XML filters are rendered below
                            _deferred_coll_query = {
                                'node': params.get('node_word', ''),
                                'win': win, 'freq': freq, 'mx': mx,
                                'token_filter': params.get('token_filter', ''),
                                'pos_filter': params.get('pos_filter', ''),
                                'lemma_filter': params.get('lemma_filter', ''),
                            }
                        else:
                            st.error(f"Failed to parse query: {err}")

            if search_mode == "Natural Language (Rule)":
                st.markdown("### ⚡ Natural Language Search (Rule-Based)")
                st.caption("Fast, deterministic parsing. Use terms like 'noun', 'verb', or 'word followed by...'. Filters also support these terms.")

                with st.expander("Collocation Settings", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                         nl_query = st.text_input("Node Word Query (NL/Rule)", value=get_state('coll_nl_query_rule', ''), placeholder="e.g. adjective followed by noun", key="coll_nl_input_rule")
                    with col2:
                         # Shared settings
                         window = st.slider("Span (Window)", 1, 10, 5, key="coll_window_rule")
                         c_sub1, c_sub2 = st.columns(2)
                         with c_sub1:
                            min_freq = st.number_input("Min Co-occurrence", 1, 100, 3, key="coll_min_freq_rule")
                         with c_sub2:
                            max_rows = st.number_input("Max Collocates", 10, 50000, 100, step=10, key="coll_max_rule")
                         measures_list = ["Log-Likelihood", "Log-Dice", "Dice Coefficient", "Mutual Information"]
                         current_measure_rule = get_state('coll_stat_measure_rule', 'Log-Likelihood')
                         measure_idx_rule = measures_list.index(current_measure_rule) if current_measure_rule in measures_list else 0
                         st.radio("Association Measure", measures_list, index=measure_idx_rule, horizontal=True, key="coll_stat_measure_rule")
                         show_example_meta = st.checkbox("Show Metadata in Examples", value=get_state('coll_show_example_meta', False), key="coll_show_example_meta_rule")
                         set_state('coll_show_example_meta', show_example_meta)
                         show_all_in_conc = st.checkbox(
                             "Show all collocates in concordance",
                             value=get_state('coll_show_all_in_conc', False),
                             help="⚠️ Warning: This will query and show all occurrences in the concordance and may take significant time to load.",
                             key="coll_show_all_in_conc_rule"
                         )
                         set_state('coll_show_all_in_conc', show_all_in_conc)

                    st.markdown("---")
                    f_col1, f_col2, f_col3 = st.columns(3)
                    with f_col1:
                        token_filter_input = st.text_input("Token Filter (NL)", placeholder="e.g. not 'the'", key="coll_token_filt_rule", help="*al : Matches any collocate token ending in \"al\" (e.g., denial, rebuttal).\n-col* : Excludes all collocate tokens starting with \"col\".\nb?t : Matches any 3-letter token starting with \"b\" and ending with \"t\" (but, bat, bit).\n(word1|word2|*ing) : Union, matches word1, word2, or any token ending in ing.")
                    with f_col2:
                        pos_filter_input = st.text_input("POS Filter (NL)", placeholder="e.g. noun, verb", key="coll_pos_filt_rule", help="*VB* : Matches any POS tag containing \"VB\" (e.g., VBN, VBD).\n-NN* : Excludes all POS tags starting with \"NN\".\nN? : Matches any 2-letter POS tag starting with \"N\".\n(JJ|RB|*VB) : Union, matches JJ, RB, or any tag ending in VB.")
                        from ui_streamlit.components.pos_help import render_pos_help_button
                        render_pos_help_button(corpus_path, "collocation_rule")
                    with f_col3:
                        lemma_filter_input = st.text_input("Lemma Filter (NL)", placeholder="e.g. be, have", key="coll_lemma_filt_rule", help="*ate : Matches any lemma ending in \"ate\" (e.g., negotiate, calculate).\n-pre* : Excludes all lemmas starting with \"pre\".\ns?t : Matches any 3-letter lemma starting with \"s\" and ending with \"t\" (e.g., sit, sat).\n(run|walk|*ing) : Union, matches run, walk, or any lemma ending in ing.")

                    # Pattern Matching Section (Reusable)
                    st.markdown("---")
                    apply_patterns = st.checkbox("Apply Patterns (Advanced)", value=get_state('coll_apply_patterns', False), key="coll_apply_patterns_rule")
                    if apply_patterns:
                        pattern_text = st.text_area("Pattern Definitions", value=get_state('coll_pattern_text', ''), height=100, key="coll_pattern_input_rule")
                        set_state('coll_pattern_text', pattern_text)
                        pattern_limit = st.number_input("Max Collocates for Patterns", 10, 100, 50, key="coll_pattern_limit_rule")
                        set_state('coll_pattern_limit', pattern_limit)

            if search_mode == "Standard":
                with st.expander("Collocation Settings", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                         from ui_streamlit.components.pos_help import render_annotation_help_button
                         if not comp_mode:
                             node_word = st.text_input("Node Word", value="", placeholder="e.g. beautiful, [lemma]*, _VB*, *kan", key="coll_node", help="Use * for wildcards (e.g. run*), _TAG for POS (e.g. _NN), [lemma] for lemma (e.g. [run]), token_POS (e.g. light_V*), or <TAG> for XML tags (e.g. <PN>)")
                             render_annotation_help_button(corpus_path, "collocation_node")
                         else:
                             st.markdown("**Node Words**")
                             node_primary = st.text_input(f"Primary ({get_state('current_corpus_name', 'Corpus')})", value="", key="coll_node_primary")
                             render_annotation_help_button(corpus_path, "collocation_node_primary")
                             node_secondary = st.text_input(f"Comparison ({comp_name if comp_name else 'Secondary'})", value="", key="coll_node_secondary")
                             if comp_path:
                                 render_annotation_help_button(comp_path, "collocation_node_secondary")
                             node_word = node_primary # Default for single-path logic logic below if needed, though we split paths

                    with col2:
                         # Shared settings for now
                         window = st.slider("Span (Window)", 1, 10, 5, key="coll_window")
                         c_sub1, c_sub2 = st.columns(2)
                         with c_sub1:
                            min_freq = st.number_input("Min Co-occurrence", 1, 100, 3, key="coll_min_freq")
                         with c_sub2:
                            max_rows = st.number_input("Max Collocates", 10, 50000, 100, step=10, key="coll_max", help="Increase this limit to download more results (up to 50,000).")
                         measures_list = ["Log-Likelihood", "Log-Dice", "Dice Coefficient", "Mutual Information"]
                         current_measure = get_state('coll_stat_measure', 'Log-Likelihood')
                         measure_idx = measures_list.index(current_measure) if current_measure in measures_list else 0
                         st.radio("Association Measure", measures_list, index=measure_idx, horizontal=True, key="coll_stat_measure")
                         show_example_meta = st.checkbox("Show Metadata in Examples", value=get_state('coll_show_example_meta', False), key="coll_show_example_meta_std")
                         set_state('coll_show_example_meta', show_example_meta)
                         show_all_in_conc = st.checkbox(
                             "Show all collocates in concordance",
                             value=get_state('coll_show_all_in_conc', False),
                             help="⚠️ Warning: This will query and show all occurrences in the concordance and may take significant time to load.",
                             key="coll_show_all_in_conc_std"
                         )
                         set_state('coll_show_all_in_conc', show_all_in_conc)

                    st.markdown("---")
                    f_col1, f_col2, f_col3 = st.columns(3)
                    with f_col1:
                        token_filter = st.text_input("Token Filter", placeholder="e.g. no, non OR -no, -non", key="coll_token_filt", help="*al : Matches any collocate token ending in \"al\" (e.g., denial, rebuttal).\n-col* : Excludes all collocate tokens starting with \"col\".\nb?t : Matches any 3-letter token starting with \"b\" and ending with \"t\" (but, bat, bit).\n(word1|word2|*ing) : Union, matches word1, word2, or any token ending in ing.")
                    with f_col2:
                        pos_filter = st.text_input("POS Filter", placeholder="e.g. JJ, NN OR -JJ, -NN", key="coll_pos_filt", help="*VB* : Matches any POS tag containing \"VB\" (e.g., VBN, VBD).\n-NN* : Excludes all POS tags starting with \"NN\".\nN? : Matches any 2-letter POS tag starting with \"N\".\n(JJ|RB|*VB) : Union, matches JJ, RB, or any tag ending in VB.")
                        from ui_streamlit.components.pos_help import render_pos_help_button
                        render_pos_help_button(corpus_path, "collocation_standard")
                    with f_col3:
                        lemma_filter = st.text_input("Lemma Filter", placeholder="e.g. see OR -see", key="coll_lemma_filt", help="*ate : Matches any lemma ending in \"ate\" (e.g., negotiate, calculate).\n-pre* : Excludes all lemmas starting with \"pre\".\ns?t : Matches any 3-letter lemma starting with \"s\" and ending with \"t\" (e.g., sit, sat).\n(run|walk|*ing) : Union, matches run, walk, or any lemma ending in ing.")

                    # Pattern Matching Section
                    st.markdown("---")
                    st.markdown("### 🔍 Collocation Patterns (Optional)")

                    # Pattern syntax help
                    with st.expander("ℹ️ Pattern Syntax Guide", expanded=False):
                        st.markdown("""
                        **Pattern Format:** `label : pattern`

                        **Symbols:**
                        - `<>` : the node word
                        - `#` : the collocate
                        - `*` : optional token (any word, 0 or 1)
                        - `+` : required token (exactly 1 word)
                        - `token` : specific token (obligatory)
                        - `(token)` : optional specific token

                        **Constraints:**
                        - `[lemma]` : token must be from specified lemma
                        - `_TAG` : token must have specified POS tag
                        - `(_TAG)` : optional POS tag constraint
                        - `([lemma])` : optional lemma constraint

                        **Examples:**
                        ```
                        Agent of passive di- : <> * * #
                        Patient/theme : # * <>
                        Strict adjacency : # <>
                        Gap of one : <> + #
                        ```

                        **Note:** One pattern per line, up to 50 patterns.
                        """)

                    # Pattern input
                    pattern_text = st.text_area(
                        "Pattern Definitions",
                        value=get_state('coll_pattern_text', ''),
                        height=150,
                        placeholder="Example:\nAgent of passive di- : <> * * #\nPatient of passive di- : # * <>",
                        help="Enter one pattern per line. Format: label : pattern",
                        key="coll_pattern_input"
                    )

                    # Save pattern text to state
                    if pattern_text != get_state('coll_pattern_text', ''):
                        set_state('coll_pattern_text', pattern_text)

                    # Pattern controls
                    p_col1, p_col2 = st.columns([1, 1])
                    with p_col1:
                        pattern_limit = st.number_input(
                            "Max Collocates for Patterns",
                            min_value=10,
                            max_value=100,
                            value=get_state('coll_pattern_limit', 50),
                            step=10,
                            help="Limit pattern matching to top N collocates for performance",
                            key="coll_pattern_limit_input"
                        )
                        if pattern_limit != get_state('coll_pattern_limit', 50):
                            set_state('coll_pattern_limit', pattern_limit)

                    with p_col2:
                        apply_patterns = st.checkbox(
                            "Apply Patterns",
                            value=get_state('coll_apply_patterns', False),
                            help="Enable pattern-based clustering of collocates",
                            key="coll_apply_patterns_check"
                        )
                        if apply_patterns != get_state('coll_apply_patterns', False):
                            set_state('coll_apply_patterns', apply_patterns)


            # --- XML Restriction Filters ---
            if not comp_mode:
                xml_filters = render_xml_restriction_filters(corpus_path, "collocation", corpus_name=corpus_name)
                xml_where, xml_params = apply_xml_restrictions(xml_filters)
            else:
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    xml_filters_1 = render_xml_restriction_filters(corpus_path, "collocation_c1", corpus_name=corpus_name)
                    xml_where_1, xml_params_1 = apply_xml_restrictions(xml_filters_1)
                with col_f2:
                    if comp_path:
                        xml_filters_2 = render_xml_restriction_filters(comp_path, "collocation_c2", corpus_name=comp_name)
                        xml_where_2, xml_params_2 = apply_xml_restrictions(xml_filters_2)
                    else:
                        xml_where_2, xml_params_2 = "", []

            if not comp_mode:
                if st.button("Calculate Collocations", type="primary", key="btn_calculate_coll_advanced"):
                    # EXECUTION LOGIC

                    # Determine effective parameters based on Mode
                    to_run_node = ""
                    to_run_win = 5
                    to_run_min_freq = 3
                    to_run_max = 100

                    to_run_tok = ""
                    to_run_pos = ""
                    to_run_lem = ""

                    run_valid = False

                    if search_mode == "Standard":
                        to_run_node = node_word
                        to_run_win = window
                        to_run_min_freq = min_freq
                        to_run_max = max_rows
                        to_run_tok = token_filter
                        to_run_pos = pos_filter
                        to_run_lem = lemma_filter
                        run_valid = bool(to_run_node)

                    elif search_mode == "Natural Language (Rule)":
                        if not nl_query:
                             st.warning("Please enter a Node Word query.")
                             run_valid = False
                        else:
                            set_state('coll_nl_query_rule', nl_query)
                            # Parse Main Node Query
                            pos_defs = ov.get_pos_definitions(corpus_path) or {}
                            reverse_pos_map = {v.lower(): k for k, v in pos_defs.items() if v}

                            params, err = parse_nl_query_rules_only(nl_query, "collocation", reverse_pos_map=reverse_pos_map)
                            if params:
                                def quick_parse(txt, r_map=reverse_pos_map):
                                    if not txt: return ""
                                    p, _ = parse_nl_query_rules_only(txt, "collocation", reverse_pos_map=r_map)
                                    return p.get('node_word', txt) if p else txt

                                to_run_node = params.get('node_word', '')
                                to_run_win = window # from shared inputs in Rule block
                                to_run_min_freq = min_freq
                                to_run_max = max_rows

                                to_run_tok = quick_parse(token_filter_input)
                                to_run_pos = quick_parse(pos_filter_input)
                                to_run_lem = quick_parse(lemma_filter_input)

                                run_valid = bool(to_run_node)
                                if run_valid:
                                     st.success(f"✓ Searching for: **{to_run_node}**")
                            else:
                                st.error(f"Error parsing query: {err}")
                                run_valid = False

                    if run_valid:
                        # Get pattern settings
                        pattern_text = get_state('coll_pattern_text', '')
                        apply_patterns = get_state('coll_apply_patterns', False)
                        pattern_limit = get_state('coll_pattern_limit', 50)

                        run_collocation_query(
                            'primary', corpus_path, to_run_node, to_run_win, to_run_min_freq, to_run_max, 
                            corpus_stats, xml_where, xml_params, to_run_tok, to_run_pos, to_run_lem,
                            pattern_text if apply_patterns else '', pattern_limit,
                            stat_measure=get_state('coll_stat_measure', 'Log-Likelihood') if search_mode == "Standard" else get_state('coll_stat_measure_rule', 'Log-Likelihood')
                        )
            else:
                if st.button("Calculate Comparison Collocations", type="primary", key="btn_calculate_coll_comp_advanced"):
                    # Determine effective parameters
                    to_run_node_1 = ""
                    to_run_node_2 = ""

                    # Shared params
                    to_run_win = 5
                    to_run_min_freq = 3
                    to_run_max = 100
                    to_run_tok = ""
                    to_run_pos = ""
                    to_run_lem = ""

                    run_valid = False

                    if search_mode == "Standard":
                        to_run_node_1 = node_primary
                        to_run_node_2 = node_secondary
                        to_run_win = window
                        to_run_min_freq = min_freq
                        to_run_max = max_rows
                        to_run_tok = token_filter
                        to_run_pos = pos_filter
                        to_run_lem = lemma_filter
                        run_valid = True

                    elif search_mode == "Natural Language (Rule)":
                         # Use same node for both
                         if not nl_query:
                             st.warning("Please enter a Node Word query.")
                             run_valid = False
                         else:
                             set_state('coll_nl_query_rule', nl_query)

                             pos_defs = ov.get_pos_definitions(corpus_path) or {}
                             reverse_pos_map = {v.lower(): k for k, v in pos_defs.items() if v}

                             params, err = parse_nl_query_rules_only(nl_query, "collocation", reverse_pos_map=reverse_pos_map)
                             if params:
                                 def quick_parse(txt, r_map=reverse_pos_map):
                                    if not txt: return ""
                                    p, _ = parse_nl_query_rules_only(txt, "collocation", reverse_pos_map=r_map)
                                    return p.get('node_word', txt) if p else txt

                                 parsed_node = params.get('node_word', '')
                                 to_run_node_1 = parsed_node
                                 to_run_node_2 = parsed_node

                                 to_run_win = window
                                 to_run_min_freq = min_freq
                                 to_run_max = max_rows

                                 to_run_tok = quick_parse(token_filter_input)
                                 to_run_pos = quick_parse(pos_filter_input)
                                 to_run_lem = quick_parse(lemma_filter_input)

                                 run_valid = bool(parsed_node)
                                 if run_valid:
                                     st.success(f"✓ Searching for: **{parsed_node}** in both corpora")
                             else:
                                st.error(f"Error parsing query: {err}")
                                run_valid = False

                    if run_valid:
                        # Get pattern settings
                        pattern_text = get_state('coll_pattern_text', '')
                        apply_patterns = get_state('coll_apply_patterns', False)
                        pattern_limit = get_state('coll_pattern_limit', 50)

                        stat_measure = get_state('coll_stat_measure', 'Log-Likelihood') if search_mode == "Standard" else get_state('coll_stat_measure_rule', 'Log-Likelihood')

                        # Run Primary
                        if to_run_node_1:
                            run_collocation_query(
                                'primary', corpus_path, to_run_node_1, to_run_win, to_run_min_freq, to_run_max, 
                                corpus_stats, xml_where_1, xml_params_1, to_run_tok, to_run_pos, to_run_lem,
                                pattern_text if apply_patterns else '', pattern_limit,
                                stat_measure=stat_measure
                            )
                        else:
                             st.warning("Primary node word missing.")

                        # Run Comparison
                        if comp_path and to_run_node_2:
                            comp_stats = get_state('comp_corpus_stats')
                            run_collocation_query(
                                'secondary', comp_path, to_run_node_2, to_run_win, to_run_min_freq, to_run_max, 
                                comp_stats, xml_where_2, xml_params_2, to_run_tok, to_run_pos, to_run_lem,
                                pattern_text if apply_patterns else '', pattern_limit,
                                stat_measure=stat_measure
                            )
                        elif comp_path and not to_run_node_2:
                            st.warning("Comparison node word missing.")

            # --- Deferred NL AI Query Execution (runs AFTER xml_where/xml_params are set) ---
            if _deferred_coll_query is not None:
                _dq = _deferred_coll_query
                run_collocation_query(
                    'primary', corpus_path,
                    _dq['node'], _dq['win'], _dq['freq'], _dq['mx'],
                    corpus_stats, xml_where, xml_params,
                    _dq['token_filter'], _dq['pos_filter'], _dq['lemma_filter'],
                    '', 50,
                    stat_measure=get_state('coll_stat_measure', 'Log-Likelihood')
                )

        with tab_multi_compare:
            st.markdown("### 🔍 Compare Collocates of Multiple Nodes")
            st.write("Compare collocates from multiple node words in the active corpus to identify shared and distinct vocabulary environments.")
            
            # Input up to 5 node words
            nodes_input = st.text_input(
                "Enter Node Words (comma-separated, max 5)",
                value=get_state('coll_multi_nodes_input', ''),
                placeholder="e.g. coffee, tea, milk, water, juice",
                key="coll_multi_nodes_text_input",
                help="Type up to 5 node words separated by commas."
            )
            set_state('coll_multi_nodes_input', nodes_input)
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                window_multi = st.slider("Span (Window)", 1, 10, 5, key="coll_window_multi")
            with col_m2:
                min_freq_multi = st.number_input("Min Co-occurrence", 1, 100, 3, key="coll_min_freq_multi")
                
            col_m3, col_m4 = st.columns(2)
            with col_m3:
                max_rows_multi = st.number_input("Max Collocates per Node", 10, 50000, 100, step=10, key="coll_max_multi")
            with col_m4:
                measures_list = ["Log-Likelihood", "Log-Dice", "Dice Coefficient", "Mutual Information"]
                current_measure_multi = get_state('coll_stat_measure_multi', 'Log-Likelihood')
                measure_idx_multi = measures_list.index(current_measure_multi) if current_measure_multi in measures_list else 0
                st.radio("Association Measure", measures_list, index=measure_idx_multi, horizontal=True, key="coll_stat_measure_multi")
                
            # Filters
            st.markdown("#### ⚙️ Filters (Optional)")
            f_col_m1, f_col_m2, f_col_m3 = st.columns(3)
            with f_col_m1:
                token_filter_multi = st.text_input("Token Filter", placeholder="e.g. no, non", key="coll_token_filt_multi")
            with f_col_m2:
                pos_filter_multi = st.text_input("POS Filter", placeholder="e.g. JJ, NN", key="coll_pos_filt_multi")
                from ui_streamlit.components.pos_help import render_pos_help_button
                render_pos_help_button(corpus_path, "collocation_multi")
            with f_col_m3:
                lemma_filter_multi = st.text_input("Lemma Filter", placeholder="e.g. see", key="coll_lemma_filt_multi")
                
            # Render XML Filters for Multi-Node Compare
            xml_filters_multi = render_xml_restriction_filters(corpus_path, "collocation_multi", corpus_name=corpus_name)
            xml_where_multi, xml_params_multi = apply_xml_restrictions(xml_filters_multi)
            
            # Action button
            if st.button("Calculate Multi-Node Collocations", type="primary", key="btn_calculate_multi_node_collocations", use_container_width=True):
                # Parse nodes
                nodes = [n.strip() for n in nodes_input.split(',') if n.strip()]
                if not nodes:
                    st.warning("Please enter at least one node word.")
                elif len(nodes) > 5:
                    st.warning("Please enter a maximum of 5 node words.")
                else:
                    run_multi_node_collocation_query(
                        corpus_path, nodes, window_multi, min_freq_multi, max_rows_multi,
                        corpus_stats, xml_where_multi, xml_params_multi,
                        token_filter_multi, pos_filter_multi, lemma_filter_multi,
                        stat_measure=get_state('coll_stat_measure_multi', 'Log-Likelihood')
                    )
                    st.rerun()
                    
            # Render Multi-Node Results
            render_multi_node_results()

        # 3. Display
        if not comp_mode:
            results = st.session_state.get('last_coll_results_primary')
            if results and results.get('source') != 'simple':
                current_measure = get_state('coll_stat_measure', 'Log-Likelihood') if search_mode == "Standard" else get_state('coll_stat_measure_rule', 'Log-Likelihood')
                if results.get('stat_measure') != current_measure:
                    run_collocation_query(
                        'primary', corpus_path, results.get('node'), results.get('window', 5),
                        results.get('min_freq', 3), results.get('max_rows', 100),
                        corpus_stats, results.get('xml_where', ''), results.get('xml_params', []),
                        results.get('token_filter', ''), results.get('pos_filter', ''), results.get('lemma_filter', ''),
                        get_state('coll_pattern_text', '') if get_state('coll_apply_patterns', False) else '',
                        get_state('coll_pattern_limit', 50),
                        stat_measure=current_measure,
                        source=results.get('source', 'advanced')
                    )
                    st.rerun()

            if results:
                render_collocation_results_column(results)

                # Display pattern results if available
                pattern_results = st.session_state.get('pattern_results_primary')
                if pattern_results:
                    render_pattern_results(pattern_results, results, 'primary')
        else:
            current_measure = get_state('coll_stat_measure', 'Log-Likelihood') if search_mode == "Standard" else get_state('coll_stat_measure_rule', 'Log-Likelihood')

            res1 = st.session_state.get('last_coll_results_primary')
            if res1 and res1.get('source') != 'simple' and res1.get('stat_measure') != current_measure:
                run_collocation_query(
                    'primary', corpus_path, res1.get('node'), res1.get('window', 5),
                    res1.get('min_freq', 3), res1.get('max_rows', 100),
                    corpus_stats, res1.get('xml_where', ''), res1.get('xml_params', []),
                    res1.get('token_filter', ''), res1.get('pos_filter', ''), res1.get('lemma_filter', ''),
                    get_state('coll_pattern_text', '') if get_state('coll_apply_patterns', False) else '',
                    get_state('coll_pattern_limit', 50),
                    stat_measure=current_measure,
                    source=res1.get('source', 'advanced')
                )
                st.rerun()

            res2 = st.session_state.get('last_coll_results_secondary')
            if res2 and res2.get('source') != 'simple' and res2.get('stat_measure') != current_measure:
                comp_stats = get_state('comp_corpus_stats')
                run_collocation_query(
                    'secondary', comp_path, res2.get('node'), res2.get('window', 5),
                    res2.get('min_freq', 3), res2.get('max_rows', 100),
                    comp_stats, res2.get('xml_where', ''), res2.get('xml_params', []),
                    res2.get('token_filter', ''), res2.get('pos_filter', ''), res2.get('lemma_filter', ''),
                    get_state('coll_pattern_text', '') if get_state('coll_apply_patterns', False) else '',
                    get_state('coll_pattern_limit', 50),
                    stat_measure=current_measure,
                    source=res2.get('source', 'advanced')
                )
                st.rerun()

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.subheader(f"Primary: {get_state('current_corpus_name', 'Corpus')}")
                if res1: 
                    render_collocation_results_column(res1, key_suffix="c1")
                    # Pattern results for primary
                    pattern_res1 = st.session_state.get('pattern_results_primary')
                    if pattern_res1:
                        render_pattern_results(pattern_res1, res1, 'c1')
            with col_c2:
                st.subheader(f"Comparison: {comp_name}")
                if not comp_path:
                    st.info("Load a comparison corpus in sidebar.")
                else:
                    res2 = st.session_state.get('last_coll_results_secondary')
                    if res2: 
                        render_collocation_results_column(res2, key_suffix="c2")
                        # Pattern results for secondary
                        pattern_res2 = st.session_state.get('pattern_results_secondary')
                        if pattern_res2:
                            render_pattern_results(pattern_res2, res2, 'c2')

            # Comparison Analysis Tables
            if res1 and res2:
                st.markdown("---")
                st.header("📊 Comparison Analysis")

                from core.modules.comparison_analysis import compare_collocations, get_comparison_summary, render_comparison_tables

                # Get collocation DataFrames
                df1 = res1.get('df')
                df2 = res2.get('df')

                if df1 is not None and df2 is not None and not df1.empty and not df2.empty:
                    # Perform comparison
                    shared_df, df1_unique, df2_unique = compare_collocations(
                        df1, df2, 
                        corpus1_name=get_state('current_corpus_name', 'Primary'),
                        corpus2_name=comp_name
                    )

                    # Summary metrics
                    summary = get_comparison_summary(shared_df, df1_unique, df2_unique, 'collocates')
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    with col_s1:
                        st.metric("Shared Collocates", summary['total_shared'])
                    with col_s2:
                        st.metric(f"{get_state('current_corpus_name', 'Primary')} Only", summary['total_primary_unique'])
                    with col_s3:
                        st.metric(f"{comp_name} Only", summary['total_comparison_unique'])
                    with col_s4:
                        st.metric("Overlap %", f"{summary['overlap_percentage']}%")

                    # Display comparison tables
                    render_comparison_tables(shared_df, df1_unique, df2_unique, 
                                            get_state('current_corpus_name', 'Primary'), 
                                            comp_name)


def run_collocation_query(identifier, path, word, window, min_freq, max_rows, stats, xml_where, xml_params, 
                          token_filter="", pos_filter="", lemma_filter="", pattern_text="", pattern_limit=50,
                          stat_measure="Log-Likelihood", source='advanced'):
    with st.spinner(f"Computing collocations..."):
        stats_df, freq, node_mwu = cached_generate_collocation(
            db_path=path,
            word=word,
            window=window,
            min_freq=min_freq,
            max_rows=max_rows,
            is_raw=False,
            corpus_stats=stats,
            xml_where_clause=xml_where,
            xml_params=xml_params,
            token_filter=token_filter,
            pos_filter=pos_filter,
            lemma_filter=lemma_filter,
            stat_measure=stat_measure
        )
        st.session_state[f'last_coll_results_{identifier}'] = {
            'df': stats_df,
            'freq': freq,
            'node': node_mwu,
            'window': window,
            'min_freq': min_freq,
            'max_rows': max_rows,
            'token_filter': token_filter,
            'pos_filter': pos_filter,
            'lemma_filter': lemma_filter,
            'corpus_name': identifier,
            'xml_where': xml_where,
            'xml_params': xml_params,
            'stat_measure': stat_measure,
            'source': source
        }
        
        # Apply pattern matching if requested
        if pattern_text and not stats_df.empty:
            with st.spinner("Applying collocation patterns..."):
                from core.modules.collocation_patterns import parse_pattern_definitions, group_collocates_by_patterns
                
                # Parse patterns
                patterns, errors = parse_pattern_definitions(pattern_text)
                
                if errors:
                    st.error("Pattern Syntax Errors:")
                    for err in errors:
                        st.error(f"  • {err}")
                    st.session_state[f'pattern_results_{identifier}'] = None
                else:
                    if patterns:
                        st.info(f"⚙️ Applying {len(patterns)} pattern(s) to top {pattern_limit} collocates...")
                        
                        # Group collocates by patterns
                        pattern_groups = group_collocates_by_patterns(
                            stats_df,
                            patterns,
                            path,
                            node_mwu,
                            window,
                            max_collocates=pattern_limit,
                            xml_where_clause=xml_where,
                            xml_params=xml_params,
                            show_all_examples=get_state('coll_show_all_in_conc', False)
                        )
                        
                        st.session_state[f'pattern_results_{identifier}'] = {
                            'groups': pattern_groups,
                            'patterns': patterns,
                            'limit': pattern_limit
                        }
                        st.success(f"✓ Pattern matching complete! Found {len(pattern_groups)} pattern groups.")
                    else:
                        st.warning("No valid patterns found.")
                        st.session_state[f'pattern_results_{identifier}'] = None

        else:
            # Clear pattern results if not applying
            st.session_state[f'pattern_results_{identifier}'] = None

def render_collocation_results_column(results, key_suffix=""):
     df = results['df']
     n_freq = results['freq']
     node = results['node']
     win = results['window']
     stat_measure = results.get('stat_measure', 'Log-Likelihood')
     
     if not df.empty:
         st.markdown(f"**{len(df)} collocates** for '{node}' (Freq: {n_freq}) within ±{win}, sorted by **{stat_measure}**.")
         
         st.download_button(
             label=f"⬇ Download {results.get('corpus_name', 'Corpus')} Collocations (Excel)",
             data=df_to_excel_bytes(df),
             file_name=f"collocations_{node}.xlsx",
             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
             key=f"dl_coll_{key_suffix}_top"
         )
         
         measure_col_map = {
             "Log-Likelihood": "LL",
             "Log-Dice": "Log-Dice",
             "Dice Coefficient": "Dice",
             "Mutual Information": "MI"
         }
         y_col = measure_col_map.get(stat_measure, "LL")

         tab_table, tab_charts, tab_graph = st.tabs(["Table", f"Charts ({stat_measure})", "Network Graph"])
         with tab_table:
             df_display = df.reset_index(drop=True)
             df_display.index += 1
             st.dataframe(df_display, use_container_width=True)

         with tab_charts:
             import matplotlib.pyplot as plt
             c1, c2, c3 = st.columns(3)
             
             def plot_top_measure(data, title, color):
                 if data.empty:
                     st.info(f"No {title} data.")
                     return
                 fig, ax = plt.subplots(figsize=(5, 4))
                 top = data.head(10).sort_values(y_col, ascending=True)
                 ax.barh(top['Collocate'], top[y_col], color=color)
                 ax.set_title(f"{title} ({stat_measure})")
                 plt.tight_layout()
                 st.pyplot(fig)

             with c1:
                 plot_top_measure(df, "Overall Top Collocates", "skyblue")
             with c2:
                 plot_top_measure(df[df['Direction'].isin(['L', 'B'])], "Left-Dominant", "salmon")
             with c3:
                 plot_top_measure(df[df['Direction'].isin(['R', 'B'])], "Right-Dominant", "lightgreen")
             
         with tab_graph:
             subtab_all, subtab_left, subtab_right = st.tabs(["🌐 Overall Network", "⬅️ Left Network", "➡️ Right Network"])
             with subtab_all:
                 st.caption("Overall Collocation Network (Top 30)")
                 graph_html = create_pyvis_graph(node, df.head(30), measure_col=y_col, measure_name=stat_measure)
                 if graph_html:
                     st.components.v1.html(graph_html, height=600)
                 else:
                     st.info("No overall network graph generated.")
             with subtab_left:
                 st.caption("Left-Dominant Collocation Network (Top 25)")
                 df_l = df[df['Direction'].isin(['L', 'B'])].head(25)
                 if not df_l.empty:
                     graph_html_l = create_pyvis_graph(node, df_l, measure_col=y_col, measure_name=stat_measure)
                     if graph_html_l:
                         st.components.v1.html(graph_html_l, height=600)
                     else:
                         st.info("No Left network graph generated.")
                 else:
                     st.info("No Left (L) collocates found.")
             with subtab_right:
                 st.caption("Right-Dominant Collocation Network (Top 25)")
                 df_r = df[df['Direction'].isin(['R', 'B'])].head(25)
                 if not df_r.empty:
                     graph_html_r = create_pyvis_graph(node, df_r, measure_col=y_col, measure_name=stat_measure)
                     if graph_html_r:
                         st.components.v1.html(graph_html_r, height=600)
                     else:
                         st.info("No Right network graph generated.")
                 else:
                     st.info("No Right (R) collocates found.")

         st.markdown("### Usage Examples by Collocates")
         # Logic fix: Ensure db_path and name are passed correctly
         db_path = get_state('current_corpus_path') if key_suffix != 'c2' else get_state('comp_corpus_path')
         corpus_name = get_state('current_corpus_name') if key_suffix != 'c2' else get_state('comp_corpus_name')
         
         # Get restrictions from results
         xml_where = results.get('xml_where', "")
         xml_params = results.get('xml_params', [])

         with st.expander("Show Examples", expanded=False):
              num_examples_opt = st.radio(
                  "**Examples per Collocate:**",
                  [1, 2, 3, 4, 5, "All"],
                  index=0,
                  horizontal=True,
                  key=f"coll_examples_per_collocate_radio_{key_suffix}",
                  help="Select how many sentence examples to fetch for each collocate in the table."
              )
              
              limit_val = 999999 if num_examples_opt == "All" else int(num_examples_opt)

              kwic_table_data = []
              # Get parallel data if needed
              is_parallel = get_state('parallel_mode', False)
              target_map = get_state('target_sent_map', {})
              
              for i, coll in enumerate(df['Collocate'].tolist()): # All collocates in the table
                   with st.spinner(f"Fetching example(s) for {coll}..."):
                        # Unpack properly to get sent_ids
                        c_kwic, _, _, _, c_sent_ids, _ = cached_generate_kwic(
                            db_path=db_path, 
                            query=node, 
                            left=7, 
                            right=7, 
                            corpus_name=corpus_name,
                            pattern_collocate_input=coll, 
                            pattern_window=win, 
                            limit=limit_val,
                            xml_where_clause=xml_where,
                            xml_params=tuple(xml_params) if xml_params else ()
                        )
                        if c_kwic:
                            for idx, kwic_item in enumerate(c_kwic):
                                col_label = f"{i+1}. {coll}" if limit_val == 1 else f"{i+1}. {coll} ({idx+1})"
                                row_data = {
                                    'Collocate': col_label,
                                    'Source Corpus': corpus_name,
                                    'Left Context': kwic_item['Left'],
                                    'Node': kwic_item['Node'],
                                    'Right Context': kwic_item['Right'],
                                    'Metadata': kwic_item.get('Metadata', {})
                                }
                                
                                if is_parallel:
                                    sid = c_sent_ids[idx] if idx < len(c_sent_ids) else None
                                    trans = target_map.get(sid, "N/A") if sid is not None else ""
                                    row_data['Translation'] = trans
                                    
                                kwic_table_data.append(row_data)

              if kwic_table_data:
                  is_simple = (results.get('source') == 'simple')
                  show_meta = False if is_simple else get_state('coll_show_example_meta', False)
                  render_kwic_table(kwic_table_data, is_parallel=is_parallel, target_lang=get_state('tgt_lang_code', 'Target'), show_meta=show_meta)
              else:
                  st.info("No examples found.")

         st.markdown("---")
         if st.button("Interpret with AI", key=f"btn_coll_ai_{key_suffix}"):
              with st.spinner("Analyzing..."):
                   header_str = f"Target Node Word: '{node}' (Total Collocates Analyzed: {len(df)})\nCollocation Statistical Measure: {stat_measure}\n\n=== TOP COLLOCATES TABLE (Explicit Counts & Association Scores) ===\n"
                   top_data = header_str + df.head(15).to_string(index=False)
                   resp, err = interpret_results_llm(
                        target_word=node,
                        analysis_type="Collocation Analysis",
                        data_description=f"Collocates for '{node}' (Window: ±{win}, Metric: {stat_measure}).",
                        data=top_data,
                        ai_provider=get_state('ai_provider'),
                        gemini_api_key=get_state('gemini_api_key'),
                        ollama_url=get_state('ollama_url'),
                        ollama_model=get_state('ai_model')
                   )
                   if resp: set_state(f'llm_res_coll_{key_suffix}', resp)
                   else: st.error(err)
          
         ai_res = get_state(f'llm_res_coll_{key_suffix}')
         if ai_res: st.markdown(ai_res)
     else:
         st.info("No collocates found.")

def render_pattern_results(pattern_results, collocation_results, key_suffix=""):
    """
    Display pattern-grouped collocation results.
    """
    groups = pattern_results.get('groups', {})
    patterns = pattern_results.get('patterns', [])
    limit = pattern_results.get('limit', 50)
    
    if not groups:
        st.info("No collocates matched any patterns.")
        return
    
    st.markdown("---")
    st.header("🎯 Pattern-Based Collocation Groups")
    st.caption(f"Pattern matching applied to top {limit} collocates from 1000 concordance sample.")
    
    # Get corpus info
    node_word = collocation_results.get('node', '')
    stat_measure = collocation_results.get('stat_measure', 'Log-Likelihood')
    
    measure_col_map = {
        "Log-Likelihood": "LL",
        "Log-Dice": "Log-Dice",
        "Dice Coefficient": "Dice",
        "Mutual Information": "MI"
    }
    y_col = measure_col_map.get(stat_measure, "LL")
    
    # Display each pattern group
    for pattern in patterns:
        label = pattern['label']
        pattern_str = pattern['pattern_str']
        
        if label not in groups:
            continue
            
        group_data = groups[label]
        
        # Handle backward compatibility: check if group_data is Dict (new) or DataFrame (old)
        if isinstance(group_data, dict):
            df_group = group_data.get('df', pd.DataFrame())
            examples = group_data.get('examples', {})
        else:
            # Old structure: group_data is the DataFrame itself
            df_group = group_data
            examples = {}
        
        if df_group.empty:
            continue
        
        with st.expander(f"📌 {label} ({len(df_group)} collocates)", expanded=True):
            st.caption(f"Pattern: `{pattern_str}`")
            
            # Show examples directly (one per collocate)
            st.markdown("**Matching Examples (Representative Instance per Collocate):**")
            
            # Sort collocates by chosen measure to show top ones first
            sort_col = y_col if y_col in df_group.columns else 'LL'
            sorted_df = df_group.sort_values(sort_col, ascending=False)
            
            # Pagination logic: show 5 by default, expand if button clicked
            limit = 5
            total = len(sorted_df)
            show_all_key = f"show_all_pattern_{label}_{key_suffix}"
            show_all = st.session_state.get(show_all_key, False)
            
            display_df = sorted_df if show_all else sorted_df.head(limit)
            
            for idx, (_, row) in enumerate(display_df.iterrows(), 1):
                collocate = row['Collocate']
                score_val = row.get(sort_col, 0)
                
                # Helper for fallback matching
                def _matches_item(t_dict, val):
                    if not val: return False
                    l_token = t_dict['token'].lower()
                    l_val = val.lower()
                    if l_val.startswith('[') and l_val.endswith(']'):
                        return t_dict.get('lemma', '').lower() == l_val[1:-1]
                    if l_val.startswith('_'):
                        return t_dict.get('pos', '') == val[1:]
                    return l_token == l_val

                example_data = examples.get(collocate)
                if example_data:
                    # check if list of examples
                    if not isinstance(example_data, list):
                        example_list = [example_data]
                    else:
                        example_list = example_data

                    if len(example_list) == 1:
                        ex_item = example_list[0]
                        if isinstance(ex_item, tuple) and len(ex_item) == 3:
                            conc_line, node_idx, coll_idx = ex_item
                        else:
                            conc_line = ex_item
                            node_idx = next((i for i, t in enumerate(conc_line) if _matches_item(t, node_word)), -1)
                            coll_idx = next((i for i, t in enumerate(conc_line) if _matches_item(t, collocate)), -1)
                        
                        parts = []
                        for i, t in enumerate(conc_line):
                            txt = t['token']
                            if i == node_idx:
                                parts.append(f"<span style='color: #00FFF5; font-weight: bold'>{txt}</span>")
                            elif i == coll_idx:
                                parts.append(f"<span style='color: #FF5252; font-weight: bold'>{txt}</span>")
                            else:
                                parts.append(f"<span style='color: #888'>{txt}</span>")
                        
                        display_html = " ".join(parts)
                        st.markdown(
                            f"{idx}. **{collocate}** ({score_val:.2f}): {display_html}",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(f"{idx}. **{collocate}** ({score_val:.2f})")
                        for sub_idx, ex_item in enumerate(example_list):
                            if isinstance(ex_item, tuple) and len(ex_item) == 3:
                                conc_line, node_idx, coll_idx = ex_item
                            else:
                                conc_line = ex_item
                                node_idx = next((i for i, t in enumerate(conc_line) if _matches_item(t, node_word)), -1)
                                coll_idx = next((i for i, t in enumerate(conc_line) if _matches_item(t, collocate)), -1)
                            
                            parts = []
                            for i, t in enumerate(conc_line):
                                txt = t['token']
                                if i == node_idx:
                                    parts.append(f"<span style='color: #00FFF5; font-weight: bold'>{txt}</span>")
                                elif i == coll_idx:
                                    parts.append(f"<span style='color: #FF5252; font-weight: bold'>{txt}</span>")
                                else:
                                    parts.append(f"<span style='color: #888'>{txt}</span>")
                            
                            display_html = " ".join(parts)
                            st.markdown(
                                f"&nbsp;&nbsp;&nbsp;&nbsp;• ({sub_idx+1}) {display_html}",
                                unsafe_allow_html=True
                            )
                else:
                    # If no example stored, show the collocate info at least
                    st.markdown(f"{idx}. **{collocate}** ({score_val:.2f})")
            
            # Show more button
            if total > limit:
                if not show_all:
                    if st.button(f"Show remaining {total - limit} types", key=f"btn_more_{label}_{key_suffix}"):
                        st.session_state[show_all_key] = True
                        st.rerun()
                else:
                    if st.button("Show less", key=f"btn_less_{label}_{key_suffix}"):
                        st.session_state[show_all_key] = False
                        st.rerun()
            
            # Download button for this pattern group
            st.download_button(
                label=f"Download '{label}' Group (Excel)",
                data=df_to_excel_bytes(df_group),
                file_name=f"pattern_{label.replace(' ', '_')}_{node_word}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_pattern_{label}_{key_suffix}"
            )


def run_multi_node_collocation_query(path, nodes, window, min_freq, max_rows, stats, xml_where, xml_params,
                                     token_filter="", pos_filter="", lemma_filter="", stat_measure="Log-Likelihood"):
    with st.spinner("Computing multi-node collocations..."):
        all_results = {}
        for node in nodes:
            df, freq, node_mwu = cached_generate_collocation(
                db_path=path,
                word=node,
                window=window,
                min_freq=min_freq,
                max_rows=max_rows,
                is_raw=False,
                corpus_stats=stats,
                xml_where_clause=xml_where,
                xml_params=xml_params,
                token_filter=token_filter,
                pos_filter=pos_filter,
                lemma_filter=lemma_filter,
                stat_measure=stat_measure
            )
            all_results[node] = {
                'df': df,
                'freq': freq,
                'node_mwu': node_mwu
            }
            
        st.session_state['last_multi_node_results'] = {
            'nodes': nodes,
            'results': all_results,
            'stat_measure': stat_measure
        }


def render_multi_node_results():
    multi_data = st.session_state.get('last_multi_node_results')
    if not multi_data:
        return
        
    nodes = multi_data['nodes']
    results = multi_data['results']
    stat_measure = multi_data['stat_measure']
    
    st.markdown("---")
    st.subheader("📊 Multi-Node Collocation Analysis Results")
    
    # Gather all collocates from all dataframes
    all_collocates = set()
    collocate_to_nodes = {}
    
    measure_col_map = {
        "Log-Likelihood": "LL",
        "Log-Dice": "Log-Dice",
        "Dice Coefficient": "Dice",
        "Mutual Information": "MI"
    }
    y_col = measure_col_map.get(stat_measure, "LL")
    
    for node, data in results.items():
        df = data['df']
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                coll = row['Collocate']
                coll_lower = coll.lower()
                all_collocates.add(coll_lower)
                
                score = row[y_col]
                freq = row['Observed']
                
                if coll_lower not in collocate_to_nodes:
                    collocate_to_nodes[coll_lower] = {
                        'word_original': coll,
                        'nodes': {}
                    }
                collocate_to_nodes[coll_lower]['nodes'][node] = {
                    'score': score,
                    'freq': freq
                }
                
    if not all_collocates:
        st.info("No collocates found for the specified node words with the current settings.")
        return
        
    # Calculate overlap lists
    shared_items = []
    unique_items = {node: [] for node in nodes}
    
    for coll_lower, data in collocate_to_nodes.items():
        associated_nodes = list(data['nodes'].keys())
        overlap_degree = len(associated_nodes)
        
        row_dict = {
            'Collocate': data['word_original'],
            'Degree': f"{overlap_degree}/{len(nodes)}",
            'degree_num': overlap_degree,
            'Associated Nodes': ", ".join(associated_nodes)
        }
        
        for node in nodes:
            if node in data['nodes']:
                row_dict[f"{node} Freq"] = data['nodes'][node]['freq']
                row_dict[f"{node} Score"] = round(data['nodes'][node]['score'], 2)
            else:
                row_dict[f"{node} Freq"] = 0
                row_dict[f"{node} Score"] = 0.0
                
        if overlap_degree > 1:
            shared_items.append(row_dict)
        else:
            node = associated_nodes[0]
            unique_items[node].append({
                'Collocate': data['word_original'],
                'Observed': data['nodes'][node]['freq'],
                stat_measure: round(data['nodes'][node]['score'], 2)
            })
            
    shared_df = pd.DataFrame(shared_items)
    
    # 1. Visualization Controls at top of results section
    st.markdown("#### ⚙️ Chart Configuration & Sorting")
    
    # Get or initialize the sort state
    sort_mode = st.session_state.get('multi_node_sort_mode', 'shared')
    calc_method = st.session_state.get('multi_node_calc_method', 'Simple Aggregate')
    prettify = st.session_state.get('multi_node_prettify', False)
    
    col_btn1, col_btn2, col_prettify = st.columns([1.5, 1.5, 1])
    with col_btn1:
        if st.button(
            "🔢 Sort by Shared Nodes", 
            key="btn_sort_shared", 
            type="primary" if sort_mode == 'shared' else "secondary",
            use_container_width=True
        ):
            st.session_state['multi_node_sort_mode'] = 'shared'
            st.rerun()
            
    with col_btn2:
        if st.button(
            "⚡ Sort by Combined Strength", 
            key="btn_sort_strength", 
            type="primary" if sort_mode == 'strength' else "secondary",
            use_container_width=True
        ):
            st.session_state['multi_node_sort_mode'] = 'strength'
            st.rerun()
            
    # Method radio button
    calc_method = st.radio(
        "Combined Strength Calculation Method:",
        ["Simple Aggregate", "Harmonic Mean", "Min-Max Normalization", "Z-Score Standardization"],
        horizontal=True,
        key="multi_node_calc_method"
    )
    
    prettify = st.checkbox(
        "✨ Prettify Scale",
        value=prettify,
        help="Applies a log scale to equalize the visual lengths of bars, making smaller association scores visible next to large ones.",
        key="multi_node_prettify"
    )
    
    # Calculate baseline stats for each node (for normalization/Z-score)
    import math
    node_stats = {}
    for node in nodes:
        df_node = results[node]['df']
        if df_node is not None and not df_node.empty:
            scores = df_node[y_col].astype(float).tolist()
            mean_val = sum(scores) / len(scores) if scores else 0.0
            variance = sum((x - mean_val) ** 2 for x in scores) / len(scores) if scores else 0.0
            std_val = math.sqrt(variance) if variance > 0 else 1.0
            min_val = min(scores) if scores else 0.0
            max_val = max(scores) if scores else 1.0
            node_stats[node] = {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val
            }
        else:
            node_stats[node] = {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0}
            
    def get_calc_score(node, raw_score):
        if raw_score <= 0:
            return 0.0
        stats = node_stats.get(node, {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0})
        
        if calc_method == "Min-Max Normalization":
            denom = stats['max'] - stats['min']
            if denom == 0:
                return 0.0
            return max(0.0, (raw_score - stats['min']) / denom)
            
        elif calc_method == "Z-Score Standardization":
            return max(0.0, (raw_score - stats['mean']) / stats['std'])
            
        return raw_score

    if not shared_df.empty:
        # Calculate Combined Score for each row dynamically
        combined_scores = []
        for _, row in shared_df.iterrows():
            raw_scores = [float(row.get(f"{node} Score", 0.0)) for node in nodes]
            
            if calc_method == "Harmonic Mean":
                active = [s for s in raw_scores if s > 0]
                val = len(active) / sum(1.0 / s for s in active) if active else 0.0
            elif calc_method == "Min-Max Normalization":
                val = sum(get_calc_score(node, s) for node, s in zip(nodes, raw_scores))
            elif calc_method == "Z-Score Standardization":
                val = sum(get_calc_score(node, s) for node, s in zip(nodes, raw_scores))
            else: # Simple Aggregate
                val = sum(raw_scores)
                
            combined_scores.append(round(val, 2))
            
        shared_df['Combined Score'] = combined_scores
        
        # Apply sorting to shared_df
        if sort_mode == 'shared':
            shared_df = shared_df.sort_values(by=['degree_num', 'Combined Score'], ascending=[False, False]).reset_index(drop=True)
        else:
            shared_df = shared_df.sort_values(by=['Combined Score', 'degree_num'], ascending=[False, False]).reset_index(drop=True)
        
    tab_shared, tab_unique, tab_graph = st.tabs(["🔗 Shared Collocates", "💎 Unique Collocates", "📊 Visual Comparison"])
    
    with tab_shared:
        st.markdown(f"Collocates appearing with more than one node word, sorted by **Overlap Degree**:")
        if not shared_df.empty:
            display_cols = ['Collocate', 'Degree', 'Associated Nodes']
            for node in nodes:
                display_cols.append(f"{node} Score")
            display_cols.append('Combined Score')
            
            st.dataframe(shared_df[display_cols], use_container_width=True)
            
            from core.io_utils import df_to_excel_bytes
            st.download_button(
                label="Download Shared Collocates (Excel)",
                data=df_to_excel_bytes(shared_df[display_cols]),
                file_name="shared_multi_node_collocates.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_shared_multi_node"
            )
        else:
            st.info("No shared collocates found among these node words.")
            
    with tab_unique:
        st.markdown("Collocates exclusive to a single node word:")
        unique_tabs = st.tabs([f"Only with '{node}'" for node in nodes])
        for idx, node in enumerate(nodes):
            with unique_tabs[idx]:
                u_list = unique_items[node]
                if u_list:
                    u_df = pd.DataFrame(u_list)
                    u_df = u_df.sort_values(by=[stat_measure], ascending=False).reset_index(drop=True)
                    st.dataframe(u_df, use_container_width=True)
                    
                    from core.io_utils import df_to_excel_bytes
                    st.download_button(
                        label=f"Download Unique Collocates for {node}",
                        data=df_to_excel_bytes(u_df),
                        file_name=f"unique_collocates_{node}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"dl_unique_multi_node_{node}"
                    )
                else:
                    st.info(f"No unique collocates found for '{node}'.")

    with tab_graph:
        render_multi_node_charts(nodes, shared_df, stat_measure, calc_method, prettify, get_calc_score)


def render_multi_node_charts(nodes, shared_df, stat_measure, calc_method, prettify, get_calc_score_fn):
    if shared_df.empty:
        st.info("No shared collocates to visualize.")
        return

    import plotly.express as px
    import math
    
    # Prepare data for individual collocates (Top 25 shared collocates)
    top_n = 25
    top_shared = shared_df.head(top_n)
    
    # Extract categories for y-axis ordering (Plotly orders bottom-to-top, so reverse the list)
    collocate_order = top_shared['Collocate'].tolist()[::-1]
    
    heat_data = []
    for _, row in top_shared.iterrows():
        coll = row['Collocate']
        for node in nodes:
            raw_score = float(row.get(f"{node} Score", 0.0))
            calc_score = get_calc_score_fn(node, raw_score)
            
            # Harmonic mean is computed collectively, so segment sizes are best kept as raw values
            # otherwise display normalized segment score
            if calc_method == "Harmonic Mean":
                display_score = raw_score
            else:
                display_score = calc_score
                
            freq = row.get(f"{node} Freq", 0)
            heat_data.append({
                'Collocate': coll,
                'Node Word': node,
                'Score': raw_score,
                'Calc Score': display_score,
                'Frequency': int(freq)
            })
            
    df_heat = pd.DataFrame(heat_data)
    
    # Apply scaling for visualization if Prettify is enabled
    if prettify:
        df_heat['Viz Score'] = df_heat['Calc Score'].apply(lambda x: math.log1p(max(0, x)))
        x_col = 'Viz Score'
        x_label = f"{calc_method} (Log Scaled)"
    else:
        df_heat['Viz Score'] = df_heat['Calc Score']
        x_col = 'Viz Score'
        x_label = calc_method
        
    # Prepare data for Overlap Distribution
    overlap_counts = shared_df.groupby('Associated Nodes').size().reset_index(name='Collocate Count')
    overlap_counts = overlap_counts.sort_values(by='Collocate Count', ascending=True)
    
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "🔮 Bubble Matrix (Clean Grid)",
        "📊 Stacked Bar Chart",
        "🕸️ Overlap Size Overview",
        "🕸️ Network"
    ])
    
    with viz_tab1:
        st.markdown(f"**Bubble Matrix**: Bubble size represents **Frequency** (co-occurrence count), and bubble color represents **{stat_measure}**.")
        fig_bubble = px.scatter(
            df_heat,
            x='Node Word',
            y='Collocate',
            size='Frequency',
            color='Score',
            color_continuous_scale='Viridis',
            labels={'Score': stat_measure, 'Frequency': 'Frequency'},
            hover_data=['Collocate', 'Node Word', 'Score', 'Frequency'],
            title=f"Shared Collocates Bubble Grid (Top {top_n})"
        )
        fig_bubble.update_layout(
            height=max(450, 25 * len(top_shared) + 120),
            xaxis_title="Node Word",
            yaxis_title="Shared Collocate",
            yaxis={'categoryorder': 'array', 'categoryarray': collocate_order},
            margin=dict(l=150, r=20, t=50, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        fig_bubble.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
        fig_bubble.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
        st.plotly_chart(fig_bubble, use_container_width=True)
        
    with viz_tab2:
        st.markdown(f"**Stacked Bar Chart**: Displays the combined strength of each collocate. Segments represent individual node scores.")
        fig_bar = px.bar(
            df_heat,
            x=x_col,
            y='Collocate',
            color='Node Word',
            barmode='stack',
            orientation='h',
            labels={x_col: x_label, 'Collocate': 'Shared Collocate'},
            hover_data={'Score': True, 'Viz Score': False, 'Frequency': True, 'Node Word': True, 'Collocate': True},
            color_discrete_sequence=px.colors.qualitative.Plotly,
            title=f"Combined Association Strength (Top {top_n})"
        )
        fig_bar.update_layout(
            height=max(450, 25 * len(top_shared) + 100),
            yaxis={'categoryorder': 'array', 'categoryarray': collocate_order},
            margin=dict(l=150, r=20, t=50, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with viz_tab3:
        st.markdown("**Overlap Size Overview**: Node combinations mapped as horizontal bars containing their collocates (larger font sizes represent higher combined strength).")
        
        # Group collocates by their exact combination
        combo_groups = {}
        for _, row in shared_df.iterrows():
            combo = row['Associated Nodes']
            coll = row['Collocate']
            score = row['Combined Score']
            if combo not in combo_groups:
                combo_groups[combo] = []
            combo_groups[combo].append((coll, score))
            
        # Sort combinations by the number of collocates descending
        sorted_combos = sorted(combo_groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        if sorted_combos:
            html_lines = [
                """
                <style>
                .overlap-bar-container {
                    margin-bottom: 25px;
                    font-family: 'Inter', system-ui, -apple-system, sans-serif;
                }
                .overlap-label {
                    font-weight: bold;
                    font-size: 0.95rem;
                    color: #e0e0e0;
                    margin-bottom: 8px;
                }
                .overlap-bar {
                    background: linear-gradient(90deg, rgba(0, 255, 245, 0.08) 0%, rgba(33, 150, 243, 0.03) 100%);
                    border: 1px solid rgba(0, 255, 245, 0.2);
                    border-radius: 12px;
                    padding: 14px 18px;
                    display: flex;
                    flex-wrap: wrap;
                    align-items: center;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                }
                .overlap-bar:hover {
                    border-color: rgba(0, 255, 245, 0.5);
                    background: linear-gradient(90deg, rgba(0, 255, 245, 0.12) 0%, rgba(33, 150, 243, 0.06) 100%);
                    box-shadow: 0 6px 20px rgba(0, 255, 245, 0.1);
                    transform: translateY(-1px);
                }
                .overlap-word {
                    display: inline-block;
                    margin: 6px 12px;
                    font-weight: 600;
                    cursor: help;
                    transition: all 0.2s ease;
                }
                .overlap-word:hover {
                    transform: scale(1.15);
                    text-shadow: 0 0 8px rgba(0, 255, 245, 0.5);
                }
                </style>
                """
            ]
            
            max_count = max([len(words) for combo, words in sorted_combos]) if sorted_combos else 1
            
            for combo, words in sorted_combos:
                # Sort words by score descending
                words = sorted(words, key=lambda x: x[1], reverse=True)
                count = len(words)
                
                # Proportional width of the bar (from 50% to 100% width)
                width_pct = int(50 + 50 * (count / max_count))
                
                # Normalize font sizes for words in this specific combo
                scores = [w[1] for w in words]
                min_s = min(scores) if scores else 0
                max_s = max(scores) if scores else 1
                range_s = max_s - min_s if max_s != min_s else 1
                
                word_spans = []
                for word, score in words:
                    # Font size scaling between 12px and 28px
                    font_size = 12 + 16 * ((score - min_s) / range_s)
                    
                    # Highlight colors based on score levels
                    if score > min_s + 0.6 * range_s:
                        color = "#00FFF5"  # High strength - Cyan
                    elif score > min_s + 0.25 * range_s:
                        color = "#64B5F6"  # Medium strength - Soft blue
                    else:
                        color = "#E0E0E0"  # Low strength - Off-white
                        
                    word_spans.append(
                        f'<span class="overlap-word" style="font-size: {font_size:.1f}px; color: {color}" title="Combined Score: {score:.1f}">{word}</span>'
                    )
                
                spans_html = "".join(word_spans)
                html_lines.append(f"""
                <div class="overlap-bar-container" style="width: {width_pct}%;">
                    <div class="overlap-label">🔗 {combo} ({count} words)</div>
                    <div class="overlap-bar">
                        {spans_html}
                    </div>
                </div>
                """)
                
            st.markdown("\n".join(html_lines), unsafe_allow_html=True)
        else:
            st.info("No overlap groups detected.")

    with viz_tab4:
        render_collocation_network(nodes, shared_df, key_suffix="multi")

    # Excel download button for visual comparison data
    st.markdown("---")
    st.markdown("##### ⬇️ Download Visual Comparison Chart Data")
    from core.io_utils import df_to_excel_bytes
    dl_cols = ['Collocate', 'Degree', 'Associated Nodes'] + [f"{node} Score" for node in nodes] + ['Combined Score']
    st.download_button(
        label="Download Chart Data (Excel)",
        data=df_to_excel_bytes(top_shared[dl_cols]),
        file_name="multi_node_visual_comparison_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_multi_node_chart_data"
    )

def render_collocation_network(nodes, shared_df, key_suffix=""):
    import networkx as nx
    from pyvis.network import Network
    import tempfile
    import os
    
    st.markdown("##### 🕸️ Collocation Network")
    st.markdown(
        "Visualise how collocates are shared across the comparison node words. "
        "Shared collocates will cluster in the centre between the nodes they belong to."
    )
    
    # 1. Controls
    c1, c2 = st.columns(2)
    with c1:
        top_n = st.number_input(
            "Top N Collocates",
            min_value=5,
            max_value=200,
            value=25,
            key=f"coll_net_top_{key_suffix}"
        )
    with c2:
        show_shared_only = st.toggle(
            "Show Only Shared Collocates",
            value=False,
            help="Hides collocates that are unique to a single node word.",
            key=f"coll_net_shared_{key_suffix}"
        )
        
    # Parse Degree column (which may contain string fractions like '4/4') as an integer
    def parse_degree(d):
        d_str = str(d)
        if '/' in d_str:
            try:
                return int(d_str.split('/')[0])
            except:
                pass
        try:
            return int(float(d_str))
        except:
            return 1
            
    shared_df_copy = shared_df.copy()
    shared_df_copy['ParsedDegree'] = shared_df_copy['Degree'].apply(parse_degree)
        
    # Get top N shared collocates
    df_sorted = shared_df_copy.sort_values(by='Combined Score', ascending=False)
    
    if show_shared_only:
        df_sorted = df_sorted[df_sorted['ParsedDegree'] >= 2]
        
    df_top = df_sorted.head(top_n)
    
    if df_top.empty:
        st.info("No collocates match the current filters.")
        return
        
    G = nx.Graph()
    
    NODE_COLORS = [
        "#FF6B6B", "#4D96FF", "#6BCB77", "#FFD93D", "#9B5DE5", 
        "#F15BB5", "#00F5D4", "#00BBF9", "#F77F00", "#D62828"
    ]
    
    # Add Node Word nodes
    for i, node in enumerate(nodes):
        color = NODE_COLORS[i % len(NODE_COLORS)]
        G.add_node(
            node,
            label=str(node),
            color=color,
            size=35,
            font={'size': 44, 'color': '#ffffff', 'strokeWidth': 5, 'strokeColor': '#000000'},
            shape="dot",
            title=f"Node Word: {node}"
        )
        
    # Count collocate sharing
    added_collocates = set()
    edges_to_add = []
    
    for _, row in df_top.iterrows():
        coll = row['Collocate']
        degree = int(row['ParsedDegree'])
        combined_score = float(row['Combined Score'])
        
        # Add collocate node
        if coll not in added_collocates:
            node_size = 12 + (degree * 4)
            node_color = "#00FFF5" if degree > 1 else "#a5b4fc"
            G.add_node(
                coll,
                label=str(coll),
                color=node_color,
                size=node_size,
                font={'size': 36, 'color': '#ffffff', 'strokeWidth': 3, 'strokeColor': '#000000'},
                shape="dot",
                title=f"Collocate: {coll}\nShared by {degree} nodes\nCombined Score: {combined_score:.2f}"
            )
            added_collocates.add(coll)
            
        # Add edges to connected node words
        for node in nodes:
            score_col = f"{node} Score"
            if score_col in row and float(row[score_col]) > 0:
                edges_to_add.append((node, coll))
                
    G.add_edges_from(edges_to_add)
    
    # Clean up node words with 0 degrees
    isolated = [node for node in G.nodes() if G.degree(node) == 0]
    G.remove_nodes_from(isolated)
    
    if len(G.nodes) == 0:
        st.info("The network is empty.")
        return
        
    # Pre-calculate layout static positions using networkx spring_layout
    # Using a tighter k and coordinate scaling multiplier to shorten branch distances
    pos = nx.spring_layout(G, k=1.0 / (len(G.nodes) ** 0.5) if len(G.nodes) > 0 else 0.2, iterations=50)
    for n_id, coords in pos.items():
        G.nodes[n_id]['x'] = float(coords[0] * 550)
        G.nodes[n_id]['y'] = float(coords[1] * 550)
        
    # Render using Pyvis
    with st.spinner("Generating collocation network..."):
        net = Network(
            height="1200px", 
            width="100%", 
            bgcolor="#0f172a", 
            font_color="#ffffff", 
            notebook=False
        )
        net.from_nx(G)
        
        physics_json = """
        {
          "physics": {
            "enabled": false
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "zoomView": true
          },
          "edges": {
            "color": {
              "color": "rgba(255, 255, 255, 0.18)",
              "hover": "rgba(0, 255, 245, 0.8)",
              "highlight": "rgba(0, 255, 245, 0.8)"
            },
            "width": 1.2,
            "smooth": {
              "type": "continuous"
            }
          }
        }
        """
        net.set_options(physics_json)
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                tmp_path = tmp.name
            net.write_html(tmp_path)
            
            with open(tmp_path, "r", encoding="utf-8") as f:
                html_content = f.read()
                
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
            # Replace white background styles from pyvis template
            html_content = html_content.replace(
                "background-color: #ffffff;",
                "background-color: #0f172a;"
            )
            html_content = html_content.replace(
                "border: 1px solid lightgray;",
                "border: 1px solid rgba(255, 255, 255, 0.1);"
            )
            
            st.components.v1.html(html_content, height=1240, scrolling=False)
            
        except Exception as e:
            st.error(f"Failed to render pyvis network: {e}")
