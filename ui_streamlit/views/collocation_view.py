import streamlit as st
import pandas as pd
from ui_streamlit.state_manager import get_state, set_state
from ui_streamlit.caching import cached_generate_collocation
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.preprocessing.xml_parser import apply_xml_restrictions
from core.ai_service import interpret_results_llm, parse_nl_query, parse_nl_query_rules_only
from core.visualiser.network import create_pyvis_graph
from core.io_utils import df_to_excel_bytes
from ui_streamlit.caching import cached_generate_kwic, cached_get_subcorpus_size
from ui_streamlit.components.result_display import render_kwic_table

def render_collocation_view():
    st.header("Collocation Analysis")
    
    corpus_path = get_state('current_corpus_path')
    corpus_name = get_state('current_corpus_name', 'Corpus')
    corpus_stats = get_state('corpus_stats')
    
    if not corpus_path:
        st.warning("Please load a corpus first.")
        return

    comp_mode = get_state('comparison_mode', False)
    comp_path = get_state('comp_corpus_path')
    comp_name = get_state('comp_corpus_name')

    # 1. Inputs
    search_mode = st.radio("Search Mode", ["Standard", "Natural Language (Rule)", "Natural Language (AI)"], horizontal=True, key="coll_search_mode")
    
    if search_mode == "Natural Language (AI)":
        st.markdown("### 🧠 Natural Language Search")
        nl_query = st.text_area("Describe what you want to find", height=70, placeholder="e.g. Find adjectives that appear within 3 words of 'environment' appearing at least 5 times")
        
        col_ai1, col_ai2 = st.columns([1, 4])
        with col_ai1:
            analyze_btn = st.button("Analyze & Search", type="primary")
        
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
                    
                    # Execute Search
                    # We reuse the logic for 'primary' search since NL is typically single-corpus focus
                    xml_filters = render_xml_restriction_filters(corpus_path, "collocation", corpus_name=corpus_name)
                    xml_where, xml_params = apply_xml_restrictions(xml_filters)
                    
                    run_collocation_query(
                        'primary', corpus_path, 
                        params.get('node_word', ''), 
                        win, 
                        freq, 
                        mx, 
                        corpus_stats, 
                        xml_where, xml_params, 
                        params.get('token_filter', ''), 
                        params.get('pos_filter', ''), 
                        params.get('lemma_filter', ''),
                        '', 
                        50
                    )
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
            
            st.markdown("---")
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                token_filter_input = st.text_input("Token Filter (NL)", placeholder="e.g. not 'the'", key="coll_token_filt_rule")
            with f_col2:
                pos_filter_input = st.text_input("POS Filter (NL)", placeholder="e.g. noun, verb", key="coll_pos_filt_rule")
            with f_col3:
                lemma_filter_input = st.text_input("Lemma Filter (NL)", placeholder="e.g. be, have", key="coll_lemma_filt_rule")
                
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
                 if not comp_mode:
                     node_word = st.text_input("Node Word", value="", placeholder="e.g. beautiful, [lemma]*, _VB*, *kan", key="coll_node")
                 else:
                     st.markdown("**Node Words**")
                     node_primary = st.text_input(f"Primary ({get_state('current_corpus_name', 'Corpus')})", value="", key="coll_node_primary")
                     node_secondary = st.text_input(f"Comparison ({comp_name if comp_name else 'Secondary'})", value="", key="coll_node_secondary")
                     node_word = node_primary # Default for single-path logic logic below if needed, though we split paths
                     
            with col2:
                 # Shared settings for now
                 window = st.slider("Span (Window)", 1, 10, 5, key="coll_window")
                 c_sub1, c_sub2 = st.columns(2)
                 with c_sub1:
                    min_freq = st.number_input("Min Co-occurrence", 1, 100, 3, key="coll_min_freq")
                 with c_sub2:
                    max_rows = st.number_input("Max Collocates", 10, 50000, 100, step=10, key="coll_max", help="Increase this limit to download more results (up to 50,000).")
            
            st.markdown("---")
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                token_filter = st.text_input("Token Filter", placeholder="e.g. no, non OR -no, -non", key="coll_token_filt")
            with f_col2:
                pos_filter = st.text_input("POS Filter", placeholder="e.g. JJ, NN OR -JJ, -NN", key="coll_pos_filt")
            with f_col3:
                lemma_filter = st.text_input("Lemma Filter", placeholder="e.g. see OR -see", key="coll_lemma_filt")
            
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


    if not comp_mode:
        xml_filters = render_xml_restriction_filters(corpus_path, "collocation", corpus_name=corpus_name)
        xml_where, xml_params = apply_xml_restrictions(xml_filters)
        
        if st.button("Calculate Collocations", type="primary"):
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
                    pos_defs = get_pos_definitions(corpus_path) or {}
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
                    pattern_text if apply_patterns else '', pattern_limit
                )
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
                st.info("Load a comparison corpus in sidebar.")
                xml_where_2, xml_params_2 = "", []
            
        if st.button("Calculate Comparison Collocations", type="primary"):
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
                     
                     pos_defs = get_pos_definitions(corpus_path) or {}
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
                
                # Run Primary
                if to_run_node_1:
                    run_collocation_query(
                        'primary', corpus_path, to_run_node_1, to_run_win, to_run_min_freq, to_run_max, 
                        corpus_stats, xml_where_1, xml_params_1, to_run_tok, to_run_pos, to_run_lem,
                        pattern_text if apply_patterns else '', pattern_limit
                    )
                else:
                     st.warning("Primary node word missing.")
                
                # Run Comparison
                if comp_path and to_run_node_2:
                    comp_stats = get_state('comp_corpus_stats')
                    run_collocation_query(
                        'secondary', comp_path, to_run_node_2, to_run_win, to_run_min_freq, to_run_max, 
                        comp_stats, xml_where_2, xml_params_2, to_run_tok, to_run_pos, to_run_lem,
                        pattern_text if apply_patterns else '', pattern_limit
                    )
                elif comp_path and not to_run_node_2:
                    st.warning("Comparison node word missing.")

    # 3. Display
    if not comp_mode:
        results = st.session_state.get('last_coll_results_primary')
        if results:
            render_collocation_results_column(results)
            
            # Display pattern results if available
            pattern_results = st.session_state.get('pattern_results_primary')
            if pattern_results:
                render_pattern_results(pattern_results, results, 'primary')
    else:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.subheader(f"Primary: {get_state('current_corpus_name', 'Corpus')}")
            res1 = st.session_state.get('last_coll_results_primary')
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
                          token_filter="", pos_filter="", lemma_filter="", pattern_text="", pattern_limit=50):
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
            lemma_filter=lemma_filter
        )
        st.session_state[f'last_coll_results_{identifier}'] = {
            'df': stats_df,
            'freq': freq,
            'node': node_mwu,
            'window': window,
            'corpus_name': identifier,
            'xml_where': xml_where,
            'xml_params': xml_params
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
                            xml_params=xml_params
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
     
     if not df.empty:
         st.markdown(f"**{len(df)} collocates** for '{node}' (Freq: {n_freq}) within ±{win}).")
         
         st.download_button(
             label=f"⬇ Download {results.get('corpus_name', 'Corpus')} Collocations (Excel)",
             data=df_to_excel_bytes(df),
             file_name=f"collocations_{node}.xlsx",
             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
             key=f"dl_coll_{key_suffix}_top"
         )
         
         tab_table, tab_charts, tab_graph = st.tabs(["Table", "Charts (LL)", "Network Graph"])
         with tab_table:
             df_display = df.reset_index(drop=True)
             df_display.index += 1
             st.dataframe(df_display, use_container_width=True)

         with tab_charts:
             import matplotlib.pyplot as plt
             c1, c2, c3 = st.columns(3)
             
             def plot_top_ll(data, title, color):
                 if data.empty:
                     st.info(f"No {title} data.")
                     return
                 fig, ax = plt.subplots(figsize=(5, 4))
                 top = data.head(10).sort_values("LL", ascending=True)
                 ax.barh(top['Collocate'], top['LL'], color=color)
                 ax.set_title(title)
                 plt.tight_layout()
                 st.pyplot(fig)

             with c1:
                 plot_top_ll(df, "Overall Top Collocates", "skyblue")
             with c2:
                 plot_top_ll(df[df['Direction'].isin(['L', 'B'])], "Left-Dominant", "salmon")
             with c3:
                 plot_top_ll(df[df['Direction'].isin(['R', 'B'])], "Right-Dominant", "lightgreen")
             
         with tab_graph:
             g_all, g_left, g_right = st.columns(3)
             with g_all:
                 st.caption("Overall Network (Top 30)")
                 graph_html = create_pyvis_graph(node, df.head(30))
                 if graph_html: st.components.v1.html(graph_html, height=400)
             with g_left:
                 st.caption("Left Network")
                 df_l = df[df['Direction'].isin(['L', 'B'])].head(20)
                 if not df_l.empty:
                     graph_html_l = create_pyvis_graph(node, df_l)
                     st.components.v1.html(graph_html_l, height=400)
                 else: st.info("No L collocates.")
             with g_right:
                 st.caption("Right Network")
                 df_r = df[df['Direction'].isin(['R', 'B'])].head(20)
                 if not df_r.empty:
                     graph_html_r = create_pyvis_graph(node, df_r)
                     st.components.v1.html(graph_html_r, height=400)
                 else: st.info("No R collocates.")

         st.markdown("### Usage Examples by Collocates")
         # Logic fix: Ensure db_path and name are passed correctly
         db_path = get_state('current_corpus_path') if key_suffix != 'c2' else get_state('comp_corpus_path')
         corpus_name = get_state('current_corpus_name') if key_suffix != 'c2' else get_state('comp_corpus_name')
         
         # Get restrictions from results
         xml_where = results.get('xml_where', "")
         xml_params = results.get('xml_params', [])

         with st.expander("Show Examples", expanded=False):
             kwic_table_data = []
             # Get parallel data if needed
             is_parallel = get_state('parallel_mode', False)
             target_map = get_state('target_sent_map', {})
             
             for i, coll in enumerate(df['Collocate'].tolist()[:10]): # Top 10 collocates
                  with st.spinner(f"Fetching example for {coll}..."):
                       # Unpack properly to get sent_ids
                       c_kwic, _, _, _, c_sent_ids, _ = cached_generate_kwic(
                           db_path=db_path, 
                           query=node, 
                           left=7, 
                           right=7, 
                           corpus_name=corpus_name,
                           pattern_collocate_input=coll, 
                           pattern_window=win, 
                           limit=1,
                           xml_where_clause=xml_where,
                           xml_params=xml_params
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
                               row_data['Translation'] = trans
                               
                           kwic_table_data.append(row_data)

             if kwic_table_data:
                 render_kwic_table(kwic_table_data, is_parallel=is_parallel, target_lang=get_state('tgt_lang_code', 'Target'))
             else:
                 st.info("No examples found.")

         st.markdown("---")
         if st.button("Interpret with AI", key=f"btn_coll_ai_{key_suffix}"):
              with st.spinner("Analyzing..."):
                   top_data = df.head(10).to_string(index=False)
                   resp, err = interpret_results_llm(
                        target_word=node,
                        analysis_type="Collocation Analysis",
                        data_description=f"Collocates for '{node}' (±{win}).",
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
            
            # Sort collocates by LL to show top ones first
            sorted_df = df_group.sort_values('LL', ascending=False)
            
            # Pagination logic: show 5 by default, expand if button clicked
            limit = 5
            total = len(sorted_df)
            show_all_key = f"show_all_pattern_{label}_{key_suffix}"
            show_all = st.session_state.get(show_all_key, False)
            
            display_df = sorted_df if show_all else sorted_df.head(limit)
            
            for idx, (_, row) in enumerate(display_df.iterrows(), 1):
                collocate = row['Collocate']
                ll_score = row.get('LL', 0)
                
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
                    # example_data is (conc_line, node_idx, coll_idx)
                    if isinstance(example_data, tuple) and len(example_data) == 3:
                        conc_line, node_idx, coll_idx = example_data
                    else:
                        # Fallback for old session state or single search
                        conc_line = example_data
                        node_idx = next((i for i, t in enumerate(conc_line) if _matches_item(t, node_word)), -1)
                        coll_idx = next((i for i, t in enumerate(conc_line) if _matches_item(t, collocate)), -1)
                    
                    # Construct display string
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
                        f"{idx}. **{collocate}** (LL: {ll_score:.2f}): {display_html}",
                        unsafe_allow_html=True
                    )
                else:
                    # If no example stored, show the collocate info at least
                    st.markdown(f"{idx}. **{collocate}** (LL: {ll_score:.2f})")
            
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
