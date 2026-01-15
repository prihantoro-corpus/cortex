import streamlit as st
import re
import pandas as pd
from ui_streamlit.state_manager import get_state, set_state
from ui_streamlit.caching import cached_generate_kwic, cached_get_subcorpus_size
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.preprocessing.xml_parser import apply_xml_restrictions
from core.ai_service import interpret_results_llm, parse_nl_query, parse_nl_query_rules_only
from core.io_utils import df_to_excel_bytes
from core.modules.overview import get_pos_definitions, get_corpus_language

def render_concordance_view():
    st.header("Concordance (KWIC)")
    
    corpus_path = get_state('current_corpus_path')
    corpus_name = get_state('current_corpus_name', 'Corpus')
    
    if not corpus_path:
        st.warning("Please load a corpus first.")
        return

    # 1. Controls
    search_mode = st.radio("Search Mode", ["Standard", "Natural Language (Rule)", "Natural Language (AI)"], horizontal=True, key="kwic_search_mode")
    search_term = get_state('kwic_search_term', '')
    
    if search_mode == "Natural Language (Rule)":
        st.markdown("### ⚡ Natural Language Search (Rule-Based)")
        st.caption("Fast, deterministic parsing without AI. Supports: 'followed by', 'preceded by', 'before', 'after', and POS terms like 'noun', 'verb', 'adjective'.")
        
        with st.expander("Search Controls", expanded=True):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                 nl_query = st.text_input("Natural Language Query", value=get_state('kwic_nl_query_rule', ''), key="kwic_nl_input_rule", help="e.g. any word followed by 'adjective'")
            with col2:
                 window_size = st.slider("Context Window", 1, 20, 5, key="kwic_window_rule")
            with col3:
                 limit = st.number_input("Max Lines", 10, 5000, 100, step=10, key="kwic_limit_rule")
             
            # Advanced Filters
            c_adv1, c_adv2 = st.columns(2)
            with c_adv1:
                coll_filter_input = st.text_input("Filter by Collocate (NL/Regex)", help="e.g. 'noun' or 'very'", key="kwic_coll_rule")
            with c_adv2:
                sort_order = st.selectbox("Sort By", ["Node (Default)", "Left Context", "Right Context"], key="kwic_sort_rule")
                c_sh1, c_sh2 = st.columns(2)
                with c_sh1:
                    show_pos = st.checkbox("Show POS", value=get_state('kwic_show_pos', False), key="kwic_show_pos_rule")
                with c_sh2:
                    show_lemma = st.checkbox("Show Lemma", value=get_state('kwic_show_lemma', False), key="kwic_show_lemma_rule")
                
                set_state('kwic_show_pos', show_pos)
                set_state('kwic_show_lemma', show_lemma)
                
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
                pos_defs = get_pos_definitions(corpus_path) or {}
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
                    
                    # Execute the search
                    xml_filters = render_xml_restriction_filters(corpus_path, "concordance", corpus_name=corpus_name)
                    xml_where, xml_params = apply_xml_restrictions(xml_filters)
                    
                    run_concordance_query('primary', corpus_path, corpus_name, 
                                          query, 
                                          window_size, 
                                          window_size,
                                          limit, 
                                          coll_filter_parsed, 
                                          xml_where, xml_params,
                                          show_pos, show_lemma)
                else:
                    st.error(f"Error parsing query: {err}")
    
    if search_mode == "Natural Language (AI)":
        st.markdown("### 🧠 Natural Language Search")
        nl_query = st.text_area("Describe your concordance query", height=70, placeholder="e.g. Find examples of 'make' followed by a noun")
        
        # Display Options for AI Mode
        with st.expander("Display Options"):
            wrap_mode = st.checkbox("Wrap Text", value=get_state('kwic_wrap_mode', True), key="kwic_wrap_mode_ai")
            set_state('kwic_wrap_mode', wrap_mode)
        
        col_ai1, col_ai2 = st.columns([1, 4])
        with col_ai1:
            analyze_btn = st.button("Search with AI", type="primary")
            
        if analyze_btn:
            if not nl_query:
                st.warning("Please enter a query.")
            else:
                with st.spinner("AI is determining search parameters..."):
                    # Fetch user definitions if available
                    pos_defs = get_pos_definitions(corpus_path) or {}
                    lang = get_corpus_language(corpus_path)
                    
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
                        win = int(params.get('window', 5))
                    except (ValueError, TypeError):
                        win = 5
                    set_state('kwic_window', win)
                    
                    # Sort
                    sort = params.get('sort_order', 'Node')
                    if 'left' in sort.lower(): set_state('kwic_sort_col_primary', 'Left') # We need to check exact key usage
                    elif 'right' in sort.lower(): set_state('kwic_sort_col_primary', 'Right')
                    
                    st.success(f"✓ Executing search for '{query}'...")
                    
                    # Execute
                    # Since params are simple, we pass defaults for others
                    xml_filters = render_xml_restriction_filters(corpus_path, "concordance", corpus_name=corpus_name)
                    xml_where, xml_params = apply_xml_restrictions(xml_filters)
                    
                    run_concordance_query('primary', corpus_path, corpus_name, 
                                          query, 
                                          win, 
                                          win,
                                          100, # Default limit
                                          "", # Coll filter
                                          xml_where, xml_params,
                                          False, False) # Show POS/Lemma defaults
                else:
                    st.error(f"Could not parse query: {err}")

    if search_mode == "Standard":
        with st.expander("Search Controls", expanded=True):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                 search_term = st.text_input("Node Word(s)", value=get_state('kwic_search_term'), key="kwic_input", help="Use * for wildcards (e.g. run*)")
            with col2:
                 window_size = st.slider("Context Window", 1, 20, 5, key="kwic_window")
            with col3:
                 limit = st.number_input("Max Lines", 10, 5000, 100, step=10, key="kwic_limit")
             
            # Advanced Filters
            c_adv1, c_adv2 = st.columns(2)
            with c_adv1:
                coll_filter = st.text_input("Filter by Collocate (Regex)", help="Show only lines containing this pattern")
            with c_adv2:
                sort_order = st.selectbox("Sort By", ["Node (Default)", "Left Context", "Right Context"])
                c_sh1, c_sh2 = st.columns(2)
                with c_sh1:
                    show_pos = st.checkbox("Show POS", value=get_state('kwic_show_pos', False), key="kwic_show_pos_cb")
                with c_sh2:
                    show_lemma = st.checkbox("Show Lemma", value=get_state('kwic_show_lemma', False), key="kwic_show_lemma_cb")
                
                set_state('kwic_show_pos', show_pos)
                set_state('kwic_show_lemma', show_lemma)
                
                wrap_mode = st.checkbox("Wrap Text", value=get_state('kwic_wrap_mode', True), key="kwic_wrap_mode_cb", help="Enable to prevent text overlap by wrapping content to multiple lines")
                set_state('kwic_wrap_mode', wrap_mode)

            
    # --- XML Restriction Filters ---
    comp_mode = get_state('comparison_mode', False)
    comp_path = get_state('comp_corpus_path')
    comp_name = get_state('comp_corpus_name')

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
            xml_filters = render_xml_restriction_filters(corpus_path, "concordance", corpus_name=corpus_name)
            xml_where, xml_params = apply_xml_restrictions(xml_filters)
            
            if st.button("Generate Concordance Lines", type="primary"):
                set_state('kwic_search_term', search_term)
                run_concordance_query('primary', corpus_path, corpus_name, search_term, window_size, window_size, limit, coll_filter, xml_where, xml_params, show_pos, show_lemma)
        else:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                xml_filters_1 = render_xml_restriction_filters(corpus_path, "concordance_c1", corpus_name=corpus_name)
                xml_where_1, xml_params_1 = apply_xml_restrictions(xml_filters_1)
            with col_f2:
                xml_filters_2 = render_xml_restriction_filters(comp_path, "concordance_c2", corpus_name=comp_name)
                xml_where_2, xml_params_2 = apply_xml_restrictions(xml_filters_2)
                
            if st.button("Generate Comparison Concordance", type="primary"):
                set_state('kwic_search_term', search_term_1)
                set_state('kwic_search_term_2', search_term_2) # New state for query 2
                
                run_concordance_query('primary', corpus_path, corpus_name, search_term_1, window_size, window_size, limit, coll_filter, xml_where_1, xml_params_1, show_pos, show_lemma)
                if comp_path:
                    run_concordance_query('secondary', comp_path, comp_name, search_term_2, window_size, window_size, limit, coll_filter, xml_where_2, xml_params_2, show_pos, show_lemma)
    else:
        # For NL mode, ensure search_term_1 is defined for display logic
        search_term_1 = get_state('kwic_search_term', '')
        search_term_2 = None

    # 2. Results Display
    if not comp_mode:
        results = st.session_state.get('last_kwic_results_primary')
        if results:
            render_concordance_column(results, search_term_1)
    else:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.subheader(f"Primary: {corpus_name}")
            results_1 = st.session_state.get('last_kwic_results_primary')
            if results_1:
                render_concordance_column(results_1, search_term_1, key_suffix="c1")
        with col_c2:
            st.subheader(f"Comparison: {comp_name}")
            if not comp_path:
                st.info("Load a comparison corpus in sidebar.")
            else:
                results_2 = st.session_state.get('last_kwic_results_secondary')
                if results_2:
                    render_concordance_column(results_2, search_term_2, key_suffix="c2")

def run_concordance_query(identifier, path, name, query, left, right, limit, coll_filter, xml_where, xml_params, show_pos=False, show_lemma=False):
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
            xml_params=xml_params,
            show_pos=show_pos,
            show_lemma=show_lemma
        )
        st.session_state[f'last_kwic_results_{identifier}'] = {
            'rows': kwic_rows,
            'total': total,
            'breakdown': breakdown_df,
            'name': name,
            'search_term': query,
            'xml_where': xml_where,
            'xml_params': xml_params,
            'path': path
        }

def render_concordance_column(results, search_term, key_suffix=""):
     kwic_rows = results['rows']
     total = results['total']
     breakdown = results['breakdown']
     name = results['name']
     
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

     if not breakdown.empty:
         with st.expander("Token Breakdown Stats", expanded=True):
             st.dataframe(breakdown.head(20), use_container_width=True, hide_index=True)
     
     # Results Table
     if kwic_rows:
         # Clickable Headers for Sorting
         st.markdown("---")
         st.markdown("##### Click headers to sort results")
         h_col1, h_col2, h_col3 = st.columns([4.2, 1.6, 4.2])
         
         sort_col = get_state(f'kwic_sort_col_{key_suffix}', 'Node')
         sort_dir = get_state(f'kwic_sort_dir_{key_suffix}', 'asc')

         def set_sort(col):
             current = get_state(f'kwic_sort_col_{key_suffix}')
             if current == col:
                 new_dir = 'desc' if get_state(f'kwic_sort_dir_{key_suffix}') == 'asc' else 'asc'
                 set_state(f'kwic_sort_dir_{key_suffix}', new_dir)
             else:
                 set_state(f'kwic_sort_col_{key_suffix}', col)
                 set_state(f'kwic_sort_dir_{key_suffix}', 'asc')

         with h_col1:
             if st.button("⬅ Sort Left Context", key=f"btn_sort_l_{key_suffix}", use_container_width=True):
                 set_sort('Left')
                 st.rerun()
         with h_col2:
             if st.button("Center", key=f"btn_sort_n_{key_suffix}", use_container_width=True):
                 set_sort('Node')
                 st.rerun()
         with h_col3:
             if st.button("Sort Right Context ➡", key=f"btn_sort_r_{key_suffix}", use_container_width=True):
                 set_sort('Right')
                 st.rerun()

         # Perform Sorting
         # Clean tags for sorting
         def clean_html(raw_html):
             return re.sub(r'<[^>]*>', '', raw_html)

         sorted_rows = sorted(
             kwic_rows, 
             key=lambda x: clean_html(x[sort_col]).lower(), 
             reverse=(sort_dir == 'desc')
         )

         wrap_style = "white-space: normal !important; overflow-wrap: break-word;" if get_state('kwic_wrap_mode', True) else "white-space: nowrap;"
         
         html = f"""
         <style>
         .kwic-table {{ width: 100%; min-width: 800px; font-family: 'Courier New', monospace; font-size: 0.9em; border-collapse: collapse; table-layout: auto; }}
         .kwic-table td {{ padding: 8px 10px; border-bottom: 1px solid #333; vertical-align: middle; line-height: 1.6; }}
         .meta-col {{ text-align: left; width: 10%; font-size: 0.8em; border-right: 1px solid #444; color: #e2e8f0; vertical-align: top; }}
         .ctx-l {{ text-align: right; width: 40%; color: #bbb; {wrap_style} }}
         .node {{ text-align: center; width: auto; white-space: nowrap; font-weight: bold; background-color: #222; color: #FFEA00; border-left: 1px solid #444; border-right: 1px solid #444; padding: 8px 15px; }}
         .ctx-r {{ text-align: left; width: 40%; color: #bbb; {wrap_style} }}
         .sort-info {{ font-size: 0.8em; color: #888; text-align: center; margin-bottom: 5px; }}
         </style>
         <div class='sort-info'>Sorted by <b>{sort_col}</b> ({'Ascending' if sort_dir == 'asc' else 'Descending'})</div>
         <div style="overflow-x: auto;">
          <table class="kwic-table">
         """
         for row in sorted_rows:
             # Trimming logic to ensure visual balance even if data is long
             l_text = row['Left']
             r_text = row['Right']
             
             meta = row.get('Metadata', {})
             meta_html = ""
             if meta:
                 for k, v in meta.items():
                     meta_html += f"<div style='margin-bottom:2px;'><span style='background-color: #334155; color: #e2e8f0; font-size: 0.85em; padding: 2px 4px; border-radius: 3px; border: 1px solid #475569; display: inline-block;' title='{k}'>{v}</span></div>"
                      
             html += f"<tr><td class='meta-col'>{meta_html}</td><td class='ctx-l'>{l_text}</td><td class='node'>{row['Node']}</td><td class='ctx-r'>{r_text}</td></tr>"
         html += "</table></div>"
         st.markdown(html, unsafe_allow_html=True)
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
               sample_text = "\n".join([f"{r['Left']} [{r['Node']}] {r['Right']}" for r in kwic_rows[:10]])
               data_desc = f"KWIC lines for '{search_term}' in '{results['name']}'."
               
               resp, err = interpret_results_llm(
                   target_word=search_term,
                   analysis_type="Concordance Analysis",
                   data_description=f"Snapshot for '{search_term}' in '{name}'.",
                   data=sample_text,
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
