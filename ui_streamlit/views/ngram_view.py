import streamlit as st
import pandas as pd
from ui_streamlit.state_manager import get_state, set_state
from ui_streamlit.caching import cached_generate_ngrams
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.preprocessing.xml_parser import apply_xml_restrictions
from core.ai_service import interpret_results_llm, parse_nl_query, parse_nl_query_rules_only
from core.io_utils import df_to_excel_bytes
from core.modules.overview import get_pos_definitions

def render_ngram_view():
    st.header("N-Gram Analysis")
    
    corpus_path = get_state('current_corpus_path')
    corpus_name = get_state('current_corpus_name', 'Corpus')
    
    if not corpus_path:
        st.warning("Please load a corpus first.")
        return

    # 1. Configuration
    search_mode = st.radio("Search Mode", ["Standard", "Natural Language (Rule)", "Natural Language (AI)"], horizontal=True, key="ngram_search_mode")
    
    if search_mode == "Natural Language (AI)":
        st.markdown("### 🧠 Natural Language Search")
        nl_query = st.text_area("Describe what n-grams you want", height=70, placeholder="e.g. Show me trigrams containing 'data' appearing at least 5 times")
        
        col_ai1, col_ai2 = st.columns([1, 4])
        with col_ai1:
            analyze_btn = st.button("Search with AI", type="primary")
            
        if analyze_btn:
             if not nl_query:
                 st.warning("Please enter a query.")
             else:
                 with st.spinner("AI is configuring n-grams..."):
                     params, err = parse_nl_query(
                         nl_query, 
                         "ngram",
                         ai_provider=get_state('ai_provider'),
                         gemini_api_key=get_state('gemini_api_key'),
                         ollama_url=get_state('ollama_url'),
                         ollama_model=get_state('ai_model')
                     )
                 
                 if params:
                     # Update State
                     try:
                         n_val = int(params.get('n_size', 2))
                     except (ValueError, TypeError): n_val = 2
                     
                     set_state('ngram_n', n_val)
                     
                     st.success(f"✓ Configured for {n_val}-grams.")
                     
                     filters_primary = {}
                     if params.get('search_term'):
                         filters_primary['1'] = params.get('search_term')
                         st.info(f"Adding filter '{params.get('search_term')}' to Position 1.")
                     
                     # Basis default
                     basis = "Token"
                     positional_bases_primary = {str(i): basis for i in range(1, n_val + 1)}
                     
                     xml_filters = render_xml_restriction_filters(corpus_path, "ngram", corpus_name=corpus_name)
                     xml_where, xml_params = apply_xml_restrictions(xml_filters)
                     
                     run_ngram_query('primary', corpus_path, corpus_name, 
                                     n_val, 
                                     filters_primary, 
                                     True, # Skip Punc 
                                     basis, 
                                     positional_bases_primary, 
                                     [], # Neg filter 
                                     xml_where, xml_params)
                 else:
                     st.error(f"Error parsing query: {err}")

    if search_mode == "Natural Language (Rule)":
        st.markdown("### ⚡ Natural Language Search (Rule-Based)")
        st.caption("Fast, deterministic parsing. Describe N-gram constraints.")
        
        with st.expander("N-Gram Settings", expanded=True):
             col1, col2 = st.columns([2, 1])
             with col1:
                 nl_query = st.text_input("N-Gram Query (NL/Rule)", value=get_state('ngram_nl_query_rule', ''), placeholder="e.g. trigrams starting with 'the'", key="ngram_nl_input_rule")
             with col2:
                 n_val = st.slider("N-Gram Size (N)", 2, 5, 2, key="ngram_n_rule")
                 
             st.markdown("---")
             col_punc, col_basis = st.columns(2)
             with col_punc:
                 skip_punc = st.checkbox("Skip Punctuation", value=True, key="ngram_skip_punc_rule")
             with col_basis:
                 global_basis = st.selectbox("Output Basis", ["Token", "Lemma", "POS Tag"], index=0, key="ngram_basis_rule")
        
        if st.button("Generate N-Grams (Rule-Based)", type="primary"):
            if not nl_query:
                st.warning("Please enter a query.")
            else:
                 set_state('ngram_nl_query_rule', nl_query)
                 
                 # Parse
                 pos_defs = get_pos_definitions(corpus_path) or {}
                 reverse_pos_map = {v.lower(): k for k, v in pos_defs.items() if v}
                 
                 params, err = parse_nl_query_rules_only(nl_query, "ngram", reverse_pos_map=reverse_pos_map)
                 
                 if params:
                     # Extraction
                     # Rule parser might return 'n_size' and 'search_term'
                     parsed_n = params.get('n_size', 2)
                     
                     # Check if we should override UI N-Val? 
                     # Strategy: If UI hasn't been touched, use parsed. But syncing is hard.
                     # Let's rely on UI Slider as source of truth for N, unless it was just defaulted?
                     # Actually, for consistency with other modes: UI overrides.
                     # But if user types "trigrams", they expect 3.
                     # Implementation: We update the state of the slider IF it's different? 
                     # Streamlit widgets hard to update mid-run without rerun.
                     # We will use the SLIDER value mostly, but maybe warn if mismatch?
                     # Or simpler: Just use UI slider value.
                     
                     search_term = params.get('search_term', '')
                     
                     st.success(f"✓ Configured for {n_val}-grams.")
                     
                     filters_primary = {}
                     if search_term:
                         # Default to Position 1 filter
                         filters_primary['1'] = search_term
                         st.info(f"   + Filter (Pos 1): '{search_term}'")
                     
                     # Basis setup
                     positional_bases_primary = {str(i): global_basis for i in range(1, n_val + 1)}
                     
                     # Execute
                     xml_filters = render_xml_restriction_filters(corpus_path, "ngram", corpus_name=corpus_name)
                     xml_where, xml_params = apply_xml_restrictions(xml_filters)
                     
                     run_ngram_query('primary', corpus_path, corpus_name, 
                                     n_val, 
                                     filters_primary, 
                                     skip_punc, 
                                     global_basis, 
                                     positional_bases_primary, 
                                     [], # Neg filter (not supported in rule parser yet)
                                     xml_where, xml_params)
                 else:
                     st.error(f"Error parsing query: {err}")

    if search_mode == "Standard":
        with st.expander("N-Gram Settings", expanded=True):
            col_n, col_punc, col_basis = st.columns([1, 1, 1])
            with col_n:
                n_val = st.slider("N-Gram Size (N)", 2, 5, 2, key="ngram_n")
            with col_punc:
                skip_punc = st.checkbox("Skip Punctuation", value=True)
                neg_filter = [] # Removed explicit box as per request; relying on positional negation
            with col_basis:
                global_basis = st.selectbox("Output Basis", ["Token", "Lemma", "POS Tag"], index=0, key="ngram_basis")
                
            st.markdown("##### Positional Filters & Basis")
            st.caption("Lower filters match the selected basis. Use `*`, `%`, `_` as wildcards. Use `_TAG` for POS tags, `[lemma]` to override, or `-term` to exclude.")
        
    # --- XML Restriction Filters ---
    comp_mode = get_state('comparison_mode', False)
    comp_path = get_state('comp_corpus_path')
    comp_name = get_state('comp_corpus_name')
    
    # 2. Dynamic Filters per Corpus
    filters_primary = {}
    positional_bases_primary = {}
    
    filters_secondary = {}
    positional_bases_secondary = {}
    
    if not comp_mode:
        # Standard Single View
        cols = st.columns(n_val)
        for i in range(1, n_val + 1):
             with cols[i-1]:
                 st.markdown(f"**Pos {i}**")
                 pos_basis = st.selectbox("Basis", ["Token", "Lemma", "POS Tag"], index=["Token", "Lemma", "POS Tag"].index(global_basis), key=f"ng_b{i}")
                 positional_bases_primary[str(i)] = pos_basis
                 
                 val = st.text_input(f"Filter", key=f"ng_p{i}")
                 if val:
                     filters_primary[str(i)] = val
    else:
        # Comparison Mode: Tabs or Columns
        st.markdown("##### N-Gram Filters by Corpus")
        tab1, tab2 = st.tabs([f"Primary: {get_state('current_corpus_name', 'Corpus')}", f"Comparison: {comp_name if comp_name else 'Secondary'}"])
        
        with tab1:
            cols = st.columns(n_val)
            for i in range(1, n_val + 1):
                 with cols[i-1]:
                     st.markdown(f"**Pos {i}**")
                     pos_basis = st.selectbox("Basis", ["Token", "Lemma", "POS Tag"], index=["Token", "Lemma", "POS Tag"].index(global_basis), key=f"ng_b{i}_c1")
                     positional_bases_primary[str(i)] = pos_basis
                     
                     val = st.text_input(f"Filter", key=f"ng_p{i}_c1")
                     if val:
                         filters_primary[str(i)] = val

        with tab2:
            cols2 = st.columns(n_val)
            for i in range(1, n_val + 1):
                 with cols2[i-1]:
                     st.markdown(f"**Pos {i}**")
                     pos_basis = st.selectbox("Basis", ["Token", "Lemma", "POS Tag"], index=["Token", "Lemma", "POS Tag"].index(global_basis), key=f"ng_b{i}_c2")
                     positional_bases_secondary[str(i)] = pos_basis
                     
                     val = st.text_input(f"Filter", key=f"ng_p{i}_c2")
                     if val:
                         filters_secondary[str(i)] = val


    if not comp_mode:
        xml_filters = render_xml_restriction_filters(corpus_path, "ngram", corpus_name=corpus_name)
        xml_where, xml_params = apply_xml_restrictions(xml_filters)
        
        if st.button("Generate N-Grams", type="primary"):
            run_ngram_query('primary', corpus_path, corpus_name, n_val, filters_primary, skip_punc, global_basis, positional_bases_primary, neg_filter, xml_where, xml_params)
    else:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            xml_filters_1 = render_xml_restriction_filters(corpus_path, "ngram_c1", corpus_name=corpus_name)
            xml_where_1, xml_params_1 = apply_xml_restrictions(xml_filters_1)
        with col_f2:
            if comp_path:
                xml_filters_2 = render_xml_restriction_filters(comp_path, "ngram_c2", corpus_name=comp_name)
                xml_where_2, xml_params_2 = apply_xml_restrictions(xml_filters_2)
            else:
                st.info("Load a comparison corpus in sidebar.")
                xml_where_2, xml_params_2 = "", []
            
        if st.button("Generate Comparison N-Grams", type="primary"):
            run_ngram_query('primary', corpus_path, corpus_name, n_val, filters_primary, skip_punc, global_basis, positional_bases_primary, neg_filter, xml_where_1, xml_params_1)
            if comp_path:
                run_ngram_query('secondary', comp_path, comp_name, n_val, filters_secondary, skip_punc, global_basis, positional_bases_secondary, neg_filter, xml_where_2, xml_params_2)

    # 3. Results
    if not comp_mode:
        df_results = st.session_state.get('last_ngram_results_primary')
        if df_results is not None:
            render_ngram_results_column(df_results, n_val, corpus_name)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Primary: {corpus_name}")
            res1 = st.session_state.get('last_ngram_results_primary')
            if res1 is not None:
                render_ngram_results_column(res1, n_val, corpus_name, key_suffix="c1")
        with col2:
            st.subheader(f"Comparison: {comp_name}")
            if not comp_path:
                st.info("Load a comparison corpus in sidebar.")
            else:
                res2 = st.session_state.get('last_ngram_results_secondary')
                if res2 is not None:
                    render_ngram_results_column(res2, n_val, comp_name, key_suffix="c2")
        
        # Comparison Analysis Tables
        if res1 is not None and res2 is not None and not res1.empty and not res2.empty:
            st.markdown("---")
            st.header("📊 N-gram Comparison Analysis")
            
            from core.modules.comparison_analysis import compare_ngrams, get_comparison_summary, render_comparison_tables
            
            # Perform comparison
            shared_df, df1_unique, df2_unique = compare_ngrams(
                res1, res2,
                corpus1_name=corpus_name,
                corpus2_name=comp_name
            )
            
            # Summary metrics
            summary = get_comparison_summary(shared_df, df1_unique, df2_unique, 'N-grams')
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Shared N-grams", summary['total_shared'])
            with col_s2:
                st.metric(f"{corpus_name} Only", summary['total_primary_unique'])
            with col_s3:
                st.metric(f"{comp_name} Only", summary['total_comparison_unique'])
            with col_s4:
                st.metric("Overlap %", f"{summary['overlap_percentage']}%")
            
            # Display comparison tables
            render_comparison_tables(shared_df, df1_unique, df2_unique,
                                    corpus_name, comp_name, analysis_type='ngram')

def run_ngram_query(identifier, path, name, n, filters, skip_punc, basis, positional_bases, neg_filter, xml_where, xml_params):
    with st.spinner(f"Generating n-grams for {name}..."):
        df = cached_generate_ngrams(
            db_path=path,
            n=n,
            filters=filters,
            is_raw=False,
            corpus_name=name,
            skip_punctuation=skip_punc,
            basis=basis,
            positional_bases=positional_bases,
            negative_filter=neg_filter,
            xml_where_clause=xml_where,
            xml_params=xml_params
        )
        st.session_state[f'last_ngram_results_{identifier}'] = df

def render_ngram_results_column(df, n_val, corpus_name, key_suffix=""):
    if df is not None and not df.empty:
        total_results = len(df)
        display_limit = 100
        
        st.markdown(f"**Found {total_results:,} results.** Showing top {min(display_limit, total_results)}.")
        
        # Limit display to prevent browser freeze
        df_display = df.head(display_limit)
        
        # Color palette for positions (vibrant, distinct colors)
        position_colors = [
            "#FF6B6B",  # Red
            "#4ECDC4",  # Teal
            "#FFD93D",  # Yellow
            "#95E1D3",  # Mint
            "#A8E6CF",  # Light green
        ]
        
        # Build custom HTML table with larger fonts and colors
        html = """
        <style>
        .ngram-table { 
            width: 100%; 
            border-collapse: collapse; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 16px;
        }
        .ngram-table th { 
            background-color: #2C3E50; 
            color: white; 
            padding: 12px; 
            text-align: left;
            font-weight: 600;
        }
        .ngram-table td { 
            padding: 10px; 
            border-bottom: 1px solid #ddd;
        }
        .ngram-table tr:hover td { 
            background-color: #f8f9fa;
        }
        .ngram-token {
            font-weight: 500;
            padding: 4px 8px;
            border-radius: 4px;
            margin: 0 2px;
            display: inline-block;
            min-width: 120px;
            text-align: center;
        }
        </style>
        <table class="ngram-table">
        <thead>
            <tr>
                <th>N-Gram</th>
                <th>Frequency</th>
                <th>Relative Freq (PMW)</th>
        """
        
        # Add optional columns if they exist
        if 'Zipf Score' in df.columns:
            html += "<th>Zipf Score</th>"
        if 'Zipf Law Frequency Band' in df.columns:
            html += "<th>Frequency Band</th>"
        
        html += "</tr></thead><tbody>"
        
        # Render rows
        for _, row in df_display.iterrows():
            ngram_col = [col for col in df.columns if col.startswith('Pos')][0] if any(col.startswith('Pos') for col in df.columns) else df.columns[0]
            ngram_text = str(row[ngram_col])
            
            # Split n-gram and apply colors
            tokens = ngram_text.split()
            colored_ngram = ""
            for i, token in enumerate(tokens):
                color = position_colors[i % len(position_colors)]
                colored_ngram += f'<span class="ngram-token" style="background-color: {color}20; border-left: 3px solid {color};">{token}</span> '
            
            html += f"<tr><td>{colored_ngram}</td>"
            html += f"<td>{row['Frequency']}</td>"
            html += f"<td>{row['Relative Frequency (per M)']:.2f}</td>"
            
            if 'Zipf Score' in df.columns:
                html += f"<td>{row['Zipf Score']:.2f}</td>"
            if 'Zipf Law Frequency Band' in df.columns:
                html += f"<td>{row['Zipf Law Frequency Band']}</td>"
            
            html += "</tr>"
        
        html += "</tbody></table>"
        
        st.markdown(html, unsafe_allow_html=True)
        
        st.download_button(
            label=f"Download {corpus_name} N-Grams (Excel)",
            data=df_to_excel_bytes(df),
            file_name=f"ngrams_{corpus_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_ng_{key_suffix}"
        )
        
        if st.button("Interpret with AI", key=f"btn_ngram_ai_{key_suffix}"):
             with st.spinner("Analyzing..."):
                  top_n = df.head(10).to_string(index=False)
                  data_desc = f"Top {n_val}-grams from corpus '{corpus_name}'."
                  
                  resp, err = interpret_results_llm(
                       target_word=f"Top {n_val}-Grams",
                       analysis_type="N-Gram Frequency Analysis",
                       data_description=data_desc,
                       data=top_n,
                       ai_provider=get_state('ai_provider'),
                       gemini_api_key=get_state('gemini_api_key'),
                       ollama_url=get_state('ollama_url'),
                       ollama_model=get_state('ai_model')
                   )
                  if resp:
                      set_state(f'llm_res_ngram_{key_suffix}', resp)
                  elif err:
                      st.error(err)
                      
        llm_res = get_state(f'llm_res_ngram_{key_suffix}')
        if llm_res:
             st.markdown(llm_res)
    else:
        st.info("No N-grams found matching specific criteria.")

