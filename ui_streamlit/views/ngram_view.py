import streamlit as st
import pandas as pd
import urllib.parse
from ui_streamlit.state_manager import get_state, set_state
from ui_streamlit.caching import cached_generate_ngrams
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.preprocessing.xml_parser import apply_xml_restrictions
from core.ai_service import interpret_results_llm, parse_nl_query, parse_nl_query_rules_only
from core.io_utils import df_to_excel_bytes
import core.modules.overview as ov

def render_ngram_view():
    from ui_streamlit.components.pos_help import check_available_annotations
    st.header("N-Gram Analysis")
    
    # Guidelines Layout using shared component
    from ui_streamlit.components.guidelines import render_guidelines
    col_main = render_guidelines("N-Gram")

    with col_main:


            corpus_path = get_state('current_corpus_path')
            corpus_name = get_state('current_corpus_name', 'Corpus')

            if not corpus_path:
                st.warning("Please load a corpus first.")
                return

            active_bases = ["Token", "Lemma"] + [a for a in check_available_annotations(corpus_path) if a != "Sentiment Analysis"]

            # Initialize XML restriction variables to prevent NameError in NL search modes
            xml_where = ""
            xml_params = []
            xml_where_1 = ""
            xml_params_1 = []
            xml_where_2 = ""
            xml_params_2 = []

            # Deferred execution flag for NL modes (query runs AFTER XML filters are rendered)
            _deferred_ngram_query = None

            tab_simple, tab_advanced = st.tabs(["Simple", "Advanced"])

            with tab_simple:
                n_val_simple = st.slider("N-Gram Size (N)", 2, 5, 2, key="ngram_n_simple")
                if st.button("Generate N-Grams", type="primary", key="btn_generate_ngram_simple", use_container_width=True):
                     positional_bases_simple = {str(i): 'Token' for i in range(1, n_val_simple + 1)}
                     run_ngram_query(
                         identifier='primary',
                         path=corpus_path,
                         name=corpus_name,
                         n=n_val_simple,
                         filters={},
                         skip_punc=True,
                         basis='Token',
                         positional_bases=positional_bases_simple,
                         neg_filter=[],
                         xml_where="",
                         xml_params=[],
                         source='simple'
                     )
                     st.rerun()

            with tab_advanced:
                # 1. Configuration
                search_mode = st.radio("Search Mode", ["Standard", "Natural Language (Rule)", "Natural Language (AI)"], horizontal=True, key="ngram_search_mode")

                if search_mode == "Natural Language (AI)":
                    st.markdown("### 🧠 Natural Language Search")
                    nl_query = st.text_area("Describe what n-grams you want", height=70, placeholder="e.g. Show me trigrams containing 'data' appearing at least 5 times")
                    from ui_streamlit.components.pos_help import render_annotation_help_button, check_available_annotations
                    render_annotation_help_button(corpus_path, "ngram_ai")

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
                                 try:
                                     n_val = int(params.get('n_size', 2))
                                 except (ValueError, TypeError): n_val = 2

                                 set_state('ngram_n', n_val)

                                 st.success(f"✓ Configured for {n_val}-grams.")

                                 filters_primary = {}
                                 if params.get('search_term'):
                                     filters_primary['1'] = params.get('search_term')
                                     st.info(f"Adding filter '{params.get('search_term')}' to Position 1.")

                                 basis = "Token"
                                 positional_bases_primary = {str(i): basis for i in range(1, n_val + 1)}

                                 # Defer execution until after XML filters are rendered below
                                 _deferred_ngram_query = {
                                     'n': n_val, 'filters': filters_primary, 'skip_punc': True,
                                     'basis': basis, 'pos_bases': positional_bases_primary, 'neg': []
                                 }
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
                             global_basis = st.radio("Output Basis", active_bases, index=0, horizontal=True, key="ngram_basis_rule")
                         
                         from ui_streamlit.components.pos_help import render_annotation_help_button, check_available_annotations
                         render_annotation_help_button(corpus_path, "ngram_rule")

                    if st.button("Generate N-Grams (Rule-Based)", type="primary"):
                        if not nl_query:
                            st.warning("Please enter a query.")
                        else:
                             set_state('ngram_nl_query_rule', nl_query)

                             pos_defs = ov.get_pos_definitions(corpus_path) or {}
                             reverse_pos_map = {v.lower(): k for k, v in pos_defs.items() if v}

                             params, err = parse_nl_query_rules_only(nl_query, "ngram", reverse_pos_map=reverse_pos_map)

                             if params:
                                 search_term = params.get('search_term', '')
                                 st.success(f"✓ Configured for {n_val}-grams.")

                                 filters_primary = {}
                                 if search_term:
                                     filters_primary['1'] = search_term
                                     st.info(f"   + Filter (Pos 1): '{search_term}'")

                                 positional_bases_primary = {str(i): global_basis for i in range(1, n_val + 1)}

                                 # Defer execution until after XML filters are rendered below
                                 _deferred_ngram_query = {
                                     'n': n_val, 'filters': filters_primary, 'skip_punc': skip_punc,
                                     'basis': global_basis, 'pos_bases': positional_bases_primary, 'neg': []
                                 }
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
                            global_basis = st.radio("Output Basis", active_bases, index=0, horizontal=True, key="ngram_basis")

                        st.markdown("##### Positional Filters & Basis")
                        st.caption("Lower filters match the selected basis. Use `*`, `%`, `_` as wildcards. Use `_TAG` for POS tags, `[lemma]` to override, or `-term` to exclude.")
                        from ui_streamlit.components.pos_help import render_annotation_help_button, check_available_annotations
                        render_annotation_help_button(corpus_path, "ngram_standard")

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
                             pos_basis = st.radio("Basis", active_bases, index=(active_bases).index(global_basis), horizontal=True, key=f"ng_b{i}")
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
                                 pos_basis = st.radio("Basis", active_bases, index=(active_bases).index(global_basis), horizontal=True, key=f"ng_b{i}_c1")
                                 positional_bases_primary[str(i)] = pos_basis

                                 val = st.text_input(f"Filter", key=f"ng_p{i}_c1")
                                 if val:
                                     filters_primary[str(i)] = val

                    with tab2:
                        cols2 = st.columns(n_val)
                        for i in range(1, n_val + 1):
                             with cols2[i-1]:
                                 st.markdown(f"**Pos {i}**")
                                 pos_basis = st.radio("Basis", active_bases, index=(active_bases).index(global_basis), horizontal=True, key=f"ng_b{i}_c2")
                                 positional_bases_secondary[str(i)] = pos_basis

                                 val = st.text_input(f"Filter", key=f"ng_p{i}_c2")
                                 if val:
                                     filters_secondary[str(i)] = val


                # --- XML Restriction Filters ---
                if not comp_mode:
                    xml_filters = render_xml_restriction_filters(corpus_path, "ngram", corpus_name=corpus_name)
                    xml_where, xml_params = apply_xml_restrictions(xml_filters)
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
                            xml_where_2, xml_params_2 = "", []

                # --- Deferred NL Query Execution (runs AFTER xml_where/xml_params are set) ---
                if _deferred_ngram_query is not None:
                    _dq = _deferred_ngram_query
                    run_ngram_query(
                        'primary', corpus_path, corpus_name,
                        _dq['n'], _dq['filters'], _dq['skip_punc'],
                        _dq['basis'], _dq['pos_bases'], _dq['neg'],
                        xml_where, xml_params
                    )

                if not comp_mode:
                    if st.button("Generate N-Grams", type="primary", key="btn_generate_ngram_advanced"):
                        run_ngram_query('primary', corpus_path, corpus_name, n_val, filters_primary, skip_punc, global_basis, positional_bases_primary, neg_filter, xml_where, xml_params)
                else:
                    if st.button("Generate Comparison N-Grams", type="primary", key="btn_generate_ngram_comp_advanced"):
                        run_ngram_query('primary', corpus_path, corpus_name, n_val, filters_primary, skip_punc, global_basis, positional_bases_primary, neg_filter, xml_where_1, xml_params_1)
                        if comp_path:
                            run_ngram_query('secondary', comp_path, comp_name, n_val, filters_secondary, skip_punc, global_basis, positional_bases_secondary, neg_filter, xml_where_2, xml_params_2)

            # 3. Results
            if not comp_mode:
                df_results = st.session_state.get('last_ngram_results_primary')
                if df_results is not None:
                    n_size = df_results.attrs.get('n', n_val if 'n_val' in locals() else 2)
                    render_ngram_results_column(df_results, n_size, corpus_name)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Primary: {corpus_name}")
                    res1 = st.session_state.get('last_ngram_results_primary')
                    if res1 is not None:
                        n_size1 = res1.attrs.get('n', n_val if 'n_val' in locals() else 2)
                        render_ngram_results_column(res1, n_size1, corpus_name, key_suffix="c1")
                with col2:
                    st.subheader(f"Comparison: {comp_name}")
                    if not comp_path:
                        st.info("Load a comparison corpus in sidebar.")
                    else:
                        res2 = st.session_state.get('last_ngram_results_secondary')
                        if res2 is not None:
                            n_size2 = res2.attrs.get('n', n_val if 'n_val' in locals() else 2)
                            render_ngram_results_column(res2, n_size2, comp_name, key_suffix="c2")

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

def format_ngram_to_concordance_query(ngram_str, positional_bases=None, global_basis='Token'):
    """
    Converts an N-Gram result string (e.g. 'IN the end of' or 'at the end of')
    into a valid Concordance query string (e.g. '_IN the end of' or 'at the end of').
    """
    tokens = str(ngram_str).split()
    query_parts = []
    
    COMMON_POS_TAGS = {
        'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
        'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
        'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
        'WDT', 'WP', 'WP$', 'WRB'
    }

    for i, tok in enumerate(tokens):
        pos_idx = str(i + 1)
        basis = positional_bases.get(pos_idx, global_basis) if positional_bases else global_basis
        
        # Already formatted (starts with _ or < or inside [])
        if tok.startswith('_') or tok.startswith('<') or (tok.startswith('[') and tok.endswith(']')):
            query_parts.append(tok)
        elif basis in ['POS Tag', 'Part-of-Speech', 'POS'] or tok in COMMON_POS_TAGS or (tok.isupper() and 2 <= len(tok) <= 4 and tok.isalpha()):
            tag = tok[1:] if tok.startswith('_') else tok
            query_parts.append(f"_{tag}")
        elif basis == 'Lemma':
            query_parts.append(f"[{tok}]")
        else:
            query_parts.append(tok)
            
    return " ".join(query_parts)


def run_ngram_query(identifier, path, name, n, filters, skip_punc, basis, positional_bases, neg_filter, xml_where, xml_params, source='advanced'):
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
        df.attrs['source'] = source
        df.attrs['n'] = n
        df.attrs['basis'] = basis
        df.attrs['positional_bases'] = positional_bases
        st.session_state[f'last_ngram_results_{identifier}'] = df

def render_ngram_results_column(df, n_val, corpus_name, key_suffix=""):
    n_val_actual = df.attrs.get('n', n_val)
    global_basis = df.attrs.get('basis', 'Token')
    pos_bases = df.attrs.get('positional_bases', {})

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
        
        # Inject CSS for token badges
        st.markdown("""
        <style>
        .ngram-token {
            font-weight: 500;
            padding: 4px 8px;
            border-radius: 4px;
            margin: 0 2px;
            display: inline-block;
            min-width: 100px;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Render N-Gram results table with in-memory Streamlit buttons to prevent browser page reload and loss of uploaded files
        header_cols = st.columns([4, 2, 2, 2])
        header_cols[0].markdown("**N-Gram**")
        header_cols[1].markdown("**Concordance**")
        header_cols[2].markdown("**Frequency**")
        header_cols[3].markdown("**Relative Freq (PMW)**")
        st.markdown("<hr style='margin: 4px 0 12px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        
        for idx, row in df_display.iterrows():
            ngram_col = [col for col in df.columns if col.startswith('Pos')][0] if any(col.startswith('Pos') for col in df.columns) else df.columns[0]
            ngram_text = str(row[ngram_col])
            
            # Split n-gram and apply colors
            tokens = ngram_text.split()
            colored_ngram = ""
            for i, token in enumerate(tokens):
                color = position_colors[i % len(position_colors)]
                colored_ngram += f'<span class="ngram-token" style="background-color: {color}20; border-left: 3px solid {color};">{token}</span> '
            
            kwic_query_str = format_ngram_to_concordance_query(ngram_text, positional_bases=pos_bases, global_basis=global_basis)
            
            r_cols = st.columns([4, 2, 2, 2])
            r_cols[0].markdown(colored_ngram, unsafe_allow_html=True)
            
            if r_cols[1].button("🔍 concordance", key=f"btn_kwic_redir_{key_suffix}_{idx}"):
                from ui_streamlit.state_manager import set_state
                set_state('kwic_search_term', kwic_query_str)
                set_state('kwic_search_mode', 'Standard')
                set_state('current_module', 'Concordance')
                st.rerun()
                
            r_cols[2].write(f"{row['Frequency']:,}")
            r_cols[3].write(f"{row['Relative Frequency (per M)']:.2f}")
            
            st.markdown("<hr style='margin: 4px 0; border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
        
        # Limit export to 100,000 rows to prevent Excel and Memory crashes
        df_export = df.head(100000)
        
        st.download_button(
            label=f"Download {corpus_name} N-Grams (Excel)",
            data=df_to_excel_bytes(df_export),
            file_name=f"ngrams_{corpus_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_ng_{key_suffix}"
        )
        
        if st.button("Interpret with AI", key=f"btn_ngram_ai_{key_suffix}"):
             with st.spinner("Analyzing..."):
                  header_str = f"Corpus: '{corpus_name}' (Total N-Grams Analyzed: {len(df)})\n=== TOP {n_val_actual}-GRAMS FREQUENCY TABLE (Explicit Occurrences Count) ===\n"
                  top_n = header_str + df.head(15).to_string(index=False)
                  data_desc = f"Top {n_val_actual}-grams frequency distribution from corpus '{corpus_name}'."
                  
                  resp, err = interpret_results_llm(
                       target_word=f"Top {n_val_actual}-Grams",
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

