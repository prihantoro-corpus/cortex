import streamlit as st
import pandas as pd
from ui_streamlit.state_manager import get_state, set_state
import importlib
import core.modules.keyword
importlib.reload(core.modules.keyword)
from core.modules.keyword import generate_keyword_list, generate_grouped_keyword_list
from core.visualiser.wordcloud import generate_wordcloud
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.preprocessing.xml_parser import apply_xml_restrictions
from core.io_utils import df_to_excel_bytes
from core.config import BUILT_IN_CORPORA
from core.preprocessing.corpus_loader import load_monolingual_corpus_files, load_built_in_corpus
from core.ai_service import parse_nl_query
import io

def parse_frequency_list_file(uploaded_file):
    """Parses a word-tab-freq or word-space-freq file."""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        lines = content.strip().split('\n')
        data = []
        total = 0
        for line in lines:
            parts = line.strip().split() # Handles both tab and spaces
            if len(parts) >= 2:
                word = parts[0].lower()
                try:
                    freq = int(parts[1])
                    data.append({'token': word, 'freq': freq})
                    total += freq
                except ValueError:
                    continue
        return pd.DataFrame(data), total
    except Exception as e:
        st.error(f"Error parsing frequency list: {e}")
        return None, 0

def render_keyword_view():
    st.header("Keyword Analysis")
    
    # 1. Corpus Selection
    current_path = get_state('current_corpus_path')
    current_name = get_state('current_corpus_name', 'Corpus')
    
    if not current_path:
        st.warning("Please load a primary corpus first.")
        return
        
    # 1. Comparison Mode & Reference Selection
    comp_mode = get_state('comparison_mode', False)
    ref_path = get_state('comp_corpus_path')
    ref_name = get_state('comp_corpus_name')
    
    with st.container(border=True):
        st.markdown("#### 🎯 Reference Corpus Selection")
        
        # Priority 1: Use Global Comparison Corpus if enabled
        if comp_mode and ref_path:
            st.success(f"**Using Global Comparison Corpus:** {ref_name}")
            st.caption("You can change the comparison corpus in the Sidebar.")
        
        # Priority 2: Manual Selection if no global comparison or if user wants to override (simplified here to just show manual if not set)
        elif not ref_path:
            st.info("Select a reference corpus to compare against.")
            tabs = st.tabs(["🏛️ Pre-built", "📤 Upload"])
            
            with tabs[0]:
                sel_name = st.selectbox("Built-in Corpora", list(BUILT_IN_CORPORA.keys()))
                if st.button("Load as Reference", key="load_builtin_ref"):
                    with st.spinner("Downloading and processing..."):
                        result = load_built_in_corpus(sel_name, BUILT_IN_CORPORA[sel_name])
                        if result.get('error'):
                            st.error(result['error'])
                        else:
                            set_state('comp_corpus_path', result['db_path'])
                            set_state('comp_corpus_stats', result['stats'])
                            set_state('comp_corpus_name', sel_name)
                            set_state('comp_xml_structure_data', result.get('structure'))
                            st.success(f"Loaded {sel_name} as reference.")
                            st.rerun()
                            
            with tabs[1]:
                uploaded_ref = st.file_uploader("Upload XML or Frequency List", type=['xml', 'txt', 'csv', 'tsv'], key="upload_ref_kw")
                if uploaded_ref:
                    if st.button("Process Reference", key="btn_process_ref"):
                        with st.spinner("Processing reference..."):
                            if uploaded_ref.name.endswith('.xml'):
                                result = load_monolingual_corpus_files([uploaded_ref], explicit_lang_code=get_state('target_lang', 'en'))
                                if result.get('error'):
                                    st.error(result['error'])
                                else:
                                    set_state('comp_corpus_path', result['db_path'])
                                    set_state('comp_corpus_stats', result['stats'])
                                    set_state('comp_corpus_name', uploaded_ref.name)
                                    set_state('comp_ref_type', 'db')
                                    st.success("Reference corpus loaded.")
                                    st.rerun()
                            else:
                                df_freq, total = parse_frequency_list_file(uploaded_ref)
                                if df_freq is not None and not df_freq.empty:
                                    set_state('comp_freq_df', df_freq)
                                    set_state('comp_total_tokens', total)
                                    set_state('comp_corpus_name', uploaded_ref.name)
                                    set_state('comp_corpus_path', 'frequency_list') 
                                    set_state('comp_ref_type', 'freq_list')
                                    st.success(f"Loaded {len(df_freq)} entries.")
                                    st.rerun()
            return 
        else:
            # Manual reference set but not in global comp mode
            c1, c2 = st.columns([3, 1])
            with c1:
                st.success(f"**Reference:** {ref_name} (Manual/Previous)")
            with c2:
                if st.button("Switch Reference", key="btn_switch_ref"):
                    set_state('comp_corpus_path', None)
                    set_state('comp_corpus_name', None)
                    set_state('comp_freq_df', None)
                    set_state('comp_total_tokens', 0)
                    st.rerun()
        
    st.markdown("---")
    
    # Target and Reference labels
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Target Corpus:** {current_name}")
    with col2:
        st.markdown(f"**Reference Corpus:** {ref_name}")
        
    # 2. Settings
    search_mode = st.radio("Search Mode", ["Standard", "Natural Language (AI)"], horizontal=True, key="kw_search_mode")
    
    if search_mode == "Natural Language (AI)":
        st.markdown("### 🧠 Natural Language Search")
        nl_query = st.text_area("Describe keyword criteria", placeholder="e.g. Find keywords with very high significance (p < 0.001)")
        
        if st.button("Configure & Run", type="primary", key="kw_ai_btn"):
             if not nl_query:
                 st.warning("Please enter a query.")
             else:
                 with st.spinner("Parsing query..."):
                     params, err = parse_nl_query(
                         nl_query, 
                         "keyword",
                         ai_provider=get_state('ai_provider'),
                         gemini_api_key=get_state('gemini_api_key'),
                         ollama_url=get_state('ollama_url'),
                         ollama_model=get_state('ai_model')
                     )
                     
                 if params:
                     st.success("✓ Criteria parsed.")
                     
                     # Execute
                     try:
                        min_freq = int(params.get('min_freq', 5))
                     except (ValueError, TypeError): min_freq = 5
                     # We can't apply min_freq to state easily without rerunning standard logic, so we call generate directly
                     
                     with st.spinner("Calculating Keyness..."):
                         # Basic restrictions
                         xml_filters_t = render_xml_restriction_filters(current_path, "kw_target")
                         xml_where_t, xml_params_t = apply_xml_restrictions(xml_filters_t)
                         
                         if get_state('comp_ref_type') != 'freq_list':
                             xml_filters_r = render_xml_restriction_filters(ref_path, "kw_ref")
                             xml_where_r, xml_params_r = apply_xml_restrictions(xml_filters_r)
                         else:
                             xml_where_r, xml_params_r = "", []
                             
                         if get_state('comp_ref_type') == 'freq_list':
                            df = generate_keyword_list(
                                current_path, None, 
                                xml_where_t, xml_params_t,
                                "", [],
                                min_freq,
                                ref_freq_df=get_state('comp_freq_df'),
                                ref_total_tokens=get_state('comp_total_tokens')
                            )
                         else:
                            df = generate_keyword_list(
                                current_path, ref_path, 
                                xml_where_t, xml_params_t,
                                xml_where_r, xml_params_r,
                                min_freq
                            )
                         
                         # Apply P-Value filtering if requested explicitly
                         p_cut = params.get('p_val_cutoff')
                         if p_cut and df is not None and not df.empty:
                             if p_cut == "0.001":
                                 df = df[df['Significance'].str.contains(r'\*\*\*', na=False)]
                             elif p_cut == "0.01":
                                 df = df[df['Significance'].str.contains(r'\*\*', na=False)]
                             elif p_cut == "0.05":
                                 df = df[df['Significance'] != 'ns']
                         
                         st.session_state['keyword_results'] = df

    if search_mode == "Standard":
        with st.expander("Analysis Settings", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                min_freq = st.number_input("Min Frequency (Target)", 1, 1000, 5, help="Minimum frequency in target corpus")
            with c2:
                top_n = st.number_input("Top N Keywords", 10, 500, 100)
            with c3:
                p_val_cutoff = st.selectbox("P-Value Cutoff", ["0.05", "0.01", "0.001", "None"], index=2)
                
            st.markdown("##### 📂 Grouping (Optional)")
            
            # Prepare Group Options
            group_options = ["File (filename)"]
            
            import duckdb
            try:
                 con = duckdb.connect(current_path, read_only=True)
                 cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
                 core_cols = ['id', 'token', 'lemma', 'pos', '_token_low', 'filename']
                 attr_opts = [c for c in cols if c not in core_cols]
                 con.close()
                 # Add attributes with prefix
                 group_options.extend([f"Attribute: {c}" for c in attr_opts])
            except: pass
            
            selected_groups = st.multiselect("Group Results By:", group_options)

    # XML Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("##### Target Filters")
        xml_filters_t = render_xml_restriction_filters(current_path, "kw_target")
        xml_where_t, xml_params_t = apply_xml_restrictions(xml_filters_t)
    if get_state('comp_ref_type') != 'freq_list':
        with col_f2:
            st.markdown("##### Reference Filters")
            xml_filters_r = render_xml_restriction_filters(ref_path, "kw_ref")
            xml_where_r, xml_params_r = apply_xml_restrictions(xml_filters_r)
    else:
        xml_where_r, xml_params_r = "", []

    if st.button("Calculate Keywords", type="primary"):
        with st.spinner("Calculating Keyness..."):
            
            # 1. Global Results (Always Calculate)
            if get_state('comp_ref_type') == 'freq_list':
                df_global = generate_keyword_list(
                    current_path, None, 
                    xml_where_t, xml_params_t,
                    "", [],
                    min_freq,
                    ref_freq_df=get_state('comp_freq_df'),
                    ref_total_tokens=get_state('comp_total_tokens')
                )
            else:
                df_global = generate_keyword_list(
                    current_path, ref_path, 
                    xml_where_t, xml_params_t,
                    xml_where_r, xml_params_r,
                    min_freq
                )
            st.session_state['keyword_results'] = df_global
            st.session_state['keyword_results_grouped'] = {}

            # 2. Grouped Results (Iterate)
            if selected_groups:
                 grouped_data = {}
                 for grp_display in selected_groups:
                      # Extract column name
                      # "File (filename)" -> "filename"
                      # "Attribute: speaker" -> "speaker"
                      if "File" in grp_display: 
                          grp_col = "filename"
                      else:
                          grp_col = grp_display.split(": ")[1]
                      
                      if get_state('comp_ref_type') == 'freq_list':
                         res_dict = generate_grouped_keyword_list(
                            current_path, grp_col,
                            None, 
                            xml_where_t, xml_params_t,
                            "", [],
                            min_freq,
                            ref_freq_df=get_state('comp_freq_df'),
                            ref_total_tokens=get_state('comp_total_tokens')
                         )
                      else:
                         res_dict = generate_grouped_keyword_list(
                            current_path, grp_col, 
                            ref_path,
                            xml_where_t, xml_params_t,
                            xml_where_r, xml_params_r,
                            min_freq
                         )
                      grouped_data[grp_display] = res_dict
                 
                 st.session_state['keyword_results_grouped'] = grouped_data
            
    # 3. Results
    results = st.session_state.get('keyword_results')
    results_grouped = st.session_state.get('keyword_results_grouped', {})
    
    # --- RENDER DASHBOARD ---
    if results_grouped:
        st.markdown("---")
        st.header("📊 Grouped Dashboard")
        
        for category, group_dict in results_grouped.items():
            st.markdown(f"## {category}")
            # category example: "Attribute: speaker"
            
            if not group_dict:
                st.info("No groups found.")
                continue
                
            # Iterate group values (e.g. SpeakerA, SpeakerB)
            # Use columns or expanders? User asked for 3 clouds/tables if 3 speakers.
            # Expanders are cleaner if many. Headers if few.
            # Let's use Expanders expanded=True by default for visibility.
            
            sorted_groups = sorted(group_dict.keys())
            
            for grp_val in sorted_groups:
                stats_df = group_dict[grp_val]
                if stats_df.empty: continue
                
                with st.expander(f"**{grp_val}** ({len(stats_df)} keywords)", expanded=True):
                    # Filter Positive Only for Cloud/Table mainly
                    pos_df = stats_df[stats_df['Type'] == 'Positive'].head(top_n)
                    
                    c_viz, c_tab = st.columns([1, 1])
                    with c_viz:
                        freq_dict = dict(zip(pos_df['token'], pos_df['LL']))
                        if freq_dict:
                            wc_fig = generate_wordcloud(freq_dict, title=f"Cloud: {grp_val}", color_scheme='viridis')
                            st.pyplot(wc_fig)
                        else: st.caption("No positive keywords.")
                            
                    with c_tab:
                        st.dataframe(
                            pos_df[['token', 'freq_t', 'freq_r', 'LL', 'LogRatio']], 
                            use_container_width=True,
                            height=300
                        )
                        st.download_button(
                            f"⬇️ {grp_val}.xlsx",
                            data=df_to_excel_bytes(pos_df),
                            file_name=f"kw_{grp_val}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"dl_{category}_{grp_val}"
                        )

        st.markdown("---")
        st.subheader("Global Results (All Groups Merged)")

    if isinstance(results, pd.DataFrame) and not results.empty:
        # AI Interpretation
        st.markdown("---")
        if st.button("🧠 Interpret Keywords with AI", key="btn_kw_ai_full"):
            with st.spinner("AI is analyzing keywords..."):
                top_pos = results[results['Type'] == 'Positive'].head(20).to_string(index=False)
                from core.ai_service import interpret_results_llm
                resp, err = interpret_results_llm(
                    target_word=current_name,
                    analysis_type="Keyword Analysis",
                    data_description=f"Keywords in '{current_name}' vs '{ref_name}'.",
                    data=top_pos,
                    ai_provider=get_state('ai_provider'),
                    gemini_api_key=get_state('gemini_api_key'),
                    ollama_url=get_state('ollama_url'),
                    ollama_model=get_state('ai_model')
                )
                if resp: set_state('llm_res_kw', resp)
                else: st.error(err)
        
        ai_res = get_state('llm_res_kw')
        if ai_res:
            with st.expander("🤖 AI Keyword Interpretation", expanded=True):
                st.markdown(ai_res)
        st.markdown("---")
        # Filtering by Significance
        if p_val_cutoff != "None":
            # Filter logic depends on how association.py formats strings.
            # *** (p<0.001), ** (p<0.01), * (p<0.05), ns
            if p_val_cutoff == "0.001":
                results = results[results['Significance'].str.contains(r'\*\*\*', na=False)]
            elif p_val_cutoff == "0.01":
                results = results[results['Significance'].str.contains(r'\*\*', na=False)]
            elif p_val_cutoff == "0.05":
                results = results[results['Significance'] != 'ns']
        
        pos_keywords = results[results['Type'] == 'Positive'].head(top_n)
        neg_keywords = results[results['Type'] == 'Negative'].sort_values('LL', ascending=False).head(top_n)
        
        tab_pos, tab_neg = st.tabs([f"Positive Keywords ({len(pos_keywords)})", f"Negative Keywords ({len(neg_keywords)})"])
        
        def render_keyword_tab(df, color_scheme='viridis'):
            if df.empty:
                st.info("No keywords found.")
                return
                
            c_viz, c_data = st.columns([1, 1])
            
            with c_viz:
                freq_dict = dict(zip(df['token'], df['LL']))
                if freq_dict:
                    wc_fig = generate_wordcloud(freq_dict, title="Keyword Cloud (by LL)", color_scheme=color_scheme)
                    st.pyplot(wc_fig)
            
            with c_data:
                st.dataframe(
                    df[['token', 'freq_t', 'freq_r', 'LL', 'LogRatio', 'Significance']], 
                    use_container_width=True
                )
                
            st.download_button(
                "Download List",
                data=df_to_excel_bytes(df),
                file_name=f"keywords_{color_scheme}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_btn_{color_scheme}_{len(df)}"
            )

        with tab_pos:
            st.caption(f"Words significantly MORE frequent in {current_name} than {ref_name}")
            render_keyword_tab(pos_keywords, 'viridis')
            
        with tab_neg:
            st.caption(f"Words significantly LESS frequent in {current_name} than {ref_name}")
            # For negative, maybe red scheme? Wordcloud supports colormaps.
            render_keyword_tab(neg_keywords, 'magma')
            
    elif results is not None:
        st.warning("No results found or calculation failed.")
