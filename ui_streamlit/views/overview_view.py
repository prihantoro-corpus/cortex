import streamlit as st
import pandas as pd
from ui_streamlit.state_manager import get_state, set_state
from core.modules.overview import calculate_corpus_statistics, get_top_frequencies_v2, get_unique_pos_tags, get_pos_definitions, save_pos_definitions, get_corpus_language, set_corpus_language
from core.ai_service import interpret_results_llm, guess_pos_definitions
from core.preprocessing.xml_parser import format_structure_data_hierarchical, apply_xml_restrictions
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.visualiser.wordcloud import create_word_cloud
from core.io_utils import df_to_excel_bytes

def render_overview():
    st.header("Corpus Overview")
    
    comp_mode = get_state('comparison_mode', False)
    
    if not comp_mode:
        # Standard Single View
        corpus_path = get_state('current_corpus_path')
        if not corpus_path:
            st.info("Please load a corpus from the sidebar to view statistics.")
            return
        stats = get_state('corpus_stats')
        name = get_state('current_corpus_name')
        structure = get_state('xml_structure_data')
        error = get_state('xml_structure_error')
        render_full_overview(name, corpus_path, stats, structure, error)
    else:
        # Comparison Side-by-Side
        c1_path = get_state('current_corpus_path')
        c2_path = get_state('comp_corpus_path')
        
        if not c1_path and not c2_path:
            st.info("Please load corpora to compare.")
            return
            
        col_a, col_b = st.columns(2)
        
        with col_a:
            if c1_path:
                render_overview_stats(
                    get_state('current_corpus_name'),
                    c1_path,
                    get_state('corpus_stats'),
                    get_state('xml_structure_data'),
                    get_state('xml_structure_error'),
                    key_suffix="c1"
                )
            else:
                st.warning("Primary Corpus not loaded.")
                
        with col_b:
            if c2_path:
                render_overview_stats(
                    get_state('comp_corpus_name'),
                    c2_path,
                    get_state('comp_corpus_stats'),
                    get_state('comp_xml_structure_data'),
                    None, # Error for comp?
                    key_suffix="c2"
                )
            else:
                st.warning("Comparison Corpus not loaded.")

def render_overview_stats(name, path, stats, structure, error, key_suffix=""):
    st.subheader(f"📊 {name}")
    
    # --- XML Restriction Filters ---
    xml_filters = render_xml_restriction_filters(path, f"overview_{key_suffix}")
    xml_where, xml_params = apply_xml_restrictions(xml_filters)
    
    # Use restricted stats if filters are active
    if xml_filters:
        from core.modules.overview import get_restricted_stats
        display_stats = get_restricted_stats(path, xml_where_clause=xml_where, xml_params=xml_params)
    else:
        display_stats = calculate_corpus_statistics(stats)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Tokens", f"{display_stats.get('total_tokens', 0):,}")
    m2.metric("Types", f"{display_stats.get('unique_types', 0):,}")
    m3.metric("TTR", f"{display_stats.get('ttr', 0):.4f}")

    # Language Settings
    current_lang = get_corpus_language(path)
    with st.expander("🌐 Corpus Language Settings"):
        c_lang1, c_lang2 = st.columns([3, 1])
        with c_lang1:
            selected_lang = st.selectbox(
                "Language", 
                ["English", "Indonesian", "German", "French", "Spanish", "Chinese", "Japanese"], 
                index=["English", "Indonesian", "German", "French", "Spanish", "Chinese", "Japanese"].index(current_lang) if current_lang in ["English", "Indonesian", "German", "French", "Spanish", "Chinese", "Japanese"] else 0,
                key=f"lang_select_{key_suffix}",
                label_visibility="collapsed"
            )
        with c_lang2:
            if st.button("Confirm", key=f"lang_confirm_{key_suffix}", use_container_width=True):
                if set_corpus_language(path, selected_lang):
                    st.success(f"Set to {selected_lang}")
                    st.rerun()
                else:
                    st.error("Failed.")
                    
    set_state(f'current_language_{key_suffix}', selected_lang)

    tab1, tab2, tab3, tab4 = st.tabs(["XML", "Freq", "POS", "Cloud"])
    
    with tab1:
        if error: st.error(error)
        if structure:
            html = format_structure_data_hierarchical(structure)
            st.markdown(f'<div style="font-family: monospace; font-size: 0.85em; padding: 10px; background: #1e1e1e; border-radius: 5px; color: #d4d4d4;">{html}</div>', unsafe_allow_html=True)
        else: st.caption("No XML structure.")
        
    with tab2:
        df = get_top_frequencies_v2(path, limit=50, xml_where_clause=xml_where, xml_params=xml_params)
        if not df.empty:
            # Use restricted total for PMW calculation
            total = display_stats.get('total_tokens', 1)
            df['Rel. Freq'] = (df['frequency'] / total * 1_000_000).round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else: st.caption("No frequencies.")
        
    with tab3:
        _render_pos_management_tab(path, xml_where, xml_params, key_suffix)
        
    with tab4:
        f_df = get_top_frequencies_v2(path, limit=100, xml_where_clause=xml_where, xml_params=xml_params)
        if not f_df.empty:
            fig = create_word_cloud(f_df, 'pos' in f_df.columns)
            if fig: st.pyplot(fig)
        else: st.caption("No wordcloud.")

def render_full_overview(name, path, stats, structure, error):
    # --- XML Restriction Filters ---
    xml_filters = render_xml_restriction_filters(path, "overview_full")
    xml_where, xml_params = apply_xml_restrictions(xml_filters)

    # Use restricted stats if filters are active
    if xml_filters:
        from core.modules.overview import get_restricted_stats
        display_stats = get_restricted_stats(path, xml_where_clause=xml_where, xml_params=xml_params)
    else:
        display_stats = calculate_corpus_statistics(stats)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tokens", f"{display_stats.get('total_tokens', 0):,}")
    col2.metric("Unique Types", f"{display_stats.get('unique_types', 0):,}")
    col3.metric("Type/Token Ratio (TTR)", f"{display_stats.get('ttr', 0):.4f}")
        
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["XML Structure", "Top Frequencies", "Unique POS Tags", "Word Cloud"])
    
    with tab1:
        if error: st.error(error)
        if structure:
            st.subheader("Structure and Attributes")
            html = format_structure_data_hierarchical(structure)
            st.markdown(f'<div style="font-family: monospace; font-size: 0.9em; padding: 15px; background: #1e1e1e; border-radius: 8px; color: #d4d4d4;">{html}</div>', unsafe_allow_html=True)
            
            with st.expander("Show Raw Python Data (for diagnosis)"):
                 st.info("The data below is the Python dictionary successfully produced by the XML parser.")
                 st.json(structure)

            with st.expander("Database Diagnostics"):
                import duckdb
                st.write(f"DB Path: `{path}`")
                try:
                    c = duckdb.connect(path, read_only=True)
                    info = c.execute("PRAGMA table_info(corpus)").fetch_df()
                    st.write("Table Schema:", info)
                    
                    # Columns check
                    cols = info['name'].tolist()
                    standard = {'id', 'token', 'pos', 'lemma', 'sent_id', '_token_low', 'filename'}
                    meta = [c for c in cols if c not in standard]
                    st.write("Detected Metadata Columns:", meta)
                    
                    if meta:
                        rows = c.execute(f"SELECT {', '.join(meta)} FROM corpus LIMIT 5").fetch_df()
                        st.write("Sample Metadata:", rows)
                    c.close()
                except Exception as e:
                    st.error(str(e))
        else: st.info("No XML structure metadata available.")
            
    with tab2:
        st.subheader("Top Frequency Tokens")
        df = get_top_frequencies_v2(path, limit=100, xml_where_clause=xml_where, xml_params=xml_params)
        if not df.empty:
            # Use restricted total for PMW calculation
            total = display_stats.get('total_tokens', 1)
            df['Rel. Freq (per M)'] = (df['frequency'] / total * 1_000_000).round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("⬇ Download Top 100", data=df_to_excel_bytes(df), file_name=f"{name}_top_freq.xlsx")
        else: st.info("No frequency data.")

    with tab3:
        st.subheader("Unique POS Tags (and Definitions)")
        _render_pos_management_tab(path, xml_where, xml_params, "full")

    with tab4:
        st.subheader("Word Cloud")
        f_df = get_top_frequencies_v2(path, limit=100, xml_where_clause=xml_where, xml_params=xml_params)
        if not f_df.empty:
            fig = create_word_cloud(f_df, 'pos' in f_df.columns)
            if fig:
                if 'pos' in f_df.columns:
                     st.markdown('<div style="font-size: 0.8em; margin-bottom: 5px;"><span style="color:#33CC33;">●</span> Noun | <span style="color:#3366FF;">●</span> Verb | <span style="color:#FF33B5;">●</span> Adj | <span style="color:#FFCC00;">●</span> Adv</div>', unsafe_allow_html=True)
                st.pyplot(fig)
        else: st.info("No frequency data.")

    st.markdown("---")
    if st.button("🧠 Interpret Corpus Overview (LLM)", key="llm_overview_btn"):
        with st.spinner("AI is analyzing..."):
            overview_data = {"stats": display_stats, "top10": df.head(10).to_dict(orient='records') if not df.empty else {}}
            resp, err = interpret_results_llm(
                target_word=name, 
                analysis_type="Corpus Overview", 
                data_description="Stats and Freq", 
                data=str(overview_data),
                ai_provider=get_state('ai_provider'),
                gemini_api_key=get_state('gemini_api_key'),
                ollama_url=get_state('ollama_url'),
                ollama_model=get_state('ai_model')
            )
            if resp:
                set_state('llm_res_overview', resp)
            elif err:
                st.error(err)
            
    llm_res = get_state('llm_res_overview')
    if llm_res:
        with st.expander("🤖 AI Assistant Interpretation", expanded=True):
            st.markdown(llm_res)

def _render_pos_management_tab(path, xml_where, xml_params, key_suffix):
    """
    Helper to render the POS management tab content.
    """
    tags = get_unique_pos_tags(path, xml_where_clause=xml_where, xml_params=xml_params)
    
    if tags:
        # Load definitions
        current_defs = get_pos_definitions(path)
        
        # Prepare DataFrame
        data_rows = []
        for t in tags:
            data_rows.append({"Tag": t, "Definition": current_defs.get(t, "")})
        df_tags = pd.DataFrame(data_rows)
        
        st.info("Edit POS definitions. Use AI to guess, upload a file, or edit the table below.")
        
        # --- ACTION BUTTONS ---
        c1, c2 = st.columns([1, 1])
        
        with c1:
            if st.button("✨ AI Guess Definitions", key=f"ai_guess_pos_{key_suffix}"):
                with st.spinner("AI is guessing definitions..."):
                    guesses, err = guess_pos_definitions(
                        tags, 
                        ai_provider=get_state('ai_provider'),
                        gemini_api_key=get_state('gemini_api_key'),
                        ollama_url=get_state('ollama_url'),
                        ollama_model=get_state('ai_model')
                    )
                    if guesses:
                        for t in tags:
                            if t in guesses:
                                current_defs[t] = guesses[t]
                        set_state(f'temp_pos_defs_{path}', current_defs)
                        st.success("AI Guesses Applied! Review and Save.")
                        st.rerun()
                    else:
                        st.error(err or "AI provided no guesses.")

        with c2:
            with st.popover("📂 Upload / Parse", use_container_width=True):
                st.markdown("### Import Definitions")
                st.markdown("Format: `TAG: Definition` (one per line)")
                
                uploaded = st.file_uploader("Upload Text File", type=['txt'], key=f"pos_upload_{key_suffix}")
                if uploaded:
                    content = uploaded.read().decode('utf-8')
                    full_text_input = content
                else:
                    full_text_input = ""
                    
                text_input = st.text_area("Or Paste Here", value=full_text_input, height=150, key=f"pos_paste_{key_suffix}")
                
                if st.button("Process Input", key=f"pos_process_{key_suffix}"):
                    count = 0
                    for line in text_input.split('\n'):
                        line = line.strip()
                        if not line: continue
                        
                        parts = None
                        if '\t' in line:
                            parts = line.split('\t', 1)
                        elif ':' in line:
                            parts = line.split(':', 1)
                        
                        if parts:
                            t_key = parts[0].strip()
                            t_val = parts[1].strip()
                            if t_key in tags:
                                current_defs[t_key] = t_val
                                count += 1
                    
                    set_state(f'temp_pos_defs_{path}', current_defs)
                    st.success(f"Parsed {count} definitions.")
                    st.rerun()

        # --- EDITOR ---
        temp_defs = get_state(f'temp_pos_defs_{path}')
        if temp_defs:
            data_rows = [{"Tag": t, "Definition": temp_defs.get(t, "")} for t in tags]
            df_tags = pd.DataFrame(data_rows)
        
        edited_df = st.data_editor(
            df_tags, 
            key=f"pos_editor_{key_suffix}", 
            hide_index=True, 
            use_container_width=True,
            disabled=["Tag"]
        )
        
        if st.button("💾 Save Definitions", key=f"save_pos_{key_suffix}", type="primary", use_container_width=True):
            new_defs = dict(zip(edited_df['Tag'], edited_df['Definition']))
            if save_pos_definitions(path, new_defs):
                st.toast("Definitions Saved!", icon="✅")
                set_state(f'temp_pos_defs_{path}', None)
                st.rerun()
            else:
                st.error("Failed to save.")

    else:
        st.info("No POS tags detected.")
