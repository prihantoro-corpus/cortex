import streamlit as st
import pandas as pd
from ui_streamlit.state_manager import get_state, set_state
from core.modules.overview import calculate_corpus_statistics, get_top_frequencies_v2, get_unique_pos_tags, get_pos_definitions, save_pos_definitions, get_corpus_language, set_corpus_language
from core.ai_service import interpret_results_llm, guess_pos_definitions
from core.preprocessing.xml_parser import format_structure_data_hierarchical, apply_xml_restrictions
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.visualiser.wordcloud import create_word_cloud
from core.io_utils import df_to_excel_bytes
import duckdb
from core.modules.classification import (
    classify_sentiment_vader, 
    classify_topics_keyword_weighted, 
    classify_topics_bertopic,
    apply_classification_by_sentence,
    BERTOPIC_AVAILABLE
)

# Standard Language Mapping
LANG_MAP = {
    "EN": "English",
    "ID": "Indonesian",
    "AR": "Arabic",
    "JP": "Japanese",
    "CH": "Chinese",
    "KO": "Korean",
    "LO": "Limola"
}

def _render_language_confirmation(path, key_suffix=""):
    """
    Renders the language selection and confirmation UI.
    """
    current_lang = get_corpus_language(path)
    
    # Try to find the code from the full name if stored that way
    current_code = "EN"
    for code, name in LANG_MAP.items():
        if current_lang == name or current_lang == code:
            current_code = code
            break

    with st.expander("🌐 Corpus Language Settings"):
        st.caption("Confirm the language of this corpus to enable dictionary and thesaurus links.")
        
        # Show currently confirmed language
        st.info(f"**Currently Confirmed:** {current_code} - {LANG_MAP.get(current_code, 'English')}")
        
        c_lang1, c_lang2 = st.columns([3, 1])
        
        lang_options = [f"{code} - {name}" for code, name in LANG_MAP.items()]
        try:
            current_idx = list(LANG_MAP.keys()).index(current_code)
        except ValueError:
            current_idx = 0
            
        with c_lang1:
            selected_fmt = st.selectbox(
                "Language", 
                lang_options,
                index=current_idx,
                key=f"lang_select_{key_suffix}",
                label_visibility="collapsed"
            )
            selected_code = selected_fmt.split(" - ")[0]
            selected_name = LANG_MAP[selected_code]

        with c_lang2:
            if st.button("Confirm", key=f"lang_confirm_{key_suffix}", use_container_width=True):
                if set_corpus_language(path, selected_name):
                    set_state('target_lang', selected_code)
                    st.toast(f"✅ {selected_code} Confirmed!", icon="✅")
                    st.rerun()
                else:
                    st.error("Failed to save language.")
    
    set_state(f'current_language_{key_suffix}', selected_name)

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
    _render_language_confirmation(path, key_suffix)


    # Show classification only if English selected/detected
    show_classification = (current_lang == 'English')

    tabs_list = ["XML", "Sub-corpus Stats", "Freq", "POS", "Cloud"]
    if show_classification: tabs_list.append("🏷️ Automatic Labeling")
    
    tabs = st.tabs(tabs_list)
    
    # Unpack tabs
    tab1, tab_sub, tab2, tab3, tab4 = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4]
    tab_cls = tabs[5] if show_classification else None
        
    with tab1:
        if error: st.error(error)
        if structure:
            html = format_structure_data_hierarchical(structure)
            st.markdown(f'<div style="font-family: monospace; font-size: 0.85em; padding: 10px; background: #1e1e1e; border-radius: 5px; color: #d4d4d4;">{html}</div>', unsafe_allow_html=True)
        else: st.caption("No XML structure.")

    with tab_sub:
        _render_subcorpus_stats(path, key_suffix)
        
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
        
    if tab_cls:
        with tab_cls:
            _render_classification_tab(path, key_suffix)

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
    
    # Language Confirmation
    _render_language_confirmation(path, "full")
        
    st.markdown("---")
    

    current_lang = get_corpus_language(path)
    show_classification = (current_lang == 'English')
    
    tabs_list = ["XML Structure", "Sub-corpus Stats", "Top Frequencies", "Unique POS Tags", "Word Cloud"]
    if show_classification: tabs_list.append("🏷️ Automatic Labeling")

    tabs = st.tabs(tabs_list)
    tab1, tab_sub, tab2, tab3, tab4 = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4]
    tab_cls = tabs[5] if show_classification else None
    
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

    with tab_sub:
        _render_subcorpus_stats(path, "full")
            
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
    if tab_cls:
        with tab_cls:
            _render_classification_tab(path, "full")

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

def _render_classification_tab(db_path, key_suffix):
    """
    Renders the Topic & Sentiment Labeling UI with method selection.
    """
    st.markdown("#### 🏷️ Automatic Corpus Labeling")
    st.caption("Automatically tag sentences with **Sentiment** and **Topic** using local NLP libraries.")
    
    # Check Columns
    try:
        con = duckdb.connect(db_path, read_only=True)
        cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
        has_topic = 'topic' in cols
        has_sent = 'sentiment' in cols
        con.close()
        
        if has_topic or has_sent:
            found = []
            if has_topic: found.append("Topic")
            if has_sent: found.append("Sentiment")
            st.success(f"✅ Existing labels found: {', '.join(found)}")
    except: pass

    st.markdown("---")
    
    # Method Selection
    method = st.radio(
        "**Topic Classification Method:**",
        options=["TF-IDF (Fast)", "BERTopic (Accurate)"],
        horizontal=True,
        help="TF-IDF: Pre-defined topics, instant. BERTopic: Auto-discovered, requires 500MB model download.",
        key=f"topic_method_{key_suffix}"
    )
    
    use_bertopic = "BERTopic" in method
    
    # Show method-specific info
    if use_bertopic:
        st.warning("⚠️ **BERTopic requires ~500MB model download** and longer processing time, but provides more accurate results.")
        
        with st.expander("🛠️ BERTopic Technical Details"):
            st.info("""
            **No data is sent to external AI servers.** All processing happens locally:
            - Uses [BERTopic](https://github.com/MaartenGr/BERTopic) for topic modeling
            - Downloads sentence-transformers model (all-MiniLM-L6-v2) on first use
            - Automatically discovers topics from your corpus content
            """)
    else:
        with st.expander("🛠️ TF-IDF Technical Details"):
            st.info("""
            **No data is sent to external AI servers.** All processing happens locally:
            - **Sentiment Analysis**: Uses [NLTK VADER](https://github.com/cjhutto/vaderSentiment) (Rule-based).
            - **Topic Classification**: Uses [Scikit-learn](https://scikit-learn.org/) TF-IDF with pre-defined keyword categories.
            """)
    
    st.markdown("---")
    
    # Configuration Section
    st.write("**Configuration:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        do_sent = st.checkbox("Sentiment (Pos/Neg/Neu)", value=True, key=f"chk_sent_{key_suffix}")
    
    with col2:
        do_topic = st.checkbox("Topic Classification", value=True, key=f"chk_topic_{key_suffix}")
    
    # BERTopic-specific parameters
    if use_bertopic and do_topic:
        st.write("**BERTopic Parameters:**")
        st.caption("💡 Tip: Use fewer topics (8-12) for clearer categorization. Higher min topic size reduces noise.")
        
        bcol1, bcol2 = st.columns(2)
        
        with bcol1:
            n_topics_option = st.selectbox(
                "Number of Topics",
                options=[8, 10, 12, 15, "Auto"] + list(range(5, 21)),
                index=1,  # Default to 10
                key=f"bertopic_n_topics_{key_suffix}",
                help="Recommended: 8-12 topics. Auto may create too many."
            )
            n_topics = 'auto' if n_topics_option == "Auto" else n_topics_option
        
        with bcol2:
            min_topic_size = st.number_input(
                "Min Topic Size",
                min_value=5,
                max_value=100,
                value=20,  # Increased from 10 to reduce topic count
                key=f"bertopic_min_size_{key_suffix}",
                help="Higher values = fewer, more distinct topics"
            )
    
    # Run Labeling Button
    if st.button("🚀 Run Labeling", key=f"run_cls_{key_suffix}", disabled=not (do_sent or do_topic)):
        with st.spinner("Processing sentences..."):
            try:
                con = duckdb.connect(db_path, read_only=True)
                df_sents = con.execute("""
                    SELECT filename, sent_id, string_agg(token, ' ' ORDER BY id) as text 
                    FROM corpus 
                    GROUP BY filename, sent_id
                """).fetch_df()
                con.close()
                
                if df_sents.empty:
                    st.error("Corpus is empty.")
                    return

                texts = df_sents['text'].tolist()
                
                # Sentiment Analysis
                if do_sent:
                    st.write("Computing Sentiment...")
                    df_sents['Predicted Sentiment'] = classify_sentiment_vader(texts)
                
                # Topic Classification
                topic_info = None
                if do_topic:
                    if use_bertopic:
                        st.write("Computing Topics with BERTopic (this may take a while)...")
                        
                        if not BERTOPIC_AVAILABLE:
                            st.error("BERTopic is not installed. Please run: `pip install bertopic sentence-transformers`")
                            return
                        
                        topic_assignments, topic_info = classify_topics_bertopic(
                            texts, 
                            n_topics=n_topics,
                            min_topic_size=min_topic_size
                        )
                        df_sents['Predicted Topic'] = topic_assignments
                    else:
                        st.write("Computing Topics with TF-IDF...")
                        topic_assignments, topic_info = classify_topics_keyword_weighted(texts)
                        df_sents['Predicted Topic'] = topic_assignments
                
                # Store results
                set_state(f'cls_preview_{key_suffix}', df_sents)
                if topic_info:
                    set_state(f'cls_topic_info_{key_suffix}', topic_info)
                
                st.toast("Labeling Complete! Preview below.", icon="🎉")
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Preview & Edit Section
    preview_df = get_state(f'cls_preview_{key_suffix}')
    topic_info = get_state(f'cls_topic_info_{key_suffix}')
    
    if preview_df is not None:
        st.divider()
        st.subheader("Preview & Edit Labels")
        
        # Topic Info Editor (if topics were generated)
        if topic_info and 'Predicted Topic' in preview_df.columns:
            st.write("**Edit Topic Labels & Keywords:**")
            st.caption("Customize the topic names and keywords before applying to corpus.")
            
            # Build editable dataframe
            topic_rows = []
            for topic_key, info in topic_info.items():
                topic_rows.append({
                    'Topic ID': str(topic_key),
                    'Label': info['label'],
                    'Keywords': ', '.join(info['keywords']) if info['keywords'] else '',
                    'Count': info['count']
                })
            
            topic_edit_df = pd.DataFrame(topic_rows)
            
            edited_topics = st.data_editor(
                topic_edit_df,
                key=f"topic_editor_{key_suffix}",
                hide_index=True,
                use_container_width=True,
                disabled=['Topic ID', 'Count'],
                column_config={
                    'Label': st.column_config.TextColumn('Topic Label', width='medium'),
                    'Keywords': st.column_config.TextColumn('Keywords (comma-separated)', width='large'),
                }
            )
            
            # Update topic assignments based on edits
            if not edited_topics.equals(topic_edit_df):
                # Create mapping from old labels to new labels
                label_map = {}
                for idx, row in edited_topics.iterrows():
                    old_label = topic_edit_df.iloc[idx]['Label']
                    new_label = row['Label']
                    label_map[old_label] = new_label
                
                # Apply mapping to preview_df
                preview_df['Predicted Topic'] = preview_df['Predicted Topic'].map(
                    lambda x: label_map.get(x, x)
                )
                set_state(f'cls_preview_{key_suffix}', preview_df)
        
        # Sample Preview
        st.write("**Sample Sentences:**")
        show_cols = ['text']
        if 'Predicted Topic' in preview_df.columns: show_cols.append('Predicted Topic')
        if 'Predicted Sentiment' in preview_df.columns: show_cols.append('Predicted Sentiment')
        
        st.dataframe(preview_df[show_cols].head(20), use_container_width=True)
        
        # Apply to Corpus
        save_col1, save_col2 = st.columns([1, 1])
        with save_col1:
            st.warning("⚠️ This will modify the corpus database. User consent is required to apply these changes.")
        with save_col2:
            if st.button("✅ I Agree, Apply to Corpus", key=f"save_cls_{key_suffix}", type="primary"):
                with st.spinner("Updating database..."):
                    success = apply_classification_by_sentence(
                        db_path, 
                        preview_df['filename'].tolist(),
                        preview_df['sent_id'].tolist(),
                        topics=preview_df['Predicted Topic'].tolist() if 'Predicted Topic' in preview_df.columns else None,
                        sentiments=preview_df['Predicted Sentiment'].tolist() if 'Predicted Sentiment' in preview_df.columns else None
                    )
                    
                    if success:
                        st.success("Corpus updated successfully!")
                        set_state(f'cls_preview_{key_suffix}', None)
                        set_state(f'cls_topic_info_{key_suffix}', None)
                        st.toast("Applied! Refreshing...", icon="💾")
                        st.rerun()
                    else:
                        st.error("Database update failed.")

def _render_subcorpus_stats(db_path, key_suffix=""):
    """
    Renders charts and tables for sub-corpus statistics:
    1. By File Name
    2. By Topic & Sentiment (if available)
    3. By XML Attributes (if available)
    """
    import plotly.express as px
    
    st.subheader("Sub-Corpus Statistics")
    
    conn = duckdb.connect(db_path, read_only=True)
    try:
        # 1. By File Name
        st.markdown("##### 📂 By File Name")
        df_files = conn.execute("SELECT filename, COUNT(*) as TokenCount FROM corpus GROUP BY filename ORDER BY TokenCount DESC").fetch_df()
        
        if not df_files.empty:
            c1, c2 = st.columns([2, 1])
            with c1:
                # Use Bar Chart for files as there might be many
                fig = px.bar(df_files, x='filename', y='TokenCount', title="Tokens per File")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(df_files, use_container_width=True, hide_index=True)
        else:
            st.info("No file information available.")
            
        st.divider()
        
        # 2. By Topic & Sentiment
        cols_info = conn.execute("PRAGMA table_info(corpus)").fetch_df()
        cols = [c.lower() for c in cols_info['name'].tolist()]
        
        has_topic = 'topic' in cols
        has_sent = 'sentiment' in cols
        
        if has_topic or has_sent:
            st.markdown("##### 🏷️ By Automatic Labeling")
            
            if has_topic:
                 # Group by distinct Topic (handling NULLs)
                topic_data = conn.execute("SELECT topic, COUNT(*) as Count FROM corpus WHERE topic IS NOT NULL GROUP BY topic ORDER BY Count DESC").fetch_df()
                if not topic_data.empty:
                    st.write("**Topic Distribution**")
                    tc1, tc2 = st.columns([1, 1])
                    with tc1:
                        fig_t = px.pie(topic_data, names='topic', values='Count', title="Topic Distribution")
                        st.plotly_chart(fig_t, use_container_width=True)
                    with tc2:
                         st.dataframe(topic_data, use_container_width=True, hide_index=True)
                else:
                    st.info("Topic column exists but no topics found. Run 'Automatic Labeling'.")

            if has_sent:
                # Group by distinct Sentiment
                sent_data = conn.execute("SELECT sentiment, COUNT(*) as Count FROM corpus WHERE sentiment IS NOT NULL GROUP BY sentiment ORDER BY Count DESC").fetch_df()
                if not sent_data.empty:
                    st.write("**Sentiment Distribution**")
                    sc1, sc2 = st.columns([1, 1])
                    with sc1:
                        fig_s = px.pie(sent_data, names='sentiment', values='Count', title="Sentiment Distribution", 
                                       color='sentiment', 
                                       color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'})
                        st.plotly_chart(fig_s, use_container_width=True)
                    with sc2:
                         st.dataframe(sent_data, use_container_width=True, hide_index=True)
                else:
                    st.info("Sentiment column exists but no sentiments found. Run 'Automatic Labeling'.")
        else:
            st.info("No Topic/Sentiment labels found. Go to the 'Automatic Labeling' tab to generate them.")
            
        st.divider()

        # 3. By XML Attributes
        from core.preprocessing.xml_parser import get_xml_attribute_columns
        attr_cols = get_xml_attribute_columns(conn)
        
        if attr_cols:
            st.markdown("##### 🧱 By XML Attributes")
            st.caption("Distribution of tokens across various document attributes.")
            
            for attr in attr_cols:
                # We limit unique values to avoid crashing charts with high-cardinality attributes (like IDs)
                unique_count = conn.execute(f"SELECT COUNT(DISTINCT {attr}) FROM corpus").fetchone()[0]
                
                if unique_count > 50:
                    st.warning(f"Attribute **{attr}** has too many unique values ({unique_count}) to visualize effectively.")
                    continue
                    
                attr_data = conn.execute(f"SELECT {attr} as Value, COUNT(*) as Count FROM corpus WHERE {attr} IS NOT NULL GROUP BY {attr} ORDER BY Count DESC").fetch_df()
                
                if not attr_data.empty:
                    st.write(f"**Attribute: {attr}**")
                    ac1, ac2 = st.columns([1, 1])
                    with ac1:
                         fig_a = px.pie(attr_data, names='Value', values='Count', title=f"Distribution by {attr}")
                         st.plotly_chart(fig_a, use_container_width=True)
                    with ac2:
                         st.dataframe(attr_data, use_container_width=True, hide_index=True)
                    st.markdown("---")
        else:
            st.caption("No additional XML attributes detected.")

    except Exception as e:
        st.error(f"Error calculating stats: {e}")
    finally:
        conn.close()
