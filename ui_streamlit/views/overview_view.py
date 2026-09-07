import streamlit as st
import os
import pandas as pd
from ui_streamlit.state_manager import get_state, set_state
from ui_streamlit.utils import notify_timing
import core.modules.overview as ov
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
import core.modules.readability as rd
import core.modules.lexical_complexity as lc

get_sentence_stats = rd.get_sentence_stats
compute_readability_metrics = rd.compute_readability_metrics
apply_reading_ease_annotation = rd.apply_reading_ease_annotation
annotate_reading_ease_by_chunks = rd.annotate_reading_ease_by_chunks
get_chunk_readability_stats = rd.get_chunk_readability_stats
calculate_formulas = rd.calculate_formulas
map_score_to_level = rd.map_score_to_level

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
    current_lang = ov.get_corpus_language(path)
    
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
            selected_fmt = st.radio(
                "Language", 
                lang_options,
                index=current_idx,
                key=f"lang_select_{key_suffix}",
                horizontal=True,
                label_visibility="collapsed"
            )
            selected_code = selected_fmt.split(" - ")[0]
            selected_name = LANG_MAP[selected_code]

        with c_lang2:
            if st.button("Confirm", key=f"lang_confirm_{key_suffix}", use_container_width=True):
                if ov.set_corpus_language(path, selected_name):
                    set_state('target_lang', selected_code)
                    st.toast(f"✅ {selected_code} Confirmed!", icon="✅")
                    st.rerun()
                else:
                    st.error("Failed to save language.")
    
    set_state(f'current_language_{key_suffix}', selected_name)

def render_custom_button_tabs(tabs_list, key_suffix=""):
    """
    Renders custom horizontal button-tabs in rows of max 5 tabs.
    Returns the selected tab name.
    """
    state_key = f'selected_tab_{key_suffix}'
    current_selection = get_state(state_key, tabs_list[0])
    if current_selection not in tabs_list:
        current_selection = tabs_list[0]
        set_state(state_key, current_selection)
        
    # Render rows of 5
    for row_start in range(0, len(tabs_list), 5):
        row_tabs = tabs_list[row_start:row_start+5]
        cols = st.columns(5)
        for idx, tab_name in enumerate(row_tabs):
            global_idx = row_start + idx
            with cols[idx]:
                is_selected = (current_selection == tab_name)
                btn_type = "primary" if is_selected else "secondary"
                if st.button(tab_name, key=f"tab_btn_{global_idx}_{key_suffix}", type=btn_type, use_container_width=True):
                    set_state(state_key, tab_name)
                    st.rerun()
                    
    return get_state(state_key, tabs_list[0])

def render_overview():
    st.header("Corpus Overview")
    
    comp_mode = get_state('comparison_mode', False)
    
    if not comp_mode:
        # Standard Single View
        corpus_path = get_state('current_corpus_path')
        source_type = get_state('source_type')
        if not corpus_path:
            st.info("👋 **Welcome to CORTEX!** Please choose a corpus to get started.")
            
            if source_type == "Online Corpus":
                render_online_builder_ui()
            elif source_type == "Built-in Corpora":
                render_built_in_corpora_selection_ui()
            else:
                render_upload_ui()
                
                with st.expander("ℹ️ Supported Formats"):
                    st.markdown("""
                    - **XML**: CORTEX extracts tokens and attributes.
                    - **TXT**: Processed via Stanza for POS and Lemmatization.
                    - **CSV/XLSX**: Must contain a column named 'token'.
                    """)
            
            return
        
        # If corpus is loaded but user wants to switch in the main area
        if source_type == "Online Corpus":
            with st.expander("🌐 Online Corpus Builder (Load New)", expanded=False):
                render_online_builder_ui()
        elif source_type == "Built-in Corpora":
            with st.expander("📚 Available Built-in Corpora (Load New)", expanded=False):
                render_built_in_corpora_selection_ui()
        elif source_type == "Upload Files":
            with st.expander("📤 Upload Corpus Files (Load New)", expanded=False):
                render_upload_ui()
            
        stats = get_state('corpus_stats')
        name = get_state('current_corpus_name')
        structure = get_state('xml_structure_data')
        error = get_state('xml_structure_error')
        render_full_overview(name, corpus_path, stats, structure, error)
    else:
        # Comparison Side-by-Side
        c1_path = get_state('current_corpus_path')
        c2_path = get_state('comp_corpus_path')
        source_type = get_state('source_type')
        
        if not c1_path and not c2_path:
            st.info("👋 **Comparison Mode Enabled.** Please load two corpora to compare.")
            
            if source_type == "Online Corpus":
                render_online_builder_ui()
            elif source_type == "Built-in Corpora":
                render_built_in_corpora_selection_ui()
            else:
                render_upload_ui()
            
            return

        # Show switcher if already loaded
        if source_type == "Online Corpus":
            with st.expander("🌐 Online Corpus Builder (Load New)", expanded=False):
                render_online_builder_ui()
        elif source_type == "Built-in Corpora":
            with st.expander("📚 Available Built-in Corpora (Load New)", expanded=False):
                render_built_in_corpora_selection_ui()
        elif source_type == "Upload Files":
            with st.expander("📤 Upload Corpus Files (Load New)", expanded=False):
                render_upload_ui()
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
        display_stats = ov.get_restricted_stats(path, xml_where_clause=xml_where, xml_params=xml_params)
    else:
        display_stats = ov.calculate_corpus_statistics(stats, db_path=path)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Tokens", f"{display_stats.get('total_tokens', 0):,}")
    m2.metric("Types", f"{display_stats.get('unique_types', 0):,}")
    m3.metric("TTR", f"{display_stats.get('ttr', 0):.4f}")

    # Show database download button for user-uploaded/built corpora
    source_type = get_state('source_type')
    if source_type in ["Upload Files", "Online Corpus"] and path and os.path.exists(path):
        st.write("") # spacing
        
        col_db, col_txt = st.columns(2)
        with col_db:
            with open(path, "rb") as db_file:
                st.download_button(
                    label="📥 Download Database (.db)",
                    data=db_file,
                    file_name=f"{name.replace(' ', '_').replace('.', '_')}_compiled.db",
                    mime="application/octet-stream",
                    help="Download this corpus as a pre-compiled DuckDB database.",
                    use_container_width=True,
                    key=f"dl_btn_{key_suffix}"
                )
        with col_txt:
            xml_cache_key = f"xml_export_{key_suffix}"
            if get_state(xml_cache_key):
                st.download_button(
                    label="📥 Download Annotated Corpus (.txt)",
                    data=get_state(xml_cache_key),
                    file_name=f"{name.replace(' ', '_').replace('.', '_')}_annotated.txt",
                    mime="text/plain",
                    help="Download the raw tagged corpus text (Word\\tPOS\\tLemma) including XML tags for NER, Sentiment, etc.",
                    use_container_width=True,
                    key=f"dl_txt_btn_{key_suffix}"
                )
            else:
                if st.button("⚙️ Generate Annotated Corpus (.txt)", key=f"gen_xml_btn_{key_suffix}", use_container_width=True, help="Compiles the current database (with all new annotations like NER and Sentiment) into a downloadable XML-tagged text file."):
                    with st.spinner("Compiling database into XML format (this may take a moment for large corpora)..."):
                        import sys
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                        from core.preprocessing.export_service import export_db_to_vertical_xml
                        xml_data = export_db_to_vertical_xml(path)
                        set_state(xml_cache_key, xml_data)
                        st.rerun()
            
    # --- Corpus Narration ---
    _render_corpus_narration(name, path, display_stats, structure, condensed=True)
    
    st.markdown("---")

    # Language Settings removed: choosing is now automatic or sidebar-driven
    # _render_language_confirmation(path, key_suffix)


    # Show classification for ALL languages now (via Translation)
    tabs_list = ["XML", "Sub-corpus Stats", "Frequency List", "POS", "Cloud", "Metadata", "🏷️ Sentiment & Topic", "🏷️ Named Entities", "🔱 Dependency Parsing", "📖 Reading Ease", "📖 Lexical Complexity"]
    
    selected_tab = render_custom_button_tabs(tabs_list, key_suffix)
    
    if selected_tab == "XML":
        if error: st.error(error)
        if structure:
            import pandas as pd
            all_attrs = []
            for tag, attrs in structure.items():
                if attrs:
                    for attr_name, vals in attrs.items():
                        sample_vals = ", ".join([str(v) for v in list(vals)[:10]])
                        if len(vals) > 10: sample_vals += ", ..."
                        all_attrs.append({"Tag": f"<{tag}>", "Attribute": attr_name, "Sample Values": sample_vals})
                else:
                    all_attrs.append({"Tag": f"<{tag}>", "Attribute": "-", "Sample Values": "-"})
            
            df = pd.DataFrame(all_attrs)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else: st.caption("No XML structure.")

    elif selected_tab == "Sub-corpus Stats":
        _render_subcorpus_stats(path, key_suffix)
        
    elif selected_tab == "Frequency List":
        df = ov.get_top_frequencies_v2(path, limit=50, xml_where_clause=xml_where, xml_params=xml_params)
        if not df.empty:
            # Use restricted total for PMW calculation
            total = display_stats.get('total_tokens', 1)
            df['Rel. Freq'] = (df['frequency'] / total * 1_000_000).round(2)
            st.caption("Displaying top 50 tokens.")
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            with st.spinner("Compiling full frequency list for download..."):
                full_df = ov.get_top_frequencies_v2(path, limit=None, xml_where_clause=xml_where, xml_params=xml_params)
                full_df['Rel. Freq'] = (full_df['frequency'] / total * 1_000_000).round(2)
                st.download_button("⬇ Download Full Frequency List", data=df_to_excel_bytes(full_df), file_name=f"{name}_full_freq.xlsx", key=f"dl_freq_{key_suffix}")
        else: st.caption("No frequencies.")
        
    elif selected_tab == "POS":
        _render_pos_management_tab(path, xml_where, xml_params, key_suffix)
        
    elif selected_tab == "Cloud":
        f_df = ov.get_top_frequencies_v2(path, limit=100, xml_where_clause=xml_where, xml_params=xml_params)
        if not f_df.empty:
            fig = create_word_cloud(f_df, 'pos' in f_df.columns)
            if fig: st.pyplot(fig)
        else: st.caption("No wordcloud.")

    elif selected_tab == "Metadata":
        _render_metadata_annotation_tab(path, key_suffix)
        
    elif selected_tab == "🏷️ Sentiment & Topic":
        _render_classification_tab(path, key_suffix)

    elif selected_tab == "🏷️ Named Entities":
        _render_ner_tab(path, key_suffix)

    elif selected_tab == "🔱 Dependency Parsing":
        _render_dependency_tab(path, key_suffix)

    elif selected_tab == "📖 Reading Ease":
        _render_reading_ease_tab(path, key_suffix)

    elif selected_tab == "📖 Lexical Complexity":
        _render_lexical_complexity_tab(path, key_suffix)

def _render_corpus_narration(name, path, display_stats, structure, condensed=False):
    """
    Builds and renders a natural-language summary paragraph above the tabs,
    strictly following the user's requested template.
    """
    con = duckdb.connect(path, read_only=True)
    try:
        # --- 1. Basic corpus size ---
        total_tokens = display_stats.get('total_tokens', 0)
        file_count = con.execute("SELECT COUNT(DISTINCT filename) FROM corpus").fetchone()[0]
        
        # --- 2. Language ---
        lang_code = ov.get_corpus_language(path) or "English"
        lang_map = {
            'en': 'English', 'id': 'Indonesian', 'fr': 'French', 'de': 'German', 
            'es': 'Spanish', 'it': 'Italian', 'pt': 'Portuguese', 'zh': 'Chinese',
            'ja': 'Japanese', 'ko': 'Korean', 'ru': 'Russian', 'ar': 'Arabic',
            'ms': 'Malay', 'vi': 'Vietnamese', 'th': 'Thai'
        }
        full_lang = lang_map.get(lang_code.lower(), lang_code)
        
        # --- 3. Column & Annotation info ---
        cols_info = con.execute("PRAGMA table_info(corpus)").fetch_df()
        all_cols = set(cols_info['name'].tolist())
        
        # Post-processed status
        has_sentiment = 'sentiment' in all_cols
        has_ner = 'ent_type' in all_cols or 'ner' in all_cols
        
        status_parts = []
        if has_sentiment and has_ner:
            status_text = "This corpus has been post-processed with sentiment and Named Entity Recognition annotators."
        elif has_sentiment:
            status_text = "This corpus has been post-processed with a sentiment annotator, but does not yet include NER."
        elif has_ner:
            status_text = "This corpus has been post-processed with a Named Entity Recognition annotator, but does not yet include sentiment analysis."
        else:
            status_text = "This corpus has not been post-processed with sentiment or Named Entity Recognition annotators."

        # --- 4. Metadata Attributes & Sub-corpora ---
        from core.preprocessing.xml_parser import get_xml_attribute_columns
        attr_cols = get_xml_attribute_columns(con)
        attr_count = len(attr_cols)
        total_unique_values = 0
        total_sub_corpora = 0
        
        if attr_count > 0:
            for col in attr_cols:
                val_count = con.execute(f'SELECT COUNT(DISTINCT "{col}") FROM corpus WHERE "{col}" IS NOT NULL').fetchone()[0]
                total_unique_values += val_count
            
            # Sub-corpora estimation: for simplicity, we count unique rows across the metadata space
            cols_str = ", ".join([f'"{c}"' for c in attr_cols])
            where_clauses = [f'"{c}" IS NOT NULL' for c in attr_cols]
            where_str = " AND ".join(where_clauses)
            query = f'SELECT COUNT(*) FROM (SELECT DISTINCT {cols_str} FROM corpus WHERE {where_str})'
            total_sub_corpora = con.execute(query).fetchone()[0]
        
        # --- 5. Lexical Diversity (STTR 100) ---
        # Note: We need a quick way to get STTR. For narration, we'll try to pull from cached complexity if possible, 
        # or calculate a representative sample if corpus is huge.
        from core.modules.lexical_complexity import calculate_generic_complexity
        
        # We take a sample of up to 20k tokens to keep it fast for the overview
        lemmas_sample = [r[0] for r in con.execute("SELECT lemma FROM corpus LIMIT 20000").fetchall()]
        complexity = calculate_generic_complexity(lemmas_sample)
        sttr_100 = complexity.get('STTR_100', 0)
        
        if sttr_100 > 0.65: div_label = "high"
        elif sttr_100 > 0.45: div_label = "moderate"
        else: div_label = "low"
        
        # --- 6. Lexical Density (LD) ---
        ld_score = 0
        ld_label = "unknown"
        pos_populated = 'pos' in all_cols and con.execute("SELECT COUNT(*) FROM corpus WHERE pos IS NOT NULL AND pos != '' LIMIT 1").fetchone()[0] > 0
        
        if pos_populated:
            lexical_count = con.execute("""
                SELECT COUNT(*) FROM corpus 
                WHERE pos GLOB 'NN*' OR pos GLOB 'VB*' OR pos GLOB 'JJ*' OR pos GLOB 'RB*'
                OR pos IN ('NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN')
            """).fetchone()[0]
            if total_tokens > 0:
                ld_score = lexical_count / total_tokens
                if ld_score < 0.45: ld_label = "low"
                elif ld_score <= 0.52: ld_label = "moderate"
                else: ld_label = "high"

        # --- 7. Reading Ease ---
        re_score = 0.0
        re_label = "not available"
        if 'fre' in all_cols or 'fkgl' in all_cols:
            re_col = 'fre' if 'fre' in all_cols else 'fkgl'
            re_score_raw = con.execute(f"SELECT AVG(CAST({re_col} AS FLOAT)) FROM corpus WHERE {re_col} IS NOT NULL").fetchone()[0]
            if re_score_raw is not None:
                re_score = re_score_raw
                if re_score >= 60: re_label = "easy to read"
                elif re_score >= 30: re_label = "moderately complex"
                else: re_label = "very difficult"

        # --- 8. Tagger Information ---
        from ui_streamlit.components.pos_help import infer_tagger_and_tagset
        tagger, tagset = infer_tagger_and_tagset(path)
        tagger_text = f"This corpus was part-of-speech tagged using {tagger} ({tagset})."

        con.close()
        
        # --- Constructing the Final Template ---
        if not condensed:
            narration_text = (
                f"The language of this corpus is {full_lang}. This corpus is composed of {file_count:,} text{'s' if file_count != 1 else ''} "
                f"and composed of {total_tokens:,} tokens overall. It has {attr_count} metadata attribute{'s' if attr_count != 1 else ''} "
                f"and {total_unique_values} unique category values. This means it has {total_sub_corpora} distinct sub-corpora combinations. "
                f"The lexical diversity is {div_label} with STTR (100) = {sttr_100:.4f}. "
                f"The lexical density is {ld_label} ({ld_score:.4f}). "
                f"The reading ease is {re_label} with an average grade/score of {re_score:.1f}. "
                f"{status_text} {tagger_text}"
            )
        else:
            # Condensed remains narrative but follows the logic
            narration_text = (
                f"A {full_lang} corpus of {total_tokens:,} tokens with {attr_count} metadata facets. "
                f"Lexically {div_label} (STTR: {sttr_100:.4f}). {status_text} {tagger_text}"
            )

        # Render Narration
        st.markdown(
            f'<div style="padding: 20px 25px; background: linear-gradient(145deg, #1e2129, #1a1c22); '
            f'border-left: 5px solid #61afef; border-radius: 12px; margin: 15px 0; '
            f'font-size: 1.15em; line-height: 1.8; color: #abb2bf; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">'
            f'📋 Corpus Narration: {narration_text}'
            f'</div>',
            unsafe_allow_html=True
        )
        
    except Exception as e:
        import traceback
        print(f"NARRATION ERROR: {e}\n{traceback.format_exc()}")
    finally:
        try:
            con.close()
        except:
            pass

def render_full_overview(name, path, stats, structure, error):
    # --- XML Restriction Filters ---
    xml_filters = render_xml_restriction_filters(path, "overview_full")
    xml_where, xml_params = apply_xml_restrictions(xml_filters)

    # Use restricted stats if filters are active
    if xml_filters:
        display_stats = ov.get_restricted_stats(path, xml_where_clause=xml_where, xml_params=xml_params)
    else:
        display_stats = ov.calculate_corpus_statistics(stats, db_path=path)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tokens", f"{display_stats.get('total_tokens', 0):,}")
    col2.metric("Unique Types", f"{display_stats.get('unique_types', 0):,}")
    col3.metric("Type/Token Ratio (TTR)", f"{display_stats.get('ttr', 0):.4f}")
    
    # Show database download button for user-uploaded/built corpora
    source_type = get_state('source_type')
    if source_type in ["Upload Files", "Online Corpus"] and path and os.path.exists(path):
        st.write("") # spacing
        
        col_db, col_txt = st.columns(2)
        with col_db:
            with open(path, "rb") as db_file:
                st.download_button(
                    label="📥 Download Pre-compiled Database (.db)",
                    data=db_file,
                    file_name=f"{name.replace(' ', '_').replace('.', '_')}_compiled.db",
                    mime="application/octet-stream",
                    help="Download this corpus as a pre-compiled DuckDB database.",
                    use_container_width=True,
                    key="dl_btn_full"
                )
        with col_txt:
            xml_cache_key = "xml_export_full"
            if get_state(xml_cache_key):
                st.download_button(
                    label="📥 Download Annotated Corpus (.txt)",
                    data=get_state(xml_cache_key),
                    file_name=f"{name.replace(' ', '_').replace('.', '_')}_annotated.txt",
                    mime="text/plain",
                    help="Download the raw tagged corpus text (Word\\tPOS\\tLemma) including XML tags for NER, Sentiment, etc.",
                    use_container_width=True,
                    key="dl_txt_btn_full"
                )
            else:
                if st.button("⚙️ Generate Annotated Corpus (.txt)", key="gen_xml_btn_full", use_container_width=True, help="Compiles the current database (with all new annotations like NER and Sentiment) into a downloadable XML-tagged text file."):
                    with st.spinner("Compiling database into XML format (this may take a moment for large corpora)..."):
                        import sys
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                        from core.preprocessing.export_service import export_db_to_vertical_xml
                        xml_data = export_db_to_vertical_xml(path)
                        set_state(xml_cache_key, xml_data)
                        st.rerun()

    # Language Confirmation removed
    # _render_language_confirmation(path, "full")
        
    st.markdown("---")
    
    # --- Corpus Narration Summary ---
    _render_corpus_narration(name, path, display_stats, structure)
    
    st.markdown("---")

    current_lang = ov.get_corpus_language(path)
    show_classification = True
    
    tabs_list = ["XML Structure", "Sub-corpus Stats", "Frequency List", "Unique POS Tags", "Word Cloud", "Metadata Annotation", "🏷️ Sentiment & Topic Analysis", "🏷️ Named Entity Recognition (NER)", "🔱 Dependency Parsing", "📖 Reading Ease", "📖 Lexical Complexity"]

    selected_tab = render_custom_button_tabs(tabs_list, "full")
    
    # Guidelines Layout using shared component
    from ui_streamlit.components.guidelines import render_guidelines
    col_main = render_guidelines("Overview", sub_tab=selected_tab)

    with col_main:
        if selected_tab == "XML Structure":
            if error: st.error(error)
            if structure:
                st.subheader("Structure and Attributes")
                
                import pandas as pd
                all_attrs = []
                for tag, attrs in structure.items():
                    if attrs:
                        for attr_name, vals in attrs.items():
                            sample_vals = ", ".join([str(v) for v in list(vals)[:10]])
                            if len(vals) > 10: sample_vals += ", ..."
                            all_attrs.append({"Tag": f"<{tag}>", "Attribute": attr_name, "Sample Values": sample_vals})
                    else:
                        all_attrs.append({"Tag": f"<{tag}>", "Attribute": "-", "Sample Values": "-"})
                
                df = pd.DataFrame(all_attrs)
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Show Raw Python Data (for diagnosis)"):
                     st.info("The data below is the Python dictionary successfully produced by the XML parser.")
                     st.json(structure)

                with st.expander("Database Diagnostics"):
                    import duckdb
                    st.write(f"DB Path: `{path}`")
                    try:
                        with duckdb.connect(path, read_only=True) as c:
                            info = c.execute("PRAGMA table_info(corpus)").fetch_df()
                            st.write("Table Schema:", info)

                        # Columns check
                        cols = info['name'].tolist()
                        standard = {'id', 'token', 'pos', 'lemma', 'sent_id', '_token_low', 'filename', 'dep_head_id'}
                        meta = [c for c in cols if c not in standard]
                        st.write("Detected Metadata Columns:", meta)

                        if meta:
                            rows = c.execute(f"SELECT {', '.join(meta)} FROM corpus LIMIT 5").fetch_df()
                            st.write("Sample Metadata:", rows)
                        c.close()
                    except Exception as e:
                        st.error(str(e))
            else: st.info("No XML structure metadata available.")

        elif selected_tab == "Sub-corpus Stats":
            _render_subcorpus_stats(path, "full")

        elif selected_tab == "Frequency List":
            st.subheader("Frequency List")
            df = ov.get_top_frequencies_v2(path, limit=100, xml_where_clause=xml_where, xml_params=xml_params)
            if not df.empty:
                # Use restricted total for PMW calculation
                total = display_stats.get('total_tokens', 1)
                df['Rel. Freq (per M)'] = (df['frequency'] / total * 1_000_000).round(2)
                st.caption("Displaying top 100 tokens. Use the button below to download the full frequency list.")
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                with st.spinner("Compiling full frequency list for download..."):
                    full_df = ov.get_top_frequencies_v2(path, limit=None, xml_where_clause=xml_where, xml_params=xml_params)
                    full_df['Rel. Freq (per M)'] = (full_df['frequency'] / total * 1_000_000).round(2)
                    st.download_button("⬇ Download Full Frequency List", data=df_to_excel_bytes(full_df), file_name=f"{name}_full_freq.xlsx")
            else: st.info("No frequency data.")

        elif selected_tab == "Unique POS Tags":
            st.subheader("Unique POS Tags (and Definitions)")
            _render_pos_management_tab(path, xml_where, xml_params, "full")

        elif selected_tab == "Word Cloud":
            st.subheader("Word Cloud")
            f_df = ov.get_top_frequencies_v2(path, limit=100, xml_where_clause=xml_where, xml_params=xml_params)
            if not f_df.empty:
                fig = create_word_cloud(f_df, 'pos' in f_df.columns)
                if fig:
                    if 'pos' in f_df.columns:
                         st.markdown('<div style="font-size: 0.8em; margin-bottom: 5px;"><span style="color:#33CC33;">●</span> Noun | <span style="color:#3366FF;">●</span> Verb | <span style="color:#FF33B5;">●</span> Adj | <span style="color:#FFCC00;">●</span> Adv</div>', unsafe_allow_html=True)
                    st.pyplot(fig)
            else: st.info("No frequency data.")

        elif selected_tab == "Metadata Annotation":
            _render_metadata_annotation_tab(path, "full")

        elif selected_tab == "🏷️ Sentiment & Topic Analysis":
            _render_classification_tab(path, "full")

        elif selected_tab == "🏷️ Named Entity Recognition (NER)":
            _render_ner_tab(path, "full")
            
        elif selected_tab == "🔱 Dependency Parsing":
            _render_dependency_tab(path, "full")

        elif selected_tab == "📖 Reading Ease":
            _render_reading_ease_tab(path, "full")

        elif selected_tab == "📖 Lexical Complexity":
            _render_lexical_complexity_tab(path, "full")

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
    tags = ov.get_unique_pos_tags(path, xml_where_clause=xml_where, xml_params=xml_params)
    
    if tags:
        # Load definitions
        current_defs = ov.get_pos_definitions(path)
        
        # Import pos_help functions
        from ui_streamlit.components.pos_help import (
            infer_tagger_and_tagset, 
            get_pos_tag_examples, 
            explain_pos_tag_via_spacy,
            UPOS_INFO, 
            PTB_INFO
        )
        
        tagger, tagset = infer_tagger_and_tagset(path)
        
        # Tagger explanation
        st.markdown(f"🤖 **Pipeline Tagger:** `{tagger}` | 🏷️ **Tagset Scheme:** `{tagset}`")
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
        tagset_lower = tagset.lower()
        if "upos" in tagset_lower or "universal" in tagset_lower:
            standard_info = UPOS_INFO
        elif "penn" in tagset_lower or "ptb" in tagset_lower:
            standard_info = PTB_INFO
        else:
            standard_info = {}

        temp_defs = get_state(f'temp_pos_defs_{path}')
        defs_to_use = temp_defs if temp_defs is not None else current_defs
        
        data_rows = []
        for t in tags:
            defn = defs_to_use.get(t, "")
            if not defn:
                if t in standard_info:
                    defn = f"{standard_info[t]['defn']} ({standard_info[t]['desc']})"
                else:
                    spacy_defn = explain_pos_tag_via_spacy(t)
                    if spacy_defn:
                        defn = spacy_defn
                # Update defs_to_use
                if defn:
                    defs_to_use[t] = defn
                
            examples = get_pos_tag_examples(path, t)
            data_rows.append({
                "Tag": t,
                "Definition": defn,
                "Examples (from corpus)": examples if examples else "None"
            })
        df_tags = pd.DataFrame(data_rows)
        
        edited_df = st.data_editor(
            df_tags, 
            key=f"pos_editor_{key_suffix}", 
            hide_index=True, 
            use_container_width=True,
            disabled=["Tag", "Examples (from corpus)"],
            column_config={
                "Tag": st.column_config.TextColumn("Tag", width=120),
                "Definition": st.column_config.TextColumn("Definition", width=350),
                "Examples (from corpus)": st.column_config.TextColumn("Examples (from corpus)", width=250)
            }
        )
        
        if st.button("💾 Save Definitions", key=f"save_pos_{key_suffix}", type="primary", use_container_width=True):
            new_defs = dict(zip(edited_df['Tag'], edited_df['Definition']))
            if ov.save_pos_definitions(path, new_defs):
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
    
    with st.expander("💡 **Method & Transparency: Classification**", expanded=False):
        st.markdown("""
        **Sentiment Analysis:** Uses the VADER lexicon to score sentences as Positive, Negative, or Neutral.
        
        **Topic Classification:**
        - **TF-IDF (Fast):** Uses pre-defined keywords to categorize text into standard topics like Sport, Politics, etc.
        - **BERTopic (Accurate):** Uses advanced embedding models to automatically discover "natural" topics in your specific corpus.
        
        **Editability:** You can rename topics or adjust keywords in the results preview before applying them.
        """)
        
    st.caption("Automatically tag sentences with **Sentiment** and **Topic** using local NLP libraries.")
    
    # Check Columns
    try:
        with duckdb.connect(db_path, read_only=True) as con:
            cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
            has_topic = 'topic' in cols
            has_sent = 'sentiment' in cols
        
        found_labels = []
        if has_topic: found_labels.append("Topic")
        if has_sent: found_labels.append("Sentiment")
        
        if found_labels:
            st.success(f"✅ Existing labels found: {', '.join(found_labels)}")
        else:
            st.info("No sentiment or topic labels found yet. Configure and run labeling below.")
    except Exception as e: 
        # st.error(f"Debug: {e}")
        pass

    # Non-English Sentiment Warning
    curr_lang = ov.get_corpus_language(db_path)
    if curr_lang and curr_lang.lower() not in ['en', 'english']:
        st.warning("⚠️ **Non-English Sentiment Analysis:** Sentences will be translated to English first. This may take significant time for large corpora and may hit translation limits.")

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
            - **Multi-language**: Non-English text is automatically translated to English for sentiment analysis.
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
            n_topics_option = st.radio(
                "Number of Topics",
                options=[8, 10, 12, 15, "Auto"] + list(range(5, 21)),
                index=1,  # Default to 10
                horizontal=True,
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
            try:
                with duckdb.connect(db_path) as con:
                    con.execute("ALTER TABLE corpus DROP COLUMN sentiment")
            except: pass
    
    # Run Labeling Button
    if st.button("🚀 Run Labeling", key=f"run_cls_{key_suffix}", disabled=not (do_sent or do_topic)):
        with st.spinner("Processing sentences..."):
            try:
                with duckdb.connect(db_path) as con:
                    df_sents = con.execute("""
                        SELECT filename, sent_id, string_agg(token, ' ' ORDER BY id) as text 
                        FROM corpus 
                        GROUP BY filename, sent_id
                    """).fetch_df()
                
                if df_sents.empty:
                    st.error("Corpus is empty.")
                    return

                texts = df_sents['text'].tolist()
                
                # Sentiment Analysis
                if do_sent:
                    st.write("Computing Sentiment...")
                    # Get current language from DB or State
                    lang_for_sent = ov.get_corpus_language(db_path)
                    df_sents['Predicted Sentiment'] = notify_timing("Sentiment analysis completed")(classify_sentiment_vader)(texts, lang=lang_for_sent)
                
                # Topic Classification
                topic_info = None
                if do_topic:
                    if use_bertopic:
                        st.write("Computing Topics with BERTopic (this may take a while)...")
                        
                        if not BERTOPIC_AVAILABLE:
                            st.error("BERTopic is not installed. Please run: `pip install bertopic sentence-transformers`")
                            return
                        
                        res = notify_timing("BERTopic classification completed")(classify_topics_bertopic)(
                            texts, 
                            n_topics=n_topics,
                            min_topic_size=min_topic_size
                        )
                        topic_assignments, topic_info = res
                        df_sents['Predicted Topic'] = topic_assignments
                    else:
                        st.write("Computing Topics with TF-IDF...")
                        res = notify_timing("TF-IDF topic classification completed")(classify_topics_keyword_weighted)(texts)
                        topic_assignments, topic_info = res
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
    
    with duckdb.connect(db_path) as conn:
        try:
            # 1. By File Name
            st.markdown("##### 📂 By File Name")
            df_files = conn.execute("""
                SELECT 
                    filename, 
                    COUNT(*) as Tokens,
                    CAST(COUNT(DISTINCT _token_low) AS FLOAT) / COUNT(*) as TTR
                FROM corpus 
                GROUP BY filename 
                ORDER BY Tokens DESC
            """).fetch_df()
            
            if not df_files.empty:
                c1, c2 = st.columns([2, 1])
                with c1:
                    # Use Bar Chart for files as there might be many
                    fig = px.bar(df_files, x='filename', y='Tokens', title="Tokens per File")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.dataframe(
                        df_files.style.format({'TTR': '{:.4f}'}), 
                        use_container_width=True, 
                        hide_index=True
                    )
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
                    unique_count = conn.execute(f'SELECT COUNT(DISTINCT "{attr}") FROM corpus').fetchone()[0]
                    
                    if unique_count > 50:
                        st.warning(f"Attribute **{attr}** has too many unique values ({unique_count}) to visualize effectively.")
                        continue
                        
                    attr_data = conn.execute(f"""
                        SELECT 
                            "{attr}" as Value, 
                            COUNT(*) as Tokens,
                            CAST(COUNT(DISTINCT _token_low) AS FLOAT) / COUNT(*) as TTR
                        FROM corpus 
                        WHERE "{attr}" IS NOT NULL 
                        GROUP BY "{attr}" 
                        ORDER BY Tokens DESC
                    """).fetch_df()
                    
                    if not attr_data.empty:
                        display_attr = "Domain" if attr == "alphabet" else attr
                        st.write(f"**Attribute: {display_attr}**")
                        ac1, ac2 = st.columns([1, 1])
                        with ac1:
                             fig_a = px.pie(attr_data, names='Value', values='Tokens', title=f"Distribution by {display_attr}")
                             st.plotly_chart(fig_a, use_container_width=True)
                        with ac2:
                             st.dataframe(
                                 attr_data.style.format({'TTR': '{:.4f}'}), 
                                 use_container_width=True, 
                                 hide_index=True
                             )
                        st.markdown("---")
            else:
                st.caption("No additional XML attributes detected.")

        except Exception as e:
            st.error(f"Error calculating stats: {e}")



def render_upload_ui():
    import core.preprocessing.corpus_loader as corpus_loader
    from core.config import STANZA_LANG_MAP
    
    st.subheader("📤 Upload Corpus Files")
    st.write("Select XML, TXT, CSV, XLSX, or DB/DUCKDB database files from your device:")
    
    uploaded_files = st.file_uploader(
        "Choose files", 
        accept_multiple_files=True,
        type=['xml', 'txt', 'csv', 'xlsx', 'db', 'duckdb', 'docx', 'pdf'],
        key="main_corpus_file_uploader"
    )
    
    # Hide options if uploading database
    is_db_upload = len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith(('.db', '.duckdb')) if uploaded_files else False
    
    # Language and Format Selection
    lang_code = 'en'
    fmt = "Raw (Natural text)"
    
    if True: # Always show language and format options
        lang_col, fmt_col = st.columns(2)
        with lang_col:
            st.markdown("**Language**")
            # Prepare language list. Add 'OTHER' at the end.
            lang_options = list(STANZA_LANG_MAP.keys()) + ["OTHER"]
            selected_lang_label = st.radio(
                "Language Select", 
                lang_options, 
                index=0,
                horizontal=True,
                key="upload_language_select",
                label_visibility="collapsed"
            )
            
            # Map label to code for processing
            if selected_lang_label == "OTHER":
                lang_code = "OTHER"
            else:
                lang_code = STANZA_LANG_MAP[selected_lang_label]
                
        with fmt_col:
            st.markdown("**Format**")
            fmt = st.radio(
                "Format Select", 
                ["Raw (Natural text)", "Tagged (Vertical)"], 
                index=0,
                horizontal=True,
                key="upload_format_select",
                label_visibility="collapsed"
            )
            
        # Custom Tagger Section
        st.markdown("---")
        tagger_tool = st.radio(
            "**Tagging Tool**",
            ["Default (TreeTagger/Stanza/Spacy)", "Custom Tagger"],
            index=0,
            horizontal=True,
            key="upload_tagger_tool_select",
            help="They are based on priority. e.g. if a language is chosen and not found in TreeTagger, we switch to Stanza, and if still not found, to Spacy."
        )
        
        custom_config = None
        if tagger_tool == "Custom Tagger":
            st.info("🔧 **Configure Custom Tagger**")
            
            custom_type = st.radio(
                "Custom Tagger Type",
                ["Data-Driven", "Rule-Based"],
                index=0,
                horizontal=True,
                key="custom_tagger_type"
            )
            
            if custom_type == "Rule-Based":
                st.warning("⚠️ Rule-based custom tagging is currently under construction.")
            else:
                custom_mode = st.radio(
                    "Model Reusability Mode",
                    ["Train New Model", "Load Existing Model"],
                    index=0,
                    horizontal=True,
                    key="custom_tagger_mode"
                )
                
                if custom_mode == "Train New Model":
                    # Data-driven tagger uploads
                    corp_col, lex_col = st.columns(2)
                    with corp_col:
                        custom_corpus_file = st.file_uploader(
                            "Upload Pre-annotated Corpus (Mandatory)",
                            type=["txt", "csv"],
                            key="custom_corpus_file_uploader",
                            help="One token per line: token TAG [lemma]. Sentences separated by blank lines."
                        )
                    with lex_col:
                        custom_lexicon_file = st.file_uploader(
                            "Upload Pre-annotated Lexicon (Optional)",
                            type=["txt", "csv"],
                            key="custom_lexicon_file_uploader",
                            help="Format: token TAG [lemma]. One entry per line."
                        )
                    
                    # Parameters
                    p_col1, p_col2 = st.columns(2)
                    with p_col1:
                        custom_guesser = st.text_input(
                            "Guesser Tag",
                            value="NN",
                            key="custom_guesser_input",
                            help="Default tag to assign to out-of-vocabulary words or low-confidence tokens."
                        )
                        
                        custom_algorithm = st.selectbox(
                            "Tagging Algorithm",
                            ["Averaged Perceptron", "Naive Bayes", "Hidden Markov Model (TnT Style)"],
                            index=0,
                            key="custom_algorithm_select"
                        )
                    with p_col2:
                        custom_window = st.slider(
                            "Context Window Size",
                            min_value=1,
                            max_value=3,
                            value=2,
                            step=1,
                            key="custom_window_slider",
                            help="Number of tokens left/right to look at. Applies to Perceptron and Naive Bayes."
                        )
                        
                        custom_threshold = st.slider(
                            "Probabilistic Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.1,
                            step=0.05,
                            key="custom_threshold_slider",
                            help="Confidence cutoff score. Below this, tagger falls back to the guesser tag."
                        )
                    
                    if custom_corpus_file:
                        try:
                            corpus_content = custom_corpus_file.read().decode('utf-8', errors='ignore')
                            lexicon_content = None
                            if custom_lexicon_file:
                                lexicon_content = custom_lexicon_file.read().decode('utf-8', errors='ignore')
                                
                            custom_config = {
                                'corpus_content': corpus_content,
                                'lexicon_content': lexicon_content,
                                'guesser_tag': custom_guesser,
                                'algorithm': custom_algorithm,
                                'context_window': custom_window,
                                'prob_threshold': custom_threshold
                            }
                        except Exception as e:
                            st.error(f"Error reading custom tagger files: {e}")
                            
                    # Download button for last trained model
                    last_model_bytes = get_state('last_trained_tagger_bytes')
                    last_model_json = get_state('last_trained_tagger_json')
                    if last_model_json or last_model_bytes:
                        st.write("") # spacer
                        d_col1, d_col2 = st.columns(2)
                        with d_col1:
                            if last_model_json:
                                st.download_button(
                                    label="📥 Download Model (.json)",
                                    data=last_model_json,
                                    file_name="custom_tagger_model.json",
                                    mime="application/json",
                                    key="download_trained_model_json_btn"
                                )
                        with d_col2:
                            if last_model_bytes:
                                st.download_button(
                                    label="📥 Download Model (.pkl)",
                                    data=last_model_bytes,
                                    file_name="custom_tagger_model.pkl",
                                    mime="application/octet-stream",
                                    key="download_trained_model_pkl_btn"
                                )
                else:
                    # Load Existing Model
                    st.write("**Load Pre-trained Model (.json or .pkl)**")
                    uploaded_model_file = st.file_uploader(
                        "Upload Trained Model File",
                        type=["json", "pkl"],
                        key="uploaded_model_file_uploader",
                        help="Upload a previously trained and downloaded .json or .pkl model file."
                    )
                    
                    if uploaded_model_file:
                        try:
                            # Read and deserialize based on file extension
                            fn = uploaded_model_file.name.lower()
                            if fn.endswith('.json'):
                                import json
                                from core.preprocessing.custom_tagger import CustomDataDrivenTagger
                                data = json.loads(uploaded_model_file.read().decode('utf-8'))
                                pre_trained_tagger = CustomDataDrivenTagger.from_json(data)
                            else:
                                import pickle
                                pre_trained_tagger = pickle.loads(uploaded_model_file.read())
                                
                            custom_config = {
                                'pre_trained_tagger': pre_trained_tagger
                            }
                            st.success("✅ Model loaded successfully! (Algorithm: " + getattr(pre_trained_tagger, 'algorithm', 'Unknown') + ")")
                        except Exception as e:
                            st.error(f"Error loading model file: {e}")
                            

    
    if uploaded_files:
        st.write("") # spacing
        if st.button("Process Uploaded Files", type="primary", use_container_width=True):
            if not is_db_upload and tagger_tool == "Custom Tagger" and not custom_config:
                if custom_mode == "Train New Model":
                    st.error("Please upload a pre-annotated corpus first to use the Custom Tagger.")
                else:
                    st.error("Please upload a pre-trained model file (.pkl) first.")
                st.stop()
            if is_db_upload:
                with st.spinner("Loading Database..."):
                    import uuid
                    import tempfile
                    import duckdb
                    import json
                    
                    first_file = uploaded_files[0]
                    unique_filename = f"corpus_{uuid.uuid4().hex}.duckdb"
                    db_path = os.path.join(tempfile.gettempdir(), unique_filename)
                    
                    try:
                        # Write uploaded bytes to temp file
                        first_file.seek(0)
                        with open(db_path, 'wb') as f:
                            f.write(first_file.read())
                            
                        try:
                            with duckdb.connect(db_path) as con:
                                con.execute("ALTER TABLE corpus DROP COLUMN ent_type")
                        except: pass
                            
                        # Query metadata
                        with duckdb.connect(db_path, read_only=True) as con:
                            tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
                        
                            if 'corpus' not in tables:
                                st.error("Uploaded database is not a valid Cortex database (missing 'corpus' table).")
                                os.remove(db_path)
                                return
                                
                            # Get language
                            lang = "English"
                            if 'corpus_metadata' in tables:
                                res_lang = con.execute("SELECT value FROM corpus_metadata WHERE key='language'").fetchone()
                                if res_lang:
                                    lang = res_lang[0]
                                    
                            # Get XML structure
                            structure = {}
                            if 'corpus_metadata' in tables:
                                res_struct = con.execute("SELECT value FROM corpus_metadata WHERE key='xml_structure'").fetchone()
                                if res_struct:
                                    try:
                                        serializable_struct = json.loads(res_struct[0])
                                        for tag in serializable_struct:
                                            structure[tag] = {}
                                            for attr in serializable_struct[tag]:
                                                structure[tag][attr] = set(serializable_struct[tag][attr])
                                    except:
                                        pass
                                        
                            # Get Stats
                            total_tokens = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
                            token_freqs = con.execute("SELECT _token_low, count(*) FROM corpus GROUP BY _token_low").fetchall()
                            token_counts = {row[0]: row[1] for row in token_freqs}
                            stats = {'token_counts': token_counts, 'total_tokens': total_tokens}
                        
                        con.close()
                        
                    except Exception as e:
                        st.error(f"Failed to read uploaded database: {e}")
                        if os.path.exists(db_path):
                            os.remove(db_path)
                        return
                    
                    # Save to state
                    if not get_state('comparison_mode'):
                        set_state('current_corpus_path', db_path)
                        set_state('corpus_stats', stats)
                        set_state('current_corpus_name', first_file.name)
                        set_state('xml_structure_data', structure)
                        set_state('target_lang', lang)
                    else:
                        if not get_state('current_corpus_path'):
                            set_state('current_corpus_path', db_path)
                            set_state('corpus_stats', stats)
                            set_state('current_corpus_name', first_file.name)
                            set_state('xml_structure_data', structure)
                        else:
                            set_state('comp_corpus_path', db_path)
                            set_state('comp_corpus_stats', stats)
                            set_state('comp_corpus_name', first_file.name)
                            set_state('comp_xml_structure_data', structure)
                            
                    st.success("Database Loaded Successfully!")
                    st.rerun()
            else:
                # Force reload logic to pick up hotfixes
                import sys
                import importlib
                try:
                    for mod in ['core.preprocessing.tagging', 'core.preprocessing.xml_parser', 'core.preprocessing.corpus_loader']:
                        if mod in sys.modules:
                            importlib.reload(sys.modules[mod])
                    st.toast("Processing modules updated! 🔄")
                except Exception as e:
                    print(f"Reload Error: {e}")

                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(val, text):
                    progress_bar.progress(val)
                    status_text.caption(text)

                with st.spinner("Processing Corpus..."):
                    result = notify_timing("Corpus loaded")(corpus_loader.load_monolingual_corpus_files)(
                        uploaded_files, 
                        explicit_lang_code=lang_code,
                        selected_format=fmt,
                        progress_callback=update_progress,
                        custom_tagger_config=custom_config
                    )
                    
                    if result.get('error'):
                        st.error(result['error'])
                    else:
                        if result.get('trained_tagger'):
                            try:
                                import pickle
                                import json
                                model_bytes = pickle.dumps(result['trained_tagger'])
                                set_state('last_trained_tagger_bytes', model_bytes)
                                
                                # Also save as JSON for maximum portability
                                model_json_str = json.dumps(result['trained_tagger'].to_json(), indent=2)
                                set_state('last_trained_tagger_json', model_json_str.encode('utf-8'))
                            except Exception as e:
                                print(f"Error serializing model: {e}")
                                
                        # Extract and save annotated corpus text
                        if result.get('annotated_corpus_text'):
                            set_state('last_annotated_corpus_text', result['annotated_corpus_text'])
                        
                        if result.get('warning'):
                            st.warning(result['warning'])
                            
                        if not get_state('comparison_mode'):
                            set_state('current_corpus_path', result['db_path'])
                            set_state('corpus_stats', result['stats'])
                            set_state('current_corpus_name', "Uploaded Batch")
                            set_state('xml_structure_data', result.get('structure'))
                            set_state('target_lang', lang_code)
                        else:
                            if not get_state('current_corpus_path'):
                                set_state('current_corpus_path', result['db_path'])
                                set_state('corpus_stats', result['stats'])
                                set_state('current_corpus_name', "Primary")
                                set_state('xml_structure_data', result.get('structure'))
                            else:
                                set_state('comp_corpus_path', result['db_path'])
                                set_state('comp_corpus_stats', result['stats'])
                                set_state('comp_corpus_name', "Comparison")
                                set_state('comp_xml_structure_data', result.get('structure'))
                        
                        st.success("Corpus Loaded Successfully!")
                        st.rerun()

def render_built_in_corpora_selection_ui():
    from core.config import get_available_corpora, BUILT_IN_CORPUS_DETAILS
    from core.preprocessing.corpus_loader import load_built_in_corpus
    
    st.subheader("📚 Available Built-in Corpora")
    st.write("Select a pre-packaged corpus below to load it directly into the session:")
    
    built_in_corpora = get_available_corpora()
    if not built_in_corpora:
        st.warning("No built-in corpora found in the local 'corpora' directory.")
        return
        
    search_query = st.text_input("🔍 Search Corpora", value="", placeholder="Type to filter corpora by name, language, description...", key="builtin_corpora_search")
    
    filtered_corpora = built_in_corpora
    if search_query:
        filtered_corpora = {}
        for name, path in built_in_corpora.items():
            detail_text = BUILT_IN_CORPUS_DETAILS.get(name, "")
            if (search_query.lower() in name.lower()) or (search_query.lower() in detail_text.lower()):
                filtered_corpora[name] = path
        if not filtered_corpora:
            st.info("No matching corpora found.")
            return
            
    for name, rel_path in filtered_corpora.items():
        with st.container(border=True):
            col_info, col_action = st.columns([4, 1])
            with col_info:
                st.markdown(f"### {name}")
                detail = BUILT_IN_CORPUS_DETAILS.get(name)
                if detail:
                    st.markdown(detail, unsafe_allow_html=True)
                else:
                    st.caption(f"Path: `{rel_path}`")
            with col_action:
                st.write("") # spacer
                st.write("") # spacer
                if st.button(f"Load {name}", key=f"load_builtin_main_{name}", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(val, text):
                        progress_bar.progress(val)
                        status_text.caption(text)
                        
                    with st.spinner(f"Loading {name}..."):
                        result = load_built_in_corpus([name], [rel_path], progress_callback=update_progress)
                        
                        if result.get('error'):
                            st.error(result['error'])
                        else:
                            if not get_state('comparison_mode'):
                                set_state('current_corpus_path', result['db_path'])
                                set_state('corpus_stats', result['stats'])
                                set_state('current_corpus_name', name)
                                set_state('xml_structure_data', result.get('structure'))
                            else:
                                if not get_state('current_corpus_path'):
                                    set_state('current_corpus_path', result['db_path'])
                                    set_state('corpus_stats', result['stats'])
                                    set_state('current_corpus_name', name)
                                    set_state('xml_structure_data', result.get('structure'))
                                else:
                                    set_state('comp_corpus_path', result['db_path'])
                                    set_state('comp_corpus_stats', result['stats'])
                                    set_state('comp_corpus_name', name)
                                    set_state('comp_xml_structure_data', result.get('structure'))
                                    
                            st.success(f"Successfully loaded {name}!")
                            st.rerun()

def render_online_builder_ui():
    import re
    import os
    mode = get_state('online_builder_mode', 'Detik.com')
    st.subheader(f"🌐 Online Corpus Builder: {mode}")
    
    if mode.startswith("Detik"):
        st.markdown("### 📰 Detik.com Tag Scraper & Corpus Builder")
        st.caption("Automatically crawl news articles by tag on Detik.com, convert multi-page articles with `?single=1`, extract title/author/date/content, and build an annotated XML corpus ready for analysis.")
        
        col1, col2 = st.columns(2)
        with col1:
            tag_val = st.text_input("Detik Tag / Category Keyword", value="ppds", placeholder="e.g. ppds, kesehatan, politik, teknologi", key="detik_tag_input", help="Base URL will be https://www.detik.com/tag/{tag}")
        with col2:
            target_count_opt = st.selectbox("Target Article Count", [50, 100, 150, 200, 300, "All (Max 500)"], index=1, key="detik_count_select")
            
        st.warning("⚠️ **Note**: Scraping larger article counts takes more processing time depending on network speed. Results are retrieved dynamically from live news feeds.")

        if st.button("🚀 Scrape Detik.com & Build Corpus", type="primary", key="btn_scrape_detik"):
            if not tag_val.strip():
                st.error("Please enter a tag keyword.")
            else:
                from core.modules.detik_scraper import build_detik_corpus_xml
                progress_bar = st.progress(0)
                status = st.empty()
                def up(m, p):
                    progress_bar.progress(min(max(p, 0.0), 1.0))
                    status.caption(m)

                with st.spinner("Scraping Detik.com news articles..."):
                    xml_content, df_summary, total_scraped = build_detik_corpus_xml(
                        tag=tag_val, 
                        target_count=target_count_opt, 
                        progress_callback=up
                    )
                    
                if xml_content and total_scraped > 0:
                    set_state('last_detik_xml_content', xml_content)
                    set_state('last_detik_df_summary', df_summary)
                    set_state('last_detik_tag', tag_val)
                    st.success(f"🎉 Successfully scraped {total_scraped} news articles for tag '{tag_val}'!")
                else:
                    st.error(f"Could not retrieve articles for tag '{tag_val}'. Please check tag spelling or try another keyword.")

        df_summary = get_state('last_detik_df_summary')
        xml_content = get_state('last_detik_xml_content')
        tag_used = get_state('last_detik_tag', 'detik')

        if df_summary is not None and not df_summary.empty:
            st.markdown(f"#### 📊 Scraped Articles Summary ({len(df_summary)} articles)")
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
            
            c_act1, c_act2 = st.columns(2)
            with c_act1:
                if st.button("📥 Load as Current Active Corpus in CORTEX", type="primary", key="btn_load_detik_active"):
                    import tempfile
                    from core.preprocessing import corpus_loader
                    temp_dir = tempfile.gettempdir()
                    xml_filename = f"Detik_{tag_used}.xml"
                    temp_xml_path = os.path.join(temp_dir, xml_filename)
                    with open(temp_xml_path, 'w', encoding='utf-8') as f:
                        f.write(xml_content)
                    
                    with open(temp_xml_path, 'rb') as f_obj:
                        f_obj.name = xml_filename
                        with st.spinner("Processing & Tagging Corpus with Stanza..."):
                            res = corpus_loader.load_monolingual_corpus_files([f_obj], explicit_lang_code='ID', selected_format='XML (Tagged)')
                            if res.get('error'):
                                st.error(res['error'])
                            else:
                                set_state('current_corpus_path', res['db_path'])
                                set_state('corpus_stats', res['stats'])
                                set_state('current_corpus_name', f"Detik_{tag_used}")
                                set_state('xml_structure_data', res.get('structure'))
                                set_state('target_lang', 'Indonesian')
                                st.success(f"🎉 'Detik_{tag_used}' loaded as current active corpus!")
                                st.rerun()
            with c_act2:
                st.download_button(
                    label=f"💾 Download Detik Corpus (XML)",
                    data=xml_content.encode('utf-8'),
                    file_name=f"Detik_{tag_used}_corpus.xml",
                    mime="application/xml",
                    key="dl_detik_xml"
                )

    elif mode == "YouTube":
        st.info("💡 **Experimental:** Max 100,000 words limit for this session.")
        url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
        opt = st.radio("Content to Download", ["Transcript only", "Comments only", "Both Transcript and Comments"], index=2)
        
        max_comments_val = 100
        strategy_val = "From top (Fastest)"
        keywords_val = []
        
        if opt in ["Comments only", "Both Transcript and Comments"]:
            col1, col2 = st.columns(2)
            with col1:
                max_comments_val = st.selectbox("Max Comments", [10, 50, 100, 250], index=2)
            with col2:
                strategy_val = st.selectbox("Comment Selection", ["From top (Fastest)", "From bottom", "Random", "By likes", "By keyword"])
                
            if strategy_val != "From top (Fastest)":
                st.warning("⚠️ Using this strategy requires downloading a large buffer of comments first. This will take extra time.")
                
            if strategy_val == "By keyword":
                kw_input = st.text_input("Enter 1-5 keywords (comma separated)")
                if kw_input:
                    keywords_val = [k.strip() for k in kw_input.split(',') if k.strip()]
                    if len(keywords_val) > 5:
                        st.warning("Only the first 5 keywords will be used.")
                        keywords_val = keywords_val[:5]
        
        mode_map = {"Transcript only": "transcript", "Comments only": "comments", "Both Transcript and Comments": "both"}
        
        if st.button("Download YouTube Data", type="primary"):
            if not url:
                st.error("Please enter a URL")
            else:
                from core.preprocessing.online_corpus import build_online_corpus
                progress_bar = st.progress(0)
                status = st.empty()
                def up(p, m):
                    progress_bar.progress(p)
                    status.caption(m)
                
                with st.spinner("Downloading..."):
                    params = {
                        "url": url, 
                        "mode": mode_map[opt],
                        "max_comments": max_comments_val,
                        "selection_strategy": strategy_val,
                        "keywords": keywords_val
                    }
                    files, warn = build_online_corpus("youtube", params, progress_callback=up)
                    if files:
                        set_state('downloaded_online_files', files)
                        st.success(f"✅ Downloaded {len(files)} components!")
                        if warn: st.warning(warn)
                    else:
                        st.error(warn or "Failed to download. Ensure the video has a transcript and comments.")

    elif mode == "Mastodon":
        st.info("💡 **Experimental:** Max 50 links and 100,000 words limit.")
        st.caption("Paste Mastodon status or profile timeline URLs (one per line).")
        urls_text = st.text_area("Mastodon URLs", height=200, placeholder="https://mastodon.social/@Mastodon/112702758163539123\nhttps://mastodon.social/@trendytoots", key="mastodon_urls_textarea")
        opt = st.radio("Content to Download", ["Post only", "Replies only", "Both Post and Replies"], index=2, key="mastodon_download_opt")
        
        max_comments_val = 100
        strategy_val = "From top (Fastest)"
        keywords_val = []
        
        if opt in ["Replies only", "Both Post and Replies"]:
            col1, col2 = st.columns(2)
            with col1:
                max_comments_val = st.selectbox("Max Replies", [10, 50, 100, 250], index=2, key="masto_max")
            with col2:
                strategy_val = st.selectbox("Reply Selection", ["From top (Fastest)", "From bottom", "Random", "By likes", "By keyword"], key="masto_strat")
                
            if strategy_val != "From top (Fastest)":
                st.warning("⚠️ Using this strategy requires fetching the entire thread first. This will take extra time.")
                
            if strategy_val == "By keyword":
                kw_input = st.text_input("Enter 1-5 keywords (comma separated)", key="masto_kw")
                if kw_input:
                    keywords_val = [k.strip() for k in kw_input.split(',') if k.strip()]
                    if len(keywords_val) > 5:
                        st.warning("Only the first 5 keywords will be used.")
                        keywords_val = keywords_val[:5]
        
        mode_map = {"Post only": "post", "Replies only": "replies", "Both Post and Replies": "both"}
        
        if st.button("Download Mastodon Data", type="primary"):
            urls = [u.strip() for u in urls_text.split('\n') if u.strip()]
            if not urls:
                st.error("No links provided")
            else:
                from core.preprocessing.online_corpus import build_online_corpus
                progress_bar = st.progress(0)
                status = st.empty()
                def up(p, m):
                    progress_bar.progress(min(p, 1.0))
                    status.caption(m)
                
                with st.spinner("Downloading..."):
                    params = {
                        "urls": urls, 
                        "mode": mode_map[opt],
                        "max_comments": max_comments_val,
                        "selection_strategy": strategy_val,
                        "keywords": keywords_val
                    }
                    files, warn = build_online_corpus("mastodon", params, progress_callback=up)
                    if files:
                        set_state('downloaded_online_files', files)
                        st.success(f"✅ Downloaded {len(files)} Mastodon components!")
                        if warn: st.warning(warn)
                    else:
                        st.error(warn or "Failed to download. Ensure the URLs are correct and public.")
 
    elif mode == "BlueSky":
        st.info("💡 **Experimental:** Max 50 links and 100,000 words limit.")
        st.caption("Paste BlueSky post or profile timeline URLs (one per line).")
        urls_text = st.text_area("BlueSky URLs", height=200, placeholder="https://bsky.app/profile/bsky.app/post/3mpok7nkjtc2o\nhttps://bsky.app/profile/academic.oup.com", key="bluesky_urls_textarea")
        opt = st.radio("Content to Download", ["Post only", "Replies only", "Both Post and Replies"], index=2, key="bluesky_download_opt")
        
        max_comments_val = 100
        strategy_val = "From top (Fastest)"
        keywords_val = []
        
        if opt in ["Replies only", "Both Post and Replies"]:
            col1, col2 = st.columns(2)
            with col1:
                max_comments_val = st.selectbox("Max Replies", [10, 50, 100, 250], index=2, key="bsky_max")
            with col2:
                strategy_val = st.selectbox("Reply Selection", ["From top (Fastest)", "From bottom", "Random", "By likes", "By keyword"], key="bsky_strat")
                
            if strategy_val != "From top (Fastest)":
                st.warning("⚠️ Using this strategy requires fetching the entire thread first. This will take extra time.")
                
            if strategy_val == "By keyword":
                kw_input = st.text_input("Enter 1-5 keywords (comma separated)", key="bsky_kw")
                if kw_input:
                    keywords_val = [k.strip() for k in kw_input.split(',') if k.strip()]
                    if len(keywords_val) > 5:
                        st.warning("Only the first 5 keywords will be used.")
                        keywords_val = keywords_val[:5]
        
        mode_map = {"Post only": "post", "Replies only": "replies", "Both Post and Replies": "both"}
        
        if st.button("Download BlueSky Data", type="primary"):
            urls = [u.strip() for u in urls_text.split('\n') if u.strip()]
            if not urls:
                st.error("No links provided")
            else:
                from core.preprocessing.online_corpus import build_online_corpus
                progress_bar = st.progress(0)
                status = st.empty()
                def up(p, m):
                    progress_bar.progress(min(p, 1.0))
                    status.caption(m)
                
                with st.spinner("Downloading..."):
                    params = {
                        "urls": urls, 
                        "mode": mode_map[opt],
                        "max_comments": max_comments_val,
                        "selection_strategy": strategy_val,
                        "keywords": keywords_val
                    }
                    files, warn = build_online_corpus("bluesky", params, progress_callback=up)
                    if files:
                        set_state('downloaded_online_files', files)
                        st.success(f"✅ Downloaded {len(files)} BlueSky components!")
                        if warn: st.warning(warn)
                    else:
                        st.error(warn or "Failed to download. Ensure the URLs are correct and public.")
                        
    elif mode == "Link Collection":
        st.info("💡 **Experimental:** Max 50 links and 500,000 words limit.")
        st.caption("Paste URLs (one per line, numbered list, or comma-separated).")
        links_text = st.text_area("URLs", height=200, placeholder="https://example.com\nhttps://test.org")
        
        if st.button("Scrape Links", type="primary"):
            import re
            links = re.findall(r'https?://[^\s,><"\']+', links_text)
            if not links:
                st.error("No valid HTTP/HTTPS URLs found in the text area.")
            else:
                from core.preprocessing.online_corpus import build_online_corpus
                progress_bar = st.progress(0)
                status = st.empty()
                def up(p, m):
                    progress_bar.progress(min(p, 1.0))
                    status.caption(m)
                
                with st.spinner(f"Scraping {len(links)} links..."):
                    files, warn = build_online_corpus("links", {"links": links}, progress_callback=up)
                    if files:
                        set_state('downloaded_online_files', files)
                        st.success(f"✅ Scraped {len(files)} out of {len(links)} pages!")
                        if warn: st.warning(warn)
                        st.rerun()
                    else:
                        st.error(warn or f"Failed to scrape content from the {len(links)} provided links.")
 
    elif mode == "Keyword Search":
        from core.preprocessing.online_corpus import OnlineCorpusBuilder
        st.info("💡 **Experimental:** Find links first, then select which to scrape (Max 500,000 words limit).")
        st.caption("Search for text-rich webpages and scrape complete sentences.")
        
        col1, col2 = st.columns(2)
        with col1:
            lang_options = ["Any", "Indonesian", "English", "Javanese", "Sundanese", "Malay"]
            selected_lang = st.selectbox("Focus Language", lang_options)
        with col2:
            num_links = st.selectbox("Number of Links to Fetch", [25, 50, 75, 100], index=1)
            
        kw_input = st.text_input("Keywords (comma separated)", placeholder="LRT, Jabodebek, presiden, peresmian")
        
        if st.button("🔍 Find Links", type="primary"):
            keywords = [k.strip() for k in kw_input.split(',') if k.strip()]
            if not keywords:
                st.error("No keywords provided")
            elif len(keywords) > 5:
                st.error("Max 5 keywords allowed.")
            else:
                progress_bar = st.progress(0)
                status = st.empty()
                def up(p, m):
                    progress_bar.progress(min(p, 1.0))
                    status.caption(m)
                
                st.info(f"⏳ Please wait, searching for {num_links} links may take up to a minute...")
                with st.spinner("Searching for related news and articles..."):
                    builder = OnlineCorpusBuilder()
                    # Append language to keywords as requested if not Any
                    links = builder.find_keyword_links(keywords, num_links=num_links, language=selected_lang, progress_callback=up)
                    if links:
                        import pandas as pd
                        df = pd.DataFrame({"Select": [True]*len(links), "URL": links})
                        set_state('found_keyword_links', df)
                        set_state('current_keywords', keywords)
                        st.success(f"✅ Found {len(links)} links!")
                    else:
                        st.error("No matching links found.")
                        
        found_links_df = get_state('found_keyword_links')
        if found_links_df is not None:
            st.write("### Review and Select Links")
            st.caption("Uncheck any links you do not want to scrape. Known easy-to-scrape domains are pushed to the top.")
            
            edited_df = st.data_editor(
                found_links_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Select": st.column_config.CheckboxColumn("Scrape?", default=True),
                    "URL": st.column_config.LinkColumn("Website URL")
                }
            )
            
            if st.button("📥 Load Corpus", type="primary", use_container_width=True):
                selected_urls = edited_df[edited_df['Select']]['URL'].tolist()
                keywords = get_state('current_keywords', [])
                if not selected_urls:
                    st.warning("No links selected!")
                else:
                    from core.preprocessing.online_corpus import build_online_corpus
                    progress_bar = st.progress(0)
                    status = st.empty()
                    def up(p, m):
                        progress_bar.progress(min(p, 1.0))
                        status.caption(m)
                    
                    st.info(f"⏳ Scraping {len(selected_urls)} links... This may take a few minutes.")
                    with st.spinner("Scraping clean sentences..."):
                        files, warn = build_online_corpus("keyword_scrape_selected", {"links": selected_urls, "keywords": keywords}, progress_callback=up)
                        if files:
                            set_state('downloaded_online_files', files)
                            st.success(f"✅ Scraped and built corpus with {len(files)} pages!")
                            if warn: st.warning(warn)
                            st.rerun()
                        else:
                            st.error("Failed to scrape any content from the selected links.")

    # Common processing section for any downloaded online files
    downloaded_files = get_state('downloaded_online_files')
    if downloaded_files:
        st.markdown("---")
        st.subheader("⚙️ Process Downloaded Corpus")
        st.success(f"{len(downloaded_files)} components ready for processing.")
        
        # Language Selection
        st.markdown("**Language**")
        from core.config import STANZA_LANG_MAP
        lang_options = list(STANZA_LANG_MAP.keys()) + ["OTHER"]
        selected_lang_label = st.radio(
            "Language Select", 
            lang_options, 
            index=0,
            horizontal=True,
            key="online_language_select",
            label_visibility="collapsed"
        )
        
        # Tagging Tool
        tagger_tool = st.radio(
            "**Tagging Tool**",
            ["Default (TreeTagger/Stanza/Spacy)", "Custom Tagger"],
            index=0,
            horizontal=True,
            key="online_tagger_tool_select",
            help="They are based on priority. e.g. if a language is chosen and not found in TreeTagger, we switch to Stanza, and if still not found, to Spacy."
        )
        
        if tagger_tool == "Custom Tagger":
            st.warning("Custom Tagger for Online Corpus is not fully wired in this view yet. Please use Default.")
            
        if st.button("Process Downloaded Files", type="primary", use_container_width=True):
            if tagger_tool == "Custom Tagger":
                st.error("Please use the Default tagger for now.")
            else:
                lang_code = "OTHER" if selected_lang_label == "OTHER" else STANZA_LANG_MAP[selected_lang_label]
                import core.preprocessing.corpus_loader as corpus_loader
                import io
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                def update_progress(val, text):
                    progress_bar.progress(val)
                    status_text.caption(text)
                    
                files_to_process = []
                for f_dict in downloaded_files:
                    buf = io.BytesIO(f_dict['content'].encode('utf-8'))
                    buf.name = f_dict['filename']
                    files_to_process.append(buf)
                    
                with st.spinner("Processing & indexing online corpus content..."):
                    result = corpus_loader.load_monolingual_corpus_files(
                        files_to_process,
                        explicit_lang_code=lang_code,
                        selected_format="Raw (Natural text)",
                        progress_callback=update_progress
                    )
                    
                    if result.get('error'):
                        st.error(result['error'])
                    else:
                        set_state('current_corpus_path', result['db_path'])
                        set_state('corpus_stats', result['stats'])
                        set_state('current_corpus_name', "Online Scraped Batch")
                        set_state('xml_structure_data', result.get('structure'))
                        set_state('target_lang', lang_code)
                        set_state('downloaded_online_files', None) # Clear buffer
                        st.success("Online corpus loaded successfully!")
                        st.rerun()

def _render_metadata_annotation_tab(db_path, key_suffix):
    import duckdb
    st.subheader("Metadata Annotation")
    
    meta_tabs = st.tabs(["📄 File Level", "✂️ Segmental Level"])
    
    with meta_tabs[0]:
        st.info("Assign attributes (e.g. Year, Genre, Author) to individual files. These attributes can then be used in **KWIC Restricted Search** and **Sub-corpus Stats**.")
        
        files = ov.get_corpus_files(db_path)
        if not files:
            st.warning("No files found in corpus.")
        else:
            # Get current metadata columns
            conn = duckdb.connect(db_path)
            cols_info = conn.execute("PRAGMA table_info(corpus)").fetch_df()
            conn.close()
            
            standard = {
                'id', 'token', 'pos', 'lemma', 'sent_id', '_token_low', 'filename', 
                'topic', 'sentiment', 'ent_type', 'dep_rel', 'dep_head_id', 
                'dep_head_token', '_conllu_id'
            }
            meta_cols = [c for c in cols_info['name'].tolist() if c.lower() not in standard and not c.lower().startswith('in_') and not c.lower().endswith(('_len', '_start', '_id'))]
            
            # State key for the working dataframe
            state_key = f"meta_editor_df_{db_path}_{key_suffix}"
            
            if get_state(state_key) is None:
                conn = duckdb.connect(db_path)
                # Fetch one sample value per filename for each metadata column
                if meta_cols:
                    select_cols = ", ".join([f"MAX({c}) as {c}" for c in meta_cols])
                    query = f"SELECT filename, {select_cols} FROM corpus GROUP BY filename ORDER BY filename"
                else:
                    query = "SELECT DISTINCT filename FROM corpus ORDER BY filename"
                    
                try:
                    df = conn.execute(query).fetch_df()
                except:
                    df = pd.DataFrame({'filename': files})
                conn.close()
                
                # Ensure all files are represented
                missing = [f for f in files if f not in df['filename'].values]
                if missing:
                    missing_df = pd.DataFrame({'filename': missing})
                    df = pd.concat([df, missing_df], ignore_index=True)
                
                set_state(state_key, df)
            
            df = get_state(state_key)
            
            # Defensive check: remove stale columns from cached session state (e.g. ent_type)
            # Only remove columns that are explicitly invalid (token-level linguistic features) 
            # instead of everything not in DB, so we don't accidentally remove newly added un-saved columns.
            stale_cols = [c for c in df.columns if c != 'filename' and (c.lower() in standard or c.lower().startswith('in_') or c.lower().endswith(('_len', '_start', '_id')))]
            if stale_cols:
                df = df.drop(columns=stale_cols)
                set_state(state_key, df)
                

            # 1. Add New Column UI
            c_add1, c_add2 = st.columns([3, 1])
            with c_add1:
                new_col_name = st.text_input("New Attribute Name (e.g. 'Genre')", key=f"new_col_input_{key_suffix}")
            with c_add2:
                st.write(" ") # alignment
                if st.button("➕ Add Attribute", key=f"add_col_btn_{key_suffix}", use_container_width=True):
                    if new_col_name and new_col_name not in df.columns:
                        df[new_col_name] = ""
                        set_state(state_key, df)
                        st.rerun()
                    elif not new_col_name:
                        st.error("Enter a name")
                    else:
                        st.warning("Already exists")

            # 2. Data Editor
            st.write("**Edit File Metadata:**")
            edited_df = st.data_editor(
                df, 
                key=f"meta_editor_widget_{key_suffix}", 
                use_container_width=True, 
                hide_index=True,
                disabled=["filename"]
            )
            
            # 3. Save Button
            if st.button("💾 Apply Metadata Annotation", type="primary", use_container_width=True, key=f"save_meta_btn_{key_suffix}"):
                with st.spinner("Applying to database..."):
                    if ov.apply_metadata_to_files(db_path, edited_df):
                        st.success("✅ Metadata successfully applied to the database!")
                        set_state(state_key, None) # Clear state to force refresh
                        st.info("The new attributes are now available for KWIC filtering and stats.")
                        st.rerun()
                    else:
                        st.error("Failed to apply metadata.")

    with meta_tabs[1]:
        st.info("Annotate specific segments within a file. You can select individual words or whole sentences.")
        
        all_files = ov.get_corpus_files(db_path)
        if not all_files:
            st.warning("No files found.")
            return

        selected_file = st.radio("Select File for Segmental Annotation", all_files, horizontal=True, key=f"seg_file_select_{key_suffix}")
        
        if selected_file:
            # 1. Word Count Check
            word_count = ov.get_file_word_count(db_path, selected_file)
            st.write(f"**File Size:** {word_count:,} words")
            
            if word_count > 5000:
                st.warning(f"⚠️ This file is too large for segmental annotation ({word_count} > 5000 words).")
                if st.button(f"✂️ Slice '{selected_file}' into 5000-word segments", key=f"slice_btn_{key_suffix}"):
                    with st.spinner("Slicing file..."):
                        if ov.slice_corpus_file(db_path, selected_file, max_words=5000):
                            st.success("File sliced successfully! Please select one of the parts.")
                            st.rerun()
                        else:
                            st.error("Failed to slice file.")
                return

            # 2. Load Tokens and Metadata
            tokens_df = ov.get_file_tokens(db_path, selected_file)
            if tokens_df.empty:
                st.info("No tokens found in this file.")
                return

            # Identify metadata columns
            standard = {'id', 'token', 'pos', 'lemma', 'sent_id', '_token_low', 'filename', 'topic', 'sentiment'}
            meta_cols = [c for c in tokens_df.columns if c.lower() not in standard]
            
            # Helper to check if a row has metadata
            def has_meta(row):
                if not meta_cols: return False
                return any(pd.notna(row[c]) and str(row[c]).strip() != "" for c in meta_cols)

            # --- SELECTION MODE ---
            sel_mode = st.radio("Selection Mode:", ["Word Selection (Natural Grid)", "Sentence Selection (List)"], horizontal=True, key=f"sel_mode_{key_suffix}")
            
            selected_token_ids = []

            if "Word Selection" in sel_mode:
                st.markdown("#### 🖱️ Word Selection Grid")
                st.caption("💡 **How to select:** Click and drag to block a range. Hold **Ctrl** to select multiple separate segments. Words with 🏷️ already have annotations.")
                
                # Prepare grid data with icons for annotated words
                words_per_row = 10
                display_tokens = []
                for _, row in tokens_df.iterrows():
                    token_text = str(row['token'])
                    if has_meta(row):
                        token_text = "🏷️ " + token_text
                    display_tokens.append(token_text)
                
                # Create 2D array
                grid_data = []
                for i in range(0, len(display_tokens), words_per_row):
                    chunk = display_tokens[i:i + words_per_row]
                    if len(chunk) < words_per_row:
                        chunk += [""] * (words_per_row - len(chunk))
                    grid_data.append(chunk)
                
                grid_df = pd.DataFrame(grid_data)
                
                selection_event = st.dataframe(
                    grid_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={i: st.column_config.TextColumn(label="", width="small") for i in range(words_per_row)},
                    on_select="rerun",
                    selection_mode="multi-cell",
                    key=f"token_grid_{selected_file}_{key_suffix}"
                )
                
                selected_cells = selection_event.get("selection", {}).get("cells", [])
                if selected_cells:
                    for cell in selected_cells:
                        r, c = (int(cell["row"]), int(cell["column"])) if isinstance(cell, dict) else (int(cell[0]), int(cell[1]))
                        token_idx = r * words_per_row + c
                        if token_idx < len(tokens_df):
                            selected_token_ids.append(int(tokens_df['id'].iloc[token_idx]))
            
            else:
                st.markdown("#### 📑 Sentence Selection List")
                st.caption("Select one or more sentences to annotate them entirely.")
                
                # Group tokens by sentence
                sentences = []
                for sid, group in tokens_df.groupby('sent_id'):
                    sent_text = " ".join(group['token'].astype(str).tolist())
                    sent_has_meta = group.apply(has_meta, axis=1).any()
                    sentences.append({
                        "sent_id": sid,
                        "Status": "🏷️ Annotated" if sent_has_meta else "Empty",
                        "Text": sent_text,
                        "_ids": group['id'].tolist()
                    })
                
                sent_df = pd.DataFrame(sentences)
                
                # Display sentence list with row selection
                sent_selection = st.dataframe(
                    sent_df[["Status", "Text"]],
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="multi-row",
                    key=f"sent_list_{selected_file}_{key_suffix}"
                )
                
                selected_rows = sent_selection.get("selection", {}).get("rows", [])
                if selected_rows:
                    for r_idx in selected_rows:
                        selected_token_ids.extend(sentences[r_idx]["_ids"])

            # --- ANNOTATION FORM ---
            if selected_token_ids:
                selected_token_ids = sorted(list(set(selected_token_ids)))
                # Preview selection
                sel_mask = tokens_df['id'].isin(selected_token_ids)
                selected_text = " ".join(tokens_df[sel_mask]['token'].astype(str).tolist())
                
                if len(selected_text) > 300:
                    selected_text = selected_text[:300] + "..."
                
                st.success(f"📌 **Selected Segment ({len(selected_token_ids)} tokens):** {selected_text}")

                with st.container(border=True):
                    st.write("**Annotate Selection**")
                    
                    history_key = f"seg_meta_history_{db_path}"
                    if history_key not in st.session_state:
                        st.session_state[history_key] = {"attributes": [], "values": {}}
                    hist = st.session_state[history_key]
                    
                    attr_val_state_key = f"cur_seg_attr_{key_suffix}"
                    val_val_state_key = f"cur_seg_val_{key_suffix}"
                    if attr_val_state_key not in st.session_state: st.session_state[attr_val_state_key] = ""
                    if val_val_state_key not in st.session_state: st.session_state[val_val_state_key] = ""
                    
                    col_attr, col_val = st.columns(2)
                    
                    with col_attr:
                        if hist["attributes"]:
                            st.caption("Reuse attribute:")
                            attr_cols = st.columns(min(len(hist["attributes"]), 4))
                            for i, a in enumerate(hist["attributes"][:4]):
                                if attr_cols[i].button(a, key=f"reuse_attr_{a}_{key_suffix}", use_container_width=True):
                                    st.session_state[attr_val_state_key] = a
                                    st.rerun()
                        
                        attr_input = st.text_input("Attribute (e.g. 'Speaker')", value=st.session_state[attr_val_state_key], key=f"seg_attr_input_{key_suffix}")
                        st.session_state[attr_val_state_key] = attr_input
                    
                    with col_val:
                        if attr_input in hist["values"] and hist["values"][attr_input]:
                            st.caption(f"Reuse for '{attr_input}':")
                            v_list = hist["values"][attr_input]
                            val_cols = st.columns(min(len(v_list), 4))
                            for i, v in enumerate(v_list[:4]):
                                if val_cols[i].button(v, key=f"reuse_val_{v}_{key_suffix}", use_container_width=True):
                                    st.session_state[val_val_state_key] = v
                                    st.rerun()
                        
                        val_input = st.text_input("Value (e.g. 'John')", value=st.session_state[val_val_state_key], key=f"seg_val_input_{key_suffix}")
                        st.session_state[val_val_state_key] = val_input

                    if st.button("💾 Apply Metadata to Selection", type="primary", use_container_width=True, key=f"apply_seg_btn_{key_suffix}"):
                        if not attr_input or not val_input:
                            st.error("Please provide both attribute and value.")
                        else:
                            with st.spinner("Applying..."):
                                meta_dict = {attr_input: val_input}
                                if ov.apply_token_metadata(db_path, selected_token_ids, meta_dict):
                                    st.toast("Metadata applied!", icon="✅")
                                    # Update history
                                    if attr_input not in hist["attributes"]: hist["attributes"].insert(0, attr_input)
                                    if attr_input not in hist["values"]: hist["values"][attr_input] = []
                                    if val_input not in hist["values"][attr_input]: hist["values"][attr_input].insert(0, val_input)
                                    st.session_state[history_key] = hist
                                    st.rerun()
                                else:
                                    st.error("Failed to apply metadata.")
            else:
                st.info("👆 Use the selection tool above to highlight words or sentences for annotation.")

            # --- CURRENT ANNOTATIONS SUMMARY ---
            st.divider()
            st.markdown("#### 📜 Current Segmental Annotations")
            
            if not meta_cols:
                st.info("No segmental metadata has been encoded for this file yet.")
            else:
                # Group tokens into segments with same metadata
                segments = []
                current_seg = None
                for _, row in tokens_df.iterrows():
                    row_meta = {c: row[c] for c in meta_cols if pd.notna(row[c]) and str(row[c]).strip() != ""}
                    if not row_meta:
                        current_seg = None
                        continue
                    if current_seg and current_seg['meta'] == row_meta:
                        current_seg['tokens'].append(row['token'])
                        current_seg['end_id'] = row['id']
                    else:
                        current_seg = {'start_id': row['id'], 'end_id': row['id'], 'tokens': [row['token']], 'meta': row_meta}
                        segments.append(current_seg)
                
                if segments:
                    summary_data = []
                    for seg in segments:
                        for attr, val in seg['meta'].items():
                            summary_data.append({"Range": f"{seg['start_id']}-{seg['end_id']}", "Text": " ".join(seg['tokens']), "Attribute": attr, "Value": val})
                    
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
                    
                    with st.expander("🛠️ Advanced: Edit Individual Tokens"):
                        mask = tokens_df[meta_cols].notna().any(axis=1) | (tokens_df[meta_cols] != "").any(axis=1)
                        editable_tokens = tokens_df[mask].copy()
                        if not editable_tokens.empty:
                            edited_tokens = st.data_editor(editable_tokens[['id', 'token'] + meta_cols], key=f"token_editor_{selected_file}_{key_suffix}", hide_index=True, disabled=['id', 'token'], use_container_width=True)
                            if st.button("💾 Save Token Edits", key=f"save_token_edits_{selected_file}"):
                                with st.spinner("Saving..."):
                                    success = True
                                    for col in meta_cols:
                                        val_groups = edited_tokens.groupby(col)
                                        for val, group in val_groups:
                                            ids = group['id'].tolist()
                                            if not ov.apply_token_metadata(db_path, ids, {col: val}): success = False
                                    if success:
                                        st.toast("Edits saved!", icon="✅")
                                        st.rerun()
                                    else:
                                        st.error("Failed to save.")
                else:
                    st.info("No segmental metadata found for this file.")

def _render_reading_ease_tab(db_path, key_suffix=""):
    import duckdb
    import pandas as pd
    
    st.subheader("📖 Reading Ease Analysis")
    
    # 1. Language Warning
    curr_lang = ov.get_corpus_language(db_path)
    if curr_lang and curr_lang.lower() not in ['en', 'english']:
        st.warning("⚠️ **Non-English Language Warning:** Readability formulas are designed and calibrated for English. For other languages, they serve as structural estimations (based on word, sentence, character, and syllable ratios) but do not strictly correspond to standard English school grades.")
    
    # 2. Clickable Transparency Link/Popover
    st.markdown("For full transparency on how readability metrics are calculated and categorized:")
    with st.expander("🔍 Click here to view mathematical formulas and difficulty classification mapping", expanded=False):
        st.markdown("""
        ### Readability Metrics & Classification Transparency
        
        #### 1. Flesch-Kincaid Grade Level
        Calculates U.S. school grade level difficulty.
        * **Formula**: `0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59`
        * **Interpretation**: Represents the educational grade level required to understand the text (e.g. 6 = 6th grade, 12 = high school senior, 16+ = university level).
        
        #### 2. Gunning Fog Index
        Measures text complexity based on sentence length and complex words.
        * **Formula**: `0.4 * ((words / sentences) + 100 * (complex_words / words))`
        * *Complex words* are defined as words containing 3 or more syllables.
        * **Interpretation**: Under 8 is easy, 8–12 is standard, 12–16 is difficult, and 17+ is very difficult.
        
        #### 3. Coleman-Liau Index
        Measures readability based on character counts and sentence ratios instead of syllables.
        * **Formula**: `0.0588 * L - 0.296 * S - 15.8`
        * *L* = average number of letters per 100 words.
        * *S* = average number of sentences per 100 words.
        * **Interpretation**: Standard grade level output (e.g. 6 = 6th grade).
        
        #### 4. Automated Readability Index (ARI)
        Calculates grade level based on characters per word and words per sentence.
        * **Formula**: `4.71 * (characters / words) + 0.5 * (words / sentences) - 21.43`
        * **Interpretation**: Standard grade level output.
        
        #### 5. SMOG Grade
        Predicts comprehension based on the count of polysyllabic words.
        * **Formula**: `1.0430 * sqrt(complex_words * 30 / sentences) + 3.1291`
        * **Interpretation**: Standard grade level output (e.g. 10 = 10th grade).
        
        ---
        
        ### Unified Difficulty Classification Matrix
        We calculate all 5 formulas for each sentence, average them, and assign the sentence to one of these 5 discrete brackets:
        
        | Bracket Name | Numerical Range (Average Grade Level) | Education / Comprehension Level |
        | :--- | :--- | :--- |
        | **1. Very Easy** | 6.0 or less | Elementary School level (up to Grade 6) |
        | **2. Easy** | from 6.1 to 8.0 | Junior High / Middle School level (Grades 6–8) |
        | **3. Standard** | from 8.1 to 12.0 | High School level (Grades 8–12) |
        | **4. Difficult** | from 12.1 to 16.0 | College / University Prep level (Grades 12–16) |
        | **5. Very Difficult** | greater than 16.0 | Graduate / Professional level (Grades 16+) |
        """)
        
    st.markdown("---")
    
    # 3. Perform Calculations
    with st.spinner("Analyzing readability metrics..."):
        sentence_df = get_sentence_stats(db_path)
        
    if sentence_df.empty:
        st.info("No valid text data found in the corpus.")
        return
        
    metrics_data = compute_readability_metrics(sentence_df)
    
    # 4. Display Overall metrics
    st.markdown("#### 📊 Overall Corpus Readability")
    overall = metrics_data['overall']['metrics']
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Flesch-Kincaid", f"{overall['Flesch-Kincaid Grade Level']}")
    m2.metric("Gunning Fog", f"{overall['Gunning Fog']}")
    m3.metric("Coleman-Liau", f"{overall['Coleman-Liau']}")
    m4.metric("ARI", f"{overall['ARI']}")
    m5.metric("SMOG", f"{overall['SMOG']}")
    
    # Combined average and bracket
    avg_score = round(sum(overall.values()) / len(overall), 2)
    overall_bracket = map_score_to_level(avg_score)
    st.info(f"**Overall Corpus Classification:** {overall_bracket} (Average Grade Level: **{avg_score}**)")
    
    # 5. Display File-level table
    st.markdown("#### 📄 Corpus Files Readability")
    file_rows = []
    for fname, fmetrics in metrics_data['files'].items():
        favg = round(sum(fmetrics.values()) / len(fmetrics), 2)
        fbracket = map_score_to_level(favg)
        file_rows.append({
            'Filename': fname,
            'Flesch-Kincaid': fmetrics['Flesch-Kincaid Grade Level'],
            'Gunning Fog': fmetrics['Gunning Fog'],
            'Coleman-Liau': fmetrics['Coleman-Liau'],
            'ARI': fmetrics['ARI'],
            'SMOG': fmetrics['SMOG'],
            'Average GL': favg,
            'Difficulty Level': fbracket
        })
    st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)
    
    # 5.1. Chunk-Level Readability Breakdown
    st.markdown("#### ✂️ Chunk-Level Readability Breakdown")
    st.caption("Analyze the text in sequential blocks of words. This is useful for single-file corpora to see readability progression and identify difficult passages.")
    
    col_ch1, col_ch2 = st.columns([1, 2])
    with col_ch1:
        selected_chunk_size = st.radio(
            "Select Chunk Size (Words):",
            options=[100, 1000, 10000, 100000],
            index=1, # Default to 1000
            horizontal=True,
            key=f"chunk_size_select_{key_suffix}"
        )
        
    with st.spinner(f"Analyzing readability per {selected_chunk_size:,} words..."):
        chunk_stats = get_chunk_readability_stats(db_path, selected_chunk_size)
        
    if chunk_stats:
        chunk_df = pd.DataFrame(chunk_stats)
        st.dataframe(chunk_df, use_container_width=True, hide_index=True)
    else:
        st.info("No chunk statistics generated.")
        
    # 6. Display Sub-corpora grouping
    st.markdown("#### 🧱 Sub-corpora Readability")
    sub_options = list(metrics_data['subcorpora'].keys())
    
    if not sub_options:
        st.info("No sub-corpora attributes (such as Topic, Sentiment, or XML Attributes) detected.")
    else:
        selected_sub = st.radio(
            "Select Sub-corpus Grouping Category:",
            options=sub_options,
            horizontal=True,
            key=f"sub_readability_select_{key_suffix}"
        )
        if selected_sub:
            sub_rows = []
            for gval, smetrics in metrics_data['subcorpora'][selected_sub].items():
                savg = round(sum(smetrics.values()) / len(smetrics), 2)
                sbracket = map_score_to_level(savg)
                sub_rows.append({
                    selected_sub: gval,
                    'Flesch-Kincaid': smetrics['Flesch-Kincaid Grade Level'],
                    'Gunning Fog': smetrics['Gunning Fog'],
                    'Coleman-Liau': smetrics['Coleman-Liau'],
                    'ARI': smetrics['ARI'],
                    'SMOG': smetrics['SMOG'],
                    'Average GL': savg,
                    'Difficulty Level': sbracket
                })
            st.dataframe(pd.DataFrame(sub_rows), use_container_width=True, hide_index=True)
            
    # 7. Database Annotation Section
    st.divider()
    st.markdown("#### 🚀 Database Readability Annotation")
    st.caption("Annotate the corpus database with Reading Ease difficulty levels. Once annotated, the levels will appear as sub-corpora, and you can restrict searches to specific difficulty ranges using the filter panels.")
    
    conn = duckdb.connect(db_path)
    cols = [c[1] for c in conn.execute("PRAGMA table_info(corpus)").fetchall()]
    conn.close()
    
    has_reading_ease_level = 'reading_ease_level' in cols
    if has_reading_ease_level:
        st.success("✅ **Reading Ease levels are already annotated in this corpus.** You can re-run annotation at any time if the corpus text changes.")
    else:
        st.info("Reading Ease levels have not been annotated yet. Run annotation below to enable difficulty-level filtering.")
    
    # Selection of annotation unit / scope
    col_ann1, col_ann2 = st.columns([1, 1])
    with col_ann1:
        ann_scope = st.radio(
            "**Annotation Granularity:**",
            options=["Sentence Level", "Chunk Level"],
            help=f"Sentence Level: Calculates difficulty for every individual sentence (great for multi-document/varied corpora). Chunk Level: Breaks the text into segments of the selected size ({selected_chunk_size:,} words) (recommended for single large texts/flat corpora to see difficulty segments).",
            key=f"ann_scope_{key_suffix}"
        )
        
    if st.button("Annotate Reading Ease Levels", key=f"btn_annotate_reading_ease_{key_suffix}", type="primary"):
        if ann_scope == "Sentence Level":
            with st.spinner("Analyzing and annotating sentences..."):
                filenames = []
                sent_ids = []
                levels = []
                
                for _, row in sentence_df.iterrows():
                    smetrics = calculate_formulas(
                        int(row['words']),
                        int(row['sentences']),
                        int(row['syllables']),
                        int(row['characters']),
                        int(row['complex_words'])
                    )
                    savg = sum(smetrics.values()) / len(smetrics)
                    slevel = map_score_to_level(savg)
                    
                    filenames.append(row['filename'])
                    sent_ids.append(row['sent_id'])
                    levels.append(slevel)
                    
                if apply_reading_ease_annotation(db_path, filenames, sent_ids, levels):
                    st.toast("Reading Ease Levels Annotated successfully!", icon="✅")
                    st.rerun()
                else:
                    st.error("Failed to write annotations to database.")
        else:
            with st.spinner(f"Analyzing and annotating {selected_chunk_size:,}-word chunks..."):
                if annotate_reading_ease_by_chunks(db_path, chunk_size=selected_chunk_size):
                    st.toast(f"Reading Ease Levels Annotated by {selected_chunk_size:,}-word chunks successfully!", icon="✅")
                    st.rerun()
                else:
                    st.error("Failed to write annotations to database.")


def _render_ner_tab(db_path, key_suffix=""):
    """
    Renders the Named Entity Recognition UI and output views.
    """
    import plotly.express as px
    import core.modules.ner_service as ner
    
    st.markdown("#### 🏷️ Named Entity Recognition (NER)")
    
    with st.expander("💡 **Method & Transparency: NER**", expanded=False):
        st.markdown("""
        **Dependency-based NER (spaCy):** Extracts entities using grammatical dependency trees and pre-trained models. Automatically classifies entities into categories (e.g., PERSON, ORG, GPE/Location).
        
        **Regex-based NER:** Scans the corpus text using custom regular expressions to match entities (e.g., Emails, URLs, Dates) under custom labels.
        """)
        
    method = st.radio(
        "**NER Extraction Method:**",
        options=["Dependency-based (spaCy)", "Regex-based (Custom Patterns)"],
        horizontal=True,
        key=f"ner_method_{key_suffix}"
    )
    
    is_spacy = "spaCy" in method
    
    if is_spacy:
        st.caption("Extract standard semantic entities using a local spaCy pipeline.")
        model_name = st.radio(
            "spaCy Pipeline Model",
            options=["en_core_web_sm", "en_core_web_md", "xx_ent_wiki_sm"],
            index=0,
            horizontal=True,
            key=f"spacy_model_{key_suffix}",
            help="en_core_web_sm: Fast & lightweight. xx_ent_wiki_sm: Multilingual entity detector."
        )
    else:
        st.caption("Identify entities by matching regular expression patterns.")
        default_regex_input = (
            "Emails: \\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b\n"
            "URLs: https?://[^\\s<>\"]+|www\\.[^\\s<>\"]+\n"
            "Dates: \\b\\d{4}[-/.]\\d{2}[-/.]\\d{2}\\b|\\b\\d{2}[-/.]\\d{2}[-/.]\\d{4}\\b"
        )
        regex_input = st.text_area(
            "Define Regex Categories (Format: `Label: Pattern` per line)",
            value=default_regex_input,
            height=120,
            key=f"ner_regex_patterns_{key_suffix}"
        )
        
    if st.button("🚀 Run NER Analysis", key=f"run_ner_btn_{key_suffix}", type="primary"):
        with st.spinner("Running Named Entity Recognition on corpus sentences..."):
            try:
                if is_spacy:
                    df_flat, df_matrix_files, df_matrix_top, raw_ents = ner.run_spacy_ner(db_path, model_name=model_name)
                else:
                    # Parse regex input
                    patterns_dict = {}
                    for line in regex_input.split('\n'):
                        line = line.strip()
                        if not line or ':' not in line:
                            continue
                        cat, pat = line.split(':', 1)
                        patterns_dict[cat.strip()] = pat.strip()
                        
                    if not patterns_dict:
                        st.error("Please define at least one valid Category: Pattern line.")
                        return
                        
                    df_flat, df_matrix_files, df_matrix_top, raw_ents = ner.run_regex_ner(db_path, patterns_dict)
                    
                set_state(f'ner_flat_{key_suffix}', df_flat)
                set_state(f'ner_matrix_files_{key_suffix}', df_matrix_files)
                set_state(f'ner_matrix_top_{key_suffix}', df_matrix_top)
                set_state(f'ner_raw_entities_{key_suffix}', raw_ents)
                
                st.toast("Named Entity Recognition completed successfully!", icon="🎉")
                st.rerun()
            except Exception as e:
                st.error(f"NER failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                
    # Display Results
    df_flat = get_state(f'ner_flat_{key_suffix}')
    df_matrix_files = get_state(f'ner_matrix_files_{key_suffix}')
    df_matrix_top = get_state(f'ner_matrix_top_{key_suffix}')
    
    if df_flat is not None and not df_flat.empty:
        st.divider()
        st.subheader("📊 NER Findings & Distribution")
        
        # High-level Metrics
        total_ents = df_flat['Frequency'].sum()
        uniq_ents = len(df_flat['Entity'].unique())
        top_row = df_flat.iloc[0] if not df_flat.empty else None
        
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Total Entities Found", f"{total_ents:,}")
        mcol2.metric("Unique Entities", f"{uniq_ents:,}")
        if top_row is not None:
            mcol3.metric("Top Entity (Freq)", f"{top_row['Entity']} ({top_row['Frequency']})")
            
        r_tab1, r_tab2, r_tab3, r_tab4 = st.tabs([
            "📊 Frequency Distribution", 
            "📁 Matrix: Category vs. Files", 
            "🏆 Matrix: Top Entities by Category", 
            "📋 All Matches"
        ])
        
        with r_tab1:
            st.markdown("##### Entity Category Distribution")
            df_cat = df_flat.groupby('Category')['Frequency'].sum().reset_index()
            fig_pie = px.pie(df_cat, names='Category', values='Frequency', title="Entity Counts by Category", hole=0.4)
            fig_pie.update_layout(margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("##### Top 15 Overall Entities")
            fig_bar = px.bar(df_flat.head(15), x='Entity', y='Frequency', color='Category', title="Top 15 Most Frequent Entities")
            fig_bar.update_layout(xaxis={'categoryorder':'total descending'}, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with r_tab2:
            st.markdown("##### Category Counts per File")
            st.caption("This pivot matrix shows the occurrence of entity categories across the files in the corpus.")
            st.dataframe(df_matrix_files, use_container_width=True, hide_index=True)
            
        with r_tab3:
            st.markdown("##### Top Entities side-by-side by Category")
            st.caption("Wide matrix representation showing the top recognized terms and their frequencies for each category.")
            st.dataframe(df_matrix_top, use_container_width=True, hide_index=True)
            
        with r_tab4:
            st.markdown("##### Complete Identified Entities")
            st.dataframe(df_flat, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇ Download NER Results (Excel)",
                data=df_to_excel_bytes(df_flat),
                file_name=f"cortex_ner_results_{key_suffix}.xlsx"
            )

        # Database Annotation Section
        st.divider()
        st.markdown("#### 🚀 Database XML Annotation")
        st.caption("Annotate the corpus database with the identified Named Entities. Once annotated, they will be searchable in the Concordance tab as `<NER CATEGORY=\"Entity\">` tags (e.g. `<NER PERSON=\"Sarah Johnson\">` or `<NER ORG=\"*\">`).")
        
        raw_ents = get_state(f'ner_raw_entities_{key_suffix}')
        if st.button("Annotate Corpus with XML Tags", key=f"btn_annotate_ner_{key_suffix}", type="primary"):
            with st.spinner("Annotating database with XML tags..."):
                if ner.annotate_ner_tags_in_db(db_path, raw_ents):
                    st.toast("Corpus Annotated with XML tags successfully! 🚀", icon="✅")
                    # Clear query cache to reload tag definitions
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Failed to write XML annotations to the database.")
    elif df_flat is not None:
        st.info("No entities were detected in the corpus matching the selected criteria.")

def _render_dependency_tab(db_path, key_suffix=""):
    import core.modules.dependency_service as dep
    st.subheader("🔱 Dependency Parsing")
    
    with st.expander("💡 Method & Transparency: Dependency Parsing", expanded=False):
        st.markdown("""
        **spaCy Dependency Parsing:** Analyzes the grammatical structure of sentences, identifying relationships between "head" words and their "dependents".
        * **dep_rel**: The type of relationship (e.g., `nsubj` for nominal subject, `obj` for object).
        * **dep_head_id**: The unique ID of the word that governs this token.
        """)

    # Check current status
    stats = dep.get_dependency_stats(db_path)
    
    if stats is not None and not stats.empty:
        st.success("✅ **Dependency relations are already annotated in this corpus.**")
        st.write("You can use the 'Restrictions' panel in the Concordance or Statistics tabs to filter searches by specific relations (e.g. searching only for nominal subjects).")
        
        st.markdown("#### 📊 Dependency Relation Distribution")
        st.dataframe(stats, use_container_width=True, hide_index=True)
    else:
        st.info("Dependency relations have not been annotated for this corpus yet.")

    st.divider()
    st.write("**Annotate Dependency Relations**")
    model_name = st.radio(
        "spaCy Pipeline Model",
        options=["en_core_web_sm", "en_core_web_md", "xx_ent_wiki_sm"],
        index=0,
        horizontal=True,
        key=f"dep_spacy_model_{key_suffix}",
        help="en_core_web_sm: Fast. xx_ent_wiki_sm: Multilingual."
    )
    
    if st.button("🚀 Run Dependency Parsing Annotation", key=f"run_dep_btn_{key_suffix}", type="primary"):
        with st.spinner("Analyzing grammatical structure..."):
            if dep.run_dependency_parsing(db_path, model_name=model_name):
                st.toast("Dependency parsing completed successfully!", icon="🔱")
                st.rerun()
            else:
                st.error("Failed to run dependency parsing.")


def _render_lexical_complexity_tab(db_path, key_suffix=""):
    st.subheader("📖 Lexical Complexity Analysis")
    
    # Scan for available wordlists in the corpus language folder
    lang_name = ov.get_corpus_language(db_path)
    lang_clean = lang_name.strip().lower()
    
    lang_map = {
        "en": "english", "id": "indonesian", "ar": "arabic", 
        "jp": "japanese", "ch": "chinese", "ko": "korean", 
        "lo": "limola", "hi": "hindi", "jv": "javanese"
    }
    mapped_lang = lang_map.get(lang_clean, lang_clean)
    
    wl_dir = os.path.join("wordlist", mapped_lang)
    if not os.path.isdir(wl_dir):
        wl_dir = os.path.join("..", "wordlist", mapped_lang)
        
    available_wordlists = {}
    if os.path.isdir(wl_dir):
        for f in os.listdir(wl_dir):
            if f.endswith(".txt") or f.endswith(".csv"):
                # Clean name for display
                clean_name = f.replace("_wordlist.txt", "").replace("_stats.csv", "").replace("_", " ").title()
                available_wordlists[clean_name] = os.path.join(wl_dir, f)
                
    selected_wl_path = None
    if available_wordlists:
        selected_wl_name = st.radio(
            "📚 **Select Lexical Sophistication Reference Wordlist:**",
            options=list(available_wordlists.keys()),
            horizontal=True,
            key=f"wl_complexity_select_{key_suffix}"
        )
        selected_wl_path = available_wordlists[selected_wl_name]
    else:
        st.info(f"No reference wordlists found in '{os.path.join('wordlist', mapped_lang)}' for sophistication analysis.")

    # Scan for sub-corpus XML attributes
    from core.preprocessing.xml_parser import get_xml_attribute_columns
    
    conn = duckdb.connect(db_path)
    attr_cols = get_xml_attribute_columns(conn)
    conn.close()
    
    # Filter filename out of XML attributes
    if attr_cols and "filename" in attr_cols:
        attr_cols.remove("filename")
        
    st.markdown("### 📊 Analysis Level & Grouping")
    grouping_basis = st.radio(
        "Group comparative analysis by:",
        options=["File-by-file", "By Sub-corpus"],
        index=0,
        horizontal=True,
        key=f"complexity_grouping_basis_{key_suffix}"
    )
        
    group_col = "filename"
    if grouping_basis == "By Sub-corpus":
        if attr_cols:
            selected_attr = st.radio(
                "Select XML attribute to group sub-corpora:",
                options=attr_cols,
                horizontal=True,
                key=f"complexity_group_attr_{key_suffix}"
            )
            group_col = selected_attr
        else:
            st.warning("No sub-corpus attributes (XML tags) found in the database. Defaulting to File-by-file grouping.")

    with st.spinner("Analyzing lexical complexity metrics..."):
        results = lc.calculate_corpus_lexical_complexity(db_path, selected_wl_path, group_by_column=group_col)
        
    if not results:
        st.info("No text data found in the corpus.")
        return
        
    lang = results.get("language", "English")
    overall = results.get("overall", {})
    files = results.get("files", {})
    
    # Render two side-by-side columns/boxes
    box1, box2 = st.columns(2)
    
    with box1:
        st.markdown('<div style="padding: 15px; border: 1px solid #3b3f46; border-radius: 8px; background: #1a1c23; margin-bottom: 20px;">'
                    '<h4 style="margin: 0; color: #4f8bf9;">⚙️ Generic Lexical Complexity</h4>'
                    '<p style="font-size: 0.9em; color: #888; margin: 5px 0 0 0;">Language-agnostic measures calculated using lemma type and token distributions (no POS tagging required).</p>'
                    '</div>', unsafe_allow_html=True)
        
        gen_overall = overall.get("generic", {})
        if gen_overall:
            def get_ttr_label(v):
                if v < 0.40: return "Low Variation"
                if v <= 0.70: return "Average Variation"
                return "High Variation"

            def get_rttr_label(v):
                if v < 4.5: return "Low Variation"
                if v <= 7.0: return "Average Variation"
                return "High Variation"

            def get_cttr_label(v):
                if v < 3.0: return "Low Variation"
                if v <= 5.0: return "Average Variation"
                return "High Variation"

            def get_logttr_label(v):
                if v < 0.80: return "Low Variation"
                if v <= 0.90: return "Average Variation"
                return "High Variation"

            def get_uber_label(v):
                if v < 20.0: return "Low Variation"
                if v <= 35.0: return "Average Variation"
                return "High Variation"

            def get_mtld_label(v):
                if v < 50.0: return "Low / Repetitive"
                if v <= 80.0: return "Medium / Intermediate"
                if v <= 110.0: return "High / Academic"
                return "Very High Variation"

            def get_sttr_label(v):
                if v < 0.65: return "Low Variation"
                if v <= 0.78: return "Average Variation"
                return "High Variation"

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Tokens (N)", f"{gen_overall.get('N', 0):,}", help="Total number of word tokens (running words) in the corpus.")
                st.caption("Total Word Count")
            with m2:
                st.metric("Types (V)", f"{gen_overall.get('V', 0):,}", help="Total number of unique word lemmas (types) in the corpus.")
                st.caption("Unique Word Count")
            with m3:
                val = gen_overall.get('TTR', 0.0)
                st.metric("TTR", f"{val:.4f}", help="Type-Token Ratio (V/N). Measures vocabulary variation. Highly sensitive to text length (longer texts have lower TTR).")
                st.caption(get_ttr_label(val))
            
            m4, m5, m6 = st.columns(3)
            with m4:
                val = gen_overall.get('RTTR', 0.0)
                st.metric("Guiraud (RTTR)", f"{val:.4f}", help="Root Type-Token Ratio (V / sqrt(N)). Controls for text length better than TTR.")
                st.caption(get_rttr_label(val))
            with m5:
                val = gen_overall.get('CTTR', 0.0)
                st.metric("Carroll (CTTR)", f"{val:.4f}", help="Corrected Type-Token Ratio (V / sqrt(2*N)). Adjusts for sample size.")
                st.caption(get_cttr_label(val))
            with m6:
                val = gen_overall.get('LogTTR', 0.0)
                st.metric("Herdan (LogTTR)", f"{val:.4f}", help="Logarithmic Type-Token Ratio (log(V) / log(N)). Reduces effect of sample size.")
                st.caption(get_logttr_label(val))
            
            m7, m8, m9 = st.columns(3)
            with m7:
                val = gen_overall.get('Uber', 0.0)
                st.metric("Uber Index", f"{val:.4f}", help="Uber Index (log(N)^2 / (log(N) - log(V))). A highly stable measure across different text lengths.")
                st.caption(get_uber_label(val))
            with m8:
                val = gen_overall.get('MTLD', 0.0)
                st.metric("MTLD", f"{val:.4f}", help="Measure of Textual Lexical Diversity. Calculates the mean length of sequential word runs that maintain a target TTR. Highly robust to text length.")
                st.caption(get_mtld_label(val))
            with m9:
                val = gen_overall.get('STTR_50', 0.0)
                st.metric("STTR / MSTTR (50)", f"{val:.4f}", help="Mean Segmental Type-Token Ratio calculated across non-overlapping segments of 50 words.")
                st.caption(get_sttr_label(val))

            m10, m11 = st.columns(2)
            with m10:
                val = gen_overall.get('STTR_100', 0.0)
                st.metric("STTR / MSTTR (100)", f"{val:.4f}", help="Mean Segmental Type-Token Ratio calculated across non-overlapping segments of 100 words. Ideal benchmark standard.")
                st.caption(get_sttr_label(val))
            with m11:
                val = gen_overall.get('STTR_200', 0.0)
                st.metric("STTR / MSTTR (200)", f"{val:.4f}", help="Mean Segmental Type-Token Ratio calculated across non-overlapping segments of 200 words.")
                st.caption(get_sttr_label(val))
            
    with box2:
        st.markdown('<div style="padding: 15px; border: 1px solid #3b3f46; border-radius: 8px; background: #1a1c23; margin-bottom: 20px;">'
                    '<h4 style="margin: 0; color: #e06c75;"><a href="https://sites.psu.edu/xxl13/lca/" target="_blank" style="color: #e06c75; text-decoration: none;">🏷️ Lu\'s LCA</a></h4>'
                    '<p style="font-size: 0.9em; color: #888; margin: 5px 0 0 0;">POS-tag and language dependent metrics. Computes Lexical Density, Class Variations, and Sophistication.</p>'
                    '</div>', unsafe_allow_html=True)
        
        # User warning
        st.warning("⚠️ **Linguistic Interpretation Warning:** Specific metrics rely on the accuracy of the corpus language settings, POS tagging, lemmatization, and POS category mappings. Please check again these conditions before making conclusions.")
        
        spec_overall = overall.get("specific", {})
        if not spec_overall:
            st.info("Specific POS complexity metrics are not available for this corpus (check that POS tags are populated and defined).")
        else:
            def get_ld_label(v):
                if v < 0.45: return "Sparse / Spoken Style"
                if v <= 0.52: return "Standard Density"
                return "Dense / Written Prose"

            def get_lv_label(v):
                if v < 0.60: return "Low Variation"
                if v <= 0.80: return "Average Variation"
                return "High Variation"

            def get_nv_label(v):
                if v < 0.15: return "Low Noun Diversity"
                if v <= 0.25: return "Average Noun Diversity"
                return "High Noun Diversity"

            def get_vv1_label(v):
                if v < 0.50: return "Low Verb Diversity"
                if v <= 0.75: return "Average Verb Diversity"
                return "High Verb Diversity"

            def get_vv2_label(v):
                if v < 0.10: return "Low Verb Diversity"
                if v <= 0.20: return "Average Verb Diversity"
                return "High Verb Diversity"

            def get_cvv1_label(v):
                if v < 2.0: return "Low Verb Diversity"
                if v <= 3.5: return "Average Verb Diversity"
                return "High Verb Diversity"

            def get_adjv_label(v):
                if v < 0.05: return "Low Adj Diversity"
                if v <= 0.12: return "Average Adj Diversity"
                return "High Adj Diversity"

            def get_advv_label(v):
                if v < 0.02: return "Low Adv Diversity"
                if v <= 0.08: return "Average Adv Diversity"
                return "High Adv Diversity"

            def get_modv_label(v):
                if v < 0.08: return "Low Mod Diversity"
                if v <= 0.20: return "Average Mod Diversity"
                return "High Mod Diversity"

            def get_ls_label(v):
                if v < 0.15: return "Low Sophistication"
                if v <= 0.35: return "Average Sophistication"
                return "High Sophistication"

            def get_vs_label(v):
                if v < 0.10: return "Low Verb Sophistication"
                if v <= 0.30: return "Average Verb Sophistication"
                return "High Verb Sophistication"

            def get_cvs1_label(v):
                if v < 1.0: return "Low Verb Sophistication"
                if v <= 2.2: return "Average Verb Sophistication"
                return "High Verb Sophistication"

            m1, m2, m3 = st.columns(3)
            with m1:
                val = spec_overall.get('LD', 0.0)
                st.metric("Lexical Density (LD)", f"{val:.4f}", help="Ratio of lexical words (nouns, verbs, adjectives, adverbs) to total words. Measures information density.")
                st.caption(get_ld_label(val))
            with m2:
                val = spec_overall.get('LV', 0.0)
                st.metric("Lexical Variation (LV)", f"{val:.4f}", help="Ratio of unique lexical words to total lexical words. Measures content word vocabulary range.")
                st.caption(get_lv_label(val))
            with m3:
                val = spec_overall.get('NV', 0.0)
                st.metric("Noun Variation (NV)", f"{val:.4f}", help="Ratio of unique nouns to total lexical words. Measures noun diversity.")
                st.caption(get_nv_label(val))
            
            m4, m5, m6 = st.columns(3)
            with m4:
                val = spec_overall.get('VV1', 0.0)
                st.metric("Verb Variation (VV1)", f"{val:.4f}", help="Ratio of unique verbs to total verbs. Measures verb vocabulary variation.")
                st.caption(get_vv1_label(val))
            with m5:
                val = spec_overall.get('VV2', 0.0)
                st.metric("Verb Variation (VV2)", f"{val:.4f}", help="Ratio of unique verbs to total lexical words. Verb diversity scaled to corpus size.")
                st.caption(get_vv2_label(val))
            with m6:
                val = spec_overall.get('CVV1', 0.0)
                st.metric("Corr. Verb Var (CVV1)", f"{val:.4f}", help="Corrected Verb Variation (unique verbs / sqrt(2 * total verbs)). Controls for text length.")
                st.caption(get_cvv1_label(val))
            
            m7, m8, m9 = st.columns(3)
            with m7:
                val = spec_overall.get('AdjV', 0.0)
                st.metric("Adj Variation (AdjV)", f"{val:.4f}", help="Ratio of adjectives to total lexical words. Measures adjective diversity.")
                st.caption(get_adjv_label(val))
            with m8:
                val = spec_overall.get('AdvV', 0.0)
                st.metric("Adv Variation (AdvV)", f"{val:.4f}", help="Ratio of adverbs to total lexical words. Measures adverb diversity.")
                st.caption(get_advv_label(val))
            with m9:
                val = spec_overall.get('ModV', 0.0)
                st.metric("Modifier Var (ModV)", f"{val:.4f}", help="Ratio of modifiers (adjectives + adverbs) to total lexical words. Measures modifier density.")
                st.caption(get_modv_label(val))
            
            if "LS1" in spec_overall:
                st.markdown("**Sophistication Metrics (NGSL Reference)**")
                m10, m11, m12 = st.columns(3)
                with m10:
                    val = spec_overall.get('LS1', 0.0)
                    st.metric("Lexical Soph. (LS1)", f"{val:.4f}", help="Ratio of sophisticated word tokens (not in top frequency bands of the selected reference wordlist) to total lexical words.")
                    st.caption(get_ls_label(val))
                with m11:
                    val = spec_overall.get('LS2', 0.0)
                    st.metric("Lexical Soph. (LS2)", f"{val:.4f}", help="Ratio of sophisticated word types to total lexical word types.")
                    st.caption(get_ls_label(val))
                with m12:
                    vs1_val = spec_overall.get('VS1')
                    if vs1_val is not None:
                        st.metric("Verb Soph. (VS1)", f"{vs1_val:.4f}", help="Ratio of sophisticated verb tokens to total verb tokens.")
                        st.caption(get_vs_label(vs1_val))
                    else:
                        st.metric("Verb Soph. (VS1)", "N/A", help="No verbs were detected by POS tagging. Check POS definitions.")
                        st.caption("⚠️ No verbs detected")
                
                m13, m14 = st.columns([1, 2])
                with m13:
                    vs2_val = spec_overall.get('VS2')
                    if vs2_val is not None:
                        st.metric("Verb Soph. (VS2)", f"{vs2_val:.4f}", help="Ratio of sophisticated verb types to total verb types.")
                        st.caption(get_vs_label(vs2_val))
                    else:
                        st.metric("Verb Soph. (VS2)", "N/A", help="No verbs were detected by POS tagging. Check POS definitions.")
                        st.caption("⚠️ No verbs detected")
                with m14:
                    cvs1_val = spec_overall.get('CVS1')
                    if cvs1_val is not None:
                        st.metric("Corr. Verb Soph. (CVS1)", f"{cvs1_val:.4f}", help="Corrected Verb Sophistication (sophisticated verb types / sqrt(2 * total verbs)). Controls for length.")
                        st.caption(get_cvs1_label(cvs1_val))
                    else:
                        st.metric("Corr. Verb Soph. (CVS1)", "N/A", help="No verbs were detected by POS tagging. Check POS definitions.")
                        st.caption("⚠️ No verbs detected")
                
                # Diagnostic info for verb sophistication
                _verb_tokens = spec_overall.get('_verb_tokens', 0)
                if _verb_tokens == 0:
                    st.info("ℹ️ **Verb Sophistication is N/A** because no verbs were identified by the POS tagger. "
                            "This can happen if POS tag definitions are missing or don't map to the 'verb' category. "
                            "Go to the **Unique POS Tags** tab to review and correct POS definitions (ensure verb tags have 'verb' in their definition).")
                elif vs1_val == 0.0 and vs2_val == 0.0:
                    st.info(f"ℹ️ **Verb Sophistication is 0.0** — all {_verb_tokens} verb tokens in the corpus use lemmas "
                            f"found in the reference wordlist (NGSL). This is normal for corpora with basic/common vocabulary.")

    st.markdown("---")
    if group_col == "filename":
        st.markdown("### 📄 File-by-file Comparative Analysis")
        group_col_name = "Filename"
    else:
        st.markdown(f"### 🧱 Sub-corpus Comparative Analysis (grouped by '{group_col}')")
        group_col_name = f"Sub-corpus ({group_col})"
    
    # Prepare comparative dataframe
    file_rows = []
    for fname, f_metrics in files.items():
        row = {group_col_name: fname}
        
        # Generic
        f_gen = f_metrics.get("generic", {})
        row.update({
            "Tokens (N)": f_gen.get("N", 0),
            "Types (V)": f_gen.get("V", 0),
            "TTR": f_gen.get("TTR", 0.0),
            "MTLD": f_gen.get("MTLD", 0.0),
            "STTR (50)": f_gen.get("STTR_50", 0.0),
            "STTR (100)": f_gen.get("STTR_100", 0.0),
            "STTR (200)": f_gen.get("STTR_200", 0.0),
            "Guiraud (RTTR)": f_gen.get("RTTR", 0.0),
            "Carroll (CTTR)": f_gen.get("CTTR", 0.0),
            "Herdan (LogTTR)": f_gen.get("LogTTR", 0.0),
            "Uber Index": f_gen.get("Uber", 0.0),
        })
        
        # Specific
        f_spec = f_metrics.get("specific", {})
        if f_spec:
            row.update({
                "Lexical Density (LD)": f_spec.get("LD", 0.0),
                "Lexical Variation (LV)": f_spec.get("LV", 0.0),
                "Noun Variation (NV)": f_spec.get("NV", 0.0),
                "Verb Variation (VV1)": f_spec.get("VV1", 0.0),
                "Verb Variation (VV2)": f_spec.get("VV2", 0.0),
                "Adj Variation (AdjV)": f_spec.get("AdjV", 0.0),
                "Adv Variation (AdvV)": f_spec.get("AdvV", 0.0),
                "Modifier Var (ModV)": f_spec.get("ModV", 0.0),
            })
            if "LS1" in f_spec:
                row.update({
                    "Lexical Soph. (LS1)": f_spec.get("LS1", 0.0),
                    "Lexical Soph. (LS2)": f_spec.get("LS2", 0.0),
                    "Verb Soph. (VS1)": f_spec.get("VS1"),
                    "Verb Soph. (VS2)": f_spec.get("VS2"),
                    "Corr. Verb Soph. (CVS1)": f_spec.get("CVS1"),
                })
        file_rows.append(row)
        
    df_files = pd.DataFrame(file_rows)
    st.dataframe(df_files, use_container_width=True, hide_index=True)
    
    st.download_button(
        "⬇ Download Lexical Complexity (Excel)",
        data=df_to_excel_bytes(df_files),
        file_name=f"lexical_complexity_{group_col}_{key_suffix}.xlsx"
    )

    st.markdown("---")
    with st.expander("📚 Metric Definitions & Academic References"):
        st.markdown(
            "### Metric Formulas & Definitions\n"
            "Here $N$ is the total count of tokens, and $V$ is the total count of unique lemmas (types).\n\n"
            "#### ⚙️ Generic Lexical Complexity Formulas\n"
            "- **Type-Token Ratio (TTR)**:  \n"
            "  $$TTR = \\frac{V}{N}$$\n"
            "- **Guiraud's Root TTR (RTTR)**:  \n"
            "  $$RTTR = \\frac{V}{\\sqrt{N}}$$\n"
            "- **Carroll's Corrected TTR (CTTR)**:  \n"
            "  $$CTTR = \\frac{V}{\\sqrt{2N}}$$\n"
            "- **Herdan's LogTTR (LogTTR)**:  \n"
            "  $$LogTTR = \\frac{\\log(V)}{\\log(N)}$$\n"
            "- **Uber Index**:  \n"
            "  $$Uber = \\frac{\\log(N)^2}{\\log(N) - \\log(V)}$$\n"
            "- **MTLD (Measure of Textual Lexical Diversity)**:  \n"
            "  Calculates the average length of text segments (runs) that maintain a Type-Token Ratio above a threshold (typically $0.72$), computed in both forward and backward directions.\n"
            "- **MSTTR (Mean Segmental TTR)**:  \n"
            "  $$\\text{MSTTR}(W) = \\frac{1}{K} \\sum_{i=1}^{K} \\text{TTR}_i$$\n"
            "  where the text is divided into $K$ non-overlapping segments of fixed size $W$ (e.g., $50, 100, 200$).\n\n"
            "#### 🏷️ Lu's LCA (POS-Specific) Formulas\n"
            "Let $N_{lex}$ and $V_{lex}$ denote the total tokens and unique types of lexical (content) words: nouns, verbs, adjectives, and adverbs.\n\n"
            "- **Lexical Density (LD)**:  \n"
            "  $$LD = \\frac{N_{lex}}{N}$$\n"
            "- **Lexical Variation (LV)**:  \n"
            "  $$LV = \\frac{V_{lex}}{N_{lex}}$$\n"
            "- **Noun Variation (NV)**:  \n"
            "  $$NV = \\frac{V_{noun}}{N_{lex}}$$\n"
            "- **Verb Variation (VV1, VV2, CVV1)**:  \n"
            "  $$VV1 = \\frac{V_{verb}}{N_{verb}}, \\quad VV2 = \\frac{V_{verb}}{N_{lex}}, \\quad CVV1 = \\frac{V_{verb}}{\\sqrt{2N_{verb}}}$$\n"
            "- **Adjective Variation (AdjV)**:  \n"
            "  $$AdjV = \\frac{V_{adj}}{N_{lex}}$$\n"
            "- **Adverb Variation (AdvV)**:  \n"
            "  $$AdvV = \\frac{V_{adv}}{N_{lex}}$$\n"
            "- **Modifier Variation (ModV)**:  \n"
            "  $$ModV = \\frac{V_{adj} + V_{adv}}{N_{lex}}$$\n"
            "- **Lexical Sophistication (LS1, LS2)**:  \n"
            "  Let $N_{soph}$ and $V_{soph}$ represent lexical tokens and types that are not found within the high-frequency bands of the selected reference wordlist:\n"
            "  $$LS1 = \\frac{N_{soph}}{N_{lex}}, \\quad LS2 = \\frac{V_{soph}}{V_{lex}}$$\n"
            "- **Verb Sophistication (VS1, VS2, CVS1)**:  \n"
            "  $$VS1 = \\frac{N_{v, soph}}{N_{verb}}, \\quad VS2 = \\frac{V_{v, soph}}{V_{verb}}, \\quad CVS1 = \\frac{V_{v, soph}}{\\sqrt{2N_{verb}}}$$\n\n"
            "#### 📊 Interpretation Ranges & Benchmark Ratings\n"
            "These empirical rating thresholds are used for the labels displayed beneath each metric:\n\n"
            "* **Type-Token Ratio (TTR)**: Low ($<0.40$), Average ($0.40 - 0.70$), High ($>0.70$)\n"
            "* **Guiraud (RTTR)**: Low ($<4.5$), Average ($4.5 - 7.0$), High ($>7.0$)\n"
            "* **Carroll (CTTR)**: Low ($<3.0$), Average ($3.0 - 5.0$), High ($>5.0$)\n"
            "* **Herdan (LogTTR)**: Low ($<0.80$), Average ($0.80 - 0.90$), High ($>0.90$)\n"
            "* **Uber Index**: Low ($<20.0$), Average ($20.0 - 35.0$), High ($>35.0$)\n"
            "* **MTLD**: Low / Repetitive ($<50.0$), Medium / Intermediate ($50.0 - 80.0$), High / Academic ($80.0 - 110.0$), Very High ($>110.0$)\n"
            "* **MSTTR (50, 100, 200)**: Low ($<0.65$), Average ($0.65 - 0.78$), High ($>0.78$)\n"
            "* **Lexical Density (LD)**: Sparse / Spoken ($<0.45$), Standard ($0.45 - 0.52$), Dense / Written ($>0.52$)\n"
            "* **Lexical Variation (LV)**: Low ($<0.60$), Average ($0.60 - 0.80$), High ($>0.80$)\n"
            "* **Noun Variation (NV)**: Low ($<0.15$), Average ($0.15 - 0.25$), High ($>0.25$)\n"
            "* **Verb Variation (VV1)**: Low ($<0.50$), Average ($0.50 - 0.75$), High ($>0.75$)\n"
            "* **Verb Variation (VV2)**: Low ($<0.10$), Average ($0.10 - 0.20$), High ($>0.20$)\n"
            "* **Corrected Verb Variation (CVV1)**: Low ($<2.0$), Average ($2.0 - 3.5$), High ($>3.5$)\n"
            "* **Adjective Variation (AdjV)**: Low ($<0.05$), Average ($0.05 - 0.12$), High ($>0.12$)\n"
            "* **Adverb Variation (AdvV)**: Low ($<0.02$), Average ($0.02 - 0.08$), High ($>0.08$)\n"
            "* **Modifier Variation (ModV)**: Low ($<0.08$), Average ($0.08 - 0.20$), High ($>0.20$)\n"
            "* **Lexical Sophistication (LS1, LS2)**: Low ($<0.15$), Average ($0.15 - 0.35$), High ($>0.35$)\n"
            "* **Verb Sophistication (VS1, VS2)**: Low ($<0.10$), Average ($0.10 - 0.30$), High ($>0.30$)\n"
            "* **Corrected Verb Sophistication (CVS1)**: Low ($<1.0$), Average ($1.0 - 2.2$), High ($>2.2$)\n\n"
            "### Academic Reference Guidelines & Benchmark Studies\n"
            "The standard indices and interpretation labels used in this module are based on the following established "
            "studies in second language acquisition (SLA), learner corpus linguistics, and vocabulary profiling:\n\n"
            "- **Lexical Complexity Analyzer (LCA)**:\n"
            "  * Lu, X. (2010). Automatic analysis of lexical complexity in second language writing. *International Journal of Corpus Linguistics*, 15(4), 474-496. [Lu's LCA Website](https://sites.psu.edu/xxl13/lca/)\n"
            "- **Measure of Textual Lexical Diversity (MTLD)**:\n"
            "  * McCarthy, P. M., & Jarvis, S. (2010). MTLD, D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. *Behavior Research Methods*, 42(2), 381-392.\n"
            "- **Type-Token Ratio & Standard Variations (RTTR, CTTR, Herdan, Uber)**:\n"
            "  * Carroll, J. B. (1964). *Language and thought*. Englewood Cliffs, NJ: Prentice-Hall. (Carroll's CTTR index)\n"
            "  * Guiraud, P. (1954). *Les caractères statistiques du vocabulaire*. Paris: Presses Universitaires de France. (Guiraud's RTTR index)\n"
            "  * Herdan, G. (1960). *Type-token mathematics*. The Hague: Mouton. (Herdan's LogTTR C-index)\n"
            "  * Dugast, D. (1979). *Vocabulaire et stylistique*. Paris: Slatkine. (Uber Index derivation)\n"
            "- **Lexical Sophistication & Vocabulary Profiling**:\n"
            "  * Laufer, B., & Nation, P. (1995). Vocabulary size and use: Lexical richness in L2 written production. *Applied Linguistics*, 16(3), 307-322. (Lexical Frequency Profile concept)\n"
            "- **Lexical Density & Register Variation**:\n"
            "  * Biber, D., Johansson, S., Leech, G., Conrad, S., & Finegan, E. (1999). *Longman Grammar of Spoken and Written English*. Longman. (Density benchmarks in written vs. spoken registers)\n"
        )




