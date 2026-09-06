import streamlit as st
import pandas as pd
import os
import altair as alt
import re

def natural_sort_key(s):
    """Sort strings containing numbers naturally (e.g. 'Sublist 2' < 'Sublist 10')"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
from ui_streamlit.state_manager import get_state, set_state
from ui_streamlit.utils import notify_timing
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.preprocessing.xml_parser import apply_xml_restrictions, get_xml_attribute_columns
from core.modules.word_profiler import load_wordlist, run_word_profiler_analysis, load_wordlist_from_file_object
from core.io_utils import df_to_excel_bytes
from core.ai_service import interpret_results_llm

def render_word_profiler_view():
    st.header("Word Profiler")
    
    corpus_path = get_state('current_corpus_path')
    corpus_name = get_state('current_corpus_name', 'Corpus')
    
    if not corpus_path:
        st.warning("Please load a corpus first.")
        return

    # Guidelines Layout using shared component
    from ui_streamlit.components.guidelines import render_guidelines
    col_main = render_guidelines("Word Profiler")

    with col_main:


        st.markdown("Analyze your corpus coverage using one or more wordlists.")

        # 1. Configuration
        with st.expander("Analysis Settings", expanded=True):
            basis = st.radio("Analysis Basis", ["Whole Corpus", "By Filename", "By Metadata"], horizontal=True, key="wp_basis")

            metadata_col = None
            if basis == "By Metadata":
                import duckdb
                con = duckdb.connect(corpus_path, read_only=True)
                attr_cols = get_xml_attribute_columns(con)
                if attr_cols:
                    selected_cols = st.multiselect("Select Metadata Attribute(s)", attr_cols, default=[attr_cols[0]] if attr_cols else [], key="wp_metadata_cols")
                    
                    # For each selected column, show available values with checklist (defaulting to all checked)
                    metadata_col = []
                    filtered_xml_filters = {}
                    
                    for col in selected_cols:
                        st.markdown(f"**Filter values for: {col}**")
                        # Fetch distinct values
                        try:
                            res = con.execute(f'SELECT DISTINCT "{col}" FROM corpus WHERE "{col}" IS NOT NULL ORDER BY "{col}"').fetchall()
                            unique_vals = [str(r[0]).strip() for r in res if str(r[0]).strip() and str(r[0]).lower() != 'nan']
                        except:
                            unique_vals = []
                            
                        if unique_vals:
                            # Use streamlit multiselect defaulting to all values
                            chosen_vals = st.multiselect(f"Select values for {col}", options=unique_vals, default=unique_vals, key=f"wp_filter_vals_{col}")
                            if chosen_vals:
                                filtered_xml_filters[col] = {'type': 'list', 'values': chosen_vals}
                                metadata_col.append(col)
                        else:
                            metadata_col.append(col)
                            
                    st.session_state['wp_metadata_value_filters'] = filtered_xml_filters
                else:
                    st.warning("No metadata attributes found in this corpus.")
                    metadata_col = None
                con.close()

        # 2. Wordlist Selection
        with st.expander("Wordlist Selection", expanded=True):
            wordlist_source = st.radio("Source", ["Existing Wordlist", "Upload Your Own"], horizontal=True, key="wp_wl_source")

            selected_wordlists = {} # name: wordlist_dict
            if wordlist_source == "Existing Wordlist":
                from core.modules.overview import get_corpus_language
                corpus_lang = get_corpus_language(corpus_path)
                st.caption(f"Detected corpus language: **{corpus_lang}**")
                
                base_wl_dir = "wordlist"
                if not os.path.exists(base_wl_dir) and os.path.exists(os.path.join("..", "wordlist")):
                    base_wl_dir = os.path.join("..", "wordlist")
                
                lang_map = {
                    "en": "english", "id": "indonesian", "ar": "arabic", 
                    "jp": "japanese", "ch": "chinese", "ko": "korean", 
                    "lo": "limola", "hi": "hindi", "jv": "javanese"
                }
                mapped_lang = lang_map.get(corpus_lang.lower(), corpus_lang.lower())
                
                wl_dir = os.path.join(base_wl_dir, mapped_lang)
                available_lists = []
                if os.path.exists(wl_dir):
                    for file in os.listdir(wl_dir):
                        full_path = os.path.join(wl_dir, file)
                        if os.path.isfile(full_path) and file.endswith((".txt", ".csv", ".xlsx", ".xls")):
                            if file.endswith("_stats.csv"):
                                continue
                            available_lists.append(file)
 
                if available_lists:
                    st.write("**Choose Wordlist(s):**")
                    chosen_lists = []
                    for chosen in sorted(available_lists):
                        if st.checkbox(chosen, key=f"wp_wl_chk_{chosen}"):
                            chosen_lists.append(chosen)
                    for chosen in chosen_lists:
                        full_path = os.path.join(wl_dir, chosen)
                        with open(full_path, 'rb') as f:
                            selected_wordlists[chosen] = load_wordlist_from_file_object(f, chosen)
                else:
                    st.info(f"No wordlists found in the `wordlist/{mapped_lang}/` directory.")
            else:
                uploaded_files = st.file_uploader(
                    "Upload Wordlist(s) (.txt, .csv, .xlsx, .xls)", 
                    type=["txt", "csv", "xlsx", "xls"], 
                    accept_multiple_files=True
                )
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        selected_wordlists[uploaded_file.name] = load_wordlist_from_file_object(uploaded_file, uploaded_file.name)

        # 3. Filtering
        xml_filters = render_xml_restriction_filters(corpus_path, "word_profiler", corpus_name=corpus_name)
        
        # Combine global filters with the metadata filters specified in settings
        combined_filters = {}
        if xml_filters:
            combined_filters.update(xml_filters)
        
        wp_meta_filters = st.session_state.get('wp_metadata_value_filters')
        if wp_meta_filters:
            combined_filters.update(wp_meta_filters)
            
        xml_where, xml_params = apply_xml_restrictions(combined_filters)

        # 4. Run Analysis
        if st.button("Run Analysis", type="primary"):
            if not selected_wordlists:
                st.error("Please select or upload at least one wordlist.")
            else:
                with st.spinner("Analyzing..."):
                    all_results = {}
                    for wl_name, wl_dict in selected_wordlists.items():
                        res_tuple = notify_timing(f"Word Profiler analysis for '{wl_name}' completed")(run_word_profiler_analysis)(
                            db_path=corpus_path,
                            wordlist=wl_dict,
                            basis=basis,
                            metadata_col=metadata_col,
                            xml_where_clause=xml_where,
                            xml_params=xml_params,
                            return_detailed=True
                        )
                        if isinstance(res_tuple, tuple) and len(res_tuple) == 2:
                            df_res, detailed = res_tuple
                        else:
                            df_res = res_tuple
                            detailed = None
                        all_results[wl_name] = {'summary': df_res, 'detailed': detailed}
                    set_state('last_wp_results_multiple', all_results)

        # 5. Results
        all_results = get_state('last_wp_results_multiple')
        if all_results is not None:
            if not all_results:
                st.info("No results found.")
            else:
                st.subheader("Analysis Results")
                for wl_name, res_data in all_results.items():
                    if isinstance(res_data, dict):
                        df_results = res_data['summary']
                        detailed = res_data.get('detailed')
                    else:
                        df_results = res_data
                        detailed = None

                    with st.expander(f"📊 Results for: {wl_name}", expanded=True):
                        if df_results.empty:
                            st.info(f"No results for {wl_name}")
                            continue

                        st.dataframe(df_results, use_container_width=True)

                        # --- AI Interpretation (High Visibility) ---
                        st.markdown("#### 🧠 AI Interpretation")
                        if st.button("🤖 Interpret Results with AI", key=f"btn_wp_ai_{wl_name}", type="primary"):
                            with st.spinner("AI is analyzing Word Profiler results..."):
                                description = f"Word Profiler coverage analysis for wordlist '{wl_name}' on corpus '{corpus_name}' (Analysis basis: {basis})."
                                wp_data_text = f"=== WORD PROFILER VOCABULARY COVERAGE TABLE ===\nWordlist: '{wl_name}'\nCorpus: '{corpus_name}'\nBasis: {basis}\n\n" + df_results.to_string(index=False)
                                response, error = interpret_results_llm(
                                    target_word=wl_name,
                                    analysis_type="Word Profiler Vocabulary Coverage Analysis",
                                    data_description=description,
                                    data=wp_data_text,
                                    ai_provider=get_state('ai_provider'),
                                    gemini_api_key=get_state('gemini_api_key'),
                                    ollama_url=get_state('ollama_url'),
                                    ollama_model=get_state('ai_model')
                                )
                                if error:
                                    st.error(error)
                                else:
                                    set_state(f'wp_ai_res_{wl_name}', response)

                        ai_res = get_state(f'wp_ai_res_{wl_name}')
                        if ai_res:
                            st.markdown(ai_res)

                        st.markdown("---")

                        # --- Visualization ---
                        st.markdown("#### 📊 Visualization")
                        render_word_profiler_chart(df_results, wl_name)

                        # Download Button for this specific wordlist
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button(
                                label=f"Download {wl_name} Results (Excel)",
                                data=df_to_excel_bytes(df_results),
                                file_name=f"word_profiler_{corpus_name}_{wl_name.split('.')[0]}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"dl_{wl_name}"
                            )
                        with col_dl2:
                            if detailed and 'zip_bytes' in detailed:
                                st.download_button(
                                    label="Download Detailed Domain Reports (ZIP)",
                                    data=detailed['zip_bytes'],
                                    file_name=f"word_profiler_detailed_{corpus_name}_{wl_name.split('.')[0]}.zip",
                                    mime="application/zip",
                                    key=f"dl_zip_{wl_name}"
                                )

                        # Render top 50 visualizations
                        if detailed and 'top_10_lists' in detailed:
                            st.markdown("#### 🔝 Top 50 Most Frequent Words by Level")
                            st.info("💡 Note: Only the top 50 words are shown here. You can download the full frequency results for all levels and domains from the ZIP file above.")
                            segments = list(detailed['top_10_lists'].keys())
                            if len(segments) > 1:
                                selected_seg = st.radio(
                                    "Select Domain/Segment for Top 50 lists:",
                                    options=segments,
                                    horizontal=True,
                                    key=f"wp_top50_seg_{wl_name}"
                                )
                            else:
                                selected_seg = segments[0] if segments else None
                            
                            if selected_seg:
                                cat_data = detailed['top_10_lists'][selected_seg]
                                cat_tabs = st.tabs(list(cat_data.keys()))
                                for cat_tab, cat_name in zip(cat_tabs, cat_data.keys()):
                                    with cat_tab:
                                        df_top = cat_data[cat_name]
                                        if not df_top.empty:
                                            df_top_display = df_top.copy()
                                            df_top_display.index = range(1, len(df_top_display) + 1)
                                            col1, col2 = st.columns([1, 2])
                                            with col1:
                                                st.dataframe(df_top_display, use_container_width=True)
                                            with col2:
                                                chart = alt.Chart(df_top).mark_bar().encode(
                                                    x=alt.X('Raw Freq:Q', title='Raw Frequency'),
                                                    y=alt.Y('Word:N', sort='-x', title='Word'),
                                                    color=alt.value('#1f77b4')
                                                ).properties(height=400) # taller for 50 words
                                                st.altair_chart(chart, use_container_width=True)
                                        else:
                                            st.info(f"No words found in this category for {selected_seg}.")

                        # Summary Metrics
                        if basis == "Whole Corpus" and not df_results.empty:
                            st.markdown("#### Coverage Summary")
                            # Filter out 'Segment' and 'Total Tokens'
                            res_cols = [c for c in df_results.columns if c not in ['Segment', 'Total Tokens']]
                            cols = st.columns(min(len(res_cols) // 2, 6)) # Max 6 metrics per row
                            for i in range(0, len(res_cols), 2):
                                cat_name = str(res_cols[i]).replace(" Freq", "")
                                freq = df_results.iloc[0][res_cols[i]]
                                perc = df_results.iloc[0][res_cols[i+1]]
                                with cols[(i // 2) % len(cols)]:
                                    st.metric(cat_name, f"{freq:,}", f"{perc}%")

def render_word_profiler_chart(df, wl_name):
    """
    Renders an interactive Altair bar chart for Word Profiler results with multi-level filtering and dynamic percentage basis.
    """
    if df.empty:
        return

    # Extract all categories present in df
    cat_names = [c.replace(" %", "") for c in df.columns if c.endswith(" %")]
    cats_sorted = sorted(cat_names, key=natural_sort_key)
    if 'OOV' in cats_sorted:
        cats_sorted.remove('OOV')
        cats_sorted.append('OOV')

    c_v1, c_v2 = st.columns([2, 1])
    with c_v1:
        selected_cats = st.multiselect(
            "Select Categories / Levels to Display in Chart:",
            options=cats_sorted,
            default=cats_sorted,
            key=f"wp_cats_multiselect_{wl_name}"
        )
    with c_v2:
        pct_basis = st.radio(
            "Percentage Calculation Basis",
            ["Selected Levels Only (Sum = 100%)", "% of Total Corpus Tokens"],
            horizontal=True,
            key=f"wp_pct_basis_{wl_name}",
            help="'Selected Levels Only' rescales percentages so the selected levels sum to 100%, allowing direct comparison (e.g. Level 4 vs Level 5). '% of Total Corpus Tokens' preserves original total coverage."
        )

    if not selected_cats:
        st.info("Please select at least one category to display in the chart.")
        return

    rescale_selected = (pct_basis == "Selected Levels Only (Sum = 100%)")
    chart_data = []

    for _, row in df.iterrows():
        segment = row['Segment']

        if rescale_selected:
            plotted_freqs = [row[f"{c} Freq"] for c in selected_cats if f"{c} Freq" in row]
            segment_total = sum(plotted_freqs)
        else:
            segment_total = row['Total Tokens'] if 'Total Tokens' in row else 0

        for cat_name in selected_cats:
            freq_col = f"{cat_name} Freq"
            frequency = row[freq_col] if freq_col in row else 0

            percentage = (frequency / segment_total * 100) if segment_total > 0 else 0

            chart_data.append({
                'Segment': segment,
                'Category': cat_name,
                'Percentage': round(percentage, 2),
                'Frequency': int(frequency)
            })

    plot_df = pd.DataFrame(chart_data)
    cats_plotted = [c for c in cats_sorted if c in selected_cats]

    if len(df) == 1:
        # Whole Corpus - Simple Bar Chart (Horizontal)
        chart = alt.Chart(plot_df).mark_bar(color='#00ADB5').encode(
            x=alt.X('Percentage:Q', title='Percentage (%)', scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('Category:N', title='Category', sort=cats_plotted),
            tooltip=['Category', 'Percentage', 'Frequency']
        ).properties(height=max(200, len(cats_plotted) * 35))
    else:
        # Multiple Segments - Stacked Bar Chart (Horizontal Stacked)
        num_segments = len(df)
        dynamic_height = max(150, num_segments * 30)

        chart = alt.Chart(plot_df).mark_bar().encode(
            x=alt.X('Percentage:Q', title='Percentage (%)', stack="normalize" if rescale_selected else None),
            y=alt.Y('Segment:N', title='Segment', sort=None),
            color=alt.Color('Category:N', sort=cats_plotted, scale=alt.Scale(scheme='category20')),
            tooltip=['Segment', 'Category', 'Percentage', 'Frequency']
        ).properties(height=dynamic_height)

    st.altair_chart(chart, use_container_width=True)


