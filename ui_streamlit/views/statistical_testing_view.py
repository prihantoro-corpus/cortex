import streamlit as st
import pandas as pd
from ui_streamlit.state_manager import get_state, set_state
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.preprocessing.xml_parser import apply_xml_restrictions, get_xml_attribute_columns

# Force reload of statistical_testing module to ensure latest changes are picked up
import importlib
import core.modules.statistical_testing
importlib.reload(core.modules.statistical_testing) 

from core.modules.statistical_testing import (
    compare_groups_by_word, preview_query_matches,
    get_document_frequency_vector, get_document_metadata_vector,
    get_document_metric_vector, calculate_correlation,
    compare_groups_by_word, preview_query_matches,
    get_document_frequency_vector, get_document_metadata_vector,
    get_document_metric_vector, calculate_correlation,
    get_feature_matrix, perform_clustering, generate_clustering_interpretation,
    perform_correspondence_analysis, generate_ca_interpretation, perform_burrows_delta
)
from core.io_utils import df_to_excel_bytes
import duckdb
from scipy import stats
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff


def render_statistical_testing_view():
    """
    Main view for Statistical Testing module.
    Phase 1: Group Comparison.
    Phase 2: Correlation (Planned).
    """
    st.header("📊 Statistical Testing")
    
    corpus_path = get_state('current_corpus_path')
    corpus_name = get_state('current_corpus_name', 'Corpus')
    
    if not corpus_path:
        st.warning("Please load a corpus first.")
        return
    
    st.markdown("""
    Compare continuous variables (word frequencies) across categorical groups, or find correlations between variables.
    """)
    
    # --- SHARED SLICING/GROUPING ---
    # Moved outside of tabs so it's available for all analysis types
    with st.expander("📝 **Corpus Config & Slicing**", expanded=False):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("### Slicing")
            # Reuse grouping logic
            con = duckdb.connect(corpus_path, read_only=True)
            cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
            con.close()
            
            grouping_opts = ['filename']
            if 'doc_id' in cols: grouping_opts.insert(0, 'doc_id')
            
            grouping_key = st.selectbox(
                "Slice Corpus By", 
                options=grouping_opts + sorted([c for c in cols if c not in grouping_opts and c not in ['token', 'pos', 'lemma', 'sent_id', '_token_low', 'id', 'filename']]),
                key="shared_grouping"
            )
            
            # Get unique values and count
            con = duckdb.connect(corpus_path, read_only=True)
            try:
                unique_vals = [r[0] for r in con.execute(
                    f"SELECT DISTINCT {grouping_key} FROM corpus WHERE {grouping_key} IS NOT NULL ORDER BY {grouping_key}"
                ).fetchall()]
                unique_vals = [str(v) for v in unique_vals if str(v).strip()]
                segment_count = len(unique_vals)
            finally:
                con.close()
            
            st.info(f"found **{segment_count}** segments.")

        with col_s2:
            st.markdown("### Selection")
            
            # Selection Mode
            select_mode = st.radio(
                "Selection Mode",
                ["All Segments", "Manual Selection"],
                key="shared_select_mode",
                horizontal=True
            )
            
            selected_segments = None
            slice_ready = True
            slice_error = None
            
            if select_mode == "All Segments":
                if segment_count > 100:
                    st.warning(f"⚠️ **{segment_count} segments** is high for visualization. Consider filtering.")
            else:
                selected_segments = st.multiselect(
                    "Select segments to analyze",
                    options=unique_vals,
                    default=unique_vals[:5] if len(unique_vals) >= 5 else unique_vals,
                    key="shared_manual_select"
                )
                if not selected_segments:
                    slice_error = "Please select at least 2 segments."
                    slice_ready = False
                elif len(selected_segments) < 2:
                    slice_error = "Requires at least 2 segments."
                    slice_ready = False

    # Tabs for different analysis modes
    tab_group, tab_corr, tab_cluster, tab_ca, tab_attrib = st.tabs([
        "📊 Group Difference", 
        "📈 Correlation", 
        "🌳 Clustering",
        "🗺️ Correspondence Analysis",
        "🕵️ Authorship Attribution"
    ])

    # ---------------- GROUP DIFFERENCE TAB ----------------
    with tab_group:
        with st.expander("💡 **Method & Transparency: Group Difference**", expanded=False):
            st.markdown("""
            **Goal:** Compare lexical usage between two categorical groups (e.g., Male vs. Female, Sci-Fi vs. Horror).
            
            **Data Used:** 
            - Frequency of tokens/queries across the specified sub-corpora.
            
            **Statistical Tests:**
            - **Chi-square Test:** Used for comparing proportions (frequencies relative to total tokens).
            - **Effect Size (Cohen's h):** Measures the magnitude of the difference (Small, Medium, or Large).
            - Includes Bonferroni or FDR (Benjamini-Hochberg) correction for multiple comparisons.
            """)
    
    # ---------------- CONNECTION TAB ----------------
    # ---------------- CORRELATION TAB ----------------
    with tab_corr:
        with st.expander("💡 **Method & Transparency: Correlation**", expanded=False):
            st.markdown("""
            **Goal:** Analyze the relationship between two variables (frequencies, metadata, or metrics) across documents.
            
            **Data Used:** 
            - Numeric metrics (like TTR or token count)
            - Lexical frequencies (count of specific words/queries)
            - Numeric XML metadata
            
            **Statistical Tests:**
            - **Pearson (Linear):** Measures linear relationship strength.
            - **Spearman (Rank):** Measures monotonic relationship, better for non-linear trends.
            - Outputs: Coefficient (r/rho), P-value (significance), and Sample Size (n).
            """)
        st.caption("**Goal:** Analyze relationship between two variables across documents (Pearson/Spearman).")
        st.caption("Each point in the scatter plot represents one document.")
        
        # --- Grouping Key Selection ---
        con = duckdb.connect(corpus_path, read_only=True)
        cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
        con.close()
        
        grouping_opts = ['filename']
        if 'doc_id' in cols: grouping_opts.insert(0, 'doc_id')
        elif 'id' in cols and 'id' not in ['id', 'sent_id']: grouping_opts.append('id') # careful with internal id
        
        # Identify probable candidates (high cardinality, not token-level unique)
        # Actually simplest is to just let user choose or default to doc_id if present
        
        grouping_key = st.selectbox(
            "Document Identifier (Grouping Key)", 
            grouping_opts + sorted([c for c in cols if c not in grouping_opts and c not in ['token', 'pos', 'lemma', 'sent_id', '_token_low', 'id', 'filename']]),
            index=0,
            help="What represents a 'Document'? For single-file corpora (like KOSLAT), use 'doc_id' or similar."
        )
        
        st.markdown("---")
        
        col_x, col_y = st.columns(2)
        
        # --- Variable X ---
        with col_x:
            st.markdown("### Variable X (Horizontal)")
            type_x = st.selectbox("Type", ["Word Frequency", "Metadata (Numeric)", "Text Metric"], key="corr_type_x")
            
            x_config = {}
            if type_x == "Word Frequency":
                x_config['query'] = st.text_input("Query X", value="good", key="corr_q_x", help="Any CORTEX query")
            elif type_x == "Metadata (Numeric)":
                con = duckdb.connect(corpus_path, read_only=True)
                attr_cols = get_xml_attribute_columns(con)
                con.close()
                x_config['attr'] = st.selectbox("Attribute X", attr_cols, key="corr_attr_x")
            elif type_x == "Text Metric":
                x_config['metric'] = st.selectbox("Metric X", ["ttr", "token_count", "type_count"], key="corr_metric_x")

        # --- Variable Y ---
        with col_y:
            st.markdown("### Variable Y (Vertical)")
            type_y = st.selectbox("Type", ["Word Frequency", "Metadata (Numeric)", "Text Metric"], key="corr_type_y")
            
            y_config = {}
            if type_y == "Word Frequency":
                y_config['query'] = st.text_input("Query Y", value="bad", key="corr_q_y", help="Any CORTEX query")
            elif type_y == "Metadata (Numeric)":
                con = duckdb.connect(corpus_path, read_only=True)
                attr_cols = get_xml_attribute_columns(con)
                con.close()
                y_config['attr'] = st.selectbox("Attribute Y", attr_cols, key="corr_attr_y")
            elif type_y == "Text Metric":
                y_config['metric'] = st.selectbox("Metric Y", ["ttr", "token_count", "type_count"], key="corr_metric_y")
         
        # --- Run Correlation ---
        st.markdown("---")
        corr_method = st.radio("Correlation Method", ["Pearson (Linear)", "Spearman (Rank)"], horizontal=True)
        method_code = 'pearson' if "Pearson" in corr_method else 'spearman'
        
        if st.button("🚀 Run Correlation", type="primary"):
            with st.spinner("Calculating correlation..."):
                try:
                    df_x = pd.DataFrame()
                    df_y = pd.DataFrame()
                    
                    # Fetch X
                    if type_x == "Word Frequency":
                        if not x_config['query']: st.error("Enter Query X"); st.stop()
                        df_x = get_document_frequency_vector(corpus_path, x_config['query'], group_by=grouping_key)
                    elif type_x == "Metadata (Numeric)":
                        df_x = get_document_metadata_vector(corpus_path, x_config['attr'], group_by=grouping_key)
                    elif type_x == "Text Metric":
                        df_x = get_document_metric_vector(corpus_path, x_config['metric'], group_by=grouping_key)
                        
                    # Fetch Y
                    if type_y == "Word Frequency":
                        if not y_config['query']: st.error("Enter Query Y"); st.stop()
                        df_y = get_document_frequency_vector(corpus_path, y_config['query'], group_by=grouping_key)
                    elif type_y == "Metadata (Numeric)":
                        df_y = get_document_metadata_vector(corpus_path, y_config['attr'], group_by=grouping_key)
                    elif type_y == "Text Metric":
                        df_y = get_document_metric_vector(corpus_path, y_config['metric'], group_by=grouping_key)
                        
                    # Calculate
                    res = calculate_correlation(df_x, df_y, method=method_code)
                    
                    if 'error' in res:
                        st.error(res['error'])
                    else:
                        r_val = res['r']
                        p_val = res['p_value']
                        n_docs = res['n']
                        
                        # --- Interpretation Logic ---
                        abs_r = abs(r_val)
                        strength = "Negligible"
                        if abs_r >= 0.8: strength = "Very Strong"
                        elif abs_r >= 0.6: strength = "Strong"
                        elif abs_r >= 0.4: strength = "Moderate"
                        elif abs_r >= 0.2: strength = "Weak"
                        
                        direction = "Positive" if r_val > 0 else "Negative"
                        sig_text = "statistically significant" if p_val < 0.05 else "not statistically significant"
                        
                        x_label = x_config.get('query') or x_config.get('attr') or x_config.get('metric')
                        y_label = y_config.get('query') or y_config.get('attr') or y_config.get('metric')
                        
                        # Generate Report
                        st.markdown("---")
                        st.subheader("📋 Statistical Report")
                        
                        with st.container(border=True):
                            report = f"""
                            **1. Methodology**
                            - **Test Used:** {corr_method} Correlation
                            - **Variables Analyzed:** 
                                - X: **{x_label}** ({type_x})
                                - Y: **{y_label}** ({type_y})
                            - **Sample Size:** {n_docs} documents
                            
                            **2. Results**
                            - **Correlation Coefficient ({'r' if method_code=='pearson' else 'rho'}):** {r_val:.3f}
                            - **P-Value:** {p_val:.4f} ({sig_text})
                            
                            **3. Interpretation**
                            Processing the data revealed a **{strength.lower()} {direction.lower()} correlation** between the two variables.
                            """
                            
                            if p_val < 0.05:
                                report += f"\nSince p < 0.05, this relationship is **statistically significant**, meaning it is unlikely to be due to chance."
                                if abs_r > 0.5:
                                    report += f"\n\n**Conclusion:** As **{x_label}** increases, **{y_label}** tends to {'increase' if r_val > 0 else 'decrease'} significantly."
                            else:
                                report += f"\nSince p >= 0.05, the relationship is **not significant**. Any observed trend might be random noise."
                                
                            st.markdown(report)
                        
                        st.markdown("---")
                        
                        # Plot
                        df_plot = res['df_plot']
                        col_map = {df_plot.columns[1]: f"X ({x_label})", df_plot.columns[2]: f"Y ({y_label})"}
                        df_plot = df_plot.rename(columns=col_map)
                        
                        # Create Scatter Plot
                        fig = px.scatter(
                            df_plot, 
                            x=f"X ({x_label})", 
                            y=f"Y ({y_label})", 
                            hover_data=['group_id'],
                            title=f"Correlation: {x_label} vs {y_label}"
                        )
                        
                        # Add Trendline manually to avoid statsmodels dependency
                        if method_code == 'pearson':
                            # Safe dropna for calculation
                            clean_data = df_plot.dropna(subset=[f"X ({x_label})", f"Y ({y_label})"])
                            if len(clean_data) > 1:
                                x_vals = clean_data[f"X ({x_label})"]
                                y_vals = clean_data[f"Y ({y_label})"]
                                slope, intercept, r_val, p_val, std_err = stats.linregress(x_vals, y_vals)
                                line_x = np.array([x_vals.min(), x_vals.max()])
                                line_y = slope * line_x + intercept
                                
                                import plotly.graph_objects as go
                                fig.add_trace(go.Scatter(
                                    x=line_x, 
                                    y=line_y, 
                                    mode='lines', 
                                    name='Trendline (OLS)',
                                    line=dict(color='red', dash='dash')
                                ))

                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Data Table
                        with st.expander("View Data"):
                            st.dataframe(df_plot)
                            
                except Exception as e:
                    st.error(f"Correlation Failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # ---------------- CLUSTERING TAB ----------------
    with tab_cluster:
        with st.expander("💡 **Method & Transparency: Clustering**", expanded=False):
            st.markdown("""
            **Goal:** Discover natural groupings of documents or segments based on similarities in word usage.
            
            **Data Used:** 
            - Word frequency matrix of the Most Frequent N-grams (Features).
            - Relative frequencies or standardized Z-scores.
            
            **Statistical Method:**
            - **Hierarchical Clustering (Ward's Method):** Minimizes variance within clusters as they are merged.
            - **Distance Measures:** Euclidean (straight line) or Manhattan (city block) distance between lexical profiles.
            """)
        st.caption("**Goal:** Group documents or segments based on similarity of word usage (Ward's Method).")
        
        # --- Configuration ---
        with st.expander("📝 **Clustering Features**", expanded=True):
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.markdown("### Status")
                if not slice_ready:
                    st.error(slice_error)
                    cluster_ready = False
                else:
                    st.success(f"Ready to cluster **{len(selected_segments) if selected_segments else segment_count}** segments.")
                    cluster_ready = True 
                
            with col_c2:
                st.markdown("### Features & Distance")
                dist_metric = st.radio(
                    "Distance Measure", 
                    ["Euclidean (Standard)", "Manhattan (City Block)"],
                    key="cluster_dist"
                )
                metric_code = 'cityblock' if "Manhattan" in dist_metric else 'euclidean'
                
                top_n = st.slider("Number of Top Words (Features)", 10, 200, 50, help="Clustering is based on the frequency of these most common words.")
                
                ngram_options = {1: "1-gram (Word)", 2: "2-gram (Bigram)", 3: "3-gram (Trigram)"}
                ngram_help = "Cluster based on words, bigrams, or trigrams. You can select multiple to combine features (e.g., Words + Bigrams)."
                
                ngram_selected = st.multiselect(
                    "Features to Include",
                    options=[1, 2, 3],
                    format_func=lambda x: ngram_options[x],
                    default=[1],
                    help=ngram_help
                )
                
                if not ngram_selected:
                    ngram_selected = [1] # Fallback
                
                z_score = st.checkbox(
                    "Standardize Features (Z-Score)", 
                    value=True, 
                    help="Normalize word frequencies to Z-scores. Essential for Burrows' Delta style analysis (Stylo R default)."
                )

        # --- Run Button ---
        if st.button("🚀 Run Clustering", type="primary", disabled=not cluster_ready):
            with st.spinner("Clustering segments..."):
                try:
                    # 1. Get Feature Matrix
                    df_matrix, top_words = get_feature_matrix(
                        corpus_path, 
                        group_by=grouping_key,
                        top_n_features=top_n,
                        selected_groups=selected_segments,
                        ngram_sizes=ngram_selected
                    )
                    
                    if df_matrix.empty:
                        st.error("No data found for clustering. Check if the corpus is empty or filters are too strict.")
                    else:
                        st.session_state['last_cluster_results'] = {
                            'matrix': df_matrix,
                            'top_words': top_words,
                            'metric': metric_code,
                            'grouping': grouping_key,
                            'z_score': z_score
                        }
                        
                        # 2. Perform Clustering (to get linkage Z)
                        res = perform_clustering(
                            df_matrix, 
                            distance_metric=metric_code, 
                            method='ward',
                            use_z_scores=z_score
                        )
                        
                        if 'error' in res:
                            st.error(res['error'])
                        else:
                            st.session_state['last_cluster_results'].update(res)
                            st.success(f"✅ clustered {len(df_matrix)} segments based on {len(top_words)} words.")
                            
                except Exception as e:
                    st.error(f"Clustering failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

        # --- Display Results ---
        if 'last_cluster_results' in st.session_state:
            res = st.session_state['last_cluster_results']
            matrix = res['matrix']
            Z = res['linkage']
            labels = res['labels']
            
            st.markdown("---")
            
            # Layout: Dendrogram on top, Interpretation below
            
            # Dendrogram
            st.subheader("🌳 Dendrogram")
            try:
                # Use plotly figure factory, but force it to use our pre-calculated Z?
                # create_dendrogram calculates Z internally. 
                # To ensure consistency specifically with our Z (which used specific metric logic),
                # we pass a lambda that just returns our Z.
                # NOTE: create_dendrogram expects data X as first arg.
                
                import plotly.figure_factory as ff
                
                # We need to wrap Z in a lambda because linkagefun receives X
                fig = ff.create_dendrogram(
                    matrix.values,
                    orientation='bottom',
                    labels=labels,
                    linkagefun=lambda x: Z
                )
                
                fig.update_layout(
                    title=f"Hierarchical Clustering (Ward's Method, {res['metric'].capitalize()})",
                    xaxis_title="Segments",
                    yaxis_title="Distance",
                    width=800,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error plotting dendrogram: {str(e)}")
            
            # Interpretation Box
            st.markdown("### 💡 Interpretation")
            with st.container(border=True):
                interpretation = generate_clustering_interpretation(
                    res['norm_matrix'], 
                    Z, 
                    labels,
                    z_matrix=res.get('z_matrix')
                )
                st.markdown(interpretation)
                
    with tab_ca:
        with st.expander("💡 **Method & Transparency: Correspondence Analysis**", expanded=False):
            st.markdown("""
            **Goal:** Map the relationship between multiple documents and multiple words in a compressed 2D space.
            
            **Data Used:** 
            - Contingency table of documents (rows) vs. top words/N-grams (columns).
            
            **Statistical Method:**
            - **Singular Value Decomposition (SVD):** Applied to Chi-square residuals to identify dimensions that explain the most lexical variance.
            - Visualizes which words are most associated with which documents.
            """)
        st.caption("**Goal:** Visualize relationship between documents (Rows) and words (Columns) in a 2D map.")
        col_ca1, col_ca2 = st.columns([1, 2])
        
        with col_ca1:
            st.markdown("### Configuration")
            
            # Reusing corpus path and grouping key from main scope
            ca_top_n = st.slider("Number of Top Features", 10, 100, 30, key="ca_top_n", 
                               help="CA works best with 20-50 high-frequency words.")
            
            ca_ngrams = st.multiselect(
                "N-gram Features",
                options=[1, 2, 3],
                format_func=lambda x: {1: "Words", 2: "Bigrams", 3: "Trigrams"}[x],
                default=[1],
                key="ca_ngrams"
            )
            if not ca_ngrams: ca_ngrams = [1]

            ca_ready = False
            if not slice_ready:
                 st.error(slice_error)
            else:
                 ca_ready = True
                 
            if st.button("🗺️ Run Correspondence Analysis", disabled=not ca_ready, type="primary"):
                with st.spinner("Computing SVD..."):
                    try:
                        # 1. Reuse existing get_feature_matrix backend
                        # CA requires raw frequencies, NO z-scores.
                        ca_matrix, ca_features = get_feature_matrix(
                            corpus_path,
                            group_by=grouping_key,
                            top_n_features=ca_top_n,
                            selected_groups=selected_segments,
                            ngram_sizes=ca_ngrams
                        )
                        
                        if ca_matrix.empty:
                            st.error("Matrix is empty. Try flexible filters.")
                        else:
                            # 2. Run CA
                            ca_res = perform_correspondence_analysis(ca_matrix)
                            
                            if 'error' in ca_res:
                                st.error(ca_res['error'])
                            else:
                                # Add metadata to results
                                ca_res['top_words'] = ca_features
                                st.session_state['last_ca_results'] = ca_res
                                st.success("Analysis Complete.")
                                
                    except Exception as e:
                        st.error(f"CA Failed: {str(e)}")
        
        with col_ca2:
            if 'last_ca_results' in st.session_state:
                res = st.session_state['last_ca_results']
                rows = res['row_coords']
                cols = res['col_coords']
                inertia = res['explained_inertia']
                
                # Check dimensions
                if rows.shape[1] < 2:
                    st.warning("Not enough dimensions to plot (Need at least 2).")
                else:
                    # Prepare Data for Plotly
                    # We plot Dim 1 vs Dim 2
                    x_expl = inertia[0] * 100
                    y_expl = inertia[1] * 100
                    
                    # Row Points (Documents)
                    fig = px.scatter(
                        rows, x='Dim 1', y='Dim 2', 
                        text=rows.index,
                        title=f"Correspondence Analysis Biplot ({x_expl:.1f}% + {y_expl:.1f}%)"
                    )
                    fig.update_traces(marker=dict(size=12, color='blue', symbol='circle'), textposition='top center', name='Documents')
                    
                    # Column Points (Words) - Add as separate trace
                    fig.add_scatter(
                        x=cols['Dim 1'], 
                        y=cols['Dim 2'],
                        mode='markers+text',
                        text=cols.index,
                        textposition='top center',
                        marker=dict(size=8, color='red', symbol='triangle-up'),
                        name='Features'
                    )
                    
                    fig.update_layout(
                        xaxis_title=f"Dimension 1 ({x_expl:.1f}%)",
                        yaxis_title=f"Dimension 2 ({y_expl:.1f}%)",
                        height=600,
                        showlegend=True
                    )
                    
                    # Add Zero lines
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("Show Inertia (Variance Explained)"):
                        st.write(pd.DataFrame({
                            'Dimension': range(1, len(inertia)+1),
                            'Explained Variance': inertia,
                            'Cumulative': np.cumsum(inertia)
                        }))
                
                # Report Box
                if 'top_words' in res:
                    st.divider()
                    with st.container(border=True):
                         report = generate_ca_interpretation(
                             rows, 
                             cols, 
                             inertia, 
                             res['top_words'],
                             grouping_key,
                             ca_ngrams # Assuming these are available in scope. If re-rendered, they are.
                         )
                         st.markdown(report)
                         
    with tab_attrib:
        with st.expander("💡 **Method & Transparency: Authorship Attribution**", expanded=False):
            st.markdown("""
            **Goal:** Determine the most likely author of a 'Questioned' text by comparing it to 'Known' author profiles.
            
            **Data Used:** 
            - Most Frequent Words (MFW) of the whole corpus (Stylometric features).
            
            **Statistical Method:**
            - **Burrows' Delta:** Calculates the Manhattan distance between the Z-scores of the MFW in the Questioned text and the Known texts.
            - Smaller Delta values indicate higher stylistic similarity.
            """)
        # Selection State - Use Session State for stability
        if 'aa_known_texts' not in st.session_state: st.session_state['aa_known_texts'] = []
        if 'aa_questioned_texts' not in st.session_state: st.session_state['aa_questioned_texts'] = []
        
        known_texts = st.session_state['aa_known_texts']
        questioned_texts = st.session_state['aa_questioned_texts']
        
        st.caption("**Goal:** Attribute 'Questioned' texts to 'Known' candidate authors using Burrows' Delta (Stylometry).")
        
        col_aa1, col_aa2 = st.columns([1, 2])
        
        with col_aa1:
            st.markdown("### Configuration")
            
            # Step 0: Choose selection method
            selection_method = st.radio(
                "How to define Known/Questioned texts?",
                options=["By Filename", "By XML Metadata"],
                key="aa_selection_method",
                help="Choose how to distinguish Known texts from Questioned texts"
            )
            
            st.markdown("---")
            
            # Get metadata columns
            con = duckdb.connect(corpus_path, read_only=True)
            cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
            all_files = [r[0] for r in con.execute("SELECT DISTINCT filename FROM corpus ORDER BY filename").fetchall()]
            con.close()
            
            # Guess author column
            default_auth_idx = 0
            possible_auth = [c for c in cols if 'auth' in c.lower() or 'class' in c.lower() or 'category' in c.lower()]
            if possible_auth:
                default_auth_idx = cols.index(possible_auth[0])
            
            author_col = st.selectbox(
                "Author Label Column",
                options=cols,
                index=default_auth_idx,
                key="aa_auth_col",
                help="Column containing author names for Known texts"
            )

            aa_grouping_key = st.selectbox(
                "Document Identifier",
                options=cols,
                index=cols.index('filename') if 'filename' in cols else 0,
                key="aa_grouping_key_box",
                help="Unique identifier for each document/segment (e.g., 'doc_id' or 'filename')"
            )
            
            # Reset selection if identifier changed
            if "last_aa_grouping_key" not in st.session_state:
                st.session_state["last_aa_grouping_key"] = aa_grouping_key
            if st.session_state["last_aa_grouping_key"] != aa_grouping_key:
                st.session_state['aa_known_texts'] = []
                st.session_state['aa_questioned_texts'] = []
                st.session_state["last_aa_grouping_key"] = aa_grouping_key
                st.rerun()
            
            st.markdown("---")
            
            if selection_method == "By Filename":
                # Get unique values for identifier
                con = duckdb.connect(corpus_path, read_only=True)
                all_ids = [r[0] for r in con.execute(f"SELECT DISTINCT {aa_grouping_key} FROM corpus ORDER BY {aa_grouping_key}").fetchall()]
                con.close()

                known_texts = st.multiselect(
                    "Known Texts",
                    options=all_ids,
                    default=all_ids[:min(5, len(all_ids))],
                    key="aa_known_files_multi"
                )
                st.session_state['aa_known_texts'] = known_texts
                
                # Show author mapping
                if known_texts:
                    con = duckdb.connect(corpus_path, read_only=True)
                    author_map = con.execute(
                        f"SELECT {aa_grouping_key} as id, {author_col} FROM corpus WHERE {aa_grouping_key} IN ({','.join(['?']*len(known_texts))}) GROUP BY {aa_grouping_key}, {author_col}",
                        known_texts
                    ).fetchdf()
                    con.close()
                    
                    with st.expander("📋 Known Text Authors"):
                        st.dataframe(author_map, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown("#### Step 2: Select Questioned Texts")
                st.caption("Select files to attribute")
                
                available_questioned = [f for f in all_ids if f not in known_texts]
                questioned_texts = st.multiselect(
                    "Questioned Texts",
                    options=available_questioned,
                    key="aa_questioned_files_multi"
                )
                st.session_state['aa_questioned_texts'] = questioned_texts
                
            else:  # By XML Metadata
                st.markdown("#### Step 1: Define Known Texts (by Author)")
                st.caption(f"Select author values from '{author_col}' that represent Known texts")
                
                # Get unique author values
                con = duckdb.connect(corpus_path, read_only=True)
                all_authors = [r[0] for r in con.execute(
                    f"SELECT DISTINCT {author_col} FROM corpus WHERE {author_col} IS NOT NULL ORDER BY {author_col}"
                ).fetchall()]
                con.close()
                
                known_authors = st.multiselect(
                    f"Known Authors ({author_col})",
                    options=all_authors,
                    key="aa_known_authors",
                    help="Select author values that identify Known texts"
                )
                
                # Preview Known texts
                if known_authors:
                    con = duckdb.connect(corpus_path, read_only=True)
                    known_preview = con.execute(
                        f"SELECT DISTINCT {aa_grouping_key}, {author_col} FROM corpus WHERE {author_col} IN ({','.join(['?']*len(known_authors))}) ORDER BY {aa_grouping_key}",
                        known_authors
                    ).fetchdf()
                    con.close()
                    
                    known_texts = known_preview[aa_grouping_key].tolist()
                    st.session_state['aa_known_texts'] = known_texts
                    
                    with st.expander(f"📋 Known Texts ({len(known_texts)} segments)"):
                        st.dataframe(known_preview, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown("#### Step 2: Define Questioned Texts (by Author)")
                st.caption(f"Select author values from '{author_col}' that represent Questioned texts")
                
                # Filter out already-selected known authors
                available_questioned_authors = [a for a in all_authors if a not in known_authors]
                
                questioned_authors = st.multiselect(
                    f"Questioned Authors ({author_col})",
                    options=available_questioned_authors,
                    key="aa_questioned_authors",
                    help="Select author values that identify Questioned texts"
                )
                
                # Preview Questioned texts
                if questioned_authors:
                    con = duckdb.connect(corpus_path, read_only=True)
                    questioned_preview = con.execute(
                        f"SELECT DISTINCT {aa_grouping_key}, {author_col} FROM corpus WHERE {author_col} IN ({','.join(['?']*len(questioned_authors))}) ORDER BY {aa_grouping_key}",
                        questioned_authors
                    ).fetchdf()
                    con.close()
                    
                    questioned_texts = questioned_preview[aa_grouping_key].tolist()
                    st.session_state['aa_questioned_texts'] = questioned_texts
                    
                    with st.expander(f"📄 Questioned Texts ({len(questioned_texts)} segments)"):
                        st.dataframe(questioned_preview, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            aa_top_n = st.slider("Top Features (MFW)", 50, 500, 100, key="aa_topn", 
                               help="Most Frequent Words. Stylo typically uses 100-1000.")
            
            # Step 3: Run Attribution
            aa_ready = False
            if not known_texts:
                st.warning("⚠️ Define at least 1 Known text.")
            elif not questioned_texts:
                st.warning("⚠️ Define at least 1 Questioned text.")
            else:
                aa_ready = True
                st.success(f"✅ Ready: {len(known_texts)} Known, {len(questioned_texts)} Questioned")
                
            if st.button("🕵️ Run Attribution", disabled=not aa_ready, type="primary"):
                with st.spinner("Calculating Stylo-Delta..."):
                    try:
                        status_placeholder = st.empty()
                        
                        # 1. Build restricted matrix
                        all_selected = sorted(list(set(known_texts + questioned_texts)))
                        
                        # CRITICAL DEBUG: Inform user of exactly what files are going in
                        st.subheader("🕵️ Calculation Scope Debug")
                        st.write(f"- Selection Type: {selection_method}")
                        st.write(f"- Known Text Count: {len(known_texts)}")
                        st.write(f"- Questioned Text Count: {len(questioned_texts)}")
                        st.write(f"- Total Unique: {len(all_selected)}")
                        
                        if len(all_selected) == 0:
                            st.error("No texts selected. Calculation aborted.")
                            st.stop()
                            
                        if len(all_selected) > 100:
                            st.error(f"🛑 FATAL: Analysis scope ({len(all_selected)} files) exceeds the safety limit of 100. This is likely due to a selection error. Aborting.")
                            st.stop()
                        
                        status_placeholder.info(f"📊 Building restricted feature matrix for {len(all_selected)} segments...")
                        raw_matrix, features = get_feature_matrix(
                            corpus_path,
                            group_by=aa_grouping_key,
                            top_n_features=aa_top_n,
                            selected_groups=all_selected,
                            ngram_sizes=[1]
                        )
                        
                        if raw_matrix.empty:
                            st.error("Matrix empty. Try increasing Top MFW.")
                        else:
                            # 2. Get Metadata for the restricted list only
                            status_placeholder.info("🏷️ Fetching author labels for selected docs...")
                            con = duckdb.connect(corpus_path, read_only=True)
                            placeholders = ', '.join(['?'] * len(all_selected))
                            # Crucial: Use MAX() or similar to ensure 1 label per ID if duplicates exist
                            meta_df = con.execute(
                                f"SELECT {aa_grouping_key}, MAX({author_col}) as author_label FROM corpus WHERE {aa_grouping_key} IN ({placeholders}) GROUP BY {aa_grouping_key}",
                                all_selected
                            ).fetchdf()
                            con.close()
                            
                            # 3. Align Matrix and Labels
                            # We join locally to guarantee match. meta_df.index is now definitely unique.
                            annotated_matrix = raw_matrix.join(meta_df.set_index(aa_grouping_key), how='inner')
                            
                            # Drop author column for numeric operations
                            author_final_col = 'author_label'
                            numeric_matrix = annotated_matrix.drop(columns=[author_final_col])
                            
                            # Z-score standardization (optimized)
                            status_placeholder.info(f"📐 Standardizing {numeric_matrix.shape[0]} docs × {numeric_matrix.shape[1]} features...")
                            matrix_values = numeric_matrix.values
                            means = np.mean(matrix_values, axis=0)
                            stds = np.std(matrix_values, axis=0, ddof=1)
                            stds[stds == 0] = 1
                            z_matrix = pd.DataFrame((matrix_values - means) / stds, index=numeric_matrix.index, columns=numeric_matrix.columns)
                            
                            # 4. Final Alignment and Indices
                            # Re-verify known/questioned based on what survived the matrix build
                            final_known = [f for f in z_matrix.index if f in known_texts]
                            final_questioned = [f for f in z_matrix.index if f in questioned_texts]
                            
                            labels_list = []
                            for fname in z_matrix.index:
                                if fname in known_texts:
                                    val = annotated_matrix.loc[fname, author_final_col]
                                    # Fallback defense
                                    if hasattr(val, 'tolist'): val = val.tolist()[0] if len(val)>0 else "UNKNOWN"
                                    labels_list.append(str(val))
                                else:
                                    labels_list.append("QUESTIONED")
                            
                            train_indices = [i for i, f in enumerate(z_matrix.index) if f in final_known]
                            test_indices = [i for i, f in enumerate(z_matrix.index) if f in final_questioned]
                            
                            # 5. Calculate Results
                            # Delta
                            status_placeholder.info(f"🔍 Computing Delta ({len(train_indices)} Known → {len(test_indices)} Questioned)...")
                            delta_res = perform_burrows_delta(z_matrix, labels_list, train_indices, test_indices)
                            
                            # PCA
                            status_placeholder.info("📍 Calculating PCA Map...")
                            from core.modules.statistical_testing import perform_pca
                            df_pca, variance = perform_pca(z_matrix)
                            
                            # Network
                            status_placeholder.info("🕸️ Creating Similarity Network...")
                            from core.modules.statistical_testing import perform_network_similarity
                            nodes, edges = perform_network_similarity(z_matrix, threshold=0.4)
                            
                            # Dendrogram
                            status_placeholder.info("🌳 Growing Tree...")
                            from core.modules.statistical_testing import perform_clustering
                            clustering_res = perform_clustering(z_matrix, distance_metric='cityblock', method='ward')
                            
                            status_placeholder.empty()
                            st.session_state['aa_results'] = {
                                'delta': delta_res,
                                'pca': (df_pca, variance),
                                'network': (nodes, edges),
                                'clustering': clustering_res,
                                'labels': labels_list,
                                'matrix_shape': z_matrix.shape,
                                'selected_counts': (len(final_known), len(final_questioned))
                            }
                            st.success("✅ Attribution Complete.")

                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        # --- RESULTS AREA ---
        if 'aa_results' in st.session_state:
            res = st.session_state['aa_results']
            
            # Show a persistent debug summary for verification
            with st.expander("🛠️ System Verification (Scope Debug)", expanded=True):
                col_db1, col_db2 = st.columns(2)
                with col_db1:
                    st.write("**Selection Data (Before Matrix)**")
                    st.write(f"- Known Texts Count: {len(known_texts)}")
                    st.write(f"- Questioned Texts Count: {len(questioned_texts)}")
                    if known_texts: st.write(f"- Sample Known: {known_texts[:3]}")
                with col_db2:
                    st.write("**Processing Data (In Matrix)**")
                    st.write(f"- total Rows: {res['matrix_shape'][0]}")
                    st.write(f"- train (Known) rows: {res['selected_counts'][0]}")
                    st.write(f"- test (Questioned) rows: {res['selected_counts'][1]}")
                
                if res['matrix_shape'][0] > 100:
                    st.error("🚨 WARNING: Scope is too large (whole corpus?). Filtering failed.")
                
            tab_verdict, tab_pca, tab_dendro, tab_net = st.tabs([
                "🏆 Authorship Verdict", "📍 PCA Map", "🌳 Dendrogram", "🕸️ Similarity Network"
            ])
            
            with tab_verdict:
                st.markdown("### 🏆 Authorship Attribution Verdict")
                st.caption("Lower distance = Closer stylistic match.")
                
                delta_df = res['delta']
                summary_data = []
                for test_doc, row in delta_df.iterrows():
                    best_match = row.idxmin()
                    best_score = row.min()
                    
                    if len(row) > 1:
                        sorted_scores = row.nsmallest(2)
                        second_score = sorted_scores.values[1]
                        confidence = ((second_score - best_score) / second_score * 100)
                    else:
                        confidence = 100.0
                    
                    verdict = "Closely Similar" if confidence >= 10.0 else "Outlier"
                    
                    summary_data.append({
                        'Questioned Text': test_doc,
                        'Best Match (Known Author)': best_match,
                        'Confidence': f"{confidence:.2f}%",
                        'Verdict': verdict
                    })
                
                summary_df = pd.DataFrame(summary_data)
                
                def color_verdict(val):
                    # Use standard colors for better accessibility
                    bg_color = '#c6f6d5' if val == 'Closely Similar' else '#fed7d7' # Light green / light red
                    text_color = '#22543d' if val == 'Closely Similar' else '#822727' # Dark green / dark red
                    return f'background-color: {bg_color}; color: {text_color}; font-weight: bold; border-radius: 4px'

                st.dataframe(
                    summary_df.style.applymap(color_verdict, subset=['Verdict']),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=delta_df.to_csv(),
                    file_name="authorship_attribution_results.csv",
                    mime="text/csv"
                )
                
                st.info("""
                **Interpretation Guide:**
                - **Confidence**: Measures the relative gap between the primary candidate and the second-closest match.
                - **Closely Similar**: Strong stylistic alignment with a single candidate (Confidence ≥ 10%).
                - **Outlier**: The text shows either weak association or ambiguous similarity across multiple candidates (Confidence < 10%).
                """)

            with tab_pca:
                st.markdown("### 📍 Principal Component Analysis (PCA)")
                st.caption("2D Projection of documents based on stylistic variation.")
                
                df_pca, variance = res['pca']
                # Add labels for coloring, ensuring string types for all comparisons
                plot_df = df_pca.copy()
                plot_df['Label'] = [str(l) for l in res['labels']]
                plot_df['Type'] = ["Known" if str(l).strip().upper() != "QUESTIONED" else "Questioned" for l in res['labels']]
                
                import plotly.express as px
                fig_pca = px.scatter(
                    plot_df, x='PC1', y='PC2',
                    color='Label',
                    symbol='Type',
                    hover_name=plot_df.index,
                    title=f"PCA stylometric map (Variance: PC1={variance[0]*100:.1f}%, PC2={variance[1]*100:.1f}%)"
                )
                st.plotly_chart(fig_pca, use_container_width=True)

            with tab_dendro:
                st.markdown("### 🌳 Hierarchical Clustering (Dendrogram)")
                st.caption("Ward's Method / Manhattan Distance (Stylometric proximity tree)")
                
                import matplotlib.pyplot as plt
                from scipy.cluster.hierarchy import dendrogram
                
                fig_den, ax = plt.subplots(figsize=(10, 6))
                dendrogram(
                    res['clustering']['linkage'],
                    labels=res['clustering']['labels'],
                    orientation='left',
                    leaf_font_size=10,
                    ax=ax
                )
                plt.tight_layout()
                st.pyplot(fig_den)

            with tab_net:
                st.markdown("### 🕸️ Similarity Network")
                st.caption("Links show strong stylistic proximity between documents.")
                
                nodes, edges = res['network']
                if edges.empty:
                    st.warning("No strong connections found between documents at the current threshold.")
                else:
                    # Simple 2D network plot using Plotly
                    # For a better network viz, we could use Pyvis, but Plotly is safer here
                    st.write(f"Found {len(edges)} strong stylistic links.")
                    st.dataframe(edges, use_container_width=True)


    # ---------------- GROUP DIFFERENCE TAB ----------------
    with tab_group:
        st.caption("**Goal:** Compare word usage between two distinct groups (e.g. Male vs Female).")
        
        # Main configuration
        with st.expander("📝 **Configure Statistical Test**", expanded=True):
            st.markdown("### Continuous Variable (Dependent)")
            
            # Search Mode Selection
            search_mode = st.radio("Search Mode", ["Standard", "Natural Language (Rule)", "Natural Language (AI)"], horizontal=True, key="stats_search_mode")
            
            query = ""
            
            if search_mode == "Natural Language (AI)":
                nl_query = st.text_area("Describe your statistical query", height=70, placeholder="e.g. Compare usage of 'make' followed by a noun between groups", key="stats_nl_query_ai")
                
                col_ai1, col_ai2 = st.columns([1, 3])
                with col_ai1:
                    analyze_btn = st.button("Interpret Query", type="primary", key="stats_ai_btn")
                
                if analyze_btn and nl_query:
                    with st.spinner("AI is determining search parameters..."):
                        pos_defs = get_pos_definitions(corpus_path) or {}
                        lang = get_corpus_language(corpus_path)
                        if lang: pos_defs['__language_context__'] = lang

                        from core.ai_service import parse_nl_query
                        params, err = parse_nl_query(
                            nl_query, 
                            "concordance", # Reusing concordance parser as it returns 'query' string
                            ai_provider=get_state('ai_provider'),
                            gemini_api_key=get_state('gemini_api_key'),
                            ollama_url=get_state('ollama_url'),
                            ollama_model=get_state('ai_model'),
                            pos_definitions=pos_defs
                        )
                        
                        if params:
                            query = params.get('query', '')
                            set_state('stats_query', query)
                            st.success(f"✓ AI interpreted: '{query}'")
                        else:
                            st.error(f"Could not parse query: {err}")
                
                # Use query from state if available
                query = get_state('stats_query', '')

            elif search_mode == "Natural Language (Rule)":
                nl_query = st.text_input("Enter natural language rule", placeholder="e.g. adjective followed by noun", key="stats_nl_query_rule")
                if nl_query:
                    pos_defs = get_pos_definitions(corpus_path) or {}
                    reverse_pos_map = {v.lower(): k for k, v in pos_defs.items() if v}
                    
                    from core.modules.concordance import parse_nl_query_rules_only
                    params, err = parse_nl_query_rules_only(nl_query, "concordance", reverse_pos_map=reverse_pos_map)
                    
                    if params:
                        query = params.get('query', '')
                        st.caption(f"✓ Rule parsed: `{query}`")
                    else:
                        st.error(f"Rule Error: {err}")
            
            else: # Standard
                query = st.text_input(
                    "Search Query",
                    value=get_state('stats_query', '_JJ*'),
                    help="Use * for wildcards (e.g. run*), _TAG for POS (e.g. _NN*), [lemma] for lemma, token_POS (e.g. light_V*), or <TAG> for XML tags (e.g. <EVAL sentiment=\"positive\">)",
                    key="stats_query_input"
                )
                set_state('stats_query', query)

            # Query syntax help
            with st.expander("💡 Query Syntax Examples"):
                st.markdown("""
                | Syntax | Description | Example |
                |--------|-------------|---------|
                | `_POS*` | POS tag filter | `_JJ*` (all adjectives) |
                | `*wildcard*` | Wildcard matching | `*ing` (words ending in 'ing') |
                | `[lemma]` | Lemma search | `[run]` (all forms of 'run') |
                | `token_POS` | Token + POS combined | `light_V*` ('light' as verb) |
                | `<TAG>` | XML tag search | `<PN>` (all person/place names) |
                | `<TAG attr="val">` | XML tag with attributes | `<PN type="human">` (people only) |
                | `(word1\|word2)` | OR pattern (NEW) | `(small\|big\|little)` |
                | `_POS1\|POS2` | Multiple POS tags | `_NN*\|VB*` (nouns OR verbs) |
                """)
            
            # Query preview
            if query:
                col_prev1, col_prev2 = st.columns([3, 1])
                with col_prev1:
                    if st.button("🔍 Preview Query Matches", key="preview_btn"):
                        with st.spinner("Analyzing query..."):
                            try:
                                preview = preview_query_matches(
                                    corpus_path, 
                                    query, 
                                    min_freq=min_freq if 'min_freq' in locals() else 3
                                )
                                
                                if 'error' in preview:
                                    st.error(preview['error'])
                                else:
                                    st.success(f"✅ **{preview['total_words']} unique words** matched (total freq: {preview['total_freq']:,})")
                                    st.info(f"📊 With min_freq: **{preview['words_above_threshold']} words** will be tested")
                                    st.caption(f"Sample: {', '.join(preview['sample_words'][:10])}")
                            except Exception as e:
                                st.error(f"Error previewing query: {str(e)}")
            
            # Advanced options
            st.markdown("### Grouping Variable (Independent)")
            
            # Get XML attributes for grouping
            con = duckdb.connect(corpus_path, read_only=True)
            try:
                attr_cols = get_xml_attribute_columns(con)
            finally:
                con.close()
            
            if not attr_cols:
                st.warning("No XML attributes found. Cannot perform group comparison.")
                return
            
            grouping_attr = st.selectbox(
                "Select grouping attribute",
                options=attr_cols,
                key="stats_grouping_attr"
            )
            
            # Get unique values for selected attribute
            con = duckdb.connect(corpus_path, read_only=True)
            try:
                unique_vals = [r[0] for r in con.execute(
                    f"SELECT DISTINCT {grouping_attr} FROM corpus WHERE {grouping_attr} IS NOT NULL ORDER BY {grouping_attr}"
                ).fetchall()]
                unique_vals = [str(v) for v in unique_vals if str(v).strip()]
            finally:
                con.close()
            
            # Group selection
            groups = st.multiselect(
                f"Select groups to compare (from '{grouping_attr}')",
                options=unique_vals,
                default=unique_vals[:2] if len(unique_vals) >= 2 else unique_vals,
                key="stats_groups"
            )
            
            if len(groups) != 2:
                st.warning("⚠️ Phase 1 MVP supports only 2-group comparisons. Please select exactly 2 groups.")
                # We don't return here because we want to show the rest of the UI (like xml filters)? 
                # Actually previously we returned. Let's keep returning within the tab.
                st.stop() # Only stop this script execution, but inside a tab/function it stops rendering.
            
            # Test configuration
            st.markdown("### Test Configuration")
            
            st.info("📊 **Statistical Test:** Chi-square test for proportions (automatically selected for count data)")
            
            col_test1, col_test2 = st.columns(2)
            
            with col_test1:
                correction_method = st.selectbox(
                    "Multiple Comparison Correction",
                    options=["Bonferroni (conservative)", "FDR (Benjamini-Hochberg)", "None"],
                    key="stats_correction"
                )
                correction_key = None if "None" in correction_method else ("bonferroni" if "Bonferroni" in correction_method else "fdr_bh")
            
            with col_test2:
                min_freq = st.number_input(
                    "Minimum Frequency Threshold",
                    min_value=1,
                    max_value=100,
                    value=3,
                    help="Exclude words with total frequency below this threshold",
                    key="stats_min_freq"
                )
        
        # XML Restrictions (Optional)
        xml_filters = render_xml_restriction_filters(corpus_path, "statistical_testing", corpus_name=corpus_name)
        xml_where, xml_params = apply_xml_restrictions(xml_filters) if xml_filters else ("", [])
        
        # Run Analysis Button
        if st.button("🚀 Run Statistical Test", type="primary", key="run_stats_btn"):
            if not query:
                st.error("Please enter a search query.")
                # return # Don't return, just show error
            
            elif len(groups) != 2:
                st.error("Please select exactly 2 groups.")
                
            else:
                with st.spinner(f"Running Chi-square test: {groups[0]} vs {groups[1]}..."):
                    try:
                        results_df = compare_groups_by_word(
                            corpus_db_path=corpus_path,
                            query=query,
                            grouping_attr=grouping_attr,
                            groups=groups,
                            min_freq=min_freq,
                            freq_measure='absolute',
                            test_type='chi2',  # Always Chi-square for count data
                            multiple_comparison=correction_key,
                            xml_where_clause=xml_where,
                            xml_params=xml_params
                        )
                        
                        if results_df.empty:
                            st.warning("No words matched the query with the specified minimum frequency threshold.")
                        else:
                            # Store results in session
                            st.session_state['last_stats_results'] = {
                                'df': results_df,
                                'query': query,
                                'groups': groups,
                                'grouping_attr': grouping_attr,
                                'test_type': "Chi-square (proportion test)",
                                'correction': correction_method,
                                'corpus_name': corpus_name
                            }
                            
                            st.success(f"✅ Analysis complete! Tested {len(results_df)} words.")
                            
                    except Exception as e:
                        st.error(f"Error running statistical test: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Display Results
        if 'last_stats_results' in st.session_state:
            results = st.session_state['last_stats_results']
            df = results['df']
            
            st.markdown("---")
            st.markdown(f"## 📊 Results: {results['query']}")
            st.caption(f"**Test:** Chi-square (proportion test) | **Groups:** {results['groups'][0]} vs {results['groups'][1]} | **Correction:** {results['correction']}")
            
            # Summary stats
            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
            
            sig_count = len(df[df['significance'] != 'ns'])
            total_count = len(df)
            
            with col_sum1:
                st.metric("Total Words Tested", total_count)
            with col_sum2:
                st.metric("Significant Results", sig_count)
            with col_sum3:
                st.metric("Significance Rate", f"{(sig_count/total_count*100):.1f}%")
            with col_sum4:
                avg_effect = df['effect_size'].abs().mean()
                st.metric("Avg Effect Size", f"{avg_effect:.3f}")
            
            # Filter controls
            col_f1, col_f2 = st.columns([2, 1])
            with col_f1:
                show_filter = st.selectbox(
                    "Show Results",
                    options=["All", "Significant Only (p < 0.05)", "Highly Significant (p < 0.01)"],
                    key="stats_filter"
                )
            with col_f2:
                sort_by = st.selectbox(
                    "Sort By",
                    options=["P-value (ascending)", "Effect Size (absolute)", "Word (alphabetical)"],
                    key="stats_sort"
                )
            
            # Apply filters
            df_display = df.copy()
            
            if "Significant Only" in show_filter:
                df_display = df_display[df_display['p_value_corrected'] < 0.05]
            elif "Highly Significant" in show_filter:
                df_display = df_display[df_display['p_value_corrected'] < 0.01]
            
            # Apply sorting
            if "P-value" in sort_by:
                df_display = df_display.sort_values('p_value_corrected')
            elif "Effect Size" in sort_by:
                df_display = df_display.sort_values('effect_size', key=lambda x: x.abs(), ascending=False)
            else:
                df_display = df_display.sort_values('word')
            
            # Format display
            df_display_formatted = df_display.copy()
            if f'{results["groups"][0]}_prop' in df_display_formatted.columns:
                 # Convert to percentage
                 df_display_formatted[f'{results["groups"][0]}_prop'] = df_display_formatted[f'{results["groups"][0]}_prop'].apply(lambda x: f"{x*100:.2f}%")
                 df_display_formatted[f'{results["groups"][1]}_prop'] = df_display_formatted[f'{results["groups"][1]}_prop'].apply(lambda x: f"{x*100:.2f}%")
            
            df_display_formatted['p_value'] = df_display_formatted['p_value'].apply(lambda x: f"{x:.4f}" if x >= 0.001 else "<.001")
            df_display_formatted['p_value_corrected'] = df_display_formatted['p_value_corrected'].apply(lambda x: f"{x:.4f}" if x >= 0.001 else "<.001")
            df_display_formatted['effect_size'] = df_display_formatted['effect_size'].apply(lambda x: f"{x:.3f}")
            df_display_formatted['test_statistic'] = df_display_formatted['test_statistic'].apply(lambda x: f"{x:.3f}")
            
            # Rename columns for clarity
            col_names = {
                'word': 'Word',
                f'{results["groups"][0]}_freq': f'{results["groups"][0]} Counts',
                f'{results["groups"][1]}_freq': f'{results["groups"][1]} Counts',
                f'{results["groups"][0]}_prop': f'{results["groups"][0]} %',
                f'{results["groups"][1]}_prop': f'{results["groups"][1]} %',
                'test_statistic': 'Test Stat',
                'p_value': 'p-value',
                'p_value_corrected': 'p-value (corrected)',
                'effect_size': "Cohen's h",
                'significance': 'Sig'
            }
            df_display_formatted = df_display_formatted.rename(columns=col_names)
            
            # Display table
            st.dataframe(
                df_display_formatted,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # Download button
            st.download_button(
                label=f"📥 Download Results (Excel)",
                data=df_to_excel_bytes(df),
                file_name=f"stats_test_{results['query']}_{results['corpus_name']}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_stats"
            )
            
            st.markdown("---")
            st.subheader("📋 Statistical Report")
            
            # automated report generation
            with st.container(border=True):
                avg_h = df['effect_size'].abs().mean()
                max_h = df['effect_size'].abs().max() if not df.empty else 0
                
                report = f"""
                **1. Methodology**
                A **Chi-square test of independence** was performed to compare the relative frequencies of **{total_count} words** between groups **'{results['groups'][0]}'** and **'{results['groups'][1]}'**.
                
                **2. Data & Settings**
                - **Variable Checked:** Word frequencies matching query `"{results['query']}"`
                - **Grouping Variable:** `{results['grouping_attr']}`
                - **Correction Method:** `{results['correction']}` (Used to control false positive rate across {total_count} tests)
                
                **3. Results Summary**
                - **Significant Differences:** {sig_count} words found ({sig_count/total_count*100:.1f}%)
                - **Average Effect Size (Cohen's h):** {avg_h:.3f} (Interpretation: <0.2 Small, 0.5 Medium, >0.8 Large)
                - **Maximum Effect Size:** {max_h:.3f}
                """
                
                report += "\n**4. Interpretation**\n"
                
                # Overall conclusion logic
                
                # Omnibus Test (Chi-square on the entire Words x Groups table)
                # This answers: "Is the overall vocabulary distribution significantly different between groups?"
                try:
                    counts_matrix = df[[f'{results["groups"][0]}_freq', f'{results["groups"][1]}_freq']].values
                    # Removing rows with all zeros just in case (though shouldn't happen with min_freq)
                    counts_matrix = counts_matrix[~np.all(counts_matrix == 0, axis=1)]
                    chi2_omnibus, p_omnibus, _, _ = stats.chi2_contingency(counts_matrix)
                    p_report = "< .001" if p_omnibus < 0.001 else f"= {p_omnibus:.3f}"
                    sig_text = "statistically significant" if p_omnibus < 0.05 else "not statistically significant"
                except:
                    p_report = "N/A"
                    sig_text = "indeterminate"

                if sig_count == 0:
                    report += f"❌ **Conclusion:** The difference between the two groups is **{sig_text}** (χ² = {chi2_omnibus:.1f}, p {p_report}).\n\n"
                    report += f"- No individual words showed significant differences after correction."
                    if results['correction'] == 'Bonferroni (conservative)':
                        report += " (Note: The strict Bonferroni correction may be masking subtle differences.)"
                else:
                    sig_rate = sig_count / total_count
                    
                    report += f"✅ **Conclusion:** The difference between the two groups is **{sig_text}** (χ² = {chi2_omnibus:.1f}, p {p_report}).\n\n"
                    
                    if sig_rate > 0.10 or avg_h > 0.2:
                        report += f"- **Distinctiveness:** {sig_count} words ({sig_rate*100:.1f}%) showed statistically significant differences."
                    else:
                        report += f"- **Distinctiveness:** Only {sig_count} specific words ({sig_rate*100:.1f}%) differed significantly, suggesting localized differences within a broadly similar vocabulary."
                    
                    top_words = df_display_formatted.head(3)['Word'].tolist()
                    report += f"\n- **Key Differentiators:** The most distinct words were **{', '.join(top_words)}**."
                    
                    if max_h < 0.2:
                         report += "\n- **Note:** While statistically significant, the effect sizes are small (< 0.2), meaning the differences in frequency are subtle."
                
                st.markdown(report)
                
                # AI Interpretation Button
                if st.button("✨ Generate AI Narrative Report"):
                    with st.spinner("Generating narrative..."):
                        from core.ai_service import interpret_results_llm
                        
                        # Prepare data summary for LLM
                        data_summary = f"""
                        Comparison: {results['groups'][0]} vs {results['groups'][1]}
                        Test: Chi-square
                        Total Words: {total_count}
                        Significant: {sig_count}
                        Query: {results['query']}
                        Top Results:
                        {df[['word', 'p_value_corrected', 'effect_size']].head(10).to_string()}
                        """
                        
                        narrative, err = interpret_results_llm(
                            target_word=results['query'],
                            analysis_type="Statistical Comparison",
                            data_description=f"Comparison of {results['groups'][0]} vs {results['groups'][1]}",
                            data=data_summary,
                            ai_provider=get_state('ai_provider'),
                            gemini_api_key=get_state('gemini_api_key'),
                            ollama_url=get_state('ollama_url'),
                            ollama_model=get_state('ai_model')
                        )
                        
                        if narrative:
                            st.markdown("### 🤖 AI Analysis")
                            st.markdown(narrative)
                        else:
                            st.error(f"AI Error: {err}")
