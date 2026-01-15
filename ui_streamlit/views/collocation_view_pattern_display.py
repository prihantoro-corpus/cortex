
def render_pattern_results(pattern_results, collocation_results, key_suffix=""):
    """
    Display pattern-grouped collocation results.
    
    Args:
        pattern_results: Dict with 'groups', 'patterns', 'limit'
        collocation_results: Original collocation results dict
        key_suffix: Unique key suffix for Streamlit widgets
    """
    groups = pattern_results.get('groups', {})
    patterns = pattern_results.get('patterns', [])
    limit = pattern_results.get('limit', 50)
    
    if not groups:
        st.info("No collocates matched any patterns.")
        return
    
    st.markdown("---")
    st.header("🎯 Pattern-Based Collocation Groups")
    st.caption(f"Pattern matching applied to top {limit} collocates")
    
    # Get corpus info for concordance fetching
    db_path = get_state('current_corpus_path') if key_suffix != 'c2' else get_state('comp_corpus_path')
    corpus_name = get_state('current_corpus_name') if key_suffix != 'c2' else get_state('comp_corpus_name')
    node_word = collocation_results.get('node', '')
    window = collocation_results.get('window', 5)
    
    # Display each pattern group
    for pattern in patterns:
        label = pattern['label']
        pattern_str = pattern['pattern_str']
        
        if label not in groups or groups[label].empty:
            continue
        
        df_group = groups[label]
        
        with st.expander(f"📌 {label} ({len(df_group)} collocates)", expanded=True):
            st.caption(f"Pattern: `{pattern_str}`")
            
            # Display table
            display_cols = ['Collocate', 'LL', 'MI', 'Observed', 'Direction', 'POS']
            available_cols = [col for col in display_cols if col in df_group.columns]
            
            df_display = df_group[available_cols].copy()
            df_display = df_display.sort_values('LL', ascending=False)
            df_display.index = range(1, len(df_display) + 1)
            
            st.dataframe(df_display, use_container_width=True)
            
            # Show concordance examples for top collocates
            st.markdown("**Example Concordances:**")
            
            top_collocates = df_display.head(5)
            
            for idx, (_, row) in enumerate(top_collocates.iterrows(), 1):
                collocate = row['Collocate']
                ll_score = row.get('LL', 0)
                
                # Fetch one concordance example
                try:
                    c_kwic, _, _, _, c_sent_ids, _ = cached_generate_kwic(
                        db_path=db_path,
                        query=node_word,
                        left=7,
                        right=7,
                        corpus_name=corpus_name,
                        pattern_collocate_input=collocate,
                        pattern_window=window,
                        limit=1
                    )
                    
                    if c_kwic:
                        left_ctx = c_kwic[0]['Left']
                        node_ctx = c_kwic[0]['Node']
                        right_ctx = c_kwic[0]['Right']
                        
                        st.markdown(
                            f"{idx}. **{collocate}** (LL: {ll_score:.2f}): "
                            f"<span style='color: #888'>{left_ctx}</span> "
                            f"<span style='color: #00FFF5; font-weight: bold'>{node_ctx}</span> "
                            f"<span style='color: #888'>{right_ctx}</span>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(f"{idx}. **{collocate}** (LL: {ll_score:.2f}): _No example found_")
                except Exception as e:
                    st.markdown(f"{idx}. **{collocate}** (LL: {ll_score:.2f}): _Error fetching example_")
            
            # Download button for this pattern group
            from core.io_utils import df_to_excel_bytes
            st.download_button(
                label=f"Download '{label}' Group (Excel)",
                data=df_to_excel_bytes(df_group),
                file_name=f"pattern_{label.replace(' ', '_')}_{node_word}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_pattern_{label}_{key_suffix}"
            )
