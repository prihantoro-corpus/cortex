import pandas as pd

def compare_collocations(df1, df2, collocate_column='Collocate', corpus1_name='Primary', corpus2_name='Comparison'):
    """
    Compare two collocation DataFrames to identify shared and unique collocates.
    
    Args:
        df1: Primary corpus collocation DataFrame
        df2: Comparison corpus collocation DataFrame
        collocate_column: Name of the column containing collocate words
        corpus1_name: Name of primary corpus (for column labels)
        corpus2_name: Name of comparison corpus (for column labels)
    
    Returns:
        tuple: (shared_df, df1_unique, df2_unique)
            - shared_df: Collocates in both corpora with stats from each
            - df1_unique: Collocates only in primary corpus
            - df2_unique: Collocates only in comparison corpus
    """
    if df1 is None or df2 is None or df1.empty or df2.empty:
        return pd.DataFrame(), df1 if df1 is not None else pd.DataFrame(), df2 if df2 is not None else pd.DataFrame()
    
    # Get sets of collocates
    set1 = set(df1[collocate_column].str.lower())
    set2 = set(df2[collocate_column].str.lower())
    
    # Find shared and unique
    shared = set1 & set2
    unique1 = set1 - set2
    unique2 = set2 - set1
    
    # Filter DataFrames
    df1_lower = df1.copy()
    df1_lower['_lower'] = df1_lower[collocate_column].str.lower()
    df2_lower = df2.copy()
    df2_lower['_lower'] = df2_lower[collocate_column].str.lower()
    
    # Get unique DataFrames
    df1_unique = df1_lower[df1_lower['_lower'].isin(unique1)].drop(columns=['_lower']).copy()
    df2_unique = df2_lower[df2_lower['_lower'].isin(unique2)].drop(columns=['_lower']).copy()
    
    # Merge shared items with stats from both corpora
    if shared:
        df1_shared = df1_lower[df1_lower['_lower'].isin(shared)].drop(columns=['_lower']).copy()
        df2_shared = df2_lower[df2_lower['_lower'].isin(shared)].drop(columns=['_lower']).copy()
        
        # Get all columns except the collocate column
        df1_cols = [col for col in df1_shared.columns if col != collocate_column]
        df2_cols = [col for col in df2_shared.columns if col != collocate_column]
        
        # Rename columns to distinguish corpora
        rename_map1 = {col: f'{col}_{corpus1_name}' for col in df1_cols}
        rename_map2 = {col: f'{col}_{corpus2_name}' for col in df2_cols}
        
        df1_shared = df1_shared.rename(columns=rename_map1)
        df2_shared = df2_shared.rename(columns=rename_map2)
        
        # Merge on collocate
        shared_df = pd.merge(
            df1_shared,
            df2_shared,
            on=collocate_column,
            how='inner'
        )
        
        # Calculate difference metrics for common columns
        # Look for Frequency columns
        freq_col1 = next((col for col in shared_df.columns if 'Frequency' in col and corpus1_name in col), None)
        freq_col2 = next((col for col in shared_df.columns if 'Frequency' in col and corpus2_name in col), None)
        
        if freq_col1 and freq_col2:
            shared_df['Freq_Diff'] = shared_df[freq_col1] - shared_df[freq_col2]
            # Sort by absolute frequency difference
            shared_df = shared_df.sort_values('Freq_Diff', key=abs, ascending=False).reset_index(drop=True)
        else:
            shared_df = shared_df.reset_index(drop=True)
    else:
        shared_df = pd.DataFrame()
    
    return shared_df, df1_unique, df2_unique


def compare_ngrams(df1, df2, ngram_column='N-Gram', corpus1_name='Primary', corpus2_name='Comparison'):
    """
    Compare two N-gram DataFrames to identify shared and unique N-grams.
    
    Args:
        df1: Primary corpus N-gram DataFrame
        df2: Comparison corpus N-gram DataFrame
        ngram_column: Name of the column containing N-grams
        corpus1_name: Name of primary corpus (for column labels)
        corpus2_name: Name of comparison corpus (for column labels)
    
    Returns:
        tuple: (shared_df, df1_unique, df2_unique)
            - shared_df: N-grams in both corpora with stats from each
            - df1_unique: N-grams only in primary corpus
            - df2_unique: N-grams only in comparison corpus
    """
    if df1 is None or df2 is None or df1.empty or df2.empty:
        return pd.DataFrame(), df1 if df1 is not None else pd.DataFrame(), df2 if df2 is not None else pd.DataFrame()
    
    # Get sets of N-grams (case-insensitive comparison)
    set1 = set(df1[ngram_column].str.lower())
    set2 = set(df2[ngram_column].str.lower())
    
    # Find shared and unique
    shared = set1 & set2
    unique1 = set1 - set2
    unique2 = set2 - set1
    
    # Create lowercase column for comparison
    df1_lower = df1.copy()
    df1_lower['_lower'] = df1_lower[ngram_column].str.lower()
    df2_lower = df2.copy()
    df2_lower['_lower'] = df2_lower[ngram_column].str.lower()
    
    # Get unique DataFrames
    df1_unique = df1_lower[df1_lower['_lower'].isin(unique1)].drop(columns=['_lower']).copy()
    df2_unique = df2_lower[df2_lower['_lower'].isin(unique2)].drop(columns=['_lower']).copy()
    
    # Merge shared items with stats from both corpora
    if shared:
        df1_shared = df1_lower[df1_lower['_lower'].isin(shared)].drop(columns=['_lower'])
        df2_shared = df2_lower[df2_lower['_lower'].isin(shared)].drop(columns=['_lower'])
        
        # Rename columns to distinguish corpora
        df1_shared = df1_shared.rename(columns={
            'Frequency': f'Freq_{corpus1_name}',
            'Relative Frequency (per M)': f'RelFreq_{corpus1_name}'
        })
        df2_shared = df2_shared.rename(columns={
            'Frequency': f'Freq_{corpus2_name}',
            'Relative Frequency (per M)': f'RelFreq_{corpus2_name}'
        })
        
        # Merge on N-gram
        shared_df = pd.merge(
            df1_shared[[ngram_column, f'Freq_{corpus1_name}', f'RelFreq_{corpus1_name}']],
            df2_shared[[ngram_column, f'Freq_{corpus2_name}', f'RelFreq_{corpus2_name}']],
            on=ngram_column,
            how='inner'
        )
        
        # Calculate difference metrics
        shared_df['Freq_Diff'] = shared_df[f'Freq_{corpus1_name}'] - shared_df[f'Freq_{corpus2_name}']
        shared_df['RelFreq_Diff'] = shared_df[f'RelFreq_{corpus1_name}'] - shared_df[f'RelFreq_{corpus2_name}']
        
        # Sort by absolute frequency difference
        shared_df = shared_df.sort_values('Freq_Diff', key=abs, ascending=False).reset_index(drop=True)
    else:
        shared_df = pd.DataFrame()
    
    return shared_df, df1_unique, df2_unique


def get_comparison_summary(shared_df, df1_unique, df2_unique, item_name='items'):
    """
    Generate a summary dictionary of comparison statistics.
    
    Args:
        shared_df: DataFrame of shared items
        df1_unique: DataFrame of primary-unique items
        df2_unique: DataFrame of comparison-unique items
        item_name: Name of items being compared (e.g., 'collocates', 'N-grams')
    
    Returns:
        dict: Summary statistics
    """
    return {
        'total_shared': len(shared_df),
        'total_primary_unique': len(df1_unique),
        'total_comparison_unique': len(df2_unique),
        'overlap_percentage': round(len(shared_df) / (len(shared_df) + len(df1_unique) + len(df2_unique)) * 100, 2) if (len(shared_df) + len(df1_unique) + len(df2_unique)) > 0 else 0,
        'item_name': item_name
    }


def render_comparison_tables(shared_df, df1_unique, df2_unique, corpus1_name, corpus2_name, analysis_type='collocation'):
    """
    Render comparison tables with AI interpretation buttons.
    
    Args:
        shared_df: DataFrame of shared items
        df1_unique: DataFrame of primary-unique items
        df2_unique: DataFrame of comparison-unique items
        corpus1_name: Name of primary corpus
        corpus2_name: Name of comparison corpus
        analysis_type: Type of analysis ('collocation' or 'ngram')
    """
    import streamlit as st
    from ui_streamlit.state_manager import get_state, set_state
    from core.ai_service import interpret_results_llm
    from core.io_utils import df_to_excel_bytes
    
    # 1. Shared Items
    with st.expander(f"🔗 Shared {analysis_type.title()}s ({len(shared_df)} items)", expanded=True):
        if not shared_df.empty:
            st.dataframe(shared_df.head(50), use_container_width=True)
            
            col_dl, col_ai = st.columns([1, 1])
            with col_dl:
                st.download_button(
                    label=f"Download Shared {analysis_type.title()}s",
                    data=df_to_excel_bytes(shared_df),
                    file_name=f"shared_{analysis_type}s.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_shared_{analysis_type}"
                )
            with col_ai:
                if st.button(f"🤖 Interpret Shared {analysis_type.title()}s", key=f"ai_shared_{analysis_type}"):
                    with st.spinner("Analyzing shared patterns..."):
                        data_str = shared_df.head(10).to_string(index=False)
                        prompt_desc = f"Shared {analysis_type}s appearing in both {corpus1_name} and {corpus2_name}."
                        
                        resp, err = interpret_results_llm(
                            target_word=f"Shared {analysis_type.title()}s",
                            analysis_type=f"Comparative {analysis_type.title()} Analysis",
                            data_description=prompt_desc,
                            data=data_str,
                            ollama_url=get_state('ollama_url'),
                            ollama_model=get_state('ai_model')
                        )
                        if resp:
                            set_state(f'ai_shared_{analysis_type}', resp)
                        elif err:
                            st.error(err)
            
            ai_resp = get_state(f'ai_shared_{analysis_type}')
            if ai_resp:
                st.markdown(ai_resp)
        else:
            st.info(f"No shared {analysis_type}s found.")
    
    # 2. Primary-Specific Items
    with st.expander(f"📘 {corpus1_name}-Specific {analysis_type.title()}s ({len(df1_unique)} items)", expanded=False):
        if not df1_unique.empty:
            st.dataframe(df1_unique.head(50), use_container_width=True)
            
            col_dl, col_ai = st.columns([1, 1])
            with col_dl:
                st.download_button(
                    label=f"Download {corpus1_name} Only",
                    data=df_to_excel_bytes(df1_unique),
                    file_name=f"{corpus1_name}_{analysis_type}s_unique.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_primary_{analysis_type}"
                )
            with col_ai:
                if st.button(f"🤖 Interpret {corpus1_name} Patterns", key=f"ai_primary_{analysis_type}"):
                    with st.spinner("Analyzing unique patterns..."):
                        data_str = df1_unique.head(10).to_string(index=False)
                        prompt_desc = f"{analysis_type.title()}s unique to {corpus1_name}, not found in {corpus2_name}."
                        
                        resp, err = interpret_results_llm(
                            target_word=f"{corpus1_name} Unique {analysis_type.title()}s",
                            analysis_type=f"Distinctive {analysis_type.title()} Analysis",
                            data_description=prompt_desc,
                            data=data_str,
                            ollama_url=get_state('ollama_url'),
                            ollama_model=get_state('ai_model')
                        )
                        if resp:
                            set_state(f'ai_primary_{analysis_type}', resp)
                        elif err:
                            st.error(err)
            
            ai_resp = get_state(f'ai_primary_{analysis_type}')
            if ai_resp:
                st.markdown(ai_resp)
        else:
            st.info(f"No unique {analysis_type}s in {corpus1_name}.")
    
    # 3. Secondary-Specific Items
    with st.expander(f"📗 {corpus2_name}-Specific {analysis_type.title()}s ({len(df2_unique)} items)", expanded=False):
        if not df2_unique.empty:
            st.dataframe(df2_unique.head(50), use_container_width=True)
            
            col_dl, col_ai = st.columns([1, 1])
            with col_dl:
                st.download_button(
                    label=f"Download {corpus2_name} Only",
                    data=df_to_excel_bytes(df2_unique),
                    file_name=f"{corpus2_name}_{analysis_type}s_unique.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_secondary_{analysis_type}"
                )
            with col_ai:
                if st.button(f"🤖 Interpret {corpus2_name} Patterns", key=f"ai_secondary_{analysis_type}"):
                    with st.spinner("Analyzing unique patterns..."):
                        data_str = df2_unique.head(10).to_string(index=False)
                        prompt_desc = f"{analysis_type.title()}s unique to {corpus2_name}, not found in {corpus1_name}."
                        
                        resp, err = interpret_results_llm(
                            target_word=f"{corpus2_name} Unique {analysis_type.title()}s",
                            analysis_type=f"Distinctive {analysis_type.title()} Analysis",
                            data_description=prompt_desc,
                            data=data_str,
                            ollama_url=get_state('ollama_url'),
                            ollama_model=get_state('ai_model')
                        )
                        if resp:
                            set_state(f'ai_secondary_{analysis_type}', resp)
                        elif err:
                            st.error(err)
            
            ai_resp = get_state(f'ai_secondary_{analysis_type}')
            if ai_resp:
                st.markdown(ai_resp)
        else:
            st.info(f"No unique {analysis_type}s in {corpus2_name}.")
