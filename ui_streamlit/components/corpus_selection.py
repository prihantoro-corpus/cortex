import streamlit as st
from ui_streamlit.state_manager import get_state, set_state, reset_tool_states

def render_corpus_selection_main():
    """
    Renders the Corpus Selection controls on the main screen.
    """
    current_path = get_state('current_corpus_path')
    active_corpus_name = get_state('current_corpus_name')
    comp_path = get_state('comp_corpus_path')
    comp_name = get_state('comp_corpus_name')
    comp_mode = get_state('comparison_mode', False)

    # Active Corpus Display Banner on Main Screen
    if current_path:
        display_name = "USER CORPUS" if active_corpus_name == "Uploaded Batch" else active_corpus_name
        if comp_mode and comp_path:
            display_comp = "USER CORPUS" if comp_name == "Uploaded Batch" else comp_name
            banner_html = f"""
            <div style='background-color:#1e293b; padding:12px 18px; border-radius:10px; border:1px solid #00ADB5; margin-bottom:15px;'>
                📂 <span style='color:#00FFF5; font-weight:bold;'>Active:</span> 
                <span style='color:white; font-weight:bold; font-size:1.05em;'>{display_name}</span> 
                <span style='color:#00ADB5; font-weight:bold;'> vs </span> 
                <span style='color:white; font-weight:bold; font-size:1.05em;'>{display_comp}</span>
            </div>
            """
        else:
            banner_html = f"""
            <div style='background-color:#1e293b; padding:12px 18px; border-radius:10px; border:1px solid #00ADB5; margin-bottom:15px;'>
                📂 <span style='color:#00FFF5; font-weight:bold;'>Active:</span> 
                <span style='color:white; font-weight:bold; font-size:1.05em;'>{display_name}</span>
            </div>
            """
        st.markdown(banner_html, unsafe_allow_html=True)
        
        # Collapsed expander if already loaded
        expander_title = "⚙️ Corpus Selection & Settings"
        expander_expanded = False
    else:
        st.markdown("<div style='background-color:#1e293b; padding:12px 18px; border-radius:10px; border:1px solid #475569; margin-bottom:15px;'><span style='font-size:1.05em; color:#94a3b8; font-weight:bold;'>📂 Active Corpus: None Loaded</span></div>", unsafe_allow_html=True)
        expander_title = "⚙️ Choose Corpus & Settings"
        expander_expanded = True

    # Render Corpus settings in an expander
    with st.expander(expander_title, expanded=expander_expanded):
        col1, col2, col3 = st.columns(3)
        with col1:
            corpus_type = st.radio(
                "Corpus Type", 
                ["Monolingual", "Parallel"],
                index=0 if get_state('corpus_type') == "Monolingual" else 1,
                horizontal=True,
                key="main_corpus_type"
            )
            if corpus_type != get_state('corpus_type'):
                set_state('corpus_type', corpus_type)
                reset_tool_states()
                st.rerun()
                
        with col2:
            comparison_mode = st.checkbox("Enable Comparison Mode", value=comp_mode, key="main_comparison_mode")
            if comparison_mode != get_state('comparison_mode'):
                set_state('comparison_mode', comparison_mode)
                st.rerun()

        with col3:
            if 'sidebar_source_selectbox' not in st.session_state:
                st.session_state['sidebar_source_selectbox'] = "Upload Files"
            source_type = st.radio(
                "Source", 
                ["Upload Files", "Built-in Corpora", "Online Corpus"],
                key="sidebar_source_selectbox",
                horizontal=True
            )
            set_state('source_type', source_type)
            
        if source_type == "Online Corpus":
            online_mode = st.radio("Builder Mode", ["YouTube", "Mastodon", "BlueSky", "Link Collection", "Keyword Search", "Detik.com"], horizontal=True, key="main_online_builder_mode")
            set_state('online_builder_mode', online_mode)
