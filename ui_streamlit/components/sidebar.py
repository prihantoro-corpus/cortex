import streamlit as st
import os
import shutil
from ui_streamlit.state_manager import set_state, get_state, reset_tool_states
from core.preprocessing.corpus_loader import load_monolingual_corpus_files, load_built_in_corpus
from core.modules.overview import calculate_corpus_statistics
from core.config import get_available_corpora, BUILT_IN_CORPUS_DETAILS

def render_sidebar():
    """
    Renders the sidebar for corpus selection and settings.
    Returns: The selected view name.
    """
    # 1. Navigation (Tools) - MOVED TO TOP
    st.sidebar.title("Tools")
    view = st.sidebar.radio(
        "Go to", 
        ["Overview", "Concordance", "N-Gram", "Collocation", "Dictionary", "Keyword", "Distribution"]
    )
    
    st.sidebar.markdown("---")
    
    # 2. Corpus Selection
    st.sidebar.title("Corpus Selection")
    
    # Corpus Type Selection
    corpus_type = st.sidebar.radio(
        "Corpus Type", 
        ["Monolingual", "Parallel"],
        index=0 if get_state('corpus_type') == "Monolingual" else 1
    )
    
    if corpus_type != get_state('corpus_type'):
        set_state('corpus_type', corpus_type)
        reset_tool_states()
        st.rerun()

    # Comparison Mode Toggle
    comparison_mode = st.sidebar.checkbox("Enable Comparison Mode", value=get_state('comparison_mode', False))
    if comparison_mode != get_state('comparison_mode'):
        set_state('comparison_mode', comparison_mode)
        st.rerun()
        
    # Corpus Source
    source_type = st.sidebar.selectbox("Source", ["Upload Files", "Built-in Corpora"])
    
    current_path = get_state('current_corpus_path')
    
    if source_type == "Upload Files":
        uploaded_files = st.sidebar.file_uploader(
            "Upload Corpus Files (XML, TXT, CSV, XLSX)", 
            accept_multiple_files=True,
            type=['xml', 'txt', 'csv', 'xlsx']
        )
        
        # New: Language and Format Selection
        lang_col, fmt_col = st.sidebar.columns(2)
        with lang_col:
            lang_code = st.selectbox("Language", ["en", "id", "jp", "OTHER"], index=0)
        with fmt_col:
            fmt = st.selectbox("Format", [".txt / auto", "verticalised (T/P/L)", "XML (Tagged)", "Excel Parallel"], index=0)
        
        if uploaded_files:
            if st.sidebar.button("Process Uploaded Files"):
                progress_bar = st.sidebar.progress(0)
                status_text = st.sidebar.empty()
                
                def update_progress(val, text):
                    progress_bar.progress(val)
                    status_text.caption(text)

                with st.spinner("Processing Corpus..."):
                    result = load_monolingual_corpus_files(
                        uploaded_files, 
                        explicit_lang_code=lang_code,
                        selected_format=fmt,
                        progress_callback=update_progress
                    )
                    
                    if result.get('error'):
                        st.error(result['error'])
                    else:
                        if not get_state('comparison_mode'):
                            set_state('current_corpus_path', result['db_path'])
                            set_state('corpus_stats', result['stats'])
                            set_state('current_corpus_name', "Uploaded Batch")
                            set_state('xml_structure_data', result.get('structure'))
                            set_state('target_lang', lang_code)
                        else:
                            # In Comparison mode, we might need to decide which one to load.
                            # Standard App logic: Load into 'comparison' slots if primary exists?
                            # For simplicity, let's offer "Load as Primary" / "Load as Comparison" buttons or detect if primary exists.
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

    elif source_type == "Built-in Corpora":
        built_in_corpora = get_available_corpora()
        
        if not built_in_corpora:
            st.warning("No corpora found in local 'corpora' directory.")
        else:
            selected_names = st.sidebar.multiselect(
                "Select Corpus (one or more)", 
                list(built_in_corpora.keys()),
                default=[]
            )
            
            # Show info for first selected corpus
            if selected_names:
                detail = BUILT_IN_CORPUS_DETAILS.get(selected_names[0])
                if detail:
                    with st.sidebar.expander("ℹ️ Corpus Info"):
                        st.markdown(detail, unsafe_allow_html=True)

            if st.sidebar.button("Load Built-in", disabled=not selected_names):
                # Force reload of parser logic to pick up hotfixes
                import sys
                import importlib
                try:
                    if 'core.preprocessing.xml_parser' in sys.modules:
                        importlib.reload(sys.modules['core.preprocessing.xml_parser'])
                    
                    if 'core.preprocessing.corpus_loader' in sys.modules:
                        importlib.reload(sys.modules['core.preprocessing.corpus_loader'])
                        
                    # Re-import the function from the reloaded module
                    from core.preprocessing.corpus_loader import load_built_in_corpus
                    
                except Exception as e:
                    print(f"Reload Error: {e}")

                progress_bar = st.sidebar.progress(0)
                status_text = st.sidebar.empty()
                
                def update_progress(val, text):
                    progress_bar.progress(val)
                    status_text.caption(text)

                with st.spinner("Downloading and processing..."):
                    # Get URLs for all selected corpora
                    selected_urls = [built_in_corpora[name] for name in selected_names]
                    
                    result = load_built_in_corpus(
                        selected_names, 
                        selected_urls,
                        progress_callback=update_progress
                    )
                    
                    if result.get('error'):
                        st.error(result['error'])
                    else:
                        # Create combined name
                        combined_name = " + ".join(selected_names)
                        
                        if not get_state('comparison_mode'):
                            set_state('current_corpus_path', result['db_path'])
                            set_state('corpus_stats', result['stats'])
                            set_state('current_corpus_name', combined_name)
                            set_state('xml_structure_data', result.get('structure'))
                            
                            # Infer Language from first corpus
                            if "ID-" in selected_names[0] or "Indonesian" in selected_names[0]:
                                set_state('target_lang', 'ID')
                            else:
                                set_state('target_lang', 'EN')
                        else:
                            if not get_state('current_corpus_path'):
                                set_state('current_corpus_path', result['db_path'])
                                set_state('corpus_stats', result['stats'])
                                set_state('current_corpus_name', combined_name)
                                set_state('xml_structure_data', result.get('structure'))
                                if "ID-" in selected_names[0] or "Indonesian" in selected_names[0]: 
                                    set_state('target_lang', 'ID')
                                else: 
                                    set_state('target_lang', 'EN')
                            else:
                                set_state('comp_corpus_path', result['db_path'])
                                set_state('comp_corpus_stats', result['stats'])
                                set_state('comp_corpus_name', combined_name)
                                set_state('comp_xml_structure_data', result.get('structure'))
                        
                        st.success("Built-in Corpus Loaded!")
                        st.rerun()

    # 3. Current Status Info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Active Corpus")
    if current_path:
        st.sidebar.success(f"Primary: **{get_state('current_corpus_name')}**")
        
    comp_path = get_state('comp_corpus_path')
    if get_state('comparison_mode') and comp_path:
        st.sidebar.info(f"Comparison: **{get_state('comp_corpus_name')}**")
    elif get_state('comparison_mode'):
        st.sidebar.warning("Load a 2nd corpus to compare.")
    
    if not current_path and not comp_path:
        st.sidebar.warning("No Corpus Loaded")
        
    st.sidebar.markdown("---")
    
    # 4. AI Interpretation Settings
    st.sidebar.title("AI Interpretation")
    
    # AI Provider Selection
    ai_provider = st.sidebar.radio("AI Provider", ["Ollama", "Gemini"], 
                                   index=0 if get_state('ai_provider') == "Ollama" else 1,
                                   key="sidebar_ai_provider")
    set_state('ai_provider', ai_provider)

    if ai_provider == "Gemini":
        gemini_key = st.sidebar.text_input("Gemini API Key", value=get_state('gemini_api_key', ''), type="password", key="sidebar_gemini_key")
        set_state('gemini_api_key', gemini_key)
        st.sidebar.caption("Google Gemini 1.5 Flash (Cloud fallback)")
    else:
        # Connection Check Button (Always Visible)
        if st.sidebar.button("Check Local AI Status"):
            from core.ai_service import test_ollama_connection
            current_url = get_state('ollama_url')
            success, msg = test_ollama_connection(current_url)
            if success: st.sidebar.success(msg)
            else: st.sidebar.error(msg)
                
        with st.sidebar.expander("Local AI Settings", expanded=False):
            o_url = st.text_input("Ollama URL", value=get_state('ollama_url'), key="sidebar_ollama_url")
            from core.ai_service import get_available_models
            available_models = get_available_models(o_url)
            current_model = get_state('ai_model')
            if available_models:
                if current_model not in available_models: available_models.append(current_model)
                index = available_models.index(current_model) if current_model in available_models else 0
                o_model = st.selectbox("Ollama Model", available_models, index=index, key="sidebar_ai_model_select")
            else:
                o_model = st.text_input("Model (Manual)", value=current_model, key="sidebar_ai_model")
            
            if o_url != get_state('ollama_url'): set_state('ollama_url', o_url)
            if o_model != get_state('ai_model'): set_state('ai_model', o_model)

    st.sidebar.markdown("---")
    
    # 5. CORTEX Assistant (App Usage Chat)
    st.sidebar.title("🧠 CORTEX Assistant")
    st.sidebar.caption("Ask how to use the app or about corpus linguistics.")
    
    chat_hist = get_state('sidebar_chat_history', [])
    with st.sidebar.container(height=250):
        for msg in chat_hist:
            with st.chat_message("user" if "user" in msg else "assistant"):
                st.markdown(msg["content"])
    
    if prompt := st.sidebar.chat_input("How do I...?", key="sidebar_chat_input"):
        chat_hist.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            from core.ai_service import app_guide_chat
            response, err = app_guide_chat(
                user_query=prompt, 
                chat_history=[{"user": m["content"], "ai": chat_hist[i+1]["content"]} for i, m in enumerate(chat_hist[:-1]) if m["role"] == "user"],
                api_key=get_state('gemini_api_key') if get_state('ai_provider') == "Gemini" else None
            )
            if response:
                chat_hist.append({"role": "assistant", "content": response})
            else:
                chat_hist.append({"role": "assistant", "content": f"Sorry, I encountered an error: {err}"})
        set_state('sidebar_chat_history', chat_hist)
        st.rerun()

    return view
