import streamlit as st
import os
import shutil
from ui_streamlit.state_manager import set_state, get_state, reset_tool_states
from ui_streamlit.utils import notify_timing
from core.preprocessing.corpus_loader import load_monolingual_corpus_files, load_built_in_corpus
from core.modules.overview import calculate_corpus_statistics
from core.config import get_available_corpora, BUILT_IN_CORPUS_DETAILS, STANZA_LANG_MAP

def render_sidebar():
    """
    Renders the sidebar for corpus selection and settings.
    Returns: The selected view name.
    """

    # 1. Navigation (Tools) - MOVED TO TOP
    logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "cortex_logo.png")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=150)
    st.sidebar.title("CORTEX")
    st.sidebar.caption("App Version: v150626")
    
    # Environment detection
    import socket
    is_local = True
    if os.getenv("STREAMLIT_SHARING_AUTHOR") or os.getenv("IS_SERVER"):
        is_local = False
    else:
        hostname = ""
        try:
            hostname = socket.gethostname().lower()
            if any(h in hostname for h in ['render', 'heroku', 'aws', 'gcp', 'azure', 'kubernetes', 'k8s', 'container', 'server']):
                is_local = False
        except Exception:
            pass
            
        try:
            from streamlit import context
            if hasattr(context, "headers"):
                host = context.headers.get("host", "").lower()
                host_name = host.split(":")[0] if ":" in host else host
                
                is_host_local = False
                if host_name in ["localhost", "127.0.0.1", "::1"]:
                    is_host_local = True
                elif hostname and host_name == hostname:
                    is_host_local = True
                elif host_name.endswith(".local"):
                    is_host_local = True
                else:
                    import ipaddress
                    try:
                        ip = ipaddress.ip_address(host_name)
                        if ip.is_private or ip.is_loopback:
                            is_host_local = True
                    except ValueError:
                        pass
                
                if not is_host_local:
                    is_local = False
        except ImportError:
            pass

    module_list = ["Overview", "Concordance", "N-Gram", "Collocation", "Word Trend", "Word Profiler", "Dictionary", "Keyword", "Distribution", "Statistical Testing", "Summarisation", "Quiz Creation"]
    
    current_module = get_state('current_module', 'Overview')
    if current_module not in module_list:
        current_module = 'Overview'
        
    index = module_list.index(current_module)
    view = st.sidebar.radio("Modules", module_list, index=index)
    
    if view != current_module:
        set_state('current_module', view)
        st.rerun()
    
    st.sidebar.title("AI Interpretation")
    
    # AI Provider Selection
    ai_providers = ["Ollama", "Gemini", "OpenRouter"]
    curr_prov = get_state('ai_provider', 'Ollama')
    prov_idx = ai_providers.index(curr_prov) if curr_prov in ai_providers else 0
    ai_provider = st.sidebar.radio("AI Provider", ai_providers, 
                                   index=prov_idx,
                                   key="sidebar_ai_provider")
    set_state('ai_provider', ai_provider)

    if ai_provider == "OpenRouter":
        openrouter_key = st.sidebar.text_input("OpenRouter API Key", value=get_state('openrouter_api_key', ''), type="password", key="sidebar_openrouter_key")
        
        # Model Selection
        openrouter_models = [
            "google/gemma-4-26b-a4b-it:free",
            "google/gemini-2.5-flash", 
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash-lite", 
            "anthropic/claude-3.5-sonnet", 
            "meta-llama/llama-3.1-8b-instruct", 
            "qwen/qwen-2.5-7b-instruct", 
            "openai/gpt-4o-mini", 
            "Custom Model"
        ]
        current_or_model = get_state('openrouter_model', 'google/gemini-2.5-flash')
        
        if current_or_model in openrouter_models:
            or_index = openrouter_models.index(current_or_model)
        else:
            or_index = openrouter_models.index("Custom Model")
            
        selected_or_option = st.sidebar.selectbox("OpenRouter Model", openrouter_models, index=or_index, key="sidebar_openrouter_model_select")
        
        if selected_or_option == "Custom Model":
            custom_or_model = st.sidebar.text_input("Enter Model Slug", value=current_or_model if current_or_model not in openrouter_models[:-1] else "google/gemini-2.5-flash", key="sidebar_openrouter_custom_model")
            final_or_model = custom_or_model
        else:
            final_or_model = selected_or_option
            
        if openrouter_key != get_state('openrouter_api_key', '') or final_or_model != get_state('openrouter_model', ''):
            set_state('openrouter_connected', False)
            set_state('openrouter_api_key', openrouter_key)
            set_state('openrouter_model', final_or_model)
            
        if get_state('openrouter_connected', False):
            st.sidebar.markdown(f"<div style='font-size:0.9em; color:#4CAF50; font-weight:bold; margin-bottom:10px;'>🟢 Connected: {get_state('openrouter_model')}</div>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown("<div style='font-size:0.9em; color:#FF9800; font-weight:bold; margin-bottom:10px;'>🔴 Not Connected</div>", unsafe_allow_html=True)

        col1, col2 = st.sidebar.columns(2)
        if col1.button("Connect to API", key="openrouter_connect_btn", use_container_width=True):
            if not openrouter_key:
                st.sidebar.error("Please enter an API Key.")
            else:
                with st.spinner("Connecting..."):
                    from core.ai_service import test_openrouter_connection
                    success, msg = test_openrouter_connection(openrouter_key, final_or_model)
                    if success:
                        set_state('openrouter_connected', True)
                        set_state('openrouter_api_key', openrouter_key)
                        set_state('openrouter_model', final_or_model)
                        st.sidebar.success("Connected successfully!")
                    else:
                        set_state('openrouter_connected', False)
                        st.sidebar.error(f"Connection failed: {msg}")
                             
        if col2.button("Test Connection", key="openrouter_test_btn", use_container_width=True):
            if not openrouter_key:
                st.sidebar.error("Please enter an API Key first.")
            else:
                with st.spinner("Testing..."):
                    from core.ai_service import test_openrouter_connection
                    success, msg = test_openrouter_connection(openrouter_key, final_or_model)
                    if success:
                        st.sidebar.success(msg)
                    else:
                        st.sidebar.error(msg)
    elif ai_provider == "Gemini":
        gemini_key = st.sidebar.text_input("Gemini API Key", value=get_state('gemini_api_key', ''), type="password", key="sidebar_gemini_key")
        
        # Model Selection
        gemini_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "Custom Model"]
        current_g_model = get_state('gemini_model', 'gemini-2.5-flash')
        
        # Determine index
        if current_g_model in gemini_models:
            model_index = gemini_models.index(current_g_model)
        else:
            model_index = gemini_models.index("Custom Model")
            
        selected_model_option = st.sidebar.selectbox("Gemini Model", gemini_models, index=model_index, key="sidebar_gemini_model_select")
        
        if selected_model_option == "Custom Model":
            custom_model = st.sidebar.text_input("Enter Model Name", value=current_g_model if current_g_model not in gemini_models[:-1] else "gemini-2.5-flash", key="sidebar_gemini_custom_model")
            final_model = custom_model
        else:
            final_model = selected_model_option
            
        # Reset connected status if key or model changes
        if gemini_key != get_state('gemini_api_key', '') or final_model != get_state('gemini_model', ''):
            set_state('gemini_connected', False)
            set_state('gemini_api_key', gemini_key)
            set_state('gemini_model', final_model)
            
        # Status Display
        if get_state('gemini_connected', False):
            st.sidebar.markdown(f"<div style='font-size:0.9em; color:#4CAF50; font-weight:bold; margin-bottom:10px;'>🟢 Connected: {get_state('gemini_model')}</div>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown("<div style='font-size:0.9em; color:#FF9800; font-weight:bold; margin-bottom:10px;'>🔴 Not Connected</div>", unsafe_allow_html=True)

        col1, col2 = st.sidebar.columns(2)
        if col1.button("Connect to API", key="gemini_connect_btn", use_container_width=True):
            if not gemini_key:
                st.sidebar.error("Please enter an API Key.")
            else:
                with st.spinner("Connecting..."):
                    from core.ai_service import test_gemini_connection
                    success, msg = test_gemini_connection(gemini_key, final_model)
                    if success:
                        set_state('gemini_connected', True)
                        set_state('gemini_api_key', gemini_key)
                        set_state('gemini_model', final_model)
                        st.sidebar.success("Connected successfully!")
                    else:
                        set_state('gemini_connected', False)
                        st.sidebar.error(f"Connection failed: {msg}")
                             
        if col2.button("Test Connection", key="gemini_test_btn", use_container_width=True):
            if not gemini_key:
                st.sidebar.error("Please enter an API Key first.")
            else:
                with st.spinner("Testing..."):
                    from core.ai_service import test_gemini_connection
                    success, msg = test_gemini_connection(gemini_key, final_model)
                    if success:
                        st.sidebar.success(msg)
                    else:
                        st.sidebar.error(msg)
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
            
            # Cache model fetching to avoid network lag on every rerun
            @st.cache_data(ttl=60, show_spinner=False)
            def get_cached_models(url):
                return get_available_models(url)
            
            available_models = get_cached_models(o_url)
            
            col_m1, col_m2 = st.columns([3, 1])
            with col_m2:
                if st.button("🔄", key="btn_refresh_ollama_models", help="Refresh installed Ollama models list"):
                    st.cache_data.clear()
                    st.rerun()

            current_model = get_state('ai_model')
            if available_models:
                if current_model not in available_models: available_models.append(current_model)
                index = available_models.index(current_model) if current_model in available_models else 0
                o_model = st.radio("Ollama Model", available_models, index=index, horizontal=True, key="sidebar_ai_model_select")
            else:
                o_model = st.text_input("Model (Manual)", value=current_model, key="sidebar_ai_model")
            
            if o_url != get_state('ollama_url'): set_state('ollama_url', o_url)
            if o_model != get_state('ai_model'): set_state('ai_model', o_model)

            # Initialize install states
            if 'ollama_install_step' not in st.session_state:
                st.session_state['ollama_install_step'] = None
            if 'ollama_download_thread' not in st.session_state:
                st.session_state['ollama_download_thread'] = None

            # Force reload the installer utility module to prevent Streamlit module caching issues
            import sys
            import importlib
            if 'core.utils.installer' in sys.modules:
                try:
                    importlib.reload(sys.modules['core.utils.installer'])
                except Exception:
                    pass

            st.markdown("---")
            st.markdown("**Local AI Installer**")

            # Show installation status messages if any
            if 'ollama_install_message' in st.session_state and st.session_state['ollama_install_message']:
                msg_type, msg_text = st.session_state['ollama_install_message']
                if msg_type == "info":
                    st.info(msg_text)
                elif msg_type == "error":
                    st.error(msg_text)
                st.session_state['ollama_install_message'] = None

            # Step 1: Initial state
            if st.session_state['ollama_install_step'] is None:
                if st.button("Get Ollama (One-Click)", key="sidebar_install_ollama_btn"):
                    st.session_state['ollama_install_step'] = "confirm"
                    st.rerun()

            # Step 2: Confirmation state
            elif st.session_state['ollama_install_step'] == "confirm":
                st.warning("This will consume 3 GB of your hard drive (including models). Continue?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Continue", key="confirm_yes"):
                        from core.utils.installer import get_ollama_download_url, OllamaDownloadThread, is_ollama_installed
                        import tempfile
                        import platform
                        
                        if is_ollama_installed():
                            st.session_state['ollama_install_message'] = ("info", "Ollama is already installed on your system!")
                            st.session_state['ollama_install_step'] = None
                            st.rerun()
                        
                        url, ext = get_ollama_download_url()
                        if url:
                            temp_dir = tempfile.gettempdir()
                            dest_path = os.path.join(temp_dir, f"OllamaSetup{ext}")
                            
                            # Start background thread
                            thread = OllamaDownloadThread(url, dest_path)
                            thread.start()
                            
                            st.session_state['ollama_download_thread'] = thread
                            st.session_state['ollama_install_step'] = "downloading"
                        else:
                            st.error(f"Unsupported platform: {platform.system()}. Please install manually from https://ollama.com")
                            st.session_state['ollama_install_step'] = None
                        st.rerun()
                with col2:
                    if st.button("No, Cancel", key="confirm_no"):
                        st.session_state['ollama_install_step'] = None
                        st.rerun()

            # Step 3: Downloading state
            elif st.session_state['ollama_install_step'] == "downloading":
                thread = st.session_state['ollama_download_thread']
                if thread is not None:
                    # Show progress bar and status
                    progress_bar = st.progress(thread.progress)
                    st.caption(thread.status)
                    
                    # Show Cancel button
                    if st.button("Cancel Download", key="cancel_download_btn"):
                        thread.cancelled = True
                        st.session_state['ollama_install_step'] = "cancelled"
                        st.rerun()
                    
                    # If thread finished
                    if not thread.is_alive():
                        if thread.completed:
                            from core.utils.installer import run_ollama_installer
                            success, run_err = run_ollama_installer(thread.dest_path)
                            if success:
                                st.success("Installer launched!")
                                if run_err:
                                    st.info(run_err)
                                else:
                                    st.info("Please follow the setup wizard. Once installed, start the Ollama application and click 'Check Local AI Status' above.")
                            else:
                                st.error(run_err)
                            st.session_state['ollama_install_step'] = None
                            st.session_state['ollama_download_thread'] = None
                        elif thread.error:
                            st.error(f"Download failed: {thread.error}")
                            st.session_state['ollama_install_step'] = None
                            st.session_state['ollama_download_thread'] = None
                        st.rerun()
                    else:
                        # Rerun to update progress bar
                        import time
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.session_state['ollama_install_step'] = None
                    st.rerun()

            # Step 4: Cancelled cleanup state
            elif st.session_state['ollama_install_step'] == "cancelled":
                st.info("Download cancelled. Cleaned up temporary files.")
                st.session_state['ollama_install_step'] = None
                st.session_state['ollama_download_thread'] = None
                st.rerun()



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
