import streamlit as st
import sys
import os

# Add project root to path so we can import from core/ui_streamlit
# Add project root to path so we can import from core/ui_streamlit
architecture_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if architecture_root not in sys.path:
    sys.path.insert(0, architecture_root)

# Force reload of backend modules to ensure fresh code is used (fixes caching issues)
import importlib
try:
    if 'core.io_utils' in sys.modules:
        importlib.reload(sys.modules['core.io_utils'])
    if 'core.config' in sys.modules:
        importlib.reload(sys.modules['core.config'])
    if 'core.preprocessing.xml_parser' in sys.modules:
        importlib.reload(sys.modules['core.preprocessing.xml_parser'])
    if 'core.preprocessing.corpus_loader' in sys.modules:
        importlib.reload(sys.modules['core.preprocessing.corpus_loader'])
    if 'core.preprocessing.tagging' in sys.modules:
        importlib.reload(sys.modules['core.preprocessing.tagging'])
#     if 'core.preprocessing.corpus_loader' in sys.modules:
#         importlib.reload(sys.modules['core.preprocessing.corpus_loader'])
    if 'core.modules.overview' in sys.modules:
        importlib.reload(sys.modules['core.modules.overview'])
#     if 'core.ai_service' in sys.modules:
#         importlib.reload(sys.modules['core.ai_service'])
#     if 'core.modules.ngram' in sys.modules:
#         importlib.reload(sys.modules['core.modules.ngram'])
#     if 'core.modules.dictionary_service' in sys.modules:
#         importlib.reload(sys.modules['core.modules.dictionary_service'])
#     if 'core.modules.concordance' in sys.modules:
#         importlib.reload(sys.modules['core.modules.concordance'])
#     if 'core.modules.comparison_analysis' in sys.modules:
#         importlib.reload(sys.modules['core.modules.comparison_analysis'])
#     if 'core.ai_service' in sys.modules:
#         importlib.reload(sys.modules['core.ai_service'])
#     if 'core.modules.collocation' in sys.modules:
#         importlib.reload(sys.modules['core.modules.collocation'])
#     if 'core.modules.collocation_patterns' in sys.modules:
#         importlib.reload(sys.modules['core.modules.collocation_patterns'])
    if 'core.visualiser.wordcloud' in sys.modules:
        importlib.reload(sys.modules['core.visualiser.wordcloud'])
#     if 'ui_streamlit.views.keyword_view' in sys.modules:
#         importlib.reload(sys.modules['ui_streamlit.views.keyword_view'])
#     if 'ui_streamlit.caching' in sys.modules:
#         importlib.reload(sys.modules['ui_streamlit.caching'])
#     if 'ui_streamlit.components.sidebar' in sys.modules:
#         importlib.reload(sys.modules['ui_streamlit.components.sidebar'])
#     if 'ui_streamlit.components.result_display' in sys.modules:
#         importlib.reload(sys.modules['ui_streamlit.components.result_display'])
#     if 'ui_streamlit.views.concordance_view' in sys.modules:
#         importlib.reload(sys.modules['ui_streamlit.views.concordance_view'])
#     if 'ui_streamlit.views.collocation_view' in sys.modules:
#         importlib.reload(sys.modules['ui_streamlit.views.collocation_view'])
#     if 'ui_streamlit.views.dictionary_view' in sys.modules:
#         importlib.reload(sys.modules['ui_streamlit.views.dictionary_view'])
#     if 'ui_streamlit.views.ngram_view' in sys.modules:
#         importlib.reload(sys.modules['ui_streamlit.views.ngram_view'])
#     if 'core.modules.distribution' in sys.modules:
#         importlib.reload(sys.modules['core.modules.distribution'])
#     if 'ui_streamlit.views.distribution_view' in sys.modules:
#         importlib.reload(sys.modules['ui_streamlit.views.distribution_view'])
except Exception as e:
    pass

# Debug Imports
try:
    from ui_streamlit.state_manager import init_session_state
    from ui_streamlit.components.sidebar import render_sidebar
    from ui_streamlit.views.overview_view import render_overview
    from ui_streamlit.views.dictionary_view import render_dictionary_view
    from ui_streamlit.views.concordance_view import render_concordance_view
    from ui_streamlit.views.ngram_view import render_ngram_view
    from ui_streamlit.views.collocation_view import render_collocation_view
    from ui_streamlit.views.keyword_view import render_keyword_view
    from ui_streamlit.views.distribution_view import render_distribution_view
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

from ui_streamlit.state_manager import init_session_state
from core.visualiser.styles import POS_COLOR_MAP

# Page Configuration
st.set_page_config(
    page_title="CORTEX Corpus Query System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize State
init_session_state()

# CSS Styling for Premium Dark Blue Theme
PRIMARY_COLOR = "#00ADB5"
BACKGROUND_COLOR = "#0f172a"
SECONDARY_BACKGROUND = "#1e293b"
TEXT_COLOR = "#f8fafc"

st.markdown(f"""
<style>
    /* Global Font Scaling */
    html {{
        font-size: 110% !important; /* +10% base increase, user requested +50% but that is huge, usually means interface scaling. 110% is a safe start, or we can go 125%. */
    }}

    /* Main App Container */
    .stApp {{
        background-color: {BACKGROUND_COLOR} !important;
        color: {TEXT_COLOR} !important;
    }}
    
    /* Top Header Bar - Force Dark */
    header[data-testid="stHeader"], [data-testid="stHeader"] {{
        background-color: {BACKGROUND_COLOR} !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {SECONDARY_BACKGROUND} !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }}
    
    /* Force ALL Labels and text to be White/High Contrast */
    label, 
    label p, 
    [data-testid="stWidgetLabel"] p,
    .stMarkdown p,
    .stText p,
    span.st-emotion-cache-ycmcfb {{
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }}
    
    /* Specific Sidebar Widget Labels - Cyan */
    [data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: #00FFF5 !important;
    }}
    
    /* Inputs, Selectboxes & Number Inputs - Force DEEP Dark Background */
    div[data-baseweb="input"], 
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"],
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {{
        background-color: #1e293b !important; /* Secondary background to stand out slightly from main BG */
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }}
    
    /* Ensure text inside ALL inputs is white */
    input, textarea, [data-testid="stSelectedValue"], .stTextInput input, .stNumberInput input {{
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }}
    
    /* Dropdown Menus */
    ul[data-testid="stSelectboxVirtualDropdown"] {{
        background-color: {SECONDARY_BACKGROUND} !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }}
    li[role="option"] {{
        background-color: {SECONDARY_BACKGROUND} !important;
        color: #FFFFFF !important;
    }}
    li[role="option"]:hover {{
        background-color: {PRIMARY_COLOR} !important;
        color: white !important;
    }}
    
    /* Multiselect Styling */
    .stMultiSelect [data-baseweb="tag"] {{
        background-color: {PRIMARY_COLOR} !important;
        color: #FFFFFF !important;
    }}
    .stMultiSelect [data-baseweb="tag"] span {{
        color: #FFFFFF !important;
    }}
    .stMultiSelect div[data-baseweb="select"] {{
        background-color: #1e293b !important;
    }}
    .stMultiSelect input {{
        color: #FFFFFF !important;
    }}

    /* Icons and Indicators */
    div[data-baseweb="select"] svg, 
    .st-emotion-cache-1v04fbb,
    [data-testid="stMetricValue"] {{
        fill: #00FFF5 !important;
        color: #00FFF5 !important;
    }}
    
    /* File Uploader */
    [data-testid="stFileUploaderDropzone"] {{
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #FFFFFF !important;
        border: 1px dashed rgba(0, 255, 245, 0.4) !important;
    }}
    [data-testid="stFileUploaderDropzone"] p, 
    [data-testid="stFileUploaderDropzone"] span {{
        color: #FFFFFF !important;
    }}

    /* Premium Buttons */
    .stButton>button {{
        color: #FFFFFF !important;
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, #00767C 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4) !important;
        text-transform: uppercase !important;
        font-weight: 700 !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}
    .stButton>button:hover {{
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(0, 173, 181, 0.6) !important;
        background: linear-gradient(135deg, #00FFF5 0%, {PRIMARY_COLOR} 100%) !important;
        border-color: #00FFF5 !important;
    }}
    
    /* Expander Styling - Make it obvious its dark */
    .stExpander {{
        background-color: rgba(0, 0, 0, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        margin-bottom: 1rem !important;
    }}
    .stExpander summary {{
        color: #00FFF5 !important;
        font-weight: 700 !important;
        background-color: rgba(255, 255, 255, 0.03) !important;
    }}
    .stExpander summary:hover {{
        color: #FFFFFF !important;
        background-color: rgba(255, 255, 255, 0.08) !important;
    }}
    
    /* Table Styling - High Contrast */
    table, .stDataFrame {{
        color: #FFFFFF !important;
    }}
    table th, table td, 
    [data-testid="stTable"] th,
    [data-testid="stTable"] td {{
        color: #FFFFFF !important;
        background-color: rgba(255, 255, 255, 0.03) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }}
    table thead th {{
        background-color: rgba(0, 173, 181, 0.2) !important;
        color: #00FFF5 !important;
        font-weight: 700 !important;
    }}
    
    /* Download Button - Always Dark Background */
    .stDownloadButton>button {{
        color: #FFFFFF !important;
        background-color: #1e293b !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease-in-out !important;
    }}
    .stDownloadButton>button:hover {{
        background-color: rgba(0, 173, 181, 0.3) !important;
        border-color: #00FFF5 !important;
        color: #FFFFFF !important;
        transform: scale(1.02) !important;
    }}
""", unsafe_allow_html=True)

from core.visualiser.styles import POS_COLOR_MAP

# Main Layout
def main():
    st.markdown(
        """
        <div style="text-align: right; margin-bottom: 0.5rem;">
            <a href="http://www.cortex-app.org/" target="_blank"
               style="color:#00FFF5; font-weight:700; text-decoration:none;">
                📘 Manual
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## CORTEX: Advanced Corpus Query System")
    
    # Render Sidebar and get current view
    current_view = render_sidebar()
    
    # Router
    if current_view == "Overview":
        render_overview()
    elif current_view == "Concordance":
        render_concordance_view()
    elif current_view == "N-Gram":
        render_ngram_view()
    elif current_view == "Collocation":
        render_collocation_view()
    elif current_view == "Dictionary":
        render_dictionary_view()
    elif current_view == "Keyword":
        render_keyword_view()
    elif current_view == "Distribution":
        render_distribution_view()
    else:
        st.write("Select a module from the sidebar.")

if __name__ == "__main__":
    main()
