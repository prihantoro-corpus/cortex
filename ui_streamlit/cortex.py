import streamlit as st

st.set_page_config(page_title="Redirecting to CORTEX...", layout="centered")

st.warning("🚀 **CORTEX has moved!** For better performance and higher memory limits, this app is now hosted on Hugging Face Spaces.")
st.info("Redirecting you to the new home in 3 seconds... If nothing happens, [**click here**](https://huggingface.co/spaces/prihantoro-corpus/cortex)!")

# Method 1: HTML Meta Refresh (Works natively without JS, affects the iframe)
st.markdown(
    """<meta http-equiv="refresh" content="3; url='https://huggingface.co/spaces/prihantoro-corpus/cortex'" />""",
    unsafe_allow_html=True
)

# Method 2: Javascript redirect (Without parent, avoids cross-origin iframe blocks)
import streamlit.components.v1 as components
components.html(
    '''
    <script>
        setTimeout(function() {
            window.top.location.href = "https://huggingface.co/spaces/prihantoro-corpus/cortex";
        }, 3000);
        
        // Fallback if top navigation is blocked by Streamlit's iframe sandbox
        setTimeout(function() {
            window.location.href = "https://huggingface.co/spaces/prihantoro-corpus/cortex";
        }, 3500);
    </script>
    ''',
    height=0
)
st.stop()
