import sys
import traceback
from pathlib import Path
# pyrefly: ignore [missing-import]
import streamlit as st

# set_page_config must be the absolute first Streamlit command
st.set_page_config(
    page_title="TalentLens AI",
    page_icon=" ",
    layout="wide",
)

try:
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import main_app
except Exception as e:
    print("--------------------------------------------------")
    print("CRITICAL: APP STARTUP CRASH DETECTED!")
    print(traceback.format_exc())
    print("--------------------------------------------------")
    st.error("### 🔴 App Startup Crashed")
    st.markdown("The application failed to initialize on the server. Below is the traceback:")
    st.code(traceback.format_exc(), language="python")
    st.stop()
