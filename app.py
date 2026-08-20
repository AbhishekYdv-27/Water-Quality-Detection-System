import streamlit as st
from pathlib import Path

from pages.Home import render_home
from pages.Prediction import render_prediction
from pages.Dashboard import render_dashboard
from pages.About import render_about
from utils.helper import load_css, get_project_root

st.set_page_config(page_title="AI Water Quality Monitoring", page_icon="💧", layout="wide")

load_css()

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = True

st.sidebar.title("🌊 Water Quality AI")
st.sidebar.caption("Predict potable water quality with machine learning")

st.session_state.theme_mode = st.sidebar.toggle("🌗 Dark mode", value=st.session_state.theme_mode, key="theme_toggle")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Prediction", "Dashboard", "About"],
    index=0,
    label_visibility="visible",
)

if page == "Home":
    render_home()
elif page == "Prediction":
    render_prediction()
elif page == "Dashboard":
    render_dashboard()
else:
    render_about()
