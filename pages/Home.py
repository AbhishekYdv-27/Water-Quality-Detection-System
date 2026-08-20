import streamlit as st
import pandas as pd

from utils.helper import DATA_PATH, FEATURE_COLUMNS


def render_home():
    st.markdown("""
    <div class="hero-card">
        <h1 style="margin-bottom:0.2rem;">💧 Smart Water Quality Monitoring</h1>
        <p style="margin-top:0.3rem; font-size:1.05rem;">A modern AI-powered assistant for predicting whether water is safe or unsafe for drinking.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <h3>Project Overview</h3>
        <p>This platform uses machine learning to analyze core water parameters such as pH, hardness, chloramines, solids, conductivity, and turbidity. It predicts whether a sample is potable or non-potable with a confidence score and tailored guidance.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>Why it matters</h3>
            <ul>
                <li>Detect unsafe water early and reduce health risk.</li>
                <li>Support treatment decisions with actionable insights.</li>
                <li>Make water quality monitoring faster and more reliable.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Dataset Snapshot</h3>
        </div>
        """, unsafe_allow_html=True)
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])
            st.metric("Potable Samples", int(df["Potability"].sum()))

    st.markdown("""
    <div class="section-card">
        <h3>Included Water Parameters</h3>
        <p>pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity</p>
    </div>
    """, unsafe_allow_html=True)
