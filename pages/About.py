import streamlit as st


def render_about():
    st.markdown("""
    <div class="hero-card">
        <h1 style="margin-bottom:0.2rem;">About the Monitoring System</h1>
        <p style="margin-top:0.3rem; color:#dff5f1;">A practical machine-learning workflow for faster water quality screening.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>What it does</h3>
            <p>The system evaluates nine physicochemical measurements and estimates whether a sample is potable. Each result includes confidence, risk level, guidance, and an exportable report.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Built with</h3>
            <p>Python · Streamlit · Scikit-learn · Pandas · NumPy · Plotly</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <h3>Machine learning workflow</h3>
        <p>1. Clean and prepare the dataset<br>2. Train and compare classifiers<br>3. Select and persist the best model<br>4. Serve predictions through an interactive dashboard</p>
    </div>
    """, unsafe_allow_html=True)
