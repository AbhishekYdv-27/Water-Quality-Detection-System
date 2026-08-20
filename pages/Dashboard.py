import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc

from utils.helper import DATA_PATH, FEATURE_COLUMNS, MODEL_PATH, load_model_bundle
from utils.visualization import plot_correlation, plot_distribution, plot_feature_importance, plot_missing_values


def render_dashboard():
    st.markdown("""
    <div class="hero-card">
        <h1 style="margin-bottom:0.2rem;">📊 Analytics Dashboard</h1>
        <p style="margin-top:0.3rem;">Explore the data, evaluate model performance, and inspect water-quality trends.</p>
    </div>
    """, unsafe_allow_html=True)
    if not DATA_PATH.exists():
        st.error("Dataset not found. Please add the water potability CSV to the data folder.")
        return

    data = pd.read_csv(DATA_PATH)
    st.markdown("""
    <div class="section-card">
        <h3>Dataset Overview</h3>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", data.shape[0])
    col2.metric("Columns", data.shape[1])
    col3.metric("Potable", int(data["Potability"].sum()))
    col4.metric("Non-Potable", int((data["Potability"] == 0).sum()))

    st.markdown("""
    <div class="section-card">
        <h3>Missing Values</h3>
    </div>
    """, unsafe_allow_html=True)
    missing_plot = plot_missing_values(data)
    if missing_plot is not None:
        st.plotly_chart(missing_plot, use_container_width=True)
    else:
        st.success("No missing values found in the dataset.")

    st.markdown("""
    <div class="section-card">
        <h3>Correlation Heatmap</h3>
    </div>
    """, unsafe_allow_html=True)
    fig_corr = plot_correlation(data)
    st.pyplot(fig_corr)

    st.markdown("""
    <div class="section-card">
        <h3>Feature Distributions</h3>
    </div>
    """, unsafe_allow_html=True)
    selected_feature = st.selectbox("Choose a feature", FEATURE_COLUMNS)
    dist_plot = plot_distribution(data, selected_feature)
    st.plotly_chart(dist_plot, use_container_width=True)

    st.markdown("""
    <div class="section-card">
        <h3>Model Accuracy Comparison</h3>
    </div>
    """, unsafe_allow_html=True)
    bundle = load_model_bundle()
    results = bundle["results"]
    comparison_df = pd.DataFrame([(name, values["accuracy"], values["f1"], values["roc_auc"]) for name, values in results.items()], columns=["Model", "Accuracy", "F1", "ROC AUC"])
    st.dataframe(comparison_df, use_container_width=True)
    fig_acc = px.bar(comparison_df, x="Model", y="Accuracy", color="Model", title="Accuracy by Model")
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("""
    <div class="section-card">
        <h3>Feature Importance</h3>
    </div>
    """, unsafe_allow_html=True)
    if MODEL_PATH.exists():
        model = bundle["model"]
        if hasattr(model, "feature_importances_"):
            st.plotly_chart(plot_feature_importance(model, FEATURE_COLUMNS), use_container_width=True)
        else:
            st.info("Feature importance is not available for the selected model.")

    st.markdown("""
    <div class="section-card">
        <h3>ROC Curve and Confusion Matrix</h3>
    </div>
    """, unsafe_allow_html=True)
    if MODEL_PATH.exists():
        from sklearn.metrics import roc_curve, auc
        from sklearn.model_selection import train_test_split
        from utils.preprocessing import prepare_data

        X_train, X_test, y_train, y_test = prepare_data(data)
        model = bundle["model"]
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        fig_roc = px.area(
            x=fpr,
            y=tpr,
            title=f"ROC Curve (AUC={roc_auc:.3f})",
            labels={"x": "False Positive Rate", "y": "True Positive Rate"},
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        cm = confusion_matrix(y_test, model.predict(X_test))
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df)
