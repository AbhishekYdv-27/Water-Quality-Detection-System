import streamlit as st


def render_about():
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        This project demonstrates an end-to-end machine learning workflow for water potability prediction.
        It combines data preprocessing, training, evaluation, and an interactive Streamlit dashboard into one system.
        """
    )

    st.subheader("Technologies Used")
    st.write("- Python")
    st.write("- Streamlit")
    st.write("- Scikit-learn")
    st.write("- Pandas, NumPy")
    st.write("- Plotly, Matplotlib, Seaborn")
    st.write("- Joblib")

    st.subheader("Machine Learning Workflow")
    st.write("1. Load and clean the dataset")
    st.write("2. Handle missing values")
    st.write("3. Train several classifiers")
    st.write("4. Tune hyperparameters with GridSearchCV")
    st.write("5. Select the best model and save it")

    st.subheader("Author")
    st.write("Built as a complete AI-powered water quality monitoring system for academic and demonstration purposes.")
