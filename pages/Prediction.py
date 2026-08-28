import streamlit as st
from datetime import datetime

from predict import predict_water_quality
from utils.helper import (
    FEATURE_COLUMNS,
    create_pdf_report,
    load_prediction_history,
    recommendation,
    risk_level,
    save_prediction_history,
    validate_user_inputs,
)


def render_prediction():
    st.markdown("""
    <div class="hero-card">
        <h1 style="margin-bottom:0.2rem;">🔮 Predict Water Potability</h1>
        <p style="margin-top:0.3rem;">Fill in the physicochemical values below to classify the sample instantly.</p>
    </div>
    """, unsafe_allow_html=True)

    sample_values = {
        "ph": 7.0,
        "Hardness": 190.0,
        "Solids": 22000.0,
        "Chloramines": 8.0,
        "Sulfate": 330.0,
        "Conductivity": 450.0,
        "Organic_carbon": 15.0,
        "Trihalomethanes": 100.0,
        "Turbidity": 3.5,
    }
    form_values = st.session_state.get("sample_values", sample_values)
    report_data = None

    with st.form("prediction_form"):
        st.markdown("""
        <div class="section-card">
            <h3>Input Water Parameters</h3>
        </div>
        """, unsafe_allow_html=True)
        values = {}
        for feature in FEATURE_COLUMNS:
            values[feature] = st.number_input(feature, value=float(form_values[feature]), step=0.01, format="%.2f")

        submitted = st.form_submit_button("Predict Water Quality", width="stretch")
        if submitted:
            errors = validate_user_inputs(values)
            if errors:
                for error in errors:
                    st.error(error)
                return

            with st.spinner("Analyzing sample..."):
                prediction, probability = predict_water_quality(values)
            confidence = round(float(probability) * 100, 2)
            pred_label = "Potable" if prediction == 1 else "Non-Potable"
            risk = risk_level(float(probability))
            rec = recommendation(prediction, float(probability))

            st.markdown("""
            <div class="section-card">
                <h3>Prediction Result</h3>
            </div>
            """, unsafe_allow_html=True)
            st.success(f"Prediction: {pred_label}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction Confidence", f"{confidence:.2f}%")
            col2.metric("Risk Level", risk)
            col3.metric("Status", pred_label)
            st.info(rec)

            history_record = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **values,
                "Prediction": pred_label,
                "Confidence": confidence,
                "Risk": risk,
                "Recommendation": rec,
            }
            save_prediction_history(history_record)
            report_data = create_pdf_report({"prediction": pred_label, "confidence": confidence, "risk": risk, "recommendation": rec})

    if report_data is not None:
        st.download_button(
            label="Download PDF Report",
            data=report_data,
            file_name="prediction_report.pdf",
            mime="application/pdf",
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧪 Autofill Sample", width="stretch"):
            st.session_state["sample_values"] = sample_values
            st.rerun()
    with col2:
        if st.button("🔄 Reset Form", width="stretch"):
            st.session_state.pop("sample_values", None)
            st.rerun()

    history = load_prediction_history()
    if not history.empty:
        st.subheader("Prediction History")
        st.dataframe(history.tail(5), width="stretch")
        csv_data = history.to_csv(index=False).encode("utf-8")
        st.download_button("Download History CSV", csv_data, file_name="prediction_history.csv", mime="text/csv")
