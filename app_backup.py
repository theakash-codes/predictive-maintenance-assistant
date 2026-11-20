
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Predictive Maintenance Assistant",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Manufacturing Predictive Maintenance Assistant")

st.markdown(
    "AI-powered machine failure prediction, anomaly detection, "
    "failure diagnosis and maintenance decision support."
)

# ==========================================
# PROJECT PATHS
# ==========================================

MODEL_DIR = "models"
REPORT_DIR = "reports"

# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():

    failure_model = joblib.load(
        os.path.join(MODEL_DIR, "optimized_failure_model.pkl")
    )

    metadata = joblib.load(
        os.path.join(MODEL_DIR, "optimized_model_metadata.pkl")
    )

    anomaly_model = joblib.load(
        os.path.join(MODEL_DIR, "isolation_forest_model.pkl")
    )

    diagnosis_data = joblib.load(
        os.path.join(MODEL_DIR, "failure_diagnosis_model.pkl")
    )

    health_config = joblib.load(
        os.path.join(MODEL_DIR, "health_scoring_config.pkl")
    )

    return (
        failure_model,
        metadata,
        anomaly_model,
        diagnosis_data,
        health_config
    )


try:
    (
        failure_model,
        metadata,
        anomaly_model,
        diagnosis_data,
        health_config
    ) = load_models()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# ==========================================
# MODEL CONFIGURATION
# ==========================================

FEATURES = metadata["features"]
FAILURE_THRESHOLD = float(metadata["threshold"])

diagnosis_model = diagnosis_data["model"]
diagnosis_targets = diagnosis_data["targets"]

HEALTH_THRESHOLDS = health_config["health_thresholds"]

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("⚙️ Machine Configuration")

st.sidebar.write("Enter the machine sensor measurements.")

air_temp = st.sidebar.number_input(
    "Air Temperature (K)",
    min_value=250.0,
    max_value=350.0,
    value=300.0,
    step=0.1
)

process_temp = st.sidebar.number_input(
    "Process Temperature (K)",
    min_value=250.0,
    max_value=400.0,
    value=310.0,
    step=0.1
)

rotational_speed = st.sidebar.number_input(
    "Rotational Speed (rpm)",
    min_value=0,
    max_value=5000,
    value=1500
)

torque = st.sidebar.number_input(
    "Torque (Nm)",
    min_value=0.0,
    max_value=150.0,
    value=40.0,
    step=0.1
)

tool_wear = st.sidebar.number_input(
    "Tool Wear (min)",
    min_value=0,
    max_value=500,
    value=100
)

analyze = st.sidebar.button(
    "Analyze Machine",
    type="primary",
    use_container_width=True
)

st.sidebar.divider()

st.sidebar.caption(
    "Dataset: AI4I 2020 Predictive Maintenance Dataset"
)

# ==========================================
# DASHBOARD NAVIGATION
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Dashboard",
    "🔍 Failure Diagnosis",
    "📊 Model Performance",
    "🧠 Explainable AI"
])

# ==========================================
# MACHINE ANALYSIS
# ==========================================

if analyze:

    input_data = pd.DataFrame(
        [[
            air_temp,
            process_temp,
            rotational_speed,
            torque,
            tool_wear
        ]],
        columns=FEATURES
    )

    # --------------------------------------
    # 1. FAILURE PREDICTION
    # --------------------------------------

    failure_probability = float(
        failure_model.predict_proba(input_data)[0][1]
    )

    failure_percentage = failure_probability * 100

    failure_prediction = (
        failure_probability >= FAILURE_THRESHOLD
    )

    # --------------------------------------
    # 2. ANOMALY DETECTION
    # --------------------------------------

    anomaly_prediction = int(
        anomaly_model.predict(input_data)[0]
    )

    anomaly_score = float(
        -anomaly_model.decision_function(input_data)[0]
    )

    reference_scores = np.asarray(
        health_config["anomaly_reference_scores"]
    )

    anomaly_percentile = (
        np.searchsorted(
            reference_scores,
            anomaly_score,
            side="right"
        ) / len(reference_scores)
    ) * 100

    anomaly_status = (
        "Anomaly Detected"
        if anomaly_prediction == -1
        else "Normal Pattern"
    )

    # --------------------------------------
    # 3. MACHINE HEALTH SCORE
    # --------------------------------------

    failure_weight = health_config["failure_weight"]
    anomaly_weight = health_config["anomaly_weight"]

    combined_risk = (
        failure_weight * failure_percentage
        + anomaly_weight * anomaly_percentile
    )

    health_score = float(
        np.clip(100 - combined_risk, 0, 100)
    )

    if health_score >= HEALTH_THRESHOLDS["healthy"]:
        health_status = "Healthy"

    elif health_score >= HEALTH_THRESHOLDS["monitor"]:
        health_status = "Monitor"

    elif health_score >= HEALTH_THRESHOLDS["warning"]:
        health_status = "Warning"

    else:
        health_status = "Critical"

    # --------------------------------------
    # 4. FAILURE MODE DIAGNOSIS
    # --------------------------------------

    diagnosis_probabilities = diagnosis_model.predict_proba(
        input_data
    )

    diagnosis_results = []

    for i, target in enumerate(diagnosis_targets):

        probability = float(
            diagnosis_probabilities[i][0][1]
        )

        diagnosis_results.append({
            "Failure Type": target,
            "Probability (%)": round(probability * 100, 2),
            "Model Assessment": (
                "Possible"
                if probability >= 0.5
                else "Not predicted"
            )
        })

    diagnosis_df = pd.DataFrame(diagnosis_results)

    # ======================================
    # TAB 1: DASHBOARD
    # ======================================

    with tab1:

        st.subheader("Machine Status Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Failure Probability",
            f"{failure_percentage:.2f}%"
        )

        col2.metric(
            "Machine Health",
            f"{health_score:.1f}/100"
        )

        col3.metric(
            "Failure Prediction",
            "Failure" if failure_prediction else "Normal"
        )

        col4.metric(
            "Anomaly Detection",
            anomaly_status
        )

        st.divider()

        st.subheader("Machine Health Assessment")

        st.progress(
            int(round(health_score))
        )

        st.write(f"**Health Status: {health_status}**")

        st.write(
            f"Combined Risk Score: {combined_risk:.2f}%"
        )

        st.caption(
            "Health score is a prototype indicator using the "
            "configured failure-risk and anomaly weights. "
            "It is not a validated industrial safety limit."
        )

        st.divider()

        st.subheader("Failure Prediction Result")

        if failure_prediction:
            st.error(
                "The model predicts a possible machine failure."
            )
        else:
            st.success(
                "The model does not predict machine failure."
            )

        st.write(
            f"Classification threshold: "
            f"{FAILURE_THRESHOLD:.4f}"
        )

        st.divider()

        st.subheader("Sensor Measurements")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Anomaly Detection Details")

        st.write(f"Anomaly status: **{anomaly_status}**")

        st.write(
            f"Anomaly score: {anomaly_score:.5f}"
        )

        st.write(
            f"Training-reference anomaly percentile: "
            f"{anomaly_percentile:.2f}%"
        )

        st.caption(
            "Anomaly percentile measures unusualness relative "
            "to training data. It is not a failure probability."
        )

        st.divider()

        st.subheader("Maintenance Recommendation")

        if failure_prediction and anomaly_prediction == -1:

            st.error(
                "High attention required: inspect the machine "
                "and review its operating conditions."
            )

        elif failure_prediction:

            st.warning(
                "Schedule an inspection and monitor the machine "
                "closely."
            )

        elif anomaly_prediction == -1:

            st.warning(
                "Unusual operating pattern detected. Inspect "
                "sensor readings and machine operating conditions."
            )

        else:

            st.success(
                "No immediate issue identified by the models. "
                "Continue routine monitoring and maintenance."
            )

        # Download report

        report = input_data.copy()

        report["Failure Probability (%)"] = failure_percentage
        report["Failure Prediction"] = (
            "Failure" if failure_prediction else "Normal"
        )
        report["Anomaly Status"] = anomaly_status
        report["Anomaly Percentile (%)"] = anomaly_percentile
        report["Health Score"] = health_score
        report["Health Status"] = health_status

        st.download_button(
            "⬇️ Download Analysis Report",
            data=report.to_csv(index=False),
            file_name="machine_analysis_report.csv",
            mime="text/csv"
        )

    # ======================================
    # TAB 2: FAILURE DIAGNOSIS
    # ======================================

    with tab2:

        st.subheader("Failure Mode Diagnosis")

        st.write(
            "The model estimates the likelihood of five "
            "different machine failure categories."
        )

        st.dataframe(
            diagnosis_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            diagnosis_df.set_index("Failure Type")[
                "Probability (%)"
            ]
        )

        st.warning(
            "TWF and RNF have very few training examples and "
            "showed weak test performance. Treat their estimates "
            "as experimental, not reliable diagnoses."
        )

        st.caption(
            "Failure categories: TWF – Tool Wear Failure, "
            "HDF – Heat Dissipation Failure, "
            "PWF – Power Failure, "
            "OSF – Overstrain Failure, "
            "RNF – Random Failure."
        )

    # ======================================
    # TAB 3: MODEL PERFORMANCE
    # ======================================

    with tab3:

        st.subheader("Optimized XGBoost Model Performance")

        metrics = metadata["metrics"]

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        metric_col2.metric(
            "Precision",
            f"{metrics['precision'] * 100:.2f}%"
        )

        metric_col3.metric(
            "Recall",
            f"{metrics['recall'] * 100:.2f}%"
        )

        metric_col4, metric_col5, metric_col6 = st.columns(3)

        metric_col4.metric(
            "F1 Score",
            f"{metrics['f1_score'] * 100:.2f}%"
        )

        metric_col5.metric(
            "PR-AUC",
            f"{metrics['pr_auc'] * 100:.2f}%"
        )

        metric_col6.metric(
            "ROC-AUC",
            f"{metrics['roc_auc'] * 100:.2f}%"
        )

        st.caption(
            "Metrics are from the held-out test set. "
            "The classification threshold was selected "
            "using validation data."
        )

        st.divider()

        st.subheader("Confusion Matrix")

        confusion_path = os.path.join(
            REPORT_DIR,
            "optimized_confusion_matrix.csv"
        )

        if os.path.exists(confusion_path):

            confusion_df = pd.read_csv(confusion_path)

            st.dataframe(
                confusion_df,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("Confusion matrix file not found.")

        st.subheader("Model Comparison")

        comparison_path = os.path.join(
            REPORT_DIR,
            "optimized_model_comparison.csv"
        )

        if os.path.exists(comparison_path):

            comparison_df = pd.read_csv(comparison_path)

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("Model comparison report not found.")

    # ======================================
    # TAB 4: EXPLAINABLE AI
    # ======================================

    with tab4:

        st.subheader("Explainable AI using SHAP")

        st.write(
            "SHAP identifies which sensor features contribute "
            "most to the trained model's predictions."
        )

        shap_image = os.path.join(
            REPORT_DIR,
            "shap_summary.png"
        )

        if os.path.exists(shap_image):

            st.image(
                shap_image,
                caption="SHAP Summary Plot",
                use_container_width=True
            )

        else:
            st.info("SHAP summary image not found.")

        importance_path = os.path.join(
            REPORT_DIR,
            "shap_feature_importance.csv"
        )

        if os.path.exists(importance_path):

            importance_df = pd.read_csv(importance_path)

            st.subheader("Global Feature Importance")

            st.dataframe(
                importance_df,
                use_container_width=True,
                hide_index=True
            )

        beeswarm_path = os.path.join(
            REPORT_DIR,
            "shap_beeswarm.png"
        )

        if os.path.exists(beeswarm_path):

            st.subheader("SHAP Feature Impact")

            st.image(
                beeswarm_path,
                use_container_width=True
            )

        st.caption(
            "SHAP values explain model behaviour and learned "
            "associations. They do not establish physical causation."
        )

else:

    # ======================================
    # INITIAL DASHBOARD
    # ======================================

    with tab1:

        st.info(
            "Enter machine sensor values in the sidebar and "
            "click Analyze Machine to generate predictions."
        )

        st.subheader("Project Modules")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("""
            **1. Machine Failure Prediction**

            XGBoost-based failure probability estimation.

            **2. Anomaly Detection**

            Isolation Forest-based unusual pattern identification.

            **3. Machine Health Scoring**

            Combined risk-based health indicator.
            """)

        with col2:

            st.markdown("""
            **4. Failure Mode Diagnosis**

            Prediction of five failure categories.

            **5. Explainable AI**

            SHAP-based feature interpretation.

            **6. Maintenance Decision Support**

            Basic recommendations based on model outputs.
            """)

    with tab2:

        st.info("Run machine analysis to view failure diagnosis.")

    with tab3:

        st.subheader("Optimized Model Performance")

        metrics = metadata["metrics"]

        st.write(f"Model: **{metadata['model_name']}**")

        st.write(
            f"Accuracy: **{metrics['accuracy'] * 100:.2f}%**"
        )

        st.write(
            f"Precision: **{metrics['precision'] * 100:.2f}%**"
        )

        st.write(
            f"Recall: **{metrics['recall'] * 100:.2f}%**"
        )

        st.write(
            f"F1 Score: **{metrics['f1_score'] * 100:.2f}%**"
        )

        st.write(
            f"PR-AUC: **{metrics['pr_auc'] * 100:.2f}%**"
        )

        st.write(
            f"ROC-AUC: **{metrics['roc_auc'] * 100:.2f}%**"
        )

    with tab4:

        st.subheader("SHAP Explainability")

        shap_image = os.path.join(
            REPORT_DIR,
            "shap_summary.png"
        )

        if os.path.exists(shap_image):
            st.image(
                shap_image,
                use_container_width=True
            )

st.divider()

st.caption(
    "B.Tech Major Project | AI4I 2020 Predictive Maintenance Dataset"
)
