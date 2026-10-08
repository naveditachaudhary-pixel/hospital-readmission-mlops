import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import mlflow.xgboost
import mlflow.sklearn
import shap

# ── Page Configuration & CSS Styling ──────────────────────────────
st.set_page_config(
    page_title="Hospital Readmission Risk Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern clinical visual aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-badge-low {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .risk-badge-moderate {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .risk-badge-high {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .recommendation-box {
        background: #EFF6FF;
        border-left: 5px solid #2563EB;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏥 Hospital Readmission Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">MLOps Pipeline Clinical Intelligence & Real-time Risk Assessment</div>', unsafe_allow_html=True)

# ── Load Model & Data ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load model from MLflow Registry or fallback joblib artifact."""
    mlflow.set_tracking_uri("mlruns")
    try:
        model_uri = "models:/HospitalReadmissionChampion/latest"
        model = mlflow.xgboost.load_model(model_uri)
        return model, "MLflow Model Registry (Champion XGBoost)"
    except Exception:
        fallback_path = "artifacts/champion_model.joblib"
        if os.path.exists(fallback_path):
            data = joblib.load(fallback_path)
            return data["model"], f"Local Artifact ({data.get('model_name', 'XGBoost')})"
        else:
            return None, "No Model Found"

@st.cache_data
def load_dataset():
    if os.path.exists("data/processed/diabetic_data_clean.csv"):
        return pd.read_csv("data/processed/diabetic_data_clean.csv")
    return None

@st.cache_data
def load_metadata():
    if os.path.exists("data/processed/feature_metadata.json"):
        with open("data/processed/feature_metadata.json", "r") as f:
            return json.load(f)
    return {}

model, model_source = load_model()
df_clean = load_dataset()
meta = load_metadata()

if model is None or df_clean is None:
    st.error("⚠️ Pipeline artifacts missing! Please run data_pipeline.py and train.py first.")
    st.code("python src/data_pipeline.py && python src/train.py", language="bash")
    st.stop()

# Model Source Status Bar
st.sidebar.markdown("### ⚙️ Pipeline Configuration")
st.sidebar.success(f"**Active Model:** {model_source}")
st.sidebar.info(f"**Dataset Rows:** {len(df_clean):,}")

# ── App Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Real-Time Patient Risk Assessment", "📊 Model Performance Benchmarks", "ℹ️ Clinical Architecture"])

# ── TAB 1: Prediction & Explanation ──────────────────────────────
with tab1:
    st.subheader("Patient Clinical Profile")
    
    col_input_type = st.radio("Select Input Mode:", ["🎲 Random Patient Sample", "🎛️ Interactive Clinical Builder"], horizontal=True)
    
    if col_input_type == "🎲 Random Patient Sample":
        if st.button("Sample Next Patient", type="primary"):
            st.session_state["sample_idx"] = np.random.randint(0, len(df_clean))
            
        sample_idx = st.session_state.get("sample_idx", 0)
        patient_row = df_clean.iloc[[sample_idx]].copy()
        X_sample = patient_row.drop(columns=["readmitted"], errors="ignore")
        y_true = patient_row["readmitted"].values[0] if "readmitted" in patient_row else None

        st.markdown(f"**Viewing Patient #{sample_idx + 1}**")
        st.dataframe(X_sample, use_container_width=True)

    else:
        st.markdown("Customize patient clinical parameters below:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age_val = st.slider("Age Group", 0, 9, 6, help="0: 0-10, 5: 50-60, 6: 60-70, 7: 70-80, etc.")
            time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 4)
            num_lab_procedures = st.number_input("Lab Procedures Count", 1, 130, 45)
            num_procedures = st.number_input("Procedures Count", 0, 10, 1)

        with col2:
            num_medications = st.number_input("Medications Count", 1, 80, 15)
            number_outpatient = st.number_input("Outpatient Visits (past year)", 0, 20, 0)
            number_emergency = st.number_input("Emergency Visits (past year)", 0, 20, 0)
            number_inpatient = st.number_input("Inpatient Visits (past year)", 0, 20, 1)

        with col3:
            number_diagnoses = st.number_input("Number of Diagnoses", 1, 16, 7)
            diag_1_cat = st.selectbox("Primary Diagnosis Category", [0, 1, 2, 3, 4, 5, 6, 7, 8], index=0, help="0: Circulatory, 1: Diabetes, 2: Digestive, 3: Genitourinary, etc.")
            is_high_risk_discharge = st.selectbox("Discharge Destination Risk", [0, 1], format_func=lambda x: "High Risk (SNF/Hospice/Home Health)" if x==1 else "Standard Routine Discharge")
            num_med_changes = st.slider("Medication Dosage Changes", 0, 5, 1)

        # Construct input row matching clean schema
        base_sample = df_clean.drop(columns=["readmitted"]).iloc[0].copy()
        base_sample["age"] = age_val
        base_sample["time_in_hospital"] = time_in_hospital
        base_sample["num_lab_procedures"] = num_lab_procedures
        base_sample["num_procedures"] = num_procedures
        base_sample["num_medications"] = num_medications
        base_sample["number_outpatient"] = number_outpatient
        base_sample["number_emergency"] = number_emergency
        base_sample["number_inpatient"] = number_inpatient
        base_sample["number_diagnoses"] = number_diagnoses
        base_sample["is_high_risk_discharge"] = is_high_risk_discharge
        base_sample["num_med_changes"] = num_med_changes
        
        # Calculate derived clinical features
        service_util = number_outpatient + number_emergency + number_inpatient
        base_sample["service_utilization"] = service_util
        base_sample["emergency_ratio"] = number_emergency / (service_util + 1)
        base_sample["inpatient_ratio"] = number_inpatient / (service_util + 1)
        base_sample["med_per_day"] = num_medications / (time_in_hospital + 1e-5)
        base_sample["lab_per_day"] = num_lab_procedures / (time_in_hospital + 1e-5)
        base_sample["treatment_intensity"] = num_procedures + num_medications + num_lab_procedures

        X_sample = pd.DataFrame([base_sample])
        y_true = None

    # ── Inference Execution ──────────────────────────────────────────
    st.markdown("---")
    if st.button("🚀 Calculate Readmission Risk Score", type="primary", use_container_width=True):
        try:
            # Handle model prediction call
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_sample)[0][1]
            elif hasattr(model, "predict"):
                proba = float(model.predict(X_sample)[0])
            else:
                proba = 0.5

            pred = 1 if proba >= 0.5 else 0
            risk_pct = proba * 100

            # Determine Risk Tier
            if risk_pct < 30.0:
                risk_tier = "Low Risk"
                badge_style = "risk-badge-low"
            elif risk_pct < 60.0:
                risk_tier = "Moderate Risk"
                badge_style = "risk-badge-moderate"
            else:
                risk_tier = "High Risk"
                badge_style = "risk-badge-high"

            # Display Results Cards
            st.markdown("### 📋 Prediction Results & Assessment")
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)

            with res_col1:
                st.metric("Readmission Likelihood", f"{risk_pct:.1f}%")

            with res_col2:
                st.markdown(f"**Risk Tier Category:**")
                st.markdown(f'<span class="{badge_style}">{risk_tier}</span>', unsafe_allow_html=True)

            with res_col3:
                st.metric("Model Decision", "Readmit <30 Days" if pred == 1 else "No Early Readmit")

            with res_col4:
                if y_true is not None:
                    actual_str = "Readmitted" if y_true == 1 else "Not Readmitted"
                    match = "✅ Correct" if pred == y_true else "⚠️ Misclassified"
                    st.metric("Ground Truth", actual_str, delta=match)
                else:
                    st.metric("Ground Truth", "N/A (Custom)")

            # Clinical Decision Recommendations
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.markdown("#### 🩺 Recommended Post-Discharge Clinical Protocol:")
            if risk_tier == "High Risk":
                st.markdown("""
                * 🔴 **Mandatory 48-Hour Phone Check-in:** Schedule nurse follow-up call within 48h of discharge.
                * 👨‍⚕️ **7-Day Outpatient Visit:** Fast-track primary care or endocrinology appointment within 7 days.
                * 💊 **Medication Reconciliation:** Complete pharmacy consultation for diabetes medication adherence.
                * 📊 **Glucose Monitoring Plan:** Provide continuous glucose monitoring (CGM) or daily log instructions.
                """)
            elif risk_tier == "Moderate Risk":
                st.markdown("""
                * 🟡 **14-Day Outpatient Follow-up:** Ensure appointment with primary care provider within 14 days.
                * 📞 **Telehealth Monitoring:** Schedule routine post-discharge check-in at 7 days.
                * 📋 **Patient Education:** Review warning signs of acute hyper/hypoglycemia.
                """)
            else:
                st.markdown("""
                * 🟢 **Standard Discharge Care:** Provide standard written discharge instructions and emergency contacts.
                * 🗓️ **Routine 30-Day Follow-up:** Schedule regular primary care visit within 30 days.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

            # SHAP Explainable AI Breakdown
            st.markdown("### 🔍 Explainable AI (SHAP Feature Drivers)")
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_sample)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                st.pyplot(fig)
            except Exception as e:
                st.info(f"SHAP explanation preview unavailable for this model type ({e}).")

        except Exception as e:
            st.error(f"Error during prediction calculation: {e}")

# ── TAB 2: Model Benchmarks ───────────────────────────────────────
with tab2:
    st.subheader("Model Evaluation & Comparison Metrics")
    
    # Static metric summary comparison table
    metrics_data = {
        "Model Architecture": ["Logistic Regression (Baseline)", "Random Forest (Challenger)", "XGBoost (👑 Champion)"],
        "AUC-ROC": [0.6491, 0.6710, "0.6722"],
        "PR-AUC": [0.2415, 0.2840, "0.2915"],
        "Macro F1": [0.4834, 0.5454, "0.5574"],
        "CV AUC Mean": [0.6352, 0.6551, "0.6611"],
        "Status": ["Baseline", "Challenger", "👑 Champion Registered"]
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

    # Artifact Screenshots Preview
    st.markdown("### Artifact Previews")
    art_col1, art_col2 = st.columns(2)
    with art_col1:
        cm_file = "artifacts/XGBoost_confusion_matrix.png"
        if os.path.exists(cm_file):
            st.image(cm_file, caption="XGBoost Champion — Confusion Matrix")
    with art_col2:
        curve_file = "artifacts/XGBoost_performance_curves.png"
        if os.path.exists(curve_file):
            st.image(curve_file, caption="XGBoost Champion — ROC & PR Curves")

# ── TAB 3: Architecture & MLOps Pipeline ─────────────────────────
with tab3:
    st.subheader("System Architecture & Data Lineage")
    st.markdown("""
    ```text
    ┌─────────────────────────┐
    │ UCI Diabetes Dataset    │ (101,766 Clinical Records)
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ src/data_pipeline.py    │ (ICD-9 Mapping, Imputation, Feature Eng.)
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ src/train.py            │ (Cross-Val, MLflow Tracking, SHAP Logging)
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ MLflow Model Registry   │ (HospitalReadmissionChampion Model)
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │ Streamlit Dashboard UI  │ (Real-Time Inference & SHAP Explanation)
    └─────────────────────────┘
    ```
    """)
