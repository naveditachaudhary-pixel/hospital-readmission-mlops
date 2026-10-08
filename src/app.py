import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ── Page Configuration ─────────────────────────────────────────────
LOGO_PATH = "assets/logo.png"

st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean healthcare UI
st.markdown("""
<style>
    .header-container {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 1.25rem;
    }
    .header-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
    }
    .header-sub {
        font-size: 0.95rem;
        color: #64748B;
        margin-top: 2px;
    }
    .risk-banner-low {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        text-align: center;
    }
    .risk-banner-moderate {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        text-align: center;
    }
    .risk-banner-high {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        text-align: center;
    }
    .protocol-card {
        background: #F0F9FF;
        border-left: 4px solid #0284C7;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model & Data ─────────────────────────────────────────────
@st.cache_resource
def load_champion_model():
    """Load model from fallback joblib artifact or MLflow registry."""
    fallback_path = "artifacts/champion_model.joblib"
    if os.path.exists(fallback_path):
        try:
            data = joblib.load(fallback_path)
            return data["model"], data.get("model_name", "Champion Model"), data.get("metrics", {})
        except Exception:
            pass

    # Try MLflow Registry
    try:
        import mlflow.sklearn
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        model = mlflow.sklearn.load_model("models:/HospitalReadmissionChampion/latest")
        return model, "HospitalReadmissionChampion", {}
    except Exception:
        return None, "None", {}

@st.cache_data
def load_dataset():
    if os.path.exists("data/processed/diabetic_data_clean.csv"):
        return pd.read_csv("data/processed/diabetic_data_clean.csv")
    return None

model, model_name, model_metrics = load_champion_model()
df_clean = load_dataset()

if model is None or df_clean is None:
    st.error("Model or clean dataset missing. Run `python src/data_pipeline.py && python src/train.py` first.")
    st.stop()

# Sidebar Info
st.sidebar.markdown("### Hospital Readmission MLOps")
st.sidebar.success(f"**Active Model:** {model_name}")
if "auc_roc" in model_metrics:
    st.sidebar.metric("Validation AUC-ROC", f"{model_metrics['auc_roc']:.4f}")
    st.sidebar.metric("Validation PR-AUC", f"{model_metrics['pr_auc']:.4f}")
st.sidebar.info(f"**Dataset Size:** {len(df_clean):,} clinical records")

# Header with Logo
head_col1, head_col2 = st.columns([1, 11])
with head_col1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=64)
with head_col2:
    st.markdown('<h1 class="header-title">Hospital Readmission Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-sub">Predicting 30-day early hospital readmissions for diabetic patients</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Patient Risk Scoring", "Model Benchmarks", "Pipeline Architecture"])

# ── TAB 1: Patient Risk Assessment ───────────────────────────────
with tab1:
    col_mode, col_btn = st.columns([3, 1])
    
    with col_mode:
        input_mode = st.radio(
            "Select Patient Source:",
            ["Sample Real Patient from Dataset", "Customize Patient Profile"],
            horizontal=True
        )

    if input_mode == "Sample Real Patient from Dataset":
        with col_btn:
            st.write("") # Spacing
            if st.button("Sample Random Patient", type="primary", use_container_width=True):
                st.session_state["patient_idx"] = np.random.randint(0, len(df_clean))
                
        p_idx = st.session_state.get("patient_idx", 0)
        patient_data = df_clean.iloc[p_idx].copy()
        X_sample = pd.DataFrame([patient_data.drop("readmitted", errors="ignore")])
        y_true = patient_data.get("readmitted", None)

        # Patient Summary Card
        st.markdown(f"#### Clinical Profile — Patient #{p_idx + 1}")
        c1, c2, c3, c4 = st.columns(4)
        
        age_map_rev = {0:"0-10", 1:"10-20", 2:"20-30", 3:"30-40", 4:"40-50", 5:"50-60", 6:"60-70", 7:"70-80", 8:"80-90", 9:"90-100"}
        age_str = age_map_rev.get(int(patient_data.get("age", 6)), "60-70")

        with c1:
            st.markdown(f"**Demographics**\n* Age Group: `{age_str} yrs`\n* Hospital Stay: `{int(patient_data.get('time_in_hospital', 3))} days`")
        with c2:
            st.markdown(f"**Prior Encounters (Past Year)**\n* Outpatient: `{int(patient_data.get('number_outpatient', 0))}`\n* Emergency: `{int(patient_data.get('number_emergency', 0))}`\n* Inpatient: `{int(patient_data.get('number_inpatient', 0))}`")
        with c3:
            st.markdown(f"**Clinical Interventions**\n* Lab Tests: `{int(patient_data.get('num_lab_procedures', 40))}`\n* Prescriptions: `{int(patient_data.get('num_medications', 15))}`\n* Med Changes: `{int(patient_data.get('num_med_changes', 0))}`")
        with c4:
            hr_discharge = "High Risk (SNF/Home Care)" if patient_data.get("is_high_risk_discharge", 0) == 1 else "Standard Discharge"
            st.markdown(f"**Care & Discharge**\n* Diagnoses: `{int(patient_data.get('number_diagnoses', 7))}`\n* Discharge: `{hr_discharge}`")

    else:
        st.markdown("#### Customize Patient Parameters")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age_val = st.selectbox("Age Group (years)", [0,1,2,3,4,5,6,7,8,9], index=6, format_func=lambda x: f"{x*10}-{x*10+10} yrs")
            time_in_hospital = st.slider("Hospital Stay (Days)", 1, 14, 4)
            num_lab_procedures = st.slider("Lab Tests Count", 1, 130, 45)
            num_procedures = st.slider("Diagnostic Procedures", 0, 10, 1)

        with col2:
            num_medications = st.slider("Prescribed Medications", 1, 80, 15)
            number_outpatient = st.number_input("Prior Outpatient Visits", 0, 20, 0)
            number_emergency = st.number_input("Prior Emergency Visits", 0, 20, 0)
            number_inpatient = st.number_input("Prior Inpatient Stays", 0, 20, 1)

        with col3:
            number_diagnoses = st.number_input("Total Diagnoses Count", 1, 16, 7)
            diag_1_cat = st.selectbox("Primary Diagnosis Category", [0,1,2,3,4,5,6,7,8], index=0, format_func=lambda x: ["Circulatory", "Diabetes", "Digestive", "Genitourinary", "Neoplasms", "Injury", "Musculoskeletal", "Respiratory", "Other"][x])
            is_high_risk_discharge = st.selectbox("Discharge Destination", [0, 1], format_func=lambda x: "High Risk (SNF / Hospice / Home Health)" if x==1 else "Standard Routine Discharge")
            num_med_changes = st.slider("Medication Dosage Adjustments", 0, 5, 1)

        # Build feature vector
        base = df_clean.drop(columns=["readmitted"], errors="ignore").iloc[0].copy()
        base["age"] = age_val
        base["time_in_hospital"] = time_in_hospital
        base["num_lab_procedures"] = num_lab_procedures
        base["num_procedures"] = num_procedures
        base["num_medications"] = num_medications
        base["number_outpatient"] = number_outpatient
        base["number_emergency"] = number_emergency
        base["number_inpatient"] = number_inpatient
        base["number_diagnoses"] = number_diagnoses
        base["is_high_risk_discharge"] = is_high_risk_discharge
        base["num_med_changes"] = num_med_changes
        base["diag_1_cat"] = diag_1_cat

        # Derived features
        serv = number_outpatient + number_emergency + number_inpatient
        base["service_utilization"] = serv
        base["emergency_ratio"] = number_emergency / (serv + 1)
        base["inpatient_ratio"] = number_inpatient / (serv + 1)
        base["med_per_day"] = num_medications / (time_in_hospital + 1e-5)
        base["lab_per_day"] = num_lab_procedures / (time_in_hospital + 1e-5)
        base["treatment_intensity"] = num_procedures + num_medications + num_lab_procedures

        X_sample = pd.DataFrame([base])
        y_true = None

    st.markdown("---")

    # Predict Probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_sample)[0][1]
    else:
        proba = float(model.predict(X_sample)[0])

    risk_pct = proba * 100
    pred = 1 if proba >= 0.5 else 0

    if risk_pct < 30.0:
        risk_tier = "LOW RISK"
        banner_class = "risk-banner-low"
    elif risk_pct < 60.0:
        risk_tier = "MODERATE RISK"
        banner_class = "risk-banner-moderate"
    else:
        risk_tier = "HIGH RISK"
        banner_class = "risk-banner-high"

    # Risk Assessment Output Section
    st.markdown("### Risk Assessment Results")
    
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Readmission Probability", f"{risk_pct:.1f}%")
    with r2:
        st.markdown(f'<div class="{banner_class}">{risk_tier}</div>', unsafe_allow_html=True)
    with r3:
        st.metric("Prediction", "Readmit <30 Days" if pred == 1 else "No Early Readmit")
    with r4:
        if y_true is not None:
            gt_text = "Readmitted" if y_true == 1 else "Not Readmitted"
            gt_delta = "Match" if pred == y_true else "Misclassified"
            st.metric("Ground Truth", gt_text, delta=gt_delta)
        else:
            st.metric("Ground Truth", "Custom Input")

    # Progress bar indicator
    st.progress(min(max(int(risk_pct), 0), 100))

    # Actionable Clinical Protocol Card
    st.markdown('<div class="protocol-card">', unsafe_allow_html=True)
    st.markdown("#### Recommended Clinical Protocol:")
    if risk_tier == "HIGH RISK":
        st.markdown("""
        * **Mandatory 48-Hour Nurse Phone Call:** Contact patient within 48 hours post-discharge.
        * **7-Day Primary Care / Endocrinology Visit:** Schedule follow-up appointment within 7 days.
        * **Medication Reconciliation:** Complete pharmacy review for insulin & oral diabetes med compliance.
        * **Glucose Log & CGM Tracking:** Provide continuous blood glucose monitoring instructions.
        """)
    elif risk_tier == "MODERATE RISK":
        st.markdown("""
        * **14-Day Outpatient Follow-up:** Schedule primary care appointment within 14 days.
        * **Telehealth Check-in:** Conduct routine 7-day post-discharge phone call.
        * **Diabetes Care Plan:** Review hyperglycemia/hypoglycemia warning symptoms with patient.
        """)
    else:
        st.markdown("""
        * **Standard Discharge Instructions:** Provide standard written discharge summaries and clinic contact numbers.
        * **Routine 30-Day Appointment:** Schedule regular follow-up visit within 30 days.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Feature Contribution Breakdown
    st.markdown("### Key Clinical Drivers (Feature Importance)")
    try:
        feat_labels = {
            "inpatient_ratio": "Prior Inpatient Stays Ratio",
            "number_inpatient": "Inpatient Stays (Past Year)",
            "number_emergency": "Emergency Visits (Past Year)",
            "service_utilization": "Total Prior Encounters",
            "treatment_intensity": "Treatment Intensity Score",
            "num_medications": "Prescribed Medications Count",
            "num_lab_procedures": "Lab Procedures Count",
            "time_in_hospital": "Hospital Stay Duration",
            "is_high_risk_discharge": "Discharge to Nursing/Home Health",
            "num_med_changes": "Medication Dosage Adjustments",
            "age": "Patient Age Group"
        }
        
        cols_to_show = [c for c in feat_labels if c in X_sample.columns]
        val_sample = X_sample[cols_to_show].iloc[0]
        val_mean = df_clean[cols_to_show].mean()
        val_std = df_clean[cols_to_show].std() + 1e-5
        
        scores = (val_sample - val_mean) / val_std
        top_indices = np.argsort(np.abs(scores))[-6:]
        
        display_names = [feat_labels[cols_to_show[i]] for i in top_indices]
        display_scores = [scores.iloc[i] for i in top_indices]
        
        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = ["#EF4444" if s > 0 else "#10B981" for s in display_scores]
        ax.barh(display_names, display_scores, color=colors, height=0.55)
        ax.axvline(0, color="#64748B", linestyle="--", alpha=0.7)
        ax.set_xlabel("Impact on Readmission Risk Score (Standard Deviations vs Population Mean)")
        ax.set_title("Top Risk Drivers for This Patient", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.info("Feature impact visualization preview.")

# ── TAB 2: Model Performance Benchmarks ───────────────────────────
with tab2:
    st.subheader("Model Benchmark Comparison")
    
    benchmarks = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "HistGradientBoosting"],
        "Validation AUC-ROC": [0.6543, 0.6694, 0.6804],
        "Validation PR-AUC": [0.2006, 0.2144, 0.2293],
        "Macro F1": [0.4783, 0.5196, 0.4780],
        "CV AUC Mean": [0.6416, 0.6545, 0.6659],
        "Status": ["Baseline", "Challenger", "Champion Registered"]
    })
    st.dataframe(benchmarks, use_container_width=True)

    # Artifact Screenshots
    st.markdown("### Artifact Previews")
    a1, a2 = st.columns(2)
    with a1:
        cm_path = "artifacts/HistGradientBoosting_confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption="HistGradientBoosting — Confusion Matrix")
    with a2:
        curve_path = "artifacts/HistGradientBoosting_performance_curves.png"
        if os.path.exists(curve_path):
            st.image(curve_path, caption="HistGradientBoosting — ROC & PR Curves")

# ── TAB 3: Pipeline Architecture ─────────────────────────────────
with tab3:
    st.subheader("System Design & Data Lineage")
    st.markdown("""
    ```text
    ┌───────────────────────────────┐
    │ UCI Diabetes Dataset (101.7k) │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │ src/data_pipeline.py          │ (ICD-9 Mapping, Clinical Feature Eng.)
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │ src/train.py                  │ (Stratified CV, MLflow SQLite Tracking)
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │ MLflow Model Registry         │ (HospitalReadmissionChampion)
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │ Streamlit Clinical App        │ (Real-Time Inference & Risk Assessment)
    └───────────────┬───────────────┘
    ```
    """)
