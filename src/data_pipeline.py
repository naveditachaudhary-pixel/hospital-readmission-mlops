import pandas as pd
import numpy as np
import os
import json
import urllib.request
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────
RAW_DIR        = "data/raw"
RAW_PATH       = "data/raw/diabetic_data.csv"
PROCESSED_DIR  = "data/processed"
PROCESSED_PATH = "data/processed/diabetic_data_clean.csv"
LOG_PATH       = "data/processed/pipeline_log.json"
META_PATH      = "data/processed/feature_metadata.json"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Clinical ICD-9 Code Mapping ───────────────────────────────────
def map_icd9_to_category(code):
    """
    Map ICD-9 diagnosis codes to 9 clinical categories:
    Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal,
    Genitourinary, Neoplasms, and Other.
    """
    if pd.isna(code) or str(code).strip() in ["?", "", "None"]:
        return "Missing"
    
    code_str = str(code).strip()
    
    # E and V codes (supplementary classifications)
    if code_str.startswith("E") or code_str.startswith("V"):
        return "Other"
    
    try:
        val = float(code_str)
        if 390 <= val <= 459 or val == 785:
            return "Circulatory"
        elif 460 <= val <= 519 or val == 786:
            return "Respiratory"
        elif 520 <= val <= 579 or val == 787:
            return "Digestive"
        elif 250 <= val < 251:
            return "Diabetes"
        elif 800 <= val <= 999:
            return "Injury"
        elif 710 <= val <= 739:
            return "Musculoskeletal"
        elif 580 <= val <= 629 or val == 788:
            return "Genitourinary"
        elif 140 <= val <= 239:
            return "Neoplasms"
        else:
            return "Other"
    except ValueError:
        return "Other"

def ensure_raw_data():
    """Download raw dataset if missing."""
    if not os.path.exists(RAW_PATH):
        print(f"[0/5] Raw dataset not found at {RAW_PATH}. Attempting download...")
        url = "https://raw.githubusercontent.com/naveditachaudhary-pixel/hospital-readmission-mlops/main/data/raw/diabetic_data.csv"
        try:
            urllib.request.urlretrieve(url, RAW_PATH)
            print(f"      Downloaded raw data to {RAW_PATH}")
        except Exception as e:
            print(f"      Warning: Could not download raw file ({e}). Will check processed file.")

def load_data(path=RAW_PATH):
    print(f"[1/5] Loading raw data from {path}...")
    if not os.path.exists(path):
        if os.path.exists(PROCESSED_PATH):
            print(f"      Raw path not found, using existing processed data from {PROCESSED_PATH}")
            return pd.read_csv(PROCESSED_PATH)
        else:
            raise FileNotFoundError(f"Neither {path} nor {PROCESSED_PATH} exists!")
            
    df = pd.read_csv(path, na_values=["?", "None", ""])
    print(f"      Loaded raw shape: {df.shape}")
    return df

def validate_data(df):
    print("[2/5] Validating dataset integrity...")
    report = {}
    
    # Target presence
    if "readmitted" not in df.columns:
        raise ValueError("Dataset missing target column 'readmitted'")
        
    null_counts = df.isnull().sum()
    report["null_counts"] = null_counts[null_counts > 0].to_dict()
    report["class_distribution"] = df["readmitted"].value_counts().to_dict()
    print(f"      Columns with nulls: {len(report['null_counts'])}")
    print(f"      Target distribution: {report['class_distribution']}")
    return report

def preprocess_data(df):
    print("[3/5] Preprocessing & Engineering Clinical Features...")
    df = df.copy()

    # Drop identifiers and uninformative high-null columns if present
    drop_cols = ["encounter_id", "patient_nbr", "payer_code", "weight", "medical_specialty"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Binary Target Encoding: 1 = Readmitted <30 days, 0 = Otherwise (NO or >30)
    if df["readmitted"].dtype == object or isinstance(df["readmitted"].iloc[0], str):
        df["readmitted"] = df["readmitted"].apply(lambda x: 1 if str(x).strip() == "<30" else 0)

    # Age ordinal mapping
    if "age" in df.columns and df["age"].dtype == object:
        age_map = {"[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
                   "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9}
        df["age"] = df["age"].map(age_map)

    # ICD-9 Clinical Diagnosis Categories
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[f"{col}_cat"] = df[col].apply(map_icd9_to_category)
            df.drop(columns=[col], inplace=True)

    # Clinical Feature Engineering
    df["service_utilization"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    df["emergency_ratio"] = df["number_emergency"] / (df["service_utilization"] + 1)
    df["inpatient_ratio"] = df["number_inpatient"] / (df["service_utilization"] + 1)
    df["med_per_day"] = df["num_medications"] / (df["time_in_hospital"] + 1e-5)
    df["lab_per_day"] = df["num_lab_procedures"] / (df["time_in_hospital"] + 1e-5)
    df["treatment_intensity"] = df["num_procedures"] + df["num_medications"] + df["num_lab_procedures"]
    
    # Discharge Disposition Risk Flag (SNF, Hospice, Home Health Transfer)
    high_risk_discharges = [3, 5, 6, 11, 13, 14, 22]
    if "discharge_disposition_id" in df.columns:
        df["is_high_risk_discharge"] = df["discharge_disposition_id"].isin(high_risk_discharges).astype(int)

    # Count Medication Adjustments (Up or Down changes)
    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'gliclazide',
                'glipizide-metformin', 'troglitazone', 'tolazamide', 'insulin',
                'rosiglitazone', 'pioglitazone']
    available_meds = [m for m in med_cols if m in df.columns]
    if available_meds:
        df["num_med_changes"] = df[available_meds].apply(
            lambda row: sum(1 for v in row if str(v).strip() in ["Up", "Down"]), axis=1
        )

    # Impute missing numeric features with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col != "readmitted" and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Impute and Label Encode categorical features
    cat_cols = df.select_dtypes(include=["object", "category", "string", "str"]).columns.tolist()
    cat_mappings = {}
    for col in cat_cols:
        df[col] = df[col].fillna("Missing")
        categories = sorted(df[col].astype(str).unique().tolist())
        mapping = {cat: i for i, cat in enumerate(categories)}
        cat_mappings[col] = mapping
        df[col] = df[col].astype(str).map(mapping)

    # Save feature metadata for reproducible prediction
    with open(META_PATH, "w") as f:
        json.dump({
            "numeric_features": [c for c in df.columns if c != "readmitted"],
            "categorical_mappings": cat_mappings,
            "engineered_features": [
                "service_utilization", "emergency_ratio", "inpatient_ratio",
                "med_per_day", "lab_per_day", "treatment_intensity",
                "is_high_risk_discharge", "num_med_changes"
            ]
        }, f, indent=2)

    print(f"      Final Shape: {df.shape}")
    print(f"      Engineered 8 clinical domain features.")
    return df

def save_data(df, path=PROCESSED_PATH):
    print(f"[4/5] Saving cleaned data to {path}...")
    df.to_csv(path, index=False)
    print(f"      Saved {df.shape[0]:,} rows × {df.shape[1]} columns")

def log_run(report, df):
    print("[5/5] Logging data pipeline metadata...")
    log = {
        "run_timestamp": datetime.now().isoformat(),
        "processed_shape": list(df.shape),
        "target_distribution": df["readmitted"].value_counts().to_dict(),
        "null_counts": report.get("null_counts", {}),
        "engineered_features": [
            "service_utilization", "emergency_ratio", "inpatient_ratio",
            "med_per_day", "lab_per_day", "treatment_intensity",
            "is_high_risk_discharge", "num_med_changes",
            "diag_1_cat", "diag_2_cat", "diag_3_cat"
        ]
    }
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"      Pipeline log saved to {LOG_PATH}")

if __name__ == "__main__":
    ensure_raw_data()
    df_raw = load_data()
    report = validate_data(df_raw)
    df_clean = preprocess_data(df_raw)
    save_data(df_clean)
    log_run(report, df_clean)
    print("\n✅ Data pipeline execution finished successfully!")
