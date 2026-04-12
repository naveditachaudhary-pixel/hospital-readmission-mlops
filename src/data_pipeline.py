import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────
RAW_PATH       = "data/raw/diabetic_data.csv"
PROCESSED_PATH = "data/processed/diabetic_data_clean.csv"
LOG_PATH       = "data/processed/pipeline_log.json"

os.makedirs("data/processed", exist_ok=True)

def load_data(path):
    print(f"[1/5] Loading data from {path}...")
    df = pd.read_csv(path, na_values=["?", "None", ""])
    print(f"      Shape: {df.shape}")
    return df

def validate_data(df):
    print("[2/5] Validating data...")
    report = {}

    # Check expected columns exist
    required = ["encounter_id", "patient_nbr", "readmitted", "age",
                "num_medications", "num_lab_procedures", "num_procedures",
                "time_in_hospital", "number_diagnoses"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Null report
    null_counts = df.isnull().sum()
    report["null_counts"] = null_counts[null_counts > 0].to_dict()
    print(f"      Columns with nulls: {len(report['null_counts'])}")

    # Value range checks
    assert df["time_in_hospital"].between(1, 14).all(), "time_in_hospital out of range"
    assert df["num_medications"].ge(0).all(),           "num_medications has negatives"
    assert df["num_lab_procedures"].ge(0).all(),        "num_lab_procedures has negatives"

    # Class distribution
    report["class_distribution"] = df["readmitted"].value_counts().to_dict()
    print(f"      Class distribution: {report['class_distribution']}")

    return report

def preprocess_data(df):
    print("[3/5] Preprocessing...")

    # Drop low-value columns
    drop_cols = ["encounter_id", "patient_nbr", "payer_code",
                 "weight", "medical_specialty"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Target encoding: readmitted → binary (1 = readmitted <30 days, 0 = otherwise)
    df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

    # Age: map range strings to ordinal integers
    age_map = {"[0-10)":0,"[10-20)":1,"[20-30)":2,"[30-40)":3,"[40-50)":4,
               "[50-60)":5,"[60-70)":6,"[70-80)":7,"[80-90)":8,"[90-100)":9}
    df["age"] = df["age"].map(age_map)

    # Impute remaining nulls
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Feature engineering
    df["treatment_intensity"] = df["num_procedures"] + df["num_medications"]

    # Label-encode remaining categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    print(f"      Final shape: {df.shape}")
    print(f"      Engineered features added: treatment_intensity")
    return df

def save_data(df, path):
    print(f"[4/5] Saving processed data to {path}...")
    df.to_csv(path, index=False)
    print(f"      Saved {df.shape[0]:,} rows × {df.shape[1]} columns")

def log_run(validation_report, df):
    print("[5/5] Logging pipeline run...")
    log = {
        "run_timestamp": datetime.now().isoformat(),
        "raw_path": RAW_PATH,
        "processed_path": PROCESSED_PATH,
        "raw_shape": validation_report.get("raw_shape"),
        "processed_shape": list(df.shape),
        "null_counts": validation_report.get("null_counts"),
        "class_distribution": validation_report.get("class_distribution"),
        "features_engineered": ["treatment_intensity"],
    }
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"      Log saved to {LOG_PATH}")

# ── Run pipeline ───────────────────────────────────────────────────
if __name__ == "__main__":
    df_raw = load_data(RAW_PATH)
    report = validate_data(df_raw)
    report["raw_shape"] = list(df_raw.shape)
    df_clean = preprocess_data(df_raw)
    save_data(df_clean, PROCESSED_PATH)
    log_run(report, df_clean)
    print("\nPipeline complete!")
