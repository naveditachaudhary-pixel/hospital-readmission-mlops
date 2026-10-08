import os
import json
import pandas as pd
import pytest
from src.data_pipeline import map_icd9_to_category, preprocess_data

def test_icd9_category_mapping():
    assert map_icd9_to_category("414.01") == "Circulatory"
    assert map_icd9_to_category("486") == "Respiratory"
    assert map_icd9_to_category("530") == "Digestive"
    assert map_icd9_to_category("250.02") == "Diabetes"
    assert map_icd9_to_category("174.9") == "Neoplasms"
    assert map_icd9_to_category("820") == "Injury"
    assert map_icd9_to_category("715") == "Musculoskeletal"
    assert map_icd9_to_category("590") == "Genitourinary"
    assert map_icd9_to_category("V45") == "Other"
    assert map_icd9_to_category("?") == "Missing"

def test_preprocess_data_schema():
    # Construct mini dummy dataframe matching raw schema
    dummy_data = {
        "encounter_id": [1, 2],
        "patient_nbr": [100, 101],
        "readmitted": ["<30", "NO"],
        "age": ["[60-70)", "[70-80)"],
        "time_in_hospital": [3, 5],
        "num_lab_procedures": [40, 50],
        "num_procedures": [1, 2],
        "num_medications": [10, 15],
        "number_outpatient": [0, 1],
        "number_emergency": [1, 0],
        "number_inpatient": [1, 0],
        "number_diagnoses": [5, 8],
        "diag_1": ["414.01", "250.0"],
        "diag_2": ["486", "715"],
        "diag_3": ["E930", "590"],
        "discharge_disposition_id": [1, 6]
    }
    df_dummy = pd.DataFrame(dummy_data)
    df_clean = preprocess_data(df_dummy)

    # Check target binary conversion
    assert list(df_clean["readmitted"]) == [1, 0]
    
    # Check engineered features presence
    expected_engineered = [
        "service_utilization", "emergency_ratio", "inpatient_ratio",
        "med_per_day", "lab_per_day", "treatment_intensity",
        "is_high_risk_discharge"
    ]
    for feat in expected_engineered:
        assert feat in df_clean.columns

def test_processed_files_exist():
    assert os.path.exists("data/processed/diabetic_data_clean.csv")
    assert os.path.exists("data/processed/pipeline_log.json")
    assert os.path.exists("artifacts/champion_model.joblib")
