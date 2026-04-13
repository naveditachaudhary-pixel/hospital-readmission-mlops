# 🏥 Hospital Readmission Predictor - MLOps Pipeline

This repository contains an end-to-end Machine Learning pipeline designed to predict early hospital readmissions (<30 days) for diabetic patients using the UCI Diabetes 130-US Hospitals dataset.

## 🚀 Project Overview

Hospital readmissions are incredibly costly and dangerous for patients. This MLOps project establishes a reproducible pipeline to clean data, train models, track experiments, and serve predictions via a live interactive dashboard.

### 🛠️ Architecture & Technologies
*   **Data Pipeline:** `pandas` and `numpy` (Handling missing values, creating clinical features, balancing class weights).
*   **Modeling:** `scikit-learn` and `xgboost` (Logistic Regression baseline, Random Forest, XGBoost Champion).
*   **Experiment Tracking & Registry:** `MLflow` (Tracking hyperparams, logging Model AUC-ROC/F1 metrics, and saving model artifacts).
*   **Live Web App:** `Streamlit` (Interactive clinical prediction UI loading models dynamically from MLflow).

## 📁 Repository Structure
```text
hospital-readmission-mlops/
├── data/
│   ├── raw/                 # Raw downloaded diabetic_data.csv
│   └── processed/           # Cleaned CSV and pipeline logs
├── src/
│   ├── data_pipeline.py     # Data cleaning and feature engineering
│   ├── train.py             # Model training, evaluating, and MLflow logging
│   └── app.py               # Streamlit extra-credit interactive dashboard
├── artifacts/               # Generated png screenshots of the UI and MLflow
├── mlruns/                  # MLflow tracking backend (local)
└── requirements.txt         # Project dependencies
```

## ⚙️ How to Run Locally

**1. Clone the repository and setup the environment**
```powershell
git clone https://github.com/naveditachaudhary-pixel/hospital-readmission-mlops.git
cd hospital-readmission-mlops
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run the Data Pipeline**
```powershell
python src/data_pipeline.py
```

**3. Train Models and Track with MLflow**
```powershell
python src/train.py
```

**4. View MLflow Dashboard**
```powershell
mlflow ui
```
Navigate to `http://localhost:5000` to compare Logistic Regression, Random Forest, and XGBoost experiments.

**5. Launch the Streamlit App (Extra Credit)**
```powershell
streamlit run src/app.py
```
Navigate to `http://localhost:8501` to randomly sample patients and simulate real-time model inference.

## 📉 Challenges Solved
1. **Missing Data:** Handled systematically by imputing missing numeric features using their median, and categorical features via mode.
2. **Class Imbalance:** Significant imbalance existed between the 'Early Readmission' positive targets vs negatives. We fixed this using `scale_pos_weight` inside the XGBoost Champion and strict class balancing during Training pipelines.

## 📈 Evaluation Metrics
The predictive models were evaluated using AUC-ROC and Macro F1 scores over a robust 5-fold cross-validation scheme:

| Model | AUC-ROC | Macro F1 | CV AUC Mean |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** (Baseline) | 0.6491 | 0.4834 | 0.6352 |
| **Random Forest** | 0.6710 | 0.5454 | 0.6551 |
| **XGBoost** (👑 Champion) | **0.6722** | **0.5574** | **0.6611** |

> XGBoost significantly outperformed the baseline logic to become the registered Champion Model inside our MLflow Registry!