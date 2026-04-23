import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, log_loss, confusion_matrix,
                              ConfusionMatrixDisplay)
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────
PROCESSED_PATH = "data/processed/diabetic_data_clean.csv"
os.makedirs("artifacts", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(PROCESSED_PATH)
    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# ── Evaluate & log one run ─────────────────────────────────────────
def evaluate_and_log(model, model_name, params,
                     X_train, X_val, y_train, y_val,
                     log_model_fn):
    with mlflow.start_run(run_name=model_name):

        # Log metadata
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset", "UCI Diabetes 130-US Hospitals")
        mlflow.set_tag("author", "Navedita")

        # Log hyperparameters
        mlflow.log_params(params)

        # Train
        model.fit(X_train, y_train)

        # Predict on validation set
        y_pred  = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        # Metrics
        metrics = {
            "auc_roc":   roc_auc_score(y_val, y_proba),
            "f1_macro":  f1_score(y_val, y_pred, average="macro"),
            "f1_weighted": f1_score(y_val, y_pred, average="weighted"),
            "precision": precision_score(y_val, y_pred),
            "recall":    recall_score(y_val, y_pred),
            "log_loss":  log_loss(y_val, y_proba),
        }

        # 5-fold cross-validation AUC on training set
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=5, scoring="roc_auc")
        metrics["cv_auc_mean"] = cv_scores.mean()
        metrics["cv_auc_std"]  = cv_scores.std()

        mlflow.log_metrics(metrics)

        # Confusion matrix artifact
        cm = confusion_matrix(y_val, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Readmit", "Readmit"])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"{model_name} — Confusion Matrix")
        cm_path = f"artifacts/{model_name}_confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(cm_path)

        # SHAP artifact (XGBoost and RF only — skip for LR)
        if model_name != "LogisticRegression":
            explainer = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(X_val[:500])
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_vals, X_val[:500],
                              show=False, plot_size=None)
            shap_path = f"artifacts/{model_name}_shap_summary.png"
            plt.savefig(shap_path, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(shap_path)

        # Log model
        log_model_fn(model, model_name)

        print(f"\n{model_name}")
        print(f"  AUC-ROC : {metrics['auc_roc']:.4f}")
        print(f"  F1 macro: {metrics['f1_macro']:.4f}")
        print(f"  CV AUC  : {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")

        return metrics

# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("hospital_readmission_prediction")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())

    # ── 1. Logistic Regression (baseline) ─────────────────────────
    lr_params = {"C": 1.0, "penalty": "l2", "solver": "saga",
                 "max_iter": 1000, "random_state": 42}
    lr = LogisticRegression(**lr_params)
    evaluate_and_log(lr, "LogisticRegression", lr_params,
                     X_train, X_val, y_train, y_val,
                     mlflow.sklearn.log_model)

    # ── 2. XGBoost (champion) ──────────────────────────────────────
    xgb_params = {"max_depth": 8, "learning_rate": 0.05,
                  "n_estimators": 300, "subsample": 0.8,
                  "colsample_bytree": 0.8,
                  "scale_pos_weight": scale_pos,
                  "eval_metric": "logloss", "random_state": 42}
    xgb = XGBClassifier(**xgb_params)
    xgb_metrics = evaluate_and_log(xgb, "XGBoost", xgb_params,
                                   X_train, X_val, y_train, y_val,
                                   mlflow.xgboost.log_model)

    # ── 3. Random Forest (challenger) ─────────────────────────────
    rf_params = {"n_estimators": 200, "max_depth": 10,
                 "min_samples_split": 5,
                 "class_weight": "balanced", "random_state": 42}
    rf = RandomForestClassifier(**rf_params)
    evaluate_and_log(rf, "RandomForest", rf_params,
                     X_train, X_val, y_train, y_val,
                     mlflow.sklearn.log_model)

    # ── Register best model ────────────────────────────────────────
    print("\nRegistering XGBoost to Model Registry...")
    runs = mlflow.search_runs(experiment_names=["hospital_readmission_prediction"],
                              filter_string="tags.model_type = 'XGBoost'",
                              order_by=["metrics.auc_roc DESC"])
    best_run_id = runs.iloc[0]["run_id"]
    model_uri   = f"runs:/{best_run_id}/XGBoost"
    mlflow.register_model(model_uri, "HospitalReadmissionChampion")

    print("\nAll experiments complete! Run: mlflow ui")
