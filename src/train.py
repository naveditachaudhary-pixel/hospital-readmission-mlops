import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, log_loss, brier_score_loss,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

# XGBoost with fallback support
XGB_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    import mlflow.xgboost
    XGB_AVAILABLE = True
except Exception as e:
    print(f"XGBoost notice: {e}. Will use HistGradientBoosting as boosting champion alternative.")

# ── Paths ──────────────────────────────────────────────────────────
PROCESSED_PATH = "data/processed/diabetic_data_clean.csv"
ARTIFACTS_DIR  = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── Load and Split Data ────────────────────────────────────────────
def load_and_split_data():
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at {PROCESSED_PATH}. Run data_pipeline.py first!")

    df = pd.read_csv(PROCESSED_PATH)
    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]

    # Stratified 70/15/15 Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"Data Loaded successfully:")
    print(f"  Train Set : {X_train.shape} (Positives: {y_train.sum()})")
    print(f"  Val Set   : {X_val.shape} (Positives: {y_val.sum()})")
    print(f"  Test Set  : {X_test.shape} (Positives: {y_test.sum()})")

    return X_train, X_val, X_test, y_train, y_val, y_test

# ── Metric Calculation Utility ────────────────────────────────────
def calculate_metrics(y_true, y_pred, y_proba):
    precision_vec, recall_vec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall_vec, precision_vec)
    
    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(pr_auc),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary")),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
    }

# ── Evaluate and Log Model ────────────────────────────────────────
def evaluate_and_log(model, model_name, params, X_train, X_val, y_train, y_val, log_model_fn):
    with mlflow.start_run(run_name=model_name) as run:
        print(f"\n--- Training & Evaluating: {model_name} ---")

        # Tags & Parameters
        mlflow.set_tag("model_family", model_name)
        mlflow.set_tag("dataset", "UCI Diabetes 130-US Hospitals")
        mlflow.set_tag("pipeline", "Hospital Readmission MLOps")
        mlflow.log_params(params)

        # Fit Model
        model.fit(X_train, y_train)

        # Validation Predictions
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        # Calculate Validation Metrics
        metrics = calculate_metrics(y_val, y_pred, y_proba)

        # 3-Fold Stratified Cross-Validation on Training Set
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        metrics["cv_auc_mean"] = float(cv_scores.mean())
        metrics["cv_auc_std"] = float(cv_scores.std())

        mlflow.log_metrics(metrics)

        # Plot 1: Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Readmit", "Readmitted <30d"])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{model_name} — Confusion Matrix")
        cm_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches="tight", dpi=150)
        plt.close()
        mlflow.log_artifact(cm_path)

        # Plot 2: ROC & PR Curves Combined
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        precision_vec, recall_vec, _ = precision_recall_curve(y_val, y_proba)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        
        ax1.plot(fpr, tpr, color="#2563EB", lw=2, label=f"AUC = {metrics['auc_roc']:.4f}")
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.6)
        ax1.set_title(f"{model_name} — ROC Curve")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        ax2.plot(recall_vec, precision_vec, color="#059669", lw=2, label=f"PR-AUC = {metrics['pr_auc']:.4f}")
        ax2.set_title(f"{model_name} — Precision-Recall Curve")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        curve_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_performance_curves.png")
        plt.savefig(curve_path, bbox_inches="tight", dpi=150)
        plt.close()
        mlflow.log_artifact(curve_path)

        # Log Model with Signature if available
        try:
            from mlflow.models import infer_signature
            signature = infer_signature(X_val[:5], y_proba[:5])
            if log_model_fn == mlflow.sklearn.log_model:
                log_model_fn(model, model_name, signature=signature, serialization_format="cloudpickle")
            else:
                log_model_fn(model, model_name, signature=signature)
        except Exception:
            if log_model_fn == mlflow.sklearn.log_model:
                log_model_fn(model, model_name, serialization_format="cloudpickle")
            else:
                log_model_fn(model, model_name)

        print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")
        print(f"  PR-AUC   : {metrics['pr_auc']:.4f}")
        print(f"  Macro F1 : {metrics['f1_macro']:.4f}")
        print(f"  CV AUC   : {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")

        return model, metrics, run.info.run_id

# ── Main Training Routine ──────────────────────────────────────────
if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("hospital_readmission_prediction")

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = neg_count / pos_count
    print(f"Class ratio (Negative / Positive): {scale_pos:.2f}")

    models_trained = {}

    # 1. Logistic Regression (Scaled Baseline)
    lr_params = {"C": 0.5, "solver": "lbfgs", "max_iter": 500, "random_state": 42}
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(**lr_params))
    ])
    lr_model, lr_metrics, lr_run_id = evaluate_and_log(
        lr_pipeline, "LogisticRegression", lr_params,
        X_train, X_val, y_train, y_val,
        mlflow.sklearn.log_model
    )
    models_trained["LogisticRegression"] = (lr_model, lr_metrics, lr_run_id)

    # 2. Random Forest (Balanced Challenger)
    rf_params = {"n_estimators": 50, "max_depth": 8, "min_samples_split": 6, "class_weight": "balanced", "n_jobs": 1, "random_state": 42}
    rf = RandomForestClassifier(**rf_params)
    rf_model, rf_metrics, rf_run_id = evaluate_and_log(
        rf, "RandomForest", rf_params,
        X_train, X_val, y_train, y_val,
        mlflow.sklearn.log_model
    )
    models_trained["RandomForest"] = (rf_model, rf_metrics, rf_run_id)

    # 3. Boosting Champion (XGBoost or HistGradientBoosting fallback)
    if XGB_AVAILABLE:
        try:
            xgb_params = {
                "max_depth": 6,
                "learning_rate": 0.04,
                "n_estimators": 350,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": scale_pos,
                "eval_metric": "logloss",
                "random_state": 42
            }
            xgb = XGBClassifier(**xgb_params)
            xgb_model, xgb_metrics, xgb_run_id = evaluate_and_log(
                xgb, "XGBoost", xgb_params,
                X_train, X_val, y_train, y_val,
                mlflow.xgboost.log_model
            )
            models_trained["XGBoost"] = (xgb_model, xgb_metrics, xgb_run_id)
        except Exception as e:
            print(f"XGBoost training warning: {e}. Falling back to HistGradientBoosting.")
            hgb_params = {"max_depth": 6, "learning_rate": 0.04, "max_iter": 300, "random_state": 42}
            hgb = HistGradientBoostingClassifier(**hgb_params)
            hgb_model, hgb_metrics, hgb_run_id = evaluate_and_log(
                hgb, "HistGradientBoosting", hgb_params,
                X_train, X_val, y_train, y_val,
                mlflow.sklearn.log_model
            )
            models_trained["HistGradientBoosting"] = (hgb_model, hgb_metrics, hgb_run_id)
    else:
        hgb_params = {"max_depth": 6, "learning_rate": 0.04, "max_iter": 300, "random_state": 42}
        hgb = HistGradientBoostingClassifier(**hgb_params)
        hgb_model, hgb_metrics, hgb_run_id = evaluate_and_log(
            hgb, "HistGradientBoosting", hgb_params,
            X_train, X_val, y_train, y_val,
            mlflow.sklearn.log_model
        )
        models_trained["HistGradientBoosting"] = (hgb_model, hgb_metrics, hgb_run_id)

    # Determine Champion Model based on AUC-ROC
    champion_name = max(models_trained, key=lambda k: models_trained[k][1]["auc_roc"])
    champ_model, champ_metrics, champ_run_id = models_trained[champion_name]

    print(f"\n=======================================================")
    print(f"👑 CHAMPION MODEL SELECTED: {champion_name}")
    print(f"   Validation AUC-ROC : {champ_metrics['auc_roc']:.4f}")
    print(f"   Validation PR-AUC  : {champ_metrics['pr_auc']:.4f}")
    print(f"=======================================================")

    # Test Set Final Benchmark
    y_test_pred = champ_model.predict(X_test)
    y_test_proba = champ_model.predict_proba(X_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, y_test_proba))
    test_f1 = float(f1_score(y_test, y_test_pred, average="macro"))
    print(f"🏆 Final Held-Out Test Set Performance:")
    print(f"   Test Set AUC-ROC : {test_auc:.4f}")
    print(f"   Test Set Macro F1: {test_f1:.4f}")

    # Save fallback joblib artifact for reliable dashboard loading
    fallback_model_path = os.path.join(ARTIFACTS_DIR, "champion_model.joblib")
    joblib.dump({
        "model": champ_model,
        "model_name": champion_name,
        "metrics": champ_metrics,
        "features": list(X_train.columns)
    }, fallback_model_path)
    print(f"Saved fallback champion artifact to {fallback_model_path}")

    # MLflow Model Registry Registration
    try:
        model_uri = f"runs:/{champ_run_id}/{champion_name}"
        registered_model = mlflow.register_model(model_uri, "HospitalReadmissionChampion")
        print(f"Successfully registered model '{champion_name}' (Version {registered_model.version}) to MLflow Registry!")
    except Exception as e:
        print(f"Note: Could not register model in MLflow Registry ({e}). Fallback artifact ready.")

    print("\n✅ All model experiments completed successfully! Launch dashboard with: streamlit run src/app.py")
