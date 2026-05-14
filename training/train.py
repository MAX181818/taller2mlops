import os
import logging
import tempfile
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURE_COLUMNS = [
    "age", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "race_encoded",
    "gender_encoded", "admission_type_encoded", "discharge_encoded",
    "admission_source_encoded", "a1c_result_encoded",
    "metformin_encoded", "insulin_encoded"
]
TARGET_COLUMN = "readmitted"

MODEL_CANDIDATES = [
    {
        "model_type": "LogisticRegression",
        "params": {"C": 1.0, "max_iter": 200, "solver": "lbfgs"},
        "build": lambda p: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", n_jobs=1, **p))
        ])
    },
    {
        "model_type": "RandomForest",
        "params": {"n_estimators": 50, "max_depth": 10},
        "build": lambda p: RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=1, **p
        )
    },
    {
        "model_type": "RandomForest",
        "params": {"n_estimators": 100, "max_depth": 15},
        "build": lambda p: RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=1, **p
        )
    },
]


def _log_artifacts(model, model_type, X_val, y_val, X_test, y_test, feature_names, tmpdir):
    """Log ROC curve, confusion matrix and feature importance as MLflow artifacts."""
    proba_val = model.predict_proba(X_val)[:, 1]
    preds_val = model.predict(X_val)

    # ROC Curve
    fpr_v, tpr_v, _ = roc_curve(y_val, proba_val)
    auc_val = roc_auc_score(y_val, proba_val)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_v, tpr_v, label=f"Val AUC = {auc_val:.3f}")
    if X_test is not None:
        proba_test = model.predict_proba(X_test)[:, 1]
        fpr_t, tpr_t, _ = roc_curve(y_test, proba_test)
        auc_test = roc_auc_score(y_test, proba_test)
        ax.plot(fpr_t, tpr_t, linestyle="--", label=f"Test AUC = {auc_test:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_type}")
    ax.legend()
    fig.tight_layout()
    roc_path = os.path.join(tmpdir, "roc_curve.png")
    fig.savefig(roc_path, dpi=100)
    plt.close(fig)
    mlflow.log_artifact(roc_path, artifact_path="plots")

    # Confusion Matrix (validation set)
    cm = confusion_matrix(y_val, preds_val)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No readmit", "Readmit"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix (Val) — {model_type}")
    fig.tight_layout()
    cm_path = os.path.join(tmpdir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=100)
    plt.close(fig)
    mlflow.log_artifact(cm_path, artifact_path="plots")

    # Feature Importance
    fig, ax = plt.subplots(figsize=(9, 5))
    if model_type == "RandomForest":
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]
        ax.bar(range(len(importances)), importances[idx])
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Feature Importance — {model_type}")
        ax.set_ylabel("Importance")
    else:
        clf = model.named_steps["clf"]
        coefs = np.abs(clf.coef_[0])
        idx = np.argsort(coefs)[::-1]
        ax.bar(range(len(coefs)), coefs[idx])
        ax.set_xticks(range(len(coefs)))
        ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"|Coefficients| — {model_type}")
        ax.set_ylabel("|Coefficient|")
    fig.tight_layout()
    fi_path = os.path.join(tmpdir, "feature_importance.png")
    fig.savefig(fi_path, dpi=100)
    plt.close(fig)
    mlflow.log_artifact(fi_path, artifact_path="plots")


def run_training(db_conn, mlflow_uri, experiment_name, model_name, batch_id, split_path=None) -> dict:
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    logger = logging.getLogger(__name__)

    if split_path and os.path.isdir(split_path):
        X_train = pd.read_csv(f"{split_path}/X_train.csv")
        X_val   = pd.read_csv(f"{split_path}/X_val.csv")
        X_test  = pd.read_csv(f"{split_path}/X_test.csv")
        y_train = pd.read_csv(f"{split_path}/y_train.csv").squeeze()
        y_val   = pd.read_csv(f"{split_path}/y_val.csv").squeeze()
        y_test  = pd.read_csv(f"{split_path}/y_test.csv").squeeze()
        logger.info(
            f"Split de t6 cargado | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
        )
    else:
        # Fallback cuando split_path no está disponible
        logger.warning("split_path no disponible — split 80/20 interno (sin test set separado)")
        engine = create_engine(f"postgresql://mlops_user:mlops_pass@{db_conn}:5432/mlops_db")
        df = pd.read_sql(
            f"SELECT {', '.join(FEATURE_COLUMNS + [TARGET_COLUMN])} FROM clean.diabetes_clean",
            engine
        )
        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_test, y_test = None, None

    results = []

    for candidate in MODEL_CANDIDATES:
        model_type = candidate["model_type"]
        params     = candidate["params"]
        model      = candidate["build"](params)

        with mlflow.start_run(run_name=f"{model_type}_{batch_id}") as run:
            model.fit(X_train, y_train)

            preds_val = model.predict(X_val)
            proba_val = model.predict_proba(X_val)[:, 1]

            val_metrics = {
                "val_roc_auc": roc_auc_score(y_val, proba_val),
                "val_f1":      f1_score(y_val, preds_val),
                "val_recall":  recall_score(y_val, preds_val),
            }

            test_metrics = {}
            if X_test is not None:
                preds_test = model.predict(X_test)
                proba_test = model.predict_proba(X_test)[:, 1]
                test_metrics = {
                    "test_roc_auc": roc_auc_score(y_test, proba_test),
                    "test_f1":      f1_score(y_test, preds_test),
                    "test_recall":  recall_score(y_test, preds_test),
                }

            log_params = {"model_type": model_type, "batch_id": batch_id}
            log_params.update(params)
            mlflow.log_params(log_params)
            mlflow.log_metrics({**val_metrics, **test_metrics})
            mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

            with tempfile.TemporaryDirectory() as tmpdir:
                _log_artifacts(
                    model, model_type,
                    X_val, y_val,
                    X_test, y_test,
                    FEATURE_COLUMNS, tmpdir,
                )

            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(model_name)
            version = versions[0].version if versions else "1"

            results.append({
                "run_id":        run.info.run_id,
                "model_version": version,
                "model_type":    model_type,
                "params":        params,
                "metrics":       val_metrics,
                "test_metrics":  test_metrics,
            })

    best = max(results, key=lambda r: r["metrics"]["val_roc_auc"])
    test_info = (
        f" | Test ROC-AUC: {best['test_metrics']['test_roc_auc']:.4f}"
        if best["test_metrics"]
        else ""
    )
    logger.info(
        f"Modelos entrenados: {len(results)} | "
        f"Ganador: {best['model_type']} {best['params']} | "
        f"Val ROC-AUC: {best['metrics']['val_roc_auc']:.4f}{test_info}"
    )

    return {
        "run_id":        best["run_id"],
        "model_version": best["model_version"],
        "metrics":       best["metrics"],
        "test_metrics":  best["test_metrics"],
        "promoted":      False,
    }
