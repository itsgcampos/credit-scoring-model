import joblib
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

from src.models.predict import prepare_new_data


def evaluate_model(df: pd.DataFrame, artifact_path: str) -> dict:
    artifact = joblib.load(artifact_path)

    target_col = artifact["target_col"]
    model = artifact["model"]
    threshold = artifact["threshold"]

    y_true = df[target_col].copy()
    X = df.drop(columns=[target_col, artifact["group_col"]], errors="ignore").copy()

    X_prepared = prepare_new_data(X, artifact)

    y_proba = model.predict_proba(X_prepared)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)

    metrics = {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_proba),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

    return metrics