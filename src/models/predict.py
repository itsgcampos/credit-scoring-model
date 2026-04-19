from pathlib import Path
import joblib
import pandas as pd


def prepare_new_data(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    X = df.copy()

    cols_to_drop = artifact["cols_to_drop"]
    categorical_cols = artifact["categorical_cols"]
    feature_columns = artifact["feature_columns"]

    X = X.drop(columns=cols_to_drop, errors="ignore")
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    X = X.reindex(columns=feature_columns, fill_value=0)

    return X


def predict_scores(df: pd.DataFrame, artifact_path: str):
    artifact = joblib.load(artifact_path)

    model = artifact["model"]
    threshold = artifact["threshold"]

    X_prepared = prepare_new_data(df, artifact)

    proba = model.predict_proba(X_prepared)[:, 1]
    pred = (proba >= threshold).astype(int)

    result = df.copy()
    result["predict_proba"] = proba
    result["prediction"] = pred

    return result


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    artifact_path = project_root / "models" / "xgboost_artifact.pkl"

    print("Módulo de predição pronto para uso.")