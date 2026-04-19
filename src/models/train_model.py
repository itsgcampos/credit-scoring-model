from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def prepare_features(df: pd.DataFrame):
    target_col = "target"
    group_col = "Customer_ID"
    cols_to_drop = ["Type_of_Loan", "Month"]

    X = df.drop(columns=[target_col, group_col]).copy()
    y = df[target_col].copy()
    groups = df[group_col].copy()

    X = X.drop(columns=cols_to_drop, errors="ignore")

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    return X, y, groups, categorical_cols, cols_to_drop, target_col, group_col


def one_hot_encode_train_test(X_train, X_test, categorical_cols):
    X_train_model = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
    X_test_model = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)

    X_train_model, X_test_model = X_train_model.align(
        X_test_model, join="left", axis=1, fill_value=0
    )

    return X_train_model, X_test_model


def train_xgboost_model(df: pd.DataFrame):
    X, y, groups, categorical_cols, cols_to_drop, target_col, group_col = prepare_features(df)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    X_train_model, X_test_model = one_hot_encode_train_test(X_train, X_test, categorical_cols)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Utilizando os valores do grid
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_model, y_train)

    y_proba = model.predict_proba(X_test_model)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    artifact = {
        "model": model,
        "feature_columns": X_train_model.columns.tolist(),
        "categorical_cols": categorical_cols,
        "cols_to_drop": cols_to_drop,
        "target_col": target_col,
        "group_col": group_col,
        "threshold": 0.5,
        "test_auc": auc
    }

    return artifact


def save_artifact(artifact: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "credit_score_features.csv"
    model_path = project_root / "models" / "xgboost_artifact.pkl"

    df = pd.read_csv(data_path)
    artifact = train_xgboost_model(df)
    save_artifact(artifact, model_path)

    print(f"Modelo salvo em: {model_path}")
    print(f"AUC de teste interno: {artifact['test_auc']:.4f}")