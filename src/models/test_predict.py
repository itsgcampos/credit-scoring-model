from pathlib import Path
import pandas as pd

from src.data.preprocess import process_pipeline
from src.features.build_features import build_features_pipeline
from src.models.predict import predict_scores


def run_test_prediction(input_csv: str, output_csv: str, artifact_path: str) -> pd.DataFrame:
    """
    Lê um CSV de observações novas (sem target), aplica o pipeline de processamento
    e feature engineering, e gera predições com o modelo treinado.
    """
    df_raw = pd.read_csv(input_csv)

    print(f"Base bruta carregada: {df_raw.shape}")

    df_clean = process_pipeline(df_raw)
    print(f"Base após preprocessamento: {df_clean.shape}")

    df_features = build_features_pipeline(df_clean)
    print(f"Base após feature engineering: {df_features.shape}")

    id_cols = [col for col in ["ID", "Customer_ID", "Name"] if col in df_raw.columns]

    predictions = predict_scores(df_features, artifact_path)

    for col in reversed(id_cols):
        predictions.insert(0, col, df_raw[col].values)

    predictions.to_csv(output_csv, index=False)
    print(f"Predições salvas em: {output_csv}")

    return predictions


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    input_csv = project_root / "data" / "raw" / "clients_to_predict.csv"
    output_csv = project_root / "data" / "predictions" / "clients_to_predict.csv"
    artifact_path = project_root / "models" / "xgboost_artifact.pkl"

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    predictions = run_test_prediction(
        input_csv=str(input_csv),
        output_csv=str(output_csv),
        artifact_path=str(artifact_path)
    )

    cols_to_show = [col for col in ["ID", "Customer_ID", "score_default", "prediction"] if col in predictions.columns]
    print("\nAmostra das predições:")
    print(predictions[cols_to_show].head())