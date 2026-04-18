from src.data.load_data import load_raw_data
from src.data.preprocess import process_pipeline
from src.features.build_features import build_features_pipeline


if __name__ == "__main__":
    df_raw = load_raw_data("data/raw/credit_score.csv")

    df_clean = process_pipeline(df_raw)
    df_clean.to_csv("data/processed/credit_score_clean.csv", index=False)
    print("\nBase tratada salva em 'data/processed/credit_score_clean.csv'")

    df_features = build_features_pipeline(df_clean)
    df_features.to_csv("data/processed/credit_score_features.csv", index=False)
    print("\nBase pronta para modelagem salva em 'data/processed/credit_score_features.csv'")
