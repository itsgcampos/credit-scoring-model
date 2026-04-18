import pandas as pd
from src.data.load_data import load_raw_data
from src.data.preprocess import process_pipeline

if __name__ == "__main__":
    # 1. Carregar dados
    df_raw = load_raw_data("data/raw/credit_score.csv")
    
    # 2. Pré-processar
    df_clean = process_pipeline(df_raw)
    
    # 3. Salvar dados processados
    df_clean.to_csv("data/processed/credit_score_clean.csv", index=False)
    print("Dados limpos salvos com sucesso em 'data/processed/credit_score_clean.csv'")