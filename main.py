import pandas as pd
from src.data.load_data import load_raw_data
from src.data.preprocess import process_pipeline
from src.features.build_features import build_features_pipeline # <--- NOVA IMPORTAÇÃO

if __name__ == "__main__":
    # 1. Carregar dados
    df_raw = load_raw_data("data/raw/credit_score.csv")
    
    # 2. Pré-processar (Limpeza)
    df_clean = process_pipeline(df_raw)
    
    # 3. Engenharia de Features
    df_features = build_features_pipeline(df_clean)
    
    # 4. Salvar base final para modelagem
    df_features.to_csv("data/processed/credit_score_features.csv", index=False)
    print("\n✅ Base pronta para modelagem salva em 'data/processed/credit_score_features.csv'")