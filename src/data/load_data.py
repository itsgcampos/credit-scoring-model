import pandas as pd
import os

def load_raw_data(filepath: str = "data/raw/credit_score.csv") -> pd.DataFrame:
    """
    Carrega o dataset bruto de credit scoring.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    
    print(f"Carregando dados de {filepath}...")
    
    # low_memory=False adicionado para evitar o DtypeWarning em bases sujas
    df = pd.read_csv(filepath, low_memory=False)
    
    print(f"Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")
    
    return df