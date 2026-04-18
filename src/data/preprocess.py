import pandas as pd
import numpy as np

def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma a variável Credit_Score em um alvo binário (1 para Default, 0 para Bom).
    """
    # Vamos considerar "Poor" como Inadimplência (Default = 1)
    # E "Standard" e "Good" como Bom (0)
    df['target'] = df['Credit_Score'].apply(lambda x: 1 if x == 'Poor' else 0)
    return df

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove caracteres indesejados de colunas numéricas e converte os tipos.
    """
    # Colunas que deveriam ser numéricas mas estão como object
    numeric_cols = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 
                    'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly', 
                    'Monthly_Balance']
    
    for col in numeric_cols:
        if col in df.columns:
            # Substituir strings indesejadas como '_' por NaN
            df[col] = df[col].replace({'_': np.nan, '': np.nan})
            
            # Remover caracteres não numéricos (exceto ponto e sinal de menos) e converter
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.extract(r'(-?\d+\.?\d*)')[0].astype(float)
                
    # Tratamento de regras de negócio lógicas
    # Idade não pode ser negativa ou maior que 100
    if 'Age' in df.columns:
        df.loc[(df['Age'] < 18) | (df['Age'] > 100), 'Age'] = np.nan
        
    return df

def process_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa todo o pipeline de pré-processamento inicial (Limpeza Básica).
    """
    df = df.copy()
    df = clean_target(df)
    df = clean_numeric_columns(df)
    
    # Remover colunas inúteis para o modelo (IDs, Nomes, SSN)
    cols_to_drop = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Credit_Score']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    return df