import pandas as pd
import numpy as np

def parse_credit_history_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte a coluna de texto '22 Years and 1 Months' para um valor inteiro (Total de Meses).
    """
    if 'Credit_History_Age' in df.columns:
        # Usa Regex para capturar os números de anos e meses
        extracted = df['Credit_History_Age'].str.extract(r'(\d+) Years and (\d+) Months')
        years = extracted[0].astype(float)
        months = extracted[1].astype(float)
        
        # Converte tudo para meses e cria a nova feature
        df['Credit_History_Age_Months'] = (years * 12) + months
        df = df.drop(columns=['Credit_History_Age'])
    return df

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche valores nulos. Em produção, salvaríamos esses valores (fit/transform),
    mas para a base estática usamos a mediana (numéricos) e moda/desconhecido (categóricos).
    """
    df = df.copy()
    
    # Numéricas: preencher nulos com a mediana da coluna
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0 and col != 'target':
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # Categóricas: preencher com 'Unknown'
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna('Unknown')
            
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma variáveis de texto em numéricas (One-Hot e Ordinal Encoding).
    """
    df = df.copy()
    
    # 1. Feature Extraction: Contar quantos tipos de empréstimo o cliente tem
    if 'Type_of_Loan' in df.columns:
        df['Loan_Types_Count'] = df['Type_of_Loan'].apply(
            lambda x: len(str(x).split(',')) if x != 'Unknown' else 0
        )
        df = df.drop(columns=['Type_of_Loan'])

    # 2. Ordinal Encoding: Variáveis que têm uma "ordem" de qualidade
    if 'Credit_Mix' in df.columns:
        # Mapeando do pior para o melhor
        map_credit_mix = {'Bad': 0, 'Standard': 1, 'Good': 2, 'Unknown': -1}
        df['Credit_Mix_Encoded'] = df['Credit_Mix'].map(map_credit_mix).fillna(-1)
        df = df.drop(columns=['Credit_Mix'])

    # 3. One-Hot Encoding: Variáveis nominais
    cols_to_dummy = ['Payment_of_Min_Amount', 'Occupation', 'Payment_Behaviour']
    cols_to_dummy = [c for c in cols_to_dummy if c in df.columns]
    
    if cols_to_dummy:
        # drop_first=True evita multicolinearidade (Dummy Variable Trap)
        df = pd.get_dummies(df, columns=cols_to_dummy, drop_first=True)
        
    return df

def build_features_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa todo o pipeline de engenharia de features.
    """
    print("\n[Feature Engineering] Iniciando construção de variáveis...")
    df = parse_credit_history_age(df)
    df = impute_missing_values(df)
    df = encode_categorical_features(df)
    
    # Converte colunas booleanas geradas pelo get_dummies para 0 e 1 (facilita pro XGBoost/LogReg)
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    print(f"[Feature Engineering] Concluído! Novo formato da base: {df.shape[0]} linhas e {df.shape[1]} colunas.")
    return df