from __future__ import annotations

import numpy as np
import pandas as pd


def _count_loan_types(value: object) -> int:
    if pd.isna(value) or value == "Unknown":
        return 0
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    return len(parts)


def build_features_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features simples e interpretaveis a partir da base tratada.
    """
    df = df.copy()

    df["Loan_Type_Count"] = df["Type_of_Loan"].apply(_count_loan_types)
    df["Has_Min_Payment_Only"] = (df["Payment_of_Min_Amount"] == "Yes").astype(int)
    df["Debt_to_Income_Ratio"] = df["Outstanding_Debt"] / df["Annual_Income"].replace(0, np.nan)
    df["EMI_to_Income_Ratio"] = df["Total_EMI_per_month"] / df["Monthly_Inhand_Salary"].replace(0, np.nan)
    df["Invested_to_Income_Ratio"] = df["Amount_invested_monthly"] / df["Monthly_Inhand_Salary"].replace(0, np.nan)
    df["Balance_to_Income_Ratio"] = df["Monthly_Balance"] / df["Monthly_Inhand_Salary"].replace(0, np.nan)

    ratio_cols = [
        "Debt_to_Income_Ratio",
        "EMI_to_Income_Ratio",
        "Invested_to_Income_Ratio",
        "Balance_to_Income_Ratio",
    ]
    for col in ratio_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())

    return df
