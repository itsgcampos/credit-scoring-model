from __future__ import annotations

import numpy as np
import pandas as pd


def _count_loan_types(value: object) -> int:
    if pd.isna(value) or value == "Unknown":
        return 0
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    return len(parts)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def build_features_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features simples e interpretaveis a partir da base tratada.
    """
    df = df.copy()

    df["Loan_Type_Count"] = df["Type_of_Loan"].apply(_count_loan_types)
    df["Has_Min_Payment_Only"] = (df["Payment_of_Min_Amount"] == "Yes").astype(int)
    df["Is_Bad_Credit_Mix"] = (df["Credit_Mix"] == "Bad").astype(int)

    df["Debt_to_Income_Ratio"] = _safe_ratio(df["Outstanding_Debt"], df["Annual_Income"])
    df["EMI_to_Income_Ratio"] = _safe_ratio(df["Total_EMI_per_month"], df["Monthly_Inhand_Salary"])
    df["Invested_to_Income_Ratio"] = _safe_ratio(df["Amount_invested_monthly"], df["Monthly_Inhand_Salary"])
    df["Balance_to_Income_Ratio"] = _safe_ratio(df["Monthly_Balance"], df["Monthly_Inhand_Salary"])

    df["Delay_x_Inquiries"] = df["Delay_from_due_date"] * df["Num_Credit_Inquiries"]
    df["Utilization_x_Debt"] = df["Credit_Utilization_Ratio"] * df["Outstanding_Debt"]

    numeric_feature_cols = [
        "Debt_to_Income_Ratio",
        "EMI_to_Income_Ratio",
        "Invested_to_Income_Ratio",
        "Balance_to_Income_Ratio",
        "Delay_x_Inquiries",
        "Utilization_x_Debt",
    ]
    for col in numeric_feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())

    return df
