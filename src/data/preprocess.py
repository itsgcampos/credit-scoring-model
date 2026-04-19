from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
]

NUMERIC_RULES = {
    "Age": (18, 100),
    "Annual_Income": (1_000, 1_000_000),
    "Monthly_Inhand_Salary": (0, 200_000),
    "Num_Bank_Accounts": (0, 20),
    "Num_Credit_Card": (0, 20),
    "Interest_Rate": (0, 100),
    "Num_of_Loan": (0, 20),
    "Delay_from_due_date": (0, 365),
    "Num_of_Delayed_Payment": (0, 100),
    "Changed_Credit_Limit": (-100, 100),
    "Num_Credit_Inquiries": (0, 100),
    "Outstanding_Debt": (0, 1_000_000),
    "Credit_Utilization_Ratio": (0, 100),
    "Total_EMI_per_month": (0, 200_000),
    "Amount_invested_monthly": (0, 50_000),
    "Monthly_Balance": (0, 200_000),
}

INTEGER_COLUMNS = [
    "Age",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Num_Credit_Inquiries",
    "Credit_History_Months",
    "Month_Num",
    "target",
]

PLACEHOLDER_TOKENS = {
    "Occupation": {"_______": np.nan},
    "Credit_Mix": {"_": np.nan},
    "Payment_of_Min_Amount": {"NM": np.nan},
    "Payment_Behaviour": {"!@9#%8": np.nan},
}


def _clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        df[col] = df[col].apply(lambda value: value.strip() if isinstance(value, str) else value)
    return df


def _clean_numeric_series(series: pd.Series, lower: float | None, upper: float | None) -> pd.Series:
    cleaned = series.astype(str).str.replace("_", "", regex=False).str.replace(",", "", regex=False)
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if lower is not None:
        numeric = numeric.where(numeric >= lower)
    if upper is not None:
        numeric = numeric.where(numeric <= upper)
    return numeric


def _group_mode(series: pd.Series) -> object:
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else np.nan


def _fill_by_customer_mode(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        customer_mode = df.groupby("Customer_ID")[col].transform(_group_mode)
        df[col] = df[col].fillna(customer_mode)
    return df


def _fill_by_customer_median(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        customer_median = df.groupby("Customer_ID")[col].transform("median")
        df[col] = df[col].fillna(customer_median)
    return df


def _parse_credit_history_to_months(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.match(r"^\s*(\d+)\s+Years?\s+and\s+(\d+)\s+Months?\s*$", str(value))
    if not match:
        return np.nan
    years, months = match.groups()
    return int(years) * 12 + int(months)


def _preprocess_credit_history(df: pd.DataFrame) -> pd.DataFrame:
    df["Credit_History_Months"] = df["Credit_History_Age"].apply(_parse_credit_history_to_months)
    df["Credit_History_Months"] = (
        df.groupby("Customer_ID")["Credit_History_Months"]
        .transform(lambda series: series.interpolate(limit_direction="both"))
    )
    df["Credit_History_Months"] = df["Credit_History_Months"].fillna(df["Credit_History_Months"].median())
    return df


def _normalize_type_of_loan(series: pd.Series) -> pd.Series:
    return series.apply(
        lambda value: str(value).replace(", and ", ", ").replace(" and ", ", ")
        if isinstance(value, str)
        else value
    )


def process_pipeline(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Limpa a base bruta de credit scoring e devolve um dataset tratado para uso
    em analise e engenharia de features.
    """
    df = df.copy()
    df = _clean_string_columns(df)

    for col, replacements in PLACEHOLDER_TOKENS.items():
        df[col] = df[col].replace(replacements)

    df["Amount_invested_monthly"] = df["Amount_invested_monthly"].replace("__10000__", np.nan)
    df["Monthly_Balance"] = df["Monthly_Balance"].replace("__-333333333333333333333333333__", np.nan)

    month_map = {month: index + 1 for index, month in enumerate(MONTH_ORDER)}
    df["Month_Num"] = df["Month"].map(month_map)
    df = df.sort_values(["Customer_ID", "Month_Num", "ID"]).reset_index(drop=True)

    for col, (lower, upper) in NUMERIC_RULES.items():
        df[col] = _clean_numeric_series(df[col], lower, upper)

    df = _preprocess_credit_history(df)
    df["Type_of_Loan"] = _normalize_type_of_loan(df["Type_of_Loan"])

    stable_numeric_cols = [
        "Age",
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
    ]
    dynamic_numeric_cols = [
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Num_Credit_Inquiries",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
    ]
    stable_categorical_cols = ["Occupation", "Type_of_Loan"]
    dynamic_categorical_cols = ["Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]

    df = _fill_by_customer_median(df, stable_numeric_cols)
    df = _fill_by_customer_mode(df, stable_categorical_cols)
    df = _fill_by_customer_median(df, dynamic_numeric_cols)
    df = _fill_by_customer_mode(df, dynamic_categorical_cols)

    salary_from_income = df["Annual_Income"] / 12
    df["Monthly_Inhand_Salary"] = df["Monthly_Inhand_Salary"].fillna(salary_from_income)

    for col in stable_numeric_cols + dynamic_numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in stable_categorical_cols + dynamic_categorical_cols:
        df[col] = df[col].fillna("Unknown")

    if is_training:
        df["target"] = (df["Credit_Score"] == "Poor").astype(int)

    df = df.drop(
        columns=["ID", "Name", "SSN", "Credit_History_Age", "Credit_Score"],
        errors="ignore"
    )

    for col in INTEGER_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")

    ordered_columns = [
        "Customer_ID",
        "Month",
        "Month_Num",
        "Age",
        "Occupation",
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Type_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Num_Credit_Inquiries",
        "Credit_Mix",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Credit_History_Months",
        "Payment_of_Min_Amount",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Payment_Behaviour",
        "Monthly_Balance",
    ]

    if is_training:
        ordered_columns.append("target")

    return df[ordered_columns]
