"""Data loading, schema validation, and missing value handling for Airline Passenger Satisfaction."""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_COLS,
    DELAY_COLS,
    DISTANCE_COL,
    ID_COL,
    RATING_COLS,
    TARGET_COL,
)

logger = logging.getLogger(__name__)

RAW_FEATURE_COLUMNS = [
    "Gender",
    "Customer Type",
    "Age",
    "Type of Travel",
    "Class",
    "Flight Distance",
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]


def convert_target(series: pd.Series) -> pd.Series:
    """Safely converts target series to binary integer format (0 or 1).

    Supports boolean, numeric 0/1, strings ('satisfied'/'dissatisfied',
    'neutral or dissatisfied', 'true'/'false').

    Raises:
        ValueError: If series contains NaNs or unsupported values.
    """
    if series.isna().any():
        raise ValueError("Target contains missing values.")

    if pd.api.types.is_bool_dtype(series):
        out = series.astype(int)
    elif set(pd.unique(series)).issubset({0, 1}):
        out = series.astype(int)
    else:
        mapping = {
            "satisfied": 1,
            "dissatisfied": 0,
            "neutral or dissatisfied": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        }
        cleaned = series.astype(str).str.strip().str.lower()
        out = cleaned.map(mapping)

    if out.isna().any() or out.nunique() > 2:
        unmapped = series[out.isna()].unique()[:5]
        raise ValueError(f"Unsupported target values encountered: {unmapped}")

    return out.astype(int).reset_index(drop=True)


def validate_schema(
    df: pd.DataFrame,
    is_training: bool = False,
    require_id: bool = False,
) -> Tuple[bool, List[str]]:
    """Validates dataframe columns against required schema.

    Returns:
        (is_valid, missing_columns)
    """
    missing = []
    if is_training and TARGET_COL not in df.columns:
        missing.append(TARGET_COL)

    if require_id and ID_COL not in df.columns:
        missing.append(ID_COL)

    # Check key expected raw features
    for col in RAW_FEATURE_COLUMNS:
        if col not in df.columns:
            missing.append(col)

    return len(missing) == 0, missing


def clean_raw_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes types, cleans missing values, and prepares raw dataframe."""
    df_clean = df.copy()

    # Coerce numeric columns
    numeric_candidates = [
        "Age",
        DISTANCE_COL,
        *DELAY_COLS,
        *RATING_COLS,
    ]
    for col in numeric_candidates:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Clean categorical columns
    for col in CATEGORICAL_COLS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("__MISSING__").astype(str).str.strip()

    return df_clean


def load_dataset(
    file_path: Union[str, pd.DataFrame],
    is_training: bool = False,
) -> pd.DataFrame:
    """Loads a CSV or returns dataframe with initial validation."""
    if isinstance(file_path, pd.DataFrame):
        df = file_path.copy()
    else:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)

    if is_training and TARGET_COL not in df.columns:
        raise ValueError(f"Training data must contain target column '{TARGET_COL}'.")

    return clean_raw_inputs(df)

