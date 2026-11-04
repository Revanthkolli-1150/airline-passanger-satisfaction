"""Prediction and inference engine for single passengers, batch dataframes, and uploaded CSVs."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from config import ID_COL, TARGET_COL
from src.model_loader import ModelBundle, load_model_bundle
from src.preprocessing import RAW_FEATURE_COLUMNS, clean_raw_inputs, validate_schema

logger = logging.getLogger(__name__)


def predict_passengers(
    df: pd.DataFrame,
    bundle: Optional[ModelBundle] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Predicts airline passenger satisfaction for a dataframe of passengers.

    Returns a DataFrame containing:
    - Probability (0.0 to 1.0)
    - Prediction (1: Satisfied, 0: Neutral or Dissatisfied)
    - Label ("Satisfied" vs "Neutral or Dissatisfied")
    - Original ID if present
    """
    if bundle is None:
        bundle = load_model_bundle()

    # Pre-clean inputs
    clean_df = clean_raw_inputs(df)

    # 1. LightGBM inference
    X_lgb = bundle.preprocessor.transform_lightgbm(clean_df)
    lgb_probs = bundle.lgb_model.predict_proba(X_lgb)[:, 1]

    # 2. CatBoost inference (if available)
    if bundle.has_catboost:
        X_cat = bundle.preprocessor.transform_catboost(clean_df)
        cat_probs = bundle.cat_model.predict_proba(X_cat)[:, 1]
        final_probs = bundle.lgb_weight * lgb_probs + bundle.cat_weight * cat_probs
    else:
        final_probs = lgb_probs

    # Clip to valid probability bounds
    final_probs = np.clip(final_probs, 0.0, 1.0)
    preds_binary = (final_probs >= threshold).astype(int)
    labels = [
        "Satisfied" if p == 1 else "Neutral or Dissatisfied"
        for p in preds_binary
    ]

    results = pd.DataFrame({
        "satisfaction_probability": np.round(final_probs, 5),
        "prediction": preds_binary,
        "satisfaction_label": labels,
    }, index=df.index)

    if ID_COL in df.columns:
        results.insert(0, ID_COL, df[ID_COL].values)

    return results


def predict_single_passenger(
    passenger_dict: Dict[str, Any],
    bundle: Optional[ModelBundle] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Infers satisfaction for a single passenger represented as a dictionary."""
    df = pd.DataFrame([passenger_dict])
    res_df = predict_passengers(df, bundle=bundle, threshold=threshold)
    row = res_df.iloc[0]

    return {
        "satisfaction_probability": float(row["satisfaction_probability"]),
        "prediction": int(row["prediction"]),
        "satisfaction_label": str(row["satisfaction_label"]),
        "model_used": "Ensemble (LightGBM + CatBoost)"
        if (bundle and bundle.has_catboost)
        else "LightGBM",
        "threshold": threshold,
    }


def predict_csv(
    input_csv_path: Union[str, Path],
    output_csv_path: Optional[Union[str, Path]] = None,
    bundle: Optional[ModelBundle] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Reads a CSV file, validates input structure, computes predictions, and optionally saves output."""
    input_path = Path(input_csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")

    # Validate presence of features
    is_valid, missing = validate_schema(df, is_training=False, require_id=False)
    if not is_valid:
        logger.warning(
            f"Some raw feature columns are missing from the input: {missing}. "
            f"They will be filled with standard default values."
        )

    results = predict_passengers(df, bundle=bundle, threshold=threshold)

    # Combine with original passenger data
    combined = pd.concat([df.reset_index(drop=True), results.drop(columns=[ID_COL], errors="ignore")], axis=1)

    if output_csv_path:
        out_p = Path(output_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_p, index=False)
        logger.info(f"Saved predictions to {out_p}")

    return combined

