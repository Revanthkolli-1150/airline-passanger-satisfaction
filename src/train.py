"""Training pipeline for Airline Passenger Satisfaction models (LightGBM, CatBoost, Ensemble)."""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import lightgbm as lgb
from lightgbm import LGBMClassifier

from config import (
    CATBOOST_MODEL_PATH,
    CATBOOST_PARAMS,
    DISTANCE_COL,
    ENSEMBLE_CONFIG_PATH,
    EVALUATION_METRICS_PATH,
    FEATURE_SCHEMA_PATH,
    FINAL_CAT_WEIGHT,
    FINAL_LGB_WEIGHT,
    LGBM_MODEL_PATH,
    LIGHTGBM_PARAMS,
    MODEL_COMPARISON_PATH,
    MODELS_DIR,
    ARTIFACTS_DIR,
    PREPROCESSOR_ARTIFACTS_PATH,
    RANDOM_SEED,
    TARGET_COL,
    TRAIN_DATA_PATH,
    VALIDATION_SPLIT_SIZE,
)
from src.evaluation import compute_classification_metrics, save_model_comparison_report
from src.feature_engineering import DistanceTargetEncoder, TabularDataPipeline, make_features
from src.preprocessing import clean_raw_inputs, convert_target, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Check if CatBoost is installed
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost is not installed in the current Python environment.")

# Check if XGBoost is installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def train_pipeline(
    train_data_path: Path = TRAIN_DATA_PATH,
    output_models_dir: Path = MODELS_DIR,
    output_artifacts_dir: Path = ARTIFACTS_DIR,
    save_artifacts: bool = True,
    quick_run: bool = False,
) -> Dict[str, Any]:
    """Runs the complete training and validation pipeline.

    1. Loads raw training data and converts target.
    2. Runs feature engineering and builds stratified train-validation split.
    3. Fits leakage-safe target encoder on Flight Distance.
    4. Trains LightGBM model with early stopping.
    5. Trains CatBoost model (if available) with early stopping.
    6. Evaluates individual models and the weighted ensemble blend.
    7. Retrains on all training rows (or saves best estimator).
    8. Serializes model artifacts, preprocessor, schema, and evaluation reports.
    """
    if not Path(train_data_path).exists():
        raise FileNotFoundError(
            f"Training dataset not found at '{train_data_path}'. "
            f"Please place train.csv into the 'data/' directory."
        )

    output_models_dir.mkdir(parents=True, exist_ok=True)
    output_artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading training data from {train_data_path}...")
    raw_df = pd.read_csv(train_data_path)
    if TARGET_COL not in raw_df.columns:
        raise ValueError(f"Missing required target column '{TARGET_COL}'.")

    y = convert_target(raw_df[TARGET_COL])
    X_raw = raw_df.drop(columns=[TARGET_COL], errors="ignore").copy()

    # Step 1: Base Feature Engineering
    logger.info("Computing engineered features...")
    X_base = make_features(X_raw)

    cat_cols = X_base.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()
    for col in cat_cols:
        X_base[col] = X_base[col].fillna("__MISSING__").astype(str)

    logger.info(f"Base feature matrix shape: {X_base.shape}, Categorical columns: {len(cat_cols)}")

    # Step 2: Stratified Validation Split (80/20)
    all_idx = np.arange(len(X_base))
    train_idx, valid_idx = train_test_split(
        all_idx,
        test_size=VALIDATION_SPLIT_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    X_tr = X_base.iloc[train_idx].copy()
    X_va = X_base.iloc[valid_idx].copy()
    y_tr = y.iloc[train_idx].copy()
    y_va = y.iloc[valid_idx].copy()

    logger.info(f"Train split: {X_tr.shape}, Validation split: {X_va.shape}")

    metrics_list = []

    # Step 3: Leakage-Safe Target Encoding on Flight Distance for LightGBM
    target_encoder_val = DistanceTargetEncoder(seed=RANDOM_SEED)
    X_tr_lgb = X_tr.copy()
    X_va_lgb = X_va.copy()

    logger.info("Fitting out-of-fold target encoding for LightGBM...")
    X_tr_lgb["Distance_Target_Encoding"] = target_encoder_val.fit_transform_oof(
        X_tr_lgb[DISTANCE_COL], y_tr
    )
    X_va_lgb["Distance_Target_Encoding"] = target_encoder_val.transform(
        X_va_lgb[DISTANCE_COL]
    )

    # One-hot encoding for LightGBM
    lgb_combined = pd.concat([X_tr_lgb, X_va_lgb], ignore_index=True)
    lgb_combined_enc = pd.get_dummies(
        lgb_combined, columns=cat_cols, dummy_na=False, dtype=np.uint8
    )
    X_tr_enc = lgb_combined_enc.iloc[:len(X_tr_lgb)].copy()
    X_va_enc = lgb_combined_enc.iloc[len(X_tr_lgb):].copy()
    lgb_feature_names = X_tr_enc.columns.tolist()

    # Step 4: Train LightGBM
    logger.info("Training LightGBM model...")
    lgb_params = LIGHTGBM_PARAMS.copy()
    if quick_run:
        lgb_params["n_estimators"] = 50

    start_lgb = time.time()
    lgb_model = LGBMClassifier(**lgb_params)
    callbacks = [
        lgb.early_stopping(150, verbose=False),
        lgb.log_evaluation(250),
    ]
    lgb_model.fit(
        X_tr_enc,
        y_tr,
        eval_set=[(X_va_enc, y_va)],
        callbacks=callbacks,
    )
    dur_lgb = time.time() - start_lgb
    lgb_valid_pred = lgb_model.predict_proba(X_va_enc)[:, 1]
    lgb_metrics = compute_classification_metrics(
        y_va, lgb_valid_pred, model_name="LightGBM", duration_seconds=dur_lgb
    )
    metrics_list.append(lgb_metrics)
    logger.info(f"LightGBM Validation ROC-AUC: {lgb_metrics['roc_auc']}")

    # Step 5: Train CatBoost (if available)
    cat_valid_pred = None
    cat_model = None
    if CATBOOST_AVAILABLE:
        logger.info("Training CatBoost model...")
        cat_params = CATBOOST_PARAMS.copy()
        if quick_run:
            cat_params["iterations"] = 50
        start_cat = time.time()
        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(
            X_tr,
            y_tr,
            cat_features=cat_cols,
            eval_set=(X_va, y_va),
            use_best_model=True,
            verbose=250,
        )
        dur_cat = time.time() - start_cat
        cat_valid_pred = cat_model.predict_proba(X_va)[:, 1]
        cat_metrics = compute_classification_metrics(
            y_va, cat_valid_pred, model_name="CatBoost", duration_seconds=dur_cat
        )
        metrics_list.append(cat_metrics)
        logger.info(f"CatBoost Validation ROC-AUC: {cat_metrics['roc_auc']}")
    else:
        logger.warning(
            "CatBoost is not available. Ensemble will use LightGBM alone or with documented fallback."
        )

    # Step 6: Ensemble evaluation
    ensemble_weights = {"LightGBM": 1.0, "CatBoost": 0.0}
    if cat_valid_pred is not None:
        blend_pred = FINAL_LGB_WEIGHT * lgb_valid_pred + FINAL_CAT_WEIGHT * cat_valid_pred
        ensemble_weights = {
            "LightGBM": FINAL_LGB_WEIGHT,
            "CatBoost": FINAL_CAT_WEIGHT,
        }
        ensemble_metrics = compute_classification_metrics(
            y_va,
            blend_pred,
            model_name="Ensemble (LightGBM 0.475 + CatBoost 0.525)",
            duration_seconds=dur_lgb + (dur_cat if CATBOOST_AVAILABLE else 0),
        )
        metrics_list.append(ensemble_metrics)
        logger.info(f"Ensemble Validation ROC-AUC: {ensemble_metrics['roc_auc']}")

    # Save validation reports
    comparison_df = save_model_comparison_report(metrics_list)

    # Step 7: Train Final Models on all rows and serialize
    logger.info("Fitting final preprocessor on entire dataset...")
    pipeline = TabularDataPipeline(seed=RANDOM_SEED)
    pipeline.fit(X_raw, y)

    # Train final LightGBM on all rows
    X_full_lgb = pipeline.transform_lightgbm(X_raw)
    final_lgb_params = LIGHTGBM_PARAMS.copy()
    best_iter = getattr(lgb_model, "best_iteration_", None) or 1000
    final_lgb_params["n_estimators"] = best_iter if not quick_run else 50
    final_lgb = LGBMClassifier(**final_lgb_params)
    logger.info(f"Fitting final LightGBM on all data with {final_lgb_params['n_estimators']} trees...")
    final_lgb.fit(X_full_lgb, y)

    # Train final CatBoost if available
    final_cat = None
    if CATBOOST_AVAILABLE:
        X_full_cat = pipeline.transform_catboost(X_raw)
        final_cat_params = CATBOOST_PARAMS.copy()
        best_cat_iter = getattr(cat_model, "best_iteration_", None)
        final_cat_params["iterations"] = (
            int(best_cat_iter + 1) if best_cat_iter is not None else 1000
        )
        if quick_run:
            final_cat_params["iterations"] = 50
        final_cat = CatBoostClassifier(**final_cat_params)
        logger.info(f"Fitting final CatBoost on all data with {final_cat_params['iterations']} trees...")
        final_cat.fit(X_full_cat, y, cat_features=cat_cols, verbose=250)

    if save_artifacts:
        logger.info("Saving model artifacts...")
        joblib.dump(final_lgb, LGBM_MODEL_PATH)
        logger.info(f"Saved LightGBM model to {LGBM_MODEL_PATH}")

        if final_cat is not None:
            joblib.dump(final_cat, CATBOOST_MODEL_PATH)
            logger.info(f"Saved CatBoost model to {CATBOOST_MODEL_PATH}")

        # Save preprocessor
        joblib.dump(pipeline.to_dict(), PREPROCESSOR_ARTIFACTS_PATH)
        logger.info(f"Saved preprocessor artifacts to {PREPROCESSOR_ARTIFACTS_PATH}")

        # Save feature schema
        schema_info = {
            "raw_feature_columns": list(X_raw.columns),
            "base_feature_columns": pipeline.base_columns,
            "categorical_columns": pipeline.cat_cols,
            "lgb_encoded_columns": pipeline.lgb_encoded_columns,
            "target_column": TARGET_COL,
            "random_seed": RANDOM_SEED,
        }
        with open(FEATURE_SCHEMA_PATH, "w", encoding="utf-8") as f:
            json.dump(schema_info, f, indent=2)
        logger.info(f"Saved feature schema to {FEATURE_SCHEMA_PATH}")

        # Save ensemble config
        ensemble_config = {
            "weights": ensemble_weights,
            "models_present": {
                "LightGBM": True,
                "CatBoost": final_cat is not None,
            },
            "established_kaggle_leaderboard_benchmark": 0.96032,
            "benchmark_note": "Kaggle public leaderboard result (not local validation).",
        }
        with open(ENSEMBLE_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(ensemble_config, f, indent=2)
        logger.info(f"Saved ensemble configuration to {ENSEMBLE_CONFIG_PATH}")

    return {
        "metrics": metrics_list,
        "ensemble_weights": ensemble_weights,
        "lgb_model": final_lgb,
        "cat_model": final_cat,
        "preprocessor": pipeline,
    }


if __name__ == "__main__":
    train_pipeline()

