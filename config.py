"""Centralized configuration for Airline Passenger Satisfaction ML Project.

All paths are relative to the project root directory.
Hyperparameters and constants strictly mirror the clean Colab notebook:
notebooks/Airline_Satisfaction_Clean_Model.ipynb
"""

from pathlib import Path

# Base Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# File Paths
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
SYNTHETIC_SAMPLE_PATH = DATA_DIR / "synthetic_sample_passengers.csv"

MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
FINAL_SUBMISSION_PATH = ARTIFACTS_DIR / "airline_satisfaction_lgb_catboost_475_525.csv"

# Model & Artifact File Paths
LGBM_MODEL_PATH = MODELS_DIR / "lightgbm_model.joblib"
CATBOOST_MODEL_PATH = MODELS_DIR / "catboost_model.joblib"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
ENSEMBLE_CONFIG_PATH = ARTIFACTS_DIR / "ensemble_config.json"
PREPROCESSOR_ARTIFACTS_PATH = ARTIFACTS_DIR / "preprocessor_artifacts.joblib"
FEATURE_SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
EVALUATION_METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"

# Global Constants
TARGET_COL = "satisfaction"
ID_COL = "id"
DISTANCE_COL = "Flight Distance"
RANDOM_SEED = 42
VALIDATION_SPLIT_SIZE = 0.20

# Ensemble Blend Weights (established from notebook validation & Kaggle benchmark)
# Benchmark Kaggle public leaderboard ROC-AUC: ~0.96032
FINAL_LGB_WEIGHT = 0.475
FINAL_CAT_WEIGHT = 0.525

# Target Encoding Parameters
TARGET_ENCODING_FOLDS = 5
TARGET_ENCODING_SMOOTHING = 20

# Survey Rating Columns (13 indicators, 0-5 scale)
RATING_COLS = [
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
]

# Domain specific rating sub-indices
COMFORT_RATING_COLS = [
    "Seat comfort",
    "Leg room service",
    "Inflight entertainment",
]

CONVENIENCE_RATING_COLS = [
    "Ease of Online booking",
    "Online boarding",
    "Departure/Arrival time convenient",
]

SERVICE_RATING_COLS = [
    "On-board service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]

# Delay Columns
DELAY_COLS = [
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

# Core Categorical Columns
CATEGORICAL_COLS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

# Model Hyperparameters mirroring notebook exactly
CATBOOST_PARAMS = {
    "iterations": 3000,
    "depth": 8,
    "learning_rate": 0.04,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "l2_leaf_reg": 5,
    "random_seed": RANDOM_SEED,
    "verbose": 250,
    "early_stopping_rounds": 150,
    "allow_writing_files": False,
    "thread_count": -1,
}

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 3000,
    "learning_rate": 0.025,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 50,
    "colsample_bytree": 0.85,
    "subsample": 0.85,
    "subsample_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 3.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}
