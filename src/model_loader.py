"""Model artifact loading and integrity verification."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib

from config import (
    ARTIFACTS_DIR,
    CATBOOST_MODEL_PATH,
    ENSEMBLE_CONFIG_PATH,
    FEATURE_SCHEMA_PATH,
    FINAL_CAT_WEIGHT,
    FINAL_LGB_WEIGHT,
    LGBM_MODEL_PATH,
    MODELS_DIR,
    PREPROCESSOR_ARTIFACTS_PATH,
)
from src.feature_engineering import TabularDataPipeline

logger = logging.getLogger(__name__)


class ModelArtifactsNotFoundError(FileNotFoundError):
    """Raised when expected trained model artifacts are not found on disk."""
    pass


class ModelBundle:
    """Holds loaded model artifacts, preprocessor, and ensemble configuration."""

    def __init__(
        self,
        lgb_model: Any,
        cat_model: Optional[Any],
        preprocessor: TabularDataPipeline,
        ensemble_config: Dict[str, Any],
        schema_config: Dict[str, Any],
    ):
        self.lgb_model = lgb_model
        self.cat_model = cat_model
        self.preprocessor = preprocessor
        self.ensemble_config = ensemble_config
        self.schema_config = schema_config

    @property
    def has_catboost(self) -> bool:
        return self.cat_model is not None

    @property
    def lgb_weight(self) -> float:
        weights = self.ensemble_config.get("weights", {})
        return float(weights.get("LightGBM", FINAL_LGB_WEIGHT if self.has_catboost else 1.0))

    @property
    def cat_weight(self) -> float:
        weights = self.ensemble_config.get("weights", {})
        return float(weights.get("CatBoost", FINAL_CAT_WEIGHT if self.has_catboost else 0.0))


def are_artifacts_available() -> Tuple[bool, Dict[str, bool]]:
    """Checks the availability of model and preprocessor artifacts."""
    status = {
        "lightgbm_model": LGBM_MODEL_PATH.exists(),
        "catboost_model": CATBOOST_MODEL_PATH.exists(),
        "preprocessor_artifacts": PREPROCESSOR_ARTIFACTS_PATH.exists(),
        "feature_schema": FEATURE_SCHEMA_PATH.exists(),
        "ensemble_config": ENSEMBLE_CONFIG_PATH.exists(),
    }
    # Base requirements: LightGBM + preprocessor + schema
    core_ready = (
        status["lightgbm_model"]
        and status["preprocessor_artifacts"]
        and status["feature_schema"]
    )
    return core_ready, status


def load_model_bundle(
    models_dir: Path = MODELS_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> ModelBundle:
    """Loads all model artifacts, preprocessors, and configurations.

    Raises:
        ModelArtifactsNotFoundError: If trained model files or preprocessor are missing.
    """
    ready, status = are_artifacts_available()
    if not ready:
        missing = [k for k, v in status.items() if not v and k != "catboost_model"]
        msg = (
            f"Required trained model artifacts are missing: {missing}.\n"
            f"Please run the model training pipeline first using:\n"
            f"  python -m src.train\n"
            f"Or ensure training data is in 'data/train.csv'."
        )
        raise ModelArtifactsNotFoundError(msg)

    # 1. Load Preprocessor
    logger.info(f"Loading preprocessor from {PREPROCESSOR_ARTIFACTS_PATH}")
    prep_data = joblib.load(PREPROCESSOR_ARTIFACTS_PATH)
    preprocessor = TabularDataPipeline.from_dict(prep_data)

    # 2. Load LightGBM
    logger.info(f"Loading LightGBM model from {LGBM_MODEL_PATH}")
    lgb_model = joblib.load(LGBM_MODEL_PATH)

    # 3. Load CatBoost (optional)
    cat_model = None
    if CATBOOST_MODEL_PATH.exists():
        logger.info(f"Loading CatBoost model from {CATBOOST_MODEL_PATH}")
        try:
            cat_model = joblib.load(CATBOOST_MODEL_PATH)
        except Exception as e:
            logger.warning(f"Could not load CatBoost model: {e}")

    # 4. Load Schema Config
    with open(FEATURE_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_config = json.load(f)

    # 5. Load Ensemble Config
    if ENSEMBLE_CONFIG_PATH.exists():
        with open(ENSEMBLE_CONFIG_PATH, "r", encoding="utf-8") as f:
            ensemble_config = json.load(f)
    else:
        ensemble_config = {
            "weights": {
                "LightGBM": FINAL_LGB_WEIGHT if cat_model else 1.0,
                "CatBoost": FINAL_CAT_WEIGHT if cat_model else 0.0,
            }
        }

    return ModelBundle(
        lgb_model=lgb_model,
        cat_model=cat_model,
        preprocessor=preprocessor,
        ensemble_config=ensemble_config,
        schema_config=schema_config,
    )

