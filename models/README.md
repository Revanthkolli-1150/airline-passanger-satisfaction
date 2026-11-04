# Models Directory

This directory stores serialized model binaries produced by the training pipeline (`src/train.py`).

## Generated Artifacts
- `lightgbm_model.joblib`: Serialized LightGBM classifier fitted with tuned hyper-parameters (num_leaves=127, learning_rate=0.025, colsample_bytree=0.85, subsample=0.85).
- `catboost_model.joblib`: Serialized CatBoost classifier fitted with depth=8, l2_leaf_reg=5, learning_rate=0.04 (when CatBoost is available).
- `xgboost_model.joblib`: (Optional) Serialized XGBoost model.

## Model Reproducibility
- All models use fixed random seed `SEED = 42`.
- Models are trained on out-of-fold target encoded features and aligned one-hot categories.
- Files in this directory are git-ignored by default to prevent large binary bloat.

