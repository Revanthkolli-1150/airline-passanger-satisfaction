# Artifacts Directory

This directory stores reusable pipeline transformations, feature schema mappings, ensemble configurations, and competition submissions.

## Files
- `preprocessor_artifacts.joblib`: Persisted `TabularDataPipeline` containing fitted distance target encoding mappings, global prior, and exact one-hot encoded dummy column lists.
- `feature_schema.json`: Complete specification of raw features, engineered base features, categoricals, and model columns.
- `ensemble_config.json`: Configuration specifying model weights (LightGBM 0.475 + CatBoost 0.525) and active models.
- `airline_satisfaction_lgb_catboost_475_525.csv`: Generated competition submission file matching Kaggle format.

