"""Tests for model loading, prediction pipeline, and error handling."""

from pathlib import Path
import lightgbm as lgb
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import TabularDataPipeline
from src.model_loader import (
    ModelArtifactsNotFoundError,
    ModelBundle,
    are_artifacts_available,
    load_model_bundle,
)
from src.predict import (
    predict_csv,
    predict_passengers,
    predict_single_passenger,
)


@pytest.fixture
def dummy_model_bundle():
    """Builds a lightweight in-memory model bundle for testing inference."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "Gender": np.random.choice(["Male", "Female"], size=n),
        "Customer Type": np.random.choice(["Loyal Customer", "Disloyal Customer"], size=n),
        "Age": np.random.randint(18, 70, size=n),
        "Type of Travel": np.random.choice(["Business travel", "Personal Travel"], size=n),
        "Class": np.random.choice(["Business", "Eco", "Eco Plus"], size=n),
        "Flight Distance": np.random.choice([100, 200, 500, 1000], size=n),
        "Departure Delay in Minutes": [0] * n,
        "Arrival Delay in Minutes": [0] * n,
        "Inflight wifi service": [3] * n,
        "Departure/Arrival time convenient": [4] * n,
        "Ease of Online booking": [3] * n,
        "Gate location": [3] * n,
        "Food and drink": [4] * n,
        "Online boarding": [4] * n,
        "Seat comfort": [4] * n,
        "Inflight entertainment": [5] * n,
        "On-board service": [4] * n,
        "Leg room service": [4] * n,
        "Baggage handling": [5] * n,
        "Checkin service": [4] * n,
        "Cleanliness": [5] * n,
    })
    y = pd.Series(np.random.choice([0, 1], size=n))

    pipe = TabularDataPipeline(seed=42)
    pipe.fit(df, y)

    X_lgb = pipe.transform_lightgbm(df)
    model = LGBMClassifier(n_estimators=10, random_state=42, verbosity=-1)
    model.fit(X_lgb, y)

    bundle = ModelBundle(
        lgb_model=model,
        cat_model=None,
        preprocessor=pipe,
        ensemble_config={"weights": {"LightGBM": 1.0, "CatBoost": 0.0}},
        schema_config={"features": list(df.columns)},
    )
    return bundle, df


def test_missing_artifacts_raises():
    """Verify that attempting to load non-existent artifacts raises ModelArtifactsNotFoundError."""
    dummy_dir = Path("non_existent_folder_xyz_123")
    with pytest.raises(ModelArtifactsNotFoundError):
        load_model_bundle(models_dir=dummy_dir, artifacts_dir=dummy_dir)


def test_predict_single_passenger(dummy_model_bundle):
    bundle, sample_df = dummy_model_bundle
    passenger = sample_df.iloc[0].to_dict()

    result = predict_single_passenger(passenger, bundle=bundle)
    assert "satisfaction_probability" in result
    assert "prediction" in result
    assert "satisfaction_label" in result
    assert 0.0 <= result["satisfaction_probability"] <= 1.0
    assert result["prediction"] in (0, 1)
    assert result["satisfaction_label"] in ("Satisfied", "Neutral or Dissatisfied")


def test_predict_passengers_batch(dummy_model_bundle):
    bundle, sample_df = dummy_model_bundle
    batch_df = sample_df.copy()
    batch_df["id"] = range(1000, 1000 + len(batch_df))

    predictions = predict_passengers(batch_df, bundle=bundle)
    assert len(predictions) == len(batch_df)
    assert "id" in predictions.columns
    assert "satisfaction_probability" in predictions.columns
    assert "prediction" in predictions.columns
    assert "satisfaction_label" in predictions.columns
    assert (predictions["satisfaction_probability"].between(0.0, 1.0)).all()
    assert set(predictions["prediction"].unique()).issubset({0, 1})


def test_predict_csv_invalid_file_raises():
    with pytest.raises(FileNotFoundError):
        predict_csv("invalid_path_that_does_not_exist.csv")

