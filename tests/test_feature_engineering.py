"""Tests for feature engineering, leakage safety, and tabular data pipelines."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    DistanceTargetEncoder,
    TabularDataPipeline,
    make_features,
)


@pytest.fixture
def sample_raw_dataframe():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "id": range(n),
        "Gender": np.random.choice(["Male", "Female"], size=n),
        "Customer Type": np.random.choice(["Loyal Customer", "Disloyal Customer"], size=n),
        "Age": np.random.randint(18, 70, size=n),
        "Type of Travel": np.random.choice(["Business travel", "Personal Travel"], size=n),
        "Class": np.random.choice(["Business", "Eco", "Eco Plus"], size=n),
        "Flight Distance": np.random.choice([100, 200, 500, 1000], size=n),
        "Departure Delay in Minutes": [0 if i % 2 == 0 else i for i in range(n)],
        "Arrival Delay in Minutes": [0 if i % 3 == 0 else i + 2 for i in range(n)],
        "Inflight wifi service": np.random.randint(0, 6, size=n),
        "Departure/Arrival time convenient": np.random.randint(0, 6, size=n),
        "Ease of Online booking": np.random.randint(0, 6, size=n),
        "Gate location": np.random.randint(0, 6, size=n),
        "Food and drink": np.random.randint(0, 6, size=n),
        "Online boarding": np.random.randint(0, 6, size=n),
        "Seat comfort": np.random.randint(0, 6, size=n),
        "Inflight entertainment": np.random.randint(0, 6, size=n),
        "On-board service": np.random.randint(0, 6, size=n),
        "Leg room service": np.random.randint(0, 6, size=n),
        "Baggage handling": np.random.randint(0, 6, size=n),
        "Checkin service": np.random.randint(0, 6, size=n),
        "Cleanliness": np.random.randint(0, 6, size=n),
    })


def test_make_features_delays_and_ratings(sample_raw_dataframe):
    features = make_features(sample_raw_dataframe)

    # Assert delay features
    assert "Total_Delay" in features.columns
    assert "Delay_Difference" in features.columns
    assert "Any_Delay" in features.columns
    assert (features["Any_Delay"].isin([0, 1])).all()

    # Assert log distance
    assert "Log_Flight_Distance" in features.columns
    assert (features["Log_Flight_Distance"] >= 0).all()

    # Assert ratings aggregation
    assert "Rating_Mean" in features.columns
    assert "Rating_Min" in features.columns
    assert "Rating_Max" in features.columns
    assert "Rating_Std" in features.columns
    assert "Low_Rating_Count" in features.columns
    assert "High_Rating_Count" in features.columns
    assert "Zero_Rating_Count" in features.columns

    # Assert domain scores & interactions
    assert "Comfort_Score" in features.columns
    assert "Convenience_Score" in features.columns
    assert "Service_Score" in features.columns
    assert "Comfort_x_Service" in features.columns
    assert "Convenience_x_Comfort" in features.columns
    assert "Age_x_Distance" in features.columns


def test_target_encoder_leakage_safety():
    """Verify that Out-Of-Fold target encoding doesn't leak target values."""
    # Construct a dataset where distance = 999 has a single unique target value
    # and all others have 0
    distances = pd.Series([100, 100, 100, 100, 200, 200, 200, 200, 300, 300, 300, 300, 400, 400, 400, 400])
    targets = pd.Series([0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1])

    encoder = DistanceTargetEncoder(smoothing=5.0, folds=4, seed=42)
    oof_encoded = encoder.fit_transform_oof(distances, targets)

    assert len(oof_encoded) == len(distances)
    # Check that OOF values are between 0 and 1
    assert (oof_encoded >= 0.0).all() and (oof_encoded <= 1.0).all()

    # Check transform on unseen distance uses global prior
    unseen_dist = pd.Series([99999])
    transformed_unseen = encoder.transform(unseen_dist)
    assert transformed_unseen.iloc[0] == pytest.approx(encoder.global_prior)

    # Check transform on test data does NOT require test labels
    test_distances = pd.Series([100, 200, 300])
    test_encoded = encoder.transform(test_distances)
    assert len(test_encoded) == 3


def test_tabular_data_pipeline_fit_transform(sample_raw_dataframe):
    np.random.seed(42)
    y = pd.Series(np.random.choice([0, 1], size=len(sample_raw_dataframe)))

    pipeline = TabularDataPipeline(seed=42)
    pipeline.fit(sample_raw_dataframe, y)

    assert pipeline.is_fitted

    # Test LightGBM transformation
    X_lgb = pipeline.transform_lightgbm(sample_raw_dataframe)
    assert "Distance_Target_Encoding" in X_lgb.columns
    assert X_lgb.shape[0] == len(sample_raw_dataframe)
    assert len(X_lgb.columns) == len(pipeline.lgb_encoded_columns)

    # Test CatBoost transformation
    X_cat = pipeline.transform_catboost(sample_raw_dataframe)
    assert X_cat.shape[0] == len(sample_raw_dataframe)
    assert "Total_Delay" in X_cat.columns

    # Test serialization and deserialization
    pipeline_dict = pipeline.to_dict()
    restored_pipeline = TabularDataPipeline.from_dict(pipeline_dict)
    X_lgb_restored = restored_pipeline.transform_lightgbm(sample_raw_dataframe)
    pd.testing.assert_frame_equal(X_lgb, X_lgb_restored)

