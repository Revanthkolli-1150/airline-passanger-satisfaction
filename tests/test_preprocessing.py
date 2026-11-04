"""Tests for data preprocessing, schema validation, and target conversion."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    RAW_FEATURE_COLUMNS,
    clean_raw_inputs,
    convert_target,
    validate_schema,
)


def test_convert_target_binary_integers():
    s = pd.Series([0, 1, 1, 0, 1])
    res = convert_target(s)
    assert res.tolist() == [0, 1, 1, 0, 1]
    assert res.dtype == int


def test_convert_target_strings():
    s = pd.Series(["satisfied", "dissatisfied", "neutral or dissatisfied", "Satisfied "])
    res = convert_target(s)
    assert res.tolist() == [1, 0, 0, 1]


def test_convert_target_boolean():
    s = pd.Series([True, False, True])
    res = convert_target(s)
    assert res.tolist() == [1, 0, 1]


def test_convert_target_missing_raises():
    s = pd.Series(["satisfied", None, "dissatisfied"])
    with pytest.raises(ValueError, match="missing values"):
        convert_target(s)


def test_convert_target_invalid_labels_raises():
    s = pd.Series(["unknown_label", "satisfied"])
    with pytest.raises(ValueError, match="Unsupported target values"):
        convert_target(s)


def test_validate_schema():
    # Valid dataframe
    data = {col: [0] for col in RAW_FEATURE_COLUMNS}
    data["id"] = [101]
    data["satisfaction"] = ["satisfied"]
    df = pd.DataFrame(data)

    is_valid, missing = validate_schema(df, is_training=True, require_id=True)
    assert is_valid is True
    assert len(missing) == 0

    # Missing required target
    df_no_target = df.drop(columns=["satisfaction"])
    is_valid, missing = validate_schema(df_no_target, is_training=True)
    assert is_valid is False
    assert "satisfaction" in missing


def test_clean_raw_inputs():
    df = pd.DataFrame({
        "Gender": ["Male", None],
        "Age": ["45", "invalid_num"],
        "Flight Distance": ["1000", "2000"],
        "Departure Delay in Minutes": [None, "15"],
    })
    cleaned = clean_raw_inputs(df)
    assert cleaned["Gender"].tolist() == ["Male", "__MISSING__"]
    assert cleaned["Age"].iloc[0] == 45.0
    assert np.isnan(cleaned["Age"].iloc[1])
    assert cleaned["Flight Distance"].tolist() == [1000.0, 2000.0]
    assert np.isnan(cleaned["Departure Delay in Minutes"].iloc[0])
    assert cleaned["Departure Delay in Minutes"].iloc[1] == 15.0

