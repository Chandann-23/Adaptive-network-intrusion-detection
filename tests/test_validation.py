import pandas as pd
import pytest

from src.features.validation import DataValidator, ValidationException


def test_validator_empty_dataset():
    """Verifies that an empty dataset triggers a ValidationException."""
    df_empty = pd.DataFrame()
    validator = DataValidator(expected_columns=["col1", "col2"])

    with pytest.raises(ValidationException, match="dataset is completely empty"):
        validator.validate_schema(df_empty)


def test_validator_missing_columns(mock_nsl_kdd_dataframe):
    """Verifies that a dataset missing expected columns triggers a ValidationException."""
    df = mock_nsl_kdd_dataframe.copy()
    # Drop required columns
    df = df.drop(columns=["duration", "protocol_type"])

    # Expected columns include duration
    expected_cols = ["duration", "protocol_type", "src_bytes"]
    validator = DataValidator(expected_columns=expected_cols)

    with pytest.raises(ValidationException, match="Missing columns"):
        validator.validate_schema(df)


def test_validator_type_inconsistency(mock_nsl_kdd_dataframe):
    """Verifies that numeric categories trigger ValidationException."""
    df = mock_nsl_kdd_dataframe.copy()
    # Convert expected categorical string columns to numeric values
    df["protocol_type"] = 1.0  # Mismatch

    expected_cols = ["duration", "protocol_type", "class"]
    validator = DataValidator(expected_columns=expected_cols)

    with pytest.raises(ValidationException, match="Type inconsistency"):
        validator.validate_schema(df)


def test_validator_success(mock_nsl_kdd_dataframe):
    """Verifies that a valid dataset schema passes successfully."""
    df = mock_nsl_kdd_dataframe
    validator = DataValidator(expected_columns=df.columns.tolist(), target_column="class")

    report = validator.validate_schema(df)

    assert report["status"] == "success"
    assert report["num_samples"] == len(df)
