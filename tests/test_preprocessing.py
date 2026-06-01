import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import build_nsl_kdd_preprocessor
from src.features.transformers import FrequencyEncoder, Log1pTransformer


def test_frequency_encoder_rare_consolidation():
    """
    Tests that FrequencyEncoder groups low-frequency categories under the
    fill value, while preserving high-frequency ones.
    """
    # Sample series with protocols: 9 'tcp', 1 'icmp'
    data = pd.DataFrame({"protocol": ["tcp"] * 9 + ["icmp"]})

    # Setting threshold at 15% (icmp has 10%, should be consolidation target)
    encoder = FrequencyEncoder(threshold=0.15, fill_value="other")
    encoder.fit(data)
    transformed = encoder.transform(data)

    assert "tcp" in encoder.frequent_categories_["protocol"]
    assert "icmp" not in encoder.frequent_categories_["protocol"]
    assert transformed["protocol"].iloc[9] == "other"


def test_frequency_encoder_unseen_categories():
    """
    Tests that FrequencyEncoder maps unseen test categories to 'other' safely.
    """
    train_data = pd.DataFrame({"service": ["http"] * 10})
    test_data = pd.DataFrame({"service": ["ssh"]})  # Unseen

    encoder = FrequencyEncoder(threshold=0.01, fill_value="other")
    encoder.fit(train_data)
    transformed = encoder.transform(test_data)

    assert transformed["service"].iloc[0] == "other"


def test_log1p_transformer_math():
    """Tests element-wise Log1p operations and negative checks."""
    data = np.array([0.0, 9.0, 99.0])
    transformer = Log1pTransformer()
    transformed = transformer.transform(data)

    # log1p(0) = 0, log1p(9) = log(10) ~ 2.3025, log1p(99) = log(100) ~ 4.605
    assert np.allclose(transformed, [0.0, np.log(10), np.log(100)])

    # Negative value checks
    with pytest.raises(ValueError, match="negative values"):
        transformer.transform([-1.0, 5.0])


def test_pipeline_serialization_and_inference(mock_nsl_kdd_dataframe):
    """
    Tests that the preprocessor pipeline compiles, fits, transforms, serializes,
    and is loadable to perform consistent inference transformations.
    """
    df = mock_nsl_kdd_dataframe
    X_raw = df.drop(columns=["class", "difficulty_score", "num_outbound_cmds"])

    # Configure groups
    cat = ["protocol_type", "service", "flag"]
    log = ["duration", "src_bytes", "dst_bytes"]
    robust = ["count", "srv_count", "hot", "num_failed_logins", "num_compromised"]
    rate = ["serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate"]
    binary = ["land", "logged_in", "root_shell", "is_hot_login", "is_guest_login"]

    preprocessor = build_nsl_kdd_preprocessor(
        categorical_cols=cat,
        log_cols=log,
        robust_cols=robust,
        rate_cols=rate,
        binary_cols=binary
    )

    X_train_proc = preprocessor.fit_transform(X_raw)

    # Save the pipeline to temporary storage
    with tempfile.TemporaryDirectory() as tmp_dir:
        pkl_path = os.path.join(tmp_dir, "test_pipeline.joblib")
        joblib.dump(preprocessor, pkl_path)

        # Load pipeline back
        loaded_preprocessor = joblib.load(pkl_path)

        # Transform again (inference simulation)
        X_infer_proc = loaded_preprocessor.transform(X_raw)

        # Assert absolute equality between fits and inference mappings
        assert np.allclose(X_train_proc, X_infer_proc)
