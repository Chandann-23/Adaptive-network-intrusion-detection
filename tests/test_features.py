import numpy as np

from src.features.preprocessing import build_nsl_kdd_preprocessor


def test_preprocessing_pipeline_dimensions(mock_nsl_kdd_dataframe):
    """
    Tests that the preprocessing pipeline successfully fits and transforms
    categorical, log-scaled, and robust-scaled columns, discarding columns
    configured for removal.
    """
    df = mock_nsl_kdd_dataframe
    X_raw = df.drop(columns=["class", "difficulty_score", "num_outbound_cmds"])

    # Configure mock categories
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

    # Fit and transform mock data
    X_processed = preprocessor.fit_transform(X_raw)

    # Verify dimensions and completeness
    assert X_processed.shape[0] == len(df)
    assert not np.isnan(X_processed).any(), "Preprocessing introduced NaN values."
    assert X_processed.shape[1] > len(log) + len(robust) + len(rate) + len(binary), \
        "One-hot expansion did not occur correctly."
