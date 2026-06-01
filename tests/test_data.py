import numpy as np

from src.features.build_features import prepare_labels


def test_prepare_labels_binary_mapping(mock_nsl_kdd_dataframe):
    """
    Tests that label preparation correctly maps 'normal' to 0 and
    anomalous class names (like 'neptune', 'satan') to 1.
    """
    df = mock_nsl_kdd_dataframe
    labels = prepare_labels(df, target_col="class")

    assert len(labels) == len(df)
    assert labels.dtype == np.int64 or labels.dtype == np.int32

    # Confirm exact mapping
    for i in range(len(df)):
        if df.iloc[i]["class"] == "normal":
            assert labels[i] == 0
        else:
            assert labels[i] == 1
