import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_nsl_kdd_dataframe() -> pd.DataFrame:
    """
    Constructs a small, representative mock NSL-KDD dataset containing
    all 43 columns to facilitate offline unit testing of the data pipelines.
    """
    np.random.seed(42)
    rows = 10

    mock_data = {
        "duration": np.random.randint(0, 100, size=rows),
        "protocol_type": np.random.choice(["tcp", "udp", "icmp"], size=rows),
        "service": np.random.choice(["http", "smtp", "ftp", "private"], size=rows),
        "flag": np.random.choice(["SF", "S0", "REJ"], size=rows),
        "src_bytes": np.random.randint(100, 10000, size=rows),
        "dst_bytes": np.random.randint(100, 10000, size=rows),
        "land": np.random.choice([0, 1], size=rows),
        "wrong_fragment": np.random.choice([0, 1, 2], size=rows),
        "urgent": np.zeros(rows),
        "hot": np.random.randint(0, 3, size=rows),
        "num_failed_logins": np.random.choice([0, 1], size=rows),
        "logged_in": np.random.choice([0, 1], size=rows),
        "num_compromised": np.zeros(rows),
        "root_shell": np.zeros(rows),
        "su_attempted": np.zeros(rows),
        "num_root": np.zeros(rows),
        "num_file_creations": np.zeros(rows),
        "num_shells": np.zeros(rows),
        "num_access_files": np.zeros(rows),
        "num_outbound_cmds": np.zeros(rows),  # Dead column simulation
        "is_hot_login": np.zeros(rows),
        "is_guest_login": np.zeros(rows),
        "count": np.random.randint(1, 10, size=rows),
        "srv_count": np.random.randint(1, 10, size=rows),
        "serror_rate": np.random.rand(rows),
        "srv_serror_rate": np.random.rand(rows),
        "rerror_rate": np.random.rand(rows),
        "srv_rerror_rate": np.random.rand(rows),
        "same_srv_rate": np.random.rand(rows),
        "diff_srv_rate": np.random.rand(rows),
        "srv_diff_host_rate": np.random.rand(rows),
        "dst_host_count": np.random.randint(0, 255, size=rows),
        "dst_host_srv_count": np.random.randint(0, 255, size=rows),
        "dst_host_same_srv_rate": np.random.rand(rows),
        "dst_host_diff_srv_rate": np.random.rand(rows),
        "dst_host_same_src_port_rate": np.random.rand(rows),
        "dst_host_srv_diff_host_rate": np.random.rand(rows),
        "dst_host_serror_rate": np.random.rand(rows),
        "dst_host_srv_serror_rate": np.random.rand(rows),
        "dst_host_rerror_rate": np.random.rand(rows),
        "dst_host_srv_rerror_rate": np.random.rand(rows),
        "class": np.random.choice(["normal", "neptune", "satan", "warezclient"], size=rows),
        "difficulty_score": np.random.randint(1, 21, size=rows)
    }

    return pd.DataFrame(mock_data)
