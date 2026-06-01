import os
import urllib.request
from typing import Tuple

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("data_pipeline")

# Full 43-column NSL-KDD schema
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_hot_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "class", "difficulty_score"
]

DOWNLOAD_MIRROR = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master"


def download_nsl_kdd(target_dir: str = "data/raw") -> Tuple[str, str]:
    """
    Downloads training and test sets of NSL-KDD from standard mirrors if not present locally.
    
    Args:
        target_dir: Absolute or relative directory path to save raw files.
        
    Returns:
        Tuple of (train_file_path, test_file_path).
    """
    os.makedirs(target_dir, exist_ok=True)

    train_path = os.path.join(target_dir, "KDDTrain+.txt")
    test_path = os.path.join(target_dir, "KDDTest+.txt")

    for filename, local_path in [("KDDTrain+.txt", train_path), ("KDDTest+.txt", test_path)]:
        if not os.path.exists(local_path):
            url = f"{DOWNLOAD_MIRROR}/{filename}"
            logger.info(f"Downloading {filename} from official mirror: {url}")
            try:
                urllib.request.urlretrieve(url, local_path)
                logger.info(f"Successfully downloaded to {local_path}")
            except Exception as e:
                logger.error(f"Failed to fetch dataset from mirror: {e}")
                raise e
        else:
            logger.info(f"Local file verified at: {local_path}")

    return train_path, test_path


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads raw NSL-KDD files into structured pandas DataFrame applying standard column schemas.
    
    Args:
        file_path: Target path to KDD txt file.
        
    Returns:
        DataFrame with full column titles.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file for loading: {file_path}")

    logger.info(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path, names=NSL_KDD_COLUMNS, header=None)
    logger.info(f"Loaded DataFrame with dimensions: {df.shape}")
    return df
