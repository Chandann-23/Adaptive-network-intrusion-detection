import json
import os
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.make_dataset import download_nsl_kdd, load_dataset
from src.features.preprocessing import build_nsl_kdd_preprocessor
from src.features.validation import DataValidator


def prepare_labels(df: pd.DataFrame, target_col: str = "class") -> np.ndarray:
    """
    Encodes raw target labels into binary numerical values.
    'normal' maps to 0 (benign), and any other security alert maps to 1 (anomaly).
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    binary_labels = (df[target_col] != "normal").astype(int).to_numpy()
    return binary_labels  # type: ignore


def prepare_multiclass_labels(df: pd.DataFrame, target_col: str = "class") -> Any:
    """
    Encodes raw target labels into 5-class numerical values:
    0: Normal
    1: DoS (Denial of Service)
    2: Probe
    3: R2L (Remote to Local)
    4: U2R (User to Root)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
        
    attack_mapping = {
        # DoS
        "apache2": 1, "back": 1, "land": 1, "neptune": 1, "pod": 1, "smurf": 1, 
        "teardrop": 1, "mailbomb": 1, "processtable": 1, "udpstorm": 1, "worm": 1,
        # Probe
        "ipsweep": 2, "mscan": 2, "nmap": 2, "portsweep": 2, "saint": 2, "satan": 2,
        # R2L
        "ftp_write": 3, "guess_passwd": 3, "imap": 3, "multihop": 3, "phf": 3, 
        "spy": 3, "warezclient": 3, "warezmaster": 3, "sendmail": 3, "named": 3, 
        "snmpgetattack": 3, "snmpguess": 3, "xlock": 3, "xsnoop": 3, "httptunnel": 3,
        # U2R
        "buffer_overflow": 4, "loadmodule": 4, "perl": 4, "ps": 4, "rootkit": 4, 
        "sqlattack": 4, "xterm": 4,
        # Normal
        "normal": 0
    }
    
    raw_labels = df[target_col].astype(str).str.strip().str.lower()
    mapped_labels = raw_labels.map(lambda x: attack_mapping.get(x, 3)).to_numpy()
    return mapped_labels
from src.utils.logger import setup_logger

logger = setup_logger("build_features")

# Full expected raw columns in NSL-KDD
RAW_COLUMNS = [
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


def load_yaml_config(path: str) -> dict:
    """Loads configuration yaml file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_and_save_features(data_config_path: str = "configs/data_config.yaml"):
    """
    Executes full pipeline: validates data integrity, applies splits, transforms features
    with ColumnTransformers, and persists all processed Parquet datasets, binary pipeline,
    and metadata JSON files.
    """
    logger.info("Initializing Phase 2 feature preprocessing execution...")

    # 1. Load Configurations
    if not os.path.exists(data_config_path):
        raise FileNotFoundError(f"Missing data configuration at: {data_config_path}")
    data_cfg = load_yaml_config(data_config_path)

    paths_cfg = data_cfg['paths']
    feat_cfg = data_cfg['features']

    # 2. Ingest Data
    raw_dir = paths_cfg['raw_data_dir']
    processed_dir = paths_cfg['processed_data_dir']
    os.makedirs(processed_dir, exist_ok=True)

    # Verify/download datasets
    train_raw_path, test_raw_path = download_nsl_kdd(raw_dir)
    df_train_raw = load_dataset(train_raw_path)
    df_test_raw = load_dataset(test_raw_path)

    # 3. Invoke Data Validation Layer
    validator = DataValidator(expected_columns=RAW_COLUMNS, target_column=feat_cfg['target'])
    train_audit = validator.validate_schema(df_train_raw)
    test_audit = validator.validate_schema(df_test_raw)

    # 4. Prepare targets and features
    logger.info("Splitting features and target classes...")
    y_train_full = prepare_labels(df_train_raw, feat_cfg['target'])
    y_test = prepare_labels(df_test_raw, feat_cfg['target'])
    
    y_train_multi_full = prepare_multiclass_labels(df_train_raw, feat_cfg['target'])
    y_test_multi = prepare_multiclass_labels(df_test_raw, feat_cfg['target'])

    # Explicitly drop leakage columns from training inputs
    cols_to_drop = feat_cfg['columns_to_drop'] + [feat_cfg['target']]
    X_train_full = df_train_raw.drop(columns=cols_to_drop)
    X_test_raw = df_test_raw.drop(columns=cols_to_drop)

    # 5. Split train/validation sets (Stratified)
    split_cfg = data_cfg['splitting']
    X_train_raw, X_val_raw, y_train, y_val, y_train_multi, y_val_multi = train_test_split(
        X_train_full, y_train_full, y_train_multi_full,
        test_size=1.0 - split_cfg['train_ratio'],
        random_state=split_cfg['random_state'],
        stratify=y_train_full if split_cfg['stratify'] else None  # type: ignore
    )
    
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_train_multi = np.asarray(y_train_multi)
    y_val_multi = np.asarray(y_val_multi)

    logger.info(f"Split completed: Train size={X_train_raw.shape[0]}, Val size={X_val_raw.shape[0]}, Test size={X_test_raw.shape[0]}")  # type: ignore

    # 6. Build Preprocessor Pipeline
    preprocessor = build_nsl_kdd_preprocessor(
        categorical_cols=feat_cfg['categorical'],
        log_cols=feat_cfg['skewed_numerical'],
        robust_cols=feat_cfg['robust_scale_numerical'],
        rate_cols=feat_cfg['rate_numerical'],
        binary_cols=feat_cfg['binary']
    )

    # 7. Apply transformations (Fit strictly on Train, Transform all)
    logger.info("Fitting preprocessing pipeline on training features...")
    X_train_proc = preprocessor.fit_transform(X_train_raw)  
    logger.info("Transforming validation and test features...")
    X_val_proc = preprocessor.transform(X_val_raw)  # type: ignore
    X_test_proc = preprocessor.transform(X_test_raw)  # type: ignore

    # Ensure dense numpy array format to prevent sparse matrix DataFrame creation crashes
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()  # type: ignore
    if hasattr(X_val_proc, "toarray"):
        X_val_proc = X_val_proc.toarray()  # type: ignore
    if hasattr(X_test_proc, "toarray"):
        X_test_proc = X_test_proc.toarray()  # type: ignore

    # Fetch exact generated feature names after One-Hot expansion
    feature_names = preprocessor.get_feature_names_out().tolist()  # type: ignore
    logger.info(f"Feature transformation completed. Total features expanded to: {len(feature_names)}")

    # 8. Convert processed arrays back to DataFrames and append target labels
    df_train_proc = pd.DataFrame(X_train_proc, columns=feature_names)  # type: ignore
    df_train_proc['target_label'] = y_train.tolist()
    df_train_proc['multiclass_label'] = y_train_multi.tolist()

    df_val_proc = pd.DataFrame(X_val_proc, columns=feature_names)  # type: ignore
    df_val_proc['target_label'] = y_val.tolist()
    df_val_proc['multiclass_label'] = y_val_multi.tolist()

    df_test_proc = pd.DataFrame(X_test_proc, columns=feature_names)  # type: ignore
    df_test_proc['target_label'] = y_test.tolist()
    df_test_proc['multiclass_label'] = y_test_multi.tolist()

    # 9. Persist processed Parquet files
    train_parquet_path = os.path.join(processed_dir, "train_processed.parquet")
    val_parquet_path = os.path.join(processed_dir, "val_processed.parquet")
    test_parquet_path = os.path.join(processed_dir, "test_processed.parquet")

    logger.info(f"Writing train Parquet file to: {train_parquet_path}")
    df_train_proc.to_parquet(train_parquet_path, index=False)

    logger.info(f"Writing validation Parquet file to: {val_parquet_path}")
    df_val_proc.to_parquet(val_parquet_path, index=False)

    logger.info(f"Writing test Parquet file to: {test_parquet_path}")
    df_test_proc.to_parquet(test_parquet_path, index=False)

    # 10. Persist Pipeline PKL binary
    pipeline_save_path = os.path.join(processed_dir, "preprocessing_pipeline.joblib")
    logger.info(f"Serializing preprocessor binary to: {pipeline_save_path}")
    joblib.dump(preprocessor, pipeline_save_path)

    # 11. Persist Feature Names schema JSON
    feature_names_save_path = os.path.join(processed_dir, "feature_names.json")
    logger.info(f"Writing feature names checklist to: {feature_names_save_path}")
    with open(feature_names_save_path, 'w') as f:
        json.dump(feature_names, f, indent=2)

    # 12. Generate & Persist Metadata JSON
    metadata = {
        "dataset_version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_shapes": {
            "train_raw_samples": train_audit["num_samples"],
            "test_raw_samples": test_audit["num_samples"]
        },
        "processed_shapes": {
            "train_processed_samples": df_train_proc.shape[0],
            "val_processed_samples": df_val_proc.shape[0],
            "test_processed_samples": df_test_proc.shape[0],
            "features_before_processing": X_train_raw.shape[1],  # type: ignore
            "features_after_processing": len(feature_names)
        },
        "class_distribution_train": {
            "0_benign": (y_train == 0).sum().item(),
            "1_anomaly": (y_train == 1).sum().item()
        },
        "multiclass_distribution_train": {
            "0_normal": (y_train_multi == 0).sum().item(),
            "1_dos": (y_train_multi == 1).sum().item(),
            "2_probe": (y_train_multi == 2).sum().item(),
            "3_r2l": (y_train_multi == 3).sum().item(),
            "4_u2r": (y_train_multi == 4).sum().item()
        },
        "pipeline_steps": [
            "DropLeakageAndDeadColumns",
            "DataValidationPassed",
            "FrequencyConsolidateService",
            "OneHotEncodeCategoricals",
            "Log1pSkewedNumerics",
            "RobustScaleCounts"
        ]
    }

    metadata_save_path = os.path.join(processed_dir, "metadata.json")
    logger.info(f"Saving metadata manifest file to: {metadata_save_path}")
    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # 13. Run Inference Compatibility Audits & Report Generation
    from src.features.compatibility import (
        InferenceCompatibilityChecker,
        generate_compatibility_report_file,
    )
    checker = InferenceCompatibilityChecker(
        pipeline_path=pipeline_save_path,
        feature_names_path=feature_names_save_path
    )
    compatibility_report = checker.check_payload_drift(X_val_raw)  # type: ignore
    generate_compatibility_report_file(compatibility_report, "reports/inference_compatibility_report.md")

    logger.info("Phase 2 feature pipeline execution completed successfully!")


if __name__ == "__main__":
    build_and_save_features()
