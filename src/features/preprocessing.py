from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.features.transformers import FrequencyEncoder, Log1pTransformer
from src.utils.logger import setup_logger

logger = setup_logger("preprocessing")


def build_nsl_kdd_preprocessor(
    categorical_cols: list,
    log_cols: list,
    robust_cols: list,
    rate_cols: list,
    binary_cols: list
) -> ColumnTransformer:
    """
    Assembles a production-grade ColumnTransformer pipeline:
    1. Consolidates rare service/flag categories and applies One-Hot Encoding.
    2. Log1p transforms and Robust-scales highly skewed byte/duration volumes.
    3. Robust-scales outlier-rich packet counter metrics.
    4. Passes through rate features and binary flags without scaling adjustments.
    
    Returns:
        Fitted or compilable ColumnTransformer pipeline.
    """
    logger.info("Assembling Preprocessing ColumnTransformer Pipelines...")

    # 1. Categorical Stream: Frequency Consolidation -> One-Hot Encoding
    categorical_pipeline = Pipeline([
        ('freq_consolidate', FrequencyEncoder(threshold=0.01, fill_value="other")),
        ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 2. Skewed Continuous Stream: Log1p -> RobustScaler
    skewed_pipeline = Pipeline([
        ('log1p', Log1pTransformer()),
        ('scaler', RobustScaler())
    ])

    # 3. Outlier Continuous Stream: RobustScaler
    robust_pipeline = Pipeline([
        ('scaler', RobustScaler())
    ])

    # 4. Final Combined Transformer Setup
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_pipeline, categorical_cols),
            ('log', skewed_pipeline, log_cols),
            ('robust', robust_pipeline, robust_cols),
            ('rate', 'passthrough', rate_cols),
            ('binary', 'passthrough', binary_cols)
        ],
        remainder='drop'  # Prevents leakage by dropping any undeclared features
    )

    logger.info("ColumnTransformer pipeline assembly successfully compiled.")
    return preprocessor
