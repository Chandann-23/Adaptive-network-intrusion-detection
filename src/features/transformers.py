import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import setup_logger

logger = setup_logger("transformers")


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    State-preserving categorical transformer that consolidates low-frequency
    categories into a single 'other' label. This prevents high dimensionality
    and handles unseen categories gracefully during inference.
    """
    def __init__(self, threshold: float = 0.01, fill_value: str = "other"):
        self.threshold = threshold
        self.fill_value = fill_value
        self.frequent_categories_ = {}

    def fit(self, X, y=None):
        """
        Learns the frequent categories (frequency >= threshold) for each column.
        """
        X_df = pd.DataFrame(X)
        logger.info(f"Fitting FrequencyEncoder on {X_df.shape[1]} columns with threshold={self.threshold}")

        self.frequent_categories_ = {}
        for col in X_df.columns:
            # Calculate value ratios
            freqs = X_df[col].value_counts(normalize=True)
            frequent = freqs[freqs >= self.threshold].index.tolist()  # type: ignore

            # Ensure the fill value itself doesn't get overwritten in subsequent maps
            self.frequent_categories_[col] = frequent
            logger.debug(f"Column '{col}' frequent categories: {frequent}")

        return self

    def transform(self, X):
        """
        Maps rare and unseen categories to the configured 'other' fill value.
        """
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if col not in self.frequent_categories_:
                logger.warning(f"Column '{col}' was not seen during fitting. Mapping all values to '{self.fill_value}'")
                X_df[col] = self.fill_value
                continue

            frequent = self.frequent_categories_[col]
            # Replace rare/unseen with 'other'
            X_df[col] = X_df[col].apply(lambda val: str(val) if val in frequent else self.fill_value)

        return X_df

    def get_feature_names_out(self, input_features=None):
        """Returns feature names mapping for scikit-learn pipelines."""
        if input_features is None:
            return np.array(list(self.frequent_categories_.keys()), dtype=object)
        return np.asarray(input_features, dtype=object)


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """
    Standard element-wise natural log (log(x + 1)) transformer to reduce right-skewness
    in packet metrics, keeping outputs stable for downstream scalers and models.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.debug("Applying element-wise Log1p transformation.")
        # Ensure input is continuous floating numbers
        X_numeric = np.array(X, dtype=float)
        # Prevent math domain exceptions
        if (X_numeric < 0).any():
            raise ValueError("Log1pTransformer encountered negative values in the numerical matrix.")
        return np.log1p(X_numeric)

    def get_feature_names_out(self, input_features=None):
        """Returns feature names mapping for scikit-learn pipelines."""
        if input_features is None:
            return np.array([], dtype=object)
        return np.asarray(input_features, dtype=object)
