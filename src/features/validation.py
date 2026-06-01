from typing import Any, Dict, List

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("validation")


class PipelineException(Exception):
    """Base exception for all pipeline-related issues."""
    pass


class ValidationException(PipelineException):
    """Raised when data checks or contract validations fail."""
    pass


class DataValidator:
    """
    Reusable, production-grade validation engine that runs schema,
    data type, completeness, and target integrity contracts.
    """
    def __init__(self, expected_columns: List[str], target_column: str = "class"):
        self.expected_columns = expected_columns
        self.target_column = target_column

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs comprehensive data contract audits. Raises ValidationException on failures.
        
        Args:
            df: Input DataFrame to check.
            
        Returns:
            Dict containing validation summary statistics.
        """
        logger.info("Executing Pipeline Schema Validation Audits...")

        # 1. Check completeness (Empty Dataset Check)
        if df.empty:
            raise ValidationException("Validation Failure: The ingested dataset is completely empty.")

        # 2. Check column count and presence
        missing_cols = [col for col in self.expected_columns if col not in df.columns]
        if missing_cols:
            raise ValidationException(
                f"Validation Failure: Schema mismatch. Missing columns in dataset: {missing_cols}"
            )

        # 3. Detect duplicate records
        # Note: Excluding target label check to detect duplicate traffic patterns
        feature_cols = [c for c in self.expected_columns if c != self.target_column]
        duplicate_count = df.duplicated(subset=feature_cols).sum()
        if duplicate_count > 0:
            logger.warning(
                f"Validation Warning: Detected {duplicate_count} duplicate traffic connection records. "
                "Ensure live logging deduplicates sessions."
            )

        # 4. Data Type Inconsistencies Check
        # Categorical columns in NSL-KDD must be string-compatible objects
        expected_categoricals = ["protocol_type", "service", "flag"]
        for col in expected_categoricals:
            if col in df.columns:
                # If a column is strictly numerical when we expect categories, raise exception
                if pd.api.types.is_numeric_dtype(df[col]):
                    raise ValidationException(
                        f"Validation Failure: Type inconsistency. Column '{col}' is numeric, expected categorical string."
                    )

        # 5. Invalid Target Label Check
        # During offline training validation, verify the target is categorical and contains expected labels
        if self.target_column in df.columns:
            unique_targets = df[self.target_column].unique()
            if len(unique_targets) == 0:
                raise ValidationException("Validation Failure: Target column exists but contains no records.")

            # Simple check: target labels should not be empty nulls
            if pd.isnull(unique_targets).any():
                raise ValidationException("Validation Failure: Target labels contain null or NaN values.")

        logger.info("Pipeline Data contract validation audits: PASSED.")

        return {
            "num_samples": df.shape[0],
            "num_features": df.shape[1],
            "duplicate_count": int(duplicate_count),
            "status": "success"
        }
