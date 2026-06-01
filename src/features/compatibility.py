import json
import os
from typing import Any, Dict, List

import joblib
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("compatibility")


class InferenceCompatibilityChecker:
    """
    Engine designed to verify that the fitted offline training preprocessor
    and schema configurations are completely compatible with online production
    inference, preventing silent failures during deployment.
    """
    def __init__(self, pipeline_path: str, feature_names_path: str):
        self.pipeline_path = pipeline_path
        self.feature_names_path = feature_names_path

        # Load assets
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Fitted pipeline binary not found at: {pipeline_path}")
        if not os.path.exists(feature_names_path):
            raise FileNotFoundError(f"Feature name registry not found at: {feature_names_path}")

        self.preprocessor = joblib.load(pipeline_path)
        with open(feature_names_path, 'r') as f:
            self.expected_feature_names: List[str] = json.load(f)

    def check_payload_drift(self, df_payload: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates whether an incoming inference dataframe matches our fitted preprocessor
        feature expectations.
        
        Args:
            df_payload: DataFrame representing incoming live connections.
            
        Returns:
            Dict containing compatibility report logs.
        """
        logger.info("Initiating Production Inference Compatibility Audit...")

        issues = []
        status = "COMPATIBLE"

        # 1. Perform transformation check
        try:
            transformed = self.preprocessor.transform(df_payload)
            actual_features_out = self.preprocessor.get_feature_names_out().tolist()
        except Exception as e:
            status = "FAILED"
            issues.append(f"Transformation Pipeline Failure: {str(e)}")
            logger.error(f"Transformation Pipeline Crash detected: {e}")
            return {"status": status, "issues": issues}

        # 2. Check feature ordering and dimensionality
        if len(actual_features_out) != len(self.expected_feature_names):
            status = "INCOMPATIBLE"
            issues.append(
                f"Feature dimensionality mismatch. Preprocessor output: {len(actual_features_out)} cols, "
                f"Reference schema: {len(self.expected_feature_names)} cols."
            )

        # 3. Check for exact name ordering match
        for idx, (act, exp) in enumerate(zip(actual_features_out, self.expected_feature_names)):
            if act != exp:
                status = "DEVIATION_DETECTED"
                issues.append(f"Feature alignment drift at index {idx}: Expected '{exp}', Got '{act}'.")
                break

        # 4. Generate final report summary
        report = {
            "status": status,
            "issues": issues,
            "actual_feature_count": len(actual_features_out),
            "expected_feature_count": len(self.expected_feature_names)
        }

        logger.info(f"Compatibility Audit completed. Status: {status}")
        return report


def generate_compatibility_report_file(report: Dict[str, Any], output_path: str = "reports/inference_compatibility_report.md"):
    """
    Writes a professional markdown audit report to document production readiness.
    """
    logger.info(f"Writing compatibility report to: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    markdown_content = f"""# Production Inference Compatibility Report

This document reports on the compatibility audits between the **Fitted Offline Preprocessing Pipeline** and **Online Live Inference requirements**. It verifies that models will not crash or silently degrade when deployed to production.

---

## 1. Compatibility Audit Summary

| Metric / Parameter | Value / Status |
|--------------------|----------------|
| **Audited Timestamp** | {pd.Timestamp.utcnow().isoformat()}Z |
| **System Compatibility Status** | **{report['status']}** |
| **Transformed Columns Count** | {report['actual_feature_count']} |
| **Reference Schema Count** | {report['expected_feature_count']} |

---

## 2. Issues & Schema Deviations Detected

"""
    if report["status"] == "COMPATIBLE":
        markdown_content += "> [!NOTE]\n> **Success**: No schema anomalies or feature ordering drift were detected. The pipeline is fully compatible and ready to serve live traffic.\n"
    else:
        markdown_content += "> [!WARNING]\n> **Action Required**: The following compatibility anomalies were detected during the checkoff run:\n\n"
        for issue in report["issues"]:
            markdown_content += f"- **Issue**: {issue}\n"

    markdown_content += """
---

## 3. How This Prevents Production Failures

In traditional ML deployments, minor differences in data prep (e.g., column ordering or unseen categorical levels) cause massive server failures:
1. **Feature Alignment Bugs**: Tree-based models (such as XGBoost) perform evaluations on indices, not column titles. If training puts `src_bytes` at index 4, but inference places `count` at index 4 due to ordering drift, the model will output incorrect predictions without crashing (silent failure).
2. **Category Out-of-Bound Exceptions**: If a new, unseen network service triggers (e.g., a zero-day exploit using a rare port), standard One-Hot encoders will crash. Our pipeline handles this gracefully by automatically grouping unseen levels into `'other'` indices.
3. **Inference pipeline versioning**: Compiling matching parquet feature names verifies schema agreements before updating API weights.
"""

    with open(output_path, 'w') as f:
        f.write(markdown_content)
