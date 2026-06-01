"""
NIDS Telemetry Predictor
========================
Binds preprocessors and locked hybrid classifiers, performing real-time
classification and confidence calculations on incoming telemetry records.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from typing import List, Dict, Any, Tuple
from src.utils.logger import setup_logger
from src.models.hybrid_pipeline import HybridPipeline  # Ensures scope definition during joblib load

logger = setup_logger("api_predictor")

# Attack mapping: 0=normal, 1=dos, 2=probe, 3=r2l, 4=u2r
ATTACK_FAMILY_MAP = {
    0: "normal",
    1: "dos",
    2: "probe",
    3: "r2l",
    4: "u2r"
}

class NIDSPredictionEngine:
    """
    Production serving predictor that pre-processes incoming telemetry dictionaries
    and performs real-time Zero-Day hybrid classifications.
    """
    def __init__(self, preprocessor_path: str, model_path: str):
        self.preprocessor_path = preprocessor_path
        self.model_path = model_path

        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Hybrid pipeline weights not found: {model_path}")

        logger.info("Loading preprocessor and hybrid models into memory...")
        self.preprocessor = joblib.load(preprocessor_path)

        # Workaround for pickle '__main__' namespace mismatch when unpickling
        import sys
        from src.models.hybrid_pipeline import HybridPipeline as HP
        setattr(sys.modules['__main__'], 'HybridPipeline', HP)

        self.hybrid_model: HybridPipeline = joblib.load(model_path)
        logger.info("In-memory loading successfully completed.")

    def predict_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes and scores a list of incoming connection payloads.
        
        Args:
            records: Array of raw NSL-KDD connection payload dictionaries.
            
        Returns:
            List of response payloads containing attack status, family, confidence, and anomaly scores.
        """
        t_start = time.perf_counter()
        
        # 1. Parse payloads to Pandas DataFrame
        df_in = pd.DataFrame(records)
        
        # 2. Run feature pre-processing ColumnTransformer
        # Columns must be processed through the exact standard scaling pipeline
        X_proc = self.preprocessor.transform(df_in)
        if hasattr(X_proc, "toarray"):
            X_proc = X_proc.toarray()

        # 3. Multi-stage predictions
        if_model = self.hybrid_model.if_model
        multiclass_model = self.hybrid_model.multiclass_model
        t_strict = self.hybrid_model.t_strict

        # Extract predictions programmatically
        s1_preds = if_model.predict(X_proc)            # type: ignore # 1 = normal, -1 = anomaly
        d1_scores = if_model.decision_function(X_proc)  # type: ignore # anomaly scores
        y2_preds = multiclass_model.predict(X_proc)    # type: ignore # supervised signature Router

        # Compute probabilities if available for confidence calculations
        has_proba = hasattr(multiclass_model, "predict_proba")
        if has_proba:
            y2_probas = multiclass_model.predict_proba(X_proc)  # type: ignore
        else:
            y2_probas = None

        predictions = []
        for i in range(len(records)):
            s1 = int(s1_preds[i])
            score = float(d1_scores[i])
            y2 = int(y2_preds[i])

            is_attack = False
            attack_family = "normal"
            confidence = 1.0

            # Apply hybrid classification decision tree logic
            if s1 == 1:
                # Stage 1 says Normal: classify as Normal
                is_attack = False
                attack_family = "normal"
                if has_proba and y2_probas is not None:
                    confidence = float(y2_probas[i][0])
            else:
                # Stage 1 says Suspicious/Anomaly
                if y2 != 0:
                    # Stage 2 recognizes it as a known attack category
                    is_attack = True
                    attack_family = ATTACK_FAMILY_MAP.get(y2, "normal")
                    if has_proba and y2_probas is not None:
                        confidence = float(y2_probas[i][y2])
                else:
                    # Stage 2 thinks normal, but Stage 1 flagged as suspicious
                    if t_strict is None:
                        # Override Zero-Day
                        is_attack = True
                        attack_family = "r2l"  # Novel threat group
                        confidence = 1.0
                    else:
                        # Marginal filtering threshold
                        if score < t_strict:
                            is_attack = True
                            attack_family = "r2l"  # Severe outlier -> Zero-day
                            confidence = 1.0
                        else:
                            is_attack = False
                            attack_family = "normal"  # Marginal outlier -> Normal
                            if has_proba and y2_probas is not None:
                                confidence = float(y2_probas[i][0])

            predictions.append({
                "is_attack": is_attack,
                "attack_family": attack_family,
                "confidence": confidence,
                "stage1_score": score
            })

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(f"Processed batch of size {len(records)} in {t_elapsed_ms:.2f} ms")
        return predictions
