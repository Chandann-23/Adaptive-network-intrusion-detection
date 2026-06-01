import os
import joblib
import pandas as pd
from typing import Dict, Any, Tuple
from src.features.validation import DataValidator
from src.utils.logger import setup_logger

logger = setup_logger("inference")


class NIDSInferenceEngine:
    """
    Production serving engine that loads preprocessing artifacts and models,
    validates incoming network connection payloads, and performs predictions.
    """
    def __init__(self, preprocessor_path: str, model_path: str, expected_columns: list):
        self.preprocessor_path = preprocessor_path
        self.model_path = model_path
        
        # Load fitted transformers and models
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessing pipeline not found: {preprocessor_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fitted model not found: {model_path}")
            
        self.preprocessor = joblib.load(preprocessor_path)
        self.model = joblib.load(model_path)
        
        # Instantiate validation layer for incoming live traffic
        self.validator = DataValidator(expected_columns=expected_columns, target_column="class")

    def score_connection(self, raw_connection: Dict[str, Any]) -> Tuple[int, float]:
        """
        Parses, validates, pre-processes, and scores a single incoming network connection.
        
        Args:
            raw_connection: Dictionary mapping the raw 41 features of NSL-KDD.
            
        Returns:
            Tuple of (predicted_class, anomaly_probability).
        """
        logger.debug("Parsing connection telemetry payload...")
        
        # 1. Format raw connection dictionary to DataFrame representation
        df_in = pd.DataFrame([raw_connection])
        
        # 2. Run data validation contracts
        # Expected columns must omit the target class if not provided at inference
        feature_cols = [c for c in self.validator.expected_columns if c != "class" and c != "difficulty_score"]
        for col in feature_cols:
            if col not in df_in.columns:
                raise ValueError(f"Schema drift detected. Missing feature column: '{col}'")
                
        # 3. Apply state-preserving ColumnTransformer pipeline
        # Discard irrelevant columns during active transformation
        X_proc = self.preprocessor.transform(df_in)
        
        # Force dense representation
        if hasattr(X_proc, "toarray"):
            X_proc = X_proc.toarray()
            
        # 4. Perform Model Inference
        prediction = int(self.model.predict(X_proc)[0])
        
        # Obtain probability scores
        try:
            probabilities = self.model.predict_proba(X_proc)[0]
            probability = float(probabilities[prediction])
        except Exception as e:
            logger.warning(f"Could not compute probabilities: {e}. Outputting default.")
            probability = 1.0 if prediction == 1 else 0.0
            
        logger.info(f"Connection scored successfully. Prediction: {prediction}, Confidence: {probability:.4f}")
        return prediction, probability
