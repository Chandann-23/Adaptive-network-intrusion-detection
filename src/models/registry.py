import os
import json
import joblib
from datetime import datetime, timezone
from typing import Any, Dict, List
from src.utils.logger import setup_logger

logger = setup_logger("registry")


class ModelRegistry:
    """
    State-preserving model registry engine to serialize trained classifiers
    to disk and reload them for live inference or evaluations, maintaining
    rich metadata manifests for production tracking.
    """
    def __init__(self, base_dir: str = "models"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_model(
        self,
        model: Any,
        model_name: str,
        target_type: str,
        metrics: Dict[str, Any],
        feature_names: List[str],
        version: str = "1.0.0",
        dataset_version: str = "1.0.0"
    ) -> str:
        """
        Serializes and persists a fitted estimator model object as a joblib binary
        inside a target-specific versioned directory, alongside a rich metadata JSON.
        
        Args:
            model: The fitted scikit-learn model instance.
            model_name: Base identifier for the model (e.g. 'logistic_regression').
            target_type: Target category (e.g. 'binary' or 'multiclass').
            metrics: Performance metrics holdouts dictionary.
            feature_names: Ordered list of expanded schema input columns.
            version: Model version identifier.
            dataset_version: Dataset schema version identifier.
            
        Returns:
            Relative folder pathway where model and metadata were saved.
        """
        # Build structure: models/<target_type>/<model_name>/
        model_dir = os.path.join(self.base_dir, target_type, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        model_save_path = os.path.join(model_dir, "model.joblib")
        metadata_save_path = os.path.join(model_dir, "metadata.json")
        
        logger.info(f"Registering model '{model_name}' under '{target_type}' target...")
        
        try:
            # 1. Persist weight binary
            joblib.dump(model, model_save_path)
            logger.info(f"Model weight binary saved successfully to: {model_save_path}")
            
            # 2. Build metadata manifest
            metadata = {
                "model_name": model_name,
                "target_type": target_type,
                "version": version,
                "training_timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset_version": dataset_version,
                "feature_count": len(feature_names),
                "metrics": metrics,
                "feature_names": feature_names
            }
            
            # 3. Persist metadata manifest
            with open(metadata_save_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Model metadata manifest saved successfully to: {metadata_save_path}")
            
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            raise e
            
        return model_dir

    def load_model(self, model_name: str, target_type: str) -> Any:
        """
        Loads a serialized joblib estimator model from registry storage.
        
        Args:
            model_name: Filename of the target model (e.g. 'logistic_regression').
            target_type: Target category (e.g. 'binary' or 'multiclass').
            
        Returns:
            Loaded model instance.
        """
        load_path = os.path.join(self.base_dir, target_type, model_name, "model.joblib")
        logger.info(f"Fetching model model from registry: {load_path}")
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Fitted model weight binary not found in registry: {load_path}")
            
        try:
            model = joblib.load(load_path)
            logger.info(f"Model weights loaded successfully: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise e
