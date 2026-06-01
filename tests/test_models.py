import os
import tempfile
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from src.models.metrics import calculate_binary_metrics
from src.models.registry import ModelRegistry
from src.models.inference import NIDSInferenceEngine


def test_calculate_binary_metrics():
    """Tests accuracy, recall, precision, F1, and AUC metrics calculations."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.3, 0.85])
    
    metrics = calculate_binary_metrics(y_true, y_pred, y_prob)
    
    assert metrics["accuracy"] == 0.8
    assert metrics["recall"] == 2.0 / 3.0
    assert metrics["precision"] == 1.0
    assert metrics["f1"] == 0.8
    assert metrics["roc_auc"] > 0.5


def test_model_registry_save_and_load():
    """Tests that ModelRegistry correctly saves and loads scikit-learn estimators."""
    model = LogisticRegression(max_iter=10)
    mock_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    mock_y = np.array([0, 1])
    model.fit(mock_x, mock_y)  # type: ignore
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        registry = ModelRegistry(base_dir=tmp_dir)
        
        # Save model
        save_path = registry.save_model(
            model=model,
            model_name="mock_lr",
            target_type="binary",
            metrics={"accuracy": 1.0},
            feature_names=["f1", "f2"]
        )
        assert os.path.exists(save_path)
        assert os.path.exists(os.path.join(save_path, "model.joblib"))
        assert os.path.exists(os.path.join(save_path, "metadata.json"))
        
        # Load model back
        loaded_model = registry.load_model("mock_lr", target_type="binary")
        
        # Predict using loaded model
        original_preds = model.predict(mock_x)  # type: ignore
        loaded_preds = loaded_model.predict(mock_x)  # type: ignore
        
        assert np.array_equal(original_preds, loaded_preds)  # type: ignore


def test_inference_engine_score_connection(mock_nsl_kdd_dataframe):
    """
    Tests that NIDSInferenceEngine successfully loads fitted preprocessing
    and model files, and scores an incoming connection dictionary.
    """
    df = mock_nsl_kdd_dataframe
    X_raw = df.drop(columns=["class", "difficulty_score", "num_outbound_cmds"])
    y = (df["class"] != "normal").astype(int).values
    
    # Simple model fitting
    from src.features.preprocessing import build_nsl_kdd_preprocessor
    cat = ["protocol_type", "service", "flag"]
    log = ["duration", "src_bytes", "dst_bytes"]
    robust = ["count", "srv_count", "hot", "num_failed_logins", "num_compromised"]
    rate = ["serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate"]
    binary = ["land", "logged_in", "root_shell", "is_hot_login", "is_guest_login"]
    
    preprocessor = build_nsl_kdd_preprocessor(
        categorical_cols=cat, log_cols=log, robust_cols=robust,
        rate_cols=rate, binary_cols=binary
    )
    X_proc = preprocessor.fit_transform(X_raw)
    
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
        
    model = LogisticRegression()
    model.fit(X_proc, y)  # type: ignore
    
    # Save preprocessing and model artifacts
    with tempfile.TemporaryDirectory() as tmp_dir:
        prep_path = os.path.join(tmp_dir, "preprocessor.joblib")
        model_path = os.path.join(tmp_dir, "model.joblib")
        
        import joblib
        joblib.dump(preprocessor, prep_path)
        joblib.dump(model, model_path)
        
        # Create inference engine
        engine = NIDSInferenceEngine(
            preprocessor_path=prep_path,
            model_path=model_path,
            expected_columns=df.columns.tolist()
        )
        
        # Take a raw row payload
        raw_row = df.iloc[0].to_dict()
        del raw_row["class"]  # simulate real inference
        del raw_row["difficulty_score"]
        
        # Score the connection
        pred, prob = engine.score_connection(raw_row)
        
        assert pred in [0, 1]
        assert 0.0 <= prob <= 1.0
