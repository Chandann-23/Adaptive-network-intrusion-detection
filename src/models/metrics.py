import numpy as np
from typing import Dict, Any, Union
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from src.utils.logger import setup_logger

logger = setup_logger("metrics")


def calculate_binary_metrics(y_true: Any, y_pred: Any, y_prob: Any) -> Dict[str, Any]:
    """
    Computes standard performance metrics for binary classification models.
    Priority order: Recall -> F1 -> ROC-AUC -> Accuracy.
    
    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted anomaly probabilities.
        
    Returns:
        Dict containing calculated metrics.
    """
    logger.debug("Calculating binary classification metrics...")
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))  # type: ignore
    rec = float(recall_score(y_true, y_pred, zero_division=0))  # type: ignore
    f1 = float(f1_score(y_true, y_pred, zero_division=0))  # type: ignore
    
    # Calculate ROC-AUC
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception as e:
        logger.warning(f"Failed to calculate ROC-AUC score: {e}. Setting to 0.5.")
        auc = 0.5
        
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    }


def calculate_multiclass_metrics(y_true: Any, y_pred: Any, y_prob: Any) -> Dict[str, Any]:
    """
    Computes class-specific and macro-averaged metrics for 5-class threat routing.
    
    Args:
        y_true: True multiclass labels (0 to 4).
        y_pred: Predicted multiclass labels (0 to 4).
        y_prob: Predicted class probabilities (matrix of shape (n_samples, 5)).
        
    Returns:
        Dict containing macro-averaged and class-specific metrics.
    """
    logger.debug("Calculating multiclass classification metrics...")
    
    acc = float(accuracy_score(y_true, y_pred))
    # Macro averages (give equal weight to all classes, important for rare R2L/U2R classes)
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))  # type: ignore
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))  # type: ignore
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))  # type: ignore
    
    # Multiclass ROC-AUC (One-vs-Rest)
    try:
        macro_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception as e:
        logger.warning(f"Failed to calculate multiclass ROC-AUC: {e}. Setting to 0.5.")
        macro_auc = 0.5
        
    # Class-specific recall and F1 to trace minor threats (U2R/R2L)
    unique_classes = np.unique(y_true)
    class_rec: Any = recall_score(y_true, y_pred, average=None, zero_division=0)  # type: ignore
    class_f1: Any = f1_score(y_true, y_pred, average=None, zero_division=0)  # type: ignore
    
    class_metrics = {}
    for idx, cls in enumerate(unique_classes):
        class_metrics[f"class_{cls}_recall"] = float(class_rec[idx])
        class_metrics[f"class_{cls}_f1"] = float(class_f1[idx])
        
    return {
        "accuracy": acc,
        "precision_macro": macro_prec,
        "recall_macro": macro_rec,
        "f1_macro": macro_f1,
        "roc_auc_macro": macro_auc,
        "class_specific": class_metrics
    }
