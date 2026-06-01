import os
from typing import Any
import numpy as np
import matplotlib
# Use Agg backend to ensure head-less image generation without GUI threads
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from src.utils.logger import setup_logger

logger = setup_logger("evaluate")


class ModelEvaluator:
    """
    Unified evaluation and reporting engine that calculates performance scores,
    and generates high-contrast visual metrics (Confusion Matrices, ROC curves,
    Precision-Recall curves, and Feature Importance bar charts).
    """
    def __init__(self, base_reports_dir: str = "reports/plots"):
        self.base_reports_dir = base_reports_dir
        self.cm_dir = os.path.join(self.base_reports_dir, "confusion_matrices")
        self.roc_dir = os.path.join(self.base_reports_dir, "roc_curves")
        self.pr_dir = os.path.join(self.base_reports_dir, "precision_recall_curves")
        self.fi_dir = os.path.join(self.base_reports_dir, "feature_importance")
        
        # Build directories
        os.makedirs(self.cm_dir, exist_ok=True)
        os.makedirs(self.roc_dir, exist_ok=True)
        os.makedirs(self.pr_dir, exist_ok=True)
        os.makedirs(self.fi_dir, exist_ok=True)

    def generate_confusion_matrix_plot(
        self,
        y_true: Any,
        y_pred: Any,
        model_name: str,
        is_multiclass: bool = False
    ) -> str:
        """
        Renders and exports a high-quality, high-contrast Confusion Matrix heatmap.
        """
        logger.info(f"Generating confusion matrix plot for model: {model_name}")
        
        # Calculate raw matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Set label taxonomy
        if is_multiclass:
            labels = ["Normal", "DoS", "Probe", "R2L", "U2R"]
        else:
            labels = ["Normal (0)", "Anomaly (1)"]
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels[:cm.shape[1]],
            yticklabels=labels[:cm.shape[0]],
            cbar=False,
            annot_kws={"size": 14, "weight": "bold"}
        )
        
        plt.title(f"Confusion Matrix: {model_name.replace('_', ' ').title()}", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Predicted Threat Class", fontsize=12, labelpad=10)
        plt.ylabel("True Threat Class", fontsize=12, labelpad=10)
        plt.tight_layout()
        
        save_path = os.path.join(self.cm_dir, f"{model_name}_cm.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        logger.info(f"Confusion matrix plot successfully exported to: {save_path}")
        return save_path

    def generate_roc_curve_plot(
        self,
        y_true: Any,
        y_prob: Any,
        model_name: str
    ) -> str:
        """
        Renders and exports a clean ROC Curve plot showing True Positive vs False Positive rates.
        """
        logger.info(f"Generating ROC curve plot for model: {model_name}")
        
        # Calculate curve metric boundaries
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#ff6666', lw=2.5, label=f'ROC Curve (Area = {roc_auc:.4f})')  # type: ignore
        plt.plot([0, 1], [0, 1], color='#66ccff', lw=1.5, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12, labelpad=10)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12, labelpad=10)
        plt.title(f'ROC Curve: {model_name.replace("_", " ").title()}', fontsize=14, fontweight="bold", pad=15)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        save_path = os.path.join(self.roc_dir, f"{model_name}_roc.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        logger.info(f"ROC curve plot successfully exported to: {save_path}")
        return save_path

    def generate_precision_recall_curve_plot(
        self,
        y_true: Any,
        y_prob: Any,
        model_name: str,
        is_multiclass: bool = False
    ) -> str:
        """
        Renders and exports a high-quality Precision-Recall Curve.
        Supports both binary and multi-class (One-vs-Rest) formulations.
        """
        logger.info(f"Generating Precision-Recall curve plot for model: {model_name}")
        plt.figure(figsize=(8, 6))
        
        if is_multiclass:
            labels = ["Normal", "DoS", "Probe", "R2L", "U2R"]
            # Enforce 1D numpy array and shape match
            y_true_arr = np.asarray(y_true)
            y_prob_arr = np.asarray(y_prob)
            
            for i in range(min(5, y_prob_arr.shape[1])):
                y_true_binary = (y_true_arr == i).astype(int)
                y_prob_class = y_prob_arr[:, i]
                
                # Check if class is present in true labels to avoid divide-by-zero
                if len(np.unique(y_true_binary)) > 1:
                    prec, rec, _ = precision_recall_curve(y_true_binary, y_prob_class)  # type: ignore
                    ap_score = average_precision_score(y_true_binary, y_prob_class)  # type: ignore
                    plt.plot(rec, prec, lw=2, label=f'{labels[i]} (AP = {ap_score:.4f})')  # type: ignore
        else:
            # Binary Precision-Recall curve
            prec, rec, _ = precision_recall_curve(y_true, y_prob)  # type: ignore
            ap_score = average_precision_score(y_true, y_prob)  # type: ignore
            plt.plot(rec, prec, color='#ff9933', lw=2.5, label=f'PR Curve (AP = {ap_score:.4f})')  # type: ignore
            
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Threat Capture)', fontsize=12, labelpad=10)
        plt.ylabel('Precision (Deduplication Accuracy)', fontsize=12, labelpad=10)
        plt.title(f'Precision-Recall Curve: {model_name.replace("_", " ").title()}', fontsize=14, fontweight="bold", pad=15)
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        save_path = os.path.join(self.pr_dir, f"{model_name}_pr.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        logger.info(f"Precision-Recall curve successfully exported to: {save_path}")
        return save_path

    def generate_feature_importance_plot(
        self,
        model: Any,
        feature_names: list[str],
        model_name: str
    ) -> str:
        """
        Renders a high-contrast horizontal bar chart of the top 10 most influential features.
        Supports DecisionTree feature_importances_ and LogisticRegression coefficients.
        """
        logger.info(f"Assessing feature importance visualization suitability for: {model_name}")
        importances = None
        title = ""
        
        # 1. Check for Tree Feature Importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            title = f"Top 10 Feature Importances: {model_name.replace('_', ' ').title()}"
        # 2. Check for Linear Model Coefficients
        elif hasattr(model, "coef_"):
            # If multiclass, coef_ is (n_classes, n_features). Take the mean absolute coefficient.
            if len(model.coef_.shape) > 1 and model.coef_.shape[0] > 1:
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                importances = np.abs(model.coef_[0])
            title = f"Top 10 Feature Coefficients (Abs): {model_name.replace('_', ' ').title()}"
            
        if importances is None:
            logger.warning(f"Classifier {model_name} does not natively support feature coefficients or importances. Skipping plot.")
            return ""
            
        # Map to feature names
        importances = np.asarray(importances)
        indices = np.argsort(importances)[::-1]
        
        top_indices = indices[:10]
        top_importances = importances[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=top_importances,
            y=top_features,
            palette="viridis",
            hue=top_features,
            legend=False
        )
        
        plt.title(title, fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Influence Metric Size", fontsize=12, labelpad=10)
        plt.ylabel("NSL-KDD Schema Input Columns", fontsize=12, labelpad=10)
        plt.grid(True, axis='x', linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        save_path = os.path.join(self.fi_dir, f"{model_name}_fi.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        logger.info(f"Feature influence plot successfully exported to: {save_path}")
        return save_path
