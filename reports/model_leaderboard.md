# Network Intrusion Detection Classifier Leaderboard

This report evaluates and ranks the baseline machine learning models trained on the pre-processed **NSL-KDD dataset**. 

---

## 1. Executive Summary

Based on our **cybersecurity metrics hierarchy** (prioritizing **Recall** to minimize missed intrusions, followed by **F1-Score** to manage false alarms), the top-performing baseline model is the **Decision Tree (Binary Classifier)**.

*   **Best Model**: Decision Tree
*   **Target Task**: Binary
*   **Intrusion Recall**: 99.7953%
*   **F1-Score**: 99.7698%

---

## 2. Classifier Rankings & Performance Comparison

Models are ranked in order of priority: **Recall** $	o$ **F1-Score** $	o$ **ROC-AUC** $	o$ **Accuracy**.

| Rank | Model Identifier | Task Type | Recall (Capture) | F1-Score | ROC-AUC | Global Accuracy | Fit Time (s) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Decision Tree** | BINARY | 99.7953% | 99.7698% | 0.9988 | 99.7857% | 0.39s |
| 2 | **Knn** | BINARY | 98.7634% | 98.6625% | 0.9981 | 98.7537% | 0.03s |
| 3 | **Naive Bayes** | BINARY | 96.5888% | 91.6454% | 0.9567 | 91.8039% | 0.11s |
| 4 | **Logistic Regression** | BINARY | 96.3244% | 97.0194% | 0.9960 | 97.2455% | 14.33s |
| 5 | **Naive Bayes** | MULTICLASS | 91.2762% | 59.6109% | 0.9794 | 86.5092% | 0.08s |
| 6 | **Decision Tree** | MULTICLASS | 89.5266% | 90.1319% | 0.9440 | 99.6825% | 0.52s |
| 7 | **Svm** | BINARY | 89.1694% | 87.6115% | 0.9578 | 88.2635% | 10.03s |
| 8 | **Knn** | MULTICLASS | 78.7075% | 80.2845% | 0.9949 | 98.4759% | 0.03s |
| 9 | **Logistic Regression** | MULTICLASS | 70.9765% | 70.7913% | 0.9907 | 98.1981% | 25.84s |
| 10 | **Svm** | MULTICLASS | 39.7469% | 40.2991% | 0.9495 | 83.1474% | 10.56s |

---

## 3. Engineering Recommendations for Production

1.  **Deployment Candidate**: **Decision Tree** represents the optimal balance of threat detection capabilities and precision.
2.  **Inference Latency Tradeoffs**: While models like KNN or SVM may offer competitive scores, their inference footprints scale with dataset sizes or kernel complexity. For microsecond response routing at core switches, **Logistic Regression** or **Decision Trees** should be chosen if their baseline Recall is within acceptable parameters.
3.  **Baseline Foundation**: These scores serve as the rigorous baseline. In Phase 4, we will introduce ensemble methods (Random Forest, XGBoost) and measure their relative improvements against this table.
