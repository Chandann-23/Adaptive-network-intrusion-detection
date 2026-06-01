# Binary Intrusion Detection Classification Report

This report evaluates models on the binary detection task (**Normal (0)** vs **Attack (1)**).

---

## 1. Top Performing Binary Detector: Decision Tree

*   **Holdout Recall**: 99.7953%
*   **F1-Score**: 99.7698%
*   **Global Holdout Accuracy**: 99.7857%

---

## 2. Model Performance Summary Table

| Rank | Model Name | Recall (Anomaly Capture) | F1-Score | ROC-AUC | Global Accuracy | Fit Latency |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Decision Tree** | 99.7953% | 99.7698% | 0.9988 | 99.7857% | 0.39s |
| 2 | **Knn** | 98.7634% | 98.6625% | 0.9981 | 98.7537% | 0.03s |
| 3 | **Naive Bayes** | 96.5888% | 91.6454% | 0.9567 | 91.8039% | 0.11s |
| 4 | **Logistic Regression** | 96.3244% | 97.0194% | 0.9960 | 97.2455% | 14.33s |
| 5 | **Svm** | 89.1694% | 87.6115% | 0.9578 | 88.2635% | 10.03s |

---

## 3. Threat Capture Rationale

For intrusion detection, **Recall** is prioritized over Accuracy. Missing an active attack (False Negative) has catastrophic consequences, including unauthorized lateral movement and data exfiltration. **Logistic Regression** and **Decision Trees** show exceptionally strong baselines with low fit footprints.
