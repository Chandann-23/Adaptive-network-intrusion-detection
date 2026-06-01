# Multiclass Threat Classification & Routing Report

This report details models on the multiclass routing task (**Normal**, **DoS**, **Probe**, **R2L**, **U2R**).

---

## 1. Top Performing Multiclass Routing Baseline: Naive Bayes

*   **Macro-Averaged Holdout Recall**: 91.2762%
*   **Macro-Averaged Holdout F1-Score**: 59.6109%
*   **Global Holdout Accuracy**: 86.5092%

---

## 2. Multiclass Performance Summary Table

| Rank | Model Name | Macro Recall | Macro F1-Score | Macro ROC-AUC | Accuracy | Fit Latency |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Naive Bayes** | 91.2762% | 59.6109% | 0.9794 | 86.5092% | 0.08s |
| 2 | **Decision Tree** | 89.5266% | 90.1319% | 0.9440 | 99.6825% | 0.52s |
| 3 | **Knn** | 78.7075% | 80.2845% | 0.9949 | 98.4759% | 0.03s |
| 4 | **Logistic Regression** | 70.9765% | 70.7913% | 0.9907 | 98.1981% | 25.84s |
| 5 | **Svm** | 39.7469% | 40.2991% | 0.9495 | 83.1474% | 10.56s |

---

## 3. Hardest Attacks to Classify (U2R & R2L)

A key finding across all multiclass models is that **U2R (User to Root)** and **R2L (Remote to Local)** threat categories are extremely difficult to classify, suffering from low individual F1-scores. This is directly caused by extreme sample scarcity in the NSL-KDD dataset (U2R has only 46 training samples).
