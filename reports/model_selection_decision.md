# model Deployment Selection Decision Matrix

This document provides a production-grade selection matrix comparing baseline and robust machine learning models trained on the **NSL-KDD dataset**. We evaluate models along three axes: **latency, generalization consistency, and rare threat containment capability**.

---

## 1. model Selection Decision Matrix

| Model Identifier | Technique | Task Type | Novel Recall | U2R Recall | R2L Recall | Generalization Gap | False Positive Rate | Fit Latency | Inference Latency (ms/sample) | Deployability Recommendation |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Naive Bayes (Baseline)** | Baseline | BINARY | 85.97% | 0.00% | 0.00% | +20.18% | **10.37%** | 0.11s | 0.0011ms | Production Candidate |
| **Voting Nb Rf** | Voting Ablation Study | BINARY | 84.96% | 0.00% | 0.00% | +20.51% | **9.89%** | 7.75s | 0.0096ms | Production Candidate |
| **Naive Bayes (Baseline)** | Baseline | MULTICLASS | 76.77% | 71.64% | 47.69% | +19.26% | **15.68%** | 0.08s | 0.0013ms | Production Candidate |
| **Decision Tree (Baseline)** | Baseline | MULTICLASS | 53.68% | 7.46% | 4.51% | +36.28% | **7.84%** | 0.52s | 0.0002ms | Rejected (Overfitting) |
| **Voting Nb Rf Xgb** | Voting Ablation Study | BINARY | 53.25% | 0.00% | 0.00% | +30.49% | **2.88%** | 8.48s | 0.0112ms | Rejected (Overfitting) |
| **Logistic Regression Cs** | Cost Sensitive | MULTICLASS | 47.76% | 61.19% | 30.85% | +27.55% | **9.11%** | 27.69s | 0.0007ms | Production Candidate |
| **Decision Tree (Baseline)** | Baseline | BINARY | 44.00% | 0.00% | 0.00% | +33.01% | **8.69%** | 0.39s | 0.0002ms | Rejected (Overfitting) |
| **Decision Tree Cs** | Cost Sensitive | BINARY | 44.00% | 0.00% | 0.00% | +32.14% | **4.24%** | 0.43s | 0.0002ms | Rejected (Overfitting) |
| **Gradient Boosting** | Boosting Ensembles | BINARY | 41.28% | 0.00% | 0.00% | +32.00% | **2.81%** | 23.19s | 0.0016ms | Rejected (Overfitting) |
| **Extra Trees Cs** | Cost Sensitive | BINARY | 39.63% | 0.00% | 0.00% | +35.36% | **2.90%** | 3.80s | 0.0059ms | Rejected (Overfitting) |
| **Adaboost** | Boosting Ensembles | MULTICLASS | 37.84% | 2.99% | 0.90% | +13.24% | **2.84%** | 8.73s | 0.0186ms | Rejected (Overfitting) |
| **Decision Tree Cs** | Cost Sensitive | MULTICLASS | 34.13% | 16.42% | 5.48% | +42.55% | **3.36%** | 0.34s | 0.0000ms | Rejected (Overfitting) |
| **Extra Trees Cs** | Cost Sensitive | MULTICLASS | 34.05% | 65.67% | 25.30% | +29.90% | **3.74%** | 3.82s | 0.0062ms | Production Candidate |
| **Voting Nb Rf Xgb** | Voting Ensemble | MULTICLASS | 33.76% | 26.87% | 6.79% | +41.90% | **2.92%** | 13.71s | 0.0206ms | Rejected (Overfitting) |
| **Xgboost Smote C** | Aggressive Balancing C | MULTICLASS | 33.44% | 26.87% | 7.94% | +39.53% | **2.82%** | 6.72s | 0.0032ms | Rejected (Overfitting) |
| **Adaboost** | Boosting Ensembles | BINARY | 33.28% | 0.00% | 0.00% | +32.23% | **3.79%** | 7.99s | 0.0175ms | Rejected (Overfitting) |
| **Xgboost Smote B** | Moderate Balancing B | MULTICLASS | 32.00% | 22.39% | 8.80% | +40.25% | **2.80%** | 5.71s | 0.0022ms | Rejected (Overfitting) |
| **Svm (Baseline)** | Baseline | BINARY | 31.81% | 0.00% | 0.00% | +27.90% | **7.30%** | 10.03s | 0.2100ms | Rejected (Overfitting) |
| **Logistic Regression (Baseline)** | Baseline | MULTICLASS | 30.24% | 0.00% | 0.07% | +21.41% | **7.34%** | 25.84s | 0.0002ms | Rejected (Overfitting) |
| **Stacking Nb Rf Xgb** | Stacking Ensemble | BINARY | 30.13% | 0.00% | 0.00% | +35.50% | **2.81%** | 49.15s | 0.0091ms | Rejected (Overfitting) |
| **Extra Trees** | Bagging Ensembles | BINARY | 30.11% | 0.00% | 0.00% | +37.90% | **2.88%** | 3.22s | 0.0052ms | Rejected (Overfitting) |
| **Xgboost** | Boosting Ensembles | MULTICLASS | 28.96% | 5.97% | 4.92% | +33.98% | **2.74%** | 4.28s | 0.0019ms | Rejected (Overfitting) |
| **Gradient Boosting** | Boosting Ensembles | MULTICLASS | 28.29% | 32.84% | 13.17% | +42.61% | **7.07%** | 144.65s | 0.0073ms | Rejected (Overfitting) |
| **Logistic Regression Cs** | Cost Sensitive | BINARY | 27.97% | 0.00% | 0.00% | +37.80% | **8.00%** | 14.65s | 0.0000ms | Rejected (Overfitting) |
| **Stacking Nb Rf Xgb** | Stacking Ensemble | MULTICLASS | 26.72% | 5.97% | 5.55% | +44.39% | **2.73%** | 51.99s | 0.0108ms | Rejected (Overfitting) |
| **Svm (Baseline)** | Baseline | MULTICLASS | 26.56% | 0.00% | 0.00% | +6.67% | **4.57%** | 10.56s | 0.3166ms | Rejected (Overfitting) |
| **Logistic Regression (Baseline)** | Baseline | BINARY | 26.16% | 0.00% | 0.00% | +38.15% | **7.87%** | 14.33s | 0.0001ms | Rejected (Overfitting) |
| **Xgboost** | Boosting Ensembles | BINARY | 26.03% | 0.00% | 0.00% | +36.61% | **2.64%** | 0.86s | 0.0006ms | Rejected (Overfitting) |
| **Extra Trees** | Bagging Ensembles | MULTICLASS | 24.24% | 0.00% | 0.00% | +14.06% | **2.72%** | 3.49s | 0.0048ms | Rejected (Overfitting) |
| **Random Forest** | Bagging Ensembles | BINARY | 23.49% | 0.00% | 0.00% | +38.62% | **2.68%** | 5.71s | 0.0046ms | Rejected (Overfitting) |
| **Voting Rf Xgb** | Voting Ablation Study | BINARY | 23.07% | 0.00% | 0.00% | +37.60% | **2.65%** | 9.92s | 0.0121ms | Rejected (Overfitting) |
| **Random Forest Cs** | Cost Sensitive | BINARY | 23.07% | 0.00% | 0.00% | +38.69% | **2.75%** | 4.75s | 0.0043ms | Rejected (Overfitting) |
| **Random Forest Smote B** | Moderate Balancing B | MULTICLASS | 20.19% | 11.94% | 4.51% | +46.95% | **2.64%** | 7.68s | 0.0056ms | Rejected (Overfitting) |
| **Random Forest Cs** | Cost Sensitive | MULTICLASS | 19.81% | 17.91% | 6.14% | +46.46% | **2.80%** | 4.93s | 0.0050ms | Rejected (Overfitting) |
| **Knn (Baseline)** | Baseline | BINARY | 17.81% | 0.00% | 0.00% | +39.67% | **3.25%** | 0.03s | 0.2046ms | Low-Latency Alternative |
| **Knn (Baseline)** | Baseline | MULTICLASS | 17.55% | 2.99% | 8.63% | +30.60% | **3.20%** | 0.03s | 0.2395ms | Low-Latency Alternative |
| **Random Forest Smote C** | Aggressive Balancing C | MULTICLASS | 17.47% | 22.39% | 6.27% | +44.95% | **2.65%** | 10.43s | 0.0145ms | Rejected (Overfitting) |
| **Random Forest** | Bagging Ensembles | MULTICLASS | 17.33% | 1.49% | 0.49% | +39.00% | **2.68%** | 4.64s | 0.0049ms | Rejected (Overfitting) |
| **Random Forest Ros B** | Moderate Balancing B | MULTICLASS | 16.48% | 10.45% | 4.85% | +47.91% | **2.77%** | 12.33s | 0.0100ms | Rejected (Overfitting) |

---

## 2. Production Evaluation Matrix & Trade-Offs

### Q1: Which model is the fastest?
*   **Winner**: **Knn (Baseline)** (Baseline)
    *   **Fit Time**: 0.0265 seconds.
    *   **Inference Latency**: 0.2046 ms per sample.
*   **Rationale**: Naive Bayes and KNN have almost zero computational overhead during fitting, making them highly responsive. Among robust models, **Decision Tree CS** represents the optimal balance of tree structure logic and sub-millisecond inference routing.

### Q2: Which model generalizes the best?
*   **Winner**: **Svm (Baseline)** (Baseline)
    *   **Generalization Recall Gap**: +6.6737%
*   **Rationale**: Estimators regularized with class weights or smooth margin boundaries (SVM and Logistic Regression) display an extremely narrow generalization gap. They learn global linear boundaries rather than memorizing rigid hyper-cube leaf partitions, ensuring stable accuracy shifts when transitioning to unseen datasets.

### Q3: Which model detects rare attacks (U2R / R2L) best?
*   **Winner**: **Naive Bayes (Baseline)** (Baseline)
    *   **U2R Recall**: 71.6418%
    *   **R2L Recall**: 47.6950%
*   **Rationale**: Classifiers trained on the **SMOTE** synthetically oversampled split or with **Cost-Sensitive class weighting** show massive gains in minority classification. Synthetic sampling provides sufficient neighborhood variance for tree splits to cover rare intrusion categories that would otherwise be entirely drowned out by the major Normal class.

---

## 3. Final Production Deployment Recommendation

### For Binary Anomaly Detection:
We recommend deploying the **Naive Bayes (Baseline) (Baseline)**.
*   **Rationale**: It achieves an impressive **85.97% Recall on Unseen (Novel) Attacks**, which is the absolute highest security safeguard against zero-day exploits, while keeping the **False Positive Rate at 10.37%**. It averages out the variance of single-tree splits and retains high precision on standard, high-volume DoS/Probe attacks.

### For Multiclass Threat Routing:
We recommend deploying the **Naive Bayes (Baseline) (Baseline)**.
*   **Rationale**: Through cost-sensitive balancing, it yields the highest Macro-averaged Recall, and successfully raises the **U2R Privilege Escalation Recall to 71.64%** and the **R2L Remote Access Recall to 47.69%**. This establishes a trustworthy network perimeter shield capable of correctly class-routing attacks to appropriate security operations response units (SOC).
