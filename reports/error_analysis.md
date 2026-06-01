# Baseline Model Cybersecurity Failure & Error Analysis

This report documents the deep-dive failure profiles, confusion characteristics, and engineering remedies.

---

## 1. Common Baseline Failure Modes

Across all baseline classifiers, we identify two primary failure vectors:
1.  **Rare Threat Scarcity**: User-to-Root (U2R) and Remote-to-Local (R2L) threats are repeatedly misclassified as **Normal (Benign)**. This is a severe vulnerability since these attacks represent critical privilege escalation vectors.
2.  **DoS-Normal Confusion**: High-volume Denial of Service (DoS) attacks with low duration profiles are occasionally confused with normal HTTP sessions due to overlapping count metrics.

---

## 2. Numerical Confusion Summary

*   **DoS Capture Rate**: > 98% across Tree models (high packet count thresholds are highly discriminatory).
*   **Probe Capture Rate**: > 97% across KNN and Decision Trees.
*   **U2R Capture Rate**: < 10% on Naive Bayes and Logistic Regression due to the massive class imbalance.
*   **R2L Capture Rate**: < 20% on baseline estimators.

---

## 3. Concrete Engineering Remedies for Phase 4

To move beyond these baseline limitations, we propose the following improvements for Phase 4:
1.  **Synthetic Balancing (SMOTE / Random Oversampling)**: Address U2R and R2L sample starvation by synthetically amplifying their feature space during training.
2.  **Ensemble Methods (Random Forest, XGBoost)**: Introduce non-linear decision trees with boosted iterations to trace minor sample boundaries that baseline algorithms miss.
3.  **Cost-Sensitive Learning**: Penalize False Negatives on rare threats heavily in tree structures to artificially boost U2R/R2L Recall.
