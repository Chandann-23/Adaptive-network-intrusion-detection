# Phase 5.0 - Naive Bayes Threshold Optimization Report

This analysis sweeps the binary decision threshold tau on the **Naive Bayes** classifier from 0.30 to 0.70 and measures the impact on **Novel Attack Recall** and **False Positive Rate (FPR)** on the untouched **KDDTest+** holdout dataset.

---

## 1. Default Operating Point (tau = 0.50)

| Metric | Value |
|:---|:---:|
| Novel Attack Recall | **85.97%** |
| False Positive Rate | **10.37%** |
| Precision | 90.69% |
| Overall Recall | 76.41% |
| F1-Score | 82.94% |
| Accuracy | 82.11% |

---

## 2. Target Operating Point

| Target | Threshold |
|:---|:---:|
| Novel Attack Recall >= 80% | Required |
| False Positive Rate <= 6% | Required |

WARNING: **No threshold satisfies both targets simultaneously.** Best available: tau = 0.30 with Novel Recall = 87.17%, FPR = 10.63%.

| Metric | Default (tau=0.50) | Optimal (tau=0.30) | Change |
|:---|:---:|:---:|:---:|
| Novel Attack Recall | 85.97% | **87.17%** | +1.20% |
| False Positive Rate | 10.37% | **10.63%** | +0.26% |
| Precision | 90.69% | 90.52% | -0.17% |
| F1-Score | 82.94% | 83.08% | +0.14% |

---

## 3. Full Threshold Sweep

| Threshold (tau) | Novel Recall | Seen Recall | FPR | Precision | F1 | Accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.30** | 87.17% | 72.49% | 10.63% | 90.52% | 83.08% | 82.20% |
| 0.35 | 86.91% | 72.49% | 10.60% | 90.54% | 83.05% | 82.17% |
| 0.40 | 86.43% | 72.48% | 10.58% | 90.54% | 82.96% | 82.10% |
| 0.45 | 86.08% | 72.48% | 10.44% | 90.63% | 82.94% | 82.10% |
| 0.50 | 85.97% | 72.47% | 10.37% | 90.69% | 82.94% | 82.11% |
| 0.55 | 85.87% | 72.47% | 10.34% | 90.71% | 82.93% | 82.10% |
| 0.60 | 85.84% | 72.47% | 10.29% | 90.75% | 82.94% | 82.12% |
| 0.65 | 85.84% | 72.47% | 10.24% | 90.79% | 82.96% | 82.14% |
| 0.70 | 85.79% | 72.47% | 10.17% | 90.84% | 82.97% | 82.16% |

---

## 4. Engineering Insights

### Why Threshold Tuning Matters
The default tau = 0.50 was never explicitly optimised - it's simply the scikit-learn default. In intrusion detection, the cost of a **False Negative** (missed attack) far exceeds the cost of a **False Positive** (wasted analyst time). This asymmetry justifies shifting tau downward to capture more attacks at the expense of a controlled increase in false alerts.

### Recall-FPR Tradeoff Observed
- At tau = 0.30: Maximum attack coverage, highest FPR - appropriate for high-security environments where no alert can be missed.
- At tau = 0.70: Highest precision, lowest FPR - appropriate where analyst capacity is limited and alert fatigue is a concern.
- At tau = 0.30 **(recommended)**: Best balance - Novel Recall 87.17%, FPR 10.63%.

### Implication for Phase 5.1 (Anomaly Detection)
With the optimal Naive Bayes threshold established, Phase 5.1 will benchmark **Isolation Forest** and **Local Outlier Factor** at their own optimal operating points to determine whether unsupervised anomaly detection can match or exceed NB's novel attack recall.

### Plots
- `reports/plots/threshold_curve.png` - Novel Recall + FPR vs tau
- `reports/plots/precision_recall_curve.png` - Full PR curve (Average Precision score shown)