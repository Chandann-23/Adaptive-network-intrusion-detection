"""
Phase 5.0 — Naive Bayes Threshold Optimization
================================================
Sweeps the binary classification decision threshold on Naive Bayes from 0.30 to 0.70
and measures the novel attack recall / FPR tradeoff on the untouched KDDTest+ holdout.

Outputs
-------
reports/threshold_analysis.csv          — per-threshold metrics table
reports/threshold_analysis.md           — human-readable markdown report
reports/plots/threshold_curve.png       — novel recall + FPR vs threshold
reports/plots/precision_recall_curve.png — sklearn PR-curve for NB

Usage
-----
    $env:PYTHONPATH="."; python src/models/threshold_analysis.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from src.utils.logger import setup_logger
from src.data.make_dataset import NSL_KDD_COLUMNS

logger = setup_logger("threshold_analysis")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEN_ATTACKS = {
    "back", "buffer_overflow", "ftp_write", "guess_passwd", "imap",
    "ipsweep", "land", "loadmodule", "multihop", "neptune", "nmap",
    "perl", "phf", "pod", "portsweep", "rootkit", "satan", "smurf",
    "spy", "teardrop", "warezclient", "warezmaster",
}

THRESHOLD_MIN   = 0.30
THRESHOLD_MAX   = 0.70
THRESHOLD_STEP  = 0.05
THRESHOLDS      = np.round(np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP / 2, THRESHOLD_STEP), 4)

# Target operating point
TARGET_NOVEL_RECALL = 0.80
TARGET_FPR          = 0.06

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_holdout_data() -> tuple:
    """Load processed test features and raw attack labels from KDDTest+."""
    test_path = "data/processed/test_processed.parquet"
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing processed test dataset: {test_path}. Run build_features.py first.")

    df_test = pd.read_parquet(test_path)
    X_test  = df_test.drop(columns=["target_label", "multiclass_label"]).to_numpy()
    y_test  = df_test["target_label"].to_numpy()

    # Raw labels for seen/novel split
    raw_path = "data/raw/KDDTest+.txt"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing raw holdout file: {raw_path}")

    df_raw        = pd.read_csv(raw_path, names=NSL_KDD_COLUMNS, header=None)
    raw_labels    = df_raw["class"].str.strip().str.lower().values

    is_normal      = raw_labels == "normal"
    is_seen_attack = np.array([x in SEEN_ATTACKS for x in raw_labels])
    is_novel_attack = np.logical_not(is_normal) & np.logical_not(is_seen_attack)

    n_normal  = int(is_normal.sum())
    n_seen    = int(is_seen_attack.sum())
    n_novel   = int(is_novel_attack.sum())

    logger.info(
        f"Holdout breakdown: Total={len(df_test)}, "
        f"Normal={n_normal}, Seen={n_seen}, Novel={n_novel}"
    )
    return X_test, y_test, is_normal, is_seen_attack, is_novel_attack, n_seen, n_novel


def load_nb_model():
    """Load the binary Naive Bayes model from the model registry."""
    model_path = os.path.join("models", "binary", "naive_bayes", "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Naive Bayes binary model not found: {model_path}")
    model = joblib.load(model_path)
    logger.info(f"Loaded Naive Bayes binary model from: {model_path}")
    return model


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    is_normal: np.ndarray,
    is_seen: np.ndarray,
    is_novel: np.ndarray,
    n_seen: int,
    n_novel: int,
) -> pd.DataFrame:
    """
    Iterate over THRESHOLDS and compute per-threshold metrics.

    Positive class = 1 (attack).  Threshold tau means:
        predict attack if P(attack | x) >= tau
    """
    rows = []
    for tau in THRESHOLDS:
        preds = (y_prob >= tau).astype(int)

        # Standard binary metrics
        prec  = precision_score(y_true, preds, zero_division=0)
        rec   = recall_score(y_true, preds, zero_division=0)
        f1    = f1_score(y_true, preds, zero_division=0)
        acc   = accuracy_score(y_true, preds)

        # False positive rate: FP / (FP + TN) = FP / N_normal
        fp    = int(np.sum((preds == 1) & is_normal))
        tn    = int(np.sum((preds == 0) & is_normal))
        fpr   = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Seen / novel attack recall
        seen_rec  = float(np.sum(preds[is_seen] == 1) / n_seen)  if n_seen  > 0 else 0.0
        novel_rec = float(np.sum(preds[is_novel] == 1) / n_novel) if n_novel > 0 else 0.0

        # Total predicted positives (for context)
        n_predicted_attack = int(preds.sum())

        rows.append({
            "threshold":          float(tau),
            "precision":          prec,
            "recall":             rec,
            "novel_attack_recall": novel_rec,
            "seen_attack_recall":  seen_rec,
            "f1_score":           f1,
            "fpr":                fpr,
            "accuracy":           acc,
            "n_predicted_attack": n_predicted_attack,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_threshold_curve(df: pd.DataFrame, out_path: str) -> None:
    """Novel recall and FPR vs decision threshold — the core operating-point chart."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f1117")
    ax1.set_facecolor("#1a1d27")

    color_recall = "#00d4aa"
    color_fpr    = "#ff6b6b"
    color_target = "#ffd166"

    tau   = df["threshold"].values
    novel = df["novel_attack_recall"].values
    fpr   = df["fpr"].values

    ax1.plot(tau, novel, color=color_recall, linewidth=2.5, marker="o", markersize=7,
             label="Novel Attack Recall")
    ax1.set_xlabel("Decision Threshold (tau)", color="white", fontsize=13)
    ax1.set_ylabel("Novel Attack Recall", color=color_recall, fontsize=13)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax1.tick_params(colors="white")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(tau, fpr, color=color_fpr, linewidth=2.5, marker="s", markersize=7,
             linestyle="--", label="False Positive Rate")
    ax2.set_ylabel("False Positive Rate (FPR)", color=color_fpr, fontsize=13)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax2.tick_params(colors="white")
    ax2.set_ylim(0, 0.30)

    # Mark target lines
    ax1.axhline(TARGET_NOVEL_RECALL, color=color_target, linewidth=1.2,
                linestyle=":", alpha=0.8, label=f"Target Recall >= {TARGET_NOVEL_RECALL:.0%}")
    ax2.axhline(TARGET_FPR, color=color_target, linewidth=1.2,
                linestyle="-.", alpha=0.8, label=f"Target FPR <= {TARGET_FPR:.0%}")

    # Highlight optimal operating points
    meets_target = (df["novel_attack_recall"] >= TARGET_NOVEL_RECALL) & (df["fpr"] <= TARGET_FPR)
    if meets_target.any():
        best_tau = df.loc[meets_target, "threshold"].values
        for bt in best_tau:
            ax1.axvline(bt, color=color_target, linewidth=1.5, linestyle="--", alpha=0.6)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left",
               facecolor="#1a1d27", labelcolor="white", fontsize=10, framealpha=0.8)

    ax1.set_title("Naive Bayes: Novel Recall vs FPR by Decision Threshold",
                  color="white", fontsize=15, fontweight="bold", pad=15)

    for spine in ax1.spines.values():
        spine.set_edgecolor("#444")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Threshold curve saved -> {out_path}")


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: str) -> None:
    """Full sklearn precision-recall curve for the Naive Bayes model."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    ax.step(recall, precision, color="#00d4aa", linewidth=2.5, where="post")
    ax.fill_between(recall, precision, step="post", alpha=0.15, color="#00d4aa")

    # Annotate each threshold marker every 0.05
    ax.set_xlabel("Recall", color="white", fontsize=13)
    ax.set_ylabel("Precision", color="white", fontsize=13)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(colors="white")

    ax.set_title(
        f"Naive Bayes - Precision-Recall Curve  (AP = {ap:.4f})",
        color="white", fontsize=14, fontweight="bold", pad=14
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(True, color="#333", linestyle="--", alpha=0.5)

    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Precision-Recall curve saved -> {out_path}")


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def find_optimal_threshold(df: pd.DataFrame) -> pd.Series:  # type: ignore[return]
    """
    Find the single threshold row that best balances:
      - Novel recall >= 80%
      - FPR <= 6%
    Tiebreak by maximum novel recall, then minimum FPR.
    """
    candidates = df[(df["novel_attack_recall"] >= TARGET_NOVEL_RECALL) & (df["fpr"] <= TARGET_FPR)]
    if not candidates.empty:
        return candidates.sort_values(["novel_attack_recall", "fpr"], ascending=[False, True]).iloc[0]  # type: ignore[return-value]
    # Relax: best novel recall regardless of FPR
    return df.sort_values("novel_attack_recall", ascending=False).iloc[0]  # type: ignore[return-value]


def build_markdown_report(df: pd.DataFrame, default_tau: float = 0.50) -> str:
    """Assemble the threshold_analysis.md report content."""
    optimal = find_optimal_threshold(df)
    default = df[df["threshold"] == round(default_tau, 4)].iloc[0]

    meets_target = (
        float(optimal["novel_attack_recall"]) >= TARGET_NOVEL_RECALL
        and float(optimal["fpr"]) <= TARGET_FPR
    )
    operating_point_note = (
        f"OPTIMAL: **Operating point found** at tau = {float(optimal['threshold']):.2f} -- "
        f"satisfies both Novel Recall >= {TARGET_NOVEL_RECALL:.0%} and FPR <= {TARGET_FPR:.0%}."
        if meets_target
        else
        f"WARNING: **No threshold satisfies both targets simultaneously.** "
        f"Best available: tau = {float(optimal['threshold']):.2f} with "
        f"Novel Recall = {float(optimal['novel_attack_recall']):.2%}, FPR = {float(optimal['fpr']):.2%}."
    )

    lines = [
        "# Phase 5.0 - Naive Bayes Threshold Optimization Report",
        "",
        "This analysis sweeps the binary decision threshold tau on the **Naive Bayes** classifier "
        "from 0.30 to 0.70 and measures the impact on **Novel Attack Recall** and "
        "**False Positive Rate (FPR)** on the untouched **KDDTest+** holdout dataset.",
        "",
        "---",
        "",
        "## 1. Default Operating Point (tau = 0.50)",
        "",
        f"| Metric | Value |",
        f"|:---|:---:|",
        f"| Novel Attack Recall | **{float(default['novel_attack_recall']):.2%}** |",
        f"| False Positive Rate | **{float(default['fpr']):.2%}** |",
        f"| Precision | {float(default['precision']):.2%} |",
        f"| Overall Recall | {float(default['recall']):.2%} |",
        f"| F1-Score | {float(default['f1_score']):.2%} |",
        f"| Accuracy | {float(default['accuracy']):.2%} |",
        "",
        "---",
        "",
        "## 2. Target Operating Point",
        "",
        f"| Target | Threshold |",
        f"|:---|:---:|",
        f"| Novel Attack Recall >= {TARGET_NOVEL_RECALL:.0%} | Required |",
        f"| False Positive Rate <= {TARGET_FPR:.0%} | Required |",
        "",
        operating_point_note,
        "",
        f"| Metric | Default (tau=0.50) | Optimal (tau={float(optimal['threshold']):.2f}) | Change |",
        f"|:---|:---:|:---:|:---:|",
        f"| Novel Attack Recall | {float(default['novel_attack_recall']):.2%} | **{float(optimal['novel_attack_recall']):.2%}** | "
        f"{float(optimal['novel_attack_recall']) - float(default['novel_attack_recall']):+.2%} |",
        f"| False Positive Rate | {float(default['fpr']):.2%} | **{float(optimal['fpr']):.2%}** | "
        f"{float(optimal['fpr']) - float(default['fpr']):+.2%} |",
        f"| Precision | {float(default['precision']):.2%} | {float(optimal['precision']):.2%} | "
        f"{float(optimal['precision']) - float(default['precision']):+.2%} |",
        f"| F1-Score | {float(default['f1_score']):.2%} | {float(optimal['f1_score']):.2%} | "
        f"{float(optimal['f1_score']) - float(default['f1_score']):+.2%} |",
        "",
        "---",
        "",
        "## 3. Full Threshold Sweep",
        "",
        "| Threshold (tau) | Novel Recall | Seen Recall | FPR | Precision | F1 | Accuracy |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for _, row in df.iterrows():  # type: ignore[assignment]
        tau_str = f"**{float(row['threshold']):.2f}**" if row["threshold"] == float(optimal["threshold"]) else f"{float(row['threshold']):.2f}"
        lines.append(
            f"| {tau_str} | {float(row['novel_attack_recall']):.2%} | "  # type: ignore[arg-type]
            f"{float(row['seen_attack_recall']):.2%} | "
            f"{float(row['fpr']):.2%} | "
            f"{float(row['precision']):.2%} | "
            f"{float(row['f1_score']):.2%} | "
            f"{float(row['accuracy']):.2%} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Engineering Insights",
        "",
        "### Why Threshold Tuning Matters",
        "The default tau = 0.50 was never explicitly optimised - it's simply the scikit-learn default. "
        "In intrusion detection, the cost of a **False Negative** (missed attack) far exceeds the cost "
        "of a **False Positive** (wasted analyst time). This asymmetry justifies shifting tau downward to "
        "capture more attacks at the expense of a controlled increase in false alerts.",
        "",
        "### Recall-FPR Tradeoff Observed",
        f"- At tau = {THRESHOLD_MIN:.2f}: Maximum attack coverage, highest FPR - appropriate for "
        "high-security environments where no alert can be missed.",
        f"- At tau = {THRESHOLD_MAX:.2f}: Highest precision, lowest FPR - appropriate where analyst "
        "capacity is limited and alert fatigue is a concern.",
        f"- At tau = {float(optimal['threshold']):.2f} **(recommended)**: Best balance - "
        f"Novel Recall {float(optimal['novel_attack_recall']):.2%}, FPR {float(optimal['fpr']):.2%}.",
        "",
        "### Implication for Phase 5.1 (Anomaly Detection)",
        "With the optimal Naive Bayes threshold established, Phase 5.1 will benchmark "
        "**Isolation Forest** and **Local Outlier Factor** at their own optimal operating points "
        "to determine whether unsupervised anomaly detection can match or exceed NB's novel attack recall.",
        "",
        "### Plots",
        "- `reports/plots/threshold_curve.png` - Novel Recall + FPR vs tau",
        "- `reports/plots/precision_recall_curve.png` - Full PR curve (Average Precision score shown)",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_threshold_analysis() -> None:
    logger.info("=== PHASE 5.0: NAIVE BAYES THRESHOLD OPTIMIZATION ===")

    # 1. Load data and model
    X_test, y_test, is_normal, is_seen, is_novel, n_seen, n_novel = load_holdout_data()
    model = load_nb_model()

    # 2. Get attack probabilities (column 1 = P(attack))
    y_prob: np.ndarray = model.predict_proba(X_test)[:, 1]  # type: ignore[assignment]
    logger.info(f"Probability range: min={y_prob.min():.4f}, max={y_prob.max():.4f}, mean={y_prob.mean():.4f}")

    # 3. Sweep thresholds
    logger.info(f"Sweeping thresholds: {THRESHOLDS.tolist()}")
    df_sweep = sweep_thresholds(y_test, y_prob, is_normal, is_seen, is_novel, n_seen, n_novel)

    # 4. Save CSV
    os.makedirs("reports", exist_ok=True)
    csv_path = "reports/threshold_analysis.csv"
    df_sweep.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info(f"Threshold sweep CSV saved -> {csv_path}")

    # 5. Generate plots
    os.makedirs("reports/plots", exist_ok=True)
    plot_threshold_curve(df_sweep, "reports/plots/threshold_curve.png")
    plot_precision_recall_curve(y_test, y_prob, "reports/plots/precision_recall_curve.png")

    # 6. Generate markdown report
    md_content = build_markdown_report(df_sweep)
    md_path = "reports/threshold_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Threshold analysis report saved -> {md_path}")

    # 7. Print headline to stdout
    optimal = find_optimal_threshold(df_sweep)
    print("\n" + "=" * 60)
    print("PHASE 5.0 COMPLETE - Naive Bayes Threshold Optimization")
    print("=" * 60)
    print(f"{'Threshold':>12} {'Novel Recall':>14} {'FPR':>8} {'F1':>8}")
    print("-" * 48)
    for _, row in df_sweep.iterrows():  # type: ignore[assignment]
        marker = " << OPTIMAL" if float(row["threshold"]) == float(optimal["threshold"]) else ""
        print(
            f"  tau={float(row['threshold']):.2f}   "
            f"{float(row['novel_attack_recall']):>12.2%}   "
            f"{float(row['fpr']):>6.2%}   "
            f"{float(row['f1_score']):>6.2%}{marker}"
        )
    print("=" * 60)
    print(f"Reports:  {csv_path}")
    print(f"          {md_path}")
    print(f"Plots:    reports/plots/threshold_curve.png")
    print(f"          reports/plots/precision_recall_curve.png")


if __name__ == "__main__":
    run_threshold_analysis()
