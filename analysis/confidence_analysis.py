#!/usr/bin/env python3
"""
Comprehensive Confidence-Correctness Analysis for LLM Safety Defenders.

Calculates Spearman correlation, AUROC, AUPRC, and Risk-Coverage metrics 
to evaluate whether the model's confidence ranking correlates with actual correctness.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path


def load_logits(path: str):
    """Loads logits and labels from a JSONL file."""
    samples = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def analyze_confidence_correctness(logits_file: str, output_dir: str):
    print(f"Loading logits from {logits_file}...")
    samples = load_logits(logits_file)
    print(f"Loaded {len(samples)} samples")
    
    if len(samples) == 0:
        print("Error: No samples found in the logits file.")
        return

    # Extract data
    logits = np.array([[s["logit_0"], s["logit_1"]] for s in samples])
    labels = np.array([s["true_label"] for s in samples])
    
    # Calculate predictions and correctness
    predictions = np.argmax(logits, axis=1)
    correctness = (predictions == labels).astype(float)
    
    # Calculate confidence metric (Logit Margin)
    # Higher margin = higher confidence
    logit_margin = np.abs(logits[:, 0] - logits[:, 1])
    
    print("\n" + "=" * 70)
    print("CONFIDENCE-CORRECTNESS ANALYSIS")
    print("=" * 70)
    
    # === 1. Spearman Rank Correlation ===
    spearman_corr, p_value = spearmanr(logit_margin, correctness)
    print(f"\n1. Spearman Rank Correlation (Confidence vs Correctness):")
    print(f"   ρ = {spearman_corr:.4f} (p-value = {p_value:.4e})")
    
    # === 2. AUROC and AUPRC ===
    # If there is only one class in correctness (e.g., all correct), AUROC will throw an error.
    if len(np.unique(correctness)) > 1:
        auroc = roc_auc_score(correctness, logit_margin)
        auprc = average_precision_score(correctness, logit_margin)
        print(f"\n2. Ranking Metrics:")
        print(f"   AUROC = {auroc:.4f} (0.5 = random guessing)")
        print(f"   AUPRC = {auprc:.4f}")
    else:
        print("\n2. Ranking Metrics: Skipped (All predictions are either correct or incorrect).")
        auroc, auprc = None, None

    # === 3. Risk-Coverage Curve & AURC ===
    # Sort samples by confidence (descending)
    sorted_idx = np.argsort(-logit_margin)
    sorted_correctness = correctness[sorted_idx]
    
    n_samples = len(sorted_correctness)
    
    # Calculate risk at 1000 evenly spaced coverage points for a smooth curve
    coverages = np.linspace(0.01, 1.0, 1000)
    risks = []
    
    for cov in coverages:
        n_keep = max(1, int(n_samples * cov))
        risk = 1.0 - np.mean(sorted_correctness[:n_keep])
        risks.append(risk)
        
    risks = np.array(risks)
    
    # Calculate Area Under Risk-Coverage Curve (AURC)
    aurc = np.trapz(risks, coverages)
    random_baseline_risk = 1.0 - np.mean(correctness)
    
    print(f"\n3. Risk-Coverage Metrics:")
    print(f"   Overall Accuracy: {np.mean(correctness):.4f}")
    print(f"   Random Baseline Risk: {random_baseline_risk:.4f}")
    print(f"   AURC: {aurc:.4f} (Lower is better)")

    # === 4. Save Results to JSON ===
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "spearman": {"rho": float(spearman_corr), "p_value": float(p_value)},
        "auroc": float(auroc) if auroc is not None else None,
        "auprc": float(auprc) if auprc is not None else None,
        "aurc": float(aurc),
        "random_baseline_risk": float(random_baseline_risk),
        "overall_accuracy": float(np.mean(correctness)),
        "n_samples": n_samples
    }
    
    with open(output_path / "confidence_analysis_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Metrics saved to: {output_path / 'confidence_analysis_metrics.json'}")

    # === 5. Plot Risk-Coverage Curve ===
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(coverages, risks, label="Model (Sorted by Confidence)", linewidth=2.5, color='#1f77b4')
    ax.axhline(y=random_baseline_risk, color='red', linestyle='--', linewidth=2, label=f"Random Baseline ({random_baseline_risk:.4f})")
    
    ax.set_xlabel("Coverage (Fraction of samples retained)", fontsize=14)
    ax.set_ylabel("Risk (1 - Accuracy)", fontsize=14)
    ax.set_title("Risk-Coverage Curve: Confidence vs. Correctness", fontsize=16)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path / "risk_coverage_curve.png", dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path / 'risk_coverage_curve.png'}")

    # === 6. Thesis Interpretation ===
    print("\n" + "=" * 70)
    print("THESIS INTERPRETATION")
    print("=" * 70)
    if abs(spearman_corr) < 0.1 and (auroc is None or abs(auroc - 0.5) < 0.05):
        print("✅ CONFIRMED: Confidence ordering is statistically uncorrelated with correctness.")
        print("   This proves that softmax-based selective prediction (abstention) will fail,")
        print("   as the model cannot distinguish correct from incorrect predictions via confidence.")
    else:
        print("⚠️  NOTE: Confidence shows some correlation with correctness. Selective prediction")
        print("   may yield marginal improvements, but likely insufficient for safety-critical abstention.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confidence-Correctness Analysis")
    parser.add_argument("--logits-file", type=str, required=True, help="Path to JSONL file with logits")
    parser.add_argument("--output-dir", type=str, default="confidence_analysis", help="Directory to save results and plots")
    args = parser.parse_args()
    
    analyze_confidence_correctness(args.logits_file, args.output_dir)