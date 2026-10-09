#!/usr/bin/env python3
"""
Generates a final, publication-quality Reliability Diagram comparing 
Uncalibrated vs. Calibrated (T=4.82) predictions.
"""

import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def load_logits(path: str):
    samples = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

def compute_exact_ece(probs, labels, n_bins=15):
    """Calculates the exact sample-level ECE using uniform binning."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i+1])
        if i == n_bins - 1: # Include 1.0 in the last bin
            in_bin = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
        
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_confidence = np.mean(probs[in_bin])
            avg_accuracy = np.mean(labels[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    return ece

def generate_final_diagram(logits_file: str, output_dir: str, T_calib: float = 4.82, n_bins: int = 15):
    print(f"Loading logits from {logits_file}...")
    samples = load_logits(logits_file)
    print(f"Loaded {len(samples)} samples")
    
    logits = np.array([[s["logit_0"], s["logit_1"]] for s in samples])
    labels = np.array([s["true_label"] for s in samples])
    
    # 1. Uncalibrated Probabilities
    probs_uncal = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1] # Prob of class 1 (Benign)
    preds_uncal = np.argmax(probs_uncal > 0.5, axis=0).astype(float) # 1 if prob > 0.5 else 0
    # Note: For binary, argmax of softmax is just prob > 0.5. 
    # Let's use the actual predictions from the logits to be safe:
    preds_uncal = (logits[:, 1] > logits[:, 0]).astype(float)
    correct_uncal = (preds_uncal == labels).astype(float)
    
    # 2. Calibrated Probabilities (T = 4.82)
    logits_calib = logits / T_calib
    probs_calib = torch.softmax(torch.tensor(logits_calib), dim=1).numpy()[:, 1]
    preds_calib = (logits_calib[:, 1] > logits_calib[:, 0]).astype(float)
    correct_calib = (preds_calib == labels).astype(float)
    
    # 3. Calculate Exact ECE (15 bins)
    ece_uncal = compute_exact_ece(probs_uncal, correct_uncal, n_bins)
    ece_calib = compute_exact_ece(probs_calib, correct_calib, n_bins)
    
    print(f"\n✅ Exact ECE (Uncalibrated): {ece_uncal:.4f}")
    print(f"✅ Exact ECE (Calibrated, T={T_calib}): {ece_calib:.4f}")

    # 4. Plotting
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # --- Top Plot: Reliability Diagram ---
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Perfect Calibration (y=x)')
    
    # Calculate bin accuracies for both
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    acc_uncal = []
    acc_calib = []
    
    for i in range(n_bins):
        in_bin = (probs_uncal >= bin_edges[i]) & (probs_uncal < bin_edges[i+1])
        if i == n_bins - 1: in_bin = (probs_uncal >= bin_edges[i]) & (probs_uncal <= bin_edges[i+1])
        
        acc_uncal.append(np.mean(correct_uncal[in_bin]) if np.sum(in_bin) > 0 else 0)
        acc_calib.append(np.mean(correct_calib[in_bin]) if np.sum(in_bin) > 0 else 0)
        
    # Plot Uncalibrated (Blue solid line)
    ax1.plot(bin_centers, acc_uncal, marker='o', color='#1f77b4', linewidth=2.5, markersize=6, label=f'Uncalibrated (ECE={ece_uncal:.4f})')
    
    # Plot Calibrated (Red dashed line)
    ax1.plot(bin_centers, acc_calib, marker='s', color='#d62728', linewidth=2.5, markersize=6, linestyle='--', label=f'Calibrated T={T_calib} (ECE={ece_calib:.4f})')
    
    ax1.set_ylabel('Empirical Accuracy', fontsize=14)
    ax1.set_title('Reliability Diagram: Naive-SFT Defender Before and After Temperature Scaling', fontsize=16)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # --- Bottom Plot: Histogram of Uncalibrated Confidence ---
    ax2.hist(probs_uncal, bins=50, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Predicted Probability (Class 1: Benign/Pass)', fontsize=14)
    ax2.set_ylabel('Sample Count', fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / "reliability_diagram_final.png", dpi=300, bbox_inches='tight')
    print(f"✅ Final plot saved to: {output_path / 'reliability_diagram_final.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="calibration_results_test_fixed")
    parser.add_argument("--T", type=float, default=4.82)
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()
    
    generate_final_diagram(args.logits_file, args.output_dir, args.T, args.n_bins)