#!/usr/bin/env python3
"""
Empirical analysis of class-specific metrics when abstaining 
on low-confidence samples. (Manual calculation to avoid sklearn quirks).
"""

import json
import argparse
import numpy as np
import torch

def load_logits(path: str):
    samples = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

def analyze_abstention(logits_file: str):
    print(f"Loading logits from {logits_file}...")
    samples = load_logits(logits_file)
    
    logits = np.array([[s["logit_0"], s["logit_1"]] for s in samples])
    labels = np.array([s["true_label"] for s in samples]) # 0 = Harmful, 1 = Benign
    
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    predictions = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    
    # Sort from HIGHEST confidence to LOWEST
    sorted_indices = np.argsort(-confidence)
    sorted_preds = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    total_samples = len(sorted_labels)
    total_harmful = np.sum(labels == 0)
    total_benign = np.sum(labels == 1)
    
    print(f"\nTotal Dataset: {total_samples} (Harmful: {total_harmful}, Benign: {total_benign})")
    print("=" * 70)
    
    coverages = [1.0, 0.75, 0.50, 0.25]
    
    for cov in coverages:
        n_keep = int(total_samples * cov)
        keep_preds = sorted_preds[:n_keep]
        keep_labels = sorted_labels[:n_keep]
        
        remaining_harmful = np.sum(keep_labels == 0)
        remaining_benign = np.sum(keep_labels == 1)
        
        # --- MANUAL METRIC CALCULATION ---
        # Harmful (Class 0)
        tp_0 = np.sum((keep_preds == 0) & (keep_labels == 0))
        fn_0 = np.sum((keep_preds == 1) & (keep_labels == 0))
        fp_0 = np.sum((keep_preds == 0) & (keep_labels == 1))
        
        h_recall = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) > 0 else 0.0
        h_precision = tp_0 / (tp_0 + fp_0) if (tp_0 + fp_0) > 0 else 0.0
        h_f1 = 2 * (h_precision * h_recall) / (h_precision + h_recall) if (h_precision + h_recall) > 0 else 0.0
        
        # Benign (Class 1)
        tp_1 = np.sum((keep_preds == 1) & (keep_labels == 1))
        fn_1 = np.sum((keep_preds == 0) & (keep_labels == 1))
        fp_1 = np.sum((keep_preds == 1) & (keep_labels == 0))
        
        b_recall = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) > 0 else 0.0
        b_precision = tp_1 / (tp_1 + fp_1) if (tp_1 + fp_1) > 0 else 0.0
        b_f1 = 2 * (b_precision * b_recall) / (b_precision + b_recall) if (b_precision + b_recall) > 0 else 0.0

        print(f"\n📊 Coverage: {cov*100:.0f}% (Keeping {n_keep} samples)")
        print(f"   Remaining Harmful: {remaining_harmful} ({remaining_harmful/total_harmful*100:.1f}% of total)")
        print(f"   Remaining Benign:  {remaining_benign} ({remaining_benign/total_benign*100:.1f}% of total)")
        print(f"   -> Harmful Recall: {h_recall:.4f} | Precision: {h_precision:.4f} | F1: {h_f1:.4f}")
        print(f"   -> Benign  Recall: {b_recall:.4f} | Precision: {b_precision:.4f} | F1: {b_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-file", type=str, required=True)
    args = parser.parse_args()
    analyze_abstention(args.logits_file)