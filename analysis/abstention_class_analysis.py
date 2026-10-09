#!/usr/bin/env python3
"""
Empirical analysis of what happens to the Harmful class 
when we abstain on low-confidence samples.
"""

import json
import argparse
import numpy as np
import torch
from sklearn.metrics import recall_score, f1_score

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
    # Note: In this dataset, 0 = Harmful (BLOCK), 1 = Benign (PASS)
    labels = np.array([s["true_label"] for s in samples]) 
    
    # Convert to probabilities
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    predictions = np.argmax(probs, axis=1)
    
    # Confidence = max probability (how sure the model is about its prediction)
    confidence = np.max(probs, axis=1)
    
    # Sort samples from HIGHEST confidence to LOWEST confidence
    sorted_indices = np.argsort(-confidence)
    sorted_preds = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    total_samples = len(sorted_labels)
    total_harmful = np.sum(labels == 0)
    total_benign = np.sum(labels == 1)
    
    print(f"\nTotal Dataset: {total_samples} (Harmful: {total_harmful}, Benign: {total_benign})")
    print("=" * 70)
    
    # Check coverages: 100%, 75%, 50%, 25%
    coverages = [1.0, 0.75, 0.50, 0.25]
    
    for cov in coverages:
        n_keep = int(total_samples * cov)
        
        # Subset of data we are keeping (the high-confidence samples)
        keep_preds = sorted_preds[:n_keep]
        keep_labels = sorted_labels[:n_keep]
        
        # 1. How many harmful/benign samples remain?
        remaining_harmful = np.sum(keep_labels == 0)
        remaining_benign = np.sum(keep_labels == 1)
        
        # 2 & 3. Harmful Recall and F1 (Only calculated if there are harmful samples left)
        if remaining_harmful > 0:
            # For class 0 (Harmful): True Positive = predicted 0 and actual 0
            harmful_recall = recall_score(keep_labels, keep_preds, labels=[0], average='binary', zero_division=0)
            harmful_f1 = f1_score(keep_labels, keep_preds, labels=[0], average='binary', zero_division=0)
        else:
            harmful_recall = 0.0
            harmful_f1 = 0.0
            
        # Benign Recall and F1
        if remaining_benign > 0:
            benign_recall = recall_score(keep_labels, keep_preds, labels=[1], average='binary', zero_division=0)
            benign_f1 = f1_score(keep_labels, keep_preds, labels=[1], average='binary', zero_division=0)
        else:
            benign_recall = 0.0
            benign_f1 = 0.0

        print(f"\n📊 Coverage: {cov*100:.0f}% (Keeping {n_keep} samples)")
        print(f"   Remaining Harmful: {remaining_harmful} / {total_harmful} ({remaining_harmful/total_harmful*100:.1f}% of total)")
        print(f"   Remaining Benign:  {remaining_benign} / {total_benign} ({remaining_benign/total_benign*100:.1f}% of total)")
        print(f"   -> Harmful Recall: {harmful_recall:.4f}")
        print(f"   -> Harmful F1:     {harmful_f1:.4f}")
        print(f"   -> Benign Recall:  {benign_recall:.4f}")
        print(f"   -> Benign F1:      {benign_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits-file", type=str, required=True)
    args = parser.parse_args()
    analyze_abstention(args.logits_file)