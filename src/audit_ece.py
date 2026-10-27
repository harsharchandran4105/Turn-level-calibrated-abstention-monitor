import json
import matplotlib.pyplot as plt
import numpy as np

def audit_ece(filepath):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    confidences = []
    accuracies = []
    
    for item in data:
        conf = item["confidence"]
        # True label: 1 = Benign, 0 = Harmful. 
        # We evaluate the raw model's turngate_pred against the true label.
        is_correct = (item["true_label"] == 1 and item["turngate_pred"] == "PASS") or \
                     (item["true_label"] == 0 and item["turngate_pred"] == "BLOCK")
        confidences.append(conf)
        accuracies.append(1 if is_correct else 0)
        
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)
    
    print(f"| Bin Range | Count (N) | Avg Confidence | Actual Accuracy | Gap |")
    print(f"|---|---|---|---|---|")
    
    for i in range(num_bins):
        in_bin = (np.array(confidences) >= bin_boundaries[i]) & (np.array(confidences) < bin_boundaries[i+1])
        # Fix for the last bin to include 1.0
        if i == num_bins - 1:
            in_bin = (np.array(confidences) >= bin_boundaries[i]) & (np.array(confidences) <= bin_boundaries[i+1])
            
        n_b = np.sum(in_bin)
        if n_b > 0:
            avg_conf = np.mean(np.array(confidences)[in_bin])
            avg_acc = np.mean(np.array(accuracies)[in_bin])
            gap = abs(avg_acc - avg_conf)
            ece += (n_b / total_samples) * gap
            print(f"| {bin_boundaries[i]:.1f}-{bin_boundaries[i+1]:.1f} | {n_b:4d} | {avg_conf:14.3f} | {avg_acc:15.3f} | {gap:5.3f} |")
        else:
            print(f"| {bin_boundaries[i]:.1f}-{bin_boundaries[i+1]:.1f} | {0:4d} | {'N/A':>14} | {'N/A':>15} | {'N/A':>5} |")

    print(f"\nTotal Samples: {total_samples}")
    print(f"Expected Calibration Error (ECE): {ece*100:.2f}%")

# Run on the 100% coverage data (tau=0.50)
audit_ece("sweep_results/benign_tau_0.50.jsonl")
# You should also run it on the harmful set and combine them for the full picture, 
# but this gives you the exact audit table the reviewer demanded.