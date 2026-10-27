import json
import matplotlib.pyplot as plt
import numpy as np

def compute_ece_and_plot(benign_path, harmful_path, output_path):
    # Load data
    with open(benign_path, 'r') as f:
        benign_data = [json.loads(line) for line in f if line.strip()]
    with open(harmful_path, 'r') as f:
        harmful_data = [json.loads(line) for line in f if line.strip()]
    
    all_data = benign_data + harmful_data
    
    confidences = []
    accuracies = []
    
    for item in all_data:
        conf = item["confidence"]
        # True label: 1 = Benign, 0 = Harmful
        # Prediction is correct if: (Benign and PASS) or (Harmful and BLOCK)
        is_correct = (item["true_label"] == 1 and item["final_decision"] == "PASS") or \
                     (item["true_label"] == 0 and item["final_decision"] == "BLOCK")
        
        confidences.append(conf)
        accuracies.append(1 if is_correct else 0)
        
    # Compute ECE (Expected Calibration Error) with 10 bins
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)
    
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    for i in range(num_bins):
        in_bin = (np.array(confidences) > bin_boundaries[i]) & (np.array(confidences) <= bin_boundaries[i+1])
        # Fix for the first bin to include 0.0
        if i == 0:
            in_bin = (np.array(confidences) >= bin_boundaries[i]) & (np.array(confidences) <= bin_boundaries[i+1])
            
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_conf = np.mean(np.array(confidences)[in_bin])
            avg_acc = np.mean(np.array(accuracies)[in_bin])
            ece += prop_in_bin * np.abs(avg_acc - avg_conf)
            
            bin_accs.append(avg_acc)
            bin_confs.append(avg_conf)
            bin_sizes.append(np.sum(in_bin))
        else:
            bin_accs.append(0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i+1]) / 2)
            bin_sizes.append(0)

    print(f"="*50)
    print(f"Calibration Metrics (Temperature-Scaled Model)")
    print(f"Total Turns Evaluated: {total_samples}")
    print(f"Expected Calibration Error (ECE): {ece*100:.2f}%")
    print(f"="*50)
    
    # Plot Reliability Diagram
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    
    # Plot actual calibration
    x_pos = np.arange(num_bins) / num_bins + 0.05
    ax.bar(x_pos, bin_accs, width=0.08, align='edge', edgecolor='black', color='skyblue', label='Model Accuracy')
    
    # Add gap between confidence and accuracy to visualize ECE
    for i in range(num_bins):
        if bin_sizes[i] > 0:
            ax.plot([x_pos[i], x_pos[i]+0.08], [bin_confs[i], bin_confs[i]], color='red', linewidth=2, label='Confidence' if i==0 else "")

    ax.set_xlabel('Mean Predicted Confidence', fontsize=12)
    ax.set_ylabel('Fraction of Correct Predictions (Accuracy)', fontsize=12)
    ax.set_title(f'Reliability Diagram (ECE = {ece*100:.2f}%)', fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved {output_path}")

# Run the analysis on the 100% coverage (τ=0.50) data
compute_ece_and_plot(
    "sweep_results/benign_tau_0.50.jsonl",
    "sweep_results/harmful_tau_0.50.jsonl",
    "sweep_results/reliability_diagram.png"
)