import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# ==========================================
# 1. Load Validation Data
# ==========================================
print("[*] Loading validation data...")
val_logits = np.load("val_logits.npy")  # Shape: (N, 2) -> [logit_BLOCK, logit_PASS]
val_labels = np.load("val_labels.npy")  # Shape: (N,) -> 0=BLOCK, 1=PASS

print(f"[*] Loaded {len(val_labels)} samples")
print(f"[*] Logits shape: {val_logits.shape}")
print(f"[*] Label distribution: {np.bincount(val_labels)} (0=BLOCK, 1=PASS)")

# Convert to PyTorch tensors
logits_tensor = torch.tensor(val_logits, dtype=torch.float32)
labels_tensor = torch.tensor(val_labels, dtype=torch.long)

# ==========================================
# 2. Define Calibration Metrics
# ==========================================
def compute_ece(confidences, predictions, labels, n_bins=15):
    """
    Computes Expected Calibration Error (ECE).
    
    Args:
        confidences: Array of confidence scores (max probability)
        predictions: Array of predicted classes (0 or 1)
        labels: Array of true labels (0 or 1)
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Average confidence in this bin
            avg_confidence = confidences[in_bin].mean()
            # Average accuracy in this bin
            avg_accuracy = (predictions[in_bin] == labels[in_bin]).mean()
            # Add to ECE
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
    
    return ece

def compute_nll(temperature, logits, labels):
    """
    Computes Negative Log-Likelihood (NLL) for a given temperature.
    This is the loss function we minimize to find optimal T.
    """
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    # Cross-entropy loss = NLL for classification
    nll = F.cross_entropy(scaled_logits, labels)
    return nll.item()

# ==========================================
# 3. Compute Baseline Metrics (T=1.0)
# ==========================================
print("\n[*] Computing baseline metrics (T=1.0)...")
baseline_probs = F.softmax(logits_tensor, dim=-1)
baseline_confidences, baseline_predictions = torch.max(baseline_probs, dim=-1)

baseline_ece = compute_ece(
    baseline_confidences.numpy(),
    baseline_predictions.numpy(),
    val_labels,
    n_bins=15
)
baseline_accuracy = (baseline_predictions.numpy() == val_labels).mean()
baseline_nll = compute_nll(1.0, logits_tensor, labels_tensor)

print(f"    Baseline ECE: {baseline_ece:.4f}")
print(f"    Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"    Baseline NLL: {baseline_nll:.4f}")

# ==========================================
# 4. Optimize Temperature
# ==========================================
print("\n[*] Optimizing temperature T...")

# Define the objective function for scipy
def objective(T):
    return compute_nll(T, logits_tensor, labels_tensor)

# Use scipy to find the optimal T (bounded between 0.1 and 10.0)
result = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')
optimal_T = result.x
optimal_nll = result.fun

print(f"[✓] Optimization complete!")
print(f"    Optimal Temperature (T): {optimal_T:.4f}")
print(f"    Optimal NLL: {optimal_nll:.4f}")

# ==========================================
# 5. Compute Calibrated Metrics
# ==========================================
print("\n[*] Computing calibrated metrics...")
calibrated_logits = logits_tensor / optimal_T
calibrated_probs = F.softmax(calibrated_logits, dim=-1)
calibrated_confidences, calibrated_predictions = torch.max(calibrated_probs, dim=-1)

calibrated_ece = compute_ece(
    calibrated_confidences.numpy(),
    calibrated_predictions.numpy(),
    val_labels,
    n_bins=15
)
calibrated_accuracy = (calibrated_predictions.numpy() == val_labels).mean()

print(f"    Calibrated ECE: {calibrated_ece:.4f}")
print(f"    Calibrated Accuracy: {calibrated_accuracy:.4f}")
print(f"    Calibrated NLL: {optimal_nll:.4f}")

# ==========================================
# 6. Save Results
# ==========================================
# Save the optimal temperature
with open("optimal_temperature.txt", "w") as f:
    f.write(f"{optimal_T:.6f}\n")
print(f"\n[✓] Saved optimal temperature to optimal_temperature.txt")

# Save calibrated logits for later use
calibrated_logits_np = calibrated_logits.numpy()
np.save("val_logits_calibrated.npy", calibrated_logits_np)
print(f"[✓] Saved calibrated logits to val_logits_calibrated.npy")

# ==========================================
# 7. Plot Reliability Diagrams
# ==========================================
print("\n[*] Generating reliability diagrams...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Baseline (T=1.0)
ax1 = axes[0]
bin_boundaries = np.linspace(0, 1, 16)
bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
bin_accuracies = []
bin_confidences = []

for i in range(15):
    in_bin = (baseline_confidences.numpy() > bin_boundaries[i]) & \
             (baseline_confidences.numpy() <= bin_boundaries[i + 1])
    if in_bin.sum() > 0:
        bin_confidences.append(baseline_confidences.numpy()[in_bin].mean())
        bin_accuracies.append((baseline_predictions.numpy()[in_bin] == val_labels[in_bin]).mean())
    else:
        bin_confidences.append(bin_centers[i])
        bin_accuracies.append(0)

ax1.bar(bin_centers, bin_accuracies, width=0.06, alpha=0.6, label='Accuracy')
ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
ax1.set_xlabel('Confidence')
ax1.set_ylabel('Accuracy')
ax1.set_title(f'Baseline (T=1.0)\nECE = {baseline_ece:.4f}')
ax1.legend()
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.grid(True, alpha=0.3)

# Plot 2: Calibrated (T=optimal_T)
ax2 = axes[1]
bin_accuracies_cal = []
bin_confidences_cal = []

for i in range(15):
    in_bin = (calibrated_confidences.numpy() > bin_boundaries[i]) & \
             (calibrated_confidences.numpy() <= bin_boundaries[i + 1])
    if in_bin.sum() > 0:
        bin_confidences_cal.append(calibrated_confidences.numpy()[in_bin].mean())
        bin_accuracies_cal.append((calibrated_predictions.numpy()[in_bin] == val_labels[in_bin]).mean())
    else:
        bin_confidences_cal.append(bin_centers[i])
        bin_accuracies_cal.append(0)

ax2.bar(bin_centers, bin_accuracies_cal, width=0.06, alpha=0.6, label='Accuracy', color='green')
ax2.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
ax2.set_xlabel('Confidence')
ax2.set_ylabel('Accuracy')
ax2.set_title(f'Calibrated (T={optimal_T:.2f})\nECE = {calibrated_ece:.4f}')
ax2.legend()
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reliability_diagrams.png", dpi=300, bbox_inches='tight')
print(f"[✓] Saved reliability diagrams to reliability_diagrams.png")

# ==========================================
# 8. Summary
# ==========================================
print("\n" + "=" * 60)
print("TEMPERATURE SCALING SUMMARY")
print("=" * 60)
print(f"Optimal Temperature (T): {optimal_T:.4f}")
print(f"\nBaseline (T=1.0):")
print(f"  ECE: {baseline_ece:.4f}")
print(f"  Accuracy: {baseline_accuracy:.4f}")
print(f"  NLL: {baseline_nll:.4f}")
print(f"\nCalibrated (T={optimal_T:.4f}):")
print(f"  ECE: {calibrated_ece:.4f}")
print(f"  Accuracy: {calibrated_accuracy:.4f}")
print(f"  NLL: {optimal_nll:.4f}")
print(f"\nImprovement:")
print(f"  ECE reduction: {baseline_ece - calibrated_ece:.4f} ({((baseline_ece - calibrated_ece) / baseline_ece * 100):.1f}%)")
print("=" * 60)
print("\n[✓] Temperature scaling complete!")
print("[*] Next step: Use optimal T to calibrate test set logits and apply abstention threshold.")