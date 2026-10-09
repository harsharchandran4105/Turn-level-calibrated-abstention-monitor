import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Parse the metrics files
taus = []
benign_scores = []
exact_accs = []
early_blocks = []

for tau in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
    filepath = f"sweep_results/metrics_tau_{tau}.txt"
    if not os.path.exists(filepath):
        continue
        
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Extract Benign Score
    benign_match = re.search(r'Benign Score \(1 - BFPR\)\s+\|\s+([\d\.]+)', content)
    if benign_match:
        benign_scores.append(float(benign_match.group(1)))
        
    # Extract Exact Acc (phi1)
    acc_match = re.search(r'Exact Turn Accuracy \(phi1\) \(%\)\s+\|\s+([\d\.]+)', content)
    if acc_match:
        exact_accs.append(float(acc_match.group(1)) / 100.0) # Convert to 0-1 scale
        
    # Extract Early Block
    early_match = re.search(r'Early Block Rate \(%\)\s+\|\s+([\d\.]+)', content)
    if early_match:
        early_blocks.append(float(early_match.group(1)))
        
    taus.append(tau)

# Plot 1: Safety (Exact Acc) vs. Utility (Benign Score)
plt.figure(figsize=(8, 6))
plt.plot(benign_scores, exact_accs, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)

# Annotate points with Tau values
for i, tau in enumerate(taus):
    plt.annotate(f'τ={tau}', (benign_scores[i], exact_accs[i]), textcoords="offset points", xytext=(5,5), fontsize=9)

plt.xlabel('Benign Score (Utility) ↑', fontsize=12)
plt.ylabel('Exact Turn Accuracy φ₁ (Safety) ↑', fontsize=12)
plt.title('Safety-Utility Trade-off: Calibrated Sequential Abstention', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(0.5, 0.9)
plt.ylim(0.3, 0.8)
plt.savefig('safety_utility_tradeoff.png', dpi=300, bbox_inches='tight')
print("[✓] Saved safety_utility_tradeoff.png")

# Plot 2: Early Blocking vs. Tau
plt.figure(figsize=(8, 6))
plt.plot(taus, early_blocks, marker='s', linestyle='-', color='r', linewidth=2, markersize=8)
plt.xlabel('Abstention Threshold (τ)', fontsize=12)
plt.ylabel('Early Block Rate (%) ↓', fontsize=12)
plt.title('Impact of Abstention Threshold on Premature Blocking', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('early_blocking_vs_tau.png', dpi=300, bbox_inches='tight')
print("[✓] Saved early_blocking_vs_tau.png")