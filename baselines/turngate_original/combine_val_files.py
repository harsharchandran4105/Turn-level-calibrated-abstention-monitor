import numpy as np

print("[*] Loading benign files...")
benign_logits = np.load("benign_val_logits.npy")
benign_labels = np.load("benign_val_labels.npy")

print("[*] Loading harmful files...")
harmful_logits = np.load("harmful_val_logits.npy")
harmful_labels = np.load("harmful_val_labels.npy")

print("[*] Combining files...")
# Concatenate along the sample dimension
val_logits = np.vstack([benign_logits, harmful_logits])
val_labels = np.concatenate([benign_labels, harmful_labels])

# Save combined files
np.save("val_logits.npy", val_logits)
np.save("val_labels.npy", val_labels)

print(f"\n[✓] Success! Combined validation files saved.")
print(f"[*] val_logits.npy shape: {val_logits.shape}  (N, 2) -> [logit_BLOCK, logit_PASS]")
print(f"[*] val_labels.npy shape: {val_labels.shape}  (N,)")
print(f"[*] Label distribution: {np.bincount(val_labels)}  (0=BLOCK, 1=PASS)")
print(f"[*] Total samples: {len(val_labels)}")