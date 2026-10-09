import json
import numpy as np
import os

def compute_transition_matrix(data_dir, suspicious_window=1):
    """
    Computes the empirical 3x3 transition matrix for the DMC fallback.
    States: 0=Benign, 1=Suspicious, 2=Harmful (absorbing)
    """
    # counts[i][j] = number of transitions from state i to state j
    counts = np.zeros((3, 3), dtype=np.float64)
    total_transitions = 0

    for filename in os.listdir(data_dir):
        if not filename.endswith('.jsonl'):
            continue
            
        filepath = os.path.join(data_dir, filename)
        is_harmful = "harmful" in filename.lower()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                traj = json.loads(line)
                conversation = traj.get("conversation", [])
                
                # Group by turn_id
                turns_dict = {}
                for turn in conversation:
                    tid = turn["turn_id"]
                    if tid not in turns_dict:
                        turns_dict[tid] = {}
                    turns_dict[tid][turn["role"]] = turn["content"]
                
                completed_tids = sorted([
                    tid for tid, roles in turns_dict.items()
                    if "user" in roles and "assistant" in roles
                ])
                
                if len(completed_tids) < 2:
                    continue # Need at least 2 turns for a transition
                
                # Assign states
                states = []
                if is_harmful:
                    t_star = completed_tids[-1] # Last turn is closure
                    for tid in completed_tids:
                        if tid >= t_star:
                            states.append(2) # Harmful
                        elif tid >= t_star - suspicious_window:
                            states.append(1) # Suspicious
                        else:
                            states.append(0) # Benign
                else:
                    # Benign trajectories are always in state 0
                    states = [0] * len(completed_tids)
                
                # Count transitions
                for i in range(len(states) - 1):
                    s_curr = states[i]
                    s_next = states[i+1]
                    counts[s_curr][s_next] += 1
                    total_transitions += 1

    # Normalize rows to get probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 # Avoid division by zero
    P = counts / row_sums
    
    # Force S2 (Harmful) to be an absorbing state
    P[2] = [0.0, 0.0, 1.0]
    
    return P, total_transitions

if __name__ == "__main__":
    # Point this to your MTID training data directory
    DATA_DIR = "./dataset/gpt52-gen_filter" 
    
    print("[*] Computing empirical DMC transition matrix...")
    P, total_trans = compute_transition_matrix(DATA_DIR, suspicious_window=1)
    
    print(f"[*] Processed {total_trans} state transitions.")
    print("[*] Learned Transition Matrix P:")
    print("    " + " "*12 + "→Benign".rjust(10) + "→Suspicious".rjust(12) + "→Harmful".rjust(10))
    states = ["Benign", "Suspicious", "Harmful"]
    for i in range(3):
        print(f"    {states[i]:>12} "
              f"{P[i][0]:>10.4f} "
              f"{P[i][1]:>12.4f} "
              f"{P[i][2]:>10.4f}")
              
    # Save for later use in the master pipeline
    np.save("dmc_transition_matrix.npy", P)
    print(f"\n[✓] Saved matrix to dmc_transition_matrix.npy")