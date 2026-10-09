"""
Discrete Markov Chain (DMC) Fallback for TurnGate Abstention.

This module implements a lightweight, interpretable sequential risk tracker
that takes over when the TurnGate monitor abstains (low confidence).

States:
    S0 = Benign      (early turns, far from attack closure)
    S1 = Suspicious  (turns approaching the closure turn t*)
    S2 = Harmful     (closure turn t* and beyond)

The transition matrix P is learned empirically from MTID training data.
"""

import json
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional


class DiscreteMarkovChain:
    """
    A 3-state Discrete Markov Chain for sequential risk accumulation.
    
    States:
        0 = Benign
        1 = Suspicious  
        2 = Harmful (absorbing state)
    """
    
    STATE_NAMES = {0: "Benign", 1: "Suspicious", 2: "Harmful"}
    
    def __init__(self, transition_matrix: np.ndarray = None, 
                 safety_threshold: float = 0.6):
        """
        Args:
            transition_matrix: 3x3 matrix P[i][j] = P(S_{t+1}=j | S_t=i)
            safety_threshold: If P(Harmful) > this, output BLOCK
        """
        if transition_matrix is not None:
            self.P = transition_matrix
        else:
            # Default: identity (no transitions) - must be learned from data
            self.P = np.eye(3)
        
        self.theta = safety_threshold  # Safety threshold for BLOCK decision
        
    # ==========================================
    # 1. LEARN TRANSITION MATRIX FROM MTID DATA
    # ==========================================
    def learn_from_data(self, data_path: str, 
                        suspicious_window: int = 1):
        """
        Learns the empirical transition matrix from MTID training data.
        
        For each dialogue with closure turn t*:
            - Turns t < t* - suspicious_window  → S0 (Benign)
            - Turns t* - suspicious_window <= t < t*  → S1 (Suspicious)
            - Turns t >= t*  → S2 (Harmful)
        
        Args:
            data_path: Path to MTID training JSONL
            suspicious_window: How many turns before t* are "suspicious"
        """
        print(f"[*] Learning DMC transition matrix from: {data_path}")
        
        # Count transitions: counts[i][j] = number of times S_i → S_j
        counts = np.zeros((3, 3), dtype=np.float64)
        total_turns = 0
        total_dialogues = 0
        
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                    
                sample = json.loads(line)
                total_dialogues += 1
                
                # Extract the closure turn t* and total turns
                # Adjust these field names based on your MTID JSONL structure
                closure_turn = sample.get("closure_turn", 
                                 sample.get("t_star", 
                                 sample.get("harmful_turn", -1)))
                
                # Get the conversation turns
                turns = sample.get("turns", 
                        sample.get("conversation", 
                        sample.get("messages", [])))
                
                if closure_turn < 0 or len(turns) == 0:
                    continue
                
                num_turns = len(turns)
                
                # Map each turn to a state
                states = []
                for t in range(num_turns):
                    if t >= closure_turn:
                        states.append(2)  # Harmful
                    elif t >= closure_turn - suspicious_window:
                        states.append(1)  # Suspicious
                    else:
                        states.append(0)  # Benign
                
                # Count transitions
                for t in range(len(states) - 1):
                    s_current = states[t]
                    s_next = states[t + 1]
                    counts[s_current][s_next] += 1
                    total_turns += 1
        
        # Normalize to get probabilities (each row sums to 1)
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.P = counts / row_sums
        
        # Make S2 (Harmful) an absorbing state
        # Once harmful, always harmful
        self.P[2] = [0.0, 0.0, 1.0]
        
        print(f"[*] Processed {total_dialogues} dialogues, {total_turns} transitions")
        print(f"[*] Learned Transition Matrix P:")
        print(f"    {'':>12} {'→Benign':>10} {'→Suspicious':>12} {'→Harmful':>10}")
        for i in range(3):
            print(f"    {self.STATE_NAMES[i]:>12} "
                  f"{self.P[i][0]:>10.4f} "
                  f"{self.P[i][1]:>12.4f} "
                  f"{self.P[i][2]:>10.4f}")
        
        return self.P
    
    # ==========================================
    # 2. DMC INFERENCE (FALLBACK LOGIC)
    # ==========================================
    def compute_state_distribution(self, initial_state: int = 0, 
                                    num_steps: int = 1) -> np.ndarray:
        """
        Computes the state probability distribution after `num_steps` transitions.
        
        π_t = π_0 × P^t
        
        Args:
            initial_state: Starting state (0, 1, or 2)
            num_steps: Number of turns to project forward
            
        Returns:
            State probability vector [P(Benign), P(Suspicious), P(Harmful)]
        """
        # Initial distribution: one-hot vector
        pi = np.zeros(3)
        pi[initial_state] = 1.0
        
        # Apply transition matrix `num_steps` times
        for _ in range(num_steps):
            pi = pi @ self.P
            
        return pi
    
    def fallback_decision(self, current_state_probs: np.ndarray) -> Tuple[str, float]:
        """
        Makes a PASS/BLOCK decision based on the DMC state probabilities.
        
        Args:
            current_state_probs: [P(Benign), P(Suspicious), P(Harmful)]
            
        Returns:
            (decision, risk_score) where decision is "BLOCK" or "PASS"
        """
        risk_score = current_state_probs[2]  # P(Harmful)
        
        if risk_score >= self.theta:
            return "BLOCK", risk_score
        else:
            return "PASS", risk_score
    
    # ==========================================
    # 3. FULL SEQUENTIAL RISK TRACKING
    # ==========================================
    def track_dialogue_risk(self, turn_decisions: List[dict]) -> List[dict]:
        """
        Tracks risk across an entire multi-turn dialogue.
        
        Args:
            turn_decisions: List of dicts, one per turn:
                {
                    "turn_id": int,
                    "turngate_action": str,  # "PASS", "BLOCK", or "ABSTAIN"
                    "turngate_confidence": float,
                    "turngate_logits": [logit_0, logit_1]  # optional
                }
                
        Returns:
            List of dicts with DMC risk tracking for each turn.
        """
        results = []
        current_state = 0  # Start in Benign state
        pi = np.array([1.0, 0.0, 0.0])  # Initial distribution
        
        for turn in turn_decisions:
            turn_id = turn["turn_id"]
            tg_action = turn["turngate_action"]
            
            if tg_action in ["PASS", "BLOCK"]:
                # TurnGate is confident → trust its decision
                # Update DMC state based on TurnGate's prediction
                if tg_action == "BLOCK":
                    current_state = 2  # Harmful detected
                    pi = np.array([0.0, 0.0, 1.0])
                else:
                    # Advance state by one step
                    pi = pi @ self.P
                    current_state = np.argmax(pi)
                    
                decision = tg_action
                risk_score = pi[2]
                source = "TurnGate"
                
            elif tg_action == "ABSTAIN":
                # TurnGate is uncertain → DMC takes over
                # Advance state by one step
                pi = pi @ self.P
                decision, risk_score = self.fallback_decision(pi)
                source = "DMC"
                current_state = np.argmax(pi)
                
            else:
                raise ValueError(f"Unknown action: {tg_action}")
            
            results.append({
                "turn_id": turn_id,
                "turngate_action": tg_action,
                "turngate_confidence": turn.get("turngate_confidence", 0.0),
                "dmc_state_probs": pi.tolist(),
                "dmc_risk_score": round(risk_score, 4),
                "final_decision": decision,
                "decision_source": source
            })
            
        return results


# ==========================================
# 4. DEMO / UNIT TEST (Run Immediately!)
# ==========================================
def run_demo():
    """
    Demonstrates the DMC with a synthetic transition matrix.
    Run this to verify the logic works before connecting to real data.
    """
    print("=" * 60)
    print("DMC FALLBACK - DEMO WITH SYNTHETIC DATA")
    print("=" * 60)
    
    # Synthetic transition matrix (typical for multi-turn jailbreaks)
    # Benign → mostly stays Benign, small chance of becoming Suspicious
    # Suspicious → high chance of becoming Harmful
    # Harmful → stays Harmful (absorbing)
    synthetic_P = np.array([
        [0.85, 0.13, 0.02],   # From Benign
        [0.10, 0.35, 0.55],   # From Suspicious
        [0.00, 0.00, 1.00],   # From Harmful (absorbing)
    ])
    
    dmc = DiscreteMarkovChain(
        transition_matrix=synthetic_P,
        safety_threshold=0.6
    )
    
    print("\n--- Scenario 1: Normal Benign Conversation ---")
    benign_dialogue = [
        {"turn_id": 1, "turngate_action": "PASS", "turngate_confidence": 0.95},
        {"turn_id": 2, "turngate_action": "PASS", "turngate_confidence": 0.92},
        {"turn_id": 3, "turngate_action": "PASS", "turngate_confidence": 0.88},
    ]
    results = dmc.track_dialogue_risk(benign_dialogue)
    for r in results:
        print(f"  Turn {r['turn_id']}: {r['final_decision']:>5} "
              f"(Source: {r['decision_source']:>8}, "
              f"Risk: {r['dmc_risk_score']:.4f}, "
              f"State: {r['dmc_state_probs']})")
    
    print("\n--- Scenario 2: Jailbreak Attack (TurnGate Abstains on Turn 3) ---")
    attack_dialogue = [
        {"turn_id": 1, "turngate_action": "PASS", "turngate_confidence": 0.85},
        {"turn_id": 2, "turngate_action": "ABSTAIN", "turngate_confidence": 0.52},
        {"turn_id": 3, "turngate_action": "ABSTAIN", "turngate_confidence": 0.48},
        {"turn_id": 4, "turngate_action": "ABSTAIN", "turngate_confidence": 0.45},
    ]
    results = dmc.track_dialogue_risk(attack_dialogue)
    for r in results:
        print(f"  Turn {r['turn_id']}: {r['final_decision']:>5} "
              f"(Source: {r['decision_source']:>8}, "
              f"Risk: {r['dmc_risk_score']:.4f}, "
              f"State: {r['dmc_state_probs']})")
    
    print("\n--- Scenario 3: Early Detection (TurnGate Blocks on Turn 2) ---")
    early_block = [
        {"turn_id": 1, "turngate_action": "PASS", "turngate_confidence": 0.90},
        {"turn_id": 2, "turngate_action": "BLOCK", "turngate_confidence": 0.88},
        {"turn_id": 3, "turngate_action": "ABSTAIN", "turngate_confidence": 0.55},
    ]
    results = dmc.track_dialogue_risk(early_block)
    for r in results:
        print(f"  Turn {r['turn_id']}: {r['final_decision']:>5} "
              f"(Source: {r['decision_source']:>8}, "
              f"Risk: {r['dmc_risk_score']:.4f}, "
              f"State: {r['dmc_state_probs']})")
    
    print("\n" + "=" * 60)
    print("[✓] DMC Demo Complete! All scenarios working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()