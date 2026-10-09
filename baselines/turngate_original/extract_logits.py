#!/usr/bin/env python3
"""
Extract per-sample logits from a trained SFT defender checkpoint.
Outputs a JSONL file with logits for tokens "0" (harmful) and "1" (benign)
at the position where the model makes its PASS/BLOCK decision.
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def extract_logits(
    checkpoint_path: str,
    data_path: str,
    output_path: str,
    max_seq_length: int = 2048,
    batch_size: int = 32,
):
    """
    Extract logits for "0" and "1" tokens at the decision position.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        data_path: Path to the JSONL data file (validation or test set)
        output_path: Path to save the extracted logits (JSONL format)
        max_seq_length: Maximum sequence length (must match training)
        batch_size: Batch size for inference
    """
    
    # Load model and tokenizer
    print(f"Loading model from {checkpoint_path}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    # Get token IDs for "0" (harmful) and "1" (benign)
    harmful_token_id = tokenizer.encode("0", add_special_tokens=False)[0]
    benign_token_id = tokenizer.encode("1", add_special_tokens=False)[0]
    print(f"Token IDs - 0 (harmful): {harmful_token_id}, 1 (benign): {benign_token_id}")
    
    # Load data
    print(f"Loading data from {data_path}...")
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples")
    
    # Extract logits
    print("Extracting logits...")
    results = []
    
    for i in tqdm(range(0, len(samples), batch_size)):
        batch_samples = samples[i : i + batch_size]
        batch_logits = []
        batch_prompt_lens = []
        batch_labels = []
        
        # Tokenize batch
        for sample in batch_samples:
            prompt = sample["prompt"]
            completion = sample["completion"].strip()
            
            # Apply chat template
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted_prompt = prompt
            
            full_text = formatted_prompt + completion
            
            # Tokenize with special tokens
            prompt_tokens = tokenizer(formatted_prompt, add_special_tokens=True)["input_ids"]
            full_tokens = tokenizer(full_text, add_special_tokens=True)["input_ids"]
            
            # Skip if exceeds max length
            if len(full_tokens) > max_seq_length:
                continue
            
            prompt_len = len(prompt_tokens)
            label_id = 0 if completion == "0" else 1  # 0=harmful, 1=benign            
            batch_logits.append(full_tokens)
            batch_prompt_lens.append(prompt_len)
            batch_labels.append(label_id)
        
        if not batch_logits:
            continue
        
        # Pad batch
        max_len = max(len(seq) for seq in batch_logits)
        pad_id = tokenizer.pad_token_id or 0
        
        input_ids = []
        attention_mask = []
        
        for seq in batch_logits:
            pad_amount = max_len - len(seq)
            input_ids.append(seq + [pad_id] * pad_amount)
            attention_mask.append([1] * len(seq) + [0] * pad_amount)
        
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(model.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Extract logits at decision position for each sample
        for j, (prompt_len, label) in enumerate(zip(batch_prompt_lens, batch_labels)):
            if prompt_len == 0:
                continue
            
            # Get logits at position prompt_len - 1 (predicting first completion token)
            position_logits = logits[j, prompt_len - 1, :]
            
            # Extract logits for "0" and "1"
            logit_0 = position_logits[harmful_token_id].item()
            logit_1 = position_logits[benign_token_id].item()
            
            # Compute softmax probabilities
            target_logits = torch.tensor([logit_1, logit_0])  # [benign, harmful]
            probs = torch.softmax(target_logits, dim=0)
            softmax_0 = probs[1].item()  # P(harmful)
            softmax_1 = probs[0].item()  # P(benign)
            
            results.append({
                "sample_id": i + j,
                "true_label": label,  # 0=harmful, 1=benign
                "logit_0": logit_0,
                "logit_1": logit_1,
                "softmax_0": softmax_0,
                "softmax_1": softmax_1,
            })
    
    # Save results
    print(f"Saving {len(results)} results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"Done! Extracted logits for {len(results)} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract logits from trained SFT defender")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to JSONL data file (validation or test set)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save extracted logits (JSONL)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    
    args = parser.parse_args()
    
    extract_logits(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_path=args.output,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
    )