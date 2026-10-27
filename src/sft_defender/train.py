"""
Training script for SFT Defender.
Supports both LoRA and full fine-tuning.
Uses next token prediction with cross-entropy loss on "harmful"/"benign" tokens.
OPTIMIZED: Vectorized compute_loss for massive speedup.
"""

import os
import sys
import json
import math
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import Dataset

# Import from src.sft_defender (absolute import)
try:
    from src.sft_defender.config import SFTTrainingConfig
    from src.sft_defender.data_preparation import SFTDataset
except ImportError:
    from .config import SFTTrainingConfig
    from .data_preparation import SFTDataset


def _log(msg: str):
    if int(os.environ.get("LOCAL_RANK", -1)) in (-1, 0):
        print(msg)

_SFT_DEBUG = os.environ.get("SFT_DEBUG", "0") == "1"


class SFTTrainer(Trainer):
    def __init__(self, harmful_token_id: int, benign_token_id: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.harmful_token_id = harmful_token_id
        self.benign_token_id = benign_token_id

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        target_ids = inputs.pop("target_ids", None)
        prompt_lengths = inputs.pop("prompt_lengths", None)
        inputs.pop("turn_ids", None)
        inputs.pop("total_turns", None)
        inputs.pop("source_types", None)

        outputs = model(**inputs)
        logits = outputs.logits
        batch_size = logits.shape[0]

        # --- VECTORIZED LOSS COMPUTATION ---
        batch_indices = torch.arange(batch_size, device=logits.device)
        # Clamp to avoid negative indices if prompt_length is 0
        pos_indices = torch.clamp(prompt_lengths - 1, min=0)
        
        # Gather logits at the specific positions: [batch_size, vocab_size]
        position_logits = logits[batch_indices, pos_indices, :]

        # Extract benign and harmful logits
        benign_logits = position_logits[:, self.benign_token_id]
        harmful_logits = position_logits[:, self.harmful_token_id]

        # Stack to [batch_size, 2] -> [benign, harmful]
        target_logits = torch.stack([benign_logits, harmful_logits], dim=1)

        # Filter out invalid samples (prompt_length == 0)
        valid_mask = prompt_lengths > 0
        if not valid_mask.any():
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return (loss, outputs) if return_outputs else loss

        valid_target_logits = target_logits[valid_mask]
        valid_target_ids = target_ids[valid_mask].to(device=valid_target_logits.device).long()

        # Compute cross entropy for all valid samples at once
        loss = F.cross_entropy(valid_target_logits, valid_target_ids)

        return (loss, outputs) if return_outputs else loss


class RewardShapingSFTTrainer(SFTTrainer):
    def __init__(
        self,
        harmful_token_id: int,
        benign_token_id: int,
        block_weight: float = 1.0,
        miss_block_weight: float = 3.0,
        early_block_weight: float = 0.3,
        gap_penalty_scale: float = 0.5,
        false_block_weight: float = 1.0,
        pass_weight: float = 0.3,
        balanced_mode: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(harmful_token_id, benign_token_id, *args, **kwargs)
        self.block_weight = block_weight
        self.miss_block_weight = miss_block_weight
        self.early_block_weight = early_block_weight
        self.gap_penalty_scale = gap_penalty_scale
        self.false_block_weight = false_block_weight
        self.pass_weight = pass_weight
        self.balanced_mode = balanced_mode

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        target_ids = inputs.pop("target_ids", None)
        prompt_lengths = inputs.pop("prompt_lengths", None)
        turn_ids = inputs.pop("turn_ids", None)
        total_turns = inputs.pop("total_turns", None)
        source_types = inputs.pop("source_types", None)

        outputs = model(**inputs)
        logits = outputs.logits
        batch_size = logits.shape[0]

        # --- VECTORIZED FORWARD PASS ---
        batch_indices = torch.arange(batch_size, device=logits.device)
        pos_indices = torch.clamp(prompt_lengths - 1, min=0)
        position_logits = logits[batch_indices, pos_indices, :]

        benign_logits = position_logits[:, self.benign_token_id]
        harmful_logits = position_logits[:, self.harmful_token_id]
        target_logits = torch.stack([benign_logits, harmful_logits], dim=1)

        valid_mask = prompt_lengths > 0
        valid_target_logits = target_logits[valid_mask]
        valid_target_ids = target_ids[valid_mask].to(device=valid_target_logits.device).long()

        # Base loss for all valid samples
        sample_losses = F.cross_entropy(valid_target_logits, valid_target_ids, reduction='none')

        # Predictions and true labels
        predicted = torch.argmax(valid_target_logits, dim=1)
        true_label = valid_target_ids

        # Filter metadata for valid samples
        valid_source_types = source_types[valid_mask] if source_types is not None else None
        valid_turn_ids = turn_ids[valid_mask] if turn_ids is not None else None
        valid_total_turns = total_turns[valid_mask] if total_turns is not None else None

        # --- VECTORIZED WEIGHT CALCULATION ---
        weights = torch.ones_like(sample_losses)
        device = sample_losses.device

        if self.balanced_mode:
            weights = torch.where(
                true_label == 1, 
                torch.tensor(self.block_weight, device=device),
                torch.tensor(self.pass_weight, device=device)
            )
        else:
            is_benign_conv = (valid_source_types == 1) if valid_source_types is not None else torch.zeros_like(true_label, dtype=torch.bool)

            # Benign conversation scenarios
            mask_correct_pass_benign = is_benign_conv & (predicted == 0)
            mask_false_positive = is_benign_conv & (predicted == 1)
            weights[mask_correct_pass_benign] = self.pass_weight
            weights[mask_false_positive] = self.false_block_weight

            # Harmful conversation scenarios
            mask_harmful_conv = ~is_benign_conv
            
            mask_correct_pass_harmful = mask_harmful_conv & (true_label == 0) & (predicted == 0)
            mask_accurate_block = mask_harmful_conv & (true_label == 1) & (predicted == 1)
            mask_miss_block = mask_harmful_conv & (true_label == 1) & (predicted == 0)
            mask_early_block = mask_harmful_conv & (true_label == 0) & (predicted == 1)

            weights[mask_correct_pass_harmful] = self.pass_weight
            weights[mask_accurate_block] = self.block_weight
            weights[mask_miss_block] = self.miss_block_weight

            # Early block weight calculation (vectorized)
            if mask_early_block.any():
                if valid_turn_ids is not None and valid_total_turns is not None:
                    raw_gap = (valid_total_turns[mask_early_block] - 1) - valid_turn_ids[mask_early_block]
                    gap = torch.clamp(raw_gap, min=0)
                else:
                    gap = torch.ones(mask_early_block.sum(), device=device)
                weights[mask_early_block] = self.early_block_weight * (1.0 + gap * self.gap_penalty_scale)

        weighted_loss = (sample_losses * weights).mean()
        return (weighted_loss, outputs) if return_outputs else weighted_loss


class SFTDatasetForTraining(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 2048):
        self.samples = []
        dropped = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    prompt = sample["prompt"]
                    completion = sample["completion"]

                    try:
                        messages = [{"role": "user", "content": prompt}]
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    except Exception:
                        formatted_prompt = prompt

                    full_text = formatted_prompt + completion
                    prompt_tokens = tokenizer(formatted_prompt, add_special_tokens=False)["input_ids"]
                    full_tokens = tokenizer(full_text, add_special_tokens=True)["input_ids"]

                    if len(full_tokens) <= max_length:
                        sample["_full_tokens"] = full_tokens
                        sample["_prompt_len"] = len(prompt_tokens)
                        self.samples.append(sample)
                    else:
                        dropped += 1

        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"[Dataset] Loaded {len(self.samples)} valid samples. Dropped {dropped} truncated samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        completion = sample["completion"].strip()
        full_tokens = sample["_full_tokens"]
        prompt_len = sample["_prompt_len"]

        label_id = 1 if completion == "0" else 0
        metadata = sample.get("metadata", {})

        return {
            "input_ids": full_tokens,
            "attention_mask": [1] * len(full_tokens),
            "labels": full_tokens,
            "prompt_lengths": prompt_len,
            "target_ids": label_id,
            "turn_ids": metadata.get("turn_id", 0),
            "total_turns": metadata.get("total_turns", 1),
            "source_types": 0 if metadata.get("source_type", "harmful") == "harmful" else 1,
        }


def sft_defender_data_collator(features):
    max_len = max(len(f["input_ids"]) for f in features)
    batch = {
        "input_ids": [], "attention_mask": [], "labels": [],
        "prompt_lengths": [], "target_ids": [], "turn_ids": [],
        "total_turns": [], "source_types": [],
    }

    for f in features:
        seq_len = len(f["input_ids"])
        pad_amount = max_len - seq_len
        batch["input_ids"].append(f["input_ids"] + [0] * pad_amount)
        batch["attention_mask"].append(f["attention_mask"] + [0] * pad_amount)
        batch["labels"].append(f["labels"] + [-100] * pad_amount)
        batch["prompt_lengths"].append(f["prompt_lengths"])
        batch["target_ids"].append(f["target_ids"])
        batch["turn_ids"].append(f["turn_ids"])
        batch["total_turns"].append(f["total_turns"])
        batch["source_types"].append(f["source_types"])

    for k in batch:
        if k in ["input_ids", "attention_mask", "labels"]:
            batch[k] = torch.tensor(batch[k], dtype=torch.long)
        else:
            batch[k] = torch.tensor(batch[k], dtype=torch.long)
    return batch


def load_model_and_tokenizer(config: SFTTrainingConfig):
    if config.resume and config.resume_model_path and config.training_type == "full":
        model_path = config.resume_model_path
        _log(f"[Resume] Loading model from checkpoint: {model_path}")
    else:
        model_path = config.base_model
        _log(f"Loading model and tokenizer from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=config.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if config.bf16 else torch.float16 if config.fp16 else torch.float32),
        attn_implementation="sdpa", # <--- CRITICAL SPEEDUP
        token=config.hf_token,
    )

    if config.gradient_checkpointing and not config.deepspeed:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def setup_lora(model, config: SFTTrainingConfig):
    if config.resume and config.resume_model_path:
        _log(f"[Resume] Loading LoRA adapter from {config.resume_model_path}...")
        model = PeftModel.from_pretrained(model, config.resume_model_path)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        model.train()
    else:
        _log("Setting up LoRA...")
        lora_config = LoraConfig(
            r=config.lora_r, lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules, lora_dropout=config.lora_dropout,
            bias=config.lora_bias, task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train_sft_defender(config: SFTTrainingConfig) -> str:
    torch.manual_seed(config.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = local_rank in (-1, 0)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process:
        with open(output_dir / "training_config.json", "w") as f:
            json.dump(config.__dict__, f, indent=2, default=str)

    model, tokenizer = load_model_and_tokenizer(config)

    if config.training_type == "lora":
        model = setup_lora(model, config)
    else:
        _log("Using full parameter fine-tuning...")
        _log(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    harmful_tokens = tokenizer.encode("0", add_special_tokens=False)
    benign_tokens = tokenizer.encode("1", add_special_tokens=False)
    if len(harmful_tokens) != 1 or len(benign_tokens) != 1:
        raise ValueError("Tokens must be single tokens.")

    harmful_token_id = harmful_tokens[0]
    benign_token_id = benign_tokens[0]
    _log(f"Token IDs - 0 (harmful): {harmful_token_id}, 1 (benign): {benign_token_id}")

    _log(f"Loading training data from {config.train_data_path}...")
    train_dataset = SFTDatasetForTraining(config.train_data_path, tokenizer, max_length=config.max_seq_length)
    _log(f"Training samples: {len(train_dataset)}")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps * world_size
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    eval_steps = config.eval_steps or max(1, steps_per_epoch // 10)
    _log(f"Steps per epoch: {steps_per_epoch}, eval_steps: {eval_steps}")

    wandb_enabled = config.wandb_project is not None
    if wandb_enabled and is_main_process:
        import wandb
        wandb.init(project=config.wandb_project, name=config.wandb_run_name, config={k: str(v) if isinstance(v, list) else v for k, v in config.__dict__.items()})
        wandb.define_metric("defense_step")
        for prefix in ["train/*", "val/*", "test/*", "init_train/*", "init_val/*", "init_test/*"]:
            wandb.define_metric(prefix, step_metric="defense_step")

    training_args_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": config.num_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_steps": int(config.warmup_ratio * steps_per_epoch * config.num_epochs),
        "max_grad_norm": config.max_grad_norm,
        "logging_steps": 10, # <--- REDUCED LOGGING FOR SPEED
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 3,
        "fp16": config.fp16,
        "bf16": config.bf16,
        "optim": "adamw_torch",
        "lr_scheduler_type": config.lr_scheduler_type,
        "report_to": "wandb" if wandb_enabled else "none",
        "seed": config.seed,
        "gradient_checkpointing": config.gradient_checkpointing,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": config.ddp_find_unused_parameters,
        "ddp_timeout": config.ddp_timeout,
    }

    if config.deepspeed: training_args_kwargs["deepspeed"] = config.deepspeed
    if config.fsdp: training_args_kwargs["fsdp"] = config.fsdp
    if config.fsdp_config: training_args_kwargs["fsdp_config"] = config.fsdp_config

    training_args = TrainingArguments(**training_args_kwargs)
    data_collator = sft_defender_data_collator

    if config.reward_shaping:
        _log(f"\nUsing Reward Shaping SFT Trainer:")
        trainer = RewardShapingSFTTrainer(
            harmful_token_id=harmful_token_id, benign_token_id=benign_token_id,
            block_weight=config.block_weight, miss_block_weight=config.miss_block_weight,
            early_block_weight=config.early_block_weight, gap_penalty_scale=config.gap_penalty_scale,
            false_block_weight=config.false_block_weight, pass_weight=config.pass_weight,
            balanced_mode=config.balanced_mode,
            model=model, args=training_args, train_dataset=train_dataset,
            data_collator=sft_defender_data_collator, processing_class=tokenizer,
        )
    else:
        trainer = SFTTrainer(
            harmful_token_id=harmful_token_id, benign_token_id=benign_token_id,
            model=model, args=training_args, train_dataset=train_dataset,
            data_collator=data_collator, processing_class=tokenizer,
        )

    defense_callback = None
    if config.eval_data_dir:
        from src.sft_defender.eval_callback import DefenseEvalCallback
        num_gpus = torch.cuda.device_count()
        eval_gpu_index = config.eval_gpu if config.eval_gpu is not None else (num_gpus - 1)
        defense_callback = DefenseEvalCallback(
            trainer=trainer, config=config, eval_steps=eval_steps, steps_per_epoch=steps_per_epoch,
            eval_gpu_index=eval_gpu_index, eval_data_dir=config.eval_data_dir, output_dir=str(output_dir),
        )
        trainer.add_callback(defense_callback)

    if config.resume_init_eval and defense_callback is not None:
        defense_callback.run_init_eval(config.resume_model_path, global_step=0)

    _log("\n" + "=" * 70 + "\nStarting training...\n" + "=" * 70 + "\n")

    if config.resume and config.resume_model_path:
        trainer.train(resume_from_checkpoint=config.resume_model_path)
    else:
        trainer.train()

    _log("\nSaving final model...")
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))

    if wandb_enabled and is_main_process:
        import wandb
        wandb.finish()

    _log(f"\nTraining completed! Model saved to: {final_model_path}")
    return str(final_model_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SFT Defender")
    parser.add_argument("--training-type", type=str, choices=["lora", "full"], default="lora")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default="checkpoints/sft_defender")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--fsdp", type=str, default=None)
    parser.add_argument("--fsdp-config", type=str, default=None)
    parser.add_argument("--ddp-timeout", type=int, default=7200)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-model-path", type=str, default=None)
    parser.add_argument("--resume-init-eval", action="store_true")
    parser.add_argument("--balanced-mode", action="store_true", default=False)
    parser.add_argument("--reward-shaping", action="store_true")
    parser.add_argument("--block-weight", type=float, default=1.0)
    parser.add_argument("--miss-block-weight", type=float, default=3.0)
    parser.add_argument("--early-block-weight", type=float, default=0.3)
    parser.add_argument("--gap-penalty-scale", type=float, default=0.5)
    parser.add_argument("--false-block-weight", type=float, default=1.0)
    parser.add_argument("--pass-weight", type=float, default=0.3)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--eval-data-dir", type=str, default=None)
    parser.add_argument("--eval-gpu", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--defense-batch-size", type=int, default=500)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)

    args = parser.parse_args()

    config = SFTTrainingConfig(
        training_type=args.training_type, base_model=args.base_model,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        output_dir=args.output_dir, train_data_path=args.train_data, val_data_path=args.val_data,
        num_epochs=args.num_epochs, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length, seed=args.seed, hf_token=args.hf_token,
        deepspeed=args.deepspeed, fsdp=args.fsdp, fsdp_config=args.fsdp_config,
        ddp_timeout=args.ddp_timeout, resume=args.resume, resume_model_path=args.resume_model_path,
        resume_init_eval=args.resume_init_eval, reward_shaping=args.reward_shaping,
        balanced_mode=args.balanced_mode, block_weight=args.block_weight,
        miss_block_weight=args.miss_block_weight, early_block_weight=args.early_block_weight,
        gap_penalty_scale=args.gap_penalty_scale, false_block_weight=args.false_block_weight,
        pass_weight=args.pass_weight, wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name, eval_data_dir=args.eval_data_dir,
        eval_gpu=args.eval_gpu, eval_steps=args.eval_steps,
        defense_batch_size=args.defense_batch_size, gpu_memory_utilization=args.gpu_memory_utilization,
    )

    train_sft_defender(config)