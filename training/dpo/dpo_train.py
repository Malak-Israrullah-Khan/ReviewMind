"""
ReviewMind – DPO Training Script
Loads data/dpo_pairs.jsonl, initialises the SFT model (Malak-Israr/reviewmind-lora)
as the reference policy, wraps a new LoRA adapter as the trainable policy, and
runs Direct Preference Optimisation via TRL's DPOTrainer.

After training the DPO adapter is saved to
    training/dpo/checkpoints/final_dpo_adapter
and optionally pushed to HuggingFace Hub as Malak-Israr/reviewmind-dpo.

Usage:
    python training/dpo/dpo_train.py
    python training/dpo/dpo_train.py --dry-run
    python training/dpo/dpo_train.py --no-push-hub
    python training/dpo/dpo_train.py --hf-token TOKEN
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# dpo_train.py lives at <project_root>/training/dpo/dpo_train.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

BASE_MODEL        = "meta-llama/Llama-3.1-8B-Instruct"
SFT_ADAPTER_REPO  = "Malak-Israr/reviewmind-lora"
DPO_HUB_REPO      = "Malak-Israr/reviewmind-dpo"

DPO_PAIRS_FILE    = PROJECT_ROOT / "data" / "dpo_pairs.jsonl"
OUTPUT_DIR        = PROJECT_ROOT / "training" / "dpo" / "checkpoints"
FINAL_ADAPTER_DIR = OUTPUT_DIR / "final_dpo_adapter"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

BETA                      = 0.1    # KL penalty — controls deviation from reference
LEARNING_RATE             = 5e-5
BATCH_SIZE                = 1
GRADIENT_ACCUMULATION     = 8      # effective batch = 8
EPOCHS                    = 1
MAX_LENGTH                = 1024   # total prompt+response length
MAX_PROMPT_LENGTH         = 512    # max tokens in the prompt portion
LORA_RANK                 = 16
LORA_ALPHA                = 32
LORA_DROPOUT              = 0.05
WARMUP_RATIO              = 0.03


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DPO training for ReviewMind",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load all components and print config without running training.",
    )
    parser.add_argument(
        "--hf-token", default=None, metavar="TOKEN",
        help="HuggingFace Hub write token. Falls back to $HF_TOKEN env var.",
    )
    parser.add_argument(
        "--hf-repo", default=DPO_HUB_REPO, metavar="USERNAME/REPO",
        help="HuggingFace Hub repo to push the DPO adapter to.",
    )
    parser.add_argument(
        "--no-push-hub", action="store_true",
        help="Skip pushing the adapter to HuggingFace Hub after training.",
    )
    parser.add_argument(
        "--pairs-file", default=str(DPO_PAIRS_FILE),
        help="Path to dpo_pairs.jsonl produced by data/dpo_dataset.py.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------

def require_gpu() -> None:
    if not torch.cuda.is_available():
        sys.exit(
            "\nNo CUDA GPU detected.\n"
            "DPO training requires a CUDA-capable GPU.\n"
            "On Colab: Runtime → Change runtime type → GPU."
        )
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name}  ({mem:.1f} GB)")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dpo_dataset(pairs_file: Path):
    """
    Load dpo_pairs.jsonl into a HuggingFace Dataset.
    Expected fields per line: prompt, chosen, rejected.
    """
    from datasets import load_dataset as hf_load

    if not pairs_file.exists():
        sys.exit(
            f"\nDPO pairs file not found: {pairs_file}\n"
            "Run  python data/dpo_dataset.py  first."
        )

    dataset = hf_load("json", data_files=str(pairs_file), split="train")
    print(f"  Loaded {len(dataset):,} preference pairs from {pairs_file}")

    # Validate required fields
    for field in ("prompt", "chosen", "rejected"):
        if field not in dataset.column_names:
            sys.exit(f"Missing required field '{field}' in {pairs_file}")

    # 90/10 train/eval split so we can track eval loss during training
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  Train: {len(split['train']):,}  Eval: {len(split['test']):,}")
    return split["train"], split["test"]


# ---------------------------------------------------------------------------
# Model + tokenizer loading
# ---------------------------------------------------------------------------

def load_base_model_and_tokenizer(base_model: str):
    """Load the base LLM in 4-bit NF4 with the SFT LoRA adapter already merged."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\nLoading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Left-pad so the loss is computed on the response tokens, not the prompt
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"Loading base model (4-bit): {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=False,
    )

    print(f"Attaching SFT adapter: {SFT_ADAPTER_REPO}")
    model = PeftModel.from_pretrained(base, SFT_ADAPTER_REPO)

    # prepare_model_for_kbit_training casts LayerNorms to fp32 and enables
    # gradient checkpointing — required before adding a second LoRA adapter
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


# ---------------------------------------------------------------------------
# DPO LoRA adapter
# ---------------------------------------------------------------------------

def add_dpo_lora(model):
    """
    Wrap the SFT-initialised model with a fresh LoRA adapter for DPO training.
    The SFT adapter weights remain frozen (they define the reference policy);
    only the new DPO LoRA weights are updated during training.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    return model


def print_param_summary(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct       = 100 * trainable / total if total else 0.0
    print(f"\n  Trainable params : {trainable:>15,}")
    print(f"  Frozen params    : {total - trainable:>15,}")
    print(f"  Total params     : {total:>15,}")
    print(f"  Trainable %      : {pct:>14.4f}%\n")


# ---------------------------------------------------------------------------
# DPO trainer setup
# ---------------------------------------------------------------------------

def build_dpo_trainer(model, tokenizer, train_dataset, eval_dataset):
    from transformers import TrainingArguments
    from trl import DPOTrainer

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eff_batch    = BATCH_SIZE * GRADIENT_ACCUMULATION
    total_steps  = (len(train_dataset) // eff_batch) * EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,

        fp16=True,
        bf16=False,
        optim="adamw_torch",
        fp16_full_eval=False,
        ddp_find_unused_parameters=False,
        skip_memory_metrics=True,
        torch_compile=False,

        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to="none",
        run_name="reviewmind-dpo",
        dataloader_num_workers=0,
        remove_unused_columns=False,  # DPOTrainer needs prompt/chosen/rejected columns
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,       # None → DPOTrainer uses the frozen base layers as reference
        args=training_args,
        beta=BETA,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
    )

    return trainer


# ---------------------------------------------------------------------------
# HuggingFace Hub push
# ---------------------------------------------------------------------------

def push_adapter_to_hub(adapter_dir: Path, repo_id: str, hf_token: str) -> None:
    from transformers import AutoTokenizer

    print(f"\nPushing DPO adapter → https://huggingface.co/{repo_id} …")
    # Push adapter config + weights only (not the merged model — saves bandwidth)
    from peft import AutoPeftModelForCausalLM
    peft_model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_dir),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    peft_model.push_to_hub(repo_id, token=hf_token, private=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.push_to_hub(repo_id, token=hf_token)

    print(f"  Adapter available at https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    args = parse_args()
    pairs_file = Path(args.pairs_file)

    print("\n" + "=" * 60)
    print("  ReviewMind – DPO Training")
    print("  Dry-run mode" if args.dry_run else "  TRAINING MODE")
    print("=" * 60)
    print(f"  Base model      : {BASE_MODEL}")
    print(f"  SFT adapter     : {SFT_ADAPTER_REPO}")
    print(f"  DPO hub repo    : {args.hf_repo}")
    print(f"  Pairs file      : {pairs_file}")
    print(f"  Beta            : {BETA}")
    print(f"  LR              : {LEARNING_RATE}")
    print(f"  Batch size      : {BATCH_SIZE}  (grad accum {GRADIENT_ACCUMULATION} → eff {BATCH_SIZE * GRADIENT_ACCUMULATION})")
    print(f"  Epochs          : {EPOCHS}")
    print(f"  Output dir      : {OUTPUT_DIR}")

    # ── GPU ──────────────────────────────────────────────
    print("\n[1/5] Checking GPU …")
    require_gpu()

    # ── Dataset ──────────────────────────────────────────
    print("\n[2/5] Loading DPO pairs …")
    train_dataset, eval_dataset = load_dpo_dataset(pairs_file)

    # ── Model ───────────────────────────────────────────
    print("\n[3/5] Loading model + SFT adapter …")
    model, tokenizer = load_base_model_and_tokenizer(BASE_MODEL)

    # ── DPO LoRA ────────────────────────────────────────
    print("\n[4/5] Wrapping with DPO LoRA adapter …")
    model = add_dpo_lora(model)
    print_param_summary(model)

    # ── Trainer ─────────────────────────────────────────
    print("[5/5] Configuring DPOTrainer …")
    trainer = build_dpo_trainer(model, tokenizer, train_dataset, eval_dataset)

    eff_batch   = BATCH_SIZE * GRADIENT_ACCUMULATION
    total_steps = (len(train_dataset) // eff_batch) * EPOCHS
    print(f"  total_steps ≈ {total_steps}  warmup ≈ {max(1, int(total_steps * WARMUP_RATIO))}")

    if args.dry_run:
        print("\n" + "=" * 60)
        print("  DRY-RUN COMPLETE — all components loaded successfully.")
        print("  Re-run without --dry-run to start DPO training.")
        print("=" * 60)
        return

    # ── Train ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Starting DPO training …")
    print("=" * 60 + "\n")

    trainer.train()

    # Save final DPO adapter
    FINAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(FINAL_ADAPTER_DIR))
    tokenizer.save_pretrained(str(FINAL_ADAPTER_DIR))
    print(f"\nFinal DPO adapter saved → {FINAL_ADAPTER_DIR}")

    # ── Hub push ────────────────────────────────────────
    if args.no_push_hub:
        print("Skipping Hub push (--no-push-hub).")
        return

    hf_token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not hf_token:
        print(
            "\nNo HuggingFace token found — skipping push.\n"
            "  Set $HF_TOKEN or pass --hf-token TOKEN."
        )
        return

    push_adapter_to_hub(FINAL_ADAPTER_DIR, args.hf_repo, hf_token)


if __name__ == "__main__":
    main()
