"""
ReviewMind – Fine-Tuning Script (Day 4)
QLoRA supervised fine-tuning of Llama-3.1-8B-Instruct on the prepared
CodeReviewer instruction dataset.

Usage:
    python training/finetune/train.py                  # full training run
    python training/finetune/train.py --dry-run        # validate setup, no training
    python training/finetune/train.py --config path/to/other.yaml
    python training/finetune/train.py --no-push-hub    # skip HF Hub upload
    python training/finetune/train.py --hf-repo username/my-model
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Path resolution — all relative paths in the config are anchored here
# ---------------------------------------------------------------------------
# train.py lives at  <project_root>/training/finetune/train.py
# so parent.parent.parent == project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Colab Google Drive checkpoint directory
GDRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/ReviewMind/checkpoints/"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for ReviewMind",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "training_config.yaml"),
        help="Path to the YAML training config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load model, tokenizer, dataset, and trainer — "
            "print the setup summary but do not call trainer.train()."
        ),
    )
    parser.add_argument(
        "--hf-repo",
        default=None,
        metavar="USERNAME/REPO",
        help="HuggingFace Hub repo ID. Overrides config hub.repo_id.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        metavar="TOKEN",
        help="HuggingFace Hub write token. Falls back to $HF_TOKEN env var.",
    )
    parser.add_argument(
        "--no-push-hub",
        action="store_true",
        help="Skip pushing the merged model to HuggingFace Hub after training.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        sys.exit(f"Config file not found: {path}")
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    print(f"Config loaded from {path}")
    return cfg


def resolve_path(raw: str) -> Path:
    """Return an absolute path; relative paths are resolved from PROJECT_ROOT."""
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# Google Drive mounting (Colab only)
# ---------------------------------------------------------------------------

def mount_google_drive() -> Path | None:
    """
    Check whether Google Drive is already mounted at /content/drive/MyDrive
    and return the checkpoint directory path if so.  Drive must be mounted
    manually by the user before running the script.  Returns None if the
    mount point does not exist, falling back to local checkpoint saving.
    """
    gdrive_root = Path("/content/drive/MyDrive")
    if gdrive_root.exists():
        gdrive_dir = Path(GDRIVE_CHECKPOINT_DIR)
        gdrive_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Google Drive detected. Checkpoints → {gdrive_dir}")
        return gdrive_dir
    print("  /content/drive/MyDrive not found — saving checkpoints locally.")
    return None


# ---------------------------------------------------------------------------
# GPU check — 4-bit quantization requires CUDA
# ---------------------------------------------------------------------------

def require_gpu() -> None:
    import torch
    if not torch.cuda.is_available():
        sys.exit(
            "\nNo CUDA GPU detected.\n"
            "QLoRA 4-bit training requires a CUDA-capable GPU.\n"
            "  • On a local machine: verify driver + CUDA toolkit installation.\n"
            "  • On cloud (Colab/Kaggle): switch to a GPU runtime.\n"
            "  • To inspect your setup: run  python -c 'import torch; print(torch.cuda.get_device_name(0))'\n"
        )
    import torch
    n = torch.cuda.device_count()
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name}  ({mem:.1f} GB)")


# ---------------------------------------------------------------------------
# BitsAndBytes 4-bit quantization config
# ---------------------------------------------------------------------------

def build_bnb_config(qcfg: dict):
    """
    QLoRA loads the base model in 4-bit (NF4) precision so it fits in GPU
    memory (~5 GB for 8B params), while the LoRA adapter weights stay in fp16.
    Double quantization squeezes an extra ~0.4 bits/param.
    """
    import torch
    from transformers import BitsAndBytesConfig

    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    compute_dtype_str = qcfg.get("bnb_4bit_compute_dtype", "float16")
    compute_dtype = dtype_map.get(compute_dtype_str)
    if compute_dtype is None:
        sys.exit(f"Unknown bnb_4bit_compute_dtype: '{compute_dtype_str}'")

    return BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------

def build_lora_config(lcfg: dict):
    """
    LoRA (Low-Rank Adaptation) injects small trainable rank-decomposition
    matrices into the attention projections.  With rank=16 and alpha=32 the
    adapter adds ~8M trainable parameters on top of the 8B frozen base —
    roughly 0.1% of total params.

    target_modules: we target all four attention projections (q/k/v/o) for
    richer adaptation compared to the common q+v-only baseline.
    """
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=lcfg["lora_rank"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        target_modules=list(lcfg["target_modules"]),
        bias="none",        # do not adapt bias terms — keeps adapter size small
        task_type=TaskType.CAUSAL_LM,
    )


# ---------------------------------------------------------------------------
# Model + tokenizer loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, bnb_config):
    """
    Load the base model in 4-bit and attach a matching tokenizer.

    device_map="auto" lets accelerate spread layers across available GPUs
    (or CPU offload if VRAM is tight) without manual device placement.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=False,
    )
    # Llama models ship without a pad token; we reuse eos so the tokenizer
    # can batch sequences without crashing.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model:     {model_name}  (4-bit quantized)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
        torch_dtype=torch.float16,  # dtype for any non-quantized layers
    )

    # prepare_model_for_kbit_training does three things:
    #   1. casts LayerNorm weights to fp32 for stable gradient flow
    #   2. makes the embed/lm_head output fp32
    #   3. enables gradient checkpointing to trade compute for VRAM
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Apply LoRA adapter
# ---------------------------------------------------------------------------

def apply_lora(model, lora_config):
    """Wrap the base model with the LoRA adapter using PEFT."""
    from peft import get_peft_model
    model = get_peft_model(model, lora_config)
    return model


# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------

def print_param_summary(model) -> None:
    """
    Print how many parameters are actually trainable vs frozen.
    After QLoRA setup this should be ~0.1% of total params — confirming
    that LoRA is wired correctly and the base model is frozen.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct       = 100 * trainable / total if total else 0.0

    print("\n" + "─" * 55)
    print("  MODEL PARAMETER SUMMARY")
    print("─" * 55)
    print(f"  Trainable params : {trainable:>15,}")
    print(f"  Frozen params    : {total - trainable:>15,}")
    print(f"  Total params     : {total:>15,}")
    print(f"  Trainable %      : {pct:>14.4f}%")
    print("─" * 55)

    if pct > 5.0:
        print("  WARNING: trainable% seems high — check LoRA is applied correctly.")
    elif pct == 0.0:
        print("  WARNING: no trainable parameters found — LoRA may not be attached.")
    else:
        print("  ✓ LoRA adapter is correctly applied (base model frozen).")
    print()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_splits(dcfg: dict):
    """
    Load the pre-processed JSONL splits produced by data/prepare_dataset.py.
    We load from local files rather than the Hub so training works offline.
    """
    from datasets import load_dataset as hf_load

    splits_dir = resolve_path(dcfg["splits_dir"])
    train_path = splits_dir / dcfg["train_file"]
    val_path   = splits_dir / dcfg["validation_file"]

    for p in (train_path, val_path):
        if not p.exists():
            sys.exit(
                f"\nData file not found: {p}\n"
                "Run  python data/prepare_dataset.py  first to generate the splits."
            )

    print(f"Loading train split      : {train_path}")
    print(f"Loading validation split : {val_path}")

    dataset = hf_load(
        "json",
        data_files={
            "train":      str(train_path),
            "validation": str(val_path),
        },
    )
    print(f"  train      : {len(dataset['train']):,} examples")
    print(f"  validation : {len(dataset['validation']):,} examples")
    return dataset


# ---------------------------------------------------------------------------
# Trainer setup
# ---------------------------------------------------------------------------

def build_trainer(model, tokenizer, dataset, tcfg: dict, dcfg: dict, dry_run: bool):
    """
    SFTTrainer (from TRL) wraps HuggingFace Trainer with conveniences for
    instruction fine-tuning: it integrates cleanly with PEFT adapters.

    Key scheduler choices:
      • cosine LR decay: smoother than linear; prevents the LR dropping to
        near-zero too early in training.
      • warmup_steps: 3% of total training steps ramp from 0→peak LR to
        stabilise early gradient updates before the model has adjusted.
        We compute this from the ratio so it stays dataset-size-independent.
    """
    from transformers import TrainingArguments
    from trl import SFTTrainer

    output_dir = resolve_path(tcfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recent TRL versions dropped dataset_text_field in favour of receiving a
    # dataset that already has a single 'text' column.  Map here so the trainer
    # sees exactly one column regardless of what else the source dataset carries.
    text_col = dcfg["text_column"]
    def keep_text(example):
        return {"text": example[text_col]}

    train_ds = dataset["train"].map(keep_text, remove_columns=dataset["train"].column_names)
    eval_ds  = dataset["validation"].map(keep_text, remove_columns=dataset["validation"].column_names)

    # Compute warmup_steps from warmup_ratio so we avoid the deprecated
    # warmup_ratio kwarg in Transformers 5.x (removed in favour of explicit steps).
    eff_batch      = tcfg["batch_size"] * tcfg["gradient_accumulation_steps"]
    total_steps    = (len(train_ds) // eff_batch) * tcfg["epochs"]
    warmup_steps   = max(1, int(total_steps * tcfg["warmup_ratio"]))

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        # Batch config — effective batch = batch_size × gradient_accumulation_steps
        per_device_train_batch_size=tcfg["batch_size"],
        per_device_eval_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],

        # Optimisation
        num_train_epochs=tcfg["epochs"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_steps=warmup_steps,

        # T4 GPUs do not support bf16; force fp16 for both training and BnB compute.
        fp16=True,
        bf16=False,

        # Checkpointing and evaluation
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        eval_steps=tcfg["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Misc
        report_to="none",       # swap to "mlflow" or "wandb" when tracking is ready
        run_name="reviewmind-sft",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    return trainer


# ---------------------------------------------------------------------------
# HuggingFace Hub — merge adapter and push full model
# ---------------------------------------------------------------------------

def push_merged_model_to_hub(
    adapter_dir: Path,
    model_name: str,
    repo_id: str,
    hf_token: str,
) -> None:
    """
    Reload the base model in fp16 (no quantization), merge the saved LoRA
    adapter into it, then upload the merged weights and tokenizer to the Hub.

    Why reload instead of merging in-place?  The training model is quantized
    to 4-bit — merge_and_unload() on a quantized model produces dequantized
    weights that differ from a clean fp16 merge.  Reloading in fp16 gives a
    clean, reproducible merged checkpoint that downstream users can load
    without bitsandbytes.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 55)
    print("  Merging adapter and pushing to HuggingFace Hub")
    print("=" * 55)

    print(f"\nReloading base model in fp16 for merge: {model_name} …")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=False,
    )

    print(f"Merging LoRA adapter from {adapter_dir} …")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()

    print(f"Pushing merged model → https://huggingface.co/{repo_id} …")
    merged_model.push_to_hub(repo_id, token=hf_token, private=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.push_to_hub(repo_id, token=hf_token)

    print(f"\n  ✓ Model available at https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    print("\n" + "=" * 55)
    print("  ReviewMind — QLoRA Fine-Tuning Setup")
    print("  Dry-run mode" if args.dry_run else "  TRAINING MODE")
    print("=" * 55)

    # ── 0. Mount Google Drive (Colab only) ───────────────────────
    print("\n[0] Checking Google Drive …")
    gdrive_dir = mount_google_drive()
    if gdrive_dir is not None:
        cfg["training"]["output_dir"] = str(gdrive_dir)
        print(f"  output_dir overridden → {gdrive_dir}")

    # ── 1. GPU check ────────────────────────────────────────────
    print("\n[1/6] Checking GPU availability …")
    require_gpu()

    # ── 2. BitsAndBytes quantization config ─────────────────────
    print("\n[2/6] Building quantization config (QLoRA / 4-bit NF4) …")
    bnb_config = build_bnb_config(cfg["quantization"])
    print(f"  Quantization : {cfg['quantization']['bnb_4bit_quant_type'].upper()}"
          f"  double_quant={cfg['quantization']['bnb_4bit_use_double_quant']}"
          f"  compute_dtype={cfg['quantization']['bnb_4bit_compute_dtype']}")

    # ── 3. Load base model + tokenizer ──────────────────────────
    print(f"\n[3/6] Loading model: {cfg['model']['model_name']} …")
    model, tokenizer = load_model_and_tokenizer(cfg["model"]["model_name"], bnb_config)
    tokenizer.model_max_length = cfg["training"]["max_seq_length"]
    print(f"  tokenizer.model_max_length set to {tokenizer.model_max_length}")

    # ── 4. Apply LoRA adapter ────────────────────────────────────
    print("\n[4/6] Applying LoRA adapter …")
    lora_config = build_lora_config(cfg["lora"])
    model = apply_lora(model, lora_config)

    lcfg = cfg["lora"]
    print(f"  rank={lcfg['lora_rank']}  alpha={lcfg['lora_alpha']}"
          f"  dropout={lcfg['lora_dropout']}"
          f"  targets={lcfg['target_modules']}")

    print_param_summary(model)

    # ── 5. Load dataset ──────────────────────────────────────────
    print("[5/6] Loading dataset splits …")
    dataset = load_splits(cfg["data"])

    # ── 6. Build trainer ─────────────────────────────────────────
    print("\n[6/6] Configuring SFTTrainer …")
    trainer = build_trainer(
        model, tokenizer, dataset,
        tcfg=cfg["training"],
        dcfg=cfg["data"],
        dry_run=args.dry_run,
    )

    tcfg = cfg["training"]
    eff_batch = tcfg["batch_size"] * tcfg["gradient_accumulation_steps"]
    total_steps  = (len(dataset["train"]) // eff_batch) * tcfg["epochs"]
    warmup_steps = max(1, int(total_steps * tcfg["warmup_ratio"]))
    print(f"  epochs={tcfg['epochs']}  lr={tcfg['learning_rate']}"
          f"  scheduler={tcfg['lr_scheduler_type']}")
    print(f"  batch_size={tcfg['batch_size']}  grad_accum={tcfg['gradient_accumulation_steps']}"
          f"  → effective_batch={eff_batch}")
    print(f"  total_steps≈{total_steps}  warmup_steps={warmup_steps}"
          f"  (ratio={tcfg['warmup_ratio']})")
    print(f"  max_seq_length={tcfg['max_seq_length']}  fp16={tcfg['fp16']}")
    print(f"  checkpoints → {resolve_path(tcfg['output_dir'])}")

    # ── Train (or exit if dry run) ───────────────────────────────
    if args.dry_run:
        print("\n" + "=" * 55)
        print("  DRY-RUN COMPLETE — all components loaded successfully.")
        print("  Re-run without --dry-run to start training.")
        print("=" * 55)
        return

    print("\n" + "=" * 55)
    print("  Starting training …")
    print("=" * 55 + "\n")

    trainer.train()

    # Save the final LoRA adapter weights (not the full merged model)
    adapter_dir = resolve_path(tcfg["output_dir"]) / "final_adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nFinal adapter saved to: {adapter_dir}")

    # ── Push merged model to HuggingFace Hub ────────────────────
    if args.no_push_hub:
        print("\nSkipping Hub push (--no-push-hub).")
        return

    hub_cfg = cfg.get("hub", {})
    repo_id = args.hf_repo or hub_cfg.get("repo_id")
    push_to_hub = hub_cfg.get("push_to_hub", True)

    if not push_to_hub:
        print("\nSkipping Hub push (hub.push_to_hub=false in config).")
        return

    if not repo_id:
        print(
            "\nNo Hub repo configured — skipping push.\n"
            "  Set hub.repo_id in the config or pass --hf-repo username/repo-name."
        )
        return

    hf_token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not hf_token:
        print(
            "\nNo HuggingFace token found — skipping push.\n"
            "  Set the $HF_TOKEN environment variable or pass --hf-token TOKEN."
        )
        return

    push_merged_model_to_hub(adapter_dir, cfg["model"]["model_name"], repo_id, hf_token)


if __name__ == "__main__":
    main()
