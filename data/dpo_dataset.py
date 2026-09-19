"""
ReviewMind – DPO Pair Generation
Loads examples from data/splits/test.jsonl, generates two reviews per
example using the fine-tuned LoRA model — one at temperature 0.3
(focused, chosen) and one at temperature 0.9 (diverse, rejected) —
and writes data/dpo_pairs.jsonl in the format expected by TRL's
DPOTrainer:

    {"prompt": "<|system|>...<|end|><|user|>...<|end|>",
     "chosen": "<|assistant|>...<|end|>",
     "rejected": "<|assistant|>...<|end|>"}

The prompt field ends before <|assistant|> so TRL can concatenate
prompt+chosen / prompt+rejected when computing log-probabilities.

Usage:
    python data/dpo_dataset.py
    python data/dpo_dataset.py --n-examples 100 --max-new-tokens 300
    python data/dpo_dataset.py --adapter-repo username/my-lora
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO = "Malak-Israr/reviewmind-lora"

SYSTEM_PROMPT = (
    "You are a senior software engineer conducting a thorough code review. "
    "Analyse the code carefully and provide specific, actionable feedback."
)

TEST_JSONL  = PROJECT_ROOT / "data" / "splits" / "test.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "dpo_pairs.jsonl"

DEFAULT_N_EXAMPLES    = 200
DEFAULT_MAX_NEW_TOKENS = 256

# Temperature used for chosen / rejected responses
TEMP_CHOSEN   = 0.3   # focused, higher-quality
TEMP_REJECTED = 0.9   # more random, lower-quality


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs for ReviewMind",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-examples", type=int, default=DEFAULT_N_EXAMPLES,
        help="Number of test examples to process.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum tokens to generate per response.",
    )
    parser.add_argument(
        "--adapter-repo", default=ADAPTER_REPO,
        help="HuggingFace repo ID for the LoRA adapter.",
    )
    parser.add_argument(
        "--base-model", default=BASE_MODEL,
        help="HuggingFace base model ID.",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_FILE),
        help="Path to write dpo_pairs.jsonl.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(base_model: str, adapter_repo: str):
    """Load base model in 4-bit NF4 and attach the LoRA adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\nLoading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

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

    print(f"Loading LoRA adapter: {adapter_repo}")
    model = PeftModel.from_pretrained(base, adapter_repo)
    model.eval()
    print("  Model ready.\n")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset loading and parsing
# ---------------------------------------------------------------------------

def load_test_examples(path: Path, n: int) -> list:
    if not path.exists():
        sys.exit(
            f"\nTest file not found: {path}\n"
            "Run  python data/prepare_dataset.py  first."
        )
    examples = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
            if len(examples) >= n:
                break
    print(f"Loaded {len(examples)} examples from {path}")
    return examples


def parse_example(raw: dict) -> dict | None:
    """Extract diff (and optional reference) from a formatted training example."""
    text = raw.get("text", "")
    user_m = re.search(r"<\|user\|>(.*?)<\|end\|>", text, re.DOTALL)
    asst_m = re.search(r"<\|assistant\|>(.*?)<\|end\|>", text, re.DOTALL)
    if not user_m:
        return None
    user_text = user_m.group(1).strip()
    prefix = "Review this pull request diff:"
    diff = user_text[len(prefix):].strip() if user_text.startswith(prefix) else user_text
    reference = asst_m.group(1).strip() if asst_m else ""
    return {"diff": diff, "reference": reference}


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(diff: str) -> str:
    """
    Return the prompt string up to (but not including) the assistant turn
    opening token.  TRL's DPOTrainer concatenates prompt+chosen and
    prompt+rejected when computing reference / policy log-probs, so the
    split must happen exactly here.
    """
    user_content = f"Review this pull request diff:\n\n{diff}"
    return (
        f"<|system|>{SYSTEM_PROMPT}<|end|>"
        f"<|user|>{user_content}<|end|>"
    )


def wrap_response(text: str) -> str:
    """Wrap a generated response in the assistant chat tokens."""
    return f"<|assistant|>{text}<|end|>"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(model, tokenizer, prompt: str, temperature: float,
             max_new_tokens: int) -> str:
    """Generate a response for the given prompt at the given temperature."""
    import torch

    # Append the assistant opening token so the model knows it's its turn
    full_input = prompt + "<|assistant|>"

    inputs = tokenizer(
        full_input,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(pairs: list) -> None:
    chosen_lens   = [len(p["chosen"].split())   for p in pairs]
    rejected_lens = [len(p["rejected"].split()) for p in pairs]

    avg_c = sum(chosen_lens)   / len(chosen_lens)   if chosen_lens   else 0
    avg_r = sum(rejected_lens) / len(rejected_lens) if rejected_lens else 0

    print("\n" + "=" * 65)
    print("  DPO PAIR GENERATION SUMMARY")
    print("=" * 65)
    print(f"  Pairs generated             : {len(pairs)}")
    print(f"  Avg chosen  length (t=0.3)  : {avg_c:.1f} words")
    print(f"  Avg rejected length (t=0.9) : {avg_r:.1f} words")
    print()
    print("  SAMPLE PAIR (index 0):")
    if pairs:
        p = pairs[0]
        diff_snippet = p["prompt"].split("Review this pull request diff:")[-1][:150].strip()
        print(f"\n  Diff (preview): {diff_snippet[:100].replace(chr(10), ' ')} …")
        print(f"\n  Chosen  (t=0.3, first 200 chars):")
        print(f"    {p['chosen'][:200].replace(chr(10), ' ')}")
        print(f"\n  Rejected (t=0.9, first 200 chars):")
        print(f"    {p['rejected'][:200].replace(chr(10), ' ')}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    print("\n" + "=" * 65)
    print("  ReviewMind – DPO Pair Generation")
    print("=" * 65)
    print(f"  Base model   : {args.base_model}")
    print(f"  Adapter      : {args.adapter_repo}")
    print(f"  Examples     : {args.n_examples}")
    print(f"  Max tokens   : {args.max_new_tokens}")
    print(f"  Temp chosen  : {TEMP_CHOSEN}  (→ chosen)")
    print(f"  Temp rejected: {TEMP_REJECTED}  (→ rejected)")
    print(f"  Output       : {output_path}")

    # ── Load model ──────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_repo)

    # ── Load and parse examples ─────────────────────────────────
    raw = load_test_examples(TEST_JSONL, args.n_examples)
    examples = [e for e in (parse_example(r) for r in raw) if e is not None]
    print(f"  Parsed {len(examples)} valid examples\n")

    # ── Generate pairs ──────────────────────────────────────────
    pairs = []
    for i, ex in enumerate(examples):
        print(f"[{i + 1}/{len(examples)}] Generating …", end=" ", flush=True)
        t0 = time.time()

        prompt = build_prompt(ex["diff"])

        chosen_text   = generate(model, tokenizer, prompt, TEMP_CHOSEN,   args.max_new_tokens)
        rejected_text = generate(model, tokenizer, prompt, TEMP_REJECTED, args.max_new_tokens)

        elapsed = time.time() - t0
        print(
            f"{elapsed:.1f}s  |  "
            f"chosen={len(chosen_text.split())}w  "
            f"rejected={len(rejected_text.split())}w"
        )

        pairs.append({
            "prompt":   prompt,
            "chosen":   wrap_response(chosen_text),
            "rejected": wrap_response(rejected_text),
        })

    # ── Save ────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\nPairs saved → {output_path}  ({len(pairs)} lines)")

    print_summary(pairs)


if __name__ == "__main__":
    main()
