"""
ReviewMind – Chain-of-Thought Evaluation
Compares plain prompting vs 4-step CoT prompting on 50 test examples.
Loads the fine-tuned LoRA model from HuggingFace, generates both styles
of review for each diff, saves all results to evaluation/results/,
and prints a summary with side-by-side examples.

Usage:
    python evaluation/cot_eval.py
    python evaluation/cot_eval.py --n-examples 20 --max-new-tokens 256
    python evaluation/cot_eval.py --adapter-repo username/my-model
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

COT_SUFFIX = (
    "\n\nThink through your review step by step:\n"
    "Step 1: Understand what the code does\n"
    "Step 2: Analyse for correctness, security, performance, and "
    "readability issues\n"
    "Step 3: Prioritise which issues are critical vs minor\n"
    "Step 4: Write the final structured review"
)

TEST_JSONL   = PROJECT_ROOT / "data" / "splits" / "test.jsonl"
RESULTS_DIR  = PROJECT_ROOT / "evaluation" / "results"
RESULTS_FILE = RESULTS_DIR / "cot_comparison.json"

DEFAULT_N_EXAMPLES    = 50
DEFAULT_MAX_NEW_TOKENS = 256


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CoT vs plain prompt evaluation for ReviewMind",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-examples", type=int, default=DEFAULT_N_EXAMPLES,
        help="Number of test examples to evaluate.",
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(base_model: str, adapter_repo: str):
    """Load the base model in 4-bit and attach the LoRA adapter."""
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
    """
    Extract the diff and reference review from a formatted training example.
    The text field uses <|user|>…<|end|><|assistant|>…<|end|> structure.
    """
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
# Prompt building and generation
# ---------------------------------------------------------------------------

def build_prompt(diff: str, cot: bool) -> str:
    user_content = f"Review this pull request diff:\n\n{diff}"
    if cot:
        user_content += COT_SUFFIX
    return (
        f"<|system|>{SYSTEM_PROMPT}<|end|>"
        f"<|user|>{user_content}<|end|>"
        f"<|assistant|>"
    )


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic for fair comparison
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list, n_side_by_side: int = 3) -> None:
    plain_lens = [len(r["plain_response"].split()) for r in results]
    cot_lens   = [len(r["cot_response"].split())   for r in results]

    avg_plain = sum(plain_lens) / len(plain_lens) if plain_lens else 0
    avg_cot   = sum(cot_lens)   / len(cot_lens)   if cot_lens   else 0
    pct_change = (avg_cot - avg_plain) / max(avg_plain, 1) * 100

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Examples evaluated      : {len(results)}")
    print(f"  Avg plain response      : {avg_plain:.1f} words")
    print(f"  Avg CoT response        : {avg_cot:.1f} words")
    print(f"  CoT length change       : {pct_change:+.1f}%")
    print()

    print("─" * 70)
    print(f"  SIDE-BY-SIDE EXAMPLES  (showing {n_side_by_side})")
    print("─" * 70)

    for i, r in enumerate(results[:n_side_by_side]):
        print(f"\n{'━' * 70}")
        print(f"  Example {i + 1}")
        print(f"{'━' * 70}")

        diff_preview = r["diff"][:250].replace("\n", " ")
        if len(r["diff"]) > 250:
            diff_preview += " …"
        print(f"\n  DIFF (preview):\n    {diff_preview}")

        print(f"\n  ┌─ PLAIN ({len(r['plain_response'].split())} words) " + "─" * 40)
        for line in r["plain_response"][:500].splitlines():
            print(f"  │  {line}")
        if len(r["plain_response"]) > 500:
            print("  │  …")

        print(f"\n  ┌─ COT ({len(r['cot_response'].split())} words) " + "─" * 42)
        for line in r["cot_response"][:500].splitlines():
            print(f"  │  {line}")
        if len(r["cot_response"]) > 500:
            print("  │  …")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  ReviewMind – CoT vs Plain Prompt Evaluation")
    print("=" * 70)
    print(f"  Base model   : {args.base_model}")
    print(f"  Adapter      : {args.adapter_repo}")
    print(f"  Examples     : {args.n_examples}")
    print(f"  Max tokens   : {args.max_new_tokens}")

    # ── Load model ──────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_repo)

    # ── Load and parse examples ─────────────────────────────────
    raw = load_test_examples(TEST_JSONL, args.n_examples)
    examples = [e for e in (parse_example(r) for r in raw) if e is not None]
    print(f"  Parsed {len(examples)} valid examples")

    # ── Generate ────────────────────────────────────────────────
    results = []
    for i, ex in enumerate(examples):
        print(f"\n[{i + 1}/{len(examples)}] Generating …", end=" ", flush=True)
        t0 = time.time()

        plain = generate(model, tokenizer,
                         build_prompt(ex["diff"], cot=False), args.max_new_tokens)
        cot   = generate(model, tokenizer,
                         build_prompt(ex["diff"], cot=True),  args.max_new_tokens)

        elapsed = time.time() - t0
        print(f"{elapsed:.1f}s  |  plain={len(plain.split())}w  cot={len(cot.split())}w")

        results.append({
            "index":          i,
            "diff":           ex["diff"],
            "reference":      ex["reference"],
            "plain_response": plain,
            "cot_response":   cot,
        })

    # ── Save ────────────────────────────────────────────────────
    with RESULTS_FILE.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {RESULTS_FILE}")

    # ── Summary ─────────────────────────────────────────────────
    print_summary(results)


if __name__ == "__main__":
    main()
