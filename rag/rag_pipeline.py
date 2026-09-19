"""
ReviewMind – RAG Pipeline
Loads the fine-tuned model (Malak-Israr/reviewmind-lora), retrieves relevant
coding guidelines from the vector store for each diff, injects them as context
before the diff, generates a RAG-augmented review, and saves side-by-side
comparisons with and without RAG to evaluation/results/rag_comparison.json.

Usage:
    python rag/rag_pipeline.py
    python rag/rag_pipeline.py --n-examples 5 --top-k 3 --max-new-tokens 350
    python rag/rag_pipeline.py --no-push-results     # skip writing output file

The script produces a JSON file with this structure per entry:
    {
        "index":           int,
        "diff":            str,          # the input diff
        "baseline_review": str,          # review generated WITHOUT RAG context
        "rag_review":      str,          # review generated WITH RAG context
        "retrieved_chunks": [            # chunks injected as context
            {"source": str, "chunk_index": int, "distance": float, "text": str}
        ],
        "context_block":   str           # formatted text that was injected
    }
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL       = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO     = "Malak-Israr/reviewmind-lora"
TEST_JSONL       = PROJECT_ROOT / "data" / "splits" / "test.jsonl"
OUTPUT_DIR       = PROJECT_ROOT / "evaluation" / "results"
OUTPUT_FILE      = OUTPUT_DIR / "rag_comparison.json"

DEFAULT_N_EXAMPLES    = 10
DEFAULT_TOP_K         = 3
DEFAULT_MAX_NEW_TOKENS = 350

SYSTEM_PROMPT = (
    "You are a senior software engineer conducting a thorough code review. "
    "Analyse the code carefully and provide specific, actionable feedback."
)

RAG_SYSTEM_PROMPT = (
    "You are a senior software engineer conducting a thorough code review. "
    "Use the provided coding guidelines as context when reviewing the diff. "
    "Reference relevant standards and provide specific, actionable feedback."
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RAG-augmented vs baseline code reviews",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-examples", type=int, default=DEFAULT_N_EXAMPLES,
        help="Number of test examples to process.",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help="Number of guideline chunks to retrieve per diff.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum tokens to generate per review.",
    )
    parser.add_argument(
        "--adapter-repo", default=ADAPTER_REPO, metavar="USERNAME/REPO",
        help="HuggingFace repo ID for the LoRA adapter.",
    )
    parser.add_argument(
        "--base-model", default=BASE_MODEL,
        help="HuggingFace base model ID.",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_FILE),
        help="Path to write rag_comparison.json.",
    )
    parser.add_argument(
        "--no-push-results", action="store_true",
        help="Skip writing the JSON output file.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(base_model: str, adapter_repo: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\nLoading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
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

    print(f"Attaching LoRA adapter: {adapter_repo}")
    model = PeftModel.from_pretrained(base, adapter_repo)
    model.eval()
    print("  Model ready.\n")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_test_examples(path: Path, n: int) -> list[dict]:
    if not path.exists():
        sys.exit(
            f"\nTest file not found: {path}\n"
            "Run  python data/prepare_dataset.py  first."
        )
    examples = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            parsed = _parse_example(raw)
            if parsed:
                examples.append(parsed)
            if len(examples) >= n:
                break
    print(f"Loaded {len(examples)} examples from {path}")
    return examples


def _parse_example(raw: dict) -> dict | None:
    text = raw.get("text", "")
    user_m = re.search(r"<\|user\|>(.*?)<\|end\|>", text, re.DOTALL)
    if not user_m:
        return None
    user_text = user_m.group(1).strip()
    prefix = "Review this pull request diff:"
    diff = user_text[len(prefix):].strip() if user_text.startswith(prefix) else user_text
    return {"diff": diff}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_baseline_prompt(diff: str) -> str:
    """Prompt without any retrieved context."""
    return (
        f"<|system|>{SYSTEM_PROMPT}<|end|>"
        f"<|user|>Review this pull request diff:\n\n{diff}<|end|>"
        f"<|assistant|>"
    )


def build_rag_prompt(diff: str, context_block: str) -> str:
    """Prompt with retrieved guidelines injected before the diff."""
    user_content = (
        f"{context_block}\n\n"
        f"Now review the following pull request diff using the guidelines above "
        f"where relevant:\n\n{diff}"
    )
    return (
        f"<|system|>{RAG_SYSTEM_PROMPT}<|end|>"
        f"<|user|>{user_content}<|end|>"
        f"<|assistant|>"
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1536,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # greedy decoding for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    print("\n" + "=" * 65)
    print("  ReviewMind – RAG Pipeline")
    print("=" * 65)
    print(f"  Base model    : {args.base_model}")
    print(f"  Adapter       : {args.adapter_repo}")
    print(f"  Examples      : {args.n_examples}")
    print(f"  Top-k chunks  : {args.top_k}")
    print(f"  Max tokens    : {args.max_new_tokens}")
    print(f"  Output        : {output_path}")

    # ── Retriever ──────────────────────────────────────────────────
    print("\n[1/4] Initialising retriever …")
    # Import from the same package directory
    sys.path.insert(0, str(PROJECT_ROOT))
    from rag.retriever import Retriever

    retriever = Retriever()
    print(f"  Vector store has {retriever.collection_size():,} chunks")

    # ── Model ────────────────────────────────────────────────────────
    print("\n[2/4] Loading model …")
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_repo)

    # ── Examples ───────────────────────────────────────────────────
    print("\n[3/4] Loading test examples …")
    examples = load_test_examples(TEST_JSONL, args.n_examples)
    if not examples:
        sys.exit("No valid examples found — aborting.")

    # ── Generate comparisons ─────────────────────────────────────────────
    print(f"\n[4/4] Generating {len(examples)} side-by-side comparisons …\n")
    results = []

    for i, ex in enumerate(examples):
        diff = ex["diff"]
        print(f"  [{i + 1}/{len(examples)}] Retrieving guidelines …", end=" ", flush=True)
        chunks = retriever.retrieve(diff, top_k=args.top_k)
        context_block = retriever.format_context(chunks)

        # Baseline (no RAG)
        t0 = time.time()
        baseline_prompt = build_baseline_prompt(diff)
        baseline_review = generate(model, tokenizer, baseline_prompt, args.max_new_tokens)
        t_base = time.time() - t0

        # RAG-augmented
        t0 = time.time()
        rag_prompt  = build_rag_prompt(diff, context_block)
        rag_review  = generate(model, tokenizer, rag_prompt, args.max_new_tokens)
        t_rag = time.time() - t0

        print(
            f"baseline={len(baseline_review.split())}w ({t_base:.1f}s)  "
            f"rag={len(rag_review.split())}w ({t_rag:.1f}s)"
        )

        results.append(
            {
                "index": i,
                "diff": diff,
                "baseline_review": baseline_review,
                "rag_review": rag_review,
                "retrieved_chunks": [
                    {
                        "source":      c["source"],
                        "chunk_index": c["chunk_index"],
                        "distance":    round(c["distance"], 6),
                        "text":        c["document"],
                    }
                    for c in chunks
                ],
                "context_block": context_block,
            }
        )

    # ── Save ──────────────────────────────────────────────────────────────
    if args.no_push_results:
        print("\nSkipping file write (--no-push-results).")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"\nResults saved → {output_path}  ({len(results)} entries)")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PIPELINE SUMMARY")
    print("=" * 65)
    print(f"  Examples processed  : {len(results)}")
    avg_chunks = (
        sum(len(r["retrieved_chunks"]) for r in results) / len(results)
        if results else 0
    )
    avg_base = (
        sum(len(r["baseline_review"].split()) for r in results) / len(results)
        if results else 0
    )
    avg_rag = (
        sum(len(r["rag_review"].split()) for r in results) / len(results)
        if results else 0
    )
    print(f"  Avg chunks injected : {avg_chunks:.1f} per example")
    print(f"  Avg baseline length : {avg_base:.0f} words")
    print(f"  Avg RAG length      : {avg_rag:.0f} words")

    if results:
        print("\n  SAMPLE (index 0):")
        r = results[0]
        print(f"\n  Diff preview : {r['diff'][:120].replace(chr(10), ' ')} …")
        print(f"\n  Baseline (first 250 chars):")
        print(f"    {r['baseline_review'][:250].replace(chr(10), ' ')}")
        print(f"\n  RAG review (first 250 chars):")
        print(f"    {r['rag_review'][:250].replace(chr(10), ' ')}")
        if r["retrieved_chunks"]:
            print(f"\n  Top retrieved chunk ({r['retrieved_chunks'][0]['source']}):")
            print(f"    {r['retrieved_chunks'][0]['text'][:200].replace(chr(10), ' ')} …")

    print("=" * 65)


if __name__ == "__main__":
    main()
