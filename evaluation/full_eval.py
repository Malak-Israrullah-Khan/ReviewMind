"""
ReviewMind – Full 4-Way Evaluation
Runs 4 model configurations on N test examples and scores each on
specificity, actionability, and detail (1-5 each) using the same
heuristic as evaluation/score_cot.py.

Configurations
--------------
  base      – base LLM (meta-llama/Llama-3.1-8B-Instruct), no adapter
  finetuned – base + LoRA adapter (Malak-Israr/reviewmind-lora)
  cot       – base + LoRA adapter + 4-step CoT suffix in prompt
  rag       – base + LoRA adapter + guidelines retrieved from ChromaDB

Output
------
  evaluation/results/full_comparison.json  – per-example scores + responses
  stdout                                   – side-by-side comparison table

Usage
-----
    python evaluation/full_eval.py
    python evaluation/full_eval.py --n-examples 20 --max-new-tokens 300
    python evaluation/full_eval.py --no-rag   # skip RAG (vectorstore absent)
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
sys.path.insert(0, str(PROJECT_ROOT))

BASE_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO = "Malak-Israr/reviewmind-lora"
TEST_JSONL   = PROJECT_ROOT / "data" / "splits" / "test.jsonl"
OUTPUT_DIR   = PROJECT_ROOT / "evaluation" / "results"
OUTPUT_FILE  = OUTPUT_DIR / "full_comparison.json"

DEFAULT_N_EXAMPLES     = 20
DEFAULT_MAX_NEW_TOKENS = 300
DEFAULT_TOP_K          = 3

SYSTEM_PROMPT = (
    "You are a senior software engineer conducting a thorough code review. "
    "Analyse the code carefully and provide specific, actionable feedback."
)
RAG_SYSTEM_PROMPT = (
    "You are a senior software engineer conducting a thorough code review. "
    "Use the provided coding guidelines as context when reviewing the diff. "
    "Reference relevant standards and provide specific, actionable feedback."
)
COT_SUFFIX = (
    "\n\nThink through your review step by step:\n"
    "Step 1: Understand what the code does\n"
    "Step 2: Analyse for correctness, security, performance, and "
    "readability issues\n"
    "Step 3: Prioritise which issues are critical vs minor\n"
    "Step 4: Write the final structured review"
)

VERSIONS = ["base", "finetuned", "cot", "rag"]
_LABELS = {
    "base":      "Base model (no adapter)",
    "finetuned": "Fine-tuned (LoRA)     ",
    "cot":       "Fine-tuned + CoT      ",
    "rag":       "Fine-tuned + RAG      ",
}

# ---------------------------------------------------------------------------
# Scoring – reuse functions from evaluation/score_cot.py
# ---------------------------------------------------------------------------

from evaluation.score_cot import score_response, total as score_total  # noqa: E402

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full 4-way evaluation: base / fine-tuned / CoT / RAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-examples",     type=int, default=DEFAULT_N_EXAMPLES,
                        help="Number of test examples.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="Max new tokens per generation.")
    parser.add_argument("--top-k",          type=int, default=DEFAULT_TOP_K,
                        help="RAG: guideline chunks to retrieve per diff.")
    parser.add_argument("--adapter-repo",   default=ADAPTER_REPO, metavar="USERNAME/REPO")
    parser.add_argument("--base-model",     default=BASE_MODEL)
    parser.add_argument("--output",         default=str(OUTPUT_FILE))
    parser.add_argument("--no-rag",         action="store_true",
                        help="Skip RAG version (use when vectorstore is absent).")
    parser.add_argument("--rescore-only",   action="store_true",
                        help="Re-score an existing full_comparison.json with the "
                             "current scoring functions; skip all model loading.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base(base_model_id: str):
    """Load base model in 4-bit NF4 quantisation, without any adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\nLoading tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    print(f"Loading base model (4-bit NF4): {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=False,
    )
    model.eval()
    print("  Base model ready.")
    return model, tokenizer


def attach_adapter(base_model, adapter_repo: str):
    """Wrap base model with a LoRA adapter via PEFT."""
    from peft import PeftModel
    print(f"Attaching LoRA adapter: {adapter_repo}")
    ft_model = PeftModel.from_pretrained(base_model, adapter_repo)
    ft_model.eval()
    print("  Adapter attached.")
    return ft_model


# ---------------------------------------------------------------------------
# Dataset
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
    asst_m = re.search(r"<\|assistant\|>(.*?)<\|end\|>", text, re.DOTALL)
    if not user_m:
        return None
    user_text = user_m.group(1).strip()
    prefix = "Review this pull request diff:"
    diff = user_text[len(prefix):].strip() if user_text.startswith(prefix) else user_text
    reference = asst_m.group(1).strip() if asst_m else ""
    return {"diff": diff, "reference": reference}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_plain_prompt(diff: str) -> str:
    return (
        f"<|system|>{SYSTEM_PROMPT}<|end|>"
        f"<|user|>Review this pull request diff:\n\n{diff}<|end|>"
        f"<|assistant|>"
    )


def build_cot_prompt(diff: str) -> str:
    return (
        f"<|system|>{SYSTEM_PROMPT}<|end|>"
        f"<|user|>Review this pull request diff:\n\n{diff}{COT_SUFFIX}<|end|>"
        f"<|assistant|>"
    )


def build_rag_prompt(diff: str, context_block: str) -> str:
    if not context_block:
        return build_plain_prompt(diff)
    user_content = (
        f"{context_block}\n\n"
        "Now review the following pull request diff using the guidelines "
        f"above where relevant:\n\n{diff}"
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
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# RAG initialisation
# ---------------------------------------------------------------------------

def init_retriever(no_rag: bool):
    if no_rag:
        print("\n[RAG] Skipped (--no-rag).")
        return None
    try:
        from rag.retriever import Retriever
        retriever = Retriever()
        size = retriever.collection_size()
        print(f"\n[RAG] Vector store ready ({size:,} chunks).")
        return retriever
    except SystemExit as exc:
        print(f"\n[RAG] Vector store unavailable ({exc}). "
              "RAG version will fall back to plain prompt.")
        return None
    except Exception as exc:
        print(f"\n[RAG] Retriever error ({exc}). "
              "RAG version will fall back to plain prompt.")
        return None


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _avg(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def print_comparison_table(per_version: dict[str, list[dict]], n: int) -> None:
    criteria = ["specificity", "actionability", "detail"]
    col_w = 10

    avgs: dict[str, dict[str, float]] = {}
    for v in VERSIONS:
        rows = per_version[v]
        avgs[v] = {c: _avg([r[c] for r in rows]) for c in criteria}
        avgs[v]["total"] = sum(avgs[v][c] for c in criteria)

    base_total = avgs["base"]["total"]

    print("\n" + "=" * 74)
    print(f"  FULL EVALUATION  (heuristic 1–5 per criterion · {n} examples)")
    print("=" * 74)
    print(
        f"  {'Version':<26}"
        f" {'Spec':>{col_w}}"
        f" {'Action':>{col_w}}"
        f" {'Detail':>{col_w}}"
        f" {'Total/15':>{col_w}}"
        f"  vs Base"
    )
    print("  " + "─" * 70)

    for v in VERSIONS:
        a = avgs[v]
        if v == "base":
            delta_str = "   —"
            arrow     = ""
        else:
            delta = a["total"] - base_total
            sign  = "+" if delta >= 0 else ""
            arrow = " ▲" if delta > 0 else (" ▼" if delta < 0 else " =")
            delta_str = f"{sign}{delta:.2f}"
        print(
            f"  {_LABELS[v]:<26}"
            f" {a['specificity']:>{col_w}.2f}"
            f" {a['actionability']:>{col_w}.2f}"
            f" {a['detail']:>{col_w}.2f}"
            f" {a['total']:>{col_w}.2f}"
            f"  {delta_str}{arrow}"
        )

    print("=" * 74)


# ---------------------------------------------------------------------------
# Rescore-only mode
# ---------------------------------------------------------------------------

def rescore_existing(input_path: Path, output_path: Path) -> None:
    """Load an existing full_comparison.json, re-score all responses with the
    current scoring functions, overwrite the file, and print the table."""
    if not input_path.exists():
        sys.exit(f"\nFile not found: {input_path}\nRun full_eval.py without --rescore-only first.")

    with input_path.open(encoding="utf-8") as fh:
        results = json.load(fh)

    print(f"Loaded {len(results)} entries from {input_path}")
    print("Re-scoring with updated heuristics …\n")

    per_version: dict[str, list[dict]] = {v: [] for v in VERSIONS}

    for r in results:
        sc_base = score_response(r["base_response"])
        sc_ft   = score_response(r["finetuned_response"])
        sc_cot  = score_response(r["cot_response"])
        sc_rag  = score_response(r["rag_response"])

        r["base_scores"]      = sc_base
        r["finetuned_scores"] = sc_ft
        r["cot_scores"]       = sc_cot
        r["rag_scores"]       = sc_rag
        r["base_total"]       = score_total(sc_base)
        r["finetuned_total"]  = score_total(sc_ft)
        r["cot_total"]        = score_total(sc_cot)
        r["rag_total"]        = score_total(sc_rag)

        per_version["base"].append(sc_base)
        per_version["finetuned"].append(sc_ft)
        per_version["cot"].append(sc_cot)
        per_version["rag"].append(sc_rag)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"Updated scores saved → {output_path}")

    print_comparison_table(per_version, len(results))

    print("  Average response lengths:")
    for v, key in [
        ("base",      "base_response"),
        ("finetuned", "finetuned_response"),
        ("cot",       "cot_response"),
        ("rag",       "rag_response"),
    ]:
        avg_w = _avg([len(r[key].split()) for r in results])
        print(f"    {_LABELS[v]}  {avg_w:.0f} words")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    if args.rescore_only:
        print("\n" + "=" * 74)
        print("  ReviewMind – Rescore Existing Results")
        print("=" * 74)
        print(f"  Input  : {output_path}")
        rescore_existing(output_path, output_path)
        return

    print("\n" + "=" * 74)
    print("  ReviewMind – Full 4-Way Evaluation")
    print("=" * 74)
    print(f"  Base model    : {args.base_model}")
    print(f"  Adapter       : {args.adapter_repo}")
    print(f"  Examples      : {args.n_examples}")
    print(f"  Max tokens    : {args.max_new_tokens}")
    print(f"  RAG top-k     : {args.top_k}{' (skipped)' if args.no_rag else ''}")
    print(f"  Output        : {output_path}")

    # ── Dataset ─────────────────────────────────────────────────────────────
    print("\n[1/5] Loading test examples …")
    examples = load_test_examples(TEST_JSONL, args.n_examples)
    if not examples:
        sys.exit("No valid examples found — aborting.")

    # ── RAG retriever ────────────────────────────────────────────────────────
    print("\n[2/5] Initialising RAG retriever …")
    retriever = init_retriever(args.no_rag)

    # ── Base model ───────────────────────────────────────────────────────────
    print("\n[3/5] Loading base model (no adapter) …")
    base_model, tokenizer = load_base(args.base_model)

    # Pass 1: generate base-only responses before the adapter is attached.
    # This ensures version 1 uses purely the base weights.
    print(f"\n  Generating base responses ({len(examples)} examples) …")
    base_responses: list[str] = []
    for i, ex in enumerate(examples):
        t0   = time.time()
        resp = generate(base_model, tokenizer,
                        build_plain_prompt(ex["diff"]), args.max_new_tokens)
        base_responses.append(resp)
        print(f"    [{i + 1:>2}/{len(examples)}] {len(resp.split())}w  {time.time()-t0:.0f}s")

    # ── Fine-tuned model ─────────────────────────────────────────────────────
    print("\n[4/5] Attaching LoRA adapter …")
    ft_model = attach_adapter(base_model, args.adapter_repo)

    # Pass 2: fine-tuned, CoT, RAG responses
    print(f"\n[5/5] Generating fine-tuned / CoT / RAG responses "
          f"({len(examples)} examples × 3 variants) …\n")

    results: list[dict] = []
    per_version: dict[str, list[dict]] = {v: [] for v in VERSIONS}

    for i, ex in enumerate(examples):
        diff      = ex["diff"]
        base_resp = base_responses[i]
        print(f"  [{i + 1:>2}/{len(examples)}]", end=" ", flush=True)

        # Fine-tuned (plain prompt)
        t0    = time.time()
        ft_resp = generate(ft_model, tokenizer,
                           build_plain_prompt(diff), args.max_new_tokens)
        t_ft  = time.time() - t0

        # CoT
        t0      = time.time()
        cot_resp = generate(ft_model, tokenizer,
                            build_cot_prompt(diff), args.max_new_tokens)
        t_cot   = time.time() - t0

        # RAG
        t0 = time.time()
        context_block = ""
        if retriever is not None:
            try:
                chunks        = retriever.retrieve(diff, top_k=args.top_k)
                context_block = retriever.format_context(chunks)
            except Exception:
                context_block = ""
        rag_resp = generate(ft_model, tokenizer,
                            build_rag_prompt(diff, context_block), args.max_new_tokens)
        t_rag    = time.time() - t0

        print(
            f"ft={len(ft_resp.split())}w({t_ft:.0f}s)"
            f"  cot={len(cot_resp.split())}w({t_cot:.0f}s)"
            f"  rag={len(rag_resp.split())}w({t_rag:.0f}s)"
        )

        # Score all 4 versions
        sc_base = score_response(base_resp)
        sc_ft   = score_response(ft_resp)
        sc_cot  = score_response(cot_resp)
        sc_rag  = score_response(rag_resp)

        per_version["base"].append(sc_base)
        per_version["finetuned"].append(sc_ft)
        per_version["cot"].append(sc_cot)
        per_version["rag"].append(sc_rag)

        results.append({
            "index":              i,
            "diff":               diff[:1000],
            "reference":          ex.get("reference", ""),
            "base_response":      base_resp,
            "finetuned_response": ft_resp,
            "cot_response":       cot_resp,
            "rag_response":       rag_resp,
            "base_scores":        sc_base,
            "finetuned_scores":   sc_ft,
            "cot_scores":         sc_cot,
            "rag_scores":         sc_rag,
            "base_total":         score_total(sc_base),
            "finetuned_total":    score_total(sc_ft),
            "cot_total":          score_total(sc_cot),
            "rag_total":          score_total(sc_rag),
        })

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {output_path}  ({len(results)} entries)")

    # ── Comparison table ──────────────────────────────────────────────────────
    print_comparison_table(per_version, len(results))

    # ── Word-count summary ────────────────────────────────────────────────────
    print("  Average response lengths:")
    for v, key in [
        ("base",      "base_response"),
        ("finetuned", "finetuned_response"),
        ("cot",       "cot_response"),
        ("rag",       "rag_response"),
    ]:
        avg_w = _avg([len(r[key].split()) for r in results])
        print(f"    {_LABELS[v]}  {avg_w:.0f} words")
    print()


if __name__ == "__main__":
    main()
