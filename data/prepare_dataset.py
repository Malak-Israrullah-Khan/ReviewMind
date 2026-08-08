"""
Day 3 – Data Preparation
Loads fasterinnerlooper/codereviewer (publicly accessible mirror of the
Microsoft CodeReviewer corpus), applies quality filters, formats into
instruction-tuning format, re-splits 80/10/10, and writes .jsonl files
to data/splits/.
"""

import json
import random
import re
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found — install with: pip install tqdm")
    # Minimal no-op shim so the script still runs
    def tqdm(iterable, **kwargs):
        label = kwargs.get("desc", "")
        total = kwargs.get("total", "?")
        print(f"  Processing {label} ({total} rows) …")
        return iterable

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_NAME   = "fasterinnerlooper/codereviewer"
# This is a publicly accessible mirror of the Microsoft CodeReviewer corpus.
# It ships as a flat dataset (no named sub-configs) with columns:
#   patch  – unified diff of the code change
#   msg    – the human reviewer's comment
#   label  – 1 if the patch received a comment, 0 otherwise
#   url    – link to the original GitHub pull request
#   lang   – detected programming language
DATASET_CONFIG = None   # no named config for this mirror

RANDOM_SEED  = 42
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
# test = remaining 10 %

MIN_DIFF_CHARS   = 20   # shorter diffs carry almost no signal
MIN_REVIEW_CHARS = 10   # very short reviews are usually noise

SPLITS_DIR = Path(__file__).parent / "splits"
SPLITS_DIR.mkdir(exist_ok=True)

# System prompt that frames the LLM's role for every training example
SYSTEM_PROMPT = (
    "You are a senior software engineer conducting a thorough code review. "
    "Analyse the code carefully and provide specific, actionable feedback."
)

# Candidate column names tried in priority order.
# CodeReviewer uses 'patch' for the diff and 'msg' for the comment,
# but we support alternatives in case the schema evolves.
DIFF_CANDIDATES   = ("patch", "input", "diff", "code", "old_code")
REVIEW_CANDIDATES = ("msg",   "target", "comment", "review")

# A review that matches this pattern is just a bare URL — not useful signal
_URL_RE = re.compile(r'^\s*https?://\S+\s*$', re.IGNORECASE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_field(row: dict, candidates: tuple):
    """
    Return (field_name, stripped_string) for the first candidate present in
    row that has a non-None value.  Returns (None, None) if nothing matches.
    """
    for name in candidates:
        val = row.get(name)
        if val is not None:
            return name, str(val).strip()
    return None, None


def is_valid(diff: str, review: str) -> tuple:
    """
    Return (True, None) if the example passes all quality gates, or
    (False, reason_string) so callers can bucket skipped rows by reason.
    """
    if not diff or len(diff) < MIN_DIFF_CHARS:
        return False, "diff_too_short"
    if not review or len(review) < MIN_REVIEW_CHARS:
        return False, "review_too_short"
    if _URL_RE.match(review):
        return False, "review_is_url"
    if len(review.split()) <= 1:
        return False, "review_single_word"
    return True, None


def format_example(diff: str, review: str) -> dict:
    """
    Wrap a (diff, review) pair in the instruction-tuning chat template.
    The <|system|>, <|user|>, <|assistant|> tokens match Phi-3's format,
    which is also compatible with most PEFT / TRL SFTTrainer setups via
    the 'text' field in a plain dict.
    """
    text = (
        f"<|system|>{SYSTEM_PROMPT}<|end|>"
        f"<|user|>Review this pull request diff:\n\n{diff}<|end|>"
        f"<|assistant|>{review}<|end|>"
    )
    return {"text": text}


def write_jsonl(path: Path, examples: list) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(examples):>8,} examples → {path}")


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

print("=" * 65)
cfg_label = f"config='{DATASET_CONFIG}'" if DATASET_CONFIG else "default config"
print(f"Loading  {DATASET_NAME}  ({cfg_label}) …")
print("=" * 65)

try:
    # Pass config only when one is specified — fasterinnerlooper/codereviewer
    # is a flat single-config dataset and errors if given an unexpected name.
    if DATASET_CONFIG:
        raw = load_dataset(DATASET_NAME, DATASET_CONFIG)
    else:
        raw = load_dataset(DATASET_NAME)
except Exception as exc:
    sys.exit(f"\nFailed to load dataset: {exc}\n"
             "Make sure you have network access to huggingface.co and that\n"
             "the 'datasets' package is installed (pip install datasets).")

print(f"  Available splits: { {k: len(v) for k, v in raw.items()} }\n")

# ---------------------------------------------------------------------------
# 2. Filter and format (pool all original splits, re-split ourselves)
# ---------------------------------------------------------------------------

# We pool all splits so our 80/10/10 ratio applies to the whole corpus.
# The original train/valid/test boundaries are intentionally discarded.

kept: list         = []
skip_counts: dict  = {}   # reason → count
skipped_missing    = 0

# Track which field names were actually used (for the summary)
diff_field_name   = None
review_field_name = None

print("Filtering and formatting …")

for split_name, split in raw.items():
    for row in tqdm(split, desc=f"  {split_name}", total=len(split), unit="ex"):

        d_name, diff   = resolve_field(row, DIFF_CANDIDATES)
        r_name, review = resolve_field(row, REVIEW_CANDIDATES)

        # Record field names from the first row that has them
        if d_name and diff_field_name is None:
            diff_field_name = d_name
        if r_name and review_field_name is None:
            review_field_name = r_name

        if diff is None or review is None:
            # At least one required field is absent entirely — log and skip
            skipped_missing += 1
            missing = []
            if diff is None:
                missing.append(f"diff (tried {DIFF_CANDIDATES})")
            if review is None:
                missing.append(f"review (tried {REVIEW_CANDIDATES})")
            print(f"    WARNING: missing {', '.join(missing)} — skipping row", file=sys.stderr)
            continue

        valid, reason = is_valid(diff, review)
        if not valid:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            continue

        kept.append(format_example(diff, review))

print()

# ---------------------------------------------------------------------------
# 3. Shuffle and split 80 / 10 / 10 with fixed seed
# ---------------------------------------------------------------------------

random.seed(RANDOM_SEED)
random.shuffle(kept)

n       = len(kept)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)

train_data = kept[:n_train]
val_data   = kept[n_train : n_train + n_val]
test_data  = kept[n_train + n_val :]

# ---------------------------------------------------------------------------
# 4. Write .jsonl splits
# ---------------------------------------------------------------------------

print("Writing splits …")
write_jsonl(SPLITS_DIR / "train.jsonl",      train_data)
write_jsonl(SPLITS_DIR / "validation.jsonl", val_data)
write_jsonl(SPLITS_DIR / "test.jsonl",       test_data)

# ---------------------------------------------------------------------------
# 5. Save 10 sample examples for visual inspection
# ---------------------------------------------------------------------------

SAMPLE_PATH = SPLITS_DIR / "sample_examples.txt"
with SAMPLE_PATH.open("w", encoding="utf-8") as fh:
    for i, ex in enumerate(train_data[:10]):
        fh.write(f"{'=' * 65}\nExample {i + 1}\n{'=' * 65}\n")
        fh.write(ex["text"])
        fh.write("\n\n")

print(f"  Wrote 10 samples           → {SAMPLE_PATH}")

# ---------------------------------------------------------------------------
# 6. Summary report
# ---------------------------------------------------------------------------

total_raw = sum(len(v) for v in raw.values())

print()
print("=" * 65)
print("FILTERING SUMMARY")
print("=" * 65)
print(f"  Raw examples (all original splits) : {total_raw:>8,}")
print(f"  Skipped — missing required fields  : {skipped_missing:>8,}")
for reason, count in sorted(skip_counts.items(), key=lambda x: -x[1]):
    label = {
        "diff_too_short"    : "Skipped — diff < 20 chars      ",
        "review_too_short"  : "Skipped — review < 10 chars    ",
        "review_is_url"     : "Skipped — review is bare URL   ",
        "review_single_word": "Skipped — review is single word",
    }.get(reason, f"Skipped — {reason:<26}")
    print(f"  {label} : {count:>8,}")
print(f"  {'─' * 44}")
print(f"  Kept after filtering               : {n:>8,}")
print()
print("FINAL SPLIT COUNTS  (seed={})".format(RANDOM_SEED))
print(f"  train      ({TRAIN_RATIO:.0%}) : {len(train_data):>8,}")
print(f"  validation ({VAL_RATIO:.0%}) : {len(val_data):>8,}")
print(f"  test       ({1-TRAIN_RATIO-VAL_RATIO:.0%}) : {len(test_data):>8,}")
print()
print(f"  Diff field used   : {diff_field_name!r}")
print(f"  Review field used : {review_field_name!r}")
print(f"  Splits directory  : {SPLITS_DIR}/")
