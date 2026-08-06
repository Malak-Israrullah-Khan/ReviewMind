"""
Day 2 – Dataset Exploration
Loads microsoft/CodeReviewer from Hugging Face and prints structure,
sample examples, and basic statistics. Saves a summary to
data/dataset_summary.txt. No cleaning or transformation here.
"""

import os
import sys
import textwrap
from collections import Counter
from pathlib import Path

from datasets import load_dataset

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

print("=" * 70)
print("Loading microsoft/CodeReviewer …")
print("=" * 70)

# The dataset ships two named configs:
#   'code-review'  – given a diff, generate a review comment
#   'code-change'  – given diff + comment, generate the revised code
# We load both so we can explore the full data picture.
DATASET_NAME = "microsoft/CodeReviewer"
CONFIGS = ["code-review", "code-change"]

datasets = {}
for config in CONFIGS:
    try:
        datasets[config] = load_dataset(DATASET_NAME, config)
        print(f"  ✓ loaded config '{config}'")
    except Exception as exc:
        print(f"  ✗ could not load config '{config}': {exc}")

if not datasets:
    # Fall back: try loading the default config (no name)
    try:
        datasets["default"] = load_dataset(DATASET_NAME)
        print("  ✓ loaded default config")
    except Exception as exc:
        sys.exit(f"Failed to load dataset: {exc}")

# ---------------------------------------------------------------------------
# 2. Structure
# ---------------------------------------------------------------------------

summary_lines = []

def h(title):
    line = f"\n{'=' * 70}\n{title}\n{'=' * 70}"
    print(line)
    summary_lines.append(line)

def log(text=""):
    print(text)
    summary_lines.append(str(text))

h("DATASET STRUCTURE")

for config_name, ds in datasets.items():
    log(f"\nConfig: '{config_name}'")
    log(f"  Splits : {list(ds.keys())}")
    for split_name, split in ds.items():
        log(f"  {split_name:10s}  rows={len(split):>7,}  columns={split.column_names}")

# ---------------------------------------------------------------------------
# Field-level notes built from dynamic inspection.
# These comments describe what each field contains once the data loads.
# They are printed alongside the first example of each config.
#
# CodeReviewer paper (Li et al., 2022) describes the two tasks:
#
#  code-review  columns:
#    msg    – the human reviewer's comment left on the pull request
#    patch  – the unified diff of the code change being reviewed
#    label  – binary: 1 if the patch received a comment, 0 otherwise
#    url    – link back to the original GitHub pull request
#    lang   – programming language detected for the file
#    input  – same as patch but pre-tokenised / formatted for the model
#    target – same as msg; used as the generation target during training
#
#  code-change  columns:
#    input  – concatenation of the diff and the review comment
#    target – the revised code after addressing the review comment
#    url    – link back to the original GitHub pull request
#    lang   – programming language
#
# Actual column names are printed dynamically below in case they differ.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3. Full examples (3 per config × split)
# ---------------------------------------------------------------------------

h("SAMPLE EXAMPLES")

EXAMPLES_TO_SHOW = 3

for config_name, ds in datasets.items():
    for split_name, split in ds.items():
        log(f"\n--- Config '{config_name}' / Split '{split_name}' ---")
        for i in range(min(EXAMPLES_TO_SHOW, len(split))):
            log(f"\n  [Example {i}]")
            example = split[i]
            for col, val in example.items():
                val_str = str(val)
                # Wrap long values so they stay readable in the terminal
                if len(val_str) > 300:
                    val_str = val_str[:297] + "…"
                log(f"    {col}: {val_str}")
        # Only show examples for train split to avoid redundancy
        break

# ---------------------------------------------------------------------------
# 4. Basic statistics
# ---------------------------------------------------------------------------

h("BASIC STATISTICS")

def avg_chars(split, col):
    """Mean character count for a column that contains strings."""
    lengths = [len(str(row[col])) for row in split if row.get(col)]
    return sum(lengths) / len(lengths) if lengths else 0.0

def token_estimate(char_count, chars_per_token=4):
    """Rough token count: ~4 chars per token for code."""
    return char_count / chars_per_token

for config_name, ds in datasets.items():
    log(f"\n── Config: '{config_name}' ──")
    for split_name, split in ds.items():
        log(f"\n  Split: {split_name}  ({len(split):,} rows)")

        # Identify likely diff and comment columns by name heuristics
        cols = split.column_names
        diff_candidates   = [c for c in cols if c in ("patch", "input", "diff", "code")]
        review_candidates = [c for c in cols if c in ("msg", "target", "comment", "review")]
        lang_candidates   = [c for c in cols if c in ("lang", "language", "programming_language")]

        # Lengths of diff / code columns
        for col in diff_candidates:
            avg = avg_chars(split, col)
            log(f"    avg len '{col}'   : {avg:>8.0f} chars  (~{token_estimate(avg):.0f} tokens)")

        # Lengths of review / comment columns
        for col in review_candidates:
            avg = avg_chars(split, col)
            log(f"    avg len '{col}'   : {avg:>8.0f} chars  (~{token_estimate(avg):.0f} tokens)")

        # Language distribution
        for col in lang_candidates:
            all_langs = [str(row[col]) for row in split if row.get(col)]
            dist = Counter(all_langs).most_common(15)
            log(f"\n    Language distribution ('{col}', top 15):")
            for lang, count in dist:
                pct = 100 * count / len(all_langs)
                log(f"      {lang:<25s} {count:>7,}  ({pct:5.1f}%)")

        # Label distribution (code-review has a binary label)
        if "label" in cols:
            labels = [str(row["label"]) for row in split]
            dist = Counter(labels).most_common()
            log(f"\n    Label distribution:")
            for label, count in dist:
                pct = 100 * count / len(labels)
                log(f"      label={label}  {count:>7,}  ({pct:5.1f}%)")

        # Min / max diff length to understand the range
        for col in diff_candidates[:1]:   # just the primary diff col
            lengths = [len(str(row[col])) for row in split if row.get(col)]
            if lengths:
                log(f"\n    '{col}' length range:")
                log(f"      min={min(lengths):,}  max={max(lengths):,}  median={sorted(lengths)[len(lengths)//2]:,}")

        # Rough percentage of empty / null diffs
        for col in diff_candidates[:1]:
            empty = sum(1 for row in split if not row.get(col) or str(row[col]).strip() == "")
            log(f"    empty '{col}' rows : {empty} ({100*empty/len(split):.1f}%)")

# ---------------------------------------------------------------------------
# 5. Save summary
# ---------------------------------------------------------------------------

SUMMARY_PATH = Path(__file__).parent / "dataset_summary.txt"
SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n")
print(f"\nSummary written to {SUMMARY_PATH}")
