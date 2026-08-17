"""
ReviewMind – CoT Scoring Script
Loads evaluation/results/cot_comparison.json and scores each plain and CoT
response on three heuristic criteria (1-5 each):

  specificity   — references specific variable names, function names, or
                  line numbers rather than speaking in generalities
  actionability — contains imperative instructions telling the developer
                  exactly what to change or fix
  detail        — length and depth: more words, more distinct points scored

Scores are heuristic (regex + word-count based) — fast, offline, no LLM
needed.  Results are printed as a comparison table and saved to
evaluation/results/cot_scores.json.

Usage:
    python evaluation/score_cot.py
    python evaluation/score_cot.py --input path/to/cot_comparison.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RESULTS_DIR   = PROJECT_ROOT / "evaluation" / "results"
DEFAULT_INPUT = RESULTS_DIR / "cot_comparison.json"
DEFAULT_OUTPUT = RESULTS_DIR / "cot_scores.json"

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

# Patterns that suggest the review names specific code entities
_CODE_IDENTIFIER_RE = re.compile(
    r"`[^`]+`"                        # backtick-quoted code
    r"|[a-zA-Z_]\w*\s*\("            # function calls: foo(
    r"|\b[a-z_]\w*_\w+"              # snake_case identifiers
    r"|\b[a-z][a-zA-Z0-9]{2,}\b"     # camelCase-ish identifiers (≥3 chars)
)

_LINE_NUMBER_RE = re.compile(
    r"\blines?\s*\d+"                 # "line 42", "lines 10-20"
    r"|\bL\d+\b"                      # "L42"
    r"|\bline\s*[:#]\s*\d+"          # "line: 42"
)

# Verbs that indicate concrete, actionable feedback
_ACTION_VERBS = re.compile(
    r"\b(rename|refactor|extract|replace|remove|delete|add|move|fix|change"
    r"|update|convert|use|avoid|ensure|check|validate|handle|catch|return"
    r"|consider|rewrite|simplify|split|merge|initialise|initialize)\b",
    re.IGNORECASE,
)

# Patterns like "change X to Y", "replace X with Y" — high-confidence actions
_EXPLICIT_ACTION_RE = re.compile(
    r"\b(change|replace|rename|convert)\b.{1,60}\b(to|with|into)\b",
    re.IGNORECASE,
)


def score_specificity(text: str) -> int:
    """
    1 — no identifiable code references
    2 — a handful of generic keyword mentions
    3 — multiple identifiers or one backtick-quoted term
    4 — several backtick refs or explicit function/variable names
    5 — line numbers referenced, or 5+ distinct code identifiers
    """
    line_hits = len(_LINE_NUMBER_RE.findall(text))
    if line_hits >= 1:
        return 5

    code_hits = _CODE_IDENTIFIER_RE.findall(text)
    n = len(set(code_hits))   # unique matches
    if n >= 5:
        return 4
    if n >= 3:
        return 3
    if n >= 1:
        return 2
    return 1


def score_actionability(text: str) -> int:
    """
    1 — no action words
    2 — 1 action verb
    3 — 2-3 action verbs
    4 — 4-5 action verbs or one explicit change pattern
    5 — 6+ action verbs or multiple explicit "change X to Y" patterns
    """
    explicit = len(_EXPLICIT_ACTION_RE.findall(text))
    if explicit >= 2:
        return 5

    verb_hits = len(_ACTION_VERBS.findall(text))
    if verb_hits >= 6 or explicit >= 1:
        return 5
    if verb_hits >= 4:
        return 4
    if verb_hits >= 2:
        return 3
    if verb_hits >= 1:
        return 2
    return 1


def score_detail(text: str) -> int:
    """
    Based on word count — a simple but reliable proxy for depth.
    1 — < 20 words
    2 — 20-49 words
    3 — 50-99 words
    4 — 100-199 words
    5 — 200+ words
    """
    words = len(text.split())
    if words >= 200:
        return 5
    if words >= 100:
        return 4
    if words >= 50:
        return 3
    if words >= 20:
        return 2
    return 1


def score_response(text: str) -> dict:
    return {
        "specificity":    score_specificity(text),
        "actionability":  score_actionability(text),
        "detail":         score_detail(text),
    }


def total(scores: dict) -> float:
    return scores["specificity"] + scores["actionability"] + scores["detail"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score plain vs CoT responses in cot_comparison.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", default=str(DEFAULT_INPUT),
        help="Path to cot_comparison.json produced by cot_eval.py.",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help="Path to write cot_scores.json.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _avg(values):
    return sum(values) / len(values) if values else 0.0


def print_summary(scored: list) -> None:
    criteria = ["specificity", "actionability", "detail"]

    plain_scores = {c: [r["plain_scores"][c] for r in scored] for c in criteria}
    cot_scores   = {c: [r["cot_scores"][c]   for r in scored] for c in criteria}

    plain_totals = [total(r["plain_scores"]) for r in scored]
    cot_totals   = [total(r["cot_scores"])   for r in scored]

    col_w = 14

    print("\n" + "=" * 62)
    print("  SCORING SUMMARY  (heuristic, 1-5 per criterion)")
    print("=" * 62)
    print(f"  {'Criterion':<18} {'Plain avg':>{col_w}} {'CoT avg':>{col_w}} {'Δ':>{col_w}}")
    print("  " + "─" * 58)

    for c in criteria:
        p = _avg(plain_scores[c])
        q = _avg(cot_scores[c])
        delta = q - p
        bar = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        print(f"  {c.capitalize():<18} {p:>{col_w}.2f} {q:>{col_w}.2f} {bar} {abs(delta):>6.2f}")

    print("  " + "─" * 58)
    p_tot = _avg(plain_totals)
    c_tot = _avg(cot_totals)
    delta_tot = c_tot - p_tot
    bar = "▲" if delta_tot > 0 else ("▼" if delta_tot < 0 else "=")
    print(f"  {'TOTAL (max 15)':<18} {p_tot:>{col_w}.2f} {c_tot:>{col_w}.2f} {bar} {abs(delta_tot):>6.2f}")
    print("=" * 62)

    # Score distribution
    print("\n  Score distributions  (plain | CoT):")
    for c in criteria:
        p_dist = {v: plain_scores[c].count(v) for v in range(1, 6)}
        q_dist = {v: cot_scores[c].count(v)   for v in range(1, 6)}
        print(f"\n  {c.capitalize()}:")
        for v in range(1, 6):
            bar_p = "█" * p_dist[v]
            bar_q = "█" * q_dist[v]
            print(f"    {v}  plain {bar_p:<12} ({p_dist[v]:>3})  |  cot {bar_q:<12} ({q_dist[v]:>3})")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        sys.exit(
            f"\nInput file not found: {input_path}\n"
            "Run  python evaluation/cot_eval.py  first."
        )

    with input_path.open(encoding="utf-8") as fh:
        results = json.load(fh)

    print(f"\nLoaded {len(results)} results from {input_path}")
    print("Scoring …")

    scored = []
    for r in results:
        plain_s = score_response(r["plain_response"])
        cot_s   = score_response(r["cot_response"])
        scored.append({
            "index":        r["index"],
            "diff_preview": r["diff"][:200],
            "plain_scores": plain_s,
            "plain_total":  total(plain_s),
            "cot_scores":   cot_s,
            "cot_total":    total(cot_s),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(scored, fh, indent=2, ensure_ascii=False)
    print(f"Scores saved → {output_path}")

    print_summary(scored)


if __name__ == "__main__":
    main()
