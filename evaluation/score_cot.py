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

# Backtick-quoted code terms  (e.g. `my_func`, `variable_name`)
_BACKTICK_RE = re.compile(r"`[^`]+`")

# snake_case identifiers (at least one underscore with word chars on both sides)
_SNAKE_CASE_RE = re.compile(r"\b[a-z_]\w*_\w+\b")

_LINE_NUMBER_RE = re.compile(
    r"\blines?\s*\d+"                 # "line 42", "lines 10-20"
    r"|\bL\d+\b"                      # "L42"
    r"|\bline\s*[:#]\s*\d+"          # "line: 42"
)

# Generic technical vocabulary — present in almost every review, not specific
_TECH_GENERIC_RE = re.compile(
    r"\b(function|method|class|variable|parameter|argument|import|module"
    r"|exception|error|return|type|interface|object|property|attribute"
    r"|loop|condition|statement|scope|decorator|constructor|callback)\b",
    re.IGNORECASE,
)

# Imperative verbs the user asked to target (de-duplicated via set later)
_IMPERATIVE_VERBS_RE = re.compile(
    r"\b(change|fix|rename|add|remove|delete|consider|use|replace|refactor"
    r"|extract|move|avoid|ensure|validate|handle|catch|update|convert"
    r"|rewrite|simplify|sanitize|sanitise|escape|wrap|raise|check"
    r"|initialize|initialise|inject|call|import)\b",
    re.IGNORECASE,
)

# "change X to Y" / "replace X with Y" — highest-confidence actionable pattern
_EXPLICIT_ACTION_RE = re.compile(
    r"\b(change|replace|rename|convert)\b.{1,60}\b(to|with|into)\b",
    re.IGNORECASE,
)

# Reasoning language: signals that the response explains WHY, not just WHAT
_REASONING_RE = re.compile(
    r"\b(because|since|therefore|however|instead|otherwise|which means"
    r"|this will|this may|this can|could lead to|this causes|as a result"
    r"|this introduces|this prevents|in order to|so that|unless|"
    r"this allows|this ensures|this avoids|this reduces)\b",
    re.IGNORECASE,
)


def score_specificity(text: str) -> int:
    """
    Presence-based — rewards feature TYPES, not raw identifier count.
    This avoids penalising concise reviews for being shorter.

    5 — line numbers referenced (highest precision)
    4 — 2+ backtick-quoted code terms, or backtick term + snake_case identifier
    3 — exactly 1 backtick-quoted term, or 2+ distinct snake_case identifiers
    2 — generic technical vocabulary only (function/method/class/variable etc.)
         or exactly 1 snake_case identifier
    1 — no identifiable code references or technical terms
    """
    if _LINE_NUMBER_RE.search(text):
        return 5

    backtick_terms = _BACKTICK_RE.findall(text)
    snake_terms    = set(_SNAKE_CASE_RE.findall(text))

    if len(backtick_terms) >= 2 or (backtick_terms and snake_terms):
        return 4
    if backtick_terms or len(snake_terms) >= 2:
        return 3
    if snake_terms or _TECH_GENERIC_RE.search(text):
        return 2
    return 1


def score_actionability(text: str) -> int:
    """
    Uses DISTINCT imperative verb types, not raw occurrence count.
    Repeating the same verb many times does not inflate the score.

    5 — explicit "change X to Y" pattern, or 4+ distinct action verb types
    4 — 3 distinct action verb types
    3 — 2 distinct action verb types
    2 — 1 distinct action verb type
    1 — no imperative verbs (identifies issues but gives no direction)
    """
    if _EXPLICIT_ACTION_RE.search(text):
        return 5

    distinct_verbs = {m.lower() for m in _IMPERATIVE_VERBS_RE.findall(text)}
    n = len(distinct_verbs)

    if n >= 4:
        return 5
    if n >= 3:
        return 4
    if n >= 2:
        return 3
    if n >= 1:
        return 2
    return 1


def score_detail(text: str) -> int:
    """
    Combines normalised word count (capped at 200) with reasoning-language
    presence.  A verbose response with no reasoning scores the same as a
    concise one that clearly explains *why* each issue matters.

    Word-count component (0-3):
        < 30 words  → 0
        30-49 words → 1
        50-99 words → 2
        ≥100 words  → 3  (capped — additional length beyond 200 is not rewarded)

    Reasoning component (0-2):
        0 reasoning patterns → 0
        1-2 patterns         → 1
        3+ patterns          → 2

    Final score = max(1, word_score + reasoning_score), clamped to 1-5.
    """
    words = len(text.split())

    if words >= 100:
        wc_score = 3
    elif words >= 50:
        wc_score = 2
    elif words >= 30:
        wc_score = 1
    else:
        wc_score = 0

    reasoning_hits = len(_REASONING_RE.findall(text))
    reasoning_score = 2 if reasoning_hits >= 3 else (1 if reasoning_hits >= 1 else 0)

    return max(1, min(5, wc_score + reasoning_score))


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
