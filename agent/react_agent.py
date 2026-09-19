"""
ReviewMind – ReAct Agent
Implements a ReAct (Reasoning + Acting) loop that reviews code diffs using
four tools, then produces a structured final review.

Tools
-----
file_fetcher(filename)              Returns mock file content for context.
dependency_checker(library,version) Looks up known CVEs in a hardcoded dict.
git_history(function_name)          Returns mock commit history for a function.
documentation_lookup(query)         Queries ChromaDB (rag/vectorstore/) for
                                    relevant coding guidelines.

ReAct loop format (parsed from model output)
--------------------------------------------
Thought: <reasoning step>
Action: <tool_name>
Action Input: <json args>
Observation: <tool result injected by the harness>
...
Final Answer: <structured JSON review>

Usage
-----
    python agent/react_agent.py
    python agent/react_agent.py --n-examples 3 --max-iterations 6
    python agent/react_agent.py --no-model   # use a stub LLM for smoke-testing
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO = "Malak-Israr/reviewmind-lora"
TEST_JSONL   = PROJECT_ROOT / "data" / "splits" / "test.jsonl"
OUTPUT_DIR   = PROJECT_ROOT / "evaluation" / "results"
OUTPUT_FILE  = OUTPUT_DIR / "agent_results.json"

DEFAULT_N_EXAMPLES    = 5
DEFAULT_MAX_ITER      = 6        # max tool calls before forcing Final Answer
DEFAULT_MAX_NEW_TOKENS = 300

# ---------------------------------------------------------------------------
# CVE knowledge base (dependency_checker)
# ---------------------------------------------------------------------------

_CVE_DB: dict[str, list[dict]] = {
    "requests": [
        {"below": "2.28.0", "cve": "CVE-2023-32681",
         "desc": "Proxy-Authorization header leaked to redirect target."},
        {"below": "2.20.0", "cve": "CVE-2018-18074",
         "desc": "Credentials forwarded over HTTP after HTTPS redirect."},
    ],
    "django": [
        {"below": "3.2.13", "cve": "CVE-2022-28347",
         "desc": "SQL injection via QuerySet.explain() on PostgreSQL."},
        {"below": "4.0.4",  "cve": "CVE-2022-28346",
         "desc": "SQL injection in annotate(), aggregate(), extra()."},
        {"below": "2.2.28", "cve": "CVE-2022-22818",
         "desc": "Possible XSS via {% debug %} template tag."},
    ],
    "pillow": [
        {"below": "9.0.0", "cve": "CVE-2022-22817",
         "desc": "Buffer overflow via crafted TIFF file."},
        {"below": "8.3.2", "cve": "CVE-2021-34552",
         "desc": "Buffer overflow in Convert.ImagingLibTiffDecode."},
    ],
    "pyyaml": [
        {"below": "6.0",   "cve": "CVE-2020-1747",
         "desc": "Arbitrary code execution via full_load() on untrusted input."},
    ],
    "cryptography": [
        {"below": "3.3.2", "cve": "CVE-2020-36242",
         "desc": "Buffer overflow in symmetric encryption routines."},
        {"below": "41.0.0","cve": "CVE-2023-23931",
         "desc": "Memory corruption in Rust bindings."},
    ],
    "numpy": [
        {"below": "1.22.0","cve": "CVE-2021-33430",
         "desc": "Buffer overflow in PyArray_NewLikeArray."},
    ],
    "flask": [
        {"below": "2.0.0", "cve": "CVE-2018-1000656",
         "desc": "Path traversal via crafted URL in static file serving."},
    ],
    "lxml": [
        {"below": "4.9.1", "cve": "CVE-2022-2309",
         "desc": "NULL pointer dereference in lxml.etree.tostring()."},
    ],
    "paramiko": [
        {"below": "2.10.1","cve": "CVE-2022-24302",
         "desc": "Race condition in private key file creation (world-readable)."},
    ],
    "urllib3": [
        {"below": "1.26.5","cve": "CVE-2021-33503",
         "desc": "ReDoS via malformed URL authority component."},
        {"below": "2.0.2", "cve": "CVE-2023-43804",
         "desc": "Cookie leak to third-party redirect targets."},
    ],
    "sqlalchemy": [
        {"below": "1.4.0", "cve": "CVE-2019-7164",
         "desc": "SQL injection via crafted select/order_by parameter."},
    ],
    "jinja2": [
        {"below": "3.1.2", "cve": "CVE-2024-22195",
         "desc": "XSS via HTML attribute injection in xmlattr filter."},
    ],
    "setuptools": [
        {"below": "65.5.1","cve": "CVE-2022-40897",
         "desc": "ReDoS in package_index via crafted HTML."},
    ],
}


# ---------------------------------------------------------------------------
# Mock data for file_fetcher and git_history
# ---------------------------------------------------------------------------

_MOCK_FILES: dict[str, str] = {
    "config.py": (
        "# Application configuration\n"
        "DEBUG = True\n"
        "SECRET_KEY = 'hardcoded-secret-12345'\n"
        "DATABASE_URL = 'postgresql://user:password@localhost/mydb'\n"
        "ALLOWED_HOSTS = ['*']\n"
    ),
    "requirements.txt": (
        "requests==2.25.0\n"
        "django==3.1.0\n"
        "pillow==8.0.0\n"
        "flask==1.1.2\n"
        "pyyaml==5.4.0\n"
    ),
    "utils.py": (
        "import os\n\n"
        "def execute_command(cmd):\n"
        "    return os.system(cmd)  # potential command injection\n\n"
        "def read_file(path):\n"
        "    with open(path) as f:\n"
        "        return f.read()  # no path traversal protection\n"
    ),
    "models.py": (
        "from django.db import models\n\n"
        "class User(models.Model):\n"
        "    username = models.CharField(max_length=150)\n"
        "    password = models.CharField(max_length=128)  # plain text\n"
        "    email = models.EmailField()\n\n"
        "class Order(models.Model):\n"
        "    user = models.ForeignKey(User, on_delete=models.CASCADE)\n"
        "    total = models.DecimalField(max_digits=10, decimal_places=2)\n"
    ),
    "tests.py": (
        "# No test file found for this module.\n"
        "# Unit tests are missing.\n"
    ),
}

_MOCK_GIT_HISTORY: dict[str, list[dict]] = {
    "process_payment": [
        {"hash": "a1b2c3d", "author": "alice", "date": "2024-01-15",
         "message": "fix: handle decimal rounding in payment totals"},
        {"hash": "e4f5g6h", "author": "bob",   "date": "2023-11-02",
         "message": "feat: add stripe webhook verification"},
        {"hash": "i7j8k9l", "author": "alice", "date": "2023-08-20",
         "message": "refactor: extract payment processor class"},
    ],
    "authenticate_user": [
        {"hash": "m1n2o3p", "author": "charlie", "date": "2024-02-10",
         "message": "security: migrate to bcrypt password hashing"},
        {"hash": "q4r5s6t", "author": "alice",   "date": "2023-12-05",
         "message": "fix: prevent timing attack in login comparison"},
        {"hash": "u7v8w9x", "author": "bob",     "date": "2023-09-14",
         "message": "feat: add two-factor authentication support"},
    ],
    "parse_input": [
        {"hash": "y1z2a3b", "author": "dave",  "date": "2024-03-01",
         "message": "fix: sanitize HTML in user-submitted input"},
        {"hash": "c4d5e6f", "author": "alice", "date": "2024-01-20",
         "message": "fix: escape SQL special chars in search query"},
    ],
    "upload_file": [
        {"hash": "g7h8i9j", "author": "eve",   "date": "2024-02-28",
         "message": "security: restrict allowed MIME types for uploads"},
        {"hash": "k1l2m3n", "author": "frank", "date": "2023-10-11",
         "message": "feat: add file size limit and virus scan hook"},
    ],
}

_DEFAULT_GIT_HISTORY = [
    {"hash": "a0b1c2d", "author": "unknown", "date": "2024-01-01",
     "message": "chore: initial commit"},
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def file_fetcher(filename: str) -> str:
    filename = filename.strip().lstrip("./")
    content = _MOCK_FILES.get(filename)
    if content:
        return f"=== {filename} ===\n{content}"
    return (
        f"=== {filename} ===\n"
        f"# File not in mock store \u2014 showing placeholder.\n"
        f"# Contains {len(filename) * 7} lines of application code.\n"
        f"# Last modified: 2024-03-15\n"
    )


def dependency_checker(library: str, version: str) -> str:
    lib = library.strip().lower()
    ver = version.strip().lstrip("v=~^")

    records = _CVE_DB.get(lib)
    if records is None:
        return f"No known CVEs found for '{library}' in the database."

    def _parse_ver(v: str) -> tuple[int, ...]:
        parts = re.split(r"[.\-]", v)
        result = []
        for p in parts:
            m = re.match(r"(\d+)", p)
            result.append(int(m.group(1)) if m else 0)
        return tuple(result[:3])

    try:
        query_v = _parse_ver(ver)
    except Exception:
        return f"Could not parse version '{version}'. Provide a semver string."

    hits = []
    for rec in records:
        threshold = _parse_ver(rec["below"])
        if query_v < threshold:
            hits.append(
                f"  {rec['cve']} (affects < {rec['below']}): {rec['desc']}"
            )

    if hits:
        return (
            f"VULNERABLE: {library} {version} has {len(hits)} known CVE(s):\n"
            + "\n".join(hits)
        )
    return f"OK: {library} {version} \u2014 no known CVEs in the database."


def git_history(function_name: str) -> str:
    fn = function_name.strip()
    history = _MOCK_GIT_HISTORY.get(fn, _DEFAULT_GIT_HISTORY)
    lines = [f"Git history for '{fn}' ({len(history)} commit(s)):"]
    for c in history:
        lines.append(
            f"  [{c['date']}] {c['hash'][:7]}  {c['author']}: {c['message']}"
        )
    return "\n".join(lines)


def documentation_lookup(query: str) -> str:
    """Query the ChromaDB vector store for relevant guideline chunks."""
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from rag.retriever import Retriever
        retriever = Retriever()
        chunks = retriever.retrieve(query, top_k=2)
        if not chunks:
            return "No relevant documentation found."
        return retriever.format_context(chunks, header=False)
    except SystemExit as exc:
        return f"[documentation_lookup unavailable: {exc}]"
    except Exception as exc:
        return f"[documentation_lookup error: {exc}]"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS: dict[str, dict] = {
    "file_fetcher": {
        "fn":   file_fetcher,
        "args": ["filename"],
        "desc": "Fetch the content of a source file by name (e.g. 'utils.py').",
    },
    "dependency_checker": {
        "fn":   dependency_checker,
        "args": ["library", "version"],
        "desc": "Check a library/version pair for known CVEs.",
    },
    "git_history": {
        "fn":   git_history,
        "args": ["function_name"],
        "desc": "Return recent commit history for a named function.",
    },
    "documentation_lookup": {
        "fn":   documentation_lookup,
        "args": ["query"],
        "desc": "Search the coding-guidelines vector store for relevant context.",
    },
}


def call_tool(name: str, raw_args: dict) -> str:
    entry = TOOLS.get(name)
    if entry is None:
        return f"[Unknown tool '{name}'. Available: {list(TOOLS)}]"
    try:
        return entry["fn"](**{k: raw_args.get(k, "") for k in entry["args"]})
    except Exception as exc:
        return f"[Tool '{name}' raised {type(exc).__name__}: {exc}]"


# ---------------------------------------------------------------------------
# ReAct prompt
# ---------------------------------------------------------------------------

_TOOL_DESCRIPTIONS = "\n".join(
    f"  {name}({', '.join(info['args'])}): {info['desc']}"
    for name, info in TOOLS.items()
)

_REACT_SYSTEM = f"""You are a senior software engineer performing a code review using a ReAct loop.

Available tools:
{_TOOL_DESCRIPTIONS}

Rules:
1. Always start with a Thought.
2. To call a tool write exactly:
   Action: <tool_name>
   Action Input: {{"key": "value"}}
3. After each Observation, write another Thought.
4. When you have gathered enough information, write:
   Final Answer: <JSON with keys: summary, issues, security_concerns, code_quality, recommendations>
5. Use at most {DEFAULT_MAX_ITER} tool calls before writing Final Answer."""

_REACT_USER_TEMPLATE = """\
Review this pull request diff using the ReAct loop.

Diff:
{diff}

Begin:
Thought:"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_ACTION_RE       = re.compile(r"Action\s*:\s*(\w+)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action\s+Input\s*:\s*(\{.*?\})", re.DOTALL | re.IGNORECASE)
_FINAL_RE        = re.compile(r"Final\s+Answer\s*:\s*(.*)", re.DOTALL | re.IGNORECASE)
_THOUGHT_RE      = re.compile(r"Thought\s*:\s*(.*?)(?=Action|Final Answer|$)", re.DOTALL | re.IGNORECASE)


def _parse_step(text: str) -> dict:
    """Extract the next action OR final answer from a model response chunk."""
    final_m = _FINAL_RE.search(text)
    if final_m:
        return {"type": "final", "content": final_m.group(1).strip()}

    action_m = _ACTION_RE.search(text)
    if action_m:
        tool_name = action_m.group(1).strip()
        args = {}
        input_m = _ACTION_INPUT_RE.search(text)
        if input_m:
            try:
                args = json.loads(input_m.group(1))
            except json.JSONDecodeError:
                raw = input_m.group(1)
                for kv in re.finditer(r'"?(\w+)"?\s*:\s*"([^"]*)"', raw):
                    args[kv.group(1)] = kv.group(2)
        return {"type": "action", "tool": tool_name, "args": args}

    thought_m = _THOUGHT_RE.search(text)
    if thought_m:
        return {"type": "thought", "content": thought_m.group(1).strip()}

    return {"type": "unknown", "content": text.strip()}


def _extract_final_review(text: str) -> dict:
    """
    Try to parse the model's Final Answer as JSON.
    Fall back to wrapping the raw text in a minimal dict.
    """
    json_m = re.search(r"\{.*\}", text, re.DOTALL)
    if json_m:
        try:
            return json.loads(json_m.group(0))
        except json.JSONDecodeError:
            pass
    return {
        "summary": text[:1000].strip(),
        "issues": [],
        "security_concerns": [],
        "code_quality": "See summary.",
        "recommendations": [],
    }


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


def generate_step(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Stub model (--no-model smoke-test)
# ---------------------------------------------------------------------------

class _StubModel:
    """Deterministic stub that cycles through a fixed ReAct script."""

    def __init__(self):
        self.device = "cpu"
        self._step = 0

    def eval(self):
        return self

    def generate(self, **_):
        return [[]]


class _StubTokenizer:
    eos_token_id = 0
    padding_side = "left"

    def __call__(self, text, **_):
        return {"input_ids": [[0]]}

    def decode(self, ids, **_):
        return ""


_STUB_SCRIPT = [
    'Thought: I should check if there are any vulnerable dependencies.\nAction: dependency_checker\nAction Input: {"library": "requests", "version": "2.25.0"}',
    'Thought: Let me look at the relevant style guidelines.\nAction: documentation_lookup\nAction Input: {"query": "error handling best practices"}',
    'Thought: Let me check the git history for context.\nAction: git_history\nAction Input: {"function_name": "process_payment"}',
    (
        'Thought: I have enough information to write the final review.\n'
        'Final Answer: {"summary": "The diff introduces a payment processing function with several issues.", '
        '"issues": ["Missing error handling for network timeouts", "No input validation on amount parameter"], '
        '"security_concerns": ["requests 2.25.0 has CVE-2023-32681 (proxy header leak)"], '
        '"code_quality": "Code follows PEP 8 but lacks docstrings and type hints.", '
        '"recommendations": ["Upgrade requests to >= 2.28.0", "Add try/except around network calls", "Validate and sanitise all inputs"]}'
    ),
]


def generate_step_stub(model, tokenizer, prompt: str, _max_new_tokens: int) -> str:
    n_obs = prompt.count("Observation:")
    idx = min(n_obs, len(_STUB_SCRIPT) - 1)
    return _STUB_SCRIPT[idx]


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------

def run_react_loop(
    diff: str,
    model,
    tokenizer,
    max_iterations: int,
    max_new_tokens: int,
    generate_fn,
) -> dict:
    """
    Run the full ReAct loop for a single diff.
    Returns a dict with keys: trace, final_review, n_steps, elapsed.
    """
    system_block = f"<|system|>{_REACT_SYSTEM}<|end|>"
    user_block   = f"<|user|>{_REACT_USER_TEMPLATE.format(diff=diff[:2000])}<|end|>"
    prompt = system_block + user_block + "<|assistant|>\nThought:"

    trace: list[dict] = []
    n_steps = 0
    t0 = time.time()

    for iteration in range(max_iterations):
        raw_out = generate_fn(model, tokenizer, prompt, max_new_tokens)
        step    = _parse_step(raw_out)

        if step["type"] == "final":
            trace.append({"step": n_steps, "type": "final_answer",
                          "content": step["content"]})
            final_review = _extract_final_review(step["content"])
            break

        if step["type"] == "action":
            n_steps += 1
            observation = call_tool(step["tool"], step["args"])
            trace.append({
                "step":        n_steps,
                "type":        "action",
                "tool":        step["tool"],
                "args":        step["args"],
                "observation": observation,
            })
            prompt += (
                f" {raw_out}\n"
                f"Observation: {observation}\n"
                f"Thought:"
            )
        elif step["type"] == "thought":
            trace.append({"step": n_steps, "type": "thought",
                          "content": step["content"]})
            prompt += f" {raw_out}\n"
        else:
            prompt += f" {raw_out}\n"

        if iteration == max_iterations - 1:
            prompt += (
                " I have gathered enough information. "
                "Final Answer: "
            )
            raw_out = generate_fn(model, tokenizer, prompt, max_new_tokens)
            trace.append({"step": n_steps + 1, "type": "final_answer",
                          "content": raw_out})
            final_review = _extract_final_review(raw_out)

    else:
        final_review = {"summary": "Max iterations reached without Final Answer.",
                        "issues": [], "security_concerns": [],
                        "code_quality": "N/A", "recommendations": []}

    return {
        "trace":        trace,
        "final_review": final_review,
        "n_steps":      n_steps,
        "elapsed":      round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

_SYNTHETIC_DIFFS = [
    """\
diff --git a/payment.py b/payment.py
--- a/payment.py
+++ b/payment.py
@@ -0,0 +1,18 @@
+import requests
+
+def process_payment(amount, card_number):
+    url = "https://api.payments.example.com/charge"
+    payload = {"amount": amount, "card": card_number}
+    r = requests.post(url, json=payload)
+    return r.json()
""",
    """\
diff --git a/auth.py b/auth.py
--- a/auth.py
+++ b/auth.py
@@ -1,6 +1,10 @@
+import hashlib
+
+def authenticate_user(username, password):
+    stored = db.query(f"SELECT hash FROM users WHERE name='{username}'")
+    return hashlib.md5(password.encode()).hexdigest() == stored
""",
    """\
diff --git a/utils.py b/utils.py
--- a/utils.py
+++ b/utils.py
@@ -0,0 +1,8 @@
+import os
+import yaml
+
+def load_config(path):
+    with open(path) as f:
+        return yaml.load(f)  # unsafe loader
+
+def run_cmd(cmd): return os.system(cmd)
""",
    """\
diff --git a/upload.py b/upload.py
--- a/upload.py
+++ b/upload.py
@@ -0,0 +1,10 @@
+from flask import request
+
+def upload_file():
+    f = request.files['file']
+    f.save('/uploads/' + f.filename)
+    return 'ok'
""",
    """\
diff --git a/api.py b/api.py
--- a/api.py
+++ b/api.py
@@ -0,0 +1,12 @@
+from django.db import connection
+
+def search_products(query):
+    sql = "SELECT * FROM products WHERE name LIKE '%" + query + "%'"
+    with connection.cursor() as c:
+        c.execute(sql)
+        return c.fetchall()
""",
]


def load_test_examples(path: Path, n: int) -> list[dict]:
    if not path.exists():
        print(f"  Note: {path} not found \u2014 using {n} synthetic diffs.")
        return [{"diff": d} for d in _SYNTHETIC_DIFFS[:n]]
    examples = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            text = raw.get("text", "")
            user_m = re.search(r"<\|user\|>(.*?)<\|end\|>", text, re.DOTALL)
            if not user_m:
                continue
            user_text = user_m.group(1).strip()
            prefix = "Review this pull request diff:"
            diff = user_text[len(prefix):].strip() if user_text.startswith(prefix) else user_text
            examples.append({"diff": diff})
            if len(examples) >= n:
                break
    print(f"  Loaded {len(examples)} examples from {path}")
    return examples


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ReAct agent for ReviewMind code review",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-examples",     type=int, default=DEFAULT_N_EXAMPLES)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--adapter-repo",   default=ADAPTER_REPO, metavar="USERNAME/REPO")
    parser.add_argument("--base-model",     default=BASE_MODEL)
    parser.add_argument("--output",         default=str(OUTPUT_FILE))
    parser.add_argument(
        "--no-model", action="store_true",
        help="Use a deterministic stub instead of loading the real model (smoke-test).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    print("\n" + "=" * 65)
    print("  ReviewMind \u2013 ReAct Agent")
    print("=" * 65)
    print(f"  Base model     : {args.base_model}")
    print(f"  Adapter        : {args.adapter_repo}")
    print(f"  Examples       : {args.n_examples}")
    print(f"  Max iterations : {args.max_iterations}")
    print(f"  Max new tokens : {args.max_new_tokens}")
    print(f"  Output         : {output_path}")
    print(f"  Stub mode      : {args.no_model}")

    # \u2500\u2500 Model \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if args.no_model:
        print("\n[1/3] Using stub model (--no-model).")
        model, tokenizer = _StubModel(), _StubTokenizer()
        generate_fn = generate_step_stub
    else:
        print("\n[1/3] Loading model \u2026")
        model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_repo)
        generate_fn = generate_step

    # \u2500\u2500 Examples \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    print("\n[2/3] Loading test examples \u2026")
    examples = load_test_examples(TEST_JSONL, args.n_examples)
    if not examples:
        sys.exit("No valid examples found \u2014 aborting.")

    # \u2500\u2500 ReAct loop \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    print(f"\n[3/3] Running ReAct loops for {len(examples)} example(s) \u2026\n")
    all_results = []

    for i, ex in enumerate(examples):
        diff = ex["diff"]
        print(f"  [{i + 1}/{len(examples)}] Running agent \u2026", end=" ", flush=True)

        result = run_react_loop(
            diff=diff,
            model=model,
            tokenizer=tokenizer,
            max_iterations=args.max_iterations,
            max_new_tokens=args.max_new_tokens,
            generate_fn=generate_fn,
        )

        print(
            f"{result['n_steps']} tool calls  "
            f"{len(result['trace'])} trace steps  "
            f"{result['elapsed']}s"
        )

        all_results.append({
            "index":        i,
            "diff":         diff[:1500],
            "trace":        result["trace"],
            "final_review": result["final_review"],
            "n_tool_calls": result["n_steps"],
            "elapsed_s":    result["elapsed"],
        })

    # \u2500\u2500 Save \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    print(f"\nResults saved \u2192 {output_path}  ({len(all_results)} entries)")

    # \u2500\u2500 Summary \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    print("\n" + "=" * 65)
    print("  AGENT SUMMARY")
    print("=" * 65)
    avg_calls   = sum(r["n_tool_calls"] for r in all_results) / len(all_results)
    avg_elapsed = sum(r["elapsed_s"]    for r in all_results) / len(all_results)
    print(f"  Examples processed  : {len(all_results)}")
    print(f"  Avg tool calls/diff : {avg_calls:.1f}")
    print(f"  Avg elapsed         : {avg_elapsed:.1f}s")

    if all_results:
        print("\n  SAMPLE (index 0) \u2014 final review:")
        rev = all_results[0]["final_review"]
        print(f"    summary           : {str(rev.get('summary',''))[:120]}")
        print(f"    issues            : {rev.get('issues', [])[:2]}")
        print(f"    security_concerns : {rev.get('security_concerns', [])[:2]}")
        print(f"    recommendations   : {rev.get('recommendations', [])[:2]}")
        print("\n  TRACE (index 0):")
        for step in all_results[0]["trace"][:4]:
            if step["type"] == "action":
                print(f"    [{step['step']}] {step['tool']}({step['args']!r:.60}) \u2192 {str(step['observation'])[:80]}")
            elif step["type"] == "final_answer":
                print(f"    [final] {str(step['content'])[:100]}")

    print("=" * 65)


if __name__ == "__main__":
    main()
