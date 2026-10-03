"""
ReviewMind — Gradio Demo
Loads the fine-tuned LoRA model on startup and provides an interactive
code-review interface with three generation modes.

Usage:
    python demo/app.py
    python demo/app.py --share          # public Gradio link
    python demo/app.py --no-rag         # skip RAG (vectorstore absent)
"""

import argparse
import sys
import time
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO = "Malak-Israr/reviewmind-lora"
MAX_NEW_TOKENS = 300

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

EXAMPLE_DIFF = """\
diff --git a/backend/auth.py b/backend/auth.py
--- a/backend/auth.py
+++ b/backend/auth.py
@@ -1,22 +1,30 @@
 import sqlite3
-from flask import request, session
+import hashlib
+from flask import request, session, abort

-def login(username, password):
-    conn = sqlite3.connect("app.db")
-    cur  = conn.cursor()
-    sql  = "SELECT * FROM users WHERE username='" + username + "'"
-    cur.execute(sql)
-    user = cur.fetchone()
-    conn.close()
-    if user and user[2] == password:
-        session["user"] = username
-        return True
-    return False
+def login(username: str, password: str) -> bool:
+    conn = sqlite3.connect("app.db")
+    cur  = conn.cursor()
+    cur.execute(
+        "SELECT id, username, password_hash FROM users WHERE username = ?",
+        (username,),
+    )
+    user = cur.fetchone()
+    conn.close()
+    if user is None:
+        return False
+    stored_hash = user[2]
+    incoming_hash = hashlib.md5(password.encode()).hexdigest()
+    if stored_hash == incoming_hash:
+        session["user_id"] = user[0]
+        return True
+    return False

-def get_user_data(user_id):
-    conn = sqlite3.connect("app.db")
-    cur  = conn.cursor()
-    sql  = f"SELECT * FROM users WHERE id = {user_id}"
-    cur.execute(sql)
-    return cur.fetchone()
+def get_user_data(user_id: int) -> dict | None:
+    conn = sqlite3.connect("app.db")
+    cur  = conn.cursor()
+    cur.execute("SELECT id, username, email FROM users WHERE id = ?", (user_id,))
+    row = cur.fetchone()
+    conn.close()
+    return {"id": row[0], "username": row[1], "email": row[2]} if row else None
"""

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_model     = None
_tokenizer = None
_retriever = None
_load_error: str | None = None

# ---------------------------------------------------------------------------
# Model + retriever loading
# ---------------------------------------------------------------------------

def _load_model():
    global _model, _tokenizer, _load_error
    try:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        print(f"Loading tokenizer: {BASE_MODEL}")
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token    = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "left"

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        print(f"Loading base model (4-bit NF4): {BASE_MODEL}")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=False,
        )

        print(f"Attaching LoRA adapter: {ADAPTER_REPO}")
        mdl = PeftModel.from_pretrained(base, ADAPTER_REPO)
        mdl.eval()

        _model     = mdl
        _tokenizer = tok
        print("Model ready.")
    except Exception as exc:
        _load_error = str(exc)
        print(f"[ERROR] Model failed to load: {exc}")


def _load_retriever(no_rag: bool):
    global _retriever
    if no_rag:
        print("[RAG] Skipped (--no-rag).")
        return
    try:
        from rag.retriever import Retriever
        _retriever = Retriever()
        print(f"[RAG] Vector store ready ({_retriever.collection_size():,} chunks).")
    except (SystemExit, Exception) as exc:
        print(f"[RAG] Vector store unavailable ({exc}) — RAG falls back to plain.")

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _plain_prompt(diff: str) -> str:
    return (
        f"<|system|>{SYSTEM_PROMPT}<|end|>"
        f"<|user|>Review this pull request diff:\n\n{diff}<|end|>"
        f"<|assistant|>"
    )


def _cot_prompt(diff: str) -> str:
    return (
        f"<|system|>{SYSTEM_PROMPT}<|end|>"
        f"<|user|>Review this pull request diff:\n\n{diff}{COT_SUFFIX}<|end|>"
        f"<|assistant|>"
    )


def _rag_prompt(diff: str) -> str:
    context_block = ""
    if _retriever is not None:
        try:
            chunks        = _retriever.retrieve(diff, top_k=3)
            context_block = _retriever.format_context(chunks)
        except Exception as exc:
            print(f"[RAG] Retrieval failed ({exc}) — falling back to plain.")
    if not context_block:
        return _plain_prompt(diff)
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

def _generate(prompt: str) -> str:
    import torch

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1536,
    )
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.3,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_ids, skip_special_tokens=True).strip()

# ---------------------------------------------------------------------------
# Gradio handler
# ---------------------------------------------------------------------------

_MODE_MAP = {
    "Plain Review":     _plain_prompt,
    "Chain of Thought": _cot_prompt,
    "RAG":              _rag_prompt,
}


def review(diff: str, mode: str) -> tuple[str, str]:
    """
    Returns (review_text, status_line).
    Called by the Gradio interface on Submit.
    """
    if _load_error:
        return "", f"⚠ Model failed to load: {_load_error}"

    if _model is None:
        return "", "⚠ Model is still loading — please wait a moment and try again."

    diff = diff.strip()
    if not diff:
        return "", "⚠ Please paste a code diff before submitting."

    prompt_fn = _MODE_MAP.get(mode, _plain_prompt)
    prompt    = prompt_fn(diff)

    t0     = time.perf_counter()
    review_text = _generate(prompt)
    elapsed = round(time.perf_counter() - t0, 1)

    word_count = len(review_text.split())
    status = f"✓  {mode} · {word_count} words · {elapsed}s"
    return review_text, status

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ReviewMind — AI Code Reviewer") as demo:
        gr.Markdown(
            "# ReviewMind — AI Code Reviewer\n"
            "Paste a unified diff below and choose a review mode. "
            "The model is a fine-tuned LLaMA-3.1-8B with a LoRA adapter "
            "trained on real GitHub pull request reviews."
        )

        with gr.Row():
            with gr.Column(scale=2):
                diff_input = gr.Textbox(
                    label="Code diff",
                    placeholder="Paste a unified diff here …",
                    lines=22,
                    max_lines=40,
                    value=EXAMPLE_DIFF,
                )
                with gr.Row():
                    mode_dropdown = gr.Dropdown(
                        label="Review mode",
                        choices=["Plain Review", "Chain of Thought", "RAG"],
                        value="Plain Review",
                        scale=2,
                    )
                    submit_btn = gr.Button("Submit", variant="primary", scale=1)

            with gr.Column(scale=2):
                review_output = gr.Textbox(
                    label="Generated review",
                    lines=22,
                    max_lines=40,
                    interactive=False,
                    show_copy_button=True,
                )
                status_line = gr.Textbox(
                    label="Status",
                    lines=1,
                    interactive=False,
                )

        gr.Markdown(
            "### Modes\n"
            "- **Plain Review** — fine-tuned model with a standard prompt\n"
            "- **Chain of Thought** — adds a 4-step reasoning scaffold before the review\n"
            "- **RAG** — prepends relevant coding guidelines (PEP 8 / OWASP / Google style) "
            "retrieved from a local vector store; falls back to Plain if the store is absent\n\n"
            "> The pre-filled diff shows a common security mistake: MD5 is used for password "
            "hashing. MD5 is cryptographically broken and must never be used for passwords — "
            "use `bcrypt`, `argon2`, or `hashlib.scrypt` instead."
        )

        submit_btn.click(
            fn=review,
            inputs=[diff_input, mode_dropdown],
            outputs=[review_output, status_line],
        )
        diff_input.submit(
            fn=review,
            inputs=[diff_input, mode_dropdown],
            outputs=[review_output, status_line],
        )

    return demo

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReviewMind Gradio demo")
    parser.add_argument("--share",  action="store_true", help="Create a public Gradio link.")
    parser.add_argument("--no-rag", action="store_true", help="Skip RAG (vectorstore absent).")
    parser.add_argument("--port",   type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _load_model()
    _load_retriever(no_rag=args.no_rag)
    app = build_ui()
    app.launch(server_port=args.port, share=True, server_name="0.0.0.0")
