"""
ReviewMind — Render deployment
Gradio interface backed by the HuggingFace Inference API.
No local model loading; generation is routed through HF's
serverless endpoint for meta-llama/Llama-3.1-8B-Instruct.

Environment variables
---------------------
  HF_TOKEN   HuggingFace access token (required for gated model access)
  PORT        Port to bind (set automatically by Render, default 7860)
"""

import os
import time

import gradio as gr
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID       = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 300
HF_TOKEN       = os.getenv("HF_TOKEN")
PORT           = int(os.getenv("PORT", 7860))

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
# Inference client
# ---------------------------------------------------------------------------

_client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

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


_PROMPT_FN = {
    "Plain": _plain_prompt,
    "CoT":   _cot_prompt,
}

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate(prompt: str) -> str:
    return _client.text_generation(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.3,
        stop_sequences=["<|end|>", "<|user|>"],
    )

# ---------------------------------------------------------------------------
# Gradio handler
# ---------------------------------------------------------------------------

def review(diff: str, mode: str) -> tuple[str, str]:
    diff = diff.strip()
    if not diff:
        return "", "⚠ Please paste a code diff before submitting."

    prompt = _PROMPT_FN.get(mode, _plain_prompt)(diff)

    try:
        t0          = time.perf_counter()
        review_text = _generate(prompt)
        elapsed     = round(time.perf_counter() - t0, 1)
    except Exception as exc:
        return "", f"⚠ Inference API error: {exc}"

    words  = len(review_text.split())
    status = f"✓  {mode} · {words} words · {elapsed}s"
    return review_text, status

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="ReviewMind — AI Code Reviewer") as demo:
    gr.Markdown(
        "# ReviewMind — AI Code Reviewer\n"
        "Paste a unified diff, choose a mode, and get an AI-generated code review.\n\n"
        "Powered by **meta-llama/Llama-3.1-8B-Instruct** via the HuggingFace Inference API."
    )

    with gr.Row():
        with gr.Column(scale=2):
            diff_input = gr.Textbox(
                label="Code diff",
                placeholder="Paste a unified diff here …",
                lines=20,
                max_lines=40,
                value=EXAMPLE_DIFF,
            )
            with gr.Row():
                mode_dropdown = gr.Dropdown(
                    label="Mode",
                    choices=["Plain", "CoT"],
                    value="Plain",
                    scale=2,
                )
                submit_btn = gr.Button("Submit", variant="primary", scale=1)

        with gr.Column(scale=2):
            review_output = gr.Textbox(
                label="Review",
                lines=20,
                max_lines=40,
                interactive=False,
                show_copy_button=True,
            )
            status_line = gr.Textbox(label="Status", lines=1, interactive=False)

    gr.Markdown(
        "**Plain** — standard prompt · "
        "**CoT** — 4-step chain-of-thought reasoning before the review\n\n"
        "> The example diff fixes SQL injection but introduces MD5 password hashing — "
        "a critical flaw. MD5 must never be used for passwords; use `bcrypt` or `argon2`."
    )

    submit_btn.click(fn=review, inputs=[diff_input, mode_dropdown],
                     outputs=[review_output, status_line])
    diff_input.submit(fn=review, inputs=[diff_input, mode_dropdown],
                      outputs=[review_output, status_line])

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT)
