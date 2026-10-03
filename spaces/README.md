---
title: ReviewMind AI Code Reviewer
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# ReviewMind — AI Code Reviewer

Paste a unified diff and get an AI-generated code review.

**Modes**
- **Plain Review** — direct review with a standard prompt
- **Chain of Thought** — 4-step reasoning scaffold before the review

**Model**: `meta-llama/Llama-3.1-8B-Instruct` via HF Inference API

**Full project** (fine-tuned LoRA, RAG, ReAct agent, FastAPI serving):
[github.com/Malak-Israrullah-Khan/ReviewMind](https://github.com/Malak-Israrullah-Khan/ReviewMind)
