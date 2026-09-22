"""
ReviewMind – FastAPI Serving Layer

Endpoints
---------
  GET  /           API info
  GET  /health     Model/retriever status
  POST /review     Generate a code review for a diff

Request body (POST /review)
---------------------------
  {
    "diff": "<code diff string>",
    "mode": "plain" | "cot" | "rag"   (default: "plain")
  }

Response (POST /review)
-----------------------
  {
    "review":          "<generated review text>",
    "mode":            "<mode used>",
    "word_count":      <int>,
    "latency_seconds": <float>
  }

Startup
-------
  uvicorn serving.api:app --host 0.0.0.0 --port 8000

Environment variables (optional)
---------------------------------
  BASE_MODEL    HuggingFace base model ID  (default: meta-llama/Llama-3.1-8B-Instruct)
  ADAPTER_REPO  HuggingFace LoRA adapter   (default: Malak-Israr/reviewmind-lora)
  MAX_NEW_TOKENS  tokens to generate       (default: 300)
  RAG_TOP_K     guideline chunks to fetch  (default: 3)
  NO_RAG        set to "1" to skip RAG     (default: unset)
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("reviewmind.api")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_BASE_MODEL     = os.getenv("BASE_MODEL",    "meta-llama/Llama-3.1-8B-Instruct")
_ADAPTER_REPO   = os.getenv("ADAPTER_REPO",  "Malak-Israr/reviewmind-lora")
_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "300"))
_RAG_TOP_K      = int(os.getenv("RAG_TOP_K", "3"))
_NO_RAG         = os.getenv("NO_RAG", "0") == "1"

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

# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------

_state: dict = {
    "model":     None,
    "tokenizer": None,
    "retriever": None,
    "ready":     False,
    "error":     None,
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model():
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    log.info("Loading tokenizer: %s", _BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL, use_fast=True)
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
    log.info("Loading base model (4-bit NF4): %s", _BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        _BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=False,
    )

    log.info("Attaching LoRA adapter: %s", _ADAPTER_REPO)
    model = PeftModel.from_pretrained(base, _ADAPTER_REPO)
    model.eval()
    log.info("Model ready.")
    return model, tokenizer


def _load_retriever():
    if _NO_RAG:
        log.info("[RAG] Skipped (NO_RAG=1).")
        return None
    try:
        from rag.retriever import Retriever
        retriever = Retriever()
        log.info("[RAG] Vector store ready (%d chunks).", retriever.collection_size())
        return retriever
    except SystemExit:
        log.warning("[RAG] Vector store absent — RAG mode will fall back to plain.")
        return None
    except Exception as exc:
        log.warning("[RAG] Could not load retriever (%s) — RAG falls back to plain.", exc)
        return None

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_plain_prompt(diff: str) -> str:
    return (
        f"<|system|>{SYSTEM_PROMPT}<|end|>"
        f"<|user|>Review this pull request diff:\n\n{diff}<|end|>"
        f"<|assistant|>"
    )


def _build_cot_prompt(diff: str) -> str:
    return (
        f"<|system|>{SYSTEM_PROMPT}<|end|>"
        f"<|user|>Review this pull request diff:\n\n{diff}{COT_SUFFIX}<|end|>"
        f"<|assistant|>"
    )


def _build_rag_prompt(diff: str, context_block: str) -> str:
    if not context_block:
        return _build_plain_prompt(diff)
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

    model     = _state["model"]
    tokenizer = _state["tokenizer"]

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
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== ReviewMind API starting up ===")
    try:
        _state["model"], _state["tokenizer"] = _load_model()
        _state["retriever"] = _load_retriever()
        _state["ready"] = True
        log.info("=== Startup complete — ready to serve ===")
    except Exception as exc:
        _state["error"] = str(exc)
        log.error("Startup failed: %s", exc)
    yield
    log.info("=== ReviewMind API shutting down ===")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ReviewMind",
    description="Fine-tuned code review API — supports plain, CoT, and RAG modes.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ReviewRequest(BaseModel):
    diff: str = Field(..., min_length=1, description="The code diff to review.")
    mode: str = Field("plain", description="Generation mode: plain | cot | rag.")


class ReviewResponse(BaseModel):
    review:          str
    mode:            str
    word_count:      int
    latency_seconds: float

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "ReviewMind",
        "version": "1.0.0",
        "endpoints": {
            "GET  /":        "API info",
            "GET  /health":  "Model / retriever status",
            "POST /review":  "Generate a code review (body: {diff, mode})",
        },
        "modes": {
            "plain": "Fine-tuned model, standard prompt",
            "cot":   "Fine-tuned model, chain-of-thought prompt",
            "rag":   "Fine-tuned model, RAG-augmented prompt",
        },
    }


@app.get("/health")
def health():
    return {
        "status":          "ready" if _state["ready"] else "unavailable",
        "model_loaded":    _state["model"] is not None,
        "retriever_loaded": _state["retriever"] is not None,
        "base_model":      _BASE_MODEL,
        "adapter_repo":    _ADAPTER_REPO,
        "max_new_tokens":  _MAX_NEW_TOKENS,
        "rag_top_k":       _RAG_TOP_K,
        "error":           _state["error"],
    }


@app.post("/review", response_model=ReviewResponse)
def review(request: ReviewRequest):
    if not _state["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready. {_state['error'] or 'Still loading.'}",
        )

    mode = request.mode.strip().lower()
    if mode not in {"plain", "cot", "rag"}:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Must be one of: plain, cot, rag.",
        )

    t0 = time.perf_counter()

    if mode == "cot":
        prompt = _build_cot_prompt(request.diff)
    elif mode == "rag":
        context_block = ""
        retriever = _state["retriever"]
        if retriever is not None:
            try:
                chunks        = retriever.retrieve(request.diff, top_k=_RAG_TOP_K)
                context_block = retriever.format_context(chunks)
            except Exception as exc:
                log.warning("[RAG] Retrieval failed (%s) — falling back to plain.", exc)
        prompt = _build_rag_prompt(request.diff, context_block)
    else:
        prompt = _build_plain_prompt(request.diff)

    review_text = _generate(prompt)
    latency     = round(time.perf_counter() - t0, 3)
    word_count  = len(review_text.split())

    log.info("mode=%-5s  words=%d  latency=%.2fs", mode, word_count, latency)

    return ReviewResponse(
        review=review_text,
        mode=mode,
        word_count=word_count,
        latency_seconds=latency,
    )
