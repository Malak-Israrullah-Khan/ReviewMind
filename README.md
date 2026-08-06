# ReviewMind

A fine-tuned, RAG-augmented, agentic code review system built as a capstone ML project.

---

## Architecture Overview

ReviewMind is built in layers, each adding a capability on top of the previous:

1. **Data** → curated code review dataset
2. **Fine-Tuning** → domain-adapted base model
3. **Chain of Thought** → structured reasoning traces
4. **DPO** → preference-aligned outputs
5. **RAG** → grounded, document-aware review
6. **Agent** → tool-using autonomous reviewer
7. **Evaluation** → automated quality measurement
8. **Deployment** → production-ready serving

---

## Data

_Covers dataset sourcing, cleaning, and splitting for supervised fine-tuning and preference optimization._

- Raw data: `data/raw/`
- Processed data: `data/processed/`
- Train/val/test splits: `data/splits/`

---

## Fine-Tuning

_Supervised fine-tuning of a base LLM on code review examples using LoRA/QLoRA via PEFT and TRL._

- Training scripts: `training/finetune/`
- Config: `configs/training_config.yaml`

---

## Chain of Thought

_Augmenting training data with explicit reasoning traces to improve review quality and interpretability._

---

## DPO

_Direct Preference Optimization to align model outputs with human reviewer preferences._

- DPO scripts: `training/dpo/`

---

## RAG

_Retrieval-Augmented Generation using a vector store of coding standards, docs, and past reviews._

- Source documents: `rag/documents/`
- Vector store: `rag/vectorstore/`

---

## Agent

_Agentic loop giving the model access to tools (linters, test runners, search) for deeper analysis._

- Agent tools: `agent/tools/`

---

## Evaluation

_Automated evaluation pipeline using ROUGE, BERTScore, and custom code-review–specific metrics._

- Evaluation scripts: `evaluation/`

---

## Deployment

_FastAPI serving layer with MLflow experiment tracking and Prometheus metrics._

- API: `serving/api/`
- Experiment runs: `mlflow_runs/`
