"""
ReviewMind – RAG Index Builder
Downloads three coding-guideline documents, chunks them, embeds with
sentence-transformers all-MiniLM-L6-v2, and stores everything in a
ChromaDB vector database at rag/vectorstore/.

Sources:
  • PEP 8 – Python Style Guide        https://peps.python.org/pep-0008/
  • OWASP Top 10 Security Guidelines   https://owasp.org/Top10/
  • Google Python Style Guide          https://google.github.io/styleguide/pyguide.html

Usage:
    python rag/build_index.py
    python rag/build_index.py --chunk-size 400 --overlap 40
    python rag/build_index.py --reset          # wipe and rebuild the collection
"""

import argparse
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = PROJECT_ROOT / "rag" / "vectorstore"
COLLECTION_NAME = "code_guidelines"
EMBED_MODEL     = "all-MiniLM-L6-v2"

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP    = 50

SOURCES = [
    {
        "name":  "pep8",
        "label": "PEP 8 – Style Guide for Python Code",
        "url":   "https://peps.python.org/pep-0008/",
    },
    {
        "name":  "owasp_top10",
        "label": "OWASP Top 10 Security Guidelines",
        "url":   "https://owasp.org/Top10/",
    },
    {
        "name":  "google_python_style",
        "label": "Google Python Style Guide",
        "url":   "https://google.github.io/styleguide/pyguide.html",
    },
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the ReviewMind RAG vector index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help="Maximum characters per chunk.",
    )
    parser.add_argument(
        "--overlap", type=int, default=DEFAULT_OVERLAP,
        help="Overlap in characters between consecutive chunks.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete and recreate the ChromaDB collection before indexing.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# HTML → plain text
# ---------------------------------------------------------------------------

def fetch_and_extract(url: str) -> str:
    """
    Download a URL and return clean plain text.
    Removes script/style/nav blocks, collapses whitespace.
    Falls back to a raw text response if the content is not HTML.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        sys.exit(
            "\nbeautifulsoup4 not found.\n"
            "Install with:  pip install beautifulsoup4 lxml"
        )

    try:
        import requests
    except ImportError:
        sys.exit("\nrequests not found.  Install with:  pip install requests")

    print(f"  Fetching {url} …", end=" ", flush=True)
    t0 = time.time()
    resp = requests.get(url, timeout=30, headers={"User-Agent": "ReviewMind-RAG/1.0"})
    resp.raise_for_status()
    elapsed = time.time() - t0
    print(f"{elapsed:.1f}s  ({len(resp.content) // 1024} KB)")

    content_type = resp.headers.get("content-type", "")
    if "html" not in content_type and not resp.text.lstrip().startswith("<"):
        return resp.text

    soup = BeautifulSoup(resp.text, "lxml")

    # Remove boilerplate tags that add noise without content
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "noscript", "meta", "link"]):
        tag.decompose()

    # Get text, use newlines as separators to preserve paragraph structure
    text = soup.get_text(separator="\n")

    # Normalise whitespace: collapse runs of blank lines, strip leading/trailing
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping fixed-size character chunks.
    Tries to break on whitespace so words are not split mid-token.
    """
    chunks = []
    start  = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Snap end to the nearest whitespace boundary (within ±30 chars)
        if end < length:
            snap = text.rfind(" ", start, end)
            if snap != -1 and snap > start + chunk_size // 2:
                end = snap

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def get_chroma_collection(reset: bool):
    """
    Return (or create) the ChromaDB collection backed by all-MiniLM-L6-v2.
    If reset=True, the existing collection is deleted first.
    """
    try:
        import chromadb
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )
    except ImportError:
        sys.exit(
            "\nchromadb not found.\n"
            "Install with:  pip install chromadb sentence-transformers"
        )

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nOpening ChromaDB at {VECTORSTORE_DIR} …")
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"  Existing collection '{COLLECTION_NAME}' deleted.")
        except Exception:
            pass

    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  Collection '{COLLECTION_NAME}'  "
          f"(existing docs: {collection.count()})")
    return collection


def add_chunks_to_collection(collection, source_name: str, chunks: list[str]) -> None:
    """
    Upsert chunks into the collection with IDs of the form
    <source_name>_<index> so reruns are idempotent.
    """
    if not chunks:
        return

    ids       = [f"{source_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i}
                 for i in range(len(chunks))]

    # Chroma has a default batch limit; upload in batches of 500 to be safe
    batch_size = 500
    for batch_start in range(0, len(chunks), batch_size):
        batch_end = batch_start + batch_size
        collection.upsert(
            documents=chunks[batch_start:batch_end],
            ids=ids[batch_start:batch_end],
            metadatas=metadatas[batch_start:batch_end],
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("\n" + "=" * 65)
    print("  ReviewMind – RAG Index Builder")
    print("=" * 65)
    print(f"  Embedding model : {EMBED_MODEL}")
    print(f"  Chunk size      : {args.chunk_size} chars")
    print(f"  Overlap         : {args.overlap} chars")
    print(f"  Vector store    : {VECTORSTORE_DIR}")
    print(f"  Collection      : {COLLECTION_NAME}")
    print(f"  Reset           : {args.reset}")

    collection = get_chroma_collection(args.reset)

    results = []
    print()

    for src in SOURCES:
        print(f"[{src['name']}]  {src['label']}")

        try:
            text = fetch_and_extract(src["url"])
        except Exception as exc:
            print(f"  ERROR fetching {src['url']}: {exc}")
            results.append((src["name"], src["label"], 0, "FAILED"))
            continue

        chunks = chunk_text(text, args.chunk_size, args.overlap)
        print(f"  Extracted {len(text):,} chars → {len(chunks)} chunks", end=" … ")

        t0 = time.time()
        add_chunks_to_collection(collection, src["name"], chunks)
        elapsed = time.time() - t0
        print(f"indexed in {elapsed:.1f}s")

        results.append((src["name"], src["label"], len(chunks), "OK"))
        print()

    # ── Summary ──────────────────────────────────────────
    print("=" * 65)
    print("  INDEXING SUMMARY")
    print("=" * 65)
    total_chunks = 0
    for name, label, n_chunks, status in results:
        marker = "✓" if status == "OK" else "✗"
        print(f"  {marker}  {label}")
        print(f"     {n_chunks:>5} chunks  [{name}]  {status}")
        total_chunks += n_chunks

    print("  " + "─" * 61)
    print(f"  Total chunks indexed : {total_chunks:,}")
    print(f"  Collection size      : {collection.count():,} documents")
    print(f"  Vector store path    : {VECTORSTORE_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
