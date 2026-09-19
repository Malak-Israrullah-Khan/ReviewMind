"""
ReviewMind – RAG Retriever
Loads the ChromaDB vector database built by rag/build_index.py and provides
a simple interface to retrieve the top-k most relevant guideline chunks for a
given code diff.

Usage (standalone):
    python rag/retriever.py --diff "def foo():\n  x=1\n  return x"
    python rag/retriever.py --diff-file path/to/diff.txt --top-k 5

Importable API:
    from rag.retriever import Retriever
    retriever = Retriever()
    chunks = retriever.retrieve(diff_text, top_k=3)
    print(retriever.format_context(chunks))
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = PROJECT_ROOT / "rag" / "vectorstore"
COLLECTION_NAME = "code_guidelines"
EMBED_MODEL     = "all-MiniLM-L6-v2"
DEFAULT_TOP_K   = 3

_SOURCE_LABELS = {
    "pep8":               "PEP 8 – Python Style Guide",
    "owasp_top10":        "OWASP Top 10 Security Guidelines",
    "google_python_style": "Google Python Style Guide",
}


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class Retriever:
    """
    Thin wrapper around a ChromaDB collection for cosine-similarity retrieval.

    The collection must already exist (built by rag/build_index.py).
    The embedding model is loaded lazily on first query.
    """

    def __init__(
        self,
        vectorstore_dir: Optional[Path] = None,
        collection_name: str = COLLECTION_NAME,
        embed_model: str = EMBED_MODEL,
    ) -> None:
        self._vectorstore_dir = vectorstore_dir or VECTORSTORE_DIR
        self._collection_name = collection_name
        self._embed_model     = embed_model
        self._collection      = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_collection(self):
        if self._collection is not None:
            return self._collection

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

        store = str(self._vectorstore_dir)
        if not self._vectorstore_dir.exists():
            sys.exit(
                f"\nVector store not found at {store}\n"
                "Build it first:  python rag/build_index.py"
            )

        client = chromadb.PersistentClient(path=store)

        try:
            names = [c.name for c in client.list_collections()]
        except Exception:
            names = []

        if self._collection_name not in names:
            sys.exit(
                f"\nCollection '{self._collection_name}' not found in {store}\n"
                "Build it first:  python rag/build_index.py"
            )

        ef = SentenceTransformerEmbeddingFunction(model_name=self._embed_model)
        self._collection = client.get_collection(
            name=self._collection_name,
            embedding_function=ef,
        )
        return self._collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """
        Query the vector store and return the top_k most similar chunks.

        Each result dict has keys:
            id          – chunk ID (e.g. "pep8_42")
            document    – chunk text
            source      – source name (e.g. "pep8")
            chunk_index – integer index within that source
            distance    – cosine distance (lower = more similar)
        """
        if not query.strip():
            return []

        collection = self._load_collection()
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                {
                    "id":          results["ids"][0][len(chunks)],
                    "document":    doc,
                    "source":      meta.get("source", "unknown"),
                    "chunk_index": meta.get("chunk_index", -1),
                    "distance":    dist,
                }
            )
        return chunks

    def format_context(self, chunks: list[dict], header: bool = True) -> str:
        """
        Format retrieved chunks as a block of text suitable for prompt injection.

        Example output:
            --- Relevant Guidelines ---
            [PEP 8 – Python Style Guide]
            Use 4 spaces per indentation level. ...

            [OWASP Top 10 Security Guidelines]
            A01:2021 – Broken Access Control ...
        """
        if not chunks:
            return ""

        parts = []
        if header:
            parts.append("--- Relevant Coding Guidelines ---")

        for chunk in chunks:
            label = _SOURCE_LABELS.get(chunk["source"], chunk["source"])
            parts.append(f"\n[{label}]\n{chunk['document'].strip()}")

        return "\n".join(parts)

    def collection_size(self) -> int:
        return self._load_collection().count()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve relevant guidelines for a code diff",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--diff", metavar="TEXT",
                       help="Diff text passed directly as a string.")
    group.add_argument("--diff-file", metavar="PATH",
                       help="Path to a file containing the diff.")
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve.",
    )
    parser.add_argument(
        "--show-distances", action="store_true",
        help="Print cosine distances alongside each chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.diff_file:
        diff_text = Path(args.diff_file).read_text(encoding="utf-8")
    else:
        diff_text = args.diff

    retriever = Retriever()
    print(f"Collection size: {retriever.collection_size():,} chunks\n")

    chunks = retriever.retrieve(diff_text, top_k=args.top_k)

    if not chunks:
        print("No chunks retrieved.")
        return

    print(f"Top {len(chunks)} results for query:\n  {diff_text[:120].replace(chr(10), ' ')} …\n")
    print("=" * 65)
    for i, chunk in enumerate(chunks, 1):
        label = _SOURCE_LABELS.get(chunk["source"], chunk["source"])
        dist_tag = f"  (distance={chunk['distance']:.4f})" if args.show_distances else ""
        print(f"\n[{i}] {label}{dist_tag}")
        print("-" * 65)
        print(chunk["document"][:500])
        if len(chunk["document"]) > 500:
            print("  … (truncated)")

    print("\n" + "=" * 65)
    print("Formatted context block:\n")
    print(retriever.format_context(chunks))


if __name__ == "__main__":
    main()
