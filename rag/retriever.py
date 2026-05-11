"""
RAG retriever backed by persistent ChromaDB.

Flow:
    user query → query embedding → ChromaDB similarity search
    → top-k chunks + metadata → context string for the LLM
"""

import os
from typing import Any, Dict, List, Optional

from rag.vector_store import ChromaVectorStore

_EMPTY_DB_ERROR = (
    "Vector database is empty. Please run: "
    "python -m rag.ingest --documents ./documents --db ./vector_db"
)


def retrieve_context(query: str, top_k: int = 3, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve relevant chunks and return context + source metadata.

    Returns:
        {
          "context": "- [source] chunk...",
          "sources": [{"source": "file.pdf", "page": 1, "chunk_id": 0, "score": 0.91}],
          "error": None | str
        }
    """
    store = ChromaVectorStore(db_path=db_path or os.getenv("VECTOR_DB_PATH", "./vector_db"))
    if store.count_documents() == 0:
        return {"context": "", "sources": [], "error": _EMPTY_DB_ERROR}

    matches = store.search(query, top_k=top_k)
    if not matches:
        return {
            "context": "",
            "sources": [],
            "error": "No relevant chunks were found in the vector database.",
        }

    lines: List[str] = []
    sources: List[Dict[str, Any]] = []

    for match in matches:
        text = match.get("text", "")
        meta = match.get("metadata") or {}
        score = match.get("score")

        source = meta.get("source", "unknown")
        page = meta.get("page", "")
        chunk_id = meta.get("chunk_id", meta.get("global_chunk_id", ""))

        page_label = f", page {page}" if page not in (None, "") else ""
        score_label = f", score {score}" if score is not None else ""
        lines.append(f"- [{source}{page_label}{score_label}] {text}")

        sources.append({
            "source": source,
            "file_type": meta.get("file_type", ""),
            "page": page,
            "chunk_id": chunk_id,
            "score": score,
        })

    return {"context": "\n".join(lines), "sources": sources, "error": None}


def retrieve_context_text(query: str, top_k: int = 3, db_path: Optional[str] = None) -> str:
    """Backward-compatible helper for old code that expects only a string."""
    result = retrieve_context(query=query, top_k=top_k, db_path=db_path)
    return result.get("context", "")
