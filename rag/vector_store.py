"""
Persistent ChromaDB vector store for real-document RAG.

This replaces the old sklearn NearestNeighbors files with a persistent
ChromaDB database stored on disk. The same vector_db/ folder can be reused
after restarting the scheduler and worker services.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_DEFAULT_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
_DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "project_documents")
_DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

_embedding_model = None
_embedding_lock = threading.Lock()


def _get_embedding_model():
    """Lazy-load the sentence-transformer exactly once per process."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    with _embedding_lock:
        if _embedding_model is not None:
            return _embedding_model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required. Install requirements.txt") from exc
        print(f"[VectorStore] Loading embedding model {_DEFAULT_EMBEDDING_MODEL} ...")
        _embedding_model = SentenceTransformer(_DEFAULT_EMBEDDING_MODEL)
        return _embedding_model


def _embed(texts: List[str]) -> np.ndarray:
    model = _get_embedding_model()
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)


def _safe_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata values must be primitive and non-None."""
    cleaned: Dict[str, Any] = {}
    for key, value in (meta or {}).items():
        if value is None:
            cleaned[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def _chunk_id(chunk: Dict[str, Any]) -> str:
    meta = chunk.get("metadata") or {}
    source = str(meta.get("source", "unknown"))
    page = str(meta.get("page", ""))
    cid = str(meta.get("chunk_id", meta.get("global_chunk_id", "0")))
    raw = f"{source}:{page}:{cid}:{chunk.get('text', '')[:64]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class ChromaVectorStore:
    """Small wrapper around a persistent ChromaDB collection."""

    def __init__(self, db_path: Optional[str] = None, collection_name: Optional[str] = None):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is required. Install requirements.txt") from exc

        self.db_path = str(Path(db_path or _DEFAULT_DB_PATH))
        self.collection_name = collection_name or _DEFAULT_COLLECTION
        Path(self.db_path).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 64) -> int:
        """Embed and store chunks. Existing IDs are upserted."""
        if not chunks:
            return 0

        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = _embed(texts).tolist()
            ids = [_chunk_id(c) for c in batch]
            metadatas = [_safe_metadata(c.get("metadata") or {}) for c in batch]

            self.collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            total += len(batch)
        return total

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return top-k matching chunks with source metadata."""
        if self.count_documents() == 0:
            return []

        query_embedding = _embed([query])[0].tolist()
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"],
        )

        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        matches: List[Dict[str, Any]] = []
        for text, meta, distance in zip(docs, metas, distances):
            score = 1.0 - float(distance) if distance is not None else None
            matches.append({
                "text": text,
                "metadata": meta or {},
                "score": round(score, 4) if score is not None else None,
            })
        return matches

    def count_documents(self) -> int:
        return int(self.collection.count())

    def reset_database(self) -> None:
        """Delete and recreate the collection, leaving the db folder in place."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


# Convenience functions used by the rest of the project.
def add_chunks(chunks: List[Dict[str, Any]], db_path: Optional[str] = None) -> int:
    return ChromaVectorStore(db_path=db_path).add_chunks(chunks)


def search(query: str, top_k: int = 3, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    return ChromaVectorStore(db_path=db_path).search(query, top_k=top_k)


def count_documents(db_path: Optional[str] = None) -> int:
    return ChromaVectorStore(db_path=db_path).count_documents()


def reset_database(db_path: Optional[str] = None) -> None:
    ChromaVectorStore(db_path=db_path).reset_database()
