"""Text chunking utilities for RAG ingestion."""

from typing import Any, Dict, List


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Split loaded document records into overlapping character chunks.

    Each returned item has:
        text: chunk text
        metadata: source, file_type, page, chunk_id
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[Dict[str, Any]] = []
    global_chunk_id = 0
    step = chunk_size - chunk_overlap

    for doc in documents:
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        base_meta = dict(doc.get("metadata") or {})

        start = 0
        local_chunk_id = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                meta = dict(base_meta)
                meta["chunk_id"] = local_chunk_id
                meta["global_chunk_id"] = global_chunk_id
                chunks.append({"text": chunk_text, "metadata": meta})
                global_chunk_id += 1
                local_chunk_id += 1
            if end >= len(text):
                break
            start += step

    return chunks
