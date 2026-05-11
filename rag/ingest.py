"""Command-line ingestion for the ChromaDB RAG pipeline."""

import argparse
import time
from pathlib import Path

from rag.chunker import chunk_documents
from rag.document_loader import load_documents
from rag.vector_store import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDF/TXT/DOCX files into a persistent ChromaDB vector database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--documents", default="./documents", help="Folder containing PDF/TXT/DOCX files")
    parser.add_argument("--db", default="./vector_db", help="ChromaDB persistence folder")
    parser.add_argument("--chunk-size", type=int, default=700, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap in characters")
    parser.add_argument("--reset", action="store_true", help="Delete existing collection before ingesting")
    args = parser.parse_args()

    t0 = time.time()
    documents_dir = Path(args.documents)
    db_path = Path(args.db)

    print("=" * 70)
    print("  Real Document RAG Ingestion")
    print(f"  documents = {documents_dir.resolve()}")
    print(f"  vector_db = {db_path.resolve()}")
    print("=" * 70)

    docs = load_documents(str(documents_dir))
    chunks = chunk_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    store = ChromaVectorStore(db_path=str(db_path))
    if args.reset:
        print("[Ingest] Resetting ChromaDB collection ...")
        store.reset_database()

    stored = store.add_chunks(chunks)

    print("\n========== Ingestion Summary ==========")
    print(f"Files/pages loaded : {len(docs)}")
    print(f"Chunks created    : {len(chunks)}")
    print(f"Chunks stored     : {stored}")
    print(f"Total in DB       : {store.count_documents()}")
    print(f"Elapsed           : {time.time() - t0:.2f}s")
    print("=======================================\n")


if __name__ == "__main__":
    main()
