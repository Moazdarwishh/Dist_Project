"""
Real document loader for the RAG pipeline.

Supported formats:
- .txt  via normal UTF-8 text reading
- .pdf  via pypdf, preserving page numbers
- .docx via python-docx

The loader returns page/document records. Chunking is done separately in
rag/chunker.py so metadata can be preserved cleanly.
"""

from pathlib import Path
from typing import Dict, List, Any
import re

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def _clean(text: str) -> str:
    """Collapse repeated whitespace while preserving readable text."""
    return re.sub(r"\s+", " ", text or "").strip()


def load_documents(documents_dir: str) -> List[Dict[str, Any]]:
    """
    Load all supported documents from a folder.

    Returns a list of dicts:
        {
          "text": "...",
          "metadata": {
             "source": "file.pdf",
             "file_type": "pdf",
             "page": 1
          }
        }
    """
    root = Path(documents_dir)
    if not root.exists():
        raise FileNotFoundError(f"Documents folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    records: List[Dict[str, Any]] = []
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        raise ValueError(f"No PDF, TXT, or DOCX files found in {root}")

    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            records.extend(_load_txt(path, root))
        elif suffix == ".pdf":
            records.extend(_load_pdf(path, root))
        elif suffix == ".docx":
            records.extend(_load_docx(path, root))

    if not records:
        raise ValueError("Documents were found, but no readable text was extracted.")

    return records


def _relative_source(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return path.name


def _load_txt(path: Path, root: Path) -> List[Dict[str, Any]]:
    text = _clean(path.read_text(encoding="utf-8", errors="ignore"))
    if not text:
        return []
    return [{
        "text": text,
        "metadata": {
            "source": _relative_source(path, root),
            "file_type": "txt",
            "page": None,
        },
    }]


def _load_pdf(path: Path, root: Path) -> List[Dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required for PDF ingestion. Install requirements.txt") from exc

    reader = PdfReader(str(path))
    records: List[Dict[str, Any]] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if not text:
            continue
        records.append({
            "text": text,
            "metadata": {
                "source": _relative_source(path, root),
                "file_type": "pdf",
                "page": page_idx,
            },
        })
    return records


def _load_docx(path: Path, root: Path) -> List[Dict[str, Any]]:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("python-docx is required for DOCX ingestion. Install requirements.txt") from exc

    document = Document(str(path))
    paragraphs = [_clean(p.text) for p in document.paragraphs if _clean(p.text)]
    text = _clean("\n".join(paragraphs))
    if not text:
        return []
    return [{
        "text": text,
        "metadata": {
            "source": _relative_source(path, root),
            "file_type": "docx",
            "page": None,
        },
    }]
