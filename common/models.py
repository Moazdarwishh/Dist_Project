"""
Shared data models for the distributed RAG system.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import time


@dataclass
class Request:
    """A single user query entering the system."""
    id: int
    query: str
    created_at: float = field(default_factory=time.time)


@dataclass
class Response:
    """The result returned after worker processing."""
    id: int
    result: str
    latency: float
    worker_id: Optional[int] = None
    success: bool = True
    error: Optional[str] = None

    # RAG source attribution: filename, page, chunk_id, relevance score, etc.
    sources: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict."""
        return asdict(self)
