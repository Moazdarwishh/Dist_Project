"""
CPU-only LLM inference.

The project runs google/flan-t5-small locally using PyTorch on CPU. This keeps
the project portable on macOS and normal laptops while preserving the real LLM
generation step.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict

_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/flan-t5-small")

_tokenizer = None
_model = None

_load_lock = threading.Lock()
_infer_lock = threading.Lock()


def _ensure_loaded() -> None:
    """Load tokenizer and model exactly once."""
    global _tokenizer, _model

    if _model is not None:
        return

    with _load_lock:
        if _model is not None:
            return

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers, torch, and sentencepiece are required. "
                "Run: pip install -r requirements.txt"
            ) from exc

        print(f"[LLM] Loading {_MODEL_NAME} on CPU ...")
        t0 = time.time()

        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)
        _model.eval()

        print(f"[LLM] Loaded in {time.time() - t0:.1f}s")


def run_llm(query: str, context: str, max_new_tokens: int = 80) -> Dict[str, str]:
    """
    Generate an answer grounded in retrieved context.

    Returns:
        {"answer": str}
    """
    _ensure_loaded()

    prompt = (
        "Answer the question using only the provided context.\n"
        "If the answer is not found in the context, say that the documents do not contain enough information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )

    with _infer_lock:
        import torch

        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        answer = _tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    return {"answer": answer}


def run_llm_text(query: str, context: str, max_new_tokens: int = 80) -> str:
    """Backward-compatible helper for code that needs only the answer string."""
    return run_llm(query, context, max_new_tokens=max_new_tokens)["answer"]
