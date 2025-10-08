"""Centralized model loading & caching to avoid multiple loads across modules."""
from __future__ import annotations
from functools import lru_cache

from typing import Optional

from transformers import pipeline

# Whisper handled in transcribe to allow fallback; summarizer & qa centralized here

@lru_cache(maxsize=1)
def get_summarizer(model_name: str = "t5-small"):
    task = "summarization" if not model_name.startswith("t5") else "text2text-generation"
    return pipeline(task, model=model_name)

@lru_cache(maxsize=1)
def get_qa_pipeline(model_name: str = 'distilbert-base-cased-distilled-squad'):
    try:
        return pipeline('question-answering', model=model_name)
    except Exception:
        return None

# Embedding model
_sentence_embedder = None

def get_sentence_embedder():
    global _sentence_embedder
    if _sentence_embedder is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _sentence_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception:
            _sentence_embedder = None
    return _sentence_embedder
