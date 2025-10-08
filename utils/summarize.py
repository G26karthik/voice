"""Summarization utilities.

Provides chunking and summarization using Hugging Face models (t5-small or bart-large-cnn).
Supports adjustable length: short, medium, detailed.
"""
from __future__ import annotations
from typing import List, Literal
import math

from transformers import pipeline

_SUMMARY_MODEL_DEFAULT = "t5-small"  # baseline
_FAST_MODEL = "sshleifer/distilbart-cnn-12-6"  # faster summarizer
_summary_pipe = None

LengthSetting = Literal['short','medium','detailed']


def load_summarizer(model_name: str = _SUMMARY_MODEL_DEFAULT):
    global _summary_pipe
    if _summary_pipe is None:
        task = "summarization" if not model_name.startswith("t5") else "text2text-generation"
        _summary_pipe = pipeline(task, model=model_name)
    return _summary_pipe


def chunk_text(text: str, max_tokens: int = 800) -> List[str]:
    # Simple token-ish split by words
    words = text.split()
    chunks = []
    current = []
    for w in words:
        current.append(w)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def summarize_text(text: str, length: LengthSetting = 'medium', model_name: str = _SUMMARY_MODEL_DEFAULT, fast: bool = False) -> str:
    if fast:
        model_name = _FAST_MODEL
    if not text.strip():
        return ""
    pipe = load_summarizer(model_name)
    # heuristics for length
    if length == 'short':
        max_len = 80
        min_len = 25
    elif length == 'detailed':
        max_len = 250
        min_len = 100
    else:
        max_len = 150
        min_len = 60

    chunks = chunk_text(text, max_tokens=350 if 'bart' in model_name else 250)
    summaries = []
    for c in chunks:
        if pipe.task == 'text2text-generation':
            prompt = f"summarize: {c}"[:4000]
            out = pipe(prompt, max_length=max_len, min_length=min_len, do_sample=False)
            summaries.append(out[0]['generated_text'])
        else:
            out = pipe(c, max_length=max_len, min_length=min_len, do_sample=False)
            summaries.append(out[0]['summary_text'])
    # If multiple summaries, recursively summarize
    combined = "\n".join(summaries)
    if len(summaries) > 1 and len(combined.split()) > max_len:
        return summarize_text(combined, length=length, model_name=model_name)
    return combined.strip()

__all__ = ["summarize_text", "chunk_text", "load_summarizer"]
