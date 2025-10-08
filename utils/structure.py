"""Section / topic segmentation utilities.

Approach:
- Split transcript into paragraphs (double newline or length-based).
- Use simple embedding similarity to group contiguous sentences into topics.
- Produce structured notes with inferred headings (first keyword or generated short summary line).

For deeper accuracy, a dedicated topic segmentation model could be integrated.
"""
from __future__ import annotations
from typing import List, Dict
import re

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore

_embedder = None

def _load_embedder():
    global _embedder
    if _embedder is None and SentenceTransformer is not None:
        try:
            _embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception:
            _embedder = None
    return _embedder

SENT_RE = re.compile(r'(?<=[.!?])\s+')

def segment_topics(text: str, similarity_threshold: float = 0.55, max_section_sentences: int = 12) -> List[Dict]:
    if not text.strip():
        return []
    sentences = [s.strip() for s in SENT_RE.split(text) if s.strip()]
    if len(sentences) <= 3:
        return [{"heading": "Section 1", "content": text.strip()}]
    embedder = _load_embedder()
    if not embedder or util is None:
        # fallback: chunk by fixed size
        sections = []
        for i in range(0, len(sentences), max_section_sentences):
            chunk = sentences[i:i+max_section_sentences]
            heading = chunk[0].split(':')[0][:60]
            sections.append({"heading": heading or f"Section {len(sections)+1}", "content": " ".join(chunk)})
        return sections

    embs = embedder.encode(sentences, convert_to_tensor=True)
    sections: List[Dict] = []
    current_group = [sentences[0]]
    prev_vec = embs[0]
    for sent, vec in zip(sentences[1:], embs[1:]):
        sim = util.cos_sim(prev_vec, vec).item()
        if sim < similarity_threshold or len(current_group) >= max_section_sentences:
            heading = current_group[0].split(':')[0][:60]
            sections.append({"heading": heading or f"Section {len(sections)+1}", "content": " ".join(current_group)})
            current_group = [sent]
        else:
            current_group.append(sent)
        prev_vec = vec
    if current_group:
        heading = current_group[0].split(':')[0][:60]
        sections.append({"heading": heading or f"Section {len(sections)+1}", "content": " ".join(current_group)})
    return sections

__all__ = ["segment_topics"]
