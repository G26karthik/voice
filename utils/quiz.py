"""Quiz & Flashcard generation utilities.

Uses KeyBERT for keyword extraction (fallback to spaCy if unavailable).
Generates MCQs and flashcards. For QA pair generation, a simple heuristic + summarization/backfill.
(Using full distilbert QA generation directly requires a context+question; here we synthesize questions.)
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import random
import json
import hashlib
import threading

try:
    from keybert import KeyBERT  # type: ignore
except Exception:  # pragma: no cover
    KeyBERT = None

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None

_kw_model = None
_spacy_nlp = None
_embed_model = None
_qa_pipe = None

# QA answer cache (for flashcard generation)
_qa_answer_cache: Dict[Tuple[str, str], str] = {}
_cache_lock = threading.Lock()

try:  # sentence-transformers for semantic similarity distractors
    from sentence_transformers import SentenceTransformer, util  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore

def load_keyword_model():
    global _kw_model
    if _kw_model is None and KeyBERT is not None:
        _kw_model = KeyBERT()
    return _kw_model


def load_spacy_model():  # fallback
    global _spacy_nlp
    if _spacy_nlp is None and spacy is not None:
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:
            _spacy_nlp = None
    return _spacy_nlp


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    if not text.strip():
        return []
    
    # Common stop words to filter out
    stop_words = {
        'about', 'after', 'again', 'against', 'could', 'would', 'should',
        'there', 'their', 'these', 'those', 'which', 'while', 'where',
        'other', 'some', 'such', 'very', 'even', 'just', 'more', 'most',
        'through', 'between', 'before', 'after', 'being', 'under', 'above'
    }
    
    model = load_keyword_model()
    if model:
        try:
            kws = model.extract_keywords(text, top_n=top_n * 2)  # Get more, then filter
            filtered = []
            for k, score in kws:
                # Filter: length > 3, not all lowercase generic words, not stop words
                if len(k) > 3 and k.lower() not in stop_words:
                    # Prefer multi-word phrases or capitalized terms
                    if ' ' in k or k[0].isupper() or len(k) > 6:
                        filtered.append(k)
                if len(filtered) >= top_n:
                    break
            return filtered if filtered else [k for k, _ in kws[:top_n]]
        except Exception:
            pass
    
    nlp = load_spacy_model()
    if nlp:
        doc = nlp(text)
        cands = set()
        # Prioritize named entities (PERSON, ORG, GPE, PRODUCT, etc.)
        for ent in doc.ents:
            if len(ent.text.strip()) > 3:
                cands.add(ent.text.strip())
        # Add meaningful noun chunks
        for nc in doc.noun_chunks:
            phrase = nc.text.strip()
            if 4 <= len(phrase) <= 40 and phrase.lower() not in stop_words:
                cands.add(phrase)
        return list(cands)[:top_n]
    
    # Improved fallback: filter common words, prioritize longer/capitalized
    words = [w for w in text.split() if len(w) > 4 and w.lower() not in stop_words]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    # Sort by frequency, then by length (prefer longer, more specific terms)
    sorted_words = sorted(freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


def _load_embedding_model():
    global _embed_model
    if _embed_model is None and SentenceTransformer is not None:
        try:
            _embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except Exception:
            _embed_model = None
    return _embed_model

def _load_qa_pipeline():
    global _qa_pipe
    if _qa_pipe is None:
        try:
            from transformers import pipeline  # local import to avoid cost if unused
            _qa_pipe = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
        except Exception:
            _qa_pipe = None
    return _qa_pipe

def _semantic_distractors(target: str, candidates: List[str], top_k: int = 3) -> List[str]:
    model = _load_embedding_model()
    if not model or util is None:
        # fallback random choices
        pool = [c for c in candidates if c != target]
        random.shuffle(pool)
        return pool[:top_k]
    
    # compute similarities and pick those moderately similar but not identical
    corpus = [c for c in candidates if c != target]
    if not corpus:
        return []
    
    try:
        # Encode target and corpus directly (simpler, no caching for now)
        emb_target = model.encode(target, convert_to_tensor=True)
        emb_corpus = model.encode(corpus, convert_to_tensor=True)
        
        # Compute cosine similarities
        sims = util.cos_sim(emb_target, emb_corpus)[0]
        paired = list(zip(corpus, sims.tolist()))
        
        paired.sort(key=lambda x: x[1], reverse=True)
        
        # choose top_k but ensure variety (avoid extremely high identical synonyms > 0.9 if possible)
        distractors = []
        for word, score in paired:
            if score > 0.98:  # nearly identical
                continue
            distractors.append(word)
            if len(distractors) >= top_k:
                break
        
        return distractors if distractors else paired[:top_k] if paired else []
        
    except Exception as e:
        # If anything fails, fallback to random choices
        pool = [c for c in corpus]
        random.shuffle(pool)
        return pool[:top_k]

def generate_mcqs(keywords: List[str], context: str, num: int = 5) -> List[Dict]:
    if not keywords:
        return []
    mcqs: List[Dict] = []
    base = keywords[:]
    random.shuffle(base)
    selected = base[:min(num, len(base))]
    
    for kw in selected:
        distractors = _semantic_distractors(kw, keywords, top_k=3)
        
        # Ensure distractors don't include the answer itself
        distractors = [d for d in distractors if d.lower() != kw.lower()]
        
        # If not enough distractors, pad with random different keywords
        if len(distractors) < 3:
            extras = [k for k in keywords if k.lower() != kw.lower() and k not in distractors]
            random.shuffle(extras)
            distractors += extras[: 3 - len(distractors)]
        
        # Ensure we have exactly 3 distractors
        distractors = distractors[:3]
        
        # Skip this question if we couldn't generate enough unique options
        if len(distractors) < 3:
            continue
        
        options = distractors + [kw]
        random.shuffle(options)
        
        # Create more specific question based on context
        question = f"Which term best relates to: '{kw}'?"
        
        # Try to find context sentence for better question phrasing
        sentences = [s.strip() for s in context.replace('\n', ' ').split('.') if s.strip()]
        for sent in sentences:
            if kw.lower() in sent.lower() and len(sent) < 150:
                # Extract a hint from the sentence
                question = f"Based on the lecture, what term is being described: related to {kw}?"
                break
        
        mcqs.append({
            "question": question,
            "options": options,
            "answer": kw
        })
    
    return mcqs


def generate_flashcards(keywords: List[str], context: str, num: int = 5, use_qa: bool = True) -> List[Dict]:
    cards: List[Dict] = []
    if not keywords:
        return cards
    slice_kw = keywords[:num]
    qa = _load_qa_pipeline() if use_qa else None
    # Pre-hash context (truncate huge contexts for hash stability & performance)
    ctx_hash = hashlib.sha1(context[:50000].encode('utf-8')).hexdigest()
    sentences = [s.strip() for s in context.replace('\n', ' ').split('.') if s.strip()]
    
    for kw in slice_kw:
        definition = ""
        
        # Strategy 1: Try QA model first
        if qa and len(context) < 20000:
            question = f"What is {kw}?"
            cache_key = (kw.lower(), ctx_hash)
            global _qa_answer_cache
            
            if cache_key in _qa_answer_cache:
                definition = _qa_answer_cache[cache_key]
            else:
                try:
                    answer = qa(question=question, context=context)
                    if answer and answer.get('answer') and len(answer['answer'].strip()) > 5:
                        ans_text = answer['answer'].strip()
                        # Ensure answer is meaningful (not just the keyword itself)
                        if ans_text.lower() != kw.lower():
                            definition = ans_text
                            with _cache_lock:
                                _qa_answer_cache[cache_key] = definition
                except Exception:
                    pass
        
        # Strategy 2: Find most relevant sentence containing the keyword
        if not definition:
            best_sentence = ""
            max_context_words = 0
            
            for sent in sentences:
                if kw.lower() in sent.lower() and len(sent) > 10:
                    # Prefer sentences with more content words (not just "X is Y")
                    context_words = len([w for w in sent.split() if len(w) > 3])
                    if context_words > max_context_words:
                        max_context_words = context_words
                        best_sentence = sent
            
            if best_sentence:
                # Clean up the sentence
                definition = best_sentence.strip()
                # If sentence is too long, try to extract just the relevant part
                if len(definition) > 200:
                    # Find the keyword position and extract surrounding context
                    kw_pos = definition.lower().find(kw.lower())
                    if kw_pos > 0:
                        start = max(0, kw_pos - 50)
                        end = min(len(definition), kw_pos + 150)
                        definition = "..." + definition[start:end].strip() + "..."
        
        # Strategy 3: Fallback with context-aware message
        if not definition:
            # Try to find ANY sentence with meaningful content
            if sentences:
                definition = f"A key concept discussed in the context of: {sentences[0][:100]}"
            else:
                definition = f"Important term mentioned in the lecture regarding {kw}."
        
        cards.append({"term": kw, "definition": definition})
    
    return cards


def export_flashcards_json(flashcards: List[Dict]) -> str:
    return json.dumps(flashcards, indent=2)

__all__ = [
    "extract_keywords",
    "generate_mcqs",
    "generate_flashcards",
    "export_flashcards_json"
]
