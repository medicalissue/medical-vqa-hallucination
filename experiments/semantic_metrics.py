"""Semantically-aware metrics for generative VQA outputs.

The naive `_norm`-based bit-exact comparison treats "Yes, the lesion is
wedge-shaped." and "Yes, this lesion appears wedge-shaped." as DIFFERENT —
which is wrong for generative models. We add four alternative metrics:

  1. yes_no_token        — extract the first yes/no token (closed-form only)
  2. token_jaccard       — Jaccard over tokenized words (no embeddings needed)
  3. lenient_substr      — _contains_answer-style: does GT phrase appear in pred?
  4. embedding_cosine    — sentence-BERT cosine similarity

For "answer flip" we compute, between two predictions A and B:

  - flip_naive(A,B)         = 1{ _norm(A) ≠ _norm(B) }                  (current metric)
  - flip_yes_no(A,B)        = 1{ extract_yn(A) ≠ extract_yn(B) }        (closed only)
  - flip_jaccard(A,B,τ=0.5) = 1{ jaccard(toks(A), toks(B)) < τ }
  - flip_emb(A,B,τ=0.85)    = 1{ cos(emb(A), emb(B)) < τ }

We report all four for every probe; users can pick which to trust.

For comparing pred to GT we add:

  - acc_strict          — _norm(pred) == _norm(gt)
  - acc_lenient         — _contains_answer(pred, gt)  (existing)
  - acc_yes_no          — first yes/no token of pred == GT (closed only)
  - acc_jaccard(τ=0.3)  — token Jaccard ≥ τ
  - acc_emb(τ=0.7)      — embedding cosine ≥ τ
"""
from __future__ import annotations
import re
from typing import Sequence, Iterable
import numpy as np

YES_PAT = re.compile(r"\b(yes|yeah|yep|correct|true|positive|present)\b", re.I)
NO_PAT  = re.compile(r"\b(no|nope|incorrect|false|negative|absent|not present)\b", re.I)
PUNCT_RE = re.compile(r"[^\w\s]")


def normalize(s: str) -> str:
    return " ".join(str(s).lower().strip().split()).rstrip(".!? ")


def extract_yn(s: str) -> str | None:
    """Return 'yes' / 'no' / None based on first matching token."""
    s = str(s).lower().strip()
    m_yes = YES_PAT.search(s)
    m_no = NO_PAT.search(s)
    if m_yes and m_no:
        return "yes" if m_yes.start() < m_no.start() else "no"
    if m_yes: return "yes"
    if m_no:  return "no"
    return None


def tokenize(s: str) -> set[str]:
    s = PUNCT_RE.sub(" ", str(s).lower())
    return {t for t in s.split() if t and len(t) > 1}


def jaccard(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta and not tb: return 1.0
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)


def contains_answer(pred: str, gt: str) -> bool:
    p = normalize(pred); g = normalize(gt)
    if not g: return False
    if p == g: return True
    return f" {g} " in f" {p} " or p.startswith(g + " ") or p.endswith(" " + g) or p == g


# ---- batch flip computations ---------------------------------------------
def flip_naive(a: str, b: str) -> bool:
    return normalize(a) != normalize(b)


def flip_yes_no(a: str, b: str) -> bool | None:
    ya, yb = extract_yn(a), extract_yn(b)
    if ya is None or yb is None: return None
    return ya != yb


def flip_jaccard(a: str, b: str, threshold: float = 0.5) -> bool:
    return jaccard(a, b) < threshold


# ---- accuracy ------------------------------------------------------------
def acc_strict(pred: str, gt: str) -> bool:
    return normalize(pred) == normalize(gt)


def acc_lenient(pred: str, gt: str) -> bool:
    return contains_answer(pred, gt)


def acc_yes_no(pred: str, gt: str) -> bool | None:
    yp = extract_yn(pred); yg = extract_yn(gt)
    if yp is None or yg is None: return None
    return yp == yg


def acc_jaccard(pred: str, gt: str, threshold: float = 0.3) -> bool:
    return jaccard(pred, gt) >= threshold


# ---- optional: sentence-embedding metric (lazy-loaded) -------------------
_embedder_cache = {}


def _get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if model_name not in _embedder_cache:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedder_cache[model_name] = SentenceTransformer(model_name, device=device)
    return _embedder_cache[model_name]


def embed_texts(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    enc = _get_embedder(model_name).encode(texts, batch_size=64, show_progress_bar=False,
                                            normalize_embeddings=True, convert_to_numpy=True)
    return enc


def cosine_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(-1)
