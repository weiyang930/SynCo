"""Cheap deduplication / novelty scoring for task texts.

Default uses character n-gram Jaccard overlap so we don't need any
embedding model. Returns values in [0, 1] where 1.0 means identical.
"""
from __future__ import annotations

import re
from typing import Iterable


_TOKEN_RE = re.compile(r"\w+")


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _ngrams(text: str, n: int) -> set[str]:
    norm = _normalize(text)
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def ngram_jaccard(a: str, b: str, n: int = 4) -> float:
    """Character n-gram Jaccard similarity in [0, 1]."""
    if not a or not b:
        return 0.0
    sa = _ngrams(a, n)
    sb = _ngrams(b, n)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def ngram_max_overlap(text: str, others: Iterable[str], n: int = 4) -> float:
    """Maximum Jaccard overlap of ``text`` against any element of ``others``."""
    best = 0.0
    for other in others:
        score = ngram_jaccard(text, other, n)
        if score > best:
            best = score
            if best >= 1.0:
                break
    return best


def normalized_hash(text: str) -> str:
    """Stable normalized hash of a text, useful as a strict-dup key."""
    import hashlib

    return hashlib.sha1(_normalize(text).encode("utf-8")).hexdigest()
