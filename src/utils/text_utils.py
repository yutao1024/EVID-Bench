"""Shared text utilities used by both retrieval and analysis stages.

Centralised here so prompts.py / agent_pipeline.py / tools.py don't redefine
the same tokenizer / overlap helpers.
"""

from __future__ import annotations

import re


_TEXT_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9']+")
_TEXT_TOKEN_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
        "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "this", "that", "these", "those", "it", "its", "as", "if",
        "than", "but", "not", "no", "yes", "all", "any", "some", "one",
        "two", "they", "them", "their", "there", "what", "when", "where",
        "why", "who", "how", "into", "onto",
        "video", "videos", "youtube", "watch", "vs", "ft", "feat",
        "featuring", "official", "channel", "your", "you", "our", "out",
        "have", "has", "had", "will",
    }
)


def tokenize_for_overlap(text: str) -> set[str]:
    """Lower-case alphanumeric tokens (>=3 chars) with English stopwords removed.

    Returns an empty set on empty / non-string input.
    """
    if not text:
        return set()
    toks = {t.lower() for t in _TEXT_TOKEN_RE.findall(text) if len(t) >= 3}
    return {t for t in toks if t not in _TEXT_TOKEN_STOPWORDS}


def keyword_overlap_score(query: str, observations: str, title: str) -> float:
    """Lightweight keyword-overlap scorer used as a VLM-rank fallback.

    Returns a value in [0, 1]; 0.5 when the context is too sparse to score.
    """
    context = " ".join(t for t in (query or "", observations or "") if t)
    q_tokens = tokenize_for_overlap(context)
    t_tokens = tokenize_for_overlap(title or "")
    if not q_tokens:
        return 0.5
    overlap = q_tokens & t_tokens
    return min(1.0, len(overlap) / max(len(q_tokens), 1))


AGGREGATOR_TITLE_MARKERS: tuple[str, ...] = (
    "compilation",
    "caught on camera",
    "moments",
    "montage",
    "best of",
    "top ",
    "amazing",
    "shocking",
    "epic ",
    "vlog",
    "daily life",
    "day in the life",
    "mix",
)


def is_aggregator_title(title: str) -> bool:
    if not title:
        return False
    t = title.lower()
    return any(m in t for m in AGGREGATOR_TITLE_MARKERS)


AGGREGATOR_QUERY_MARKERS: tuple[str, ...] = (
    "compilation",
    "caught on camera",
    "moments",
    "vlog",
    "tips",
    "how to",
    "tutorial",
    "best of",
)
