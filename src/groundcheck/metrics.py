"""Pure, deterministic retrieval and text-overlap metrics. No LLM calls, no I/O."""

from __future__ import annotations

import math
import re


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k retrieved ids that are relevant.

    Duplicate ids in `ranked_ids` are counted at each position they occur
    (retrieval systems shouldn't return duplicates, but we don't assume it).
    """
    if k <= 0 or not ranked_ids:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of all relevant ids captured within the top-k retrieved ids.

    Unlike `precision_at_k`, duplicate ids in `ranked_ids` are only counted
    once (recall measures set coverage, so re-retrieving the same relevant
    id twice can't push recall above 1.0).
    """
    if not relevant_ids or k <= 0 or not ranked_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant id (1-indexed); 0.0 if none found."""
    if not relevant_ids:
        return 0.0
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def dcg_at_k(gains: list[float], k: int) -> float:
    """Discounted cumulative gain over the first k entries of `gains`."""
    if k <= 0:
        return 0.0
    return sum(gain / math.log2(i + 2) for i, gain in enumerate(gains[:k]))


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, float], k: int) -> float:
    """Normalized DCG@k using graded relevance scores.

    `relevance` maps doc id -> a non-negative gain (e.g. 0-3 grade, or 1 for
    binary relevant/not). Ids absent from `relevance` are treated as gain 0.
    Returns 0.0 if no relevant ids exist (ideal DCG would be 0).
    """
    if k <= 0 or not ranked_ids or not relevance:
        return 0.0
    gains = [relevance.get(doc_id, 0.0) for doc_id in ranked_ids]
    actual = dcg_at_k(gains, k)
    ideal_gains = sorted(relevance.values(), reverse=True)
    ideal = dcg_at_k(ideal_gains, k)
    if ideal == 0.0:
        return 0.0
    return actual / ideal


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def token_overlap(text_a: str, text_b: str) -> float:
    """Jaccard similarity (intersection over union) of lowercased word tokens.

    Returns 0.0 if either text has no tokens (including two empty strings, by
    convention, since there is nothing to confirm overlap on).
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
