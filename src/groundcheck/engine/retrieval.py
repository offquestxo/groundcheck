"""Retrieval quality evaluation: Mode A (gold labels, deterministic) and Mode B (LLM-graded)."""

from __future__ import annotations

from groundcheck import metrics
from groundcheck.judge.parsing import extract_json
from groundcheck.judge.prompts import GRADE_RELEVANCE_PROMPT, GRADE_RELEVANCE_SYSTEM
from groundcheck.judge.sampler import Judge, NoJudgeAvailable
from groundcheck.schemas import RetrievalMetrics, RetrievalResult, RetrievedGrade, Source


def evaluate_with_gold_labels(
    retrieved: list[Source], relevant_ids: list[str], k_values: list[int]
) -> RetrievalResult:
    """Mode A: pure-math precision/recall/MRR/NDCG@k from a binary relevance set."""
    ranked_ids = [s.id for s in retrieved]
    relevant_set = set(relevant_ids)
    binary_relevance = {doc_id: 1.0 for doc_id in relevant_set}

    per_k = [
        RetrievalMetrics(
            k=k,
            precision=metrics.precision_at_k(ranked_ids, relevant_set, k),
            recall=metrics.recall_at_k(ranked_ids, relevant_set, k),
        )
        for k in k_values
    ]
    ndcg = {k: metrics.ndcg_at_k(ranked_ids, binary_relevance, k) for k in k_values}

    return RetrievalResult(
        mode="gold_labels",
        mrr=metrics.mrr(ranked_ids, relevant_set),
        ndcg=ndcg,
        metrics=per_k,
    )


def evaluate_with_grades(
    retrieved: list[Source], grades: dict[str, float], k_values: list[int]
) -> RetrievalResult:
    """Mode B: same metrics computed from LLM-assigned 0-3 relevance grades.

    A chunk is treated as "relevant" for precision/recall/MRR if its grade is
    >= 2 (the conventional graded-relevance cutoff for "relevant" vs
    "marginally/not relevant" on a 0-3 scale).
    """
    ranked_ids = [s.id for s in retrieved]
    relevant_set = {doc_id for doc_id, grade in grades.items() if grade >= 2}

    per_k = [
        RetrievalMetrics(
            k=k,
            precision=metrics.precision_at_k(ranked_ids, relevant_set, k),
            recall=metrics.recall_at_k(ranked_ids, relevant_set, k),
        )
        for k in k_values
    ]
    ndcg = {k: metrics.ndcg_at_k(ranked_ids, grades, k) for k in k_values}

    return RetrievalResult(
        mode="llm_graded",
        mrr=metrics.mrr(ranked_ids, relevant_set),
        ndcg=ndcg,
        metrics=per_k,
        graded_relevance=[RetrievedGrade(id=k, grade=v) for k, v in grades.items()],
    )


async def grade_relevance(judge: Judge, query: str, retrieved: list[Source]) -> dict[str, float]:
    """One sampling call: 0-3 relevance grade for every retrieved chunk."""
    if not retrieved:
        return {}
    chunks_block = "\n\n".join(f"[{s.id}]\n{s.text}" for s in retrieved)
    raw = await judge.complete(
        system=GRADE_RELEVANCE_SYSTEM,
        prompt=GRADE_RELEVANCE_PROMPT.format(query=query, chunks=chunks_block),
        max_tokens=2048,
    )
    parsed = extract_json(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"Judge returned non-list relevance grades: {raw!r}")
    return {item["id"]: float(item["grade"]) for item in parsed}


async def evaluate_retrieval(
    judge: Judge | None,
    query: str,
    retrieved: list[Source],
    relevant_ids: list[str] | None,
    k_values: list[int],
) -> RetrievalResult:
    """Single entry point: Mode A if `relevant_ids` given, else Mode B via `judge`."""
    if relevant_ids is not None:
        return evaluate_with_gold_labels(retrieved, relevant_ids, k_values)
    if judge is None:
        raise NoJudgeAvailable()
    grades = await grade_relevance(judge, query, retrieved)
    return evaluate_with_grades(retrieved, grades, k_values)
