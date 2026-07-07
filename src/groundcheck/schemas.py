"""Pydantic models shared across tools: inputs, outputs, and stored report shape."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

ResponseFormat = Literal["concise", "detailed"]

Verdict = Literal["supported", "unsupported", "contradicted"]

RetrievalMode = Literal["gold_labels", "llm_graded"]


class Source(BaseModel):
    id: str = Field(description="Stable identifier for this source chunk, e.g. 'doc1#chunk3'.")
    text: str = Field(description="The chunk text the answer may have been generated from.")


class ClaimVerdict(BaseModel):
    claim: str = Field(description="An atomic factual claim extracted from the answer.")
    verdict: Verdict
    source_id: str | None = Field(
        default=None, description="Id of the source that supports or contradicts the claim."
    )
    quoted_span: str | None = Field(
        default=None, description="Exact quoted span from the source backing the verdict."
    )
    reason: str | None = Field(
        default=None, description="One-line explanation, populated for unsupported/contradicted."
    )


class FaithfulnessResult(BaseModel):
    score: float = Field(description="supported claims / total claims, 0-1.")
    total_claims: int
    supported: int
    unsupported: int
    contradicted: int
    claims: list[ClaimVerdict] = Field(
        description="Concise: only problem claims. Detailed: every claim."
    )


class Hallucination(BaseModel):
    answer_span: str = Field(description="Exact unsupported/contradicted span from the answer.")
    closest_source_span: str = Field(
        description="Closest source passage that fails to support the span."
    )
    source_id: str | None = None
    reason: str = Field(description="One-line reason the span is unsupported or contradicted.")


class HallucinationResult(BaseModel):
    hallucinations: list[Hallucination] = Field(description="Empty list means the answer is clean.")


class RetrievedGrade(BaseModel):
    id: str
    grade: float = Field(description="0-3 LLM-judged relevance grade, only set in llm_graded mode.")


class RetrievalMetrics(BaseModel):
    k: int
    precision: float
    recall: float


class RetrievalResult(BaseModel):
    mode: RetrievalMode = Field(description="Which mode actually ran: gold_labels or llm_graded.")
    mrr: float
    ndcg: dict[int, float] = Field(description="NDCG@k for each requested k.")
    metrics: list[RetrievalMetrics] = Field(description="Precision/recall@k for each requested k.")
    graded_relevance: list[RetrievedGrade] | None = Field(
        default=None, description="Per-chunk LLM grades, only present in llm_graded mode."
    )


class CriterionVerdict(BaseModel):
    criterion: str
    winner: Literal["a", "b", "tie"]
    rationale: str


class CompareResult(BaseModel):
    winner: Literal["a", "b", "tie/uncertain"]
    criteria: list[CriterionVerdict]
    rationale: str = Field(description="Brief overall rationale, notes disagreement if any.")


class EvalCase(BaseModel):
    id: str
    query: str
    answer: str
    sources: list[Source]
    relevant_ids: list[str] | None = None


class CaseResult(BaseModel):
    id: str
    faithfulness: FaithfulnessResult
    retrieval: RetrievalResult | None = None


class SuiteSummary(BaseModel):
    report_id: str
    case_count: int
    mean_faithfulness: float
    mean_ndcg: dict[int, float] = Field(default_factory=dict)
    worst_cases: list[str] = Field(description="Ids of the 5 lowest-faithfulness cases.")


class Report(BaseModel):
    report_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = Field(default_factory=time.time)
    prompt_version: str
    cases: list[CaseResult]
    summary: SuiteSummary
