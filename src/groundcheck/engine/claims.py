"""Claim decomposition + verification core, shared by evaluate_faithfulness and
detect_hallucinations (one engine, two presentations).
"""

from __future__ import annotations

from pydantic import BaseModel

from groundcheck.judge.parsing import extract_json
from groundcheck.judge.prompts import (
    DECOMPOSE_PROMPT,
    DECOMPOSE_SYSTEM,
    VERIFY_PROMPT,
    VERIFY_SYSTEM,
)
from groundcheck.judge.sampler import Judge
from groundcheck.schemas import (
    ClaimVerdict,
    FaithfulnessResult,
    Hallucination,
    HallucinationResult,
    Source,
)


class DecomposedClaim(BaseModel):
    id: str
    claim: str
    answer_span: str


def _format_sources(sources: list[Source]) -> str:
    return "\n\n".join(f"[{s.id}]\n{s.text}" for s in sources)


async def decompose_claims(judge: Judge, answer: str) -> list[DecomposedClaim]:
    """One sampling call: split `answer` into atomic factual claims with their
    verbatim source span in the answer. Ids are assigned locally (c1, c2, ...)."""
    if not answer.strip():
        return []
    raw = await judge.complete(
        system=DECOMPOSE_SYSTEM,
        prompt=DECOMPOSE_PROMPT.format(answer=answer),
        max_tokens=2048,
    )
    parsed = extract_json(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"Judge returned non-list claims decomposition: {raw!r}")
    return [
        DecomposedClaim(id=f"c{i}", claim=item["claim"], answer_span=item["answer_span"])
        for i, item in enumerate(parsed, start=1)
    ]


async def verify_claims(
    judge: Judge, claims: list[DecomposedClaim], sources: list[Source]
) -> list[ClaimVerdict]:
    """One sampling call verifying all claims against all sources together."""
    if not claims:
        return []
    claims_block = "\n".join(f"{c.id}: {c.claim}" for c in claims)
    raw = await judge.complete(
        system=VERIFY_SYSTEM,
        prompt=VERIFY_PROMPT.format(claims=claims_block, sources=_format_sources(sources)),
        max_tokens=4096,
    )
    parsed = extract_json(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"Judge returned non-list claim verdicts: {raw!r}")

    by_id = {c.id: c for c in claims}
    verdicts = []
    for item in parsed:
        claim = by_id[item["id"]]
        verdicts.append(
            ClaimVerdict(
                claim=claim.claim,
                answer_span=claim.answer_span,
                verdict=item["verdict"],
                source_id=item.get("source_id"),
                quoted_span=item.get("quoted_span"),
                reason=item.get("reason"),
            )
        )
    return verdicts


async def run_claims_pipeline(
    judge: Judge, answer: str, sources: list[Source]
) -> list[ClaimVerdict]:
    """Decompose then verify. The shared core behind both 3.1 and 3.2."""
    claims = await decompose_claims(judge, answer)
    return await verify_claims(judge, claims, sources)


def summarize_faithfulness(verdicts: list[ClaimVerdict]) -> FaithfulnessResult:
    supported = sum(1 for v in verdicts if v.verdict == "supported")
    unsupported = sum(1 for v in verdicts if v.verdict == "unsupported")
    contradicted = sum(1 for v in verdicts if v.verdict == "contradicted")
    total = len(verdicts)
    return FaithfulnessResult(
        score=(supported / total) if total else 1.0,
        total_claims=total,
        supported=supported,
        unsupported=unsupported,
        contradicted=contradicted,
        claims=verdicts,
    )


async def evaluate_faithfulness(
    judge: Judge, answer: str, sources: list[Source]
) -> FaithfulnessResult:
    """Shared engine: decompose then verify, and tally the score."""
    verdicts = await run_claims_pipeline(judge, answer, sources)
    return summarize_faithfulness(verdicts)


def summarize_hallucinations(verdicts: list[ClaimVerdict]) -> HallucinationResult:
    problems = [v for v in verdicts if v.verdict != "supported"]
    return HallucinationResult(
        hallucinations=[
            Hallucination(
                answer_span=v.answer_span,
                closest_source_span=v.quoted_span or "(sources are silent on this)",
                source_id=v.source_id,
                reason=v.reason or "not addressed by any source",
            )
            for v in problems
        ]
    )


async def detect_hallucinations(
    judge: Judge, answer: str, sources: list[Source]
) -> HallucinationResult:
    """Shared engine, tight presentation: only the unsupported/contradicted claims."""
    verdicts = await run_claims_pipeline(judge, answer, sources)
    return summarize_hallucinations(verdicts)
