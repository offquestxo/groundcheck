import json

import pytest

from groundcheck.engine.claims import (
    decompose_claims,
    detect_hallucinations,
    evaluate_faithfulness,
    verify_claims,
)
from groundcheck.schemas import Source
from tests.fakes import FakeJudge


def _decompose_response(*pairs: tuple[str, str]) -> str:
    return json.dumps([{"claim": c, "answer_span": a} for c, a in pairs])


def _verify_response(*items: dict) -> str:
    return json.dumps(list(items))


class TestDecomposeClaims:
    async def test_parses_json_array(self):
        judge = FakeJudge([_decompose_response(("The sky is blue.", "The sky is blue"))])
        claims = await decompose_claims(judge, "The sky is blue.")
        assert len(claims) == 1
        assert claims[0].id == "c1"
        assert claims[0].claim == "The sky is blue."
        assert claims[0].answer_span == "The sky is blue"
        assert len(judge.calls) == 1

    async def test_strips_markdown_fences(self):
        judge = FakeJudge(['```json\n[{"claim": "one claim", "answer_span": "one claim"}]\n```'])
        claims = await decompose_claims(judge, "one claim")
        assert claims[0].claim == "one claim"

    async def test_empty_answer_short_circuits(self):
        judge = FakeJudge([])
        claims = await decompose_claims(judge, "   ")
        assert claims == []
        assert judge.calls == []

    async def test_non_list_response_raises(self):
        judge = FakeJudge(['{"not": "a list"}'])
        with pytest.raises(ValueError):
            await decompose_claims(judge, "some answer")

    async def test_ids_assigned_sequentially(self):
        judge = FakeJudge([_decompose_response(("A", "A"), ("B", "B"), ("C", "C"))])
        claims = await decompose_claims(judge, "A B C")
        assert [c.id for c in claims] == ["c1", "c2", "c3"]


class TestVerifyClaims:
    async def test_parses_verdicts_and_carries_answer_span(self):
        from groundcheck.engine.claims import DecomposedClaim

        claims = [
            DecomposedClaim(id="c1", claim="The sky is blue.", answer_span="sky is blue"),
            DecomposedClaim(id="c2", claim="The moon is cheese.", answer_span="moon is cheese"),
        ]
        response = _verify_response(
            {
                "id": "c1",
                "verdict": "supported",
                "source_id": "doc1",
                "quoted_span": "the sky appears blue",
                "reason": None,
            },
            {
                "id": "c2",
                "verdict": "contradicted",
                "source_id": "doc1",
                "quoted_span": "the moon is rock",
                "reason": "directly conflicts",
            },
        )
        judge = FakeJudge([response])
        sources = [Source(id="doc1", text="the sky appears blue; the moon is rock")]
        verdicts = await verify_claims(judge, claims, sources)

        assert len(verdicts) == 2
        assert verdicts[0].verdict == "supported"
        assert verdicts[0].answer_span == "sky is blue"
        assert verdicts[1].verdict == "contradicted"
        assert verdicts[1].answer_span == "moon is cheese"

    async def test_empty_claims_short_circuits(self):
        judge = FakeJudge([])
        verdicts = await verify_claims(judge, [], [Source(id="a", text="x")])
        assert verdicts == []
        assert judge.calls == []


class TestEvaluateFaithfulness:
    async def test_full_pipeline_score(self):
        decompose_response = _decompose_response(
            ("Claim A is true.", "Claim A is true"), ("Claim B is false.", "Claim B is false")
        )
        verify_response = _verify_response(
            {
                "id": "c1",
                "verdict": "supported",
                "source_id": "s1",
                "quoted_span": "A is true",
                "reason": None,
            },
            {
                "id": "c2",
                "verdict": "unsupported",
                "source_id": None,
                "quoted_span": None,
                "reason": "not mentioned in sources",
            },
        )
        judge = FakeJudge([decompose_response, verify_response])
        sources = [Source(id="s1", text="A is true")]
        result = await evaluate_faithfulness(judge, "Claim A is true. Claim B is false.", sources)

        assert result.total_claims == 2
        assert result.supported == 1
        assert result.unsupported == 1
        assert result.contradicted == 0
        assert result.score == pytest.approx(0.5)
        assert len(judge.calls) == 2

    async def test_no_claims_scores_perfect(self):
        judge = FakeJudge(["[]"])
        result = await evaluate_faithfulness(judge, "   ", [Source(id="s1", text="x")])
        assert result.total_claims == 0
        assert result.score == 1.0


class TestDetectHallucinations:
    async def test_only_problem_claims_returned(self):
        decompose_response = _decompose_response(
            ("Claim A is true.", "Claim A is true"), ("Claim B is false.", "Claim B is false")
        )
        verify_response = _verify_response(
            {
                "id": "c1",
                "verdict": "supported",
                "source_id": "s1",
                "quoted_span": "A is true",
                "reason": None,
            },
            {
                "id": "c2",
                "verdict": "contradicted",
                "source_id": "s1",
                "quoted_span": "B is actually true",
                "reason": "directly conflicts",
            },
        )
        judge = FakeJudge([decompose_response, verify_response])
        sources = [Source(id="s1", text="A is true. B is actually true.")]
        result = await detect_hallucinations(judge, "Claim A is true. Claim B is false.", sources)

        assert len(result.hallucinations) == 1
        assert result.hallucinations[0].answer_span == "Claim B is false"
        assert result.hallucinations[0].closest_source_span == "B is actually true"
        assert result.hallucinations[0].reason == "directly conflicts"

    async def test_clean_answer_returns_empty(self):
        decompose_response = _decompose_response(("Claim A is true.", "Claim A is true"))
        verify_response = _verify_response(
            {
                "id": "c1",
                "verdict": "supported",
                "source_id": "s1",
                "quoted_span": "A is true",
                "reason": None,
            }
        )
        judge = FakeJudge([decompose_response, verify_response])
        result = await detect_hallucinations(
            judge, "Claim A is true.", [Source(id="s1", text="A is true")]
        )
        assert result.hallucinations == []

    async def test_missing_quoted_span_gets_fallback_text(self):
        decompose_response = _decompose_response(("Unaddressed claim.", "Unaddressed claim"))
        verify_response = _verify_response(
            {
                "id": "c1",
                "verdict": "unsupported",
                "source_id": None,
                "quoted_span": None,
                "reason": "not addressed",
            }
        )
        judge = FakeJudge([decompose_response, verify_response])
        result = await detect_hallucinations(
            judge, "Unaddressed claim.", [Source(id="s1", text="unrelated text")]
        )
        assert result.hallucinations[0].closest_source_span == "(sources are silent on this)"
