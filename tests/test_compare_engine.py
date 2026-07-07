import json

from groundcheck.engine.compare import compare_answers
from tests.fakes import FakeJudge


def _verdict(overall: str, per_criterion: list[dict]) -> str:
    return json.dumps(
        {"overall_winner": overall, "rationale": f"{overall} wins", "per_criterion": per_criterion}
    )


class TestCompareAnswers:
    async def test_agreement_across_orderings_picks_winner(self):
        forward = _verdict(
            "a",
            [{"criterion": "faithfulness", "winner": "a", "rationale": "a is more faithful"}],
        )
        backward = _verdict(
            "a",
            [{"criterion": "faithfulness", "winner": "a", "rationale": "a is more faithful"}],
        )
        judge = FakeJudge([forward, backward])
        result = await compare_answers(
            judge, "query", "answer a", "answer b", None, ["faithfulness"]
        )
        assert result.winner == "a"
        assert result.criteria[0].winner == "a"
        assert len(judge.calls) == 2

    async def test_disagreement_across_orderings_yields_tie(self):
        forward = _verdict(
            "a",
            [{"criterion": "faithfulness", "winner": "a", "rationale": "a wins"}],
        )
        backward = _verdict(
            "b",
            [{"criterion": "faithfulness", "winner": "b", "rationale": "b wins"}],
        )
        judge = FakeJudge([forward, backward])
        result = await compare_answers(
            judge, "query", "answer a", "answer b", None, ["faithfulness"]
        )
        assert result.winner == "tie/uncertain"
        assert result.criteria[0].winner == "tie"
        assert "position" in result.criteria[0].rationale.lower()

    async def test_per_criterion_disagreement_independent_of_overall(self):
        forward = _verdict(
            "a",
            [
                {"criterion": "faithfulness", "winner": "a", "rationale": "a wins"},
                {"criterion": "completeness", "winner": "a", "rationale": "a wins"},
            ],
        )
        backward = _verdict(
            "a",
            [
                {"criterion": "faithfulness", "winner": "a", "rationale": "a wins"},
                {"criterion": "completeness", "winner": "b", "rationale": "b wins"},
            ],
        )
        judge = FakeJudge([forward, backward])
        result = await compare_answers(
            judge, "query", "a", "b", None, ["faithfulness", "completeness"]
        )
        assert result.winner == "a"
        by_criterion = {c.criterion: c for c in result.criteria}
        assert by_criterion["faithfulness"].winner == "a"
        assert by_criterion["completeness"].winner == "tie"
