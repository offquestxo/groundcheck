import json

import pytest

from groundcheck.engine.retrieval import (
    evaluate_retrieval,
    evaluate_with_gold_labels,
    evaluate_with_grades,
    grade_relevance,
)
from groundcheck.judge.sampler import NoJudgeAvailable
from groundcheck.schemas import Source
from tests.fakes import FakeJudge


def _sources(*ids: str) -> list[Source]:
    return [Source(id=i, text=f"text for {i}") for i in ids]


class TestEvaluateWithGoldLabels:
    def test_perfect_retrieval(self):
        retrieved = _sources("a", "b", "c")
        result = evaluate_with_gold_labels(retrieved, ["a", "b", "c"], [3])
        assert result.mode == "gold_labels"
        assert result.mrr == 1.0
        assert result.metrics[0].precision == 1.0
        assert result.metrics[0].recall == 1.0
        assert result.ndcg[3] == pytest.approx(1.0)

    def test_no_relevant_found(self):
        retrieved = _sources("x", "y")
        result = evaluate_with_gold_labels(retrieved, ["a"], [2])
        assert result.mrr == 0.0
        assert result.metrics[0].precision == 0.0
        assert result.metrics[0].recall == 0.0
        assert result.ndcg[2] == 0.0

    def test_multiple_k_values(self):
        retrieved = _sources("a", "x", "b", "y")
        result = evaluate_with_gold_labels(retrieved, ["a", "b"], [1, 2, 4])
        by_k = {m.k: m for m in result.metrics}
        assert by_k[1].precision == 1.0
        assert by_k[2].precision == 0.5
        assert by_k[4].recall == 1.0

    def test_empty_relevant_ids(self):
        retrieved = _sources("a", "b")
        result = evaluate_with_gold_labels(retrieved, [], [2])
        assert result.mrr == 0.0
        assert result.metrics[0].recall == 0.0
        assert result.ndcg[2] == 0.0

    def test_empty_retrieved(self):
        result = evaluate_with_gold_labels([], ["a"], [3])
        assert result.mrr == 0.0
        assert result.metrics[0].precision == 0.0


class TestEvaluateWithGrades:
    def test_high_grade_counts_as_relevant(self):
        retrieved = _sources("a", "b", "c")
        grades = {"a": 3.0, "b": 1.0, "c": 0.0}
        result = evaluate_with_grades(retrieved, grades, [3])
        assert result.mode == "llm_graded"
        # only "a" clears the >=2 relevance cutoff
        assert result.metrics[0].recall == 1.0
        assert result.mrr == 1.0
        assert result.graded_relevance is not None
        assert len(result.graded_relevance) == 3

    def test_low_grades_yield_zero_metrics(self):
        retrieved = _sources("a", "b")
        grades = {"a": 1.0, "b": 0.0}
        result = evaluate_with_grades(retrieved, grades, [2])
        assert result.metrics[0].recall == 0.0
        assert result.mrr == 0.0

    def test_ndcg_uses_full_graded_scale(self):
        retrieved = _sources("a", "b")
        grades = {"a": 3.0, "b": 1.0}
        result = evaluate_with_grades(retrieved, grades, [2])
        # ranking already matches descending grade order -> perfect NDCG
        assert result.ndcg[2] == pytest.approx(1.0)


class TestGradeRelevance:
    async def test_parses_grades(self):
        response = json.dumps([{"id": "a", "grade": 3}, {"id": "b", "grade": 0}])
        judge = FakeJudge([response])
        grades = await grade_relevance(judge, "query", _sources("a", "b"))
        assert grades == {"a": 3.0, "b": 0.0}

    async def test_empty_retrieved_short_circuits(self):
        judge = FakeJudge([])
        grades = await grade_relevance(judge, "query", [])
        assert grades == {}
        assert judge.calls == []


class TestEvaluateRetrieval:
    async def test_mode_a_when_relevant_ids_given(self):
        judge = FakeJudge([])
        result = await evaluate_retrieval(judge, "q", _sources("a", "b"), ["a"], [2])
        assert result.mode == "gold_labels"
        assert judge.calls == []

    async def test_mode_b_when_relevant_ids_none(self):
        response = json.dumps([{"id": "a", "grade": 3}, {"id": "b", "grade": 0}])
        judge = FakeJudge([response])
        result = await evaluate_retrieval(judge, "q", _sources("a", "b"), None, [2])
        assert result.mode == "llm_graded"

    async def test_mode_b_without_judge_raises(self):
        with pytest.raises(NoJudgeAvailable):
            await evaluate_retrieval(None, "q", _sources("a"), None, [2])
