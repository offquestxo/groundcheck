import math

import pytest

from groundcheck.metrics import (
    dcg_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    token_overlap,
)


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"x", "y"}, 3) == 0.0

    def test_partial(self):
        assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 4) == 0.5

    def test_k_smaller_than_list(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 2) == 1.0

    def test_k_larger_than_list(self):
        # only 2 items exist; precision is computed over what's actually there
        assert precision_at_k(["a", "b"], {"a"}, 10) == 0.5

    def test_empty_ranked(self):
        assert precision_at_k([], {"a"}, 3) == 0.0

    def test_k_zero(self):
        assert precision_at_k(["a", "b"], {"a"}, 0) == 0.0

    def test_k_negative(self):
        assert precision_at_k(["a", "b"], {"a"}, -1) == 0.0

    def test_duplicate_ids(self):
        # duplicates count at each position
        assert precision_at_k(["a", "a", "b"], {"a"}, 3) == pytest.approx(2 / 3)

    def test_empty_relevant(self):
        assert precision_at_k(["a", "b"], set(), 2) == 0.0


class TestRecallAtK:
    def test_all_captured(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_none_captured(self):
        assert recall_at_k(["a", "b"], {"x"}, 2) == 0.0

    def test_partial(self):
        assert recall_at_k(["a", "b"], {"a", "b", "c"}, 2) == pytest.approx(2 / 3)

    def test_k_smaller_than_relevant_set(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, 1) == pytest.approx(1 / 3)

    def test_empty_relevant_ids(self):
        assert recall_at_k(["a", "b"], set(), 2) == 0.0

    def test_empty_ranked(self):
        assert recall_at_k([], {"a"}, 3) == 0.0

    def test_k_zero(self):
        assert recall_at_k(["a"], {"a"}, 0) == 0.0

    def test_duplicate_ids(self):
        # recall measures set coverage; re-retrieving "a" twice is still 1 hit
        assert recall_at_k(["a", "a"], {"a"}, 2) == 1.0


class TestMRR:
    def test_first_position(self):
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_third_position(self):
        assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_relevant_found(self):
        assert mrr(["x", "y"], {"a"}) == 0.0

    def test_empty_ranked(self):
        assert mrr([], {"a"}) == 0.0

    def test_empty_relevant(self):
        assert mrr(["a", "b"], set()) == 0.0

    def test_multiple_relevant_uses_first(self):
        assert mrr(["x", "a", "b"], {"a", "b"}) == pytest.approx(1 / 2)


class TestDCGAtK:
    def test_known_values(self):
        # DCG@3 for gains [3,2,1] = 3/log2(2) + 2/log2(3) + 1/log2(4)
        gains = [3.0, 2.0, 1.0]
        expected = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
        assert dcg_at_k(gains, 3) == pytest.approx(expected)

    def test_k_truncates(self):
        assert dcg_at_k([1.0, 1.0, 1.0], 1) == pytest.approx(1.0)

    def test_empty_gains(self):
        assert dcg_at_k([], 3) == 0.0

    def test_k_zero(self):
        assert dcg_at_k([1.0], 0) == 0.0


class TestNDCGAtK:
    def test_perfect_ranking_is_one(self):
        relevance = {"a": 3.0, "b": 2.0, "c": 1.0}
        assert ndcg_at_k(["a", "b", "c"], relevance, 3) == pytest.approx(1.0)

    def test_reversed_ranking_less_than_one(self):
        relevance = {"a": 3.0, "b": 2.0, "c": 1.0}
        score = ndcg_at_k(["c", "b", "a"], relevance, 3)
        assert 0.0 < score < 1.0

    def test_binary_relevance(self):
        relevance = {"a": 1.0}
        assert ndcg_at_k(["a", "x", "y"], relevance, 3) == pytest.approx(1.0)

    def test_no_relevant_ids(self):
        assert ndcg_at_k(["a", "b"], {}, 2) == 0.0

    def test_empty_ranked(self):
        assert ndcg_at_k([], {"a": 1.0}, 2) == 0.0

    def test_k_zero(self):
        assert ndcg_at_k(["a"], {"a": 1.0}, 0) == 0.0

    def test_unlisted_ids_treated_as_zero_gain(self):
        relevance = {"a": 2.0}
        # "z" isn't in relevance map -> gain 0, shouldn't crash
        score = ndcg_at_k(["z", "a"], relevance, 2)
        assert 0.0 < score < 1.0


class TestTokenOverlap:
    def test_identical_text(self):
        assert token_overlap("the cat sat", "the cat sat") == 1.0

    def test_no_overlap(self):
        assert token_overlap("apples oranges", "bananas grapes") == 0.0

    def test_partial_overlap(self):
        # {the,cat,sat} vs {the,dog,sat} -> intersection {the,sat}=2, union=4
        assert token_overlap("the cat sat", "the dog sat") == pytest.approx(0.5)

    def test_case_insensitive(self):
        assert token_overlap("The Cat", "the cat") == 1.0

    def test_empty_a(self):
        assert token_overlap("", "text") == 0.0

    def test_empty_b(self):
        assert token_overlap("text", "") == 0.0

    def test_both_empty(self):
        assert token_overlap("", "") == 0.0

    def test_punctuation_ignored(self):
        assert token_overlap("cat, dog!", "cat dog") == 1.0
