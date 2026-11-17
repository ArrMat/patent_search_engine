"""The eval's arithmetic only; the eval itself runs against the dev stack (uv run eval-search)."""

import pytest

from search_engine.evaluate import graded_metrics, metrics, sample, to_query


def test_graded_metrics():
    rel = {"a": 2, "b": 1}
    perfect = graded_metrics(["a", "b", "x"], rel)
    assert perfect == {"ndcg@10": 1.0, "recall@10": 1.0, "mrr@10": 1.0}
    swapped = graded_metrics(["x", "b", "a"], rel)
    assert 0 < swapped["ndcg@10"] < 1 and swapped["recall@10"] == 1.0 and swapped["mrr@10"] == 0.5
    assert graded_metrics(["x"], rel) == {"ndcg@10": 0.0, "recall@10": 0.0, "mrr@10": 0.0}


def test_metrics():
    m = metrics([1, 2, None, 10])
    assert m["recall@1"] == 0.25 and m["recall@5"] == 0.5 and m["recall@10"] == 0.75
    assert m["mrr@10"] == pytest.approx((1 + 1 / 2 + 1 / 10) / 4)


def test_sample_spreads_evenly_and_defaults_to_all():
    assert sample(list(range(10)), 3) == [0, 3, 6]
    assert sample([1, 2], None) == sample([1, 2], 5) == [1, 2]


def test_to_query_truncates_and_skips_short_text():
    assert to_query("a semicircular plate (") is None
    assert to_query(" ".join(["w"] * 100)) == " ".join(["w"] * 40)
