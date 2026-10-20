import pytest

from src.evaluate import first_hit_rank, is_hit, summarize


def moment(start, end):
    return {"start_seconds": start, "end_seconds": end}


@pytest.mark.parametrize(
    "result, expected",
    [
        (moment(10, 12), True),     # inside
        (moment(14.5, 20), True),   # starts within tolerance after the label
        (moment(16, 20), False),    # too late
        (moment(0, 8.5), True),     # ends within tolerance before the label
        (moment(0, 7.5), False),    # too early
    ],
)
def test_is_hit_uses_overlap_with_tolerance(result, expected):
    assert is_hit(result, {"ranges": [[9.0, 14.0]]}) is expected


def test_first_hit_rank():
    label = {"ranges": [[30.0, 35.0]]}
    results = [moment(0, 5), moment(31, 33), moment(30, 31)]

    assert first_hit_rank(results, label) == 2
    assert first_hit_rank(results[:1], label) is None


def test_is_hit_accepts_any_of_several_ranges():
    label = {"ranges": [[5.0, 6.0], [40.0, 45.0]]}

    assert is_hit(moment(41, 43), label)
    assert not is_hit(moment(20, 25), label)


def test_summarize_computes_recall_and_mrr():
    rows = [
        {"config": "a", "type": "visual", "query": "q1", "rank": 1},
        {"config": "a", "type": "speech", "query": "q2", "rank": 4},
        {"config": "a", "type": "visual", "query": "q3", "rank": None},
    ]

    overall, _ = summarize(rows)

    assert overall.loc["a", "r@1"] == pytest.approx(1 / 3, abs=1e-3)
    assert overall.loc["a", "r@5"] == pytest.approx(2 / 3, abs=1e-3)
    assert overall.loc["a", "mrr"] == pytest.approx((1 + 0.25) / 3, abs=1e-3)
