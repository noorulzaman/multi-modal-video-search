import pytest

from src.search import (
    MIN_DISTINCT_TIME_GAP_SECONDS,
    SIMILARITY_THRESHOLD,
    STRONG_MATCH_SCORE,
    evaluate_best_match,
)


def candidate(score, timestamp):
    return {"score": score, "timestamp": timestamp, "path": None}


def test_no_candidates_is_rejected():
    result = evaluate_best_match([])

    assert result["accepted"] is False
    assert result["best_match"] is None


def test_below_threshold_is_rejected():
    result = evaluate_best_match([candidate(SIMILARITY_THRESHOLD - 0.01, 0.0)])

    assert result["accepted"] is False
    assert "minimum" in result["reason"]


def test_single_candidate_above_threshold_is_accepted():
    result = evaluate_best_match([candidate(SIMILARITY_THRESHOLD + 0.01, 3.0)])

    assert result["accepted"] is True
    assert result["score_margin"] is None


def test_clear_margin_over_distinct_competitor_is_accepted():
    far = MIN_DISTINCT_TIME_GAP_SECONDS + 1

    result = evaluate_best_match([candidate(0.24, 0.0), candidate(0.20, far)])

    assert result["accepted"] is True
    assert result["score_margin"] == pytest.approx(0.04)


def test_close_competitor_is_rejected_unless_strong():
    far = MIN_DISTINCT_TIME_GAP_SECONDS + 1

    weak = evaluate_best_match([candidate(0.215, 0.0), candidate(0.210, far)])
    strong = evaluate_best_match(
        [candidate(STRONG_MATCH_SCORE, 0.0), candidate(STRONG_MATCH_SCORE - 0.001, far)]
    )

    assert weak["accepted"] is False
    assert "competitor" in weak["reason"]
    assert strong["accepted"] is True


def test_nearby_frames_are_not_treated_as_competitors():
    near = MIN_DISTINCT_TIME_GAP_SECONDS - 1

    result = evaluate_best_match([candidate(0.215, 0.0), candidate(0.214, near)])

    assert result["competitor_score"] is None
    assert result["accepted"] is True

