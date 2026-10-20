"""Match-acceptance heuristic for visual search results."""

# Search settings. These are internal settings, not UI controls.
SIMILARITY_THRESHOLD = 0.20
MIN_SCORE_MARGIN = 0.01
STRONG_MATCH_SCORE = 0.27
MIN_DISTINCT_TIME_GAP_SECONDS = 5.0


def evaluate_best_match(candidates):
    """
    Evaluate the top candidate using absolute similarity and a
    temporal-diversity-aware score margin.

    This is a heuristic, not a calibrated probability.
    """

    if not candidates:
        return {
            "accepted": False,
            "best_match": None,
            "best_score": None,
            "competitor_score": None,
            "score_margin": None,
            "reason": "No valid indexed frame candidates were found.",
        }

    best_match = candidates[0]
    best_score = float(best_match["score"])

    distinct_competitors = [
        candidate
        for candidate in candidates[1:]
        if abs(
            float(candidate["timestamp"])
            - float(best_match["timestamp"])
        ) >= MIN_DISTINCT_TIME_GAP_SECONDS
    ]

    competitor = (
        distinct_competitors[0]
        if distinct_competitors
        else None
    )

    competitor_score = (
        float(competitor["score"])
        if competitor is not None
        else None
    )

    score_margin = (
        best_score - competitor_score
        if competitor_score is not None
        else None
    )

    passes_absolute_threshold = best_score >= SIMILARITY_THRESHOLD

    has_clear_margin = (
        score_margin is not None
        and score_margin >= MIN_SCORE_MARGIN
    )

    has_strong_absolute_score = best_score >= STRONG_MATCH_SCORE

    accepted = (
        passes_absolute_threshold
        and (
            score_margin is None
            or has_clear_margin
            or has_strong_absolute_score
        )
    )

    if not passes_absolute_threshold:
        reason = (
            "The best candidate did not meet the minimum "
            "similarity threshold."
        )
    elif accepted:
        reason = "The candidate passed the similarity heuristic."
    else:
        reason = (
            "The best candidate was too close to a distinct-time "
            "competitor and did not meet the stronger similarity "
            "requirement."
        )

    return {
        "accepted": accepted,
        "best_match": best_match,
        "best_score": best_score,
        "competitor_score": competitor_score,
        "score_margin": score_margin,
        "reason": reason,
    }
