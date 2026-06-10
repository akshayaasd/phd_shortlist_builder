"""
Feedback scorer — applies historical outcome data to boost/penalise candidates.

Two mechanisms:
1. Supervisor-level EMA boost: supervisors with positive historical outcomes
   get a small recruitment_score boost; those with WRONG_PERSON get penalised.
2. NOT_RECRUITING suppression: suppressed supervisors are removed from results.
"""
from __future__ import annotations

import logging

from src.candidate import Candidate
from src.feedback.supervisor_db import get_supervisor_score, is_suppressed

log = logging.getLogger(__name__)

EMA_BOOST_WEIGHT = 0.05  # max ±0.05 adjustment to recruitment_score


def apply_feedback_scores(candidates: list[Candidate]) -> list[Candidate]:
    """
    Apply historical outcome data to adjust recruitment_score.
    Suppressed supervisors are removed.
    Returns filtered, adjusted candidate list.
    """
    passed = []
    suppressed_count = 0
    adjusted_count = 0

    for c in candidates:
        # Check suppression
        if is_suppressed(c.supervisor_id):
            log.debug(f"Suppressed (NOT_RECRUITING): {c.name}")
            suppressed_count += 1
            continue

        # Apply EMA boost if we have historical data
        ema = get_supervisor_score(c.supervisor_id)
        if ema is not None:
            # EMA ∈ [-1, 1] → scale to [-0.05, +0.05]
            adjustment = ema * EMA_BOOST_WEIGHT
            c.recruitment_score = max(0.0, min(1.0, c.recruitment_score + adjustment))
            adjusted_count += 1

        passed.append(c)

    log.info(
        f"Feedback scorer: {suppressed_count} suppressed, "
        f"{adjusted_count} scores adjusted from history"
    )
    return passed
