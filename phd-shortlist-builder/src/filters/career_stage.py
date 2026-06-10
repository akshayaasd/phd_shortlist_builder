"""
Career-stage scorer — weighted score, NOT a hard filter.

Design decision (v2): The original plan used a hard cutoff at 8 years
of publication history, which rejects valid junior assistant professors.
We now compute a weighted career_stage_score ∈ [0, 1] and set a soft
threshold: score < 0.35 → discard, 0.35–0.60 → keep with low_confidence flag.

Confirmed grant PIs (NIH/UKRI source) get a score floor of 0.60
because grant councils do not fund PhD students as principal investigators.

Signals and weights:
  1. Years since first publication (0.30)  — longer career → more senior
  2. Last-author rate (0.30)               — PIs appear last on their students' papers
  3. Total publications, log-normalised (0.20) — students have few papers
  4. Confirmed grant PI flag (0.20)        — binary, definitive proof

Thresholds:
  < 0.35  → discard (almost certainly a student / early postdoc)
  0.35–0.60 → keep, career_stage_low_confidence = True
  ≥ 0.60  → keep, career_stage_low_confidence = False
"""
from __future__ import annotations

import logging
import math

import datetime

from src.candidate import Candidate

log = logging.getLogger(__name__)
CURRENT_YEAR = datetime.date.today().year

DISCARD_THRESHOLD = 0.30
LOW_CONFIDENCE_THRESHOLD = 0.60
GRANT_PI_FLOOR = 0.60  # NIH/UKRI confirmed PI — cannot be a student


def _years_score(first_pub_year: int | None) -> float:
    if first_pub_year is None:
        return 0.3  # unknown — moderate penalty, don't discard
    years = CURRENT_YEAR - first_pub_year
    return min(1.0, years / 12.0)  # 12 years → 1.0 (captures mid-career PIs better)


def _works_score(works_count: int) -> float:
    if works_count <= 0:
        return 0.0
    return min(1.0, math.log(works_count + 1) / math.log(41))  # log(41) ≈ 3.71


def _grant_pi_flag(candidate: Candidate) -> float:
    """1.0 if discovered from a confirmed PI grant source."""
    pi_sources = {"nih_reporter", "ukri_gtr"}
    return 1.0 if any(s in pi_sources for s in candidate.data_sources) else 0.0


def compute_career_stage_score(candidate: Candidate) -> float:
    s_years = _years_score(candidate.first_pub_year)
    s_last = min(1.0, candidate.last_author_rate)
    s_works = _works_score(candidate.works_count)
    s_grant = _grant_pi_flag(candidate)

    score = (
        0.30 * s_years
        + 0.30 * s_last
        + 0.20 * s_works
        + 0.20 * s_grant
    )

    # Floor for confirmed grant PIs
    if s_grant == 1.0:
        score = max(score, GRANT_PI_FLOOR)

    return round(score, 3)


def apply_career_stage_filter(candidates: list[Candidate]) -> list[Candidate]:
    """
    Computes career_stage_score for each candidate.
    Discards below threshold; flags low-confidence range.
    Returns filtered list.
    """
    passed, discarded = [], 0

    for c in candidates:
        score = compute_career_stage_score(c)
        c.career_stage_score = score

        if score < DISCARD_THRESHOLD:
            log.debug(
                f"Career stage DISCARDED: {c.name} (score={score:.2f}, "
                f"years_since_first_pub={CURRENT_YEAR - (c.first_pub_year or CURRENT_YEAR)}, "
                f"works={c.works_count}, last_author_rate={c.last_author_rate:.2f})"
            )
            discarded += 1
        else:
            c.career_stage_low_confidence = score < LOW_CONFIDENCE_THRESHOLD
            passed.append(c)

    log.info(f"Career-stage filter: {discarded} discarded, {len(passed)} passed "
             f"({sum(1 for c in passed if c.career_stage_low_confidence)} low-confidence)")
    return passed
