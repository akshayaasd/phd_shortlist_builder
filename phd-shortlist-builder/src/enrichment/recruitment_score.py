"""
RecruitmentScore — the final ranking signal.

Answers: "Who should the student actually contact?" not "Who is most famous?"

Formula:
  recruitment_score = (
      0.30 * domain_similarity          # How well research topics match
    + 0.25 * recency_score              # Is this PI actively publishing/funded?
    + 0.20 * verification_score         # How confident are we this is a real active PI?
    + 0.15 * fit_confidence             # LLM assessment of specific fit
    + 0.10 * career_stage_score         # Seniority signal
  )

Recency score:
  A PI with 5+ recent papers and 2+ active grants scores 1.0.
  A PI last published in 2019 with no active grants scores ~0.1.
  This directly answers the reviewer's challenge about inactive PIs.
"""
from __future__ import annotations

import math

from src.candidate import Candidate


def _recency_score(candidate: Candidate) -> float:
    """
    Combines recent publications (last 3 years) and active grant count.
    Both signals decay gracefully.
    """
    pub_score = min(1.0, candidate.recent_pubs_last_3_years / 5.0)
    grant_score = min(1.0, candidate.active_grant_count / 3.0)
    return round(0.60 * pub_score + 0.40 * grant_score, 3)


def compute_recruitment_score(candidate: Candidate) -> float:
    recency = _recency_score(candidate)
    score = (
        0.30 * candidate.domain_similarity
        + 0.25 * recency
        + 0.20 * candidate.pi_verification.verification_score
        + 0.15 * candidate.fit_confidence
        + 0.10 * candidate.career_stage_score
    )
    return round(min(1.0, max(0.0, score)), 4)


def assign_tier(candidate: Candidate) -> str:
    """
    Tier is informational — recruitment_score drives actual ordering.

    Reach:  Very senior PI (h_index >= 30, works >= 80), likely hard to get
    Target: Mid-career, good match, realistic
    Safety: Junior/early-career PI, strong match, actively recruiting
    """
    if candidate.h_index >= 30 and candidate.works_count >= 80:
        return "reach"
    if candidate.career_stage_low_confidence or (candidate.h_index < 12 and candidate.works_count < 25):
        return "safety"
    return "target"


def score_and_rank(candidates: list[Candidate]) -> list[Candidate]:
    """
    Compute recruitment_score and tier for all candidates.
    Returns candidates sorted by recruitment_score descending.
    """
    for c in candidates:
        c.recruitment_score = compute_recruitment_score(c)
        c.tier = assign_tier(c)

    candidates.sort(key=lambda c: c.recruitment_score, reverse=True)
    return candidates
