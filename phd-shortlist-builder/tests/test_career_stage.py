"""Tests for career-stage scorer."""
import pytest
from src.candidate import Candidate
from src.filters.career_stage import (
    compute_career_stage_score,
    apply_career_stage_filter,
    DISCARD_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
)

CURRENT_YEAR = __import__("datetime").date.today().year


def make_candidate(**kwargs) -> Candidate:
    defaults = dict(
        supervisor_id="test:1",
        name="Test PI",
        institution="Test University",
        country="US",
    )
    defaults.update(kwargs)
    return Candidate(**defaults)


def test_confirmed_grant_pi_never_discarded():
    """An NIH/UKRI grant PI must never be discarded regardless of publication history."""
    c = make_candidate(
        data_sources=["nih_reporter"],
        first_pub_year=CURRENT_YEAR - 4,  # only 4 years — junior
        works_count=5,
        last_author_rate=0.1,
    )
    score = compute_career_stage_score(c)
    assert score >= 0.60, f"Grant PI should have floor score of 0.60, got {score}"


def test_clear_student_discarded():
    """Someone with 2 years of publication history and 3 papers should be discarded."""
    c = make_candidate(
        data_sources=["openalex"],
        first_pub_year=CURRENT_YEAR - 2,
        works_count=3,
        last_author_rate=0.0,
        active_grant_count=0,
    )
    score = compute_career_stage_score(c)
    assert score < DISCARD_THRESHOLD, f"Student-like profile should score < {DISCARD_THRESHOLD}, got {score}"


def test_senior_pi_high_score():
    """A PI with 15+ years, 80 papers, 40% last-author rate should score well above discard."""
    c = make_candidate(
        data_sources=["openalex"],
        first_pub_year=CURRENT_YEAR - 16,
        works_count=85,
        last_author_rate=0.45,
        active_grant_count=2,
    )
    score = compute_career_stage_score(c)
    # Without a confirmed grant-source flag, score ~0.63; still well above discard threshold
    assert score >= DISCARD_THRESHOLD + 0.20, f"Senior PI should score well above discard, got {score}"
    assert score >= LOW_CONFIDENCE_THRESHOLD, f"Senior PI should not be low-confidence, got {score}"


def test_junior_assistant_professor_flagged_not_discarded():
    """Junior AP (5 years, 15 papers, some last-author) should pass but be flagged."""
    c = make_candidate(
        data_sources=["openalex"],
        first_pub_year=CURRENT_YEAR - 5,
        works_count=15,
        last_author_rate=0.30,
        active_grant_count=0,
    )
    score = compute_career_stage_score(c)
    assert score >= DISCARD_THRESHOLD, f"Junior AP should not be discarded, score={score}"
    assert score < LOW_CONFIDENCE_THRESHOLD, f"Junior AP should be low-confidence, score={score}"


def test_filter_retains_correct_counts():
    """Filter should correctly separate discarded vs passed candidates."""
    candidates = [
        make_candidate(supervisor_id="test:1", data_sources=["openalex"],
                       first_pub_year=CURRENT_YEAR - 1, works_count=2, last_author_rate=0.0),
        make_candidate(supervisor_id="test:2", data_sources=["nih_reporter"],
                       first_pub_year=CURRENT_YEAR - 3, works_count=5, last_author_rate=0.1),
        make_candidate(supervisor_id="test:3", data_sources=["openalex"],
                       first_pub_year=CURRENT_YEAR - 20, works_count=100, last_author_rate=0.5),
    ]
    passed = apply_career_stage_filter(candidates)
    # test:1 should be discarded (too junior); test:2 passes (grant PI); test:3 passes (senior)
    ids = [c.supervisor_id for c in passed]
    assert "test:1" not in ids
    assert "test:2" in ids
    assert "test:3" in ids
