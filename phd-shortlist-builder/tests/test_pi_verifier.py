"""Tests for PI verification logic."""
import pytest
from src.candidate import Candidate
from src.verification.pi_verifier import (
    _check_recent_publication,
    _check_active_grant,
    DISCARD_THRESHOLD,
)
from src.candidate import Paper, Grant

CURRENT_YEAR = __import__("datetime").date.today().year


def make_candidate(**kwargs) -> Candidate:
    return Candidate(supervisor_id="test:1", name="PI", institution="MIT", country="US", **kwargs)


def test_recent_pub_detected():
    c = make_candidate(
        papers=[Paper(title="Paper 2024", year=CURRENT_YEAR - 1, cited_by_count=10)],
        recent_pubs_last_3_years=1,
    )
    assert _check_recent_publication(c) is True


def test_old_pub_not_recent():
    c = make_candidate(
        papers=[Paper(title="Old Paper", year=CURRENT_YEAR - 6, cited_by_count=100)],
        recent_pubs_last_3_years=0,
    )
    assert _check_recent_publication(c) is False


def test_active_grant_from_nih_source():
    c = make_candidate(
        data_sources=["nih_reporter"],
        grants=[Grant(title="Active Grant", funder="NIH", active=True)],
        active_grant_count=1,
    )
    assert _check_active_grant(c) is True


def test_no_recent_pub_no_grant_would_be_discarded():
    """PI with nothing recent and no grants should fail verification threshold."""
    c = make_candidate(
        data_sources=["openalex"],
        recent_pubs_last_3_years=0,
        active_grant_count=0,
        papers=[Paper(title="Old", year=CURRENT_YEAR - 8, cited_by_count=5)],
    )
    # verification_score = 0.45*0 + 0.45*0 + 0.10*0 = 0.0 < DISCARD_THRESHOLD
    from src.candidate import PIVerification
    c.pi_verification = PIVerification(
        recent_publication_verified=False,
        active_grant_verified=False,
        faculty_page_verified=False,
        verification_score=0.0,
    )
    assert c.pi_verification.verification_score < DISCARD_THRESHOLD
