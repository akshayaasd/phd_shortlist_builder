"""Tests for country gate."""
import pytest
from src.candidate import Candidate
from src.filters.country_gate import apply_country_gate
from src.profile_parser import StudentProfile, TargetIntake


def make_profile(countries: list[str]) -> StudentProfile:
    return StudentProfile(
        student_id="test",
        education=[], skills=[], projects=[], publications=[],
        research_interests=["test"],
        target_countries=countries,
        target_intake=TargetIntake(semester="Fall", year=2025),
        nationality="IN",
        intro_call_summary="",
        raw_resume="",
    )


def make_candidate(country: str, idx: int = 1) -> Candidate:
    return Candidate(
        supervisor_id=f"test:{idx}",
        name=f"PI {idx}",
        institution="Test Univ",
        country=country,
    )


def test_us_uk_filter():
    profile = make_profile(["US", "GB"])
    candidates = [
        make_candidate("US", 1),
        make_candidate("GB", 2),
        make_candidate("AU", 3),
        make_candidate("DE", 4),
    ]
    passed = apply_country_gate(candidates, profile)
    countries = {c.country for c in passed}
    assert "AU" not in countries
    assert "DE" not in countries
    assert len(passed) == 2


def test_uk_alias_accepted():
    """'UK' in source data should match 'GB' in target countries."""
    profile = make_profile(["GB"])
    candidates = [make_candidate("UK", 1)]
    passed = apply_country_gate(candidates, profile)
    assert len(passed) == 1


def test_empty_candidates():
    profile = make_profile(["US"])
    assert apply_country_gate([], profile) == []


def test_case_insensitive():
    profile = make_profile(["US"])
    c = make_candidate("us", 1)
    passed = apply_country_gate([c], profile)
    assert len(passed) == 1
