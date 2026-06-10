"""Tests for domain guard Gate A (anchor concept check)."""
import pytest
from src.candidate import Candidate
from src.filters.domain_guard import gate_a_check
from src.profile_parser import StudentProfile, TargetIntake


def make_profile(interests: list[str]) -> StudentProfile:
    return StudentProfile(
        student_id="test",
        education=[], skills=[], projects=[], publications=[],
        research_interests=interests,
        target_countries=["US"],
        target_intake=TargetIntake(semester="Fall", year=2025),
        nationality="IN",
        intro_call_summary="",
        raw_resume="",
    )


def make_candidate(topics: list[str]) -> Candidate:
    return Candidate(
        supervisor_id="test:1", name="PI", institution="Univ", country="US",
        topics=topics,
    )


def test_correct_domain_passes():
    profile = make_profile(["computational neuroscience"])
    c = make_candidate(["Neural Coding", "Synaptic Plasticity", "Computational Neuroscience"])
    assert gate_a_check(c, profile) is True


def test_wrong_domain_rejected():
    """'DNA barcoding' for chromatin biology should not leak into a neuroscience student's list."""
    profile = make_profile(["computational neuroscience", "neuroimaging"])
    c = make_candidate(["DNA Barcoding", "Hi-C Chromatin", "Single Cell Sequencing"])
    assert gate_a_check(c, profile) is False


def test_humanities_rejected_for_stem():
    """Roman antiquity should not appear for a biomaterials student."""
    profile = make_profile(["biomaterials"])
    c = make_candidate(["Roman History", "Classical Antiquity", "Literary Analysis"])
    assert gate_a_check(c, profile) is False


def test_no_topics_passes_to_gate_b():
    """A candidate with no topics should not be discarded at Gate A (give benefit of doubt)."""
    profile = make_profile(["computational neuroscience"])
    c = make_candidate([])
    assert gate_a_check(c, profile) is True


def test_bci_matches_neuroscience():
    profile = make_profile(["brain-computer interfaces"])
    c = make_candidate(["EEG Signal Processing", "Motor Imagery BCI", "Neural Interface"])
    assert gate_a_check(c, profile) is True
