"""
Candidate dataclass — shared internal representation used across all pipeline stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paper:
    title: str
    year: int
    doi: Optional[str] = None
    cited_by_count: int = 0
    url: Optional[str] = None


@dataclass
class Grant:
    title: str
    funder: str
    grant_id: Optional[str] = None
    url: Optional[str] = None
    active: bool = True
    start_year: Optional[int] = None
    end_year: Optional[int] = None


@dataclass
class PIVerification:
    faculty_page_verified: bool = False
    active_grant_verified: bool = False
    recent_publication_verified: bool = False
    verification_sources: list[str] = field(default_factory=list)
    verification_score: float = 0.0


@dataclass
class Candidate:
    # Identity
    supervisor_id: str                      # openalex:AXXX or internal
    name: str
    institution: str
    country: str                            # ISO-3166 2-letter
    openalex_id: Optional[str] = None

    # Contact
    contact_email: Optional[str] = None
    email_inferred: bool = False

    # Bibliometrics (from OpenAlex)
    h_index: int = 0
    works_count: int = 0
    cited_by_count: int = 0
    first_pub_year: Optional[int] = None
    last_pub_year: Optional[int] = None
    recent_pubs_last_3_years: int = 0
    last_author_rate: float = 0.0           # fraction of papers as last/corresponding author
    topics: list[str] = field(default_factory=list)  # OpenAlex topic display names

    # Grants
    active_grant_count: int = 0
    grants: list[Grant] = field(default_factory=list)

    # Evidence
    papers: list[Paper] = field(default_factory=list)

    # Scores (computed during pipeline)
    career_stage_score: float = 0.0
    career_stage_low_confidence: bool = False
    domain_similarity: float = 0.0
    fit_confidence: float = 0.0

    # Verification
    pi_verification: PIVerification = field(default_factory=PIVerification)

    # Output fields
    why_match: str = ""
    tier: str = "target"
    recruitment_score: float = 0.0
    open_position_url: Optional[str] = None
    linked_program: Optional[str] = None
    disambiguation_warning: bool = False
    data_sources: list[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.supervisor_id)

    def __eq__(self, other):
        return isinstance(other, Candidate) and self.supervisor_id == other.supervisor_id
