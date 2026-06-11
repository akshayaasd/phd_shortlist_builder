"""
Output schema and JSON writer.

Pydantic v2 models enforce the documented schema at serialisation time.
The metadata block captures filter rejection logs for transparency.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.candidate import Candidate


# --- Pydantic output models ---

class PaperOut(BaseModel):
    title: str
    year: int
    doi: Optional[str] = None
    url: Optional[str] = None
    cited_by_count: int = 0


class GrantOut(BaseModel):
    title: str
    funder: str
    grant_id: Optional[str] = None
    url: Optional[str] = None
    active: bool = True


class EvidenceOut(BaseModel):
    papers: list[PaperOut] = Field(default_factory=list)
    grants: list[GrantOut] = Field(default_factory=list)


class PIVerificationOut(BaseModel):
    faculty_page_verified: bool
    active_grant_verified: bool
    recent_publication_verified: bool
    verification_score: float
    verification_sources: list[str] = Field(default_factory=list)


class RecencyOut(BaseModel):
    recent_pubs_last_3_years: int
    active_grant_count: int
    latest_publication_year: Optional[int] = None


class SupervisorOut(BaseModel):
    rank: int
    supervisor_id: str
    name: str
    institution: str
    country: str
    contact_email: Optional[str] = None
    email_inferred: bool = False
    research_focus: list[str] = Field(default_factory=list)
    evidence: EvidenceOut
    why_match: str
    fit_confidence: float
    tier: str
    recruitment_score: float
    career_stage_score: float
    career_stage_low_confidence: bool
    pi_verification: PIVerificationOut
    recency: RecencyOut
    open_position_url: Optional[str] = None
    linked_program: Optional[str] = None
    disambiguation_warning: bool
    data_sources: list[str]


class FilterLog(BaseModel):
    country_rejected: int = 0
    domain_leakage_gate_a: int = 0
    domain_leakage_gate_b: int = 0
    career_stage_rejected: int = 0
    verification_rejected: int = 0
    disambig_discarded: int = 0


class SummaryOut(BaseModel):
    total_recommendations: int
    papers_and_grants_count: int
    papers_only_count: int
    grants_only_count: int
    total_candidates_discovered: int
    total_after_filtering: int
    coverage_by_area: dict[str, int] = Field(default_factory=dict)
    country_distribution: dict[str, int] = Field(default_factory=dict)
    filter_rejection_log: FilterLog


class ShortlistGroups(BaseModel):
    papers_and_grants: list[SupervisorOut]
    papers_only: list[SupervisorOut]
    grants_only: list[SupervisorOut]


class Shortlist(BaseModel):
    student_id: str
    generated_at: str
    summary: SummaryOut
    shortlist: ShortlistGroups


# --- Builder ---

def build_shortlist(
    student_id: str,
    candidates: list[Candidate],
    metadata_dict: dict,  # We will pass the raw metadata dict from run.py
) -> Shortlist:
    
    papers_and_grants_cands = []
    papers_only_cands = []
    grants_only_cands = []

    # Categorize candidates based on evidence availability
    for c in candidates:
        has_papers = len(c.papers) > 0
        has_grants = len(c.grants) > 0
        
        if has_papers and has_grants:
            papers_and_grants_cands.append(c)
        elif has_papers:
            papers_only_cands.append(c)
        else:
            grants_only_cands.append(c)
            
    # Sort each group by recruitment_score
    papers_and_grants_cands.sort(key=lambda x: x.recruitment_score, reverse=True)
    papers_only_cands.sort(key=lambda x: x.recruitment_score, reverse=True)
    grants_only_cands.sort(key=lambda x: x.recruitment_score, reverse=True)

    def _build_sups(cand_list: list[Candidate]) -> list[SupervisorOut]:
        sups = []
        for rank, c in enumerate(cand_list, start=1):
            v = c.pi_verification
            sups.append(SupervisorOut(
                rank=rank,
                supervisor_id=c.supervisor_id,
                name=c.name,
                institution=c.institution,
                country=c.country,
                contact_email=c.contact_email,
                email_inferred=c.email_inferred,
                research_focus=c.topics[:5],
                evidence=EvidenceOut(
                    papers=[PaperOut(title=p.title, year=p.year, doi=p.doi, url=p.url, cited_by_count=p.cited_by_count) for p in c.papers],
                    grants=[GrantOut(title=g.title, funder=g.funder, grant_id=g.grant_id, url=g.url, active=g.active) for g in c.grants],
                ),
                why_match=c.why_match,
                fit_confidence=c.fit_confidence,
                tier=c.tier,
                recruitment_score=c.recruitment_score,
                career_stage_score=c.career_stage_score,
                career_stage_low_confidence=c.career_stage_low_confidence,
                pi_verification=PIVerificationOut(
                    faculty_page_verified=v.faculty_page_verified,
                    active_grant_verified=v.active_grant_verified,
                    recent_publication_verified=v.recent_publication_verified,
                    verification_score=v.verification_score,
                    verification_sources=v.verification_sources,
                ),
                recency=RecencyOut(
                    recent_pubs_last_3_years=c.recent_pubs_last_3_years,
                    active_grant_count=c.active_grant_count,
                    latest_publication_year=c.last_pub_year,
                ),
                open_position_url=c.open_position_url,
                linked_program=c.linked_program,
                disambiguation_warning=c.disambiguation_warning,
                data_sources=c.data_sources,
            ))
        return sups

    summary = SummaryOut(
        total_recommendations=len(candidates),
        papers_and_grants_count=len(papers_and_grants_cands),
        papers_only_count=len(papers_only_cands),
        grants_only_count=len(grants_only_cands),
        total_candidates_discovered=metadata_dict.get("total_candidates_discovered", 0),
        total_after_filtering=metadata_dict.get("total_after_filtering", 0),
        coverage_by_area=metadata_dict.get("coverage_by_area", {}),
        country_distribution=metadata_dict.get("country_distribution", {}),
        filter_rejection_log=FilterLog(**metadata_dict.get("filter_rejection_log", {})),
    )

    return Shortlist(
        student_id=student_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        summary=summary,
        shortlist=ShortlistGroups(
            papers_and_grants=_build_sups(papers_and_grants_cands),
            papers_only=_build_sups(papers_only_cands),
            grants_only=_build_sups(grants_only_cands),
        )
    )


def write_shortlist(shortlist: Shortlist, output_dir: str = "sample_output") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{shortlist.student_id}.json"
    out_path.write_text(
        shortlist.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return out_path
