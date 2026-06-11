"""
Profile parser — validates and normalises the student input JSON
into a structured StudentProfile dataclass used throughout the pipeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any


@dataclass
class Education:
    degree: str
    field: str
    institution: str
    graduation_year: int
    gpa: Optional[str] = None
    thesis: Optional[str] = None


@dataclass
class Publication:
    title: str
    venue: str
    year: int


@dataclass
class Project:
    title: str
    description: str


@dataclass
class TargetIntake:
    semester: str
    year: int


@dataclass
class StudentProfile:
    student_id: str
    education: list[Education]
    skills: list[str]
    projects: list[Project]
    publications: list[Publication]
    research_interests: list[str]          # 3–5 stated areas (hard constraint)
    target_countries: list[str]            # ISO-3166 2-letter codes (hard constraint)
    target_intake: TargetIntake
    nationality: str                        # student's nationality (ISO-3166)
    intro_call_summary: str
    raw_resume: str

    # Derived — populated by embedder
    interest_synonyms: dict[str, list[str]] = field(default_factory=dict)
    interest_embeddings: dict[str, list[float]] = field(default_factory=dict)

    # Optional enrichment fields (from profile JSON)
    internships: list[dict[str, Any]] = field(default_factory=list)
    achievements: list[str] = field(default_factory=list)

    def all_query_terms(self) -> list[str]:
        """Flat list of original interests + synonyms for API queries."""
        terms = list(self.research_interests)
        for synonyms in self.interest_synonyms.values():
            terms.extend(synonyms)
        return list(dict.fromkeys(terms))  # deduplicate, preserve order

    def background_summary(self) -> str:
        """Compact background string for why_match prompts."""
        degrees = " → ".join(
            f"{e.degree} in {e.field} ({e.institution})" for e in self.education
        )
        pub_titles = "; ".join(p.title for p in self.publications) or "no publications yet"
        internship_str = "; ".join(
            f"{i.get('role', '')} at {i.get('organization', '')}" for i in self.internships
        ) or ""
        achievement_str = ", ".join(self.achievements[:3]) or ""
        return (
            f"Degrees: {degrees}. "
            f"Skills: {', '.join(self.skills[:6])}. "
            f"Publications: {pub_titles}. "
            + (f"Industry experience: {internship_str}. " if internship_str else "")
            + (f"Achievements: {achievement_str}. " if achievement_str else "")
            + f"Intro: {self.intro_call_summary[:300]}"
        )


def load_profile(path: str | Path) -> StudentProfile:
    """Load and validate a student profile from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    # Validate required fields
    required = [
        "student_id", "education", "skills", "projects",
        "research_interests", "target_countries", "target_intake",
        "nationality", "intro_call_summary", "raw_resume",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Student profile missing required fields: {missing}")

    if len(data["research_interests"]) < 1:
        raise ValueError("research_interests must have at least 1 entry")

    if len(data["target_countries"]) < 1:
        raise ValueError("target_countries must have at least 1 entry")

    # Normalise country codes to uppercase
    data["target_countries"] = [c.upper() for c in data["target_countries"]]
    data["nationality"] = data["nationality"].upper()

    return StudentProfile(
        student_id=data["student_id"],
        education=[Education(**e) for e in data.get("education", [])],
        skills=data.get("skills", []),
        projects=[Project(**p) for p in data.get("projects", [])],
        publications=[Publication(**p) for p in data.get("publications", [])],
        research_interests=data["research_interests"],
        target_countries=data["target_countries"],
        target_intake=TargetIntake(**data["target_intake"]),
        nationality=data["nationality"],
        intro_call_summary=data["intro_call_summary"],
        raw_resume=data.get("raw_resume", ""),
        internships=data.get("internships", []),
        achievements=data.get("achievements", []),
    )
