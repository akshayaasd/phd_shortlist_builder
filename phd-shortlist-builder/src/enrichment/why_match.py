"""
why_match generator — GPT-4o-mini structured output.

Generates a personalised 2–3 sentence match rationale for each candidate.
Rules enforced in prompt:
  - Must reference a SPECIFIC paper title or grant topic (not generic praise)
  - Must show how that work maps to the student's stated area
  - If fit is moderate, say so — do not oversell
  - Returns fit_confidence ∈ [0, 1] used as a ranking signal
"""
from __future__ import annotations

import asyncio
import json
import logging

from openai import AsyncOpenAI
from src.candidate import Candidate
from src.profile_parser import StudentProfile
from src.llm_client import LLMClient

log = logging.getLogger(__name__)
MODEL = "gpt-4o-mini"


async def generate_why_match(
    candidate: Candidate,
    profile: StudentProfile,
    client: AsyncOpenAI | LLMClient,
) -> None:
    """Generate why_match and fit_confidence for a single candidate (in-place)."""
    if not isinstance(client, LLMClient):
        client = LLMClient(client)

    paper_titles = "\n".join(
        f"  - {p.title} ({p.year}, cited {p.cited_by_count}x)"
        for p in candidate.papers[:3]
    ) or "  (no recent papers found)"

    grant_titles = "\n".join(
        f"  - {g.title} [{g.funder}]"
        for g in candidate.grants[:2]
    ) or "  (no grants found)"

    prompt = (
        "You are writing a match rationale for a PhD applicant's cold-email outreach.\n\n"
        f"Student research areas: {json.dumps(profile.research_interests)}\n"
        f"Student background: {profile.background_summary()}\n\n"
        f"Professor: {candidate.name} ({candidate.institution})\n"
        f"Professor's recent papers:\n{paper_titles}\n"
        f"Professor's active grants:\n{grant_titles}\n\n"
        "Write a 2–3 sentence why_match for this professor.\n"
        "Rules:\n"
        "  1. Reference a SPECIFIC paper title or grant topic by name.\n"
        "  2. Show concretely how that work connects to the student's background.\n"
        "  3. No generic praise ('leading expert', 'prolific researcher', 'renowned').\n"
        "  4. If the fit is only moderate, be honest — don't oversell.\n"
        "  5. Write from the student's perspective (first person is OK).\n\n"
        "Return ONLY JSON: "
        '{\"why_match\": \"...\", \"fit_confidence\": 0.0}'
        " where fit_confidence is 0.0–1.0."
    )

    try:
        result = await client.chat_complete_json(prompt)
        candidate.why_match = result.get("why_match") or ""
        candidate.fit_confidence = max(0.0, min(1.0, float(result.get("fit_confidence", 0.5))))
    except Exception as exc:
        log.warning(f"why_match generation failed for {candidate.name}: {exc}")
        candidate.why_match = f"Research in {', '.join(candidate.topics[:2])} aligns with student interests."
        candidate.fit_confidence = 0.5


async def generate_all_why_matches(
    candidates: list[Candidate],
    profile: StudentProfile,
    client: AsyncOpenAI | LLMClient,
    batch_size: int = 20,
) -> None:
    """
    Generate why_match for all candidates.
    Uses asyncio.gather in batches to avoid hitting rate limits.
    """
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i: i + batch_size]
        tasks = [generate_why_match(c, profile, client) for c in batch]
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info(f"why_match: generated {min(i + batch_size, len(candidates))}/{len(candidates)}")

