"""
EURAXESS discovery module — timebox: 4 hours.

Scrapes EURAXESS job listings for open PhD positions with a named supervisor.
This is the only source that directly proves a professor is CURRENTLY recruiting.

We use the EURAXESS search API (undocumented but stable JSON endpoint).
Rate-limited to be polite.
"""
from __future__ import annotations

import asyncio
import logging
import re

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.candidate import Candidate
from src.profile_parser import StudentProfile

log = logging.getLogger(__name__)

EURAXESS_SEARCH = "https://euraxess.ec.europa.eu/api/jobs"

# ISO-3166 alpha-2 → EURAXESS country name mapping (partial, most common)
COUNTRY_MAP = {
    "GB": "United Kingdom",
    "US": None,        # EURAXESS is EU/EEA — US positions not listed
    "DE": "Germany",
    "FR": "France",
    "NL": "Netherlands",
    "SE": "Sweden",
    "CA": None,        # Not on EURAXESS
    "AU": None,        # Not on EURAXESS
    "CH": "Switzerland",
    "BE": "Belgium",
    "DK": "Denmark",
    "FI": "Finland",
    "NO": "Norway",
    "IT": "Italy",
    "ES": "Spain",
}


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=3, max=8))
async def _get(client: httpx.AsyncClient, params: dict) -> dict:
    resp = await client.get(EURAXESS_SEARCH, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _extract_supervisor(description: str) -> str | None:
    """
    Best-effort extraction of supervisor name from a job description.
    Looks for 'Supervisor: X', 'Contact: X', 'PI: X' patterns.
    """
    patterns = [
        r"[Ss]upervisor\s*:?\s*(?:Prof\.?|Dr\.?|Professor)?\s*([A-Z][a-z]+ [A-Z][a-z]+)",
        r"[Cc]ontact\s*:?\s*(?:Prof\.?|Dr\.?|Professor)?\s*([A-Z][a-z]+ [A-Z][a-z]+)",
        r"\bPI\b\s*:?\s*([A-Z][a-z]+ [A-Z][a-z]+)",
        r"[Gg]roup of\s+(?:Prof\.?|Dr\.?|Professor)\s+([A-Z][a-z]+ [A-Z][a-z]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, description)
        if m:
            return m.group(1).strip()
    return None


async def discover_via_euraxess(
    profile: StudentProfile,
    client: httpx.AsyncClient,
    dry_run: bool = False,
) -> list[Candidate]:
    """
    Search EURAXESS for open PhD positions in target countries.
    Yields candidates with open_position_url set — strongest recruiting signal.
    """
    euraxess_countries = [
        COUNTRY_MAP.get(c) for c in profile.target_countries
        if COUNTRY_MAP.get(c) is not None
    ]
    if not euraxess_countries:
        log.info("EURAXESS: no target countries overlap with EURAXESS — skipping")
        return []

    query_terms = profile.research_interests[:3] if not dry_run else profile.research_interests[:1]
    candidates: list[Candidate] = []
    seen_urls: set[str] = set()

    for term in query_terms:
        for country_name in euraxess_countries[:2]:  # cap to avoid too many requests
            params = {
                "keywords": term,
                "country": country_name,
                "type": "PhD",
                "page": 0,
                "size": 10 if not dry_run else 3,
            }
            try:
                data = await _get(client, params)
                jobs = data.get("content") or data.get("results") or []

                for job in jobs:
                    url = job.get("url") or job.get("link") or ""
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    title = job.get("title") or ""
                    description = job.get("description") or job.get("abstract") or ""
                    institution = job.get("organisation") or job.get("employer") or ""
                    country_code = next(
                        (k for k, v in COUNTRY_MAP.items() if v == country_name), "??"
                    )

                    supervisor_name = _extract_supervisor(description)
                    if not supervisor_name:
                        # Can't name a supervisor — still add as position-only record
                        supervisor_name = f"Supervisor at {institution}"

                    c = Candidate(
                        supervisor_id=f"euraxess:{hash(url) % 10**9}",
                        name=supervisor_name,
                        institution=institution,
                        country=country_code,
                        open_position_url=url or None,
                        linked_program=title[:200] if title else None,
                        data_sources=["euraxess"],
                    )
                    candidates.append(c)

                await asyncio.sleep(0.5)  # polite delay

            except Exception as exc:
                log.warning(f"EURAXESS query '{term}' / '{country_name}' failed: {exc}")

    log.info(f"EURAXESS: discovered {len(candidates)} open positions")
    return candidates
