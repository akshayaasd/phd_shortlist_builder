"""
PI Verification — Stage 4.

Produces the pi_verification object that appears in every output record.
This is the single biggest differentiator for contamination reduction.

Three signals (weights reflect reliability):
  recent_publication_verified (0.45) — OpenAlex data, reliable
  active_grant_verified       (0.45) — grant API data, reliable
  faculty_page_verified       (0.10) — best-effort HTTP scrape, unreliable

Discard threshold: verification_score < 0.40
(Means: must have at least one of: recent pub OR active grant)

Design note on faculty_page_verified:
  University faculty pages are inconsistent (different URL patterns,
  JS rendering, outdated info). We mark it best_effort and weight it
  low (0.10). A failed scrape does NOT eliminate a candidate who has
  active NIH grants and published last year.
"""
from __future__ import annotations

import asyncio
import logging
import re

import httpx

from src.candidate import Candidate, PIVerification

log = logging.getLogger(__name__)
CURRENT_YEAR = __import__("datetime").date.today().year

DISCARD_THRESHOLD = 0.40

# Common university faculty page URL patterns to try
FACULTY_URL_PATTERNS = [
    "https://{domain}/people/{name_slug}",
    "https://{domain}/faculty/{name_slug}",
    "https://{domain}/staff/{name_slug}",
    "https://{domain}/~{name_slug}",
]

# Institution domain hints (partial — expand as needed)
INSTITUTION_DOMAINS: dict[str, str] = {
    "mit": "web.mit.edu",
    "stanford": "stanford.edu",
    "harvard": "harvard.edu",
    "oxford": "ox.ac.uk",
    "cambridge": "cam.ac.uk",
    "ucl": "ucl.ac.uk",
    "imperial": "imperial.ac.uk",
    "edinburgh": "ed.ac.uk",
    "manchester": "manchester.ac.uk",
    "berkeley": "berkeley.edu",
    "caltech": "caltech.edu",
    "columbia": "columbia.edu",
    "yale": "yale.edu",
    "princeton": "princeton.edu",
    "cornell": "cornell.edu",
    "toronto": "utoronto.ca",
}


def _name_slug(name: str) -> str:
    parts = name.lower().split()
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[-1]}"
    return parts[0] if parts else "unknown"


def _institution_domain(institution: str) -> str | None:
    inst_lower = institution.lower()
    for keyword, domain in INSTITUTION_DOMAINS.items():
        if keyword in inst_lower:
            return domain
    return None


async def _try_faculty_page(
    candidate: Candidate,
    client: httpx.AsyncClient,
) -> bool:
    """
    Best-effort: try common faculty page URL patterns.
    Returns True if we find a page that contains the candidate's last name.
    """
    domain = _institution_domain(candidate.institution)
    if not domain:
        return False

    slug = _name_slug(candidate.name)
    last_name = candidate.name.split()[-1].lower() if candidate.name.split() else ""
    headers = {"User-Agent": "Mozilla/5.0 (academic research bot; contact@ambitio.club)"}

    for pattern in FACULTY_URL_PATTERNS:
        url = pattern.format(domain=domain, name_slug=slug)
        try:
            resp = await client.get(url, headers=headers, timeout=8, follow_redirects=True)
            if resp.status_code == 200 and last_name in resp.text.lower():
                log.debug(f"Faculty page found for {candidate.name}: {url}")
                return True
        except Exception:
            continue

    return False


def _check_recent_publication(candidate: Candidate) -> bool:
    """True if any paper or count_by_year entry is within last 3 years."""
    if candidate.recent_pubs_last_3_years > 0:
        return True
    for paper in candidate.papers:
        if paper.year >= CURRENT_YEAR - 3:
            return True
    return False


def _check_active_grant(candidate: Candidate) -> bool:
    """True if any grant is flagged active OR candidate came from grant API."""
    if candidate.active_grant_count > 0:
        return True
    for grant in candidate.grants:
        if grant.active:
            return True
    # If sourced from NIH/UKRI, grants were filtered to active already
    if any(s in {"nih_reporter", "ukri_gtr"} for s in candidate.data_sources):
        return True
    return False


async def verify_pi(
    candidate: Candidate,
    http_client: httpx.AsyncClient,
) -> None:
    """
    Compute pi_verification in-place for a single candidate.
    """
    recent_pub = _check_recent_publication(candidate)
    active_grant = _check_active_grant(candidate)
    faculty_page = await _try_faculty_page(candidate, http_client)

    score = (
        0.45 * int(recent_pub)
        + 0.45 * int(active_grant)
        + 0.10 * int(faculty_page)
    )

    sources = []
    if recent_pub:
        sources.append("openalex_recent_pub")
    if active_grant:
        grant_sources = [s for s in candidate.data_sources if s in {"nih_reporter", "ukri_gtr"}]
        if grant_sources:
            sources.extend(grant_sources)
        else:
            sources.append("grant_data")
    if faculty_page:
        sources.append("faculty_page")

    candidate.pi_verification = PIVerification(
        faculty_page_verified=faculty_page,
        active_grant_verified=active_grant,
        recent_publication_verified=recent_pub,
        verification_sources=sources,
        verification_score=round(score, 3),
    )


async def apply_pi_verification(
    candidates: list[Candidate],
    http_client: httpx.AsyncClient,
) -> list[Candidate]:
    """
    Run PI verification on all candidates in parallel.
    Discard any with verification_score < DISCARD_THRESHOLD.
    """
    tasks = [verify_pi(c, http_client) for c in candidates]
    await asyncio.gather(*tasks, return_exceptions=True)

    passed, discarded = [], 0
    for c in candidates:
        if c.pi_verification.verification_score >= DISCARD_THRESHOLD:
            passed.append(c)
        else:
            log.debug(
                f"PI Verification DISCARDED: {c.name} "
                f"(score={c.pi_verification.verification_score:.2f}, "
                f"recent_pub={c.pi_verification.recent_publication_verified}, "
                f"active_grant={c.pi_verification.active_grant_verified})"
            )
            discarded += 1

    log.info(f"PI Verification: {discarded} discarded, {len(passed)} passed")
    return passed
