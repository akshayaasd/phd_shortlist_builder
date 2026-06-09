"""
OpenAlex discovery module.

Queries the OpenAlex API to find candidate supervisors matching
the student's research interests in target countries.

API docs: https://docs.openalex.org/
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.candidate import Candidate, Paper
from src.profile_parser import StudentProfile

log = logging.getLogger(__name__)

BASE = "https://api.openalex.org"
EMAIL = os.getenv("OPENALEX_EMAIL", "phd-shortlist@ambitio.club")

# Current year used for recency calculations
import datetime
CURRENT_YEAR = datetime.date.today().year


def _polite(params: dict) -> dict:
    """Adds the mailto param for OpenAlex polite pool (faster rate limits)."""
    params["mailto"] = EMAIL
    return params


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _get(client: httpx.AsyncClient, url: str, params: dict) -> dict:
    resp = await client.get(url, params=_polite(params), timeout=20)
    resp.raise_for_status()
    return resp.json()


async def fetch_author_details(openalex_id: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """Fetch full author profile including h_index, works_count, topics."""
    url = f"{BASE}/authors/{openalex_id}"
    data = await _get(client, url, {})
    return data


async def fetch_recent_works(openalex_id: str, client: httpx.AsyncClient, limit: int = 20) -> list[dict]:
    """Fetch recent works for an author sorted by citation count."""
    params = {
        "filter": f"author.id:{openalex_id},publication_year:>{CURRENT_YEAR - 5}",
        "sort": "cited_by_count:desc",
        "per-page": limit,
        "select": "title,publication_year,doi,cited_by_count,authorships",
    }
    data = await _get(client, f"{BASE}/works", params)
    return data.get("results", [])


def _last_author_rate(works: list[dict], openalex_id: str) -> float:
    """
    Approximate rate at which this author appears as last author.
    Last author = highest position index in authorships list.
    """
    total, last = 0, 0
    clean_id = openalex_id.split("/")[-1]  # strip URL prefix if present
    for work in works:
        auths = work.get("authorships", [])
        if not auths:
            continue
        total += 1
        max_pos = max((a.get("author_position", "first") for a in auths), key=lambda x: {"first": 0, "middle": 1, "last": 2}.get(x, 0))
        last_auth = auths[-1] if auths else {}
        last_auth_id = (last_auth.get("author", {}) or {}).get("id", "")
        if clean_id in last_auth_id:
            last += 1
    return last / total if total > 0 else 0.0


def _extract_papers(works: list[dict]) -> list[Paper]:
    """Convert OpenAlex works to Paper objects (top 3 by citations)."""
    papers = []
    for w in works[:3]:
        papers.append(Paper(
            title=w.get("title") or "Untitled",
            year=w.get("publication_year") or CURRENT_YEAR,
            doi=f"https://doi.org/{w['doi']}" if w.get("doi") else None,
            cited_by_count=w.get("cited_by_count") or 0,
        ))
    return papers


def _count_recent(works: list[dict]) -> int:
    return sum(1 for w in works if (w.get("publication_year") or 0) >= CURRENT_YEAR - 3)


async def _build_candidate_from_author(
    author: dict,
    source_query: str,
    works: list[dict],
) -> Candidate | None:
    """Build a Candidate from an OpenAlex author + works payload."""
    oa_id = author.get("id", "")
    name = author.get("display_name", "Unknown")
    if not oa_id or not name:
        return None

    # Institution
    affiliations = author.get("last_known_institutions") or author.get("affiliations") or []
    if not affiliations:
        return None
    inst = affiliations[0]
    institution = inst.get("display_name", "")
    country = (inst.get("country_code") or "").upper()
    if not country:
        return None

    # Topics
    topics = [t.get("display_name", "") for t in (author.get("topics") or [])[:10]]

    # Bibliometrics
    summary = author.get("summary_stats") or {}
    h_index = summary.get("h_index") or author.get("h_index") or 0
    works_count = author.get("works_count") or 0
    cited_by = author.get("cited_by_count") or 0

    counts_by_year = author.get("counts_by_year") or []
    pub_years = [c["year"] for c in counts_by_year if c.get("works_count", 0) > 0]
    first_pub_year = min(pub_years) if pub_years else None
    last_pub_year = max(pub_years) if pub_years else None

    last_auth_rate = _last_author_rate(works, oa_id)
    recent_pubs = _count_recent(works)
    papers = _extract_papers(works)

    return Candidate(
        supervisor_id=f"openalex:{oa_id.split('/')[-1]}",
        openalex_id=oa_id,
        name=name,
        institution=institution,
        country=country,
        h_index=h_index,
        works_count=works_count,
        cited_by_count=cited_by,
        first_pub_year=first_pub_year,
        last_pub_year=last_pub_year,
        recent_pubs_last_3_years=recent_pubs,
        last_author_rate=last_auth_rate,
        topics=topics,
        papers=papers,
        data_sources=["openalex"],
    )


async def discover_via_openalex(
    profile: StudentProfile,
    client: httpx.AsyncClient,
    per_query: int = 25,
    dry_run: bool = False,
) -> list[Candidate]:
    """
    Main entry point: queries OpenAlex for each research interest + synonyms,
    filtered to target countries. Returns a deduplicated list of Candidates.
    """
    target_countries = "|".join(profile.target_countries)
    seen_ids: set[str] = set()
    candidates: list[Candidate] = []

    query_terms = profile.research_interests.copy()
    for synonyms in profile.interest_synonyms.values():
        query_terms.extend(synonyms)
    query_terms = list(dict.fromkeys(query_terms))  # deduplicate

    if dry_run:
        query_terms = query_terms[:2]
        per_query = 5

    async def _query_term(term: str) -> list[Candidate]:
        local_candidates: list[Candidate] = []
        try:
            params = {
                "search": term,
                "filter": f"last_known_institutions.country_code:{target_countries}",
                "sort": "cited_by_count:desc",
                "per-page": per_query,
                "select": "id,display_name,last_known_institutions,affiliations,topics,works_count,cited_by_count,counts_by_year,summary_stats",
            }
            data = await _get(client, f"{BASE}/authors", params)
            authors = data.get("results", [])

            # Fetch recent works for each author in parallel
            works_tasks = [
                fetch_recent_works(a["id"], client)
                for a in authors if a.get("id")
            ]
            all_works = await asyncio.gather(*works_tasks, return_exceptions=True)

            for author, works in zip(authors, all_works):
                if isinstance(works, Exception):
                    works = []
                oa_id = author.get("id", "")
                clean_id = oa_id.split("/")[-1]
                if clean_id in seen_ids:
                    continue
                seen_ids.add(clean_id)
                c = await _build_candidate_from_author(author, term, works)
                if c:
                    local_candidates.append(c)

        except Exception as exc:
            log.warning(f"OpenAlex query '{term}' failed: {exc}")
        return local_candidates

    tasks = [_query_term(term) for term in query_terms]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for batch in results:
        if isinstance(batch, list):
            candidates.extend(batch)

    log.info(f"OpenAlex: discovered {len(candidates)} candidates across {len(query_terms)} queries")
    return candidates
