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

from src.candidate import Candidate, Paper, Grant
from src.profile_parser import StudentProfile
from src.filters.country_gate import _normalise as _normalise_country

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
        "select": "id,title,publication_year,doi,cited_by_count,authorships,primary_location,awards",
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
        max_pos = max((a.get("author_position", "first") for a in auths if a), key=lambda x: {"first": 0, "middle": 1, "last": 2}.get(x, 0))
        last_auth = auths[-1] if auths else {}
        author_info = (last_auth or {}).get("author") or {}
        last_auth_id = author_info.get("id") or ""
        if clean_id and last_auth_id and clean_id in last_auth_id:
            last += 1
    return last / total if total > 0 else 0.0


def _extract_papers(works: list[dict]) -> list[Paper]:
    """Convert OpenAlex works to Paper objects (top 3 by citations)."""
    papers = []
    for w in works[:3]:
        doi_val = w.get("doi")
        if doi_val and not doi_val.startswith("http"):
            doi_val = f"https://doi.org/{doi_val}"
            
        url_val = None
        if w.get("primary_location"):
            url_val = w["primary_location"].get("landing_page_url")
        if not url_val:
            url_val = doi_val
        if not url_val:
            url_val = w.get("id")

        papers.append(Paper(
            title=w.get("title") or "Untitled",
            year=w.get("publication_year") or CURRENT_YEAR,
            doi=doi_val,
            cited_by_count=w.get("cited_by_count") or 0,
            url=url_val,
        ))
    return papers


def _count_recent(works: list[dict]) -> int:
    return sum(1 for w in works if (w.get("publication_year") or 0) >= CURRENT_YEAR - 3)


def _extract_grants_from_works(works: list[dict]) -> list[Grant]:
    """Extract unique grants from the author's works list."""
    seen_grant_ids = set()
    grants = []
    for w in works:
        for g_data in w.get("awards") or []:
            funder_name = g_data.get("funder_display_name") or "Unknown Funder"
            award_id = g_data.get("funder_award_id")
            
            # Use (funder, award_id) as key for deduplication
            key = (funder_name.lower(), (award_id or "").lower())
            if key in seen_grant_ids:
                continue
            seen_grant_ids.add(key)
            
            pub_year = w.get("publication_year")
            # If the work was published recently, the grant might still be active or recently active
            is_active = pub_year >= CURRENT_YEAR - 2 if pub_year else False
            
            title = f"Research Funding: {award_id}" if award_id else f"Research Funding via {funder_name}"
            
            grants.append(Grant(
                title=title,
                funder=funder_name,
                grant_id=award_id,
                url=g_data.get("id") or g_data.get("funder_id"),
                active=is_active,
                start_year=pub_year,
                end_year=None,
            ))
    return grants


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
    grants = _extract_grants_from_works(works)
    active_grants_count = sum(1 for g in grants if g.active)

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
        grants=grants,
        active_grant_count=active_grants_count,
        data_sources=["openalex"],
    )


async def discover_via_openalex(
    profile: StudentProfile,
    client: httpx.AsyncClient,
    per_query: int = 25,
    dry_run: bool = False,
) -> list[Candidate]:
    """
    Main entry point: queries OpenAlex works for each research interest + synonyms,
    extracts the PIs (last authors) in target countries, and builds Candidates.
    """
    # Normalize country codes (handles "Germany" → "DE", "UK" → "GB", etc.)
    normalised_countries = [_normalise_country(c) for c in profile.target_countries]
    target_countries = "|".join(normalised_countries)
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
            # 1. Search works (publications) matching the topic
            params = {
                "search": term,
                "filter": f"institutions.country_code:{target_countries},publication_year:>{CURRENT_YEAR - 5}",
                "sort": "cited_by_count:desc",
                "per-page": per_query,
            }
            works_data = await _get(client, f"{BASE}/works", params)
            works_results = works_data.get("results", [])

            # 2. Extract unique last author (PI) OpenAlex IDs from the works
            author_ids = []
            for work in works_results:
                auths = work.get("authorships", [])
                if auths:
                    # Last author is typically the PI/senior author
                    last_auth = auths[-1]
                    pi_author = (last_auth or {}).get("author") or {}
                    author_id = pi_author.get("id")
                    if author_id:
                        author_ids.append(author_id)
            
            author_ids = list(dict.fromkeys(author_ids))[:per_query]

            # 3. Fetch author details and their works in parallel
            author_details_tasks = [fetch_author_details(aid.split("/")[-1], client) for aid in author_ids]
            author_works_tasks = [fetch_recent_works(aid.split("/")[-1], client) for aid in author_ids]

            authors_resolved = await asyncio.gather(*author_details_tasks, return_exceptions=True)
            works_resolved = await asyncio.gather(*author_works_tasks, return_exceptions=True)

            for author, works in zip(authors_resolved, works_resolved):
                if isinstance(author, Exception) or isinstance(works, Exception):
                    continue
                oa_id = author.get("id", "")
                clean_id = oa_id.split("/")[-1]
                if not clean_id or clean_id in seen_ids:
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

