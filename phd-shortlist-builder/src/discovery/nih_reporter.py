"""
NIH RePORTER discovery module.

Queries the NIH RePORTER API for active grants whose abstracts match
the student's research interests, then extracts the PI as a candidate.

API docs: https://api.reporter.nih.gov/
IMPORTANT: We block fellowship/training grants (F31, F32, T32, K99)
because those list the awardee (a junior researcher), NOT a supervisor.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.candidate import Candidate, Grant
from src.profile_parser import StudentProfile

log = logging.getLogger(__name__)

NIH_API = "https://api.reporter.nih.gov/v2/projects/search"

# Grant types that list junior researchers as PI — never supervisors
BLOCKED_ACTIVITY_CODES = frozenset({
    "F30", "F31", "F32", "F33",           # NRSA fellowships (grad students/postdocs)
    "T32", "T34", "T35", "T90",           # training grants
    "K99", "R00",                          # career development / transition
    "K01", "K08", "K23", "K25",           # career awards
    "DP5",                                 # early independence award
})

import datetime
CURRENT_YEAR = datetime.date.today().year


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _post(client: httpx.AsyncClient, payload: dict) -> dict:
    resp = await client.post(NIH_API, json=payload, timeout=25)
    resp.raise_for_status()
    return resp.json()


def _is_blocked_activity_code(project: dict) -> bool:
    code = (project.get("activity_code") or "").upper()
    return code in BLOCKED_ACTIVITY_CODES


def _extract_grant(project: dict) -> Grant:
    return Grant(
        title=project.get("project_title") or "Untitled",
        funder="NIH",
        grant_id=project.get("full_project_num") or project.get("project_num"),
        url=f"https://reporter.nih.gov/project-details/{project.get('appl_id', '')}",
        active=True,
        start_year=int(str(project.get("project_start_date") or "")[:4]) if project.get("project_start_date") else None,
        end_year=int(str(project.get("project_end_date") or "")[:4]) if project.get("project_end_date") else None,
    )


def _parse_pi(project: dict) -> dict[str, str] | None:
    """Extract PI name and institution from a NIH project record."""
    pis = project.get("principal_investigators") or []
    if not pis:
        return None
    pi = pis[0]
    full_name = f"{pi.get('first_name', '')} {pi.get('last_name', '')}".strip()
    if not full_name or full_name == " ":
        return None

    org = project.get("organization") or {}
    institution = org.get("org_name") or ""
    country_name = org.get("org_country") or ""
    # NIH is US-only
    country = "US"

    return {
        "name": full_name,
        "institution": institution,
        "country": country,
        "email": pi.get("email") or None,
    }


async def discover_via_nih(
    profile: StudentProfile,
    client: httpx.AsyncClient,
    dry_run: bool = False,
) -> list[Candidate]:
    """
    Query NIH RePORTER for each research interest, extract PIs from
    active (non-fellowship) grants in target countries (US only for NIH).
    """
    if "US" not in profile.target_countries:
        log.info("NIH RePORTER: US not in target countries — skipping")
        return []

    query_terms = profile.research_interests.copy()
    if dry_run:
        query_terms = query_terms[:1]

    candidates_by_name_inst: dict[str, Candidate] = {}

    async def _query(term: str) -> list[Candidate]:
        local: list[Candidate] = []
        payload = {
            "criteria": {
                "advanced_text_search": {
                    "operator": "and",
                    "search_field": "all",
                    "search_text": term,
                },
                "project_end_date": {
                    "from_date": f"{CURRENT_YEAR - 1}-01-01",  # active in last year+
                },
                "is_active": True,
            },
            "limit": 25 if not dry_run else 5,
            "offset": 0,
            "include_fields": [
                "ProjectTitle", "AbstractText", "ProjectNum", "FullProjectNum",
                "ApplId", "ActivityCode", "PrincipalInvestigators",
                "Organization", "ProjectStartDate", "ProjectEndDate",
            ],
            "sort_field": "project_start_date",
            "sort_order": "desc",
        }
        try:
            data = await _post(client, payload)
            results = data.get("results") or []
            for project in results:
                if _is_blocked_activity_code(project):
                    log.debug(f"NIH: blocked activity code {project.get('activity_code')} — skipped")
                    continue
                pi_info = _parse_pi(project)
                if not pi_info:
                    continue

                key = f"{pi_info['name'].lower()}|{pi_info['institution'].lower()}"
                if key in candidates_by_name_inst:
                    # Add grant to existing candidate
                    candidates_by_name_inst[key].grants.append(_extract_grant(project))
                    candidates_by_name_inst[key].active_grant_count += 1
                else:
                    c = Candidate(
                        supervisor_id=f"nih:{key.replace(' ', '_')[:60]}",
                        name=pi_info["name"],
                        institution=pi_info["institution"],
                        country=pi_info["country"],
                        contact_email=pi_info.get("email"),
                        grants=[_extract_grant(project)],
                        active_grant_count=1,
                        data_sources=["nih_reporter"],
                    )
                    candidates_by_name_inst[key] = c
                    local.append(c)
        except Exception as exc:
            log.warning(f"NIH RePORTER query '{term}' failed: {exc}")
        return local

    tasks = [_query(term) for term in query_terms]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten — deduplication is by name+inst dict above
    seen = set()
    final: list[Candidate] = []
    for batch in results:
        if isinstance(batch, list):
            for c in batch:
                if c.supervisor_id not in seen:
                    seen.add(c.supervisor_id)
                    final.append(c)

    log.info(f"NIH RePORTER: discovered {len(final)} PI candidates")
    return final
