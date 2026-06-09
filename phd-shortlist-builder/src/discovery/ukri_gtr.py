"""
UKRI Gateway to Research discovery module.

Queries the UKRI GtR API for funded projects matching research interests,
extracts the Principal Investigator as a candidate.

API docs: https://gtr.ukri.org/resources/GtRAPI.html

IMPORTANT: We skip Studentship and MSCA Postdoctoral grants —
those list the awardee (junior researcher), not a supervisor.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.candidate import Candidate, Grant
from src.profile_parser import StudentProfile

log = logging.getLogger(__name__)

UKRI_API = "https://gtr.ukri.org/gtr/api"

# Grant categories that list junior researchers — not supervisors
BLOCKED_CATEGORIES = frozenset({
    "studentship",
    "msca postdoctoral fellowship",
    "msca doctoral network",
    "fellowship",         # broad — will refine by title check
})

import datetime
CURRENT_YEAR = datetime.date.today().year


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _get(client: httpx.AsyncClient, url: str, params: dict) -> dict:
    headers = {"Accept": "application/vnd.rcuk.gtr.json-v7"}
    resp = await client.get(url, params=params, headers=headers, timeout=25)
    resp.raise_for_status()
    return resp.json()


def _is_blocked(project: dict) -> bool:
    category = (project.get("category") or "").lower()
    title = (project.get("title") or "").lower()
    if category in BLOCKED_CATEGORIES:
        return True
    # Fellowships by title when category is generic
    if "studentship" in title or "fellowship" in title and "project" not in title:
        return True
    return False


def _extract_grant(project: dict, grant_ref: str) -> Grant:
    fund = project.get("fund") or {}
    funder = (fund.get("funder") or {}).get("name") or "UKRI"
    start = (fund.get("start") or "")[:4]
    end = (fund.get("end") or "")[:4]
    return Grant(
        title=project.get("title") or "Untitled",
        funder=funder,
        grant_id=grant_ref,
        url=f"https://gtr.ukri.org/project/{project.get('id', '')}",
        active=int(end) >= CURRENT_YEAR if end.isdigit() else False,
        start_year=int(start) if start.isdigit() else None,
        end_year=int(end) if end.isdigit() else None,
    )


async def discover_via_ukri(
    profile: StudentProfile,
    client: httpx.AsyncClient,
    dry_run: bool = False,
) -> list[Candidate]:
    """
    Query UKRI GtR for each research interest, extract PIs from funded projects.
    UK only (country_code GB).
    """
    if "GB" not in profile.target_countries and "UK" not in profile.target_countries:
        log.info("UKRI: UK not in target countries — skipping")
        return []

    query_terms = profile.research_interests.copy()
    if dry_run:
        query_terms = query_terms[:1]

    candidates_by_key: dict[str, Candidate] = {}

    async def _query(term: str) -> None:
        params = {
            "q": term,
            "p": 1,
            "s": 10,  # UKRI GtR API enforces a minimum page size of 10
            "f": "pro.gr",   # projects + grants
        }
        try:
            data = await _get(client, f"{UKRI_API}/projects", params)
            projects = (data.get("project") or [])

            for project in projects:
                if _is_blocked(project):
                    log.debug(f"UKRI: blocked project '{project.get('title','')[:40]}' — skipped")
                    continue

                grant_ref = project.get("grantReference") or project.get("id") or ""

                # Extract PI from project participants
                people_links = project.get("links", {}).get("link", [])
                pi_info: dict | None = None
                for link in people_links:
                    if link.get("rel") == "PRINCIPAL_INVESTIGATOR":
                        href = link.get("href", "")
                        if href:
                            try:
                                person_data = await _get(client, href, {})
                                org_links = project.get("links", {}).get("link", [])
                                institution = ""
                                for ol in org_links:
                                    if ol.get("rel") == "LEAD_ORG":
                                        institution = ol.get("title", "")
                                        break
                                pi_info = {
                                    "name": f"{person_data.get('firstName', '')} {person_data.get('surname', '')}".strip(),
                                    "institution": institution or project.get("leadOrganisationDepartment", ""),
                                    "email": person_data.get("email"),
                                }
                            except Exception:
                                pass
                        break

                if not pi_info or not pi_info.get("name"):
                    continue

                key = f"{pi_info['name'].lower()}|{pi_info['institution'].lower()}"
                grant_obj = _extract_grant(project, grant_ref)

                if key in candidates_by_key:
                    candidates_by_key[key].grants.append(grant_obj)
                    if grant_obj.active:
                        candidates_by_key[key].active_grant_count += 1
                else:
                    c = Candidate(
                        supervisor_id=f"ukri:{key.replace(' ', '_')[:60]}",
                        name=pi_info["name"],
                        institution=pi_info["institution"],
                        country="GB",
                        contact_email=pi_info.get("email"),
                        grants=[grant_obj],
                        active_grant_count=1 if grant_obj.active else 0,
                        data_sources=["ukri_gtr"],
                    )
                    candidates_by_key[key] = c

        except Exception as exc:
            log.warning(f"UKRI GtR query '{term}' failed: {exc}")

    tasks = [_query(term) for term in query_terms]
    await asyncio.gather(*tasks, return_exceptions=True)

    final = list(candidates_by_key.values())
    log.info(f"UKRI GtR: discovered {len(final)} PI candidates")
    return final
