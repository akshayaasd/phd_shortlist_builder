import logging
import asyncio
from typing import Any
import httpx
from src.candidate import Candidate, Grant
from src.filters.disambiguation import _normalise_name, _normalise_inst
from src.discovery.nih_reporter import _is_blocked_activity_code, _extract_grant as _nih_extract_grant, _parse_pi
from src.discovery.ukri_gtr import _is_blocked as _ukri_is_blocked, _extract_grant as _ukri_extract_grant

log = logging.getLogger(__name__)

NIH_API = "https://api.reporter.nih.gov/v2/projects/search"
UKRI_API = "https://gtr.ukri.org/gtr/api"

async def enrich_candidate_grants(candidates: list[Candidate], client: httpx.AsyncClient) -> None:
    """
    Enriches candidates with active grants by performing targeted lookup
    by PI name on NIH RePORTER and UKRI GtR databases.
    """
    log.info(f"Starting targeted grant enrichment for {len(candidates)} candidates...")
    
    tasks = []
    for c in candidates:
        if c.country == "US":
            tasks.append(_enrich_nih(c, client))
        elif c.country in ("GB", "UK"):
            tasks.append(_enrich_ukri(c, client))
            
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
        
    log.info("Finished targeted grant enrichment.")

async def _enrich_nih(c: Candidate, client: httpx.AsyncClient) -> None:
    # Query NIH for PI name
    payload = {
        "criteria": {
            "pi_names": [{"any_name": c.name}],
            "is_active": True
        },
        "limit": 20,
        "include_fields": [
            "ProjectTitle", "AbstractText", "ProjectNum", "FullProjectNum",
            "ApplId", "ActivityCode", "PrincipalInvestigators",
            "Organization", "ProjectStartDate", "ProjectEndDate"
        ]
    }
    try:
        resp = await client.post(NIH_API, json=payload, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        
        c_name_norm = _normalise_name(c.name)
        c_inst_norm = _normalise_inst(c.institution)
        
        existing_ids = {g.grant_id for g in c.grants if g.grant_id}
        
        for project in results:
            if _is_blocked_activity_code(project):
                continue
                
            pi_info = _parse_pi(project)
            if not pi_info:
                continue
                
            # Verify name and institution match fuzzy
            pi_name_norm = _normalise_name(pi_info["name"])
            pi_inst_norm = _normalise_inst(pi_info["institution"])
            
            # Check if name tokens match
            name_match = (c_name_norm in pi_name_norm or pi_name_norm in c_name_norm or
                          set(c_name_norm.split()) == set(pi_name_norm.split()))
            
            inst_match = (c_inst_norm in pi_inst_norm or pi_inst_norm in c_inst_norm)
            
            if name_match and inst_match:
                grant_obj = _nih_extract_grant(project)
                if grant_obj.grant_id not in existing_ids:
                    c.grants.append(grant_obj)
                    existing_ids.add(grant_obj.grant_id)
                    if grant_obj.active:
                        c.active_grant_count += 1
                        
    except Exception as e:
        log.warning(f"Targeted NIH lookup for {c.name} failed: {e}")

async def _enrich_ukri(c: Candidate, client: httpx.AsyncClient) -> None:
    # Query UKRI for PI name as query term
    params = {
        "q": f'"{c.name}"',
        "p": 1,
        "s": 10,
        "f": "pro.gr",
    }
    headers = {"Accept": "application/vnd.rcuk.gtr.json-v7"}
    try:
        resp = await client.get(f"{UKRI_API}/projects", params=params, headers=headers, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        projects = data.get("project") or []
        
        c_name_norm = _normalise_name(c.name)
        c_inst_norm = _normalise_inst(c.institution)
        existing_ids = {g.grant_id for g in c.grants if g.grant_id}
        
        for project in projects:
            if _ukri_is_blocked(project):
                continue
                
            # Extract PI info
            people_links = project.get("links", {}).get("link", [])
            pi_name = None
            pi_inst = c.institution
            
            for link in people_links:
                if link.get("rel") == "PRINCIPAL_INVESTIGATOR":
                    href = link.get("href", "")
                    if href:
                        try:
                            # Fetch person info
                            person_resp = await client.get(href, headers=headers, timeout=20)
                            person_resp.raise_for_status()
                            person_data = person_resp.json()
                            pi_name = f"{person_data.get('firstName', '')} {person_data.get('surname', '')}".strip()
                        except Exception:
                            pass
                    break
            
            if not pi_name:
                continue
                
            # Find Lead Org
            org_links = project.get("links", {}).get("link", [])
            for ol in org_links:
                if ol.get("rel") == "LEAD_ORG":
                    pi_inst = ol.get("title", "")
                    break
                    
            pi_name_norm = _normalise_name(pi_name)
            pi_inst_norm = _normalise_inst(pi_inst)
            
            name_match = (c_name_norm in pi_name_norm or pi_name_norm in c_name_norm or
                          set(c_name_norm.split()) == set(pi_name_norm.split()))
            inst_match = (c_inst_norm in pi_inst_norm or pi_inst_norm in c_inst_norm)
            
            if name_match and inst_match:
                grant_ref = project.get("grantReference") or project.get("id") or ""
                grant_obj = _ukri_extract_grant(project, grant_ref)
                if grant_obj.grant_id not in existing_ids:
                    c.grants.append(grant_obj)
                    existing_ids.add(grant_obj.grant_id)
                    if grant_obj.active:
                        c.active_grant_count += 1
                        
    except Exception as e:
        log.warning(f"Targeted UKRI GtR lookup for {c.name} failed: {e}")
