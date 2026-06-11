"""
Paper enricher.

Searches OpenAlex for candidates discovered via grant databases
who currently have no papers in their profile.
"""
import asyncio
import logging
import httpx

from src.candidate import Candidate
from src.discovery.openalex import fetch_recent_works, _extract_papers, _count_recent, _polite, BASE

log = logging.getLogger(__name__)

from src.filters.disambiguation import _normalise_name, _normalise_inst

async def _enrich_single_candidate(candidate: Candidate, client: httpx.AsyncClient):
    # Only enrich if they don't have papers
    if candidate.papers:
        return
        
    try:
        # Search OpenAlex for author by name
        params = _polite({
            "search": candidate.name,
            "sort": "works_count:desc",
            "per-page": 5
        })
        resp = await client.get(f"{BASE}/authors", params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        
        results = data.get("results", [])
        if not results:
            return
            
        c_name_norm = _normalise_name(candidate.name)
        c_inst_norm = _normalise_inst(candidate.institution)
        
        matched_author = None
        for author in results:
            # Check name match
            author_name = author.get("display_name") or ""
            author_name_norm = _normalise_name(author_name)
            name_match = (c_name_norm in author_name_norm or author_name_norm in c_name_norm or
                          set(c_name_norm.split()) == set(author_name_norm.split()))
            if not name_match:
                continue
                
            # Check institution match
            affiliations = author.get("affiliations") or []
            last_known = author.get("last_known_institutions") or []
            
            inst_match = False
            # Check last known institutions first
            for inst_obj in last_known:
                inst_name = inst_obj.get("display_name") or ""
                inst_name_norm = _normalise_inst(inst_name)
                if (c_inst_norm in inst_name_norm or inst_name_norm in c_inst_norm or
                    any(w in inst_name_norm for w in c_inst_norm.split() if len(w) > 4)):
                    inst_match = True
                    break
            
            # If not matched, check affiliations list
            if not inst_match:
                for aff in affiliations:
                    inst_obj = aff.get("institution") or {}
                    inst_name = inst_obj.get("display_name") or ""
                    inst_name_norm = _normalise_inst(inst_name)
                    if (c_inst_norm in inst_name_norm or inst_name_norm in c_inst_norm or
                        any(w in inst_name_norm for w in c_inst_norm.split() if len(w) > 4)):
                        inst_match = True
                        break
            
            if inst_match:
                matched_author = author
                break
                
        if not matched_author:
            return
            
        oa_id = matched_author.get("id", "").split("/")[-1]
        
        if not oa_id:
            return
            
        candidate.openalex_id = oa_id
        
        # Now fetch their recent works
        works = await fetch_recent_works(oa_id, client)
        if not works:
            return
            
        candidate.papers = _extract_papers(works)
        candidate.recent_pubs_last_3_years = _count_recent(works)
        
        # We can also get their topics
        topics = [t.get("display_name", "") for t in (matched_author.get("topics") or [])[:10]]
        if not candidate.topics:
            candidate.topics = topics
            
        if "openalex" not in candidate.data_sources:
            candidate.data_sources.append("openalex")
            
    except Exception as exc:
        log.warning(f"Failed to enrich papers for {candidate.name}: {exc}")

async def enrich_candidate_papers(candidates: list[Candidate], client: httpx.AsyncClient):
    """
    Search OpenAlex to find papers for candidates who only have grants.
    """
    to_enrich = [c for c in candidates if not c.papers]
    if not to_enrich:
        return
        
    log.info(f"Starting targeted paper enrichment for {len(to_enrich)} candidates...")
    tasks = [_enrich_single_candidate(c, client) for c in to_enrich]
    await asyncio.gather(*tasks, return_exceptions=True)
    log.info("Finished targeted paper enrichment.")
