"""
Deduplication and same-name disambiguation.

Primary: deduplicate on OpenAlex author_id (trust OpenAlex's own disambiguation).
Secondary: merge NIH/UKRI candidates into matching OpenAlex records by name + institution.
Tertiary: if two records have identical (name, country) but different institutions
          and both came from OpenAlex, flag the lower-similarity one with
          disambiguation_warning=True.

Design note: We do NOT reject on name collisions alone. We use OpenAlex IDs
as the authority. Common names like "Wei Wang" only collide if they share an
OpenAlex ID — which means OpenAlex already thinks they're the same person.
If they have different IDs, they're different records and we keep both.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from src.candidate import Candidate, Grant
from src.profile_parser import StudentProfile

log = logging.getLogger(__name__)


def _normalise_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _normalise_inst(inst: str) -> str:
    # Strip common suffixes for fuzzy matching
    return (
        inst.strip().lower()
        .replace("university of ", "u ")
        .replace("the ", "")
        .split(",")[0]
        .strip()
    )


def deduplicate_grants(grants: list[Grant]) -> list[Grant]:
    """Helper to deduplicate grants by grant_id or title/funder key."""
    seen = set()
    unique_grants = []
    for g in grants:
        key = g.grant_id.strip().lower() if g.grant_id else f"{g.funder.strip().lower()}|{g.title.strip().lower()}"
        if key not in seen:
            seen.add(key)
            unique_grants.append(g)
    return unique_grants


def deduplicate_and_disambiguate(
    candidates: list[Candidate],
    profile: StudentProfile,
) -> list[Candidate]:
    """
    1. Merge candidates with the same OpenAlex ID (consolidate sources/grants).
    2. Cross-link NIH/UKRI candidates into OpenAlex records where name+institution match.
    3. Flag same-name collisions across institutions with disambiguation_warning.
    Returns deduplicated list.
    """
    # --- Pass 1: Group by OpenAlex ID ---
    by_openalex: dict[str, Candidate] = {}
    no_openalex: list[Candidate] = []

    for c in candidates:
        if c.openalex_id:
            key = c.openalex_id.split("/")[-1]
            if key in by_openalex:
                existing = by_openalex[key]
                # Merge: add grants and sources from duplicate
                existing.grants.extend(c.grants)
                existing.grants = deduplicate_grants(existing.grants)
                existing.active_grant_count = sum(1 for g in existing.grants if g.active)
                if c.contact_email and not existing.contact_email:
                    existing.contact_email = c.contact_email
                for src in c.data_sources:
                    if src not in existing.data_sources:
                        existing.data_sources.append(src)
                if c.open_position_url and not existing.open_position_url:
                    existing.open_position_url = c.open_position_url
            else:
                by_openalex[key] = c
                # Ensure its own grants are also deduplicated
                c.grants = deduplicate_grants(c.grants)
                c.active_grant_count = sum(1 for g in c.grants if g.active)
        else:
            no_openalex.append(c)

    # --- Pass 2: Try to link non-OpenAlex candidates to existing OpenAlex records ---
    remaining_no_oa: list[Candidate] = []
    openalex_list = list(by_openalex.values())

    for c in no_openalex:
        c_name = _normalise_name(c.name)
        c_inst = _normalise_inst(c.institution)
        matched = False
        for existing in openalex_list:
            e_name = _normalise_name(existing.name)
            e_inst = _normalise_inst(existing.institution)
            if c_name == e_name and (c_inst in e_inst or e_inst in c_inst):
                # Merge into existing OpenAlex record
                existing.grants.extend(c.grants)
                existing.grants = deduplicate_grants(existing.grants)
                existing.active_grant_count = sum(1 for g in existing.grants if g.active)
                if c.contact_email and not existing.contact_email:
                    existing.contact_email = c.contact_email
                for src in c.data_sources:
                    if src not in existing.data_sources:
                        existing.data_sources.append(src)
                if c.open_position_url and not existing.open_position_url:
                    existing.open_position_url = c.open_position_url
                matched = True
                break
        if not matched:
            c.grants = deduplicate_grants(c.grants)
            c.active_grant_count = sum(1 for g in c.grants if g.active)
            remaining_no_oa.append(c)

    # --- Pass 3: Flag same-name collisions in remaining pool ---
    combined = openalex_list + remaining_no_oa
    name_groups: dict[str, list[Candidate]] = defaultdict(list)
    for c in combined:
        name_groups[_normalise_name(c.name)].append(c)

    collision_count = 0
    for name, group in name_groups.items():
        if len(group) > 1:
            # Multiple candidates with the same name — flag all but the best match
            # "best" = highest career_stage_score (already computed at this point)
            group.sort(key=lambda x: x.career_stage_score, reverse=True)
            for c in group[1:]:
                c.disambiguation_warning = True
                collision_count += 1
            log.debug(f"Name collision: '{name}' has {len(group)} records — flagged {len(group)-1}")

    log.info(
        f"Deduplication: {len(candidates)} → {len(combined)} candidates "
        f"({collision_count} disambiguation warnings)"
    )
    return combined
