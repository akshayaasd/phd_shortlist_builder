"""
Country gate — hard constraint filter.

Every candidate must have their current affiliation in one of the
student's target countries. This is a binary, non-negotiable gate.
Any candidate that fails is discarded immediately — not flagged.

Why discard instead of flag: surfacing an out-of-country PI to the
student is strictly worse than missing a borderline match.
"""
from __future__ import annotations

import logging

from src.candidate import Candidate
from src.profile_parser import StudentProfile

log = logging.getLogger(__name__)

# Aliases that may appear in source data or student profile input
_ALIASES: dict[str, str] = {
    "UK": "GB",
    "UNITED KINGDOM": "GB",
    "GREAT BRITAIN": "GB",
    "UNITED STATES": "US",
    "USA": "US",
    "CANADA": "CA",
    "AUSTRALIA": "AU",
    "GERMANY": "DE",
    "SINGAPORE": "SG",
    "FRANCE": "FR",
    "NETHERLANDS": "NL",
    "SWEDEN": "SE",
    "SWITZERLAND": "CH",
    "BELGIUM": "BE",
    "DENMARK": "DK",
    "FINLAND": "FI",
    "NORWAY": "NO",
    "ITALY": "IT",
    "SPAIN": "ES",
    "JAPAN": "JP",
    "CHINA": "CN",
    "SOUTH KOREA": "KR",
    "NEW ZEALAND": "NZ",
    "IRELAND": "IE",
    "AUSTRIA": "AT",
    "PORTUGAL": "PT",
    "HONG KONG": "HK",
}


def _normalise(code: str) -> str:
    code = code.upper().strip()
    return _ALIASES.get(code, code)


def apply_country_gate(
    candidates: list[Candidate],
    profile: StudentProfile,
) -> list[Candidate]:
    """
    Discard any candidate not in the student's target countries.
    Returns the filtered list and logs rejection count.
    """
    target = {_normalise(c) for c in profile.target_countries}
    passed, rejected = [], 0

    for c in candidates:
        candidate_country = _normalise(c.country)
        if candidate_country in target:
            passed.append(c)
        else:
            log.debug(f"Country gate REJECTED: {c.name} @ {c.institution} [{c.country}]")
            rejected += 1

    log.info(f"Country gate: {rejected} rejected, {len(passed)} passed")
    return passed
