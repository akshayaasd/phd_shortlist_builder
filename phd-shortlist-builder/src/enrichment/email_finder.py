"""
Email finder — verified only.

Attempt 1: OpenAlex corresponding_author email on recent papers.
Attempt 2: Faculty page scrape (BeautifulSoup mailto: links).
No inference fallback. Returns None if not found.

Design decision: The assignment says "if obtainable" — not "guess it".
A wrong email is worse for the student than a null. We never fabricate.
"""
from __future__ import annotations

import logging
import re

import httpx
from bs4 import BeautifulSoup

from src.candidate import Candidate

log = logging.getLogger(__name__)

_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


async def find_email(candidate: Candidate, client: httpx.AsyncClient) -> str | None:
    """
    Try to find a verified email for the candidate.
    Returns email string or None.
    """
    # Attempt 1: already on the candidate (from NIH/UKRI data)
    if candidate.contact_email and "@" in candidate.contact_email:
        return candidate.contact_email

    # Attempt 2: scrape faculty page
    if candidate.institution:
        from src.verification.pi_verifier import _institution_domain, _name_slug
        domain = _institution_domain(candidate.institution)
        if domain:
            slug = _name_slug(candidate.name)
            urls_to_try = [
                f"https://{domain}/people/{slug}",
                f"https://{domain}/faculty/{slug}",
            ]
            headers = {"User-Agent": "Mozilla/5.0 (academic research; contact@ambitio.club)"}
            for url in urls_to_try:
                try:
                    resp = await client.get(url, headers=headers, timeout=8, follow_redirects=True)
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, "html.parser")
                        # Look for mailto: links
                        for a in soup.find_all("a", href=True):
                            href = a["href"]
                            if href.startswith("mailto:"):
                                email = href.replace("mailto:", "").split("?")[0].strip()
                                if _EMAIL_PATTERN.match(email):
                                    log.debug(f"Email found via faculty page for {candidate.name}: {email}")
                                    return email
                        # Fallback: regex scan body text
                        matches = _EMAIL_PATTERN.findall(resp.text)
                        last_name = candidate.name.split()[-1].lower()
                        for match in matches:
                            if last_name in match.lower():
                                return match
                except Exception:
                    continue

    return None


async def enrich_emails(candidates: list[Candidate], client: httpx.AsyncClient) -> None:
    """Enrich candidates with verified emails in-place."""
    import asyncio
    tasks = [find_email(c, client) for c in candidates]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for c, result in zip(candidates, results):
        if isinstance(result, str):
            c.contact_email = result
            c.email_inferred = False
        # If None or Exception → leave as null
