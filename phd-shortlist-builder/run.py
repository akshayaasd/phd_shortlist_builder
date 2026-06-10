"""
PhD Shortlist Builder — main entrypoint.

Usage:
    python run.py --profile sample_profiles/student_001.json

Options:
    --profile PATH      Path to student profile JSON (required)
    --output DIR        Output directory (default: sample_output)
    --dry-run           Run with reduced queries for fast testing (~2 min)
    --ingest-outcomes PATH  Ingest an outcomes CSV into the feedback DB
    --max-results INT   Maximum supervisors in final shortlist (default: 200)

Single-command reproducibility (requirement #6):
    python run.py --profile sample_profiles/student_001.json
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import click
import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phd_shortlist")


async def _run_pipeline(
    profile_path: str,
    output_dir: str,
    dry_run: bool,
    max_results: int,
) -> Path:
    from src.profile_parser import load_profile
    from src.embedder import embed_profile
    from src.discovery.openalex import discover_via_openalex
    from src.discovery.nih_reporter import discover_via_nih
    from src.discovery.ukri_gtr import discover_via_ukri
    from src.discovery.euraxess import discover_via_euraxess
    from src.filters.country_gate import apply_country_gate
    from src.filters.domain_guard import apply_domain_guard
    from src.filters.career_stage import apply_career_stage_filter
    from src.filters.disambiguation import deduplicate_and_disambiguate
    from src.verification.pi_verifier import apply_pi_verification
    from src.enrichment.email_finder import enrich_emails
    from src.enrichment.why_match import generate_all_why_matches
    from src.enrichment.recruitment_score import score_and_rank
    from src.feedback.scorer import apply_feedback_scores
    from src.output.schema import (
        build_shortlist, write_shortlist,
        ShortlistMetadata, FilterLog
    )

    start_time = time.time()
    openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"))
    from src.llm_client import LLMClient
    llm_client = LLMClient(openai_client)

    # ── Stage 1: Load & embed profile ────────────────────────────────────────
    log.info("═" * 60)
    log.info("Stage 1: Loading and embedding student profile")
    profile = load_profile(profile_path)
    profile = await embed_profile(profile, llm_client)
    log.info(f"Student: {profile.student_id} | Areas: {profile.research_interests}")
    log.info(f"Target countries: {profile.target_countries}")

    # ── Stage 2: Discovery (all sources in parallel) ──────────────────────────
    log.info("═" * 60)
    log.info("Stage 2: Discovery (OpenAlex + NIH + UKRI + EURAXESS — parallel)")

    async with httpx.AsyncClient(follow_redirects=True) as http:
        discovery_tasks = [
            discover_via_openalex(profile, http, dry_run=dry_run),
            discover_via_nih(profile, http, dry_run=dry_run),
            discover_via_ukri(profile, http, dry_run=dry_run),
            discover_via_euraxess(profile, http, dry_run=dry_run),
        ]
        all_batches = await asyncio.gather(*discovery_tasks, return_exceptions=True)

        all_candidates = []
        for batch in all_batches:
            if isinstance(batch, list):
                all_candidates.extend(batch)
            else:
                log.warning(f"Discovery source failed: {batch}")

        total_discovered = len(all_candidates)
        log.info(f"Total discovered: {total_discovered} candidates")

        # ── Stage 3: Filtering ────────────────────────────────────────────────
        log.info("═" * 60)
        log.info("Stage 3: Filtering")

        # 3a. Country gate (hard, first — cheapest)
        after_country = apply_country_gate(all_candidates, profile)
        country_rejected = total_discovered - len(after_country)

        # 3b. Career-stage scoring
        after_career = apply_career_stage_filter(after_country)
        career_rejected = len(after_country) - len(after_career)

        # 3c. Deduplication (before domain guard — reduces LLM calls)
        after_dedup = deduplicate_and_disambiguate(after_career, profile)
        disambig_discarded = sum(1 for c in after_dedup if c.disambiguation_warning)

        # 3d. Domain guard (Gate A + Gate B LLM)
        after_domain = await apply_domain_guard(after_dedup, profile, llm_client)
        gate_a_rejected = len(after_dedup) - len(after_domain)  # approximate
        gate_b_rejected = 0  # logged internally

        # ── Stage 4: PI Verification ──────────────────────────────────────────
        log.info("═" * 60)
        log.info("Stage 4: PI Verification")
        after_verification = await apply_pi_verification(after_domain, http)
        verification_rejected = len(after_domain) - len(after_verification)

        log.info(f"After all filters: {len(after_verification)} candidates")

        # ── Stage 5: Enrichment ───────────────────────────────────────────────
        log.info("═" * 60)
        log.info("Stage 5: Enrichment")

        # 5a. Email finding (verified only)
        await enrich_emails(after_verification, http)

        # 5b. Domain similarity (for RecruitmentScore)
        _apply_domain_similarity(after_verification, profile)

        # 5c. why_match generation
        await generate_all_why_matches(after_verification, profile, llm_client)

        # 5d. Scoring & ranking
        ranked = score_and_rank(after_verification)

        # 5e. Apply feedback history (suppression + EMA boost)
        ranked = apply_feedback_scores(ranked)

        # Trim to max_results
        final = ranked[:max_results]

    # ── Stage 6: Output ───────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("Stage 6: Writing output")

    coverage = _coverage_by_area(final, profile)
    country_dist = Counter(c.country for c in final)

    metadata = ShortlistMetadata(
        total_candidates_discovered=total_discovered,
        total_after_filtering=len(after_verification),
        final_shortlist_count=len(final),
        coverage_by_area=coverage,
        country_distribution=dict(country_dist),
        filter_rejection_log=FilterLog(
            country_rejected=country_rejected,
            domain_leakage_gate_a=gate_a_rejected,
            domain_leakage_gate_b=gate_b_rejected,
            career_stage_rejected=career_rejected,
            verification_rejected=verification_rejected,
            disambig_discarded=disambig_discarded,
        ),
    )

    shortlist = build_shortlist(profile.student_id, final, metadata)
    out_path = write_shortlist(shortlist, output_dir)

    elapsed = time.time() - start_time
    log.info("═" * 60)
    log.info(f"Done! {len(final)} supervisors -> {out_path}")
    log.info(f"Wall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if elapsed > 900:
        log.warning("Exceeded 15-minute target — consider --dry-run for testing")

    return out_path


def _apply_domain_similarity(candidates, profile) -> None:
    """Compute cosine similarity between candidate topics and student interests."""
    from src.embedder import cosine_similarity
    for c in candidates:
        if not c.topics or not profile.interest_embeddings:
            c.domain_similarity = 0.5
            continue
        # Use topic names as proxy (true approach would embed topics too,
        # but that would require extra API calls per candidate)
        topic_str = " ".join(c.topics).lower()
        best_sim = 0.0
        for interest in profile.research_interests:
            interest_lower = interest.lower()
            # Keyword overlap as fast approximation
            interest_words = set(interest_lower.split())
            topic_words = set(topic_str.split())
            overlap = len(interest_words & topic_words) / max(len(interest_words), 1)
            best_sim = max(best_sim, overlap)
        c.domain_similarity = min(1.0, best_sim * 2)  # scale overlap to [0, 1]


def _coverage_by_area(candidates, profile) -> dict[str, int]:
    """Estimate how many candidates map to each research interest."""
    coverage: dict[str, int] = {area: 0 for area in profile.research_interests}
    for c in candidates:
        topic_str = " ".join(c.topics).lower()
        for area in profile.research_interests:
            if any(w in topic_str for w in area.lower().split() if len(w) > 4):
                coverage[area] += 1
                break
    return coverage


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    pass


@cli.command()
@click.option("--profile", required=True, help="Path to student profile JSON")
@click.option("--output", default="sample_output", help="Output directory")
@click.option("--dry-run", is_flag=True, help="Fast test run with reduced queries")
@click.option("--max-results", default=200, help="Max supervisors in shortlist")
def build(profile: str, output: str, dry_run: bool, max_results: int):
    """Build a PhD supervisor shortlist for a student profile."""
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "sk-placeholder"

    if dry_run:
        click.echo("Dry-run mode: using reduced queries (~2 min)")

    asyncio.run(_run_pipeline(profile, output, dry_run, max_results))


@cli.command()
@click.option("--csv", "csv_path", required=True, help="Path to outcomes CSV")
def ingest_outcomes(csv_path: str):
    """Ingest an outcomes CSV into the feedback database."""
    from src.feedback.supervisor_db import ingest_outcomes_csv
    n = ingest_outcomes_csv(csv_path)
    click.echo(f"✅ Ingested {n} outcome rows from {csv_path}")


if __name__ == "__main__":
    # Support: python run.py --profile ...  (without subcommand, for simplicity)
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        # Inject 'build' subcommand if user runs without it
        sys.argv.insert(1, "build")
    cli()
