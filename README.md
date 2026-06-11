# PhD Shortlist Builder

A system that ingests a student profile and produces a ranked shortlist of PhD supervisors with grant/paper evidence and personalised match rationales.

## Recent Optimizations & Features

* **Cross-Reference Enrichment:** Cross-links grant-only candidates (from NIH/UKRI) with their verified publications on OpenAlex, and hydrates paper-only candidates with active grants.
* **Fuzzy Institution & Name Validation:** Avoids false-positive publication/grant attachments by performing strict name token matching and fuzzy institution verification.
* **Embedding-Band Domain Guard:** Uses pre-computed topic embedding cosine similarity to classify candidates (auto-passing high similarity, auto-rejecting low similarity), falling back to GPT-4o-mini only for the ambiguous middle band. This reduces LLM API costs by up to 80%.
* **Concurrency & Rate-Limit Management:** Implements a token bucket semaphore (5 concurrent requests max) with exponential backoff retries to guarantee smooth operations and prevent OpenAlex HTTP 429 rate limit triggers.

## Quick Start

```bash
# 1. Navigate to the project directory
cd phd-shortlist-builder

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
cp .env.example .env
# Edit .env and add OPENAI_API_KEY=sk-...

# 4. Run on the sample profile
python run.py --profile sample_profiles/student_001.json

# 5. Fast test (dry-run, ~2 min)
python run.py --profile sample_profiles/student_001.json --dry-run

# 6. Ingest outcome feedback
python run.py ingest-outcomes --csv outcomes.csv
```

Output: `phd-shortlist-builder/sample_output/<student_id>.json`

## Architecture

```
Student Profile JSON
        │
        ▼
Stage 1: Profile Parsing + Embedding
        │   - Validates required fields
        │   - Embeds research interests (cached)
        │   - Expands to synonyms via LLM
        ▼
Stage 2: Multi-Source Discovery (parallel)
        │   - OpenAlex (backbone: 4 sources × N query terms)
        │   - NIH RePORTER (US grants, blocks F31/F32/T32)
        │   - UKRI Gateway to Research (UK grants, blocks Studentships)
        │   - EURAXESS (EU open positions, recruiting signal)
        ▼
Stage 3: Filtering
        │   - Country gate (hard constraint, runs first)
        │   - Career-stage scoring (weighted, not hard filter)
        │   - Deduplication (OpenAlex author_id authority)
        │   - Domain guard Gate A (anchor concepts, fast)
        │   - Domain guard Gate B (LLM discipline check)
        ▼
Stage 4: PI Verification
        │   - recent_publication_verified (OpenAlex, 3yr window)
        │   - active_grant_verified (NIH/UKRI grant data)
        │   - faculty_page_verified (best-effort HTTP scrape)
        ▼
Stage 5: Enrichment & Scoring
        │   - Email finding (verified only, null if not found)
        │   - Targeted Grant Enrichment (targeted NIH/UKRI lookup by PI name & institution)
        │   - why_match generation (gpt-4o-mini, structured JSON)
        │   - RecruitmentScore = f(domain_sim, recency, verification, fit, career_stage)
        │   - Feedback history applied (EMA boost + NOT_RECRUITING suppression)
        ▼
Stage 6: Output JSON
```

## Data Sources

| Source | What it provides | Why included |
|--------|-----------------|--------------|
| **OpenAlex** | Author profiles, works, topics, institution | Backbone — already disambiguated by author_id |
| **NIH RePORTER** | US grant PIs, abstracts, active dates | Definitive PI proof; grant API blocks junior grants |
| **UKRI GtR** | UK grant PIs, project titles | Same for UK |
| **EURAXESS** | EU open PhD positions | Only source proving current active recruitment |

## Design Trade-offs

See [DECISIONS.md](phd-shortlist-builder/DECISIONS.md) for full reasoning.

Key choices:
- **Precision over recall**: hard discard on failed filters rather than flagging
- **career_stage is a score, not a filter**: protects junior assistant professors  
- **No email inference**: null is honest; wrong email hurts the student
- **RecruitmentScore weights recency**: inactive PIs rank below active ones regardless of h-index
- **3+1 data sources**: depth of verification > breadth of connectors in 72 hours

## Output Schema

See [schema.md](phd-shortlist-builder/schema.md).

## Requirements

Python 3.11+, OpenAI API key.
All other dependencies in `requirements.txt`.

## Latency

Typical wall-clock: 8–12 minutes for a full profile.
`--dry-run` mode: ~2 minutes (reduced queries, for testing).
Parallelism: all discovery sources queried concurrently via `asyncio.gather`.

## Reproducibility

Same input profile → same output structure (LLM `why_match` text may vary slightly).
Embedding cache in `.cache/embeddings/` prevents re-billing on re-runs.

## Known Limitations

1. **EURAXESS API is unofficial** — may change without notice. Gracefully degrades to empty result.
2. **Faculty page verification is best-effort** — JS-rendered pages, unusual URL patterns, and 404s on legitimate PIs are common. Weighted low (0.10) in verification score.
3. **Email coverage** — university faculty pages have inconsistent structure. Expect ~30–50% email coverage in final output.
4. **OpenAlex topic taxonomy** — some niche research areas may not map well to OpenAlex's concept graph. Synonym expansion (Stage 1) mitigates this.
5. **Global Funding (e.g., Canada, India, EU)**: While direct official APIs (like NIH/UKRI) are only queried for US/UK candidates, other international funding sources are captured automatically by extracting work-level `awards` metadata directly from OpenAlex.
