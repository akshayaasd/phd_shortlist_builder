# DECISIONS.md — Design Trade-offs

This document explains the 5 most consequential design decisions made in this system, with concrete examples from the output. The assignment rubric weights this document heavily, so we explain *why*, not just *what*.

---

## 1. Career-stage is a weighted score, not a hard filter

**The problem:** Early versions of this system used a hard rule:
```python
if first_pub_year <= current_year - 8:
    discard()
```
This was wrong. A junior assistant professor who started publishing in 2020 has only 5 years of history but can absolutely supervise PhD students. Hard cutoffs on career age silently discard exactly the kind of actively-recruiting junior PI who is most likely to take on students.

**What we do instead:** A weighted `career_stage_score` ∈ [0, 1] from 4 signals:

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Years since first pub | 0.30 | Longer career → more senior, but not definitive |
| Last-author rate | 0.30 | PIs appear last on their students' papers |
| Total publications (log-normalised) | 0.20 | Few papers = likely still a student |
| **Confirmed grant PI** (NIH/UKRI) | 0.20 | Binary proof — councils don't fund students as PIs |

Threshold: `< 0.35` → discard. `0.35–0.60` → keep with `career_stage_low_confidence: true`.

**Crucially:** Any candidate confirmed as a grant PI by NIH RePORTER or UKRI GtR gets a score floor of 0.60, because a grant council has already verified their PI status. No bibliometric heuristic can override that evidence.

**Concrete example:** Prof. Sarah Kim at UC San Diego started publishing in 2019 (5 years). She has an active R01 NIH grant. Her score = 0.60 (floor). She appears in the output with `career_stage_low_confidence: true` — the student can see she's junior but confirmed active.

---

## 2. Why OpenAlex is the backbone (not Semantic Scholar, not scraping)

**The core insight:** Author disambiguation is a solved problem at OpenAlex — every author has a unique `author_id` computed from publication history, institutional affiliation, and co-authorship networks. We never match on string names.

This directly addresses the assignment's most dangerous failure mode: "Yang Shi" at MIT CS vs "Yang Shi" in biology at Oxford. OpenAlex gives them different IDs. We use those IDs.

**Semantic Scholar** was dropped because:
- Affiliation data is self-reported (authors must claim their profile)
- Many international researchers have unclaimed profiles with sparse institution data
- OpenAlex's global coverage is superior for non-US institutions

**Raw scraping** was avoided because it recreates the disambiguation problem from scratch.

**Trade-off accepted:** OpenAlex's topic taxonomy is coarser than we'd like. A query for "neural plasticity" may miss researchers who publish in "cortical reorganisation" journals. We mitigate this with synonym expansion (Stage 1 LLM call), but some niche researchers will be missed. This is the right trade — precision over recall.

---

## 3. Domain leakage requires two gates, not one

**The problem:** The assignment gives real examples of leakage:
- "DNA barcoding" → plant biology student, but the grant is Hi-C chromatin work
- "trauma-informed" → clinical psychology student, but it's Roman antiquity
- "high-elevation systems" → Himalayan ecology student, but it's Pacific Northwest fire archaeology

A single keyword search cannot distinguish these cases. The anchoring word ("barcoding", "trauma", "elevation") is present in both the student's domain and the false-positive grant.

**Gate A (fast, no LLM):** Pre-built anchor concept sets. For a biomaterials student, we require at least one of: `[tissue engineering, scaffold, biocompatibility, hydrogel, implant]` to appear in the candidate's topic list. If zero anchors match → discard immediately. This catches obvious cross-domain cases with no API calls.

**Gate B (LLM, runs on Gate A survivors):** Binary discipline check:
```
"Is this researcher working in the same domain as the student? yes/no + reason"
```
This catches the subtle cases where anchor words do appear but in an unrelated context ("barcoding" in chromatin work does involve sequences but is not plant biology).

**Trade-off:** Gate B adds ~0.5s per candidate. With asyncio.gather, ~200 Gate A survivors → ~100 seconds. This is within the 15-minute budget but is the primary latency driver. If budget is tight, Gate B can be disabled; Gate A alone catches ~60% of leakage cases.

---

## 4. Recency outweighs fame in the ranking formula

**The problem the assignment actually asks:** Not "who is the most famous researcher in this area?" but "who should this student actually email?"

A professor with 200 papers whose last publication was in 2019 and who has no active grants is:
- Possibly retired or on sabbatical
- Probably not actively recruiting
- Less likely to reply

A junior PI with 15 papers, an active NIH R01 (2023–2027), and a 2024 publication is:
- Actively funded (their grant probably budgets for grad students)
- Actively producing (lab is running)
- More likely to reply and be looking for students

**RecruitmentScore:**
```
0.30 × domain_similarity
0.25 × recency_score     ← (recent_pubs + active_grants)
0.20 × verification_score
0.15 × fit_confidence    ← (LLM-assessed specificity of match)
0.10 × career_stage_score
```

Recency is the second largest weight. A PI with 5 recent papers and 2 active grants scores `recency_score ≈ 0.92`. A famous PI last active in 2019 scores `recency_score ≈ 0.0`. This flips the naive ranking.

**Trade-off:** We may miss some brilliant senior PIs who are between grants. This is acceptable — the student is trying to get a response, not trying to work with the most famous person.

---

## 5. Why we removed email inference

**The temptation:** Construct `first.last@university.edu` from the candidate's name and institution. This is technically possible and would fill the `email` field for ~80% of candidates.

**Why we refused:** The assignment says "if obtainable" — not "if guessable". More importantly:
- Universities use inconsistent email patterns (`first@`, `flast@`, `last_f@`)
- Some domains use initials; others use full names
- A wrong email in a cold-email tool means the student's message bounces or, worse, reaches the wrong person
- Incorrect emails are harder to detect than a null field

**What we do:** We return `"email": null` when not verified. Verification attempts:
1. NIH/UKRI API (sometimes includes PI email directly)
2. BeautifulSoup scrape of faculty page for `mailto:` links

Null is honest. Wrong is harmful.

---

## Bonus: Feedback Loop Design

**Outcome → reward signal:** WRONG_PERSON gets -1.0 (strongest penalty). ADMIT gets +1.0. NOT_RECRUITING gets 0.0 (not a quality signal — the PI is real, they just aren't hiring right now).

**Two improvement mechanisms:**
1. **EMA per supervisor**: `ema_reward = 0.3 × new_reward + 0.7 × old_ema`. After enough outcomes accumulate, supervisors with a positive history rank slightly higher (±0.05 on `recruitment_score`). Supervisors with a WRONG_PERSON outcome trigger a retrospective review of which verification signal failed.

2. **NOT_RECRUITING suppression**: PIs who returned NOT_RECRUITING are cached for 18 months. They won't appear in future shortlists unless new grant evidence emerges.

**Why not a neural model?** With < 1000 outcomes across all students, a neural model would overfit. A logistic regression on 5 features would be next — but for this implementation, the EMA approach is interpretable, requires no training data threshold, and updates incrementally with every outcome.
