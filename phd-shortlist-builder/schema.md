# Output JSON Schema

The system produces a single JSON file per student at `sample_output/<student_id>.json`.

## Top-level structure

```json
{
  "student_id": "string",
  "generated_at": "ISO-8601 UTC timestamp",
  "supervisors": [ ...SupervisorRecord ],
  "metadata": { ...Metadata }
}
```

## SupervisorRecord

| Field | Type | Notes |
|-------|------|-------|
| `rank` | integer | 1-indexed, sorted by `recruitment_score` descending |
| `supervisor_id` | string | `openalex:AXXX`, `nih:...`, or `ukri:...` |
| `name` | string | Full name |
| `institution` | string | Current primary affiliation |
| `country` | string | ISO-3166 2-letter code |
| `contact_email` | string \| null | Verified only. null = not found |
| `email_inferred` | boolean | Always false (inference was removed) |
| `research_focus` | string[] | Up to 5 OpenAlex topic labels |
| `evidence.papers` | Paper[] | Top 3 recent papers by citation count |
| `evidence.grants` | Grant[] | Active grants from NIH/UKRI |
| `why_match` | string | 2–3 sentence personalised rationale |
| `fit_confidence` | float | LLM-assessed ∈ [0, 1] |
| `tier` | string | `reach` / `target` / `safety` |
| `recruitment_score` | float | Primary ranking signal ∈ [0, 1] |
| `career_stage_score` | float | ∈ [0, 1] |
| `career_stage_low_confidence` | boolean | true if score ∈ [0.35, 0.60) |
| `pi_verification.faculty_page_verified` | boolean | Best-effort HTTP check |
| `pi_verification.active_grant_verified` | boolean | Grant API data |
| `pi_verification.recent_publication_verified` | boolean | Published ≤3 years ago |
| `pi_verification.verification_score` | float | Weighted: 0.45/0.45/0.10 |
| `pi_verification.verification_sources` | string[] | Which signals fired |
| `recency.recent_pubs_last_3_years` | integer | Papers published in last 3 years |
| `recency.active_grant_count` | integer | Number of active grants |
| `recency.latest_publication_year` | integer \| null | Year of most recent paper |
| `open_position_url` | string \| null | EURAXESS/FindAPhD link if found |
| `linked_program` | string \| null | PhD program or vacancy title |
| `disambiguation_warning` | boolean | True if name collision was detected |
| `data_sources` | string[] | Which discovery sources found this PI |

## Paper

| Field | Type |
|-------|------|
| `title` | string |
| `year` | integer |
| `doi` | string \| null | Full `https://doi.org/...` URL |
| `cited_by_count` | integer |

## Grant

| Field | Type |
|-------|------|
| `title` | string |
| `funder` | string | `NIH`, `UKRI`, `ARC`, etc. |
| `grant_id` | string \| null |
| `url` | string \| null | Funder project page |
| `active` | boolean |

## Metadata

| Field | Type | Notes |
|-------|------|-------|
| `total_candidates_discovered` | integer | Before any filtering |
| `total_after_filtering` | integer | After all filter stages |
| `final_shortlist_count` | integer | In the output |
| `coverage_by_area` | object | `{area: count}` |
| `country_distribution` | object | `{ISO-code: count}` |
| `filter_rejection_log.country_rejected` | integer | |
| `filter_rejection_log.domain_leakage_gate_a` | integer | |
| `filter_rejection_log.domain_leakage_gate_b` | integer | |
| `filter_rejection_log.career_stage_rejected` | integer | |
| `filter_rejection_log.verification_rejected` | integer | |
| `filter_rejection_log.disambig_discarded` | integer | |

## RecruitmentScore Formula

```
recruitment_score =
    0.30 × domain_similarity
  + 0.25 × recency_score
  + 0.20 × verification_score
  + 0.15 × fit_confidence
  + 0.10 × career_stage_score

recency_score = 0.60 × min(recent_pubs_last_3_years / 5, 1.0)
              + 0.40 × min(active_grant_count / 3, 1.0)
```
