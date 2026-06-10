import pytest
import httpx
from unittest.mock import AsyncMock, patch
from src.candidate import Candidate, Grant
from src.discovery.openalex import _extract_grants_from_works
from src.enrichment.grant_enricher import enrich_candidate_grants
from src.filters.disambiguation import deduplicate_grants

def test_openalex_grant_extraction():
    works = [
        {
            "publication_year": 2026,
            "awards": [
                {
                    "funder_id": "https://openalex.org/F4320306076",
                    "funder_display_name": "National Science Foundation",
                    "funder_award_id": "NSF-12345"
                }
            ]
        },
        # Duplicate grant in another paper (should be deduplicated)
        {
            "publication_year": 2025,
            "awards": [
                {
                    "funder_id": "https://openalex.org/F4320306076",
                    "funder_display_name": "National Science Foundation",
                    "funder_award_id": "NSF-12345"
                }
            ]
        },
        # New grant in an older paper (should be marked inactive if < current_year - 2)
        {
            "publication_year": 2020,
            "awards": [
                {
                    "funder_id": "https://openalex.org/F4320306076",
                    "funder_display_name": "National Science Foundation",
                    "funder_award_id": "NSF-99999"
                }
            ]
        }
    ]
    
    grants = _extract_grants_from_works(works)
    assert len(grants) == 2
    
    # Check deduplicated active grant
    nsf_12345 = next(g for g in grants if g.grant_id == "NSF-12345")
    assert nsf_12345.active is True
    assert nsf_12345.funder == "National Science Foundation"
    
    # Check old inactive grant
    nsf_99999 = next(g for g in grants if g.grant_id == "NSF-99999")
    assert nsf_99999.active is False


@pytest.mark.asyncio
async def test_targeted_enrichment_nih():
    candidate = Candidate(
        supervisor_id="openalex:A1",
        name="John Smith",
        institution="Stanford University",
        country="US",
        grants=[]
    )
    
    mock_nih_response = {
        "results": [
            {
                "project_title": "Targeted Cancer Therapy Research",
                "activity_code": "R01",
                "full_project_num": "R01CA123456",
                "appl_id": 9999,
                "project_start_date": "2024-01-01",
                "project_end_date": "2028-12-31",
                "principal_investigators": [
                    {
                        "first_name": "John",
                        "last_name": "Smith",
                        "email": "jsmith@stanford.edu"
                    }
                ],
                "organization": {
                    "org_name": "Stanford University",
                    "org_country": "United States"
                }
            }
        ]
    }
    
    async def mock_post(*args, **kwargs):
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.json = lambda: mock_nih_response
        mock_resp.raise_for_status = lambda: None
        return mock_resp
        
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = mock_post
    
    await enrich_candidate_grants([candidate], client)
    
    assert len(candidate.grants) == 1
    assert candidate.grants[0].grant_id == "R01CA123456"
    assert candidate.grants[0].funder == "NIH"
    assert candidate.active_grant_count == 1


def test_deduplicate_grants_logic():
    grants = [
        Grant(title="Grant A", funder="Funder X", grant_id="G1", active=True),
        Grant(title="Grant A", funder="Funder X", grant_id="G1", active=False),  # Duplicate ID
        Grant(title="Grant B", funder="Funder Y", grant_id=None, active=True),
        Grant(title="Grant B", funder="Funder Y", grant_id=None, active=False),  # Duplicate title/funder key
        Grant(title="Grant C", funder="Funder Z", grant_id="G2", active=True)
    ]
    
    deduped = deduplicate_grants(grants)
    assert len(deduped) == 3
    assert [g.title for g in deduped] == ["Grant A", "Grant B", "Grant C"]


if __name__ == "__main__":
    import asyncio
    print("Running test_openalex_grant_extraction...")
    test_openalex_grant_extraction()
    print("Running test_deduplicate_grants_logic...")
    test_deduplicate_grants_logic()
    print("Running test_targeted_enrichment_nih...")
    asyncio.run(test_targeted_enrichment_nih())
    print("All tests passed successfully!")

