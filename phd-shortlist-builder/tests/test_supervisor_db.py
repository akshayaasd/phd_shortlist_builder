"""Tests for outcomes feedback loop and database."""
import csv
import pytest
from pathlib import Path
from datetime import datetime, timezone
import src.feedback.supervisor_db as sdb


@pytest.fixture(autouse=True)
def mock_db_path(tmp_path, monkeypatch):
    """Fixture to redirect the database to a temp file for every test."""
    db_file = tmp_path / "test_supervisor_db.sqlite"
    monkeypatch.setattr(sdb, "DB_PATH", db_file)
    yield db_file


def test_ingest_outcomes_csv(tmp_path):
    """Test standard outcome ingestion, EMA scoring, and suppression logic."""
    csv_file = tmp_path / "outcomes.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["student_id", "supervisor_id", "institution", "area", "sent_at", "outcome"])
        writer.writerow(["s1", "nih:test_1", "MIT", "AI", "2026-06-01", "ADMIT"])
        writer.writerow(["s2", "nih:test_2", "Stanford", "NLP", "2026-06-02", "NOT_RECRUITING"])
        writer.writerow(["s3", "nih:test_1", "MIT", "AI", "2026-06-03", "INTERVIEW"])

    rows = sdb.ingest_outcomes_csv(str(csv_file))
    assert rows == 3

    # Check EMA for supervisor nih:test_1
    # First record: ADMIT (+1.0) -> inserts 1.0
    # Second record: INTERVIEW (+0.8) -> EMA_ALPHA * 0.8 + (1 - EMA_ALPHA) * 1.0
    # 0.3 * 0.8 + 0.7 * 1.0 = 0.24 + 0.7 = 0.94
    ema = sdb.get_supervisor_score("nih:test_1")
    assert ema is not None
    assert abs(ema - 0.94) < 1e-4

    # Check suppression for nih:test_2
    assert sdb.is_suppressed("nih:test_2") is True
    assert sdb.is_suppressed("nih:test_1") is False


def test_not_recruiting_capping_and_leap_year(tmp_path, monkeypatch):
    """Verify that NOT_RECRUITING suppression date handles end-of-month and year boundaries."""
    # We will mock datetime.now to return a date near the end of a month (e.g., Aug 31, 2026)
    # Adding 18 months to Aug 31, 2026 should land on Feb 28/29, 2028 (leap year).
    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 31, tzinfo=timezone.utc)

    monkeypatch.setattr(sdb, "datetime", MockDatetime)

    csv_file = tmp_path / "outcomes.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["student_id", "supervisor_id", "institution", "area", "sent_at", "outcome"])
        writer.writerow(["s1", "nih:test_suppress", "MIT", "AI", "2026-08-31", "NOT_RECRUITING"])

    sdb.ingest_outcomes_csv(str(csv_file))

    # Read the stored suppressed_until date directly from database to verify correct date calculation
    conn = sdb._get_conn()
    row = conn.execute(
        "SELECT suppressed_until FROM supervisor_scores WHERE supervisor_id = ?",
        ("nih:test_suppress",)
    ).fetchone()
    conn.close()

    assert row is not None
    suppressed_until_str = row[0]
    
    # 2026-08-31 + 18 months -> 2028-02 (Feb 2028 is a leap year, so max days is 29)
    # The day 31 gets capped to 29.
    expected_suppressed_until = datetime(2028, 2, 29, tzinfo=timezone.utc).isoformat()
    assert suppressed_until_str == expected_suppressed_until
