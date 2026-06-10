"""
Feedback loop — SQLite-backed supervisor success database.

Ingests outcomes.csv and updates supervisor EMA reward scores.
These scores feed into future ranking runs.

Outcome → reward mapping (see DECISIONS.md for rationale):
  ADMIT           +1.0   Strongest positive signal
  INTERVIEW       +0.8
  POSITIVE_REPLY  +0.6
  OUT_OF_OFFICE   +0.1   Neutral — confirms valid address
  NOT_RECRUITING   0.0   Quality-neutral; suppresses PI for 18 months
  NO_REPLY        -0.1   Weak negative
  REJECT          -0.3
  BOUNCE          -0.5   Bad contact info
  WRONG_PERSON    -1.0   Strongest negative — contamination signal
"""
from __future__ import annotations

import csv
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH = Path(".cache/supervisor_db.sqlite")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

OUTCOME_REWARDS: dict[str, float] = {
    "ADMIT": 1.0,
    "INTERVIEW": 0.8,
    "POSITIVE_REPLY": 0.6,
    "OUT_OF_OFFICE": 0.1,
    "NOT_RECRUITING": 0.0,
    "NO_REPLY": -0.1,
    "REJECT": -0.3,
    "BOUNCE": -0.5,
    "WRONG_PERSON": -1.0,
}

EMA_ALPHA = 0.3   # exponential moving average decay
NOT_RECRUITING_SUPPRESS_MONTHS = 18


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS supervisor_scores (
            supervisor_id TEXT PRIMARY KEY,
            name          TEXT,
            institution   TEXT,
            ema_reward    REAL DEFAULT 0.0,
            outcome_count INTEGER DEFAULT 0,
            last_updated  TEXT,
            suppressed_until TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS outcome_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id    TEXT,
            supervisor_id TEXT,
            institution   TEXT,
            area          TEXT,
            sent_at       TEXT,
            outcome       TEXT,
            reward        REAL,
            logged_at     TEXT
        )
    """)
    conn.commit()
    return conn


def ingest_outcomes_csv(csv_path: str) -> int:
    """
    Parse outcomes CSV and update supervisor EMA scores.
    Returns number of rows processed.
    """
    conn = _get_conn()
    rows_processed = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            supervisor_id = (row.get("supervisor_id") or "").strip()
            outcome = (row.get("outcome") or "").strip().upper()

            if not supervisor_id or outcome not in OUTCOME_REWARDS:
                continue

            reward = OUTCOME_REWARDS[outcome]
            now = datetime.now(timezone.utc).isoformat()

            # Log the raw outcome
            conn.execute(
                "INSERT INTO outcome_log (student_id, supervisor_id, institution, area, sent_at, outcome, reward, logged_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row.get("student_id", ""),
                    supervisor_id,
                    row.get("institution", ""),
                    row.get("area", ""),
                    row.get("sent_at", ""),
                    outcome,
                    reward,
                    now,
                )
            )

            # Update EMA
            existing = conn.execute(
                "SELECT ema_reward, outcome_count FROM supervisor_scores WHERE supervisor_id = ?",
                (supervisor_id,)
            ).fetchone()

            suppressed_until = None
            if outcome == "NOT_RECRUITING":
                # Suppress this PI for 18 months
                from datetime import timedelta
                suppress_dt = datetime.now(timezone.utc).replace(
                    month=((datetime.now().month - 1 + NOT_RECRUITING_SUPPRESS_MONTHS) % 12) + 1
                )
                suppressed_until = suppress_dt.isoformat()

            if existing:
                old_ema, count = existing
                new_ema = EMA_ALPHA * reward + (1 - EMA_ALPHA) * old_ema
                conn.execute(
                    "UPDATE supervisor_scores SET ema_reward=?, outcome_count=?, last_updated=?, suppressed_until=COALESCE(?, suppressed_until) WHERE supervisor_id=?",
                    (new_ema, count + 1, now, suppressed_until, supervisor_id)
                )
            else:
                conn.execute(
                    "INSERT INTO supervisor_scores (supervisor_id, ema_reward, outcome_count, last_updated, suppressed_until) VALUES (?, ?, 1, ?, ?)",
                    (supervisor_id, reward, now, suppressed_until)
                )

            rows_processed += 1

    conn.commit()
    conn.close()
    log.info(f"Ingested {rows_processed} outcome rows from {csv_path}")
    return rows_processed


def get_supervisor_score(supervisor_id: str) -> float | None:
    """Look up stored EMA reward for a supervisor. Returns None if not found."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT ema_reward FROM supervisor_scores WHERE supervisor_id = ?",
        (supervisor_id,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def is_suppressed(supervisor_id: str) -> bool:
    """Check if a PI is suppressed due to NOT_RECRUITING outcome."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT suppressed_until FROM supervisor_scores WHERE supervisor_id = ?",
        (supervisor_id,)
    ).fetchone()
    conn.close()
    if not row or not row[0]:
        return False
    try:
        suppress_dt = datetime.fromisoformat(row[0])
        return datetime.now(timezone.utc) < suppress_dt
    except Exception:
        return False
