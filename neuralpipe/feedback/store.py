"""FeedbackStore ABC + SQLiteFeedbackStore implementation.

The ABC defines the interface so a future ChromaDBFeedbackStore can be swapped
in with a single line change in NeuralPipeAgent.
"""
from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from neuralpipe.models.feedback import DQRecord


class FeedbackStore(ABC):
    """Abstract interface for DQ record persistence."""

    @abstractmethod
    def save(self, record: DQRecord) -> None:
        """Persist a DQRecord."""

    @abstractmethod
    def get_by_route(self, route_id: str) -> list[DQRecord]:
        """Return all DQRecords for a given route_id."""

    @abstractmethod
    def get_by_line(self, line_number: str) -> list[DQRecord]:
        """Return all DQRecords for a given line_number."""

    @abstractmethod
    def get_all(self) -> list[DQRecord]:
        """Return all DQRecords."""

    @abstractmethod
    def mark_applied_to_spec(self, dq_id: str) -> None:
        """Mark a DQRecord as promoted to spec."""


class SQLiteFeedbackStore(FeedbackStore):
    """SQLite-backed DQ record store.

    Schema is a single table with one row per DQRecord. All fields are stored
    as TEXT (JSON-serialised where complex). Simple enough for v1; replace with
    ChromaDB for vector-similarity search when needed.
    """

    _CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS dq_records (
        dq_id              TEXT PRIMARY KEY,
        route_id           TEXT NOT NULL,
        timestamp          TEXT NOT NULL,
        line_number        TEXT NOT NULL,
        dq_category        TEXT NOT NULL,
        dq_reason_text     TEXT NOT NULL,
        derived_rule       TEXT,
        constraint_scope   TEXT NOT NULL DEFAULT 'GLOBAL',
        applied_to_spec    INTEGER NOT NULL DEFAULT 0
    )
    """

    def __init__(self, db_path: str | Path = "neuralpipe_dq.db") -> None:
        self._db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(self._CREATE_SQL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, record: DQRecord) -> None:
        sql = """
        INSERT OR REPLACE INTO dq_records
            (dq_id, route_id, timestamp, line_number, dq_category, dq_reason_text,
             derived_rule, constraint_scope, applied_to_spec)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.execute(sql, (
                record.dq_id,
                record.route_id,
                record.timestamp,
                record.line_number,
                record.dq_category,
                record.dq_reason_text,
                record.derived_rule,
                record.constraint_scope,
                int(record.applied_to_spec),
            ))

    def get_by_route(self, route_id: str) -> list[DQRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dq_records WHERE route_id = ? ORDER BY timestamp DESC",
                (route_id,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_by_line(self, line_number: str) -> list[DQRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dq_records WHERE line_number = ? ORDER BY timestamp DESC",
                (line_number,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_all(self) -> list[DQRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dq_records ORDER BY timestamp DESC"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def mark_applied_to_spec(self, dq_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE dq_records SET applied_to_spec = 1 WHERE dq_id = ?",
                (dq_id,),
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DQRecord:
        return DQRecord(
            dq_id=row["dq_id"],
            route_id=row["route_id"],
            timestamp=row["timestamp"],
            line_number=row["line_number"],
            dq_category=row["dq_category"],
            dq_reason_text=row["dq_reason_text"],
            derived_rule=row["derived_rule"],
            constraint_scope=row["constraint_scope"],
            applied_to_spec=bool(row["applied_to_spec"]),
        )
