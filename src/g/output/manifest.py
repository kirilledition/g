"""SQLite-backed manifest for chunked output persistence."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class RunManifestRecord:
    """Persistent metadata describing one output run."""

    run_identifier: str
    association_mode: str
    schema_version: str
    chunk_size: int
    variant_limit: int | None
    configuration_fingerprint: str
    configuration_json: str


@dataclass(frozen=True)
class ManifestChunkRecord:
    """Metadata for one committed chunk."""

    chunk_identifier: int
    chunk_label: str
    variant_start_index: int
    variant_stop_index: int
    row_count: int
    file_path: str
    checksum_sha256: str
    status: str


class OutputManifest:
    """Manifest persisted as a SQLite database."""

    def __init__(self, manifest_path: Path) -> None:
        """Initialize and prepare the manifest database."""
        self.manifest_path = manifest_path
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.manifest_path)
        self.connection.execute("PRAGMA journal_mode=WAL;")
        self.connection.execute("PRAGMA synchronous=FULL;")
        self.initialize_tables()

    def initialize_tables(self) -> None:
        """Create manifest tables when absent."""
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_metadata (
                run_identifier TEXT PRIMARY KEY,
                association_mode TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                chunk_size INTEGER NOT NULL,
                variant_limit INTEGER,
                configuration_fingerprint TEXT NOT NULL,
                configuration_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_identifier INTEGER PRIMARY KEY,
                chunk_label TEXT NOT NULL,
                variant_start_index INTEGER NOT NULL,
                variant_stop_index INTEGER NOT NULL,
                row_count INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                checksum_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                committed_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def load_run_record(self) -> RunManifestRecord | None:
        """Load persisted run metadata when present."""
        cursor = self.connection.execute(
            """
            SELECT
                run_identifier,
                association_mode,
                schema_version,
                chunk_size,
                variant_limit,
                configuration_fingerprint,
                configuration_json
            FROM run_metadata
            LIMIT 1;
            """
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return RunManifestRecord(
            run_identifier=str(row[0]),
            association_mode=str(row[1]),
            schema_version=str(row[2]),
            chunk_size=int(row[3]),
            variant_limit=None if row[4] is None else int(row[4]),
            configuration_fingerprint=str(row[5]),
            configuration_json=str(row[6]),
        )

    def register_or_validate_run(self, run_record: RunManifestRecord, *, resume: bool) -> None:
        """Register a new run or validate an existing manifest for resume."""
        existing_run_record = self.load_run_record()
        if existing_run_record is None:
            if resume:
                message = f"Cannot resume run '{run_record.run_identifier}' because no manifest metadata was found."
                raise ValueError(message)
            self.insert_run_record(run_record)
            return
        if existing_run_record != run_record:
            message = (
                f"Output run '{run_record.run_identifier}' does not match the requested configuration. "
                "Choose a new run directory or resume with identical inputs and chunking."
            )
            raise ValueError(message)
        if not resume:
            message = (
                f"Output run '{run_record.run_identifier}' already exists. "
                "Use resume mode or choose a new run directory."
            )
            raise ValueError(message)

    def insert_run_record(self, run_record: RunManifestRecord) -> None:
        """Insert run metadata for a newly created manifest."""
        created_at = datetime.now(UTC).isoformat()
        self.connection.execute(
            """
            INSERT INTO run_metadata(
                run_identifier,
                association_mode,
                schema_version,
                chunk_size,
                variant_limit,
                configuration_fingerprint,
                configuration_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                run_record.run_identifier,
                run_record.association_mode,
                run_record.schema_version,
                run_record.chunk_size,
                run_record.variant_limit,
                run_record.configuration_fingerprint,
                run_record.configuration_json,
                created_at,
            ),
        )
        self.connection.commit()

    def load_committed_chunk_identifiers(self) -> set[int]:
        """Load committed chunk identifiers."""
        cursor = self.connection.execute(
            """
            SELECT chunk_identifier
            FROM chunks
            WHERE status = 'committed'
            ORDER BY chunk_identifier ASC;
            """
        )
        return {int(row[0]) for row in cursor.fetchall()}

    def insert_committed_chunk(self, chunk_record: ManifestChunkRecord) -> None:
        """Insert one committed chunk row atomically."""
        committed_at = datetime.now(UTC).isoformat()
        self.connection.execute(
            """
            INSERT OR REPLACE INTO chunks(
                chunk_identifier,
                chunk_label,
                variant_start_index,
                variant_stop_index,
                row_count,
                file_path,
                checksum_sha256,
                status,
                committed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                chunk_record.chunk_identifier,
                chunk_record.chunk_label,
                chunk_record.variant_start_index,
                chunk_record.variant_stop_index,
                chunk_record.row_count,
                chunk_record.file_path,
                chunk_record.checksum_sha256,
                chunk_record.status,
                committed_at,
            ),
        )
        self.connection.commit()

    def load_committed_chunk_records(self) -> list[ManifestChunkRecord]:
        """Return committed chunks in ascending identifier order."""
        cursor = self.connection.execute(
            """
            SELECT
                chunk_identifier,
                chunk_label,
                variant_start_index,
                variant_stop_index,
                row_count,
                file_path,
                checksum_sha256,
                status
            FROM chunks
            WHERE status = 'committed'
            ORDER BY chunk_identifier ASC;
            """
        )
        return [
            ManifestChunkRecord(
                chunk_identifier=int(row[0]),
                chunk_label=str(row[1]),
                variant_start_index=int(row[2]),
                variant_stop_index=int(row[3]),
                row_count=int(row[4]),
                file_path=str(row[5]),
                checksum_sha256=str(row[6]),
                status=str(row[7]),
            )
            for row in cursor.fetchall()
        ]

    def close(self) -> None:
        """Close the database connection."""
        self.connection.close()
