"""SQLite-backed manifest for chunked output persistence."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ManifestChunkRecord:
    """Metadata for one committed chunk."""

    chunk_identifier: int
    chunk_label: str
    row_count: int
    file_path: str
    checksum_sha256: str
    status: str


class OutputManifest:
    """Append-only manifest persisted as a SQLite database."""

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
                created_at TEXT NOT NULL
            );
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_identifier INTEGER PRIMARY KEY,
                chunk_label TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                checksum_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                committed_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def register_run(self, run_identifier: str, association_mode: str, schema_version: str) -> None:
        """Record run metadata if it is not already registered."""
        created_at = datetime.now(UTC).isoformat()
        self.connection.execute(
            """
            INSERT OR IGNORE INTO run_metadata(run_identifier, association_mode, schema_version, created_at)
            VALUES (?, ?, ?, ?);
            """,
            (run_identifier, association_mode, schema_version, created_at),
        )
        self.connection.commit()

    def load_committed_chunk_identifiers(self) -> set[int]:
        """Load chunk identifiers with committed status."""
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
                row_count,
                file_path,
                checksum_sha256,
                status,
                committed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                chunk_record.chunk_identifier,
                chunk_record.chunk_label,
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
            SELECT chunk_identifier, chunk_label, row_count, file_path, checksum_sha256, status
            FROM chunks
            WHERE status = 'committed'
            ORDER BY chunk_identifier ASC;
            """
        )
        return [
            ManifestChunkRecord(
                chunk_identifier=int(row[0]),
                chunk_label=str(row[1]),
                row_count=int(row[2]),
                file_path=str(row[3]),
                checksum_sha256=str(row[4]),
                status=str(row[5]),
            )
            for row in cursor.fetchall()
        ]

    def close(self) -> None:
        """Close the database connection."""
        self.connection.close()
