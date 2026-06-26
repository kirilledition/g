"""Pooch-backed download, cache, and archive helpers for development tooling."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import typing
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import pooch
import pooch.typing

POOCH_HASH_ALGORITHM = "sha256"
DOWNLOAD_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60
DEFAULT_DOWNLOAD_CHUNK_SIZE_BYTES = 1024 * 1024

DownloadProcessor = pooch.typing.Processor


@dataclass(frozen=True)
class DownloadRegistryEntry:
    """One named downloadable file in a tooling registry.

    Attributes:
        name: Stable registry key.
        download_url: Source URL.
        file_name: Local cache file name.
        expected_sha256: Expected SHA-256 hex digest, when available.
        kind: Human-readable file kind.
        description: Optional note for docs and manifests.

    """

    name: str
    download_url: str
    file_name: str
    expected_sha256: str | None
    kind: str
    description: str | None = None


@dataclass(frozen=True)
class DownloadManifest:
    """Manifest describing one retrieved file."""

    schema_version: int
    download_url: str
    path: str
    expected_sha256: str | None
    actual_sha256: str
    size_bytes: int
    managed_by: str


@dataclass(frozen=True)
class DownloadedFile:
    """Resolved download result.

    Attributes:
        path: Cached file path.
        download_url: Source URL.
        expected_sha256: Expected SHA-256 hex digest, when available.
        actual_sha256: Actual SHA-256 hex digest of the cached file.
        size_bytes: Cached file size.
        manifest_path: Sidecar manifest path.
        processed_paths: Paths returned by a Pooch processor, such as extracted files.

    """

    path: Path
    download_url: str
    expected_sha256: str | None
    actual_sha256: str
    size_bytes: int
    manifest_path: Path
    processed_paths: tuple[Path, ...]


def calculate_sha256(download_path: Path) -> str:
    """Calculate a file SHA-256 digest."""
    sha256_hash = hashlib.sha256()
    with download_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(DEFAULT_DOWNLOAD_CHUNK_SIZE_BYTES), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def pooch_known_hash(expected_sha256: str | None) -> str | None:
    """Convert a SHA-256 digest into Pooch's explicit known-hash format."""
    if expected_sha256 is None:
        return None
    if expected_sha256.startswith(f"{POOCH_HASH_ALGORITHM}:"):
        return expected_sha256
    return f"{POOCH_HASH_ALGORITHM}:{expected_sha256}"


def download_manifest_path(download_path: Path) -> Path:
    """Return the manifest path for a downloaded file."""
    return download_path.with_name(f"{download_path.name}.manifest.json")


def write_download_manifest(manifest_path: Path, manifest: DownloadManifest) -> None:
    """Write a sidecar manifest for one retrieved file."""
    manifest_path.write_text(
        json.dumps(dataclasses.asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_processed_paths(processed_result: object) -> tuple[Path, ...]:
    """Normalize a Pooch retrieve result into processed paths."""
    if isinstance(processed_result, str):
        return (Path(processed_result),)
    if isinstance(processed_result, Path):
        return (processed_result,)
    if isinstance(processed_result, list | tuple):
        processed_paths: list[Path] = []
        for path_value in processed_result:
            if isinstance(path_value, str):
                processed_paths.append(Path(path_value))
            elif isinstance(path_value, Path):
                processed_paths.append(path_value)
            else:
                message = f"Pooch processor returned a non-path value: {path_value!r}"
                raise TypeError(message)
        return tuple(processed_paths)
    return ()


def http_downloader_for_url(
    download_url: str,
    *,
    timeout_seconds: int,
    chunk_size_bytes: int,
) -> pooch.typing.Downloader | None:
    """Return a timeout-aware HTTP downloader when the URL scheme supports it."""
    url_scheme = urllib.parse.urlparse(download_url).scheme
    if url_scheme not in {"http", "https"}:
        return None
    return typing.cast(
        "pooch.typing.Downloader",
        pooch.HTTPDownloader(timeout=timeout_seconds, chunk_size=chunk_size_bytes),
    )


def retrieve_file(
    *,
    download_url: str,
    destination_path: Path,
    expected_sha256: str | None,
    processor: DownloadProcessor | None = None,
    timeout_seconds: int = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    chunk_size_bytes: int = DEFAULT_DOWNLOAD_CHUNK_SIZE_BYTES,
) -> DownloadedFile:
    """Retrieve one file through Pooch and write a download manifest.

    Args:
        download_url: Remote or local source URL.
        destination_path: Local cache path that Pooch should populate.
        expected_sha256: Expected SHA-256 hex digest. ``None`` allows unverified
            retrieval for legacy registries that do not yet publish hashes.
        processor: Optional Pooch processor, such as ``Unzip`` or ``Untar``.
        timeout_seconds: HTTP timeout for remote downloads.
        chunk_size_bytes: HTTP streaming chunk size.

    Returns:
        Downloaded file metadata and any processor output paths.

    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    downloader = http_downloader_for_url(
        download_url,
        timeout_seconds=timeout_seconds,
        chunk_size_bytes=chunk_size_bytes,
    )
    processed_result = pooch.retrieve(
        url=download_url,
        known_hash=pooch_known_hash(expected_sha256),
        fname=destination_path.name,
        path=destination_path.parent,
        processor=processor,
        downloader=downloader,
    )
    cached_path = destination_path.resolve()
    actual_sha256 = calculate_sha256(cached_path)
    manifest_path = download_manifest_path(cached_path)
    write_download_manifest(
        manifest_path,
        DownloadManifest(
            schema_version=DOWNLOAD_MANIFEST_SCHEMA_VERSION,
            download_url=download_url,
            path=str(cached_path),
            expected_sha256=expected_sha256,
            actual_sha256=actual_sha256,
            size_bytes=cached_path.stat().st_size,
            managed_by="pooch",
        ),
    )
    return DownloadedFile(
        path=cached_path,
        download_url=download_url,
        expected_sha256=expected_sha256,
        actual_sha256=actual_sha256,
        size_bytes=cached_path.stat().st_size,
        manifest_path=manifest_path,
        processed_paths=normalize_processed_paths(processed_result),
    )


def retrieve_registry_entry(
    *,
    registry_entry: DownloadRegistryEntry,
    destination_directory: Path,
    processor: DownloadProcessor | None = None,
) -> DownloadedFile:
    """Retrieve one file from a named tooling registry entry."""
    return retrieve_file(
        download_url=registry_entry.download_url,
        destination_path=destination_directory / registry_entry.file_name,
        expected_sha256=registry_entry.expected_sha256,
        processor=processor,
    )


def build_unzip_processor(*, members: tuple[str, ...], extract_directory: Path) -> DownloadProcessor:
    """Build a Pooch ZIP archive processor."""
    extract_directory.mkdir(parents=True, exist_ok=True)
    return typing.cast(
        "DownloadProcessor",
        pooch.Unzip(members=list(members), extract_dir=str(extract_directory)),
    )


def build_untar_processor(*, members: tuple[str, ...], extract_directory: Path) -> DownloadProcessor:
    """Build a Pooch tar archive processor."""
    extract_directory.mkdir(parents=True, exist_ok=True)
    return typing.cast(
        "DownloadProcessor",
        pooch.Untar(members=list(members), extract_dir=str(extract_directory)),
    )
