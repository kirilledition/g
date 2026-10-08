"""Byte-verified localization and success-only publication without shell commands."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import typing
from pathlib import Path

from tooling.workbench import manifest


@dataclasses.dataclass(frozen=True)
class FileDigest:
    """Stable file identity observed while hashing a regular file."""

    sha256: str
    size_bytes: int


@dataclasses.dataclass(frozen=True)
class LocalizedFile:
    """A verified copy and the immutable source generation when applicable."""

    path: Path
    source: manifest.InputFile
    generation: str | None


def file_digest(path: Path) -> FileDigest:
    """Hash a file and reject changes during the read."""
    if not path.is_file():
        raise ValueError(f"Required regular file is unavailable: {path}.")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"Input changed while being verified: {path}.")
    return FileDigest(digest.hexdigest(), after.st_size)


def verify_file(path: Path, expected: manifest.InputFile) -> FileDigest:
    """Require exact content identity rather than trusting a copied filename."""
    digest = file_digest(path)
    if digest.size_bytes != expected.size_bytes or digest.sha256 != expected.sha256:
        raise ValueError(f"Input checksum or byte size does not match the manifest: {path.name}.")
    return digest


def gcloud_arguments(arguments: list[str], billing_project: str | None) -> list[str]:
    """Build shell-free Cloud CLI arguments with an optional requester project."""
    result = ["gcloud", "--quiet", "storage", *arguments]
    if billing_project is not None:
        if not billing_project or any(ord(character) < 32 for character in billing_project):
            raise ValueError("billing_project must be nonempty printable text.")
        result.extend(["--billing-project", billing_project])
    return result


def describe_object(uri: str, billing_project: str | None) -> dict[str, manifest.JsonValue]:
    """Read Cloud CLI metadata without silently treating errors as absence."""
    result = subprocess.run(
        gcloud_arguments(["objects", "describe", uri, "--format=json"], billing_project),
        capture_output=True,
        text=True,
        check=True,
    )
    return manifest.mapping(typing.cast("manifest.JsonValue", json.loads(result.stdout)), "GCS object metadata")


def object_size(metadata: dict[str, manifest.JsonValue]) -> int:
    """Read the CLI's JSON string or integer byte count strictly."""
    size = metadata.get("size")
    if isinstance(size, str) and size.isascii() and size.isdecimal():
        size = int(size)
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError("GCS object size must be a nonnegative integer.")
    return size


def object_generation(metadata: dict[str, manifest.JsonValue]) -> str:
    """Require a concrete object generation before downloading content."""
    generation = metadata.get("generation")
    if isinstance(generation, int) and not isinstance(generation, bool):
        generation = str(generation)
    if (
        not isinstance(generation, str)
        or not generation.isascii()
        or not generation.isdecimal()
        or int(generation) <= 0
    ):
        raise ValueError("GCS object metadata does not contain a valid immutable generation.")
    return generation


def check_remote_source(source: manifest.InputFile, billing_project: str | None) -> str:
    """Validate object size and resolve one immutable generation."""
    uri = f"{source.uri}#{source.generation}" if source.generation is not None else source.uri
    metadata = describe_object(uri, billing_project)
    generation = object_generation(metadata)
    if object_size(metadata) != source.size_bytes:
        raise ValueError("GCS object byte size does not match the job manifest.")
    if source.generation is not None and generation != source.generation:
        raise ValueError("GCS object generation does not match the pinned job manifest.")
    return generation


def localize_file(source: manifest.InputFile, destination: Path, billing_project: str | None) -> LocalizedFile:
    """Copy or download one pinned input, verify bytes, then publish locally."""
    if destination.exists() or destination.is_symlink():
        raise ValueError("Refusing to replace an existing localized input.")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    partial_path = destination.with_name(destination.name + ".partial")
    if partial_path.exists() or partial_path.is_symlink():
        raise ValueError("Refusing to replace an existing partial localization.")
    generation = None
    if source.uri.startswith("gs://"):
        generation = check_remote_source(source, billing_project)
        subprocess.run(
            gcloud_arguments(
                ["cp", f"{source.uri}#{generation}", str(partial_path), "--do-not-decompress"], billing_project
            ),
            check=True,
        )
    else:
        if Path(source.uri).stat().st_size != source.size_bytes:
            raise ValueError("Local source byte size does not match the job manifest.")
        with Path(source.uri).open("rb") as source_file, partial_path.open("xb") as target_file:
            shutil.copyfileobj(source_file, target_file, length=8 * 1024 * 1024)
    verify_file(partial_path, source)
    partial_path.rename(destination)
    return LocalizedFile(destination, source, generation)


def write_json_atomic(path: Path, value: object) -> None:
    """Replace a small diagnostic document atomically on the same filesystem."""
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def regular_files(root: Path) -> tuple[Path, ...]:
    """List only regular files, rejecting links before a publication copy."""
    result: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Publication bundles must not contain symbolic links.")
        if path.is_file():
            result.append(path)
        elif not path.is_dir():
            raise ValueError("Publication bundles must contain only regular files and directories.")
    return tuple(result)


def object_md5(path: Path) -> str:
    """Calculate the GCS non-composite object checksum for upload verification."""
    checksum = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            checksum.update(block)
    return base64.b64encode(checksum.digest()).decode("ascii")


def publish_cloud_file(path: Path, uri: str, billing_project: str | None) -> None:
    """Upload a new object and require server-observed size and content checksum."""
    subprocess.run(
        gcloud_arguments(["cp", str(path), uri, "--if-generation-match=0"], billing_project),
        env={**os.environ, "CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED": "false"},
        check=True,
    )
    metadata = describe_object(uri, billing_project)
    observed_md5 = metadata.get("md5_hash", metadata.get("md5Hash"))
    if object_size(metadata) != path.stat().st_size or observed_md5 != object_md5(path):
        raise ValueError(
            "Uploaded object failed server-observed size or MD5 verification; publication remains unconfirmed."
        )


def publish_bundle(bundle: Path, output_uri: str, attempt_id: str, billing_project: str | None) -> str:
    """Publish an immutable attempt; cloud commit marker is uploaded last."""
    manifest.safe_name(attempt_id, "attempt_id")
    paths = regular_files(bundle)
    marker = bundle / "COMMITTED.json"
    if marker not in paths:
        raise ValueError("A completed, verified bundle must contain COMMITTED.json.")
    if output_uri.startswith("gs://"):
        destination = f"{output_uri}/{attempt_id}"
        for path in paths:
            if path != marker:
                publish_cloud_file(path, f"{destination}/{path.relative_to(bundle).as_posix()}", billing_project)
        publish_cloud_file(marker, f"{destination}/COMMITTED.json", billing_project)
        return destination
    parent = Path(output_uri)
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination_path = parent / attempt_id
    if destination_path.exists() or destination_path.is_symlink():
        raise ValueError("Refusing to replace a published attempt.")
    temporary_path = parent / f".{attempt_id}.publishing"
    temporary_path.mkdir(mode=0o700)
    for path in paths:
        relative_path = path.relative_to(bundle)
        copied_path = temporary_path / relative_path
        copied_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(path, copied_path)
        if file_digest(path) != file_digest(copied_path):
            raise ValueError("Local publication copy failed verification.")
    # Both paths share a filesystem. The entire success bundle appears at once.
    temporary_path.rename(destination_path)
    return str(destination_path)
