#!/usr/bin/env python3
"""Install repo-local command-line tools for Ubuntu SLURM development."""

from __future__ import annotations

import enum
import hashlib
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


class ArchiveFormat(enum.StrEnum):
    """Supported downloaded artifact formats."""

    RAW = "raw"
    TAR_GZIP = "tar_gzip"
    ZIP = "zip"


@dataclass(frozen=True)
class BinaryMember:
    """One archive member to install as an executable."""

    archive_member_name: str
    installed_name: str


@dataclass(frozen=True)
class ToolArchive:
    """Pinned downloadable tool archive."""

    tool_name: str
    download_url: str
    archive_filename: str
    expected_sha256: str
    archive_format: ArchiveFormat
    binary_members: tuple[BinaryMember, ...]


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOOLS_DIRECTORY = REPOSITORY_ROOT / ".tools"
JUST_VERSION = "1.51.0"
RUST_TOOLCHAIN_VERSION = "1.96.0"
RUSTUP_INIT_SHA256 = "4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10"
RUSTUP_INIT_URL = "https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init"
TOOL_ARCHIVES = (
    ToolArchive(
        tool_name="just",
        download_url=(
            f"https://github.com/casey/just/releases/download/{JUST_VERSION}/"
            f"just-{JUST_VERSION}-x86_64-unknown-linux-musl.tar.gz"
        ),
        archive_filename=f"just-{JUST_VERSION}-x86_64-unknown-linux-musl.tar.gz",
        expected_sha256="c8f085ca3e885723c341d06243fc291b5abfdc8bbe3b2c076b117de490387b59",
        archive_format=ArchiveFormat.TAR_GZIP,
        binary_members=(BinaryMember(archive_member_name="just", installed_name="just"),),
    ),
    ToolArchive(
        tool_name="plink",
        download_url="https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20250819.zip",
        archive_filename="plink_linux_x86_64_20250819.zip",
        expected_sha256="0be346f0ffc5d68bc3ae0098ec3f997b470e7d75d9548e0d64a5f40f6bb8caa2",
        archive_format=ArchiveFormat.ZIP,
        binary_members=(BinaryMember(archive_member_name="plink", installed_name="plink"),),
    ),
    ToolArchive(
        tool_name="plink2",
        download_url="https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_amd_avx2_20260228.zip",
        archive_filename="plink2_linux_amd_avx2_20260228.zip",
        expected_sha256="39d14618b163ca7fead4e048d8db7306b4dc91572fd24217cce3b4d4f1cacdcd",
        archive_format=ArchiveFormat.ZIP,
        binary_members=(
            BinaryMember(archive_member_name="plink2", installed_name="plink2"),
            BinaryMember(archive_member_name="vcf_subset", installed_name="vcf_subset"),
        ),
    ),
    ToolArchive(
        tool_name="regenie",
        download_url="https://github.com/rgcgithub/regenie/releases/download/v4.1/regenie_v4.1.gz_x86_64_Linux.zip",
        archive_filename="regenie_v4.1.gz_x86_64_Linux.zip",
        expected_sha256="8d5b64cebd7e33933c9b92dd97ccffedfcc727d7be68fe8a6bec1eb959d10963",
        archive_format=ArchiveFormat.ZIP,
        binary_members=(
            BinaryMember(
                archive_member_name="regenie_v4.1.gz_x86_64_Linux",
                installed_name="regenie",
            ),
        ),
    ),
)


def resolve_tools_directory() -> Path:
    """Resolve the repo-local tools directory."""
    return Path(os.environ.get("GWAS_ENGINE_TOOLS_DIR", str(DEFAULT_TOOLS_DIRECTORY))).expanduser().resolve()


def calculate_sha256(download_path: Path) -> str:
    """Calculate a file SHA256 digest."""
    sha256_hash = hashlib.sha256()
    with download_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_sha256(download_path: Path, expected_sha256: str) -> None:
    """Raise when a file SHA256 digest does not match."""
    actual_sha256 = calculate_sha256(download_path)
    if actual_sha256 != expected_sha256:
        message = f"Checksum mismatch for {download_path.name}: expected {expected_sha256}, got {actual_sha256}."
        raise RuntimeError(message)


def download_file(download_url: str, download_path: Path, expected_sha256: str) -> None:
    """Download a file if missing or checksum-invalid."""
    if download_path.exists():
        verify_sha256(download_path, expected_sha256)
        print(f"Reusing {download_path}")
        return

    temporary_download_path = download_path.with_suffix(f"{download_path.suffix}.tmp")
    print(f"Downloading {download_url}")
    with urllib.request.urlopen(download_url) as response:
        temporary_download_path.write_bytes(response.read())
    verify_sha256(temporary_download_path, expected_sha256)
    temporary_download_path.replace(download_path)


def make_executable(executable_path: Path) -> None:
    """Mark a file executable by the owner, group, and others."""
    current_mode = executable_path.stat().st_mode
    executable_path.chmod(
        current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )


def install_binary(binary_bytes: bytes, binary_path: Path) -> None:
    """Install executable bytes atomically."""
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=binary_path.parent, delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
        temporary_file.write(binary_bytes)
    make_executable(temporary_path)
    temporary_path.replace(binary_path)


def read_zip_member(archive_path: Path, archive_member_name: str) -> bytes:
    """Read one binary member from a ZIP archive."""
    with zipfile.ZipFile(archive_path) as zip_file:
        return zip_file.read(archive_member_name)


def read_tar_gzip_member(archive_path: Path, archive_member_name: str) -> bytes:
    """Read one binary member from a gzipped tar archive."""
    with tarfile.open(archive_path, mode="r:gz") as tar_file:
        tar_member = tar_file.getmember(archive_member_name)
        extracted_file = tar_file.extractfile(tar_member)
        if extracted_file is None:
            message = f"Archive member {archive_member_name} is not a file."
            raise RuntimeError(message)
        return extracted_file.read()


def read_archive_member(tool_archive: ToolArchive, archive_path: Path, archive_member_name: str) -> bytes:
    """Read one executable payload from a supported archive."""
    if tool_archive.archive_format == ArchiveFormat.ZIP:
        return read_zip_member(archive_path, archive_member_name)
    if tool_archive.archive_format == ArchiveFormat.TAR_GZIP:
        return read_tar_gzip_member(archive_path, archive_member_name)
    if tool_archive.archive_format == ArchiveFormat.RAW:
        return archive_path.read_bytes()
    message = f"Unsupported archive format: {tool_archive.archive_format}."
    raise ValueError(message)


def install_tool_archive(tool_archive: ToolArchive, downloads_directory: Path, bin_directory: Path) -> None:
    """Download, verify, and install one pinned tool archive."""
    archive_path = downloads_directory / tool_archive.archive_filename
    download_file(tool_archive.download_url, archive_path, tool_archive.expected_sha256)
    for binary_member in tool_archive.binary_members:
        binary_path = bin_directory / binary_member.installed_name
        binary_bytes = read_archive_member(tool_archive, archive_path, binary_member.archive_member_name)
        install_binary(binary_bytes, binary_path)
        print(f"Installed {tool_archive.tool_name}: {binary_path}")


def install_rust_toolchain(tools_directory: Path, downloads_directory: Path) -> None:
    """Install a minimal Rust toolchain under the repo-local tools directory."""
    cargo_home = tools_directory / "rust" / "cargo"
    rustup_home = tools_directory / "rust" / "rustup"
    cargo_path = cargo_home / "bin" / "cargo"
    rustc_path = cargo_home / "bin" / "rustc"
    rustup_path = cargo_home / "bin" / "rustup"
    environment = os.environ.copy()
    environment["CARGO_HOME"] = str(cargo_home)
    environment["RUSTUP_HOME"] = str(rustup_home)
    if cargo_path.exists() and rustc_path.exists() and rustup_path.exists():
        current_version_process = subprocess.run(
            [str(rustc_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        if current_version_process.stdout.startswith(f"rustc {RUST_TOOLCHAIN_VERSION} "):
            print(f"Reusing Rust {RUST_TOOLCHAIN_VERSION} toolchain in {tools_directory / 'rust'}")
            return
        subprocess.run(
            [
                str(rustup_path),
                "toolchain",
                "install",
                RUST_TOOLCHAIN_VERSION,
                "--profile",
                "minimal",
            ],
            check=True,
            env=environment,
        )
        subprocess.run([str(rustup_path), "default", RUST_TOOLCHAIN_VERSION], check=True, env=environment)
        print(f"Updated Rust toolchain to {RUST_TOOLCHAIN_VERSION} in {tools_directory / 'rust'}")
        return

    rustup_init_path = downloads_directory / "rustup-init"
    download_file(RUSTUP_INIT_URL, rustup_init_path, RUSTUP_INIT_SHA256)
    make_executable(rustup_init_path)
    subprocess.run(
        [
            str(rustup_init_path),
            "--no-modify-path",
            "--profile",
            "minimal",
            "--default-toolchain",
            RUST_TOOLCHAIN_VERSION,
            "-y",
        ],
        check=True,
        env=environment,
    )
    print(f"Installed Rust {RUST_TOOLCHAIN_VERSION} toolchain in {tools_directory / 'rust'}")


def install_rust_components(tools_directory: Path) -> None:
    """Install Rust components required by development recipes."""
    cargo_home = tools_directory / "rust" / "cargo"
    rustup_home = tools_directory / "rust" / "rustup"
    rustup_path = cargo_home / "bin" / "rustup"
    environment = os.environ.copy()
    environment["CARGO_HOME"] = str(cargo_home)
    environment["RUSTUP_HOME"] = str(rustup_home)
    subprocess.run(
        [
            str(rustup_path),
            "component",
            "add",
            "rustfmt",
            "clippy",
        ],
        check=True,
        env=environment,
    )
    print("Installed Rust components: rustfmt, clippy")


def ensure_installed_command(command_name: str, bin_directory: Path) -> None:
    """Raise when a command is unavailable after bootstrap."""
    command_path = bin_directory / command_name
    if not command_path.exists():
        message = f"Expected installed command is missing: {command_path}"
        raise RuntimeError(message)


def main() -> None:
    """Install all server development tools."""
    tools_directory = resolve_tools_directory()
    downloads_directory = tools_directory / "downloads"
    bin_directory = tools_directory / "bin"
    downloads_directory.mkdir(parents=True, exist_ok=True)
    bin_directory.mkdir(parents=True, exist_ok=True)

    for tool_archive in TOOL_ARCHIVES:
        install_tool_archive(tool_archive, downloads_directory, bin_directory)
    install_rust_toolchain(tools_directory, downloads_directory)
    install_rust_components(tools_directory)

    for command_name in ("just", "plink", "plink2", "regenie"):
        ensure_installed_command(command_name, bin_directory)
    for command_name in ("cargo", "cargo-clippy", "cargo-fmt", "rustc", "rustfmt"):
        if shutil.which(command_name, path=str(tools_directory / "rust" / "cargo" / "bin")) is None:
            message = f"Expected Rust command is missing: {command_name}"
            raise RuntimeError(message)

    print(f"Server tools ready in {tools_directory}")
    print("Run: source scripts/server_env.sh")


if __name__ == "__main__":
    main()
