#!/usr/bin/env python3
"""Install repo-local command-line tools for Ubuntu SLURM development."""

from __future__ import annotations

import enum
import os
import shutil
import stat
import subprocess
import tempfile
import typing
from dataclasses import dataclass
from pathlib import Path

import hydra

from tooling.common import downloads as tooling_downloads
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat

if typing.TYPE_CHECKING:
    import omegaconf


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


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOOLS_DIRECTORY = REPOSITORY_ROOT / ".tools"
JUST_VERSION = "1.51.0"
MOLD_VERSION = "2.41.0"
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
        tool_name="mold",
        download_url=(
            f"https://github.com/rui314/mold/releases/download/v{MOLD_VERSION}/mold-{MOLD_VERSION}-x86_64-linux.tar.gz"
        ),
        archive_filename=f"mold-{MOLD_VERSION}-x86_64-linux.tar.gz",
        expected_sha256="a3696680d99e692970590a178bc3a33d78d60d1c6dc9db7a11b557b02b751f5d",
        archive_format=ArchiveFormat.TAR_GZIP,
        binary_members=(
            BinaryMember(
                archive_member_name=f"mold-{MOLD_VERSION}-x86_64-linux/bin/mold",
                installed_name="mold",
            ),
            BinaryMember(
                archive_member_name=f"mold-{MOLD_VERSION}-x86_64-linux/bin/ld.mold",
                installed_name="ld.mold",
            ),
        ),
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


@dataclass(frozen=True)
class BootstrapToolsArguments:
    """Resolved parameters for server tool bootstrap.

    Attributes:
        tools_dir: Optional repo-local tools directory override.

    """

    tools_dir: Path | None


@dataclass(frozen=True)
class RetrievedToolArchive:
    """Downloaded archive and extracted executable member paths."""

    archive_path: Path
    member_paths: dict[str, Path]


def resolve_tools_directory() -> Path:
    """Resolve the repo-local tools directory."""
    return Path(os.environ.get("GWAS_ENGINE_TOOLS_DIR", str(DEFAULT_TOOLS_DIRECTORY))).expanduser().resolve()


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


def build_archive_processor(
    *,
    tool_archive: ToolArchive,
    extract_directory: Path,
) -> tooling_downloads.DownloadProcessor | None:
    """Build the Pooch processor for a supported archive."""
    member_names = tuple(binary_member.archive_member_name for binary_member in tool_archive.binary_members)
    if tool_archive.archive_format == ArchiveFormat.ZIP:
        return tooling_downloads.build_unzip_processor(members=member_names, extract_directory=extract_directory)
    if tool_archive.archive_format == ArchiveFormat.TAR_GZIP:
        return tooling_downloads.build_untar_processor(members=member_names, extract_directory=extract_directory)
    if tool_archive.archive_format == ArchiveFormat.RAW:
        return None
    message = f"Unsupported archive format: {tool_archive.archive_format}."
    raise ValueError(message)


def find_extracted_member_path(
    *,
    processed_paths: tuple[Path, ...],
    extract_directory: Path,
    archive_member_name: str,
) -> Path:
    """Find one extracted archive member returned by Pooch."""
    for processed_path in processed_paths:
        if processed_path.name == archive_member_name or processed_path.as_posix().endswith(archive_member_name):
            return processed_path
    fallback_path = extract_directory / archive_member_name
    if fallback_path.exists():
        return fallback_path
    processed_listing = ", ".join(str(processed_path) for processed_path in processed_paths)
    message = f"Archive member {archive_member_name} was not extracted. Processed paths: {processed_listing}"
    raise RuntimeError(message)


def retrieve_tool_archive(tool_archive: ToolArchive, downloads_directory: Path) -> RetrievedToolArchive:
    """Retrieve and process one pinned tool archive through Pooch."""
    archive_path = downloads_directory / tool_archive.archive_filename
    extract_directory = downloads_directory / f"{tool_archive.archive_filename}.contents"
    processor = build_archive_processor(tool_archive=tool_archive, extract_directory=extract_directory)
    downloaded_file = tooling_downloads.retrieve_file(
        download_url=tool_archive.download_url,
        destination_path=archive_path,
        expected_sha256=tool_archive.expected_sha256,
        processor=processor,
    )
    member_paths: dict[str, Path] = {}
    for binary_member in tool_archive.binary_members:
        if tool_archive.archive_format == ArchiveFormat.RAW:
            member_paths[binary_member.archive_member_name] = downloaded_file.path
        else:
            member_paths[binary_member.archive_member_name] = find_extracted_member_path(
                processed_paths=downloaded_file.processed_paths,
                extract_directory=extract_directory,
                archive_member_name=binary_member.archive_member_name,
            )
    return RetrievedToolArchive(archive_path=downloaded_file.path, member_paths=member_paths)


def install_tool_archive(tool_archive: ToolArchive, downloads_directory: Path, bin_directory: Path) -> None:
    """Download, verify, and install one pinned tool archive."""
    retrieved_archive = retrieve_tool_archive(tool_archive, downloads_directory)
    for binary_member in tool_archive.binary_members:
        binary_path = bin_directory / binary_member.installed_name
        binary_bytes = retrieved_archive.member_paths[binary_member.archive_member_name].read_bytes()
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
    tooling_downloads.retrieve_file(
        download_url=RUSTUP_INIT_URL,
        destination_path=rustup_init_path,
        expected_sha256=RUSTUP_INIT_SHA256,
    )
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


def run_tool(arguments: BootstrapToolsArguments) -> None:
    """Install all server development tools."""
    tools_directory = (
        arguments.tools_dir.expanduser().resolve() if arguments.tools_dir is not None else resolve_tools_directory()
    )
    downloads_directory = tools_directory / "downloads"
    bin_directory = tools_directory / "bin"
    downloads_directory.mkdir(parents=True, exist_ok=True)
    bin_directory.mkdir(parents=True, exist_ok=True)

    for tool_archive in TOOL_ARCHIVES:
        install_tool_archive(tool_archive, downloads_directory, bin_directory)
    install_rust_toolchain(tools_directory, downloads_directory)
    install_rust_components(tools_directory)

    for command_name in ("just", "mold", "ld.mold", "plink", "plink2", "regenie"):
        ensure_installed_command(command_name, bin_directory)
    for command_name in ("cargo", "cargo-clippy", "cargo-fmt", "rustc", "rustfmt"):
        if shutil.which(command_name, path=str(tools_directory / "rust" / "cargo" / "bin")) is None:
            message = f"Expected Rust command is missing: {command_name}"
            raise RuntimeError(message)

    print(f"Server tools ready in {tools_directory}")
    print("Run: source tooling/server/server_env.sh")


def build_arguments_from_config(config: omegaconf.DictConfig) -> BootstrapToolsArguments:
    """Resolve server bootstrap parameters from Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return BootstrapToolsArguments(
        tools_dir=tooling_hydra_arguments.path_or_none(tool_values["tools_dir"]),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="server_bootstrap_tools")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Install server development tools from Hydra configuration."""
    run_tool(build_arguments_from_config(config))


def main() -> None:
    """Install server development tools from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
