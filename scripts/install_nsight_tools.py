#!/usr/bin/env python3
"""Install Nsight Systems and Nsight Compute into the repo-local tool directory."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NsightTool:
    """One Nsight executable backed by a CUDA repository package."""

    display_name: str
    package_prefix: str
    executable_name: str


@dataclass(frozen=True)
class NsightPackage:
    """Package metadata parsed from an NVIDIA CUDA repository index."""

    package_name: str
    package_version: str
    filename: str
    sha256: str
    depends: str | None


@dataclass(frozen=True)
class CudaToolkitVersion:
    """CUDA toolkit compatibility version advertised by the NVIDIA driver.

    Attributes:
        major: CUDA toolkit major version.
        minor: CUDA toolkit minor version.

    """

    major: int
    minor: int


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOOLS_DIRECTORY = REPOSITORY_ROOT / ".tools"
CUDA_REPOSITORY_BASE_URL = "https://developer.download.nvidia.com/compute/cuda/repos"
NSIGHT_COMPUTE_META_PACKAGE_PREFIX = "cuda-nsight-compute-"
NSIGHT_TOOLS = (
    NsightTool(
        display_name="Nsight Systems",
        package_prefix="nsight-systems-",
        executable_name="nsys",
    ),
    NsightTool(
        display_name="Nsight Compute",
        package_prefix="nsight-compute-",
        executable_name="ncu",
    ),
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tools-dir",
        type=Path,
        default=Path(os.environ.get("GWAS_ENGINE_TOOLS_DIR", str(DEFAULT_TOOLS_DIRECTORY))),
        help="Repo-local tool directory that contains bin/, downloads/, and nsight/.",
    )
    parser.add_argument(
        "--repository-url",
        default=None,
        help="CUDA repository URL. Defaults to the host Ubuntu and architecture repository.",
    )
    parser.add_argument(
        "--nsight-systems-package",
        default=None,
        help="Exact nsight-systems package name to install instead of the newest package.",
    )
    parser.add_argument(
        "--nsight-compute-package",
        default=None,
        help="Exact nsight-compute package name to install instead of the newest package.",
    )
    parser.add_argument(
        "--nsight-compute-cuda-version",
        default=os.environ.get("GWAS_ENGINE_NSIGHT_COMPUTE_CUDA_VERSION"),
        help=(
            "Maximum CUDA toolkit compatibility version for Nsight Compute, such as 12.2. "
            "Defaults to GWAS_ENGINE_NSIGHT_COMPUTE_CUDA_VERSION or the CUDA version parsed from nvidia-smi."
        ),
    )
    return parser.parse_args()


def parse_os_release(os_release_path: Path) -> dict[str, str]:
    """Parse an os-release file."""
    values: dict[str, str] = {}
    if not os_release_path.exists():
        return values
    for raw_line in os_release_path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line or raw_line.startswith("#"):
            continue
        key, value = raw_line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def cuda_repository_architecture() -> str:
    """Return the CUDA repository architecture for this host."""
    machine_architecture = platform.machine()
    if machine_architecture in {"x86_64", "amd64"}:
        return "x86_64"
    if machine_architecture in {"aarch64", "arm64"}:
        return "sbsa"
    message = f"Unsupported machine architecture for CUDA repository selection: {machine_architecture}."
    raise RuntimeError(message)


def default_repository_url() -> str:
    """Build the default CUDA repository URL for the current Ubuntu host."""
    os_release = parse_os_release(Path("/etc/os-release"))
    if os_release.get("ID") != "ubuntu":
        message = "Automatic Nsight install currently supports Ubuntu hosts. Pass --repository-url to override."
        raise RuntimeError(message)
    version_identifier = os_release.get("VERSION_ID", "").replace(".", "")
    if not version_identifier:
        message = "Could not determine Ubuntu VERSION_ID. Pass --repository-url to override."
        raise RuntimeError(message)
    repository_architecture = cuda_repository_architecture()
    return f"{CUDA_REPOSITORY_BASE_URL}/ubuntu{version_identifier}/{repository_architecture}"


def read_url_text(url: str) -> str:
    """Read a URL as UTF-8 text."""
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def parse_package_index(package_index_text: str) -> list[NsightPackage]:
    """Parse Nsight package records from a Debian package index."""
    packages: list[NsightPackage] = []
    for stanza in package_index_text.split("\n\n"):
        fields: dict[str, str] = {}
        for raw_line in stanza.splitlines():
            if raw_line.startswith(" "):
                continue
            if ": " not in raw_line:
                continue
            field_name, value = raw_line.split(": ", 1)
            fields[field_name] = value
        package_name = fields.get("Package")
        if package_name is None:
            continue
        if not package_name.startswith(("nsight-systems-", "nsight-compute-", NSIGHT_COMPUTE_META_PACKAGE_PREFIX)):
            continue
        package_version = fields.get("Version")
        filename = fields.get("Filename")
        sha256 = fields.get("SHA256")
        depends = fields.get("Depends")
        if package_version is None or filename is None or sha256 is None:
            continue
        packages.append(
            NsightPackage(
                package_name=package_name,
                package_version=package_version,
                filename=filename,
                sha256=sha256,
                depends=depends,
            )
        )
    return packages


def version_components(value: str) -> list[int]:
    """Return numeric components used for package version sorting."""
    return [int(component) for component in re.findall(r"\d+", value)]


def parse_cuda_toolkit_version(value: str) -> CudaToolkitVersion:
    """Parse a CUDA toolkit compatibility version string."""
    match = re.search(r"(\d+)(?:\.(\d+))?", value)
    if match is None:
        message = f"Could not parse CUDA toolkit version from {value!r}."
        raise RuntimeError(message)
    return CudaToolkitVersion(
        major=int(match.group(1)),
        minor=int(match.group(2) or "0"),
    )


def detect_nvidia_smi_cuda_version() -> CudaToolkitVersion | None:
    """Return the CUDA compatibility version from nvidia-smi when available."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError, subprocess.SubprocessError:
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", result.stdout)
    if match is None:
        return None
    return parse_cuda_toolkit_version(match.group(1))


def nsight_compute_meta_package_version(package: NsightPackage) -> CudaToolkitVersion | None:
    """Return the CUDA version encoded in a cuda-nsight-compute package name."""
    match = re.fullmatch(r"cuda-nsight-compute-(\d+)-(\d+)", package.package_name)
    if match is None:
        return None
    return CudaToolkitVersion(major=int(match.group(1)), minor=int(match.group(2)))


def cuda_version_rank(cuda_version: CudaToolkitVersion) -> int:
    """Return a stable sortable rank for a CUDA toolkit version."""
    return cuda_version.major * 1000 + cuda_version.minor


def cuda_version_is_compatible(candidate: CudaToolkitVersion, maximum_version: CudaToolkitVersion) -> bool:
    """Return whether a CUDA toolkit version is compatible with a driver maximum."""
    return cuda_version_rank(candidate) <= cuda_version_rank(maximum_version)


def direct_nsight_compute_dependency(meta_package: NsightPackage) -> str:
    """Return the direct nsight-compute package referenced by a CUDA meta package."""
    depends = meta_package.depends or ""
    match = re.search(r"\b(nsight-compute-\d{4}\.\d+\.\d+)\b", depends)
    if match is None:
        message = f"Could not find a direct nsight-compute dependency in {meta_package.package_name}."
        raise RuntimeError(message)
    return match.group(1)


def select_driver_compatible_nsight_compute_package(
    *,
    packages: list[NsightPackage],
    maximum_cuda_version: CudaToolkitVersion,
) -> NsightPackage | None:
    """Select the newest Nsight Compute package compatible with the driver CUDA version."""
    compatible_meta_packages: list[NsightPackage] = []
    for package in packages:
        meta_package_version = nsight_compute_meta_package_version(package)
        if meta_package_version is None:
            continue
        if cuda_version_is_compatible(meta_package_version, maximum_cuda_version):
            compatible_meta_packages.append(package)
    if not compatible_meta_packages:
        return None

    selected_meta_package = max(
        compatible_meta_packages,
        key=lambda package: (
            cuda_version_rank(nsight_compute_meta_package_version(package) or CudaToolkitVersion(major=0, minor=0)),
            version_components(package.package_version),
        ),
    )
    direct_package_name = direct_nsight_compute_dependency(selected_meta_package)
    direct_candidates = [package for package in packages if package.package_name == direct_package_name]
    if not direct_candidates:
        message = f"{selected_meta_package.package_name} depends on {direct_package_name}, but it was not found."
        raise RuntimeError(message)
    return max(direct_candidates, key=lambda package: version_components(package.package_version))


def select_package(
    *,
    packages: list[NsightPackage],
    tool: NsightTool,
    requested_package_name: str | None,
    nsight_compute_cuda_version: CudaToolkitVersion | None = None,
) -> NsightPackage:
    """Select the requested or newest package for one Nsight tool."""
    candidates = [package for package in packages if package.package_name.startswith(tool.package_prefix)]
    if requested_package_name is not None:
        matching_packages = [package for package in candidates if package.package_name == requested_package_name]
        if matching_packages:
            return max(matching_packages, key=lambda package: version_components(package.package_version))
        message = f"Package {requested_package_name} was not found in the CUDA repository index."
        raise RuntimeError(message)
    if not candidates:
        message = f"No {tool.display_name} packages were found in the CUDA repository index."
        raise RuntimeError(message)
    if tool.package_prefix == "nsight-compute-" and nsight_compute_cuda_version is not None:
        compatible_package = select_driver_compatible_nsight_compute_package(
            packages=packages,
            maximum_cuda_version=nsight_compute_cuda_version,
        )
        if compatible_package is not None:
            return compatible_package
        message = (
            "No Nsight Compute package compatible with CUDA driver version "
            f"{nsight_compute_cuda_version.major}.{nsight_compute_cuda_version.minor} "
            "was found in the CUDA repository index. Pass --repository-url pointing at an older CUDA "
            "repository if the host distribution repository no longer carries that driver generation."
        )
        raise RuntimeError(message)
    return max(
        candidates,
        key=lambda package: (
            version_components(package.package_name),
            version_components(package.package_version),
        ),
    )


def calculate_sha256(download_path: Path) -> str:
    """Calculate a file SHA256 digest."""
    sha256_hash = hashlib.sha256()
    with download_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_sha256(download_path: Path, expected_sha256: str) -> None:
    """Raise when a downloaded package does not match the repository digest."""
    actual_sha256 = calculate_sha256(download_path)
    if actual_sha256 != expected_sha256:
        message = f"Checksum mismatch for {download_path.name}: expected {expected_sha256}, got {actual_sha256}."
        raise RuntimeError(message)


def download_package(*, repository_url: str, nsight_package: NsightPackage, downloads_directory: Path) -> Path:
    """Download one package if it is missing or already verified."""
    downloads_directory.mkdir(parents=True, exist_ok=True)
    download_url = urllib.parse.urljoin(repository_url.rstrip("/") + "/", nsight_package.filename.removeprefix("./"))
    download_path = downloads_directory / Path(nsight_package.filename).name
    if download_path.exists():
        verify_sha256(download_path, nsight_package.sha256)
        print(f"Reusing {download_path}")
        return download_path

    temporary_download_path = download_path.with_suffix(f"{download_path.suffix}.tmp")
    print(f"Downloading {download_url}")
    with (
        urllib.request.urlopen(download_url, timeout=60) as response,
        temporary_download_path.open("wb") as file_handle,
    ):
        shutil.copyfileobj(response, file_handle)
    verify_sha256(temporary_download_path, nsight_package.sha256)
    temporary_download_path.replace(download_path)
    return download_path


def extract_package(*, package_path: Path, nsight_package: NsightPackage, payload_parent: Path) -> Path:
    """Extract a package into the repo-local Nsight payload directory."""
    payload_directory = payload_parent / nsight_package.package_name
    marker_path = payload_directory / ".g-installed-sha256"
    if marker_path.exists() and marker_path.read_text(encoding="utf-8").strip() == nsight_package.sha256:
        print(f"Reusing extracted {nsight_package.package_name} in {payload_directory}")
        return payload_directory

    payload_directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(["dpkg-deb", "-x", str(package_path), str(payload_directory)], check=True)
    marker_path.write_text(nsight_package.sha256 + "\n", encoding="utf-8")
    print(f"Extracted {nsight_package.package_name} to {payload_directory}")
    return payload_directory


def find_executable(payload_directory: Path, executable_name: str) -> Path:
    """Find an executable inside an extracted Nsight package."""
    candidates = [
        candidate
        for candidate in payload_directory.rglob(executable_name)
        if candidate.is_file() and os.access(candidate, os.X_OK)
    ]
    if not candidates:
        message = f"Could not find executable {executable_name} in {payload_directory}."
        raise RuntimeError(message)
    return sorted(candidates, key=lambda candidate: (len(candidate.parts), str(candidate)))[0]


def link_executable(*, executable_path: Path, bin_directory: Path, executable_name: str) -> Path:
    """Link an extracted Nsight executable into the repo-local bin directory."""
    bin_directory.mkdir(parents=True, exist_ok=True)
    link_path = bin_directory / executable_name
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    relative_target = os.path.relpath(executable_path, start=bin_directory)
    link_path.symlink_to(relative_target)
    print(f"Linked {link_path} -> {relative_target}")
    return link_path


def requested_package_name_for_tool(arguments: argparse.Namespace, tool: NsightTool) -> str | None:
    """Return the requested package override for one tool."""
    if tool.package_prefix == "nsight-systems-":
        return str(arguments.nsight_systems_package) if arguments.nsight_systems_package is not None else None
    if tool.package_prefix == "nsight-compute-":
        return str(arguments.nsight_compute_package) if arguments.nsight_compute_package is not None else None
    return None


def install_tool(
    *,
    packages: list[NsightPackage],
    repository_url: str,
    tools_directory: Path,
    tool: NsightTool,
    requested_package_name: str | None,
    nsight_compute_cuda_version: CudaToolkitVersion | None,
) -> None:
    """Install one Nsight tool."""
    nsight_package = select_package(
        packages=packages,
        tool=tool,
        requested_package_name=requested_package_name,
        nsight_compute_cuda_version=nsight_compute_cuda_version,
    )
    print(f"Selected {tool.display_name}: {nsight_package.package_name} {nsight_package.package_version}")
    downloads_directory = tools_directory / "downloads" / "nsight"
    payload_parent = tools_directory / "nsight"
    package_path = download_package(
        repository_url=repository_url,
        nsight_package=nsight_package,
        downloads_directory=downloads_directory,
    )
    payload_directory = extract_package(
        package_path=package_path,
        nsight_package=nsight_package,
        payload_parent=payload_parent,
    )
    executable_path = find_executable(payload_directory, tool.executable_name)
    link_executable(
        executable_path=executable_path,
        bin_directory=tools_directory / "bin",
        executable_name=tool.executable_name,
    )


def main() -> None:
    """Install Nsight CLI tools."""
    arguments = parse_arguments()
    repository_url = str(arguments.repository_url) if arguments.repository_url is not None else default_repository_url()
    tools_directory = Path(arguments.tools_dir).expanduser().resolve()
    package_index_url = urllib.parse.urljoin(repository_url.rstrip("/") + "/", "Packages")
    packages = parse_package_index(read_url_text(package_index_url))
    if not packages:
        message = f"No Nsight packages were found in {package_index_url}."
        raise RuntimeError(message)
    if arguments.nsight_compute_cuda_version is not None:
        nsight_compute_cuda_version = parse_cuda_toolkit_version(str(arguments.nsight_compute_cuda_version))
    else:
        nsight_compute_cuda_version = detect_nvidia_smi_cuda_version()
    if nsight_compute_cuda_version is not None:
        print(
            "Selecting Nsight Compute for CUDA driver compatibility "
            f"{nsight_compute_cuda_version.major}.{nsight_compute_cuda_version.minor}"
        )
    for tool in NSIGHT_TOOLS:
        install_tool(
            packages=packages,
            repository_url=repository_url,
            tools_directory=tools_directory,
            tool=tool,
            requested_package_name=requested_package_name_for_tool(arguments, tool),
            nsight_compute_cuda_version=nsight_compute_cuda_version,
        )
    print(f"Nsight tools ready in {tools_directory / 'bin'}")


if __name__ == "__main__":
    main()
