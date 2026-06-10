"""JAX persistent-cache path helpers for profiling tools."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import socket
from pathlib import Path

CPUINFO_PATH = Path("/proc/cpuinfo")
DEFAULT_PROFILE_CPU_JAX_CACHE_PARENT = Path("/tmp/g-jax-cpu-profile-cache")
CPUINFO_FEATURE_FIELDS = (
    "vendor_id",
    "cpu family",
    "model",
    "model name",
    "stepping",
    "flags",
    "Features",
)
SAFE_CACHE_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9._=-]+")
CPU_FEATURE_FINGERPRINT_LENGTH = 16
CACHE_DIRECTORY_FINGERPRINT_LENGTH = 12


def safe_cache_component(value: str, fallback: str) -> str:
    """Return a filesystem-safe cache path component.

    Args:
        value: Candidate component text.
        fallback: Component used when the candidate normalizes to empty.

    Returns:
        Sanitized component text.

    """
    normalized_value = SAFE_CACHE_COMPONENT_PATTERN.sub("-", value.strip()).strip(".-")
    if normalized_value:
        return normalized_value
    return fallback


def read_cpuinfo_feature_fields(cpuinfo_path: Path = CPUINFO_PATH) -> tuple[tuple[str, str], ...]:
    """Read stable feature-bearing fields from the first CPU in `/proc/cpuinfo`.

    Args:
        cpuinfo_path: Path to a Linux cpuinfo file.

    Returns:
        Ordered CPU feature fields available on this host.

    """
    try:
        cpuinfo_text = cpuinfo_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ()
    field_values: dict[str, str] = {}
    for raw_line in cpuinfo_text.splitlines():
        if not raw_line.strip():
            break
        if ":" not in raw_line:
            continue
        field_name, raw_value = raw_line.split(":", 1)
        field_name = field_name.strip()
        if field_name not in CPUINFO_FEATURE_FIELDS or field_name in field_values:
            continue
        normalized_value = " ".join(raw_value.split())
        if field_name in {"flags", "Features"}:
            normalized_value = " ".join(sorted(normalized_value.split()))
        if normalized_value:
            field_values[field_name] = normalized_value
    return tuple(
        (field_name, field_values[field_name]) for field_name in CPUINFO_FEATURE_FIELDS if field_name in field_values
    )


def cpu_feature_fingerprint(cpuinfo_path: Path = CPUINFO_PATH) -> str:
    """Build a short fingerprint from host CPU identity and feature flags.

    Args:
        cpuinfo_path: Path to a Linux cpuinfo file.

    Returns:
        Short stable fingerprint for CPU cache partitioning.

    """
    feature_fields = read_cpuinfo_feature_fields(cpuinfo_path)
    if feature_fields:
        fingerprint_source = "\n".join(f"{field_name}={field_value}" for field_name, field_value in feature_fields)
    else:
        fingerprint_source = "\n".join(
            (
                f"machine={platform.machine()}",
                f"processor={platform.processor()}",
                f"platform={platform.platform()}",
            )
        )
    return hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:CPU_FEATURE_FINGERPRINT_LENGTH]


def resolve_cpu_feature_aware_cache_directory(
    base_cache_directory: Path,
    *,
    cache_parent_environment_variable: str = "G_PROFILE_CPU_JAX_CACHE_PARENT",
    default_cache_parent: Path = DEFAULT_PROFILE_CPU_JAX_CACHE_PARENT,
    cpuinfo_path: Path = CPUINFO_PATH,
) -> Path:
    """Resolve a node and CPU-feature-aware JAX cache directory.

    Args:
        base_cache_directory: Logical cache directory requested by the profiling tool.
        cache_parent_environment_variable: Environment variable that overrides the node-local parent.
        default_cache_parent: Default parent directory for isolated CPU caches.
        cpuinfo_path: Path to a Linux cpuinfo file.

    Returns:
        Effective cache directory isolated by host, CPU feature fingerprint, and logical cache path.

    """
    cache_parent_text = os.environ.get(cache_parent_environment_variable)
    cache_parent = default_cache_parent
    if cache_parent_text:
        cache_parent = Path(cache_parent_text)
    expanded_base_cache_directory = base_cache_directory.expanduser()
    host_component = safe_cache_component(socket.gethostname(), "unknown-host")
    base_name_component = safe_cache_component(expanded_base_cache_directory.name, "jax-cache")
    base_directory_fingerprint = hashlib.sha256(str(expanded_base_cache_directory).encode("utf-8")).hexdigest()[
        :CACHE_DIRECTORY_FINGERPRINT_LENGTH
    ]
    return (
        cache_parent.expanduser()
        / f"host-{host_component}"
        / f"features-{cpu_feature_fingerprint(cpuinfo_path)}"
        / f"{base_name_component}-{base_directory_fingerprint}"
    )
