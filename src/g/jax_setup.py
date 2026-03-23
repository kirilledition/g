"""JAX runtime configuration for the GWAS engine."""

from __future__ import annotations

import os
from pathlib import Path

import jax

ENABLE_X64 = True
ENABLE_PERSISTENT_COMPILATION_CACHE = os.environ.get("G_ENABLE_JAX_PERSISTENT_COMPILATION_CACHE", "1") == "1"
PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES = -1
PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS = 0


def resolve_jax_compilation_cache_directory() -> Path:
    """Resolve the persistent JAX compilation cache directory."""
    configured_cache_directory = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if configured_cache_directory is not None:
        return Path(configured_cache_directory).expanduser()
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home is not None:
        return Path(xdg_cache_home).expanduser() / "g" / "jax"
    return Path.home() / ".cache" / "g" / "jax"


jax.config.update("jax_enable_x64", ENABLE_X64)
if ENABLE_PERSISTENT_COMPILATION_CACHE:
    cache_directory = resolve_jax_compilation_cache_directory()
    cache_directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_directory))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS)
