"""JAX runtime configuration for the GWAS engine."""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

NUMERIC_MODE_FLOAT32 = "float32"
NUMERIC_MODE_BFLOAT16 = "bfloat16"
SUPPORTED_NUMERIC_MODES = {
    NUMERIC_MODE_FLOAT32,
    NUMERIC_MODE_BFLOAT16,
}
NUMERIC_MODE = os.environ.get("G_JAX_NUMERIC_MODE", NUMERIC_MODE_FLOAT32).strip().lower()
if NUMERIC_MODE not in SUPPORTED_NUMERIC_MODES:
    message = (
        f"Unsupported G_JAX_NUMERIC_MODE value '{NUMERIC_MODE}'. Expected one of {sorted(SUPPORTED_NUMERIC_MODES)}."
    )
    raise ValueError(message)

ARRAY_DTYPE = (
    jnp.float32
    if NUMERIC_MODE == NUMERIC_MODE_FLOAT32
    else jnp.bfloat16
)
SOLVER_DTYPE = ARRAY_DTYPE
HOST_NUMPY_DTYPE = np.float32
JAX_ENABLE_X64 = False
DEFAULT_MATMUL_PRECISION = os.environ.get(
    "G_JAX_DEFAULT_MATMUL_PRECISION",
    "float32",
)
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


jax.config.update("jax_enable_x64", JAX_ENABLE_X64)
jax.config.update("jax_default_matmul_precision", DEFAULT_MATMUL_PRECISION)
if ENABLE_PERSISTENT_COMPILATION_CACHE:
    cache_directory = resolve_jax_compilation_cache_directory()
    cache_directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_directory))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS)


def configure_jax_device(device: str) -> None:
    """Configure the JAX execution device.

    Args:
        device: Either 'cpu' to restrict execution to CPU, or 'gpu' to prefer
            GPU acceleration with CPU fallback.

    """
    if device == "gpu":
        # Let JAX auto-detect GPU (CUDA or ROCm) with CPU fallback
        # Don't force platform order to avoid ROCm initialization errors on NVIDIA systems
        jax.config.update("jax_platforms", "")
    else:
        jax.config.update("jax_platforms", "cpu")


def cast_array_to_runtime_dtype(array: jax.Array) -> jax.Array:
    """Cast an array to the configured runtime storage dtype."""
    return array.astype(ARRAY_DTYPE)


def cast_array_to_solver_dtype(array: jax.Array) -> jax.Array:
    """Cast an array to the configured solver dtype."""
    return array.astype(SOLVER_DTYPE)
