"""Shared pytest runtime setup."""

from __future__ import annotations

import dataclasses

from g import jax_setup
from g.interface import config

COMPUTE_CONFIG = dataclasses.replace(config.load_packaged_config().g_compute, jax_persistent_cache=False)
jax_setup.configure_jax_runtime(
    persistent_cache=COMPUTE_CONFIG.jax_persistent_cache,
    persistent_cache_min_entry_size_bytes=COMPUTE_CONFIG.jax_persistent_cache_min_entry_size_bytes,
    persistent_cache_min_compile_time_seconds=COMPUTE_CONFIG.jax_persistent_cache_min_compile_time_seconds,
)
