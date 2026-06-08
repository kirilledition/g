"""Shared pytest runtime setup."""

from __future__ import annotations

import dataclasses

from g import jax_runtime, jax_setup
from g.interface import config

COMPUTE_CONFIG = dataclasses.replace(config.load_packaged_config().g_compute, jax_persistent_cache=False)
jax_setup.configure_jax_runtime(jax_runtime.build_jax_runtime_policy(COMPUTE_CONFIG))
