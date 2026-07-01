"""Shared pytest runtime setup."""

from __future__ import annotations

from g.interface import config
from g.jax_runtime import resolution as jax_runtime_resolution
from g.jax_runtime import setup as jax_runtime_setup

COMPUTE_CONFIG = config.RegenieConfig.from_options(
    {
        "bgen": "dataset.bgen",
        "phenoFile": "phenotype.tsv",
        "phenoCol": "trait",
        "pred": "predictions.list",
        "out": "results/output",
        "jax_persistent_cache": False,
    }
).g_compute
JAX_RUNTIME_POLICY = jax_runtime_resolution.resolve_jax_runtime_policy(COMPUTE_CONFIG)
jax_runtime_setup.configure_before_backend_init(
    native_setup_session=jax_runtime_resolution.build_native_jax_runtime_setup_session(JAX_RUNTIME_POLICY),
    diagnostic_sink=None,
)
