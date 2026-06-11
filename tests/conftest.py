"""Shared pytest runtime setup."""

from __future__ import annotations

from g import jax_runtime, jax_setup
from g.interface import config

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
jax_setup.configure_jax_runtime(jax_runtime.build_jax_runtime_policy(COMPUTE_CONFIG))
