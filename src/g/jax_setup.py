"""JAX runtime configuration for the GWAS engine."""

from __future__ import annotations

import jax

ENABLE_X64 = True

jax.config.update("jax_enable_x64", ENABLE_X64)
