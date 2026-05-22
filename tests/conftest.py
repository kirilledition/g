"""Shared pytest runtime setup."""

from __future__ import annotations

from g import jax_setup

jax_setup.configure_jax_runtime(persistent_cache=False)
