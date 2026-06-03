"""Floating-point dtype helpers for JAX compute kernels."""

from __future__ import annotations

import typing

import jax.numpy as jnp

from g import types


def resolve_jax_dtype(floating_point_dtype: types.FloatingPointDtype) -> typing.Any:
    """Resolve a configured floating-point dtype to a JAX dtype."""
    if floating_point_dtype == types.FloatingPointDtype.FLOAT64:
        return jnp.float64
    return jnp.float32
