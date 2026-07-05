"""Unit tests for compute dtype adapters."""

from __future__ import annotations

import typing

import jax.numpy as jnp
import pytest

from g import types
from g.compute.common import dtype as compute_dtype


def test_resolve_jax_dtype_maps_supported_floating_point_dtypes() -> None:
    assert compute_dtype.resolve_jax_dtype(types.FloatingPointDtype.FLOAT32) == jnp.float32
    assert compute_dtype.resolve_jax_dtype(types.FloatingPointDtype.FLOAT64) == jnp.float64


def test_resolve_jax_dtype_rejects_raw_string_values() -> None:
    raw_dtype = typing.cast("types.FloatingPointDtype", "float32")

    with pytest.raises(TypeError, match="Expected FloatingPointDtype for JAX dtype resolution"):
        compute_dtype.resolve_jax_dtype(raw_dtype)


def test_resolve_jax_dtype_rejects_unsupported_values() -> None:
    unsupported_dtype = typing.cast("types.FloatingPointDtype", "float16")

    with pytest.raises(TypeError, match="Expected FloatingPointDtype for JAX dtype resolution"):
        compute_dtype.resolve_jax_dtype(unsupported_dtype)
