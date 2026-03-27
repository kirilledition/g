from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    import pytest

import jax.numpy as jnp

from g.jax_setup import (
    FLOAT_DTYPE,
    configure_jax_device,
    resolve_jax_compilation_cache_directory,
)


def test_resolve_jax_cache_uses_jax_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure JAX_COMPILATION_CACHE_DIR takes highest precedence."""
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/custom/jax/cache")
    monkeypatch.setenv("XDG_CACHE_HOME", "/custom/xdg/cache")

    result = resolve_jax_compilation_cache_directory()

    assert result == Path("/custom/jax/cache")


def test_resolve_jax_cache_expands_jax_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure JAX_COMPILATION_CACHE_DIR is expanded."""
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "~/custom/jax/cache")
    monkeypatch.setenv("HOME", "/mock/home")

    result = resolve_jax_compilation_cache_directory()

    assert result == Path("/mock/home/custom/jax/cache")


def test_resolve_jax_cache_uses_xdg_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure XDG_CACHE_HOME is used when JAX_COMPILATION_CACHE_DIR is not set."""
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/custom/xdg/cache")

    result = resolve_jax_compilation_cache_directory()

    assert result == Path("/custom/xdg/cache/g/jax")


def test_resolve_jax_cache_expands_xdg_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure XDG_CACHE_HOME is expanded."""
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "~/custom/xdg/cache")
    monkeypatch.setenv("HOME", "/mock/home")

    result = resolve_jax_compilation_cache_directory()

    assert result == Path("/mock/home/custom/xdg/cache/g/jax")


def test_resolve_jax_cache_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure fallback to ~/.cache/g/jax is used when no env vars are set."""
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", "/mock/home")
    monkeypatch.setattr(Path, "home", lambda: Path("/mock/home"))

    result = resolve_jax_compilation_cache_directory()

    assert result == Path("/mock/home/.cache/g/jax")


def test_configure_jax_device_gpu() -> None:
    """Ensure configuring for GPU sets an empty platforms string for auto-detection."""
    with patch("g.jax_setup.jax.config.update") as mock_update:
        configure_jax_device("gpu")
        mock_update.assert_called_once_with("jax_platforms", "")


def test_configure_jax_device_cpu() -> None:
    """Ensure configuring for CPU explicitly sets the CPU platform."""
    with patch("g.jax_setup.jax.config.update") as mock_update:
        configure_jax_device("cpu")
        mock_update.assert_called_once_with("jax_platforms", "cpu")


def test_configure_jax_device_unknown_fallback() -> None:
    """Ensure configuring for an unknown device falls back to CPU."""
    with patch("g.jax_setup.jax.config.update") as mock_update:
        configure_jax_device("tpu")
        mock_update.assert_called_once_with("jax_platforms", "cpu")


def test_float_dtype_is_float32() -> None:
    """Ensure the codebase-wide JAX float dtype is fixed to float32."""
    assert jnp.float32 == FLOAT_DTYPE
