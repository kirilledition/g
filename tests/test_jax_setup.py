from __future__ import annotations

import typing
from pathlib import Path
from unittest.mock import patch

if typing.TYPE_CHECKING:
    import pytest

import jax.numpy as jnp

from g.jax_setup import (
    FLOAT_DTYPE,
    configure_jax_device,
    require_gpu_device,
    resolve_jax_compilation_cache_directory,
)
from g.types import Device


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
    """Ensure configuring for GPU requires the CUDA backend."""
    with (
        patch("g.jax_setup.jax.config.update") as mock_update,
        patch("g.jax_setup.require_gpu_device") as mock_require_gpu_device,
    ):
        configure_jax_device(Device.GPU)
        mock_update.assert_called_once_with("jax_platforms", "cuda")
        mock_require_gpu_device.assert_called_once_with()


def test_configure_jax_device_cpu() -> None:
    """Ensure configuring for CPU explicitly sets the CPU platform."""
    with patch("g.jax_setup.jax.config.update") as mock_update:
        configure_jax_device(Device.CPU)
        mock_update.assert_called_once_with("jax_platforms", "cpu")


def test_require_gpu_device_accepts_gpu_platform() -> None:
    """Ensure GPU validation accepts a JAX GPU device."""

    class FakeDevice:
        platform = "gpu"

    with patch("g.jax_setup.jax.devices", return_value=[FakeDevice()]):
        require_gpu_device()


def test_require_gpu_device_rejects_cpu_only_backend() -> None:
    """Ensure GPU validation rejects CPU-only device lists."""

    class FakeDevice:
        platform = "cpu"

        def __str__(self) -> str:
            return "CpuDevice(id=0)"

    with patch("g.jax_setup.jax.devices", return_value=[FakeDevice()]):
        try:
            require_gpu_device()
        except RuntimeError as error:
            assert "did not report any GPU devices" in str(error)
        else:
            raise AssertionError("Expected GPU validation to fail for CPU-only devices.")


def test_require_gpu_device_wraps_backend_initialization_errors() -> None:
    """Ensure CUDA initialization errors get an actionable message."""
    with patch("g.jax_setup.jax.devices", side_effect=RuntimeError("Unknown backend cuda")):
        try:
            require_gpu_device()
        except RuntimeError as error:
            assert "no CUDA-enabled JAX backend" in str(error)
        else:
            raise AssertionError("Expected GPU validation to fail when CUDA initialization fails.")


def test_float_dtype_is_float32() -> None:
    """Ensure the codebase-wide JAX float dtype is fixed to float32."""
    assert jnp.float32 == FLOAT_DTYPE
