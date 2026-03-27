from __future__ import annotations

from unittest.mock import patch

from g.jax_setup import configure_jax_device


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
