from __future__ import annotations

import importlib
import sys
import typing
from pathlib import Path
from unittest.mock import patch

if typing.TYPE_CHECKING:
    import pytest

import jax.numpy as jnp

from g.jax_setup import (
    FLOAT_DTYPE,
    configure_jax_device,
    configure_jax_runtime_before_backend_init,
    require_gpu_device,
    resolve_jax_compilation_cache_directory,
    resolve_xla_cache_option,
)
from g.types import Device


def test_resolve_jax_cache_uses_explicit_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure explicit config paths are used instead of environment variables."""
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/ignored/jax/cache")
    monkeypatch.setenv("G_JAX_CACHE_DIR", "/ignored/g/cache")

    result = resolve_jax_compilation_cache_directory(Path("~/custom/g/cache"))

    assert result == Path("~/custom/g/cache").expanduser()


def test_resolve_jax_cache_uses_fallback() -> None:
    """Ensure fallback uses a node-local cache path when no config path is set."""
    result = resolve_jax_compilation_cache_directory()

    assert result.parent.parent == Path("/tmp")
    assert result.name == "g-jax-cache"


def test_resolve_xla_cache_option_defaults_to_disabled() -> None:
    """Ensure XLA auxiliary caches are opt-in even on node-local paths."""
    result = resolve_xla_cache_option(Path("/tmp/mockuser/g-jax-cache"))

    assert result == "none"


def test_resolve_xla_cache_option_enables_node_local_opt_in() -> None:
    """Ensure XLA auxiliary caches can be enabled on node-local paths."""
    result = resolve_xla_cache_option(Path("/tmp/mockuser/g-jax-cache"), enable_xla_autotune_cache=True)

    assert result == "xla_gpu_per_fusion_autotune_cache_dir"


def test_resolve_xla_cache_option_rejects_beegfs_opt_in() -> None:
    """Ensure XLA auxiliary caches stay disabled on BeeGFS paths."""
    result = resolve_xla_cache_option(Path("/mnt/beegfs/kirill/cache"), enable_xla_autotune_cache=True)

    assert result == "none"


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


def test_configure_jax_runtime_before_backend_init_sets_platform_first(tmp_path: Path) -> None:
    """Ensure platform selection happens before other JAX runtime settings."""
    cache_directory = tmp_path / "jax-cache"

    with patch("g.jax_setup.jax.config.update") as mock_update:
        configure_jax_runtime_before_backend_init(device=Device.CPU, cache_directory=cache_directory)

    assert mock_update.call_args_list[0].args == ("jax_platforms", "cpu")
    assert ("jax_enable_x64", True) in [call.args for call in mock_update.call_args_list]
    assert ("jax_compilation_cache_dir", str(cache_directory)) in [call.args for call in mock_update.call_args_list]


def test_configure_jax_runtime_before_backend_init_validates_gpu_after_runtime(tmp_path: Path) -> None:
    """Ensure GPU validation happens after platform and cache settings are applied."""
    cache_directory = tmp_path / "jax-cache"
    call_order: list[str] = []

    def record_config_update(setting_name: str, value: object) -> None:
        del value
        call_order.append(setting_name)

    def record_require_gpu_device() -> None:
        call_order.append("require_gpu_device")

    with (
        patch("g.jax_setup.jax.config.update", side_effect=record_config_update),
        patch("g.jax_setup.require_gpu_device", side_effect=record_require_gpu_device),
    ):
        configure_jax_runtime_before_backend_init(device=Device.GPU, cache_directory=cache_directory)

    assert call_order[0] == "jax_platforms"
    assert "jax_compilation_cache_dir" in call_order
    assert call_order[-1] == "require_gpu_device"


def test_compute_import_does_not_configure_jax_runtime() -> None:
    """Ensure compute modules leave JAX runtime policy to jax_setup."""
    module_name = "g.compute.regenie2_binary"
    previous_module = sys.modules.pop(module_name, None)
    try:
        with patch("jax.config.update") as mock_update:
            importlib.import_module(module_name)
        mock_update.assert_not_called()
    finally:
        if previous_module is not None:
            sys.modules[module_name] = previous_module


def test_require_gpu_device_accepts_gpu_platform() -> None:
    """Ensure GPU validation accepts a JAX GPU device."""

    class FakeDevice:
        platform = "gpu"

    with (
        patch("g.jax_setup.nvidia_driver_is_visible", return_value=True),
        patch("g.jax_setup.jax.devices", return_value=[FakeDevice()]),
    ):
        require_gpu_device()


def test_require_gpu_device_rejects_missing_nvidia_driver() -> None:
    """Ensure GPU validation fails before JAX when no NVIDIA device is visible."""
    with patch("g.jax_setup.nvidia_driver_is_visible", return_value=False):
        try:
            require_gpu_device()
        except RuntimeError as error:
            assert "cannot see the NVIDIA driver or device files" in str(error)
        else:
            raise AssertionError("Expected GPU validation to fail without visible NVIDIA devices.")


def test_require_gpu_device_rejects_cpu_only_backend() -> None:
    """Ensure GPU validation rejects CPU-only device lists."""

    class FakeDevice:
        platform = "cpu"

        def __str__(self) -> str:
            return "CpuDevice(id=0)"

    with (
        patch("g.jax_setup.nvidia_driver_is_visible", return_value=True),
        patch("g.jax_setup.jax.devices", return_value=[FakeDevice()]),
    ):
        try:
            require_gpu_device()
        except RuntimeError as error:
            assert "did not report any GPU devices" in str(error)
        else:
            raise AssertionError("Expected GPU validation to fail for CPU-only devices.")


def test_require_gpu_device_wraps_backend_initialization_errors() -> None:
    """Ensure CUDA initialization errors get an actionable message."""
    with (
        patch("g.jax_setup.nvidia_driver_is_visible", return_value=True),
        patch("g.jax_setup.jax.devices", side_effect=RuntimeError("Unknown backend cuda")),
    ):
        try:
            require_gpu_device()
        except RuntimeError as error:
            assert "no CUDA-enabled JAX backend" in str(error)
        else:
            raise AssertionError("Expected GPU validation to fail when CUDA initialization fails.")


def test_require_gpu_device_wraps_jax_plugin_assertion_errors() -> None:
    """Ensure CUDA plugin assertion failures get an actionable message."""
    with (
        patch("g.jax_setup.nvidia_driver_is_visible", return_value=True),
        patch("g.jax_setup.jax.devices", side_effect=AssertionError("plugin initialization failed")),
    ):
        try:
            require_gpu_device()
        except RuntimeError as error:
            assert "JAX CUDA plugin failed" in str(error)
        else:
            raise AssertionError("Expected GPU validation to fail when JAX raises AssertionError.")


def test_float_dtype_is_float32() -> None:
    """Ensure the codebase-wide JAX float dtype is fixed to float32."""
    assert jnp.float32 == FLOAT_DTYPE
