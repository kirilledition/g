from __future__ import annotations

import dataclasses
import importlib
import sys
import typing
from pathlib import Path
from unittest.mock import patch

if typing.TYPE_CHECKING:
    import pytest

from g import jax_runtime, types
from g.interface import config
from g.jax_setup import (
    build_jax_config_update_operations,
    configure_jax_runtime_before_backend_init,
    require_gpu_device,
    resolve_jax_compilation_cache_directory,
    resolve_jax_runtime_setup,
    resolve_xla_auxiliary_cache,
)


def build_runtime_policy(**overrides: object) -> jax_runtime.JaxRuntimePolicy:
    """Build explicit JAX runtime policy for tests."""
    compute_config = dataclasses.replace(config.load_packaged_config().g_compute, **overrides)
    return jax_runtime.build_jax_runtime_policy(compute_config)


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


def test_resolve_xla_auxiliary_cache_defaults_to_disabled() -> None:
    """Ensure XLA auxiliary caches are opt-in even on node-local paths."""
    result = resolve_xla_auxiliary_cache(
        Path("/tmp/mockuser/g-jax-cache"),
        persistent_cache=True,
        enable_xla_autotune_cache=False,
    )

    assert result.mode == jax_runtime.XlaAuxiliaryCacheMode.DISABLED
    assert result.enabled is False
    assert result.reason == "XLA auxiliary cache was not requested"


def test_resolve_xla_auxiliary_cache_disables_without_persistent_cache() -> None:
    """Ensure XLA auxiliary caches require the persistent compilation cache."""
    result = resolve_xla_auxiliary_cache(
        Path("/tmp/mockuser/g-jax-cache"),
        persistent_cache=False,
        enable_xla_autotune_cache=True,
    )

    assert result.mode == jax_runtime.XlaAuxiliaryCacheMode.DISABLED
    assert result.reason == "persistent compilation cache is disabled"


def test_resolve_xla_auxiliary_cache_enables_node_local_opt_in() -> None:
    """Ensure XLA auxiliary caches can be enabled on node-local paths."""
    result = resolve_xla_auxiliary_cache(
        Path("/tmp/mockuser/g-jax-cache"),
        persistent_cache=True,
        enable_xla_autotune_cache=True,
    )

    assert result.mode == jax_runtime.XlaAuxiliaryCacheMode.PER_FUSION_AUTOTUNE
    assert result.enabled is True
    assert result.reason == "cache directory is node-local"


def test_resolve_xla_auxiliary_cache_rejects_beegfs_opt_in() -> None:
    """Ensure XLA auxiliary caches stay disabled on BeeGFS paths."""
    result = resolve_xla_auxiliary_cache(
        Path("/mnt/beegfs/kirill/cache"),
        persistent_cache=True,
        enable_xla_autotune_cache=True,
    )

    assert result.mode == jax_runtime.XlaAuxiliaryCacheMode.DISABLED
    assert result.reason == "cache directory is on BeeGFS"


def test_resolve_jax_runtime_setup_returns_report() -> None:
    """Ensure pure setup resolution returns the expected report."""
    policy = dataclasses.replace(
        build_runtime_policy(),
        cache_directory=Path("/mnt/beegfs/kirill/cache"),
        matmul_precision=None,
        xla_autotune_cache=True,
        transfer_guard=True,
    )

    report = resolve_jax_runtime_setup(policy)

    assert report.requested_device == types.Device.CPU
    assert report.platform == jax_runtime.JaxPlatform.CPU
    assert report.cache_directory == Path("/mnt/beegfs/kirill/cache")
    assert report.matmul_precision == types.JaxMatmulPrecision.FLOAT32
    assert report.persistent_cache_enabled is True
    assert report.xla_auxiliary_cache.mode == jax_runtime.XlaAuxiliaryCacheMode.DISABLED
    assert report.transfer_guard_enabled is True
    assert report.gpu_validation.status == jax_runtime.GpuValidationStatus.SKIPPED


def test_build_jax_config_update_operations_sets_platform_first(tmp_path: Path) -> None:
    """Ensure platform selection is the first JAX config mutation."""
    cache_directory = tmp_path / "jax-cache"
    report = resolve_jax_runtime_setup(dataclasses.replace(build_runtime_policy(), cache_directory=cache_directory))

    operations = build_jax_config_update_operations(report)

    assert operations[0] == jax_runtime.JaxConfigUpdateOperation("jax_platforms", "cpu")
    assert jax_runtime.JaxConfigUpdateOperation("jax_enable_x64", value=True) in operations
    assert jax_runtime.JaxConfigUpdateOperation("jax_compilation_cache_dir", str(cache_directory)) in operations


def test_configure_jax_runtime_before_backend_init_sets_platform_first(tmp_path: Path) -> None:
    """Ensure platform selection happens before other JAX runtime settings."""
    cache_directory = tmp_path / "jax-cache"
    policy = dataclasses.replace(build_runtime_policy(), cache_directory=cache_directory)

    with patch("g.jax_setup.jax.config.update") as mock_update:
        report = configure_jax_runtime_before_backend_init(policy)

    assert report.cache_directory == cache_directory
    assert mock_update.call_args_list[0].args == ("jax_platforms", "cpu")
    assert ("jax_enable_x64", True) in [call.args for call in mock_update.call_args_list]
    assert ("jax_compilation_cache_dir", str(cache_directory)) in [call.args for call in mock_update.call_args_list]


def test_configure_jax_runtime_before_backend_init_validates_gpu_after_runtime(tmp_path: Path) -> None:
    """Ensure GPU validation happens after platform and cache settings are applied."""
    cache_directory = tmp_path / "jax-cache"
    policy = dataclasses.replace(build_runtime_policy(device=types.Device.GPU), cache_directory=cache_directory)
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
        report = configure_jax_runtime_before_backend_init(policy)

    assert report.gpu_validation.status == jax_runtime.GpuValidationStatus.SUCCEEDED
    assert call_order[0] == "jax_platforms"
    assert "jax_compilation_cache_dir" in call_order
    assert call_order[-1] == "require_gpu_device"


def test_configure_jax_runtime_emits_structured_diagnostics(tmp_path: Path) -> None:
    """Ensure setup choices are emitted as structured diagnostic events."""
    cache_directory = tmp_path / "jax-cache"
    policy = dataclasses.replace(
        build_runtime_policy(),
        cache_directory=cache_directory,
        xla_autotune_cache=False,
        transfer_guard=True,
    )
    diagnostic_events: list[jax_runtime.JaxRuntimeDiagnosticEvent] = []

    with patch("g.jax_setup.jax.config.update"):
        configure_jax_runtime_before_backend_init(policy, diagnostic_sink=diagnostic_events.append)

    event_names = [diagnostic_event.event_name for diagnostic_event in diagnostic_events]
    assert event_names == [
        "jax_platform_selected",
        "jax_persistent_cache_configured",
        "jax_xla_auxiliary_cache_configured",
        "jax_transfer_guard_configured",
        "jax_gpu_validation",
    ]
    event_fields = [jax_runtime.diagnostic_event_fields(diagnostic_event) for diagnostic_event in diagnostic_events]
    assert event_fields[0]["platform"] == "cpu"
    assert event_fields[1]["cache_directory"] == str(cache_directory)
    assert event_fields[2]["enabled"] is False
    assert event_fields[3]["enabled"] is True
    assert event_fields[4]["status"] == "skipped"


def test_configure_jax_runtime_emits_gpu_validation_failure_before_raise(tmp_path: Path) -> None:
    """Ensure GPU validation failures are logged before the original error is re-raised."""
    cache_directory = tmp_path / "jax-cache"
    policy = dataclasses.replace(build_runtime_policy(device=types.Device.GPU), cache_directory=cache_directory)
    diagnostic_events: list[jax_runtime.JaxRuntimeDiagnosticEvent] = []

    with (
        patch("g.jax_setup.jax.config.update"),
        patch("g.jax_setup.require_gpu_device", side_effect=RuntimeError("no gpu")),
    ):
        try:
            configure_jax_runtime_before_backend_init(policy, diagnostic_sink=diagnostic_events.append)
        except RuntimeError as error:
            assert str(error) == "no gpu"
        else:
            raise AssertionError("Expected GPU validation failure.")

    failure_event = diagnostic_events[-1]
    assert failure_event.event_name == "jax_gpu_validation"
    assert failure_event.level == jax_runtime.JaxRuntimeDiagnosticLevel.ERROR
    assert jax_runtime.diagnostic_event_fields(failure_event) == {
        "status": "failed",
        "message": "no gpu",
    }


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
