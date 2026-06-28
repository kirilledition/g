from __future__ import annotations

import dataclasses
import importlib
import sys
import typing
from pathlib import Path
from unittest.mock import patch

from g import runtime_paths, types
from g.interface import config
from g.jax_runtime import models, resolution, setup

if typing.TYPE_CHECKING:
    import pytest


def build_runtime_policy(**overrides: object) -> models.JaxRuntimePolicy:
    """Build explicit JAX runtime policy for tests."""
    policy = models.JaxRuntimePolicy(
        device=types.Device.CPU,
        cache_directory=None,
        matmul_precision=None,
        persistent_cache=True,
        persistent_cache_min_entry_size_bytes=0,
        persistent_cache_min_compile_time_seconds=0,
        xla_autotune_cache=False,
        transfer_guard=False,
    )
    return dataclasses.replace(policy, **overrides)


def test_resolve_jax_runtime_policy_uses_native_payload() -> None:
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
            "device": "gpu",
            "jax_cache_dir": "~/custom/g/cache",
            "jax_matmul_precision": "highest",
            "jax_persistent_cache": False,
            "jax_persistent_cache_min_entry_size_bytes": 1024,
            "jax_persistent_cache_min_compile_time_seconds": 5,
            "jax_xla_autotune_cache": True,
            "jax_transfer_guard": True,
        }
    )

    policy = resolution.resolve_jax_runtime_policy(regenie_config.g_compute)

    assert policy == models.JaxRuntimePolicy(
        device=types.Device.GPU,
        cache_directory=Path("~/custom/g/cache").expanduser(),
        matmul_precision=types.JaxMatmulPrecision.HIGHEST,
        persistent_cache=False,
        persistent_cache_min_entry_size_bytes=1024,
        persistent_cache_min_compile_time_seconds=5,
        xla_autotune_cache=True,
        transfer_guard=True,
    )


def test_resolve_jax_cache_uses_explicit_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure explicit config paths are used instead of environment variables."""
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/ignored/jax/cache")
    monkeypatch.setenv("G_JAX_CACHE_DIR", "/ignored/g/cache")

    report = resolution.resolve_jax_runtime_setup(build_runtime_policy(cache_directory=Path("~/custom/g/cache")))

    assert report.cache_directory == Path("~/custom/g/cache").expanduser()


def test_resolve_jax_cache_uses_fallback() -> None:
    """Ensure fallback uses a local temporary cache path when no config path is set."""
    report = resolution.resolve_jax_runtime_setup(build_runtime_policy())

    assert report.cache_directory.parent.parent == runtime_paths.DEFAULT_LOCAL_TEMPORARY_ROOT
    assert report.cache_directory.name == "g-jax-cache"


def test_resolve_jax_runtime_setup_defaults_xla_auxiliary_cache_to_disabled() -> None:
    """Ensure XLA auxiliary caches are opt-in."""
    report = resolution.resolve_jax_runtime_setup(
        build_runtime_policy(cache_directory=runtime_paths.DEFAULT_LOCAL_TEMPORARY_ROOT / "mockuser" / "g-jax-cache")
    )

    assert report.xla_auxiliary_cache_mode == models.XlaAuxiliaryCacheMode.DISABLED
    assert report.xla_auxiliary_cache_reason == "XLA auxiliary cache was not requested"


def test_resolve_jax_runtime_setup_disables_xla_auxiliary_cache_without_persistent_cache() -> None:
    """Ensure XLA auxiliary caches require the persistent compilation cache."""
    report = resolution.resolve_jax_runtime_setup(
        build_runtime_policy(
            cache_directory=runtime_paths.DEFAULT_LOCAL_TEMPORARY_ROOT / "mockuser" / "g-jax-cache",
            persistent_cache=False,
            xla_autotune_cache=True,
        )
    )

    assert report.xla_auxiliary_cache_mode == models.XlaAuxiliaryCacheMode.DISABLED
    assert report.xla_auxiliary_cache_reason == "persistent compilation cache is disabled"


def test_resolve_jax_runtime_setup_enables_requested_xla_auxiliary_cache() -> None:
    """Ensure XLA auxiliary caches are enabled when explicitly requested."""
    report = resolution.resolve_jax_runtime_setup(
        build_runtime_policy(
            cache_directory=Path("shared-cache"),
            xla_autotune_cache=True,
        )
    )

    assert report.xla_auxiliary_cache_mode == models.XlaAuxiliaryCacheMode.PER_FUSION_AUTOTUNE
    assert report.xla_auxiliary_cache_reason == "XLA auxiliary cache was requested"


def test_resolve_jax_runtime_setup_returns_report() -> None:
    """Ensure pure setup resolution returns the expected report."""
    policy = dataclasses.replace(
        build_runtime_policy(),
        cache_directory=Path("shared-cache"),
        matmul_precision=None,
        xla_autotune_cache=True,
        transfer_guard=True,
    )

    report = resolution.resolve_jax_runtime_setup(policy)

    assert report.requested_device == types.Device.CPU
    assert report.platform_name == models.JAX_CPU_PLATFORM_NAME
    assert report.cache_directory == Path("shared-cache")
    assert report.matmul_precision == types.JaxMatmulPrecision.FLOAT32
    assert report.persistent_cache_enabled is True
    assert report.xla_auxiliary_cache_mode == models.XlaAuxiliaryCacheMode.PER_FUSION_AUTOTUNE
    assert report.transfer_guard_enabled is True
    assert report.gpu_validation_status == models.GpuValidationStatus.SKIPPED


def test_resolve_jax_runtime_setup_maps_gpu_device_to_cuda_platform_name() -> None:
    """Ensure GPU policy resolves to the JAX CUDA platform string."""
    report = resolution.resolve_jax_runtime_setup(build_runtime_policy(device=types.Device.GPU))

    assert report.requested_device == types.Device.GPU
    assert report.platform_name == models.JAX_CUDA_PLATFORM_NAME


def test_configure_before_backend_init_sets_platform_first(tmp_path: Path) -> None:
    """Ensure platform selection happens before other JAX runtime settings."""
    cache_directory = tmp_path / "jax-cache"
    policy = dataclasses.replace(build_runtime_policy(), cache_directory=cache_directory)

    with patch("g.jax_runtime.setup.jax.config.update") as mock_update:
        report = setup.configure_before_backend_init(policy, diagnostic_sink=None)

    assert report.cache_directory == cache_directory
    assert [call.args for call in mock_update.call_args_list] == [
        ("jax_platforms", "cpu"),
        ("jax_enable_x64", True),
        ("jax_default_matmul_precision", "float32"),
        ("jax_compilation_cache_dir", str(cache_directory)),
        ("jax_persistent_cache_min_entry_size_bytes", 0),
        ("jax_persistent_cache_min_compile_time_secs", 0),
        ("jax_persistent_cache_enable_xla_caches", "none"),
    ]


def test_configure_before_backend_init_validates_gpu_after_runtime(tmp_path: Path) -> None:
    """Ensure GPU validation happens after platform and cache settings are applied."""
    cache_directory = tmp_path / "jax-cache"
    policy = dataclasses.replace(build_runtime_policy(device=types.Device.GPU), cache_directory=cache_directory)
    call_order: list[str] = []

    def record_config_update(setting_name: str, value: object) -> None:
        del value
        call_order.append(setting_name)

    def record_validate_gpu_device() -> models.JaxGpuValidationReport:
        call_order.append("validate_gpu_device")
        return models.JaxGpuValidationReport(
            status=models.GpuValidationStatus.SUCCEEDED,
            message="JAX reported at least one GPU device.",
        )

    with (
        patch("g.jax_runtime.setup.jax.config.update", side_effect=record_config_update),
        patch("g.jax_runtime.setup.validate_gpu_device", side_effect=record_validate_gpu_device),
    ):
        report = setup.configure_before_backend_init(policy, diagnostic_sink=None)

    assert report.gpu_validation_status == models.GpuValidationStatus.SUCCEEDED
    assert call_order[0] == "jax_platforms"
    assert "jax_compilation_cache_dir" in call_order
    assert call_order[-1] == "validate_gpu_device"


def test_configure_before_backend_init_emits_structured_diagnostics(tmp_path: Path) -> None:
    """Ensure setup choices are emitted as structured diagnostic events."""
    cache_directory = tmp_path / "jax-cache"
    policy = dataclasses.replace(
        build_runtime_policy(),
        cache_directory=cache_directory,
        xla_autotune_cache=False,
        transfer_guard=True,
    )
    diagnostic_events: list[models.JaxRuntimeDiagnosticEvent] = []

    with patch("g.jax_runtime.setup.jax.config.update"):
        setup.configure_before_backend_init(policy, diagnostic_sink=diagnostic_events.append)

    event_names = [diagnostic_event.event_name for diagnostic_event in diagnostic_events]
    assert event_names == [
        "jax_platform_selected",
        "jax_persistent_cache_configured",
        "jax_xla_auxiliary_cache_configured",
        "jax_transfer_guard_configured",
        "jax_gpu_validation",
    ]
    event_fields = [
        {diagnostic_field.name: diagnostic_field.value for diagnostic_field in diagnostic_event.fields}
        for diagnostic_event in diagnostic_events
    ]
    assert event_fields[0]["platform"] == "cpu"
    assert event_fields[1]["cache_directory"] == str(cache_directory)
    assert event_fields[2]["enabled"] is False
    assert event_fields[3]["enabled"] is True
    assert event_fields[4]["status"] == "skipped"


def test_configure_before_backend_init_emits_gpu_validation_failure_before_raise(tmp_path: Path) -> None:
    """Ensure GPU validation failures are logged before the original error is re-raised."""
    cache_directory = tmp_path / "jax-cache"
    policy = dataclasses.replace(build_runtime_policy(device=types.Device.GPU), cache_directory=cache_directory)
    diagnostic_events: list[models.JaxRuntimeDiagnosticEvent] = []

    with (
        patch("g.jax_runtime.setup.jax.config.update"),
        patch("g.jax_runtime.setup.validate_gpu_device", side_effect=RuntimeError("no gpu")),
    ):
        try:
            setup.configure_before_backend_init(policy, diagnostic_sink=diagnostic_events.append)
        except RuntimeError as error:
            assert str(error) == "no gpu"
        else:
            raise AssertionError("Expected GPU validation failure.")

    failure_event = diagnostic_events[-1]
    assert failure_event.event_name == "jax_gpu_validation"
    assert failure_event.level == models.JaxRuntimeDiagnosticLevel.ERROR
    assert {diagnostic_field.name: diagnostic_field.value for diagnostic_field in failure_event.fields} == {
        "status": "failed",
        "message": "no gpu",
    }


def test_compute_import_does_not_configure_jax_backend() -> None:
    """Ensure compute modules leave JAX runtime policy to setup."""
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
        patch("g.jax_runtime.setup.nvidia_driver_is_visible", return_value=True),
        patch("g.jax_runtime.setup.jax.devices", return_value=[FakeDevice()]),
    ):
        setup.require_gpu_device()


def test_validate_gpu_device_returns_native_success_report() -> None:
    """Ensure GPU validation success details come from native runtime policy."""

    class FakeDevice:
        platform = "gpu"

        def __str__(self) -> str:
            return "GpuDevice(id=0)"

    with (
        patch("g.jax_runtime.setup.nvidia_driver_is_visible", return_value=True),
        patch("g.jax_runtime.setup.jax.devices", return_value=[FakeDevice()]),
    ):
        validation_report = setup.validate_gpu_device()

    assert validation_report == models.JaxGpuValidationReport(
        status=models.GpuValidationStatus.SUCCEEDED,
        message="JAX reported at least one GPU device.",
    )


def test_require_gpu_device_rejects_missing_nvidia_driver() -> None:
    """Ensure GPU validation fails before JAX when no NVIDIA device is visible."""
    with patch("g.jax_runtime.setup.nvidia_driver_is_visible", return_value=False):
        try:
            setup.require_gpu_device()
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
        patch("g.jax_runtime.setup.nvidia_driver_is_visible", return_value=True),
        patch("g.jax_runtime.setup.jax.devices", return_value=[FakeDevice()]),
    ):
        try:
            setup.require_gpu_device()
        except RuntimeError as error:
            assert "did not report any GPU devices" in str(error)
        else:
            raise AssertionError("Expected GPU validation to fail for CPU-only devices.")


def test_require_gpu_device_wraps_backend_initialization_errors() -> None:
    """Ensure CUDA initialization errors get an actionable message."""
    with (
        patch("g.jax_runtime.setup.nvidia_driver_is_visible", return_value=True),
        patch("g.jax_runtime.setup.jax.devices", side_effect=RuntimeError("Unknown backend cuda")),
    ):
        try:
            setup.require_gpu_device()
        except RuntimeError as error:
            assert "no CUDA-enabled JAX backend" in str(error)
        else:
            raise AssertionError("Expected GPU validation to fail when CUDA initialization fails.")


def test_require_gpu_device_wraps_jax_plugin_assertion_errors() -> None:
    """Ensure CUDA plugin assertion failures get an actionable message."""
    with (
        patch("g.jax_runtime.setup.nvidia_driver_is_visible", return_value=True),
        patch("g.jax_runtime.setup.jax.devices", side_effect=AssertionError("plugin initialization failed")),
    ):
        try:
            setup.require_gpu_device()
        except RuntimeError as error:
            assert "JAX CUDA plugin failed" in str(error)
        else:
            raise AssertionError("Expected GPU validation to fail when JAX raises AssertionError.")
