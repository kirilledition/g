from __future__ import annotations

import types
import typing

import pytest

from g.compute import cuda_ffi
from tooling.cli import benchmark_firth_compute

if typing.TYPE_CHECKING:
    from pathlib import Path


def build_arguments(
    temporary_path: Path,
    *,
    device: benchmark_firth_compute.BenchmarkDevice,
    implementation: benchmark_firth_compute.BenchmarkImplementation,
) -> benchmark_firth_compute.BenchmarkArguments:
    return benchmark_firth_compute.BenchmarkArguments(
        device=device,
        implementation=implementation,
        sample_count=16,
        candidate_capacity=8,
        firth_batch_size=4,
        active_candidate_counts=(4, 8),
        warmup_trial_count=1,
        measured_trial_count=1,
        trace_active_candidate_count=None,
        output_directory=temporary_path / "output",
        jax_cache_directory=temporary_path / "cache",
    )


def test_default_config_selects_portable_jax() -> None:
    arguments = benchmark_firth_compute.build_arguments_from_overrides()

    assert arguments.implementation is benchmark_firth_compute.BenchmarkImplementation.JAX
    assert benchmark_firth_compute.register_firth_components_implementation(arguments) is None


def test_raw_cuda_mode_rejects_cpu_before_lowering(tmp_path: Path) -> None:
    arguments = build_arguments(
        tmp_path,
        device=benchmark_firth_compute.BenchmarkDevice.CPU,
        implementation=benchmark_firth_compute.BenchmarkImplementation.RAW_CUDA,
    )

    with pytest.raises(ValueError, match="raw_cuda Firth components require device=gpu"):
        benchmark_firth_compute.validate_arguments(arguments)


def test_raw_cuda_mode_requires_feature_gated_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = build_arguments(
        tmp_path,
        device=benchmark_firth_compute.BenchmarkDevice.GPU,
        implementation=benchmark_firth_compute.BenchmarkImplementation.RAW_CUDA,
    )

    def reject_import(module_name: str) -> types.ModuleType:
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(benchmark_firth_compute.importlib, "import_module", reject_import)

    with pytest.raises(RuntimeError, match="private-test-support"):
        benchmark_firth_compute.register_firth_components_implementation(arguments)


def test_raw_cuda_mode_requires_exact_registered_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arguments = build_arguments(
        tmp_path,
        device=benchmark_firth_compute.BenchmarkDevice.GPU,
        implementation=benchmark_firth_compute.BenchmarkImplementation.RAW_CUDA,
    )
    test_support = types.SimpleNamespace(register_firth_components_ffi=lambda: "wrong.target")
    monkeypatch.setattr(benchmark_firth_compute.importlib, "import_module", lambda module_name: test_support)

    with pytest.raises(RuntimeError, match="does not match"):
        benchmark_firth_compute.register_firth_components_implementation(arguments)

    test_support.register_firth_components_ffi = lambda: cuda_ffi.FIRTH_COMPONENTS_FFI_TARGET
    assert (
        benchmark_firth_compute.register_firth_components_implementation(arguments)
        == cuda_ffi.FIRTH_COMPONENTS_FFI_TARGET
    )


def test_summary_schema_version_is_integer_zero() -> None:
    payload = {"schema_version": benchmark_firth_compute.SUMMARY_SCHEMA_VERSION}

    assert payload["schema_version"] == 0
    benchmark_firth_compute._validate_summary_schema_version(payload)


def test_summary_schema_version_rejects_non_integer_payload() -> None:
    with pytest.raises(ValueError, match="integer 0"):
        benchmark_firth_compute._validate_summary_schema_version({"schema_version": True})


def test_summary_schema_version_rejects_unexpected_version() -> None:
    with pytest.raises(ValueError, match="Expected CUDA qualification schema_version"):
        benchmark_firth_compute._validate_summary_schema_version({"schema_version": 1})
