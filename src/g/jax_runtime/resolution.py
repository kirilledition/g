"""Pure JAX runtime policy and setup resolution."""

from __future__ import annotations

import typing
from pathlib import Path

from g import _core, runtime_paths, types
from g.jax_runtime import models

if typing.TYPE_CHECKING:
    from g.interface import config

DEFAULT_CACHE_DIRECTORY_NAME = "g-jax-cache"


def resolve_jax_runtime_policy(compute_config: config.GComputeConfig) -> models.JaxRuntimePolicy:
    """Resolve the process-global JAX runtime policy requested by a run.

    Args:
        compute_config: Normalized compute configuration.

    Returns:
        Requested JAX runtime policy.

    """
    policy_payload = _core.build_jax_runtime_policy_payload(
        device=compute_config.device.value,
        cache_directory=(
            None if compute_config.jax_cache_dir is None else str(compute_config.jax_cache_dir.expanduser())
        ),
        matmul_precision=None
        if compute_config.jax_matmul_precision is None
        else compute_config.jax_matmul_precision.value,
        persistent_cache=compute_config.jax_persistent_cache,
        persistent_cache_min_entry_size_bytes=compute_config.jax_persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds=compute_config.jax_persistent_cache_min_compile_time_seconds,
        xla_autotune_cache=compute_config.jax_xla_autotune_cache,
        transfer_guard=compute_config.jax_transfer_guard,
    )
    cache_directory_payload = typing.cast("str | None", policy_payload["cache_directory"])
    matmul_precision_payload = typing.cast("str | None", policy_payload["matmul_precision"])
    return models.JaxRuntimePolicy(
        device=types.Device(typing.cast("str", policy_payload["device"])),
        cache_directory=None if cache_directory_payload is None else Path(cache_directory_payload),
        matmul_precision=(
            None if matmul_precision_payload is None else types.JaxMatmulPrecision(matmul_precision_payload)
        ),
        persistent_cache=typing.cast("bool", policy_payload["persistent_cache"]),
        persistent_cache_min_entry_size_bytes=typing.cast(
            "int",
            policy_payload["persistent_cache_min_entry_size_bytes"],
        ),
        persistent_cache_min_compile_time_seconds=typing.cast(
            "int",
            policy_payload["persistent_cache_min_compile_time_seconds"],
        ),
        xla_autotune_cache=typing.cast("bool", policy_payload["xla_autotune_cache"]),
        transfer_guard=typing.cast("bool", policy_payload["transfer_guard"]),
    )


def resolve_jax_runtime_setup(policy: models.JaxRuntimePolicy) -> models.JaxRuntimeSetupReport:
    """Resolve JAX setup decisions without mutating process-global state.

    Args:
        policy: Requested runtime policy.

    Returns:
        Setup report with pure resolution decisions.

    """
    if policy.cache_directory is None:
        resolved_cache_directory = runtime_paths.default_local_cache_directory(DEFAULT_CACHE_DIRECTORY_NAME)
    else:
        resolved_cache_directory = policy.cache_directory.expanduser()
    setup_payload = _core.resolve_jax_runtime_setup_payload(
        policy.device.value,
        str(resolved_cache_directory),
        None if policy.matmul_precision is None else policy.matmul_precision.value,
        policy.persistent_cache,
        policy.persistent_cache_min_entry_size_bytes,
        policy.persistent_cache_min_compile_time_seconds,
        policy.xla_autotune_cache,
        policy.transfer_guard,
    )
    return jax_runtime_setup_report_from_native_payload(setup_payload)


def jax_runtime_setup_report_from_native_payload(payload: object) -> models.JaxRuntimeSetupReport:
    """Adapt a native JAX runtime setup payload.

    Args:
        payload: Native setup payload mapping.

    Returns:
        JAX runtime setup report.

    """
    setup_payload = dict(typing.cast("typing.Mapping[str, object]", payload))
    return models.JaxRuntimeSetupReport(
        requested_device=types.Device(typing.cast("str", setup_payload["requested_device"])),
        platform_name=typing.cast("str", setup_payload["platform_name"]),
        cache_directory=Path(typing.cast("str", setup_payload["cache_directory"])),
        matmul_precision=types.JaxMatmulPrecision(typing.cast("str", setup_payload["matmul_precision"])),
        persistent_cache_enabled=typing.cast("bool", setup_payload["persistent_cache_enabled"]),
        persistent_cache_min_entry_size_bytes=typing.cast(
            "int",
            setup_payload["persistent_cache_min_entry_size_bytes"],
        ),
        persistent_cache_min_compile_time_seconds=typing.cast(
            "int",
            setup_payload["persistent_cache_min_compile_time_seconds"],
        ),
        xla_auxiliary_cache_mode=models.XlaAuxiliaryCacheMode(
            typing.cast("str", setup_payload["xla_auxiliary_cache_mode"])
        ),
        xla_auxiliary_cache_reason=typing.cast("str", setup_payload["xla_auxiliary_cache_reason"]),
        transfer_guard_enabled=typing.cast("bool", setup_payload["transfer_guard_enabled"]),
        gpu_validation_status=models.GpuValidationStatus(typing.cast("str", setup_payload["gpu_validation_status"])),
        gpu_validation_message=typing.cast("str | None", setup_payload["gpu_validation_message"]),
    )
