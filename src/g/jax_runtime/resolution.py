"""Pure JAX runtime policy and setup resolution."""

from __future__ import annotations

import typing

from g import runtime_paths, types
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
    cache_directory = None
    if compute_config.jax_cache_dir is not None:
        cache_directory = compute_config.jax_cache_dir.expanduser()
    return models.JaxRuntimePolicy(
        device=compute_config.device,
        cache_directory=cache_directory,
        matmul_precision=compute_config.jax_matmul_precision,
        persistent_cache=compute_config.jax_persistent_cache,
        persistent_cache_min_entry_size_bytes=compute_config.jax_persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds=compute_config.jax_persistent_cache_min_compile_time_seconds,
        xla_autotune_cache=compute_config.jax_xla_autotune_cache,
        transfer_guard=compute_config.jax_transfer_guard,
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
    gpu_validation_status = models.GpuValidationStatus.SKIPPED
    gpu_validation_message = "CPU runtime requested; GPU validation skipped."
    if policy.device == types.Device.GPU:
        gpu_validation_status = models.GpuValidationStatus.PENDING
        gpu_validation_message = None
    matmul_precision = types.JaxMatmulPrecision.FLOAT32
    if policy.matmul_precision is not None:
        matmul_precision = policy.matmul_precision
    platform_name = models.JAX_CPU_PLATFORM_NAME
    if policy.device == types.Device.GPU:
        platform_name = models.JAX_CUDA_PLATFORM_NAME
    xla_auxiliary_cache_mode = models.XlaAuxiliaryCacheMode.DISABLED
    xla_auxiliary_cache_reason = "persistent compilation cache is disabled"
    if policy.persistent_cache and not policy.xla_autotune_cache:
        xla_auxiliary_cache_reason = "XLA auxiliary cache was not requested"
    elif policy.persistent_cache:
        xla_auxiliary_cache_mode = models.XlaAuxiliaryCacheMode.PER_FUSION_AUTOTUNE
        xla_auxiliary_cache_reason = "XLA auxiliary cache was requested"
    return models.JaxRuntimeSetupReport(
        requested_device=policy.device,
        platform_name=platform_name,
        cache_directory=resolved_cache_directory,
        matmul_precision=matmul_precision,
        persistent_cache_enabled=policy.persistent_cache,
        persistent_cache_min_entry_size_bytes=policy.persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds=policy.persistent_cache_min_compile_time_seconds,
        xla_auxiliary_cache_mode=xla_auxiliary_cache_mode,
        xla_auxiliary_cache_reason=xla_auxiliary_cache_reason,
        transfer_guard_enabled=policy.transfer_guard,
        gpu_validation_status=gpu_validation_status,
        gpu_validation_message=gpu_validation_message,
    )
