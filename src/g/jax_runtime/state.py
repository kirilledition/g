"""Pure JAX runtime policy formatting helpers."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from g.jax_runtime import models


def describe_jax_runtime_policy(policy: models.JaxRuntimePolicy) -> str:
    """Format a JAX runtime policy for diagnostics.

    Args:
        policy: Runtime policy to format.

    Returns:
        Stable human-readable policy description.

    """
    cache_directory = "<default>" if policy.cache_directory is None else str(policy.cache_directory)
    matmul_precision = "<default>" if policy.matmul_precision is None else policy.matmul_precision.value
    return (
        f"device={policy.device.value}, "
        f"jax-cache-dir={cache_directory}, "
        f"jax-matmul-precision={matmul_precision}, "
        f"jax-persistent-cache={policy.persistent_cache}, "
        f"jax-persistent-cache-min-entry-size-bytes={policy.persistent_cache_min_entry_size_bytes}, "
        f"jax-persistent-cache-min-compile-time-seconds={policy.persistent_cache_min_compile_time_seconds}, "
        f"jax-xla-autotune-cache={policy.xla_autotune_cache}, "
        f"jax-transfer-guard={policy.transfer_guard}"
    )
