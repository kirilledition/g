"""Process-global JAX runtime policy state and compatibility checks."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from g.jax_runtime import models

CONFIGURED_JAX_RUNTIME_POLICY: models.JaxRuntimePolicy | None = None


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


def require_compatible_jax_runtime_policy(requested_policy: models.JaxRuntimePolicy) -> None:
    """Raise when a requested policy conflicts with the configured policy.

    Args:
        requested_policy: Requested process-global JAX runtime policy.

    Raises:
        RuntimeError: If a previous run configured incompatible process-global JAX settings.

    """
    configured_policy = CONFIGURED_JAX_RUNTIME_POLICY
    if configured_policy is None or requested_policy == configured_policy:
        return
    message = (
        "JAX runtime is already configured for this Python process with "
        f"{describe_jax_runtime_policy(configured_policy)}. "
        "A later run requested incompatible settings: "
        f"{describe_jax_runtime_policy(requested_policy)}. "
        "JAX backend, platform, and compilation cache settings are process-global; start a fresh Python process "
        "for incompatible runtime settings."
    )
    raise RuntimeError(message)
