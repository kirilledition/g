"""Deep-profile diagnostics helpers."""

from __future__ import annotations

import typing

from tooling.profile_deep import jax_cache as profile_deep_jax_cache

if typing.TYPE_CHECKING:
    from tooling.profile_deep import models as profile_deep_models


def read_jax_compile_log_summary(
    stderr_log_path: str,
) -> profile_deep_models.JaxCompileLogSummary:
    """Read JAX compile/cache diagnostics from a stderr log."""
    return profile_deep_jax_cache.read_jax_compile_log_summary(stderr_log_path)
