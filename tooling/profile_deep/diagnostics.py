"""Deep-profile diagnostics helpers."""

from __future__ import annotations

import typing

import tooling.cli.profile_regenie2_deep as profile_regenie2_deep

if typing.TYPE_CHECKING:
    from tooling.profile_deep import models as profile_deep_models


def read_jax_compile_log_summary(
    stderr_log_path: str,
) -> profile_deep_models.JaxCompileLogSummary:
    """Read JAX compile/cache diagnostics from a stderr log."""
    return profile_regenie2_deep.read_jax_compile_log_summary(stderr_log_path)
