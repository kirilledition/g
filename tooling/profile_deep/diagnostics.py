"""Deep-profile diagnostics helpers."""

from __future__ import annotations

import tooling.cli.profile_regenie2_deep as profile_regenie2_deep


def read_jax_compile_log_summary(
    stderr_log_path: str,
) -> profile_regenie2_deep.JaxCompileLogSummary:
    """Read JAX compile/cache diagnostics from a stderr log."""
    return profile_regenie2_deep.read_jax_compile_log_summary(stderr_log_path)
