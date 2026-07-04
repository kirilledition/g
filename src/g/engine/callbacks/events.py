"""Callback-local diagnostic event and telemetry helpers."""

from __future__ import annotations

from g import _core

type TelemetrySession = object


def native_pipeline_diagnostic_policy() -> _core.NativePipelineDiagnosticPolicy:
    """Build the native pipeline diagnostic policy handle for callback diagnostics."""
    return _core.NativePipelineDiagnosticPolicy()


def record_callback_null_logistic_nonconvergence_warning(
    *,
    message: str,
    chromosome: str,
    nonconverged_count: int,
    phenotype_count: int,
    policy: str,
    scalar_convergence: bool,
    total_fit_count: int,
) -> None:
    """Record a null-logistic nonconvergence warning emitted by callbacks."""
    native_pipeline_diagnostic_policy().record_callback_null_logistic_nonconvergence_warning_diagnostic_event(
        message=message,
        chromosome=chromosome,
        nonconverged_count=nonconverged_count,
        phenotype_count=phenotype_count,
        policy=policy,
        scalar_convergence=scalar_convergence,
        total_fit_count=total_fit_count,
    )
