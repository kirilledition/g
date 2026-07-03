"""Preflight helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import typing

from g.engine import preflight as engine_preflight

type BgenPreflightEngineProtocol = engine_preflight.BgenPreflightEngineProtocol
type PreflightReport = engine_preflight.PreflightReport


def run_regenie2_preflight(
    *,
    run_input: typing.Any,
    prediction_source: typing.Any,
    engine: BgenPreflightEngineProtocol,
    variant_limit: int | None,
    is_binary_trait: bool,
    trusted_no_missing_diploid: bool,
) -> PreflightReport:
    """Validate aligned single-trait inputs before chunk execution."""
    return engine_preflight.run_regenie2_preflight(
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        variant_limit=variant_limit,
        is_binary_trait=is_binary_trait,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )


def run_regenie2_multi_preflight(
    *,
    run_input: typing.Any,
    prediction_source: typing.Any,
    engine: BgenPreflightEngineProtocol,
    variant_limit: int | None,
    is_binary_trait: bool,
    trusted_no_missing_diploid: bool,
) -> PreflightReport:
    """Validate aligned multi-trait inputs before chunk execution."""
    return engine_preflight.run_regenie2_multi_preflight(
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        variant_limit=variant_limit,
        is_binary_trait=is_binary_trait,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
