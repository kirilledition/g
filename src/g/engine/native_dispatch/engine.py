"""Native BGEN engine construction."""

from __future__ import annotations

import typing

from g import _core, types

if typing.TYPE_CHECKING:
    from g import execution_plan


def open_bgen_run_engine(
    *,
    genotype_source_config: execution_plan.GenotypeSourceConfig,
    chunk_size: int,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool,
) -> _core.Regenie2RunEngine:
    """Open the native BGEN run engine without running trusted validation."""
    _core.record_native_dispatch_bgen_engine_constructing_diagnostic_event(
        chunk_size=chunk_size,
        source_path=str(genotype_source_config.source_path),
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        variant_limit=variant_limit,
    )
    return _core.Regenie2RunEngine(
        str(genotype_source_config.source_path),
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )


def validate_trusted_bgen_run_engine(
    *,
    engine: _core.Regenie2RunEngine,
    genotype_source_config: execution_plan.GenotypeSourceConfig,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
) -> None:
    """Validate trusted no-missing diploid BGEN mode for an open engine."""
    _core.record_native_dispatch_trusted_bgen_validation_started_diagnostic_event(
        source_path=str(genotype_source_config.source_path),
        trusted_bgen_validation_mode=trusted_bgen_validation_mode.value,
    )
    engine.validate_trusted_no_missing_diploid_with_default_cache(
        str(genotype_source_config.source_path),
        trusted_bgen_validation_mode.value,
    )


def build_bgen_run_engine(
    *,
    genotype_source_config: execution_plan.GenotypeSourceConfig,
    chunk_size: int,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
) -> _core.Regenie2RunEngine:
    """Open the native BGEN run engine once for alignment and chunk delivery."""
    engine = open_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    if trusted_no_missing_diploid:
        validate_trusted_bgen_run_engine(
            engine=engine,
            genotype_source_config=genotype_source_config,
            trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        )
    return engine
