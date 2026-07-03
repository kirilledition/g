"""BGEN engine helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import typing

from g.engine.native_dispatch import engine as native_dispatch_engine

if typing.TYPE_CHECKING:
    from g import _core, execution_plan, types


def open_bgen_run_engine(
    *,
    genotype_source_config: execution_plan.GenotypeSourceConfig,
    chunk_size: int,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool,
) -> _core.Regenie2RunEngine:
    """Open the native BGEN run engine without trusted validation."""
    return native_dispatch_engine.open_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )


def validate_trusted_bgen_run_engine(
    *,
    engine: _core.Regenie2RunEngine,
    genotype_source_config: execution_plan.GenotypeSourceConfig,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    trusted_bgen_validator: typing.Callable[..., None] | None,
) -> None:
    """Validate trusted no-missing diploid BGEN mode for an open engine."""
    native_dispatch_engine.validate_trusted_bgen_run_engine(
        engine=engine,
        genotype_source_config=genotype_source_config,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        trusted_bgen_validator=trusted_bgen_validator,
    )


def build_bgen_run_engine(
    *,
    genotype_source_config: execution_plan.GenotypeSourceConfig,
    chunk_size: int,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    trusted_bgen_validator: typing.Callable[..., None] | None,
) -> _core.Regenie2RunEngine:
    """Open the native BGEN run engine once for alignment and chunk delivery."""
    return native_dispatch_engine.build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        trusted_bgen_validator=trusted_bgen_validator,
    )
