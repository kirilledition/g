"""Native BGEN engine construction."""

from __future__ import annotations

import logging
import typing

from g import _core, types
from g.engine import trusted_validation

if typing.TYPE_CHECKING:
    from g.io import source

logger = logging.getLogger(__name__)


def open_bgen_run_engine(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    chunk_size: int,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool,
) -> _core.Regenie2RunEngine:
    """Open the native BGEN run engine without running trusted validation."""
    logger.debug("Constructing native BGEN run engine.")
    return _core.Regenie2RunEngine(
        str(genotype_source_config.source_path),
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )


def validate_trusted_bgen_run_engine(
    *,
    engine: _core.Regenie2RunEngine,
    genotype_source_config: source.GenotypeSourceConfig,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    trusted_bgen_validator: typing.Callable[..., None] | None,
) -> None:
    """Validate trusted no-missing diploid BGEN mode for an open engine."""
    logger.debug("Validating trusted no-missing diploid BGEN mode.")
    resolved_trusted_bgen_validator = trusted_bgen_validator or trusted_validation.validate_trusted_bgen_with_cache
    resolved_trusted_bgen_validator(
        engine=engine,
        bgen_path=genotype_source_config.source_path,
        validation_mode=trusted_bgen_validation_mode,
    )


def build_bgen_run_engine(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    chunk_size: int,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    trusted_bgen_validator: typing.Callable[..., None] | None,
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
            trusted_bgen_validator=trusted_bgen_validator,
        )
    return engine
