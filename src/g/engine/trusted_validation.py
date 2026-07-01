"""Trusted BGEN validation cache helpers."""

from __future__ import annotations

from pathlib import Path

from g import _core, types


def assume_trusted_no_missing_diploid_validated() -> bool:
    """Return whether trusted BGEN validation should be treated as already completed."""
    return False


def trusted_bgen_validation_cache_directory() -> Path:
    """Return the trusted BGEN validation cache directory."""
    return Path.home() / ".cache" / "g" / "bgen_validation"


def build_trusted_bgen_validation_fingerprint(
    *,
    bgen_path: Path,
    sample_count: int,
    variant_count: int,
    trusted_no_missing_diploid: bool,
) -> str:
    """Build a stable trusted BGEN validation fingerprint."""
    return _core.build_trusted_bgen_validation_fingerprint_value(
        str(bgen_path),
        sample_count,
        variant_count,
        trusted_no_missing_diploid,
    )


def trusted_bgen_validation_cache_path(fingerprint: str) -> Path:
    """Return the validation cache path for a fingerprint."""
    return Path(
        _core.build_trusted_bgen_validation_cache_path_value(
            str(trusted_bgen_validation_cache_directory()),
            fingerprint,
        )
    )


def validate_trusted_bgen_with_cache(
    *,
    engine: _core.Regenie2RunEngine,
    bgen_path: Path,
    validation_mode: types.TrustedBgenValidationMode,
) -> None:
    """Validate or trust the no-missing diploid BGEN path according to mode."""
    if assume_trusted_no_missing_diploid_validated():
        message = "Trusted no-missing diploid validation cannot be globally assumed for calculation runs."
        raise ValueError(message)
    if validation_mode == types.TrustedBgenValidationMode.ASSUME_VALIDATED:
        message = (
            "Trusted no-missing diploid validation mode 'assume_validated' is unsafe for calculation runs. "
            "Use 'cache_on_miss' or 'force_validate' so BGEN compatibility is checked before decoding."
        )
        raise ValueError(message)
    fingerprint = build_trusted_bgen_validation_fingerprint(
        bgen_path=bgen_path,
        sample_count=int(engine.sample_count),
        variant_count=int(engine.variant_count),
        trusted_no_missing_diploid=True,
    )
    cache_path = trusted_bgen_validation_cache_path(fingerprint)
    if validation_mode == types.TrustedBgenValidationMode.CACHE_ON_MISS and cache_path.exists():
        engine.mark_trusted_no_missing_diploid_validated()
        return
    engine.validate_trusted_no_missing_diploid()
    _core.write_trusted_bgen_validation_cache_payload(
        str(cache_path),
        fingerprint,
        str(bgen_path),
        int(engine.sample_count),
        int(engine.variant_count),
    )
