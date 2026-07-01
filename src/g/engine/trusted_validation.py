"""Trusted BGEN validation cache helpers."""

from __future__ import annotations

from pathlib import Path

from g import _core, types


def trusted_bgen_validation_cache_directory() -> Path:
    """Return the trusted BGEN validation cache directory."""
    return Path(_core.default_trusted_bgen_validation_cache_directory_value())


def validate_trusted_bgen_with_cache(
    *,
    engine: _core.Regenie2RunEngine,
    bgen_path: Path,
    validation_mode: types.TrustedBgenValidationMode,
) -> None:
    """Validate or trust the no-missing diploid BGEN path according to mode."""
    engine.validate_trusted_no_missing_diploid_with_cache(
        str(bgen_path),
        validation_mode.value,
        str(trusted_bgen_validation_cache_directory()),
    )
