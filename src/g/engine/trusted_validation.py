"""Trusted BGEN validation cache helpers."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g import _core, types


def validate_trusted_bgen_with_cache(
    *,
    engine: _core.Regenie2RunEngine,
    bgen_path: Path,
    validation_mode: types.TrustedBgenValidationMode,
) -> None:
    """Validate or trust the no-missing diploid BGEN path according to mode."""
    engine.validate_trusted_no_missing_diploid_with_default_cache(
        str(bgen_path),
        validation_mode.value,
    )
