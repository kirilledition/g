"""Trusted BGEN validation cache helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from g import _core, types

ASSUME_TRUSTED_NO_MISSING_DIPLOID_VALIDATED_ENVIRONMENT_VARIABLE = (
    "G_REGENIE2_ASSUME_TRUSTED_NO_MISSING_DIPLOID_VALIDATED"
)
TRUSTED_BGEN_VALIDATION_SCHEMA_VERSION = 1


def assume_trusted_no_missing_diploid_validated() -> bool:
    """Return whether trusted BGEN validation should be treated as already completed."""
    raw_value = os.environ.get(ASSUME_TRUSTED_NO_MISSING_DIPLOID_VALIDATED_ENVIRONMENT_VARIABLE)
    if raw_value is None:
        return False
    return raw_value.lower() in {"1", "true", "yes", "on"}


def trusted_bgen_validation_cache_directory() -> Path:
    """Return the trusted BGEN validation cache directory."""
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_home = Path(xdg_cache_home) if xdg_cache_home else Path.home() / ".cache"
    return cache_home / "g" / "bgen_validation"


def build_trusted_bgen_validation_fingerprint(
    *,
    bgen_path: Path,
    sample_count: int,
    variant_count: int,
    trusted_no_missing_diploid: bool,
) -> str:
    """Build a stable trusted BGEN validation fingerprint."""
    bgen_stat = bgen_path.stat()
    fingerprint_payload = {
        "schema_version": TRUSTED_BGEN_VALIDATION_SCHEMA_VERSION,
        "bgen_path": str(bgen_path.resolve()),
        "size": bgen_stat.st_size,
        "mtime_ns": bgen_stat.st_mtime_ns,
        "sample_count": sample_count,
        "variant_count": variant_count,
        "trusted_no_missing_diploid": trusted_no_missing_diploid,
    }
    fingerprint_json = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(fingerprint_json.encode("utf-8")).hexdigest()


def trusted_bgen_validation_cache_path(fingerprint: str) -> Path:
    """Return the validation cache path for a fingerprint."""
    return trusted_bgen_validation_cache_directory() / f"{fingerprint}.json"


def validate_trusted_bgen_with_cache(
    *,
    engine: _core.Regenie2RunEngine,
    bgen_path: Path,
    validation_mode: types.TrustedBgenValidationMode,
) -> None:
    """Validate or trust the no-missing diploid BGEN path according to mode."""
    if (
        assume_trusted_no_missing_diploid_validated()
        or validation_mode == types.TrustedBgenValidationMode.ASSUME_VALIDATED
    ):
        return
    fingerprint = build_trusted_bgen_validation_fingerprint(
        bgen_path=bgen_path,
        sample_count=int(engine.sample_count),
        variant_count=int(engine.variant_count),
        trusted_no_missing_diploid=True,
    )
    cache_path = trusted_bgen_validation_cache_path(fingerprint)
    if validation_mode == types.TrustedBgenValidationMode.CACHE_ON_MISS and cache_path.exists():
        return
    engine.validate_trusted_no_missing_diploid()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "schema_version": TRUSTED_BGEN_VALIDATION_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "bgen_path": str(bgen_path.resolve()),
        "sample_count": int(engine.sample_count),
        "variant_count": int(engine.variant_count),
    }
    temporary_cache_path = cache_path.with_suffix(".json.tmp")
    temporary_cache_path.write_text(json.dumps(cache_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary_cache_path.replace(cache_path)
