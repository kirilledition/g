"""Deep-profile JAX cache helpers."""

from __future__ import annotations

import typing

import tooling.cli.profile_regenie2_deep as profile_regenie2_deep

if typing.TYPE_CHECKING:
    from pathlib import Path

    from tooling.profile_deep import models as profile_deep_models


def resolve_profile_jax_cache_directory(
    candidate: profile_deep_models.Step2Candidate,
    base_cache_directory: Path | None,
) -> Path | None:
    """Resolve the JAX cache directory for one profile candidate."""
    return profile_regenie2_deep.resolve_profile_jax_cache_directory(candidate, base_cache_directory)
