from __future__ import annotations

import dataclasses

import pytest

from g import execution_plan, types
from g.interface import config


def build_binary_config(**overrides: object) -> config.BinaryConfig:
    """Build packaged binary config with test overrides."""
    return dataclasses.replace(config.load_packaged_config().binary, **overrides)


def test_default_binary_config_normalizes_to_score_only() -> None:
    plan = execution_plan.normalize_binary_correction_config(build_binary_config())

    assert plan == types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod.SCORE_ONLY,
        p_threshold=0.05,
        firth_se=False,
    )


def test_firth_approx_maps_to_approximate_firth_plan() -> None:
    plan = execution_plan.normalize_binary_correction_config(
        build_binary_config(firth=True, approx=True, p_threshold=0.01, firth_se=True)
    )

    assert plan == types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
        p_threshold=0.01,
        firth_se=True,
    )


def test_approx_without_firth_raises() -> None:
    with pytest.raises(ValueError, match="--approx requires --firth"):
        execution_plan.normalize_binary_correction_config(build_binary_config(approx=True))


def test_firth_and_spa_raises_for_spa() -> None:
    with pytest.raises(NotImplementedError, match="SPA fallback is not implemented"):
        execution_plan.normalize_binary_correction_config(build_binary_config(firth=True, approx=True, spa=True))


@pytest.mark.parametrize("p_threshold", [0.0, 1.0, -0.01, 1.01])
def test_invalid_p_threshold_values_raise(p_threshold: float) -> None:
    with pytest.raises(ValueError, match="pThresh must be in"):
        execution_plan.normalize_binary_correction_config(build_binary_config(p_threshold=p_threshold))


def test_spa_raises_until_implemented() -> None:
    with pytest.raises(NotImplementedError, match="SPA fallback is not implemented"):
        execution_plan.normalize_binary_correction_config(build_binary_config(spa=True))


def test_exact_firth_raises_until_parity_proven() -> None:
    with pytest.raises(NotImplementedError, match="Exact REGENIE --firth"):
        execution_plan.normalize_binary_correction_config(build_binary_config(firth=True))
