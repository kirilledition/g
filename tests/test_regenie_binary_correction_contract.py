from __future__ import annotations

import pytest

from g import execution_plan, types
from g.interface import config


def build_binary_correction_plan(**overrides: object) -> types.BinaryCorrectionPlan:
    """Build the native-planned binary correction plan with test overrides."""
    normalized_overrides = dict(overrides)
    if "p_threshold" in normalized_overrides:
        normalized_overrides["pThresh"] = normalized_overrides.pop("p_threshold")
    if "firth_se" in normalized_overrides:
        normalized_overrides["firth-se"] = normalized_overrides.pop("firth_se")
    raw_options: dict[str, object] = {
        "step": 2,
        "bt": True,
        "bgen": "dataset.bgen",
        "phenoFile": "phenotype.tsv",
        "phenoCol": "trait",
        "pred": "predictions.list",
        "out": "results/output",
    }
    raw_options.update(normalized_overrides)
    run_request = execution_plan.compile_run_request(config.RegenieConfig.from_options(raw_options))
    return execution_plan.adapt_binary_correction_plan(run_request.correction)


def test_default_binary_config_normalizes_to_score_only() -> None:
    plan = build_binary_correction_plan()

    assert plan.method == types.BinaryFallbackMethod.SCORE_ONLY
    assert plan.p_threshold == pytest.approx(0.05)
    assert plan.firth_se is False


def test_firth_approx_maps_to_approximate_firth_plan() -> None:
    plan = build_binary_correction_plan(firth=True, approx=True, p_threshold=0.01, firth_se=True)

    assert plan.method == types.BinaryFallbackMethod.FIRTH_APPROXIMATE
    assert plan.p_threshold == pytest.approx(0.01)
    assert plan.firth_se is True


def test_approx_without_firth_raises() -> None:
    with pytest.raises(ValueError, match="--approx requires --firth"):
        build_binary_correction_plan(firth=False, approx=True)


@pytest.mark.parametrize("p_threshold", [0.0, 1.0, -0.01, 1.01])
def test_invalid_p_threshold_values_raise(p_threshold: float) -> None:
    with pytest.raises(ValueError, match=r"pThresh|binary\.p_threshold"):
        build_binary_correction_plan(p_threshold=p_threshold)


def test_exact_firth_raises_until_parity_proven() -> None:
    with pytest.raises(ValueError, match="Exact --firth is not implemented"):
        build_binary_correction_plan(firth=True, approx=False)
