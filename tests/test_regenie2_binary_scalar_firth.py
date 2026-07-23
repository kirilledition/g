"""Correctness tests for scalar approximate-Firth primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

import tests.numerical
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import logistic as regenie2_binary_logistic
from g.compute.regenie2_binary.firth import scalar_approx as regenie2_binary_firth_scalar_approx
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types

UPSTREAM_REGENIE_LOGISTIC_MINIMUM_ETA = -30.0
UPSTREAM_REGENIE_LOGISTIC_MAXIMUM_ETA = 30.0
UPSTREAM_REGENIE_NUMERICAL_EPSILON_MULTIPLIER = 10.0


@dataclass(frozen=True)
class ScalarFirthFixture:
    """Deterministic scalar approximate-Firth operands."""

    phenotype_vector: npt.NDArray[np.float64]
    genotype_vector: npt.NDArray[np.float64]
    offset_vector: npt.NDArray[np.float64]
    active_sample_mask: npt.NDArray[np.bool_]
    non_active_deviance: float
    beta: float


@dataclass(frozen=True)
class ScalarFirthComponentReference:
    """Independent scalar Firth component values."""

    genotype_information: float
    score_adjustment: float
    penalized_deviance: float
    score: float
    valid: bool


@dataclass(frozen=True)
class ScalarLineSearchReference:
    """Independent upstream-style scalar step-halving result."""

    beta: float
    step_size: float
    components: ScalarFirthComponentReference
    attempt_count: int
    accepted: bool
    valid: bool


def build_scalar_firth_fixture() -> ScalarFirthFixture:
    """Build a well-conditioned lane with one inactive sample."""
    return ScalarFirthFixture(
        phenotype_vector=np.asarray([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64),
        genotype_vector=np.asarray([0.2, 1.1, 0.5, 1.8, 1.3, 0.7], dtype=np.float64),
        offset_vector=np.asarray([-0.2, 0.1, -0.1, 0.3, 0.2, -0.05], dtype=np.float64),
        active_sample_mask=np.asarray([True, True, True, True, False, True], dtype=np.bool_),
        non_active_deviance=0.7,
        beta=0.4,
    )


def build_binary_kernel_config(
    *,
    approximate_firth_maximum_iterations: int = 30,
) -> regenie2_binary_config.BinaryKernelConfig:
    """Build a small but production-shaped scalar solver policy."""
    return regenie2_binary_config.BinaryKernelConfig(
        numerical=regenie2_binary_config.BinaryNumericalConfig(
            minimum_probability=1.0e-7,
            minimum_variance=1.0e-10,
            relative_variance_tolerance=1.0e-7,
        ),
        null_logistic=regenie2_binary_config.BinaryNullLogisticConfig(
            maximum_iterations=100,
            coefficient_tolerance=1.0e-6,
        ),
        firth_candidate=regenie2_binary_config.FirthCandidateConfig(
            batch_size=4,
            candidate_capacity=8,
        ),
        approximate_firth=regenie2_binary_config.ApproximateFirthConfig(
            maximum_iterations=approximate_firth_maximum_iterations,
            gradient_tolerance=1.0e-8,
            maximum_step_size=5.0,
            pseudo_maximum_iterations=20,
            pseudo_inner_maximum_iterations=30,
            line_search_maximum_attempts=20,
            sparse_carrier_dosage_threshold=0.5,
            use_cuda_components=False,
        ),
        null_firth=regenie2_binary_config.NullFirthConfig(
            maximum_iterations=30,
            gradient_tolerance=1.0e-8,
            maximum_step_size=5.0,
            fallback_iteration_multiplier=2,
            fallback_step_divisor=2.0,
            line_search_maximum_attempts=20,
            step_halving_scale=0.5,
        ),
    )


def compute_scalar_firth_component_reference(
    fixture: ScalarFirthFixture,
    *,
    minimum_variance: float = 0.0,
) -> ScalarFirthComponentReference:
    """Evaluate the scalar penalized likelihood and adjusted score in NumPy."""
    linear_predictor = fixture.offset_vector + fixture.genotype_vector * fixture.beta
    epsilon = UPSTREAM_REGENIE_NUMERICAL_EPSILON_MULTIPLIER * np.finfo(np.float64).eps
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ordinary_probability = np.reciprocal(
            1.0
            + np.exp(
                -np.clip(
                    linear_predictor,
                    UPSTREAM_REGENIE_LOGISTIC_MINIMUM_ETA,
                    UPSTREAM_REGENIE_LOGISTIC_MAXIMUM_ETA,
                )
            )
        )
        probability = np.where(
            linear_predictor < UPSTREAM_REGENIE_LOGISTIC_MINIMUM_ETA,
            epsilon / (1.0 + epsilon),
            np.where(
                linear_predictor > UPSTREAM_REGENIE_LOGISTIC_MAXIMUM_ETA,
                1.0 / (1.0 + epsilon),
                ordinary_probability,
            ),
        )
        weight = probability * (1.0 - probability)
        active_weight = np.where(fixture.active_sample_mask, weight, 0.0)
        information_diagonal = fixture.genotype_vector**2 * active_weight
        genotype_information = float(np.sum(information_diagonal))
        negative_log_likelihood = -np.where(
            fixture.phenotype_vector > 0.5,
            np.log(probability),
            np.log1p(-probability),
        )
        active_deviance = 2.0 * np.sum(np.where(fixture.active_sample_mask, negative_log_likelihood, 0.0))
        score_adjustment = float(
            np.sum(
                np.where(
                    fixture.active_sample_mask,
                    fixture.genotype_vector * information_diagonal * (0.5 - probability),
                    0.0,
                )
            )
            / genotype_information
        )
        score = float(
            np.sum(
                np.where(
                    fixture.active_sample_mask,
                    fixture.genotype_vector * (fixture.phenotype_vector - probability),
                    0.0,
                )
            )
            + score_adjustment
        )
        penalized_deviance = float(fixture.non_active_deviance + active_deviance - np.log(genotype_information))
    valid = (
        np.isfinite(genotype_information)
        and genotype_information > minimum_variance
        and np.isfinite(penalized_deviance)
        and np.isfinite(score)
    )
    return ScalarFirthComponentReference(
        genotype_information=genotype_information,
        score_adjustment=score_adjustment,
        penalized_deviance=penalized_deviance,
        score=score,
        valid=bool(valid),
    )


def run_scalar_line_search_reference(
    *,
    fixture: ScalarFirthFixture,
    initial_step_size: float,
    maximum_attempts: int,
    minimum_variance: float,
) -> ScalarLineSearchReference:
    """Apply REGENIE's scalar half-step recurrence to independent components."""
    current_components = compute_scalar_firth_component_reference(
        fixture,
        minimum_variance=minimum_variance,
    )
    retained_beta = fixture.beta
    retained_components = current_components
    step_size = initial_step_size
    accepted = False
    attempt_count = 0
    for attempt_count in range(1, maximum_attempts + 1):
        if attempt_count > 1:
            step_size /= 2.0
        candidate_beta = fixture.beta + step_size
        candidate_components = compute_scalar_firth_component_reference(
            ScalarFirthFixture(
                phenotype_vector=fixture.phenotype_vector,
                genotype_vector=fixture.genotype_vector,
                offset_vector=fixture.offset_vector,
                active_sample_mask=fixture.active_sample_mask,
                non_active_deviance=fixture.non_active_deviance,
                beta=candidate_beta,
            ),
            minimum_variance=minimum_variance,
        )
        accepted = candidate_components.valid and (
            candidate_components.penalized_deviance < current_components.penalized_deviance
        )
        if accepted:
            retained_beta = candidate_beta
            retained_components = candidate_components
            break
    return ScalarLineSearchReference(
        beta=retained_beta,
        step_size=step_size,
        components=retained_components,
        attempt_count=attempt_count,
        accepted=accepted,
        valid=current_components.valid,
    )


def compute_fixture_components(
    fixture: ScalarFirthFixture,
) -> regenie2_binary_firth_types.ScalarFirthComponents:
    """Evaluate the production pure-JAX scalar component path."""
    return regenie2_binary_firth_scalar_approx.compute_scalar_firth_components_with_minimum_variance(
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        genotype_vector=jnp.asarray(fixture.genotype_vector),
        offset_vector=jnp.asarray(fixture.offset_vector),
        active_sample_mask=jnp.asarray(fixture.active_sample_mask),
        non_active_deviance=jnp.asarray(fixture.non_active_deviance),
        beta=jnp.asarray(fixture.beta),
        minimum_variance=jnp.asarray(1.0e-10),
        use_cuda_components=False,
    )


def test_scalar_firth_components_match_independent_numpy_formula() -> None:
    """Validate information, adjustment, penalized deviance, and score together."""
    fixture = build_scalar_firth_fixture()
    reference = compute_scalar_firth_component_reference(fixture)

    observed = compute_fixture_components(fixture)

    tests.numerical.assert_absolute_difference_less_than(
        observed.genotype_information,
        reference.genotype_information,
        1.0e-12,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.score_adjustment,
        reference.score_adjustment,
        1.0e-12,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.penalized_deviance,
        reference.penalized_deviance,
        1.0e-12,
    )
    tests.numerical.assert_absolute_difference_less_than(observed.score, reference.score, 1.0e-12)
    assert reference.valid
    assert bool(np.asarray(observed.valid))


def test_scalar_firth_rejects_zero_genotype_information() -> None:
    """Reject a collinear or empty scalar genotype lane."""
    observed = regenie2_binary_firth_scalar_approx.compute_scalar_firth_components_with_minimum_variance(
        phenotype_vector=jnp.asarray([0.0, 1.0, 0.0, 1.0], dtype=jnp.float64),
        genotype_vector=jnp.zeros((4,), dtype=jnp.float64),
        offset_vector=jnp.zeros((4,), dtype=jnp.float64),
        active_sample_mask=jnp.ones((4,), dtype=jnp.bool_),
        non_active_deviance=jnp.asarray(0.0, dtype=jnp.float64),
        beta=jnp.asarray(0.0, dtype=jnp.float64),
        minimum_variance=jnp.asarray(1.0e-10, dtype=jnp.float64),
        use_cuda_components=False,
    )

    assert not bool(np.asarray(observed.valid))
    assert float(np.asarray(observed.genotype_information)) == 0.0


def test_scalar_solver_parameter_budget_is_split_between_pseudo_and_newton() -> None:
    """Keep the total configured budget bounded across both solver phases."""
    observed = regenie2_binary_firth_scalar_approx.build_scalar_approximate_firth_solver_parameters(
        build_binary_kernel_config()
    )

    assert int(np.asarray(observed.pseudo_maximum_iterations)) == 15
    assert int(np.asarray(observed.newton_raphson_maximum_iterations)) == 15
    assert int(np.asarray(observed.pseudo_inner_maximum_iterations)) == 30
    assert int(np.asarray(observed.line_search_maximum_attempts)) == 20
    assert not observed.use_cuda_components


def test_scalar_solver_parameter_budget_preserves_floor_split() -> None:
    """Floor-divide an odd valid total while leaving two iterations per phase."""
    observed = regenie2_binary_firth_scalar_approx.build_scalar_approximate_firth_solver_parameters(
        build_binary_kernel_config(approximate_firth_maximum_iterations=5)
    )

    assert int(np.asarray(observed.pseudo_maximum_iterations)) == 2
    assert int(np.asarray(observed.newton_raphson_maximum_iterations)) == 2


def test_scalar_solver_parameter_budget_accepts_exact_minimum_split() -> None:
    """Accept the minimum total as exactly two iterations for each phase."""
    observed = regenie2_binary_firth_scalar_approx.build_scalar_approximate_firth_solver_parameters(
        build_binary_kernel_config(approximate_firth_maximum_iterations=4)
    )

    assert int(np.asarray(observed.pseudo_maximum_iterations)) == 2
    assert int(np.asarray(observed.newton_raphson_maximum_iterations)) == 2


def test_scalar_solver_parameter_budget_rejects_fewer_than_four_total_iterations() -> None:
    """Reject a total budget that cannot give both phases two iterations."""
    with np.testing.assert_raises_regex(ValueError, "must be at least 4"):
        regenie2_binary_firth_scalar_approx.build_scalar_approximate_firth_solver_parameters(
            build_binary_kernel_config(approximate_firth_maximum_iterations=3)
        )


def test_scalar_line_search_with_zero_attempts_retains_trusted_state() -> None:
    """Do not move beta when the line-search budget is exhausted up front."""
    fixture = build_scalar_firth_fixture()
    current_components = compute_fixture_components(fixture)
    observed = regenie2_binary_firth_scalar_approx.run_scalar_line_search_with_minimum_variance(
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        genotype_vector=jnp.asarray(fixture.genotype_vector),
        offset_vector=jnp.asarray(fixture.offset_vector),
        active_sample_mask=jnp.asarray(fixture.active_sample_mask),
        non_active_deviance=jnp.asarray(fixture.non_active_deviance),
        current_beta=jnp.asarray(fixture.beta),
        current_penalized_deviance=current_components.penalized_deviance,
        current_genotype_information=current_components.genotype_information,
        current_score=current_components.score,
        current_valid=current_components.valid,
        initial_step_size=jnp.asarray(1.0),
        maximum_attempts=0,
        minimum_variance=jnp.asarray(1.0e-10),
        use_cuda_components=False,
    )

    tests.numerical.assert_absolute_difference_less_than(observed.beta, fixture.beta, 1.0e-15)
    tests.numerical.assert_absolute_difference_less_than(
        observed.penalized_deviance,
        current_components.penalized_deviance,
        1.0e-15,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.genotype_information,
        current_components.genotype_information,
        1.0e-15,
    )
    tests.numerical.assert_absolute_difference_less_than(observed.score, current_components.score, 1.0e-15)
    assert int(np.asarray(observed.attempt_count)) == 0
    assert not bool(np.asarray(observed.accepted))
    assert bool(np.asarray(observed.valid))


@pytest.mark.parametrize(
    "candidate_penalized_deviance",
    [11.0, 10.0],
    ids=["valid-worse", "valid-equal"],
)
def test_scalar_line_search_rejects_valid_nonimproving_proposals(
    monkeypatch: pytest.MonkeyPatch,
    candidate_penalized_deviance: float,
) -> None:
    """Require strict objective improvement from every otherwise-valid proposal."""

    def compute_candidate_components(
        *,
        phenotype_vector: jax.Array,
        genotype_vector: jax.Array,
        offset_vector: jax.Array,
        active_sample_mask: jax.Array,
        non_active_deviance: jax.Array,
        beta: jax.Array,
        minimum_variance: jax.Array,
        use_cuda_components: bool,
    ) -> regenie2_binary_firth_types.ScalarFirthComponents:
        del (
            phenotype_vector,
            genotype_vector,
            offset_vector,
            active_sample_mask,
            non_active_deviance,
            minimum_variance,
            use_cuda_components,
        )
        return regenie2_binary_firth_types.ScalarFirthComponents(
            genotype_information=jnp.asarray(2.0, dtype=beta.dtype),
            score_adjustment=jnp.asarray(0.25, dtype=beta.dtype),
            penalized_deviance=jnp.asarray(candidate_penalized_deviance, dtype=beta.dtype),
            score=jnp.asarray(3.0, dtype=beta.dtype),
            valid=jnp.asarray(1, dtype=jnp.bool_),
        )

    monkeypatch.setattr(
        regenie2_binary_firth_scalar_approx,
        "compute_scalar_firth_components_with_minimum_variance",
        compute_candidate_components,
    )
    observed = regenie2_binary_firth_scalar_approx.run_scalar_line_search_with_minimum_variance(
        phenotype_vector=jnp.ones((1,), dtype=jnp.float64),
        genotype_vector=jnp.ones((1,), dtype=jnp.float64),
        offset_vector=jnp.zeros((1,), dtype=jnp.float64),
        active_sample_mask=jnp.ones((1,), dtype=jnp.bool_),
        non_active_deviance=jnp.asarray(0.0, dtype=jnp.float64),
        current_beta=jnp.asarray(0.0, dtype=jnp.float64),
        current_penalized_deviance=jnp.asarray(10.0, dtype=jnp.float64),
        current_genotype_information=jnp.asarray(1.0, dtype=jnp.float64),
        current_score=jnp.asarray(2.0, dtype=jnp.float64),
        current_valid=jnp.asarray(1, dtype=jnp.bool_),
        initial_step_size=jnp.asarray(1.0, dtype=jnp.float64),
        maximum_attempts=1,
        minimum_variance=jnp.asarray(1.0e-10, dtype=jnp.float64),
        use_cuda_components=False,
    )

    assert int(np.asarray(observed.attempt_count)) == 1
    assert not bool(np.asarray(observed.accepted))
    assert bool(np.asarray(observed.valid))
    tests.numerical.assert_absolute_difference_less_than(observed.beta, 0.0, 1.0e-15)
    tests.numerical.assert_absolute_difference_less_than(observed.penalized_deviance, 10.0, 1.0e-15)
    tests.numerical.assert_absolute_difference_less_than(observed.genotype_information, 1.0, 1.0e-15)
    tests.numerical.assert_absolute_difference_less_than(observed.score, 2.0, 1.0e-15)


def test_scalar_line_search_accepts_valid_half_step_after_invalid_full_step() -> None:
    """Continue real step-halving after saturation invalidates the full proposal."""
    fixture = ScalarFirthFixture(
        phenotype_vector=np.ones((4,), dtype=np.float64),
        genotype_vector=np.ones((4,), dtype=np.float64),
        offset_vector=np.zeros((4,), dtype=np.float64),
        active_sample_mask=np.ones((4,), dtype=np.bool_),
        non_active_deviance=0.0,
        beta=-10.0,
    )
    initial_step_size = 60.0
    minimum_variance = 1.0e-10
    current_reference = compute_scalar_firth_component_reference(
        fixture,
        minimum_variance=minimum_variance,
    )
    full_step_reference = compute_scalar_firth_component_reference(
        ScalarFirthFixture(
            phenotype_vector=fixture.phenotype_vector,
            genotype_vector=fixture.genotype_vector,
            offset_vector=fixture.offset_vector,
            active_sample_mask=fixture.active_sample_mask,
            non_active_deviance=fixture.non_active_deviance,
            beta=fixture.beta + initial_step_size,
        ),
        minimum_variance=minimum_variance,
    )
    half_step_reference = compute_scalar_firth_component_reference(
        ScalarFirthFixture(
            phenotype_vector=fixture.phenotype_vector,
            genotype_vector=fixture.genotype_vector,
            offset_vector=fixture.offset_vector,
            active_sample_mask=fixture.active_sample_mask,
            non_active_deviance=fixture.non_active_deviance,
            beta=fixture.beta + initial_step_size / 2.0,
        ),
        minimum_variance=minimum_variance,
    )
    line_search_reference = run_scalar_line_search_reference(
        fixture=fixture,
        initial_step_size=initial_step_size,
        maximum_attempts=2,
        minimum_variance=minimum_variance,
    )
    observed = regenie2_binary_firth_scalar_approx.run_scalar_line_search_with_minimum_variance(
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        genotype_vector=jnp.asarray(fixture.genotype_vector),
        offset_vector=jnp.asarray(fixture.offset_vector),
        active_sample_mask=jnp.asarray(fixture.active_sample_mask),
        non_active_deviance=jnp.asarray(fixture.non_active_deviance),
        current_beta=jnp.asarray(fixture.beta),
        current_penalized_deviance=jnp.asarray(current_reference.penalized_deviance),
        current_genotype_information=jnp.asarray(current_reference.genotype_information),
        current_score=jnp.asarray(current_reference.score),
        current_valid=jnp.asarray(current_reference.valid),
        initial_step_size=jnp.asarray(initial_step_size),
        maximum_attempts=2,
        minimum_variance=jnp.asarray(minimum_variance),
        use_cuda_components=False,
    )

    assert current_reference.valid
    assert not full_step_reference.valid
    assert half_step_reference.valid
    assert half_step_reference.penalized_deviance < current_reference.penalized_deviance
    assert line_search_reference.attempt_count == 2
    assert line_search_reference.accepted
    assert line_search_reference.valid
    assert int(np.asarray(observed.attempt_count)) == line_search_reference.attempt_count
    assert bool(np.asarray(observed.accepted)) is line_search_reference.accepted
    assert bool(np.asarray(observed.valid)) is line_search_reference.valid
    tests.numerical.assert_absolute_difference_less_than(observed.beta, line_search_reference.beta, 1.0e-15)
    tests.numerical.assert_absolute_difference_less_than(
        observed.step_size,
        line_search_reference.step_size,
        1.0e-15,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.penalized_deviance,
        line_search_reference.components.penalized_deviance,
        1.0e-12,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.genotype_information,
        line_search_reference.components.genotype_information,
        1.0e-15,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.score,
        line_search_reference.components.score,
        1.0e-12,
    )


def test_scalar_line_search_all_invalid_candidates_retain_trusted_state() -> None:
    """Exhaust real saturated proposals without corrupting trusted quantities."""
    fixture = ScalarFirthFixture(
        phenotype_vector=np.ones((4,), dtype=np.float64),
        genotype_vector=np.ones((4,), dtype=np.float64),
        offset_vector=np.zeros((4,), dtype=np.float64),
        active_sample_mask=np.ones((4,), dtype=np.bool_),
        non_active_deviance=0.0,
        beta=-10.0,
    )
    initial_step_size = 1_000.0
    minimum_variance = 1.0e-10
    current_reference = compute_scalar_firth_component_reference(
        fixture,
        minimum_variance=minimum_variance,
    )
    for candidate_step_scale in (1.0, 0.5, 0.25):
        candidate_reference = compute_scalar_firth_component_reference(
            ScalarFirthFixture(
                phenotype_vector=fixture.phenotype_vector,
                genotype_vector=fixture.genotype_vector,
                offset_vector=fixture.offset_vector,
                active_sample_mask=fixture.active_sample_mask,
                non_active_deviance=fixture.non_active_deviance,
                beta=fixture.beta + initial_step_size * candidate_step_scale,
            ),
            minimum_variance=minimum_variance,
        )
        assert not candidate_reference.valid

    line_search_reference = run_scalar_line_search_reference(
        fixture=fixture,
        initial_step_size=initial_step_size,
        maximum_attempts=3,
        minimum_variance=minimum_variance,
    )

    observed = regenie2_binary_firth_scalar_approx.run_scalar_line_search_with_minimum_variance(
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        genotype_vector=jnp.asarray(fixture.genotype_vector),
        offset_vector=jnp.asarray(fixture.offset_vector),
        active_sample_mask=jnp.asarray(fixture.active_sample_mask),
        non_active_deviance=jnp.asarray(fixture.non_active_deviance),
        current_beta=jnp.asarray(fixture.beta),
        current_penalized_deviance=jnp.asarray(current_reference.penalized_deviance),
        current_genotype_information=jnp.asarray(current_reference.genotype_information),
        current_score=jnp.asarray(current_reference.score),
        current_valid=jnp.asarray(current_reference.valid),
        initial_step_size=jnp.asarray(initial_step_size),
        maximum_attempts=3,
        minimum_variance=jnp.asarray(minimum_variance),
        use_cuda_components=False,
    )

    assert current_reference.valid
    assert line_search_reference.attempt_count == 3
    assert not line_search_reference.accepted
    assert line_search_reference.valid
    assert int(np.asarray(observed.attempt_count)) == line_search_reference.attempt_count
    assert bool(np.asarray(observed.accepted)) is line_search_reference.accepted
    assert bool(np.asarray(observed.valid)) is line_search_reference.valid
    tests.numerical.assert_absolute_difference_less_than(observed.beta, line_search_reference.beta, 1.0e-15)
    tests.numerical.assert_absolute_difference_less_than(
        observed.step_size,
        line_search_reference.step_size,
        1.0e-15,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.penalized_deviance,
        line_search_reference.components.penalized_deviance,
        1.0e-12,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.genotype_information,
        line_search_reference.components.genotype_information,
        1.0e-15,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.score,
        line_search_reference.components.score,
        1.0e-12,
    )


def test_sparse_initialization_accounts_for_inactive_null_deviance() -> None:
    """Retain the full-sample null objective while solving on carriers only."""
    phenotype = jnp.asarray([0.0, 1.0, 0.0, 1.0, 1.0], dtype=jnp.float64)
    genotype = jnp.asarray([0.0, 1.0, 0.0, 1.5, 0.0], dtype=jnp.float64)
    offset = jnp.asarray([-0.2, 0.1, 0.0, 0.3, -0.1], dtype=jnp.float64)
    carrier_mask = genotype > 0.5
    null_probability = regenie2_binary_logistic.compute_regenie_logistic_probability(offset)
    full_null_deviance = regenie2_binary_logistic.compute_logistic_deviance(
        phenotype,
        null_probability,
        jnp.ones_like(phenotype, dtype=jnp.bool_),
    )
    solver_parameters = regenie2_binary_firth_scalar_approx.build_scalar_approximate_firth_solver_parameters(
        build_binary_kernel_config()
    )

    observed = regenie2_binary_firth_scalar_approx.initialize_single_variant_regenie_approximate_firth(
        phenotype_vector=phenotype,
        genotype_vector=genotype,
        offset_vector=offset,
        carrier_sample_mask=carrier_mask,
        full_null_deviance=full_null_deviance,
        sparse_correction=jnp.asarray(1, dtype=jnp.bool_),
        solver_parameters=solver_parameters,
    )
    active_null_deviance = regenie2_binary_logistic.compute_logistic_deviance(
        phenotype,
        null_probability,
        carrier_mask,
    )

    np.testing.assert_array_equal(np.asarray(observed.active_sample_mask), np.asarray(carrier_mask))
    tests.numerical.assert_absolute_difference_less_than(
        observed.non_active_deviance,
        np.asarray(full_null_deviance - active_null_deviance),
        1.0e-12,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.deviance_null,
        np.asarray(full_null_deviance - jnp.log(observed.components.genotype_information)),
        1.0e-12,
    )


def test_scalar_newton_solver_converges_on_regular_dense_lane() -> None:
    """Reach a valid finite terminal result from a shared initialization."""
    phenotype = jnp.asarray([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=jnp.float64)
    genotype = jnp.asarray([0.1, 1.2, 0.4, 1.8, 1.1, 0.3, 1.6, 0.7], dtype=jnp.float64)
    offset = jnp.asarray([-0.1, 0.1, -0.05, 0.2, 0.1, -0.1, 0.15, 0.0], dtype=jnp.float64)
    null_probability = regenie2_binary_logistic.compute_regenie_logistic_probability(offset)
    full_null_deviance = regenie2_binary_logistic.compute_logistic_deviance(
        phenotype,
        null_probability,
        jnp.ones_like(phenotype, dtype=jnp.bool_),
    )
    solver_parameters = regenie2_binary_firth_scalar_approx.build_scalar_approximate_firth_solver_parameters(
        build_binary_kernel_config()
    )
    initial_state = regenie2_binary_firth_scalar_approx.initialize_single_variant_regenie_approximate_firth(
        phenotype_vector=phenotype,
        genotype_vector=genotype,
        offset_vector=offset,
        carrier_sample_mask=jnp.ones_like(phenotype, dtype=jnp.bool_),
        full_null_deviance=full_null_deviance,
        sparse_correction=jnp.asarray(0, dtype=jnp.bool_),
        solver_parameters=solver_parameters,
    )

    terminal = regenie2_binary_firth_scalar_approx.run_initialized_scalar_newton_raphson_firth_solver(initial_state)
    terminal_components = regenie2_binary_firth_scalar_approx.compute_scalar_firth_components_with_minimum_variance(
        phenotype_vector=phenotype,
        genotype_vector=genotype,
        offset_vector=offset,
        active_sample_mask=jnp.ones_like(phenotype, dtype=jnp.bool_),
        non_active_deviance=jnp.asarray(0.0),
        beta=terminal.beta,
        minimum_variance=solver_parameters.minimum_variance,
        use_cuda_components=False,
    )

    assert bool(np.asarray(terminal.valid_mask))
    assert float(np.abs(np.asarray(terminal_components.score))) < 1.0e-8
    assert float(np.asarray(terminal.chi_squared)) >= 0.0
    assert float(np.asarray(terminal.standard_error)) > 0.0


def test_finalization_uses_chi_square_tail_without_changing_status() -> None:
    """Convert the terminal likelihood ratio and retain exact validity."""
    terminal = regenie2_binary_firth_types.ScalarFirthTerminalResult(
        beta=jnp.asarray([0.5, -0.2], dtype=jnp.float64),
        standard_error=jnp.asarray([0.2, 0.3], dtype=jnp.float64),
        chi_squared=jnp.asarray([4.0, 0.0], dtype=jnp.float64),
        valid_mask=jnp.asarray([True, False]),
    )
    observed = regenie2_binary_firth_scalar_approx.finalize_scalar_firth_terminal_result(terminal)
    reference_log10_p_value = np.asarray(
        [-math.log10(math.erfc(math.sqrt(2.0))), 0.0],
        dtype=np.float64,
    )

    tests.numerical.assert_absolute_difference_less_than(observed.log10_p_value, reference_log10_p_value, 1.0e-12)
    np.testing.assert_array_equal(np.asarray(observed.valid_mask), np.asarray([True, False]))
