"""Correctness tests for the covariate-only Firth null solver."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

import tests.numerical
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary.firth import null as regenie2_binary_firth_null
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types

if typing.TYPE_CHECKING:
    import pytest

PRODUCTION_NULL_FIRTH_GRADIENT_TOLERANCE = 50.0e-6
PRODUCTION_NULL_FIRTH_MAXIMUM_STEP_SIZE = 25.0
REGENIE_NULL_FIRTH_INITIAL_SCORE_MAXIMUM = 1.0e16


@dataclass(frozen=True)
class NullFirthFixture:
    """Full-rank covariate-only Firth inputs."""

    covariate_matrix: npt.NDArray[np.float64]
    phenotype_vector: npt.NDArray[np.float64]
    loco_offset: npt.NDArray[np.float64]
    coefficients: npt.NDArray[np.float64]


@dataclass(frozen=True)
class NullFirthComponentReference:
    """Independent NumPy null-Firth components."""

    information_cholesky_factor: npt.NDArray[np.float64]
    deviance: float
    modified_score: npt.NDArray[np.float64]


@dataclass(frozen=True)
class NullFirthScoreHistoryReference:
    """Independent scalar form of REGENIE's null-Firth score history."""

    previous_score_maximum: float
    score_increase_count: int
    failed: bool


def build_null_firth_fixture() -> NullFirthFixture:
    """Build a non-separated fixture with a nonzero offset and start."""
    return NullFirthFixture(
        covariate_matrix=np.asarray(
            [
                [1.0, -1.5],
                [1.0, -1.0],
                [1.0, -0.5],
                [1.0, 0.0],
                [1.0, 0.5],
                [1.0, 1.0],
                [1.0, 1.5],
                [1.0, 2.0],
            ],
            dtype=np.float64,
        ),
        phenotype_vector=np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64),
        loco_offset=np.asarray([0.04, -0.03, 0.01, 0.00, -0.02, 0.03, -0.01, 0.02], dtype=np.float64),
        coefficients=np.asarray([0.15, -0.08], dtype=np.float64),
    )


def build_null_firth_policy_config() -> regenie2_binary_config.BinaryKernelConfig:
    """Build a distinctive policy for testing every null-Firth fallback attempt."""
    return regenie2_binary_config.BinaryKernelConfig(
        numerical=regenie2_binary_config.BinaryNumericalConfig(
            minimum_probability=1.0e-7,
            minimum_variance=1.0e-10,
            relative_variance_tolerance=1.0e-7,
        ),
        null_logistic=regenie2_binary_config.BinaryNullLogisticConfig(
            maximum_iterations=17,
            coefficient_tolerance=1.0e-6,
        ),
        firth_candidate=regenie2_binary_config.FirthCandidateConfig(
            batch_size=4,
            candidate_capacity=8,
        ),
        approximate_firth=regenie2_binary_config.ApproximateFirthConfig(
            maximum_iterations=19,
            gradient_tolerance=2.5e-4,
            maximum_step_size=5.0,
            pseudo_maximum_iterations=11,
            pseudo_inner_maximum_iterations=13,
            line_search_maximum_attempts=7,
            sparse_carrier_dosage_threshold=0.5,
            use_cuda_components=False,
        ),
        null_firth=regenie2_binary_config.NullFirthConfig(
            maximum_iterations=13,
            gradient_tolerance=1.25e-7,
            maximum_step_size=9.0,
            fallback_iteration_multiplier=3,
            fallback_step_divisor=4.0,
            line_search_maximum_attempts=7,
            step_halving_scale=0.25,
        ),
    )


def compute_null_firth_component_reference(fixture: NullFirthFixture) -> NullFirthComponentReference:
    """Evaluate the Jeffreys-adjusted null score with NumPy."""
    linear_predictor = fixture.covariate_matrix @ fixture.coefficients + fixture.loco_offset
    probability = np.reciprocal(1.0 + np.exp(-linear_predictor))
    weight = probability * (1.0 - probability)
    information = (fixture.covariate_matrix.T * weight) @ fixture.covariate_matrix
    cholesky_factor = np.linalg.cholesky(information)
    negative_log_likelihood = -np.where(
        fixture.phenotype_vector > 0.5,
        np.log(probability),
        np.log1p(-probability),
    )
    deviance = 2.0 * np.sum(negative_log_likelihood) - np.linalg.slogdet(information).logabsdet
    projected_covariates = np.linalg.solve(information, fixture.covariate_matrix.T).T
    leverage = weight * np.sum(projected_covariates * fixture.covariate_matrix, axis=1)
    modified_score = fixture.covariate_matrix.T @ (
        fixture.phenotype_vector - probability + leverage * (0.5 - probability)
    )
    return NullFirthComponentReference(
        information_cholesky_factor=cholesky_factor,
        deviance=float(deviance),
        modified_score=modified_score,
    )


def update_null_firth_score_history_reference(
    *,
    state: NullFirthScoreHistoryReference,
    score_maximum: float,
    converged: bool,
    check_score_increase: bool,
) -> NullFirthScoreHistoryReference:
    """Apply the scalar recurrence and control order from upstream REGENIE."""
    if converged:
        return state
    score_increase_count = state.score_increase_count + 1 if score_maximum > state.previous_score_maximum else 0
    return NullFirthScoreHistoryReference(
        previous_score_maximum=score_maximum,
        score_increase_count=score_increase_count,
        failed=state.failed
        or (
            check_score_increase
            and score_increase_count > regenie2_binary_firth_null.NULL_FIRTH_MAXIMUM_CONSECUTIVE_SCORE_INCREASES
        ),
    )


def compare_score_history_sequence_to_regenie(
    *,
    score_maxima: list[float],
    convergence_iteration: int | None,
    check_score_increase: bool,
) -> list[regenie2_binary_firth_types.NullFirthScoreHistoryState]:
    """Compare every production transition with the independent recurrence."""
    observed_state = regenie2_binary_firth_types.NullFirthScoreHistoryState(
        previous_score_maximum=jnp.asarray(REGENIE_NULL_FIRTH_INITIAL_SCORE_MAXIMUM, dtype=jnp.float64),
        score_increase_count=jnp.asarray(0, dtype=jnp.int32),
        failed=jnp.asarray(0, dtype=jnp.bool_),
    )
    reference_state = NullFirthScoreHistoryReference(
        previous_score_maximum=REGENIE_NULL_FIRTH_INITIAL_SCORE_MAXIMUM,
        score_increase_count=0,
        failed=False,
    )
    observed_states: list[regenie2_binary_firth_types.NullFirthScoreHistoryState] = []
    for iteration, score_maximum in enumerate(score_maxima, start=1):
        converged = iteration == convergence_iteration
        observed_state = regenie2_binary_firth_null.update_null_firth_score_history(
            state=observed_state,
            score_maximum=jnp.asarray(score_maximum, dtype=jnp.float64),
            converged=converged,
            check_score_increase=check_score_increase,
        )
        reference_state = update_null_firth_score_history_reference(
            state=reference_state,
            score_maximum=score_maximum,
            converged=converged,
            check_score_increase=check_score_increase,
        )
        observed_previous_score_maximum = float(np.asarray(observed_state.previous_score_maximum))
        if np.isnan(reference_state.previous_score_maximum):
            assert np.isnan(observed_previous_score_maximum)
        else:
            assert observed_previous_score_maximum == reference_state.previous_score_maximum
        assert int(np.asarray(observed_state.score_increase_count)) == reference_state.score_increase_count
        assert bool(np.asarray(observed_state.failed)) is reference_state.failed
        observed_states.append(observed_state)
    return observed_states


def test_null_firth_score_history_tracks_the_immediately_previous_score() -> None:
    """Reset the increase count when the current score drops from the prior iterate."""
    observed_states = compare_score_history_sequence_to_regenie(
        score_maxima=[10.0, 1.0, 2.0, 1.5],
        convergence_iteration=None,
        check_score_increase=True,
    )

    observed_counts = [int(np.asarray(state.score_increase_count)) for state in observed_states]
    assert observed_counts == [0, 0, 1, 0]
    assert float(np.asarray(observed_states[-1].previous_score_maximum)) == 1.5


def test_null_firth_first_score_above_regenie_sentinel_counts_as_an_increase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start the recurrence from REGENIE's finite 1e16 score sentinel."""

    def compute_increasing_components(
        *,
        covariate_matrix: jax.Array,
        phenotype_vector: jax.Array,
        loco_offset: jax.Array,
        coefficients: jax.Array,
    ) -> regenie2_binary_firth_types.NullFirthComponents:
        del covariate_matrix, phenotype_vector, loco_offset
        score_maximum = REGENIE_NULL_FIRTH_INITIAL_SCORE_MAXIMUM + 2.0 * (coefficients[0] + 1.0)
        return regenie2_binary_firth_types.NullFirthComponents(
            information_cholesky_factor=jnp.ones((1, 1), dtype=coefficients.dtype),
            deviance=-coefficients[0],
            modified_score=jnp.asarray([score_maximum], dtype=coefficients.dtype),
            valid=jnp.asarray(1, dtype=jnp.bool_),
        )

    monkeypatch.setattr(
        regenie2_binary_firth_null,
        "compute_null_firth_components",
        compute_increasing_components,
    )
    observed = regenie2_binary_firth_null.fit_covariate_only_firth_null_model_once(
        covariate_matrix=jnp.ones((1, 1), dtype=jnp.float64),
        phenotype_vector=jnp.ones((1,), dtype=jnp.float64),
        loco_offset=jnp.zeros((1,), dtype=jnp.float64),
        initial_coefficients=jnp.zeros((1,), dtype=jnp.float64),
        maximum_iterations=26,
        maximum_step_size=1.0,
        tolerance=1.0,
        line_search_maximum_attempts=1,
        line_search_step_halving_scale=0.5,
        check_score_increase=True,
    )

    assert not bool(np.asarray(observed.converged))
    tests.numerical.assert_absolute_difference_less_than(observed.coefficients, np.asarray([25.0]), 1.0e-15)


def test_null_firth_score_history_does_not_accumulate_nonconsecutive_increases() -> None:
    """Accept a final convergence after an alternating-score trajectory."""
    score_maxima = [0.5] + ([2.0, 1.5] * 12) + [2.0, 0.75]
    observed_states = compare_score_history_sequence_to_regenie(
        score_maxima=score_maxima,
        convergence_iteration=len(score_maxima),
        check_score_increase=True,
    )

    final_state = observed_states[-1]
    assert int(np.asarray(final_state.score_increase_count)) == 1
    assert float(np.asarray(final_state.previous_score_maximum)) == 2.0
    assert not bool(np.asarray(final_state.failed))


def test_null_firth_score_history_fails_only_after_25_consecutive_increases() -> None:
    """Match REGENIE's strict greater-than-25 failure threshold."""
    score_maxima = [float(score_maximum) for score_maximum in range(27)]
    observed_states = compare_score_history_sequence_to_regenie(
        score_maxima=score_maxima,
        convergence_iteration=None,
        check_score_increase=True,
    )

    assert int(np.asarray(observed_states[-2].score_increase_count)) == 25
    assert not bool(np.asarray(observed_states[-2].failed))
    assert int(np.asarray(observed_states[-1].score_increase_count)) == 26
    assert bool(np.asarray(observed_states[-1].failed))


def test_null_firth_score_history_can_disable_the_increase_failure() -> None:
    """Track the recurrence without failing the final fallback policy."""
    score_maxima = [float(score_maximum) for score_maximum in range(27)]
    observed_states = compare_score_history_sequence_to_regenie(
        score_maxima=score_maxima,
        convergence_iteration=None,
        check_score_increase=False,
    )

    assert int(np.asarray(observed_states[-1].score_increase_count)) == 26
    assert not bool(np.asarray(observed_states[-1].failed))


def test_null_firth_convergence_precedes_the_increase_failure() -> None:
    """Accept convergence from iteration two onward before applying the heuristic."""
    for convergence_iteration in (2, 27):
        score_maxima = [float(score_maximum) for score_maximum in range(convergence_iteration)]
        observed_states = compare_score_history_sequence_to_regenie(
            score_maxima=score_maxima,
            convergence_iteration=convergence_iteration,
            check_score_increase=True,
        )

        assert not bool(np.asarray(observed_states[-1].failed))


def test_null_firth_score_history_handles_nonfinite_scores_deterministically() -> None:
    """Match scalar comparison semantics for NaN and positive infinity."""
    nan_states = compare_score_history_sequence_to_regenie(
        score_maxima=[1.0, float("nan")],
        convergence_iteration=None,
        check_score_increase=True,
    )
    infinity_states = compare_score_history_sequence_to_regenie(
        score_maxima=[1.0, float("inf")],
        convergence_iteration=None,
        check_score_increase=True,
    )

    assert np.isnan(float(np.asarray(nan_states[-1].previous_score_maximum)))
    assert int(np.asarray(nan_states[-1].score_increase_count)) == 0
    assert int(np.asarray(infinity_states[-1].score_increase_count)) == 1
    assert not bool(np.asarray(nan_states[-1].failed))
    assert not bool(np.asarray(infinity_states[-1].failed))


def test_null_firth_components_match_independent_numpy_formula() -> None:
    """Validate information, penalized deviance, leverage, and modified score."""
    fixture = build_null_firth_fixture()
    reference = compute_null_firth_component_reference(fixture)

    observed = regenie2_binary_firth_null.compute_null_firth_components(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        coefficients=jnp.asarray(fixture.coefficients),
    )

    tests.numerical.assert_absolute_difference_less_than(
        observed.information_cholesky_factor,
        reference.information_cholesky_factor,
        1.0e-12,
    )
    tests.numerical.assert_absolute_difference_less_than(observed.deviance, reference.deviance, 1.0e-12)
    tests.numerical.assert_absolute_difference_less_than(observed.modified_score, reference.modified_score, 1.0e-12)
    assert bool(np.asarray(observed.valid))


def test_null_firth_line_search_accepts_a_deviance_decreasing_newton_step() -> None:
    """Accept the first full or halved Newton step that improves the objective."""
    fixture = build_null_firth_fixture()
    current_components = regenie2_binary_firth_null.compute_null_firth_components(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        coefficients=jnp.asarray(fixture.coefficients),
    )
    information = np.asarray(current_components.information_cholesky_factor)
    information = information @ information.T
    coefficient_step = np.linalg.solve(information, np.asarray(current_components.modified_score))

    observed = regenie2_binary_firth_null.run_null_firth_line_search(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        current_coefficients=jnp.asarray(fixture.coefficients),
        current_deviance=current_components.deviance,
        coefficient_step=jnp.asarray(coefficient_step),
        maximum_attempts=8,
        step_halving_scale=0.5,
    )

    assert bool(np.asarray(observed.accepted))
    assert bool(np.asarray(observed.valid))
    assert float(np.asarray(observed.deviance)) < float(np.asarray(current_components.deviance))


def test_null_firth_zero_attempt_line_search_retains_trusted_state() -> None:
    """Leave coefficients and deviance untouched when no attempt is authorized."""
    fixture = build_null_firth_fixture()
    current_components = regenie2_binary_firth_null.compute_null_firth_components(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        coefficients=jnp.asarray(fixture.coefficients),
    )

    observed = regenie2_binary_firth_null.run_null_firth_line_search(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        current_coefficients=jnp.asarray(fixture.coefficients),
        current_deviance=current_components.deviance,
        coefficient_step=jnp.asarray([1.0, -1.0]),
        maximum_attempts=0,
        step_halving_scale=0.5,
    )

    tests.numerical.assert_absolute_difference_less_than(observed.coefficients, fixture.coefficients, 1.0e-15)
    tests.numerical.assert_absolute_difference_less_than(observed.deviance, current_components.deviance, 1.0e-15)
    assert not bool(np.asarray(observed.accepted))
    assert bool(np.asarray(observed.valid))


def test_null_firth_single_attempt_converges_to_a_small_modified_score() -> None:
    """Converge on a regular fixture under the production numerical policy."""
    fixture = build_null_firth_fixture()
    observed = regenie2_binary_firth_null.fit_covariate_only_firth_null_model_once(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        initial_coefficients=jnp.zeros((fixture.covariate_matrix.shape[1],), dtype=jnp.float64),
        maximum_iterations=100,
        maximum_step_size=PRODUCTION_NULL_FIRTH_MAXIMUM_STEP_SIZE,
        tolerance=PRODUCTION_NULL_FIRTH_GRADIENT_TOLERANCE,
        line_search_maximum_attempts=25,
        line_search_step_halving_scale=0.5,
        check_score_increase=True,
    )
    terminal_components = regenie2_binary_firth_null.compute_null_firth_components(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        coefficients=observed.coefficients,
    )

    assert bool(np.asarray(observed.converged))
    assert bool(np.asarray(terminal_components.valid))
    assert (
        float(np.max(np.abs(np.asarray(terminal_components.modified_score)))) < PRODUCTION_NULL_FIRTH_GRADIENT_TOLERANCE
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.penalized_log_likelihood,
        -0.5 * np.asarray(terminal_components.deviance),
        1.0e-12,
    )


def test_null_firth_zero_iteration_budget_returns_failure_without_moving_start() -> None:
    """Make exhaustion explicit and never present an untrusted likelihood."""
    fixture = build_null_firth_fixture()
    observed = regenie2_binary_firth_null.fit_covariate_only_firth_null_model_once(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        initial_coefficients=jnp.asarray(fixture.coefficients),
        maximum_iterations=0,
        maximum_step_size=5.0,
        tolerance=1.0e-8,
        line_search_maximum_attempts=20,
        line_search_step_halving_scale=0.5,
        check_score_increase=True,
    )

    assert not bool(np.asarray(observed.converged))
    assert bool(np.isnan(np.asarray(observed.penalized_log_likelihood)))
    tests.numerical.assert_absolute_difference_less_than(observed.coefficients, fixture.coefficients, 1.0e-15)


def test_null_firth_nonfinite_input_returns_failure_without_moving_start() -> None:
    """Reject an invalid initial state without exposing a trusted likelihood."""
    fixture = build_null_firth_fixture()
    covariate_matrix = fixture.covariate_matrix.copy()
    covariate_matrix[0, 0] = np.nan
    observed = regenie2_binary_firth_null.fit_covariate_only_firth_null_model_once(
        covariate_matrix=jnp.asarray(covariate_matrix),
        phenotype_vector=jnp.asarray(fixture.phenotype_vector),
        loco_offset=jnp.asarray(fixture.loco_offset),
        initial_coefficients=jnp.asarray(fixture.coefficients),
        maximum_iterations=100,
        maximum_step_size=PRODUCTION_NULL_FIRTH_MAXIMUM_STEP_SIZE,
        tolerance=PRODUCTION_NULL_FIRTH_GRADIENT_TOLERANCE,
        line_search_maximum_attempts=25,
        line_search_step_halving_scale=0.5,
        check_score_increase=True,
    )

    assert not bool(np.asarray(observed.converged))
    assert bool(np.isnan(np.asarray(observed.penalized_log_likelihood)))
    tests.numerical.assert_absolute_difference_less_than(observed.coefficients, fixture.coefficients, 1.0e-15)


def test_null_firth_wrapper_wires_and_lazily_selects_all_attempt_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise starts, limits, checks, ordering, and terminal fallback behavior."""
    fixture = build_null_firth_fixture()
    kernel_config = build_null_firth_policy_config()
    covariate_matrix = jnp.asarray(fixture.covariate_matrix, dtype=jnp.float64)
    phenotype_vector = jnp.asarray(fixture.phenotype_vector, dtype=jnp.float64)
    loco_offset = jnp.asarray(fixture.loco_offset, dtype=jnp.float64)
    initial_coefficients = jnp.asarray(fixture.coefficients, dtype=jnp.float64)
    zero_start_coefficients = jnp.zeros_like(initial_coefficients).at[0].set(-jnp.mean(loco_offset))
    null_firth_config = kernel_config.null_firth
    fallback_maximum_iterations = null_firth_config.maximum_iterations * null_firth_config.fallback_iteration_multiplier
    fallback_maximum_step_size = null_firth_config.maximum_step_size / null_firth_config.fallback_step_divisor

    for first_converged_attempt_index in range(1, 6):

        def probe_attempt_policy(
            *,
            covariate_matrix: jax.Array,
            phenotype_vector: jax.Array,
            loco_offset: jax.Array,
            initial_coefficients: jax.Array,
            maximum_iterations: int | jax.Array,
            maximum_step_size: float | jax.Array,
            tolerance: float | jax.Array,
            line_search_maximum_attempts: int | jax.Array,
            line_search_step_halving_scale: float | jax.Array,
            check_score_increase: bool | jax.Array,
        ) -> regenie2_binary_firth_types.NullFirthFitResult:
            shared_policy_matches = (
                jnp.all(covariate_matrix == fixture.covariate_matrix)
                & jnp.all(phenotype_vector == fixture.phenotype_vector)
                & jnp.all(loco_offset == fixture.loco_offset)
                & (jnp.asarray(tolerance, dtype=jnp.float64) == null_firth_config.gradient_tolerance)
                & (
                    jnp.asarray(line_search_maximum_attempts, dtype=jnp.int32)
                    == null_firth_config.line_search_maximum_attempts
                )
                & (
                    jnp.asarray(line_search_step_halving_scale, dtype=jnp.float64)
                    == null_firth_config.step_halving_scale
                )
            )
            initial_start_matches = jnp.all(initial_coefficients == fixture.coefficients)
            zero_start_matches = jnp.all(initial_coefficients == zero_start_coefficients)
            standard_limits_match = (
                jnp.asarray(maximum_iterations, dtype=jnp.int32) == null_firth_config.maximum_iterations
            ) & (jnp.asarray(maximum_step_size, dtype=jnp.float64) == null_firth_config.maximum_step_size)
            fallback_limits_match = (
                jnp.asarray(maximum_iterations, dtype=jnp.int32) == fallback_maximum_iterations
            ) & (jnp.asarray(maximum_step_size, dtype=jnp.float64) == fallback_maximum_step_size)
            score_increase_check = jnp.asarray(check_score_increase, dtype=jnp.bool_)
            first_attempt_matches = (
                shared_policy_matches & initial_start_matches & standard_limits_match & score_increase_check
            )
            second_attempt_matches = (
                shared_policy_matches & zero_start_matches & standard_limits_match & score_increase_check
            )
            third_attempt_matches = (
                shared_policy_matches & zero_start_matches & fallback_limits_match & score_increase_check
            )
            fourth_attempt_matches = (
                shared_policy_matches & initial_start_matches & fallback_limits_match & (~score_increase_check)
            )
            attempt_index = (
                first_attempt_matches.astype(jnp.int32)
                + second_attempt_matches.astype(jnp.int32) * 2
                + third_attempt_matches.astype(jnp.int32) * 3
                + fourth_attempt_matches.astype(jnp.int32) * 4
            )
            converged = attempt_index >= jnp.asarray(first_converged_attempt_index, dtype=jnp.int32)
            return regenie2_binary_firth_types.NullFirthFitResult(
                coefficients=jnp.full_like(initial_coefficients, attempt_index),
                penalized_log_likelihood=jnp.asarray(attempt_index, dtype=jnp.float64),
                converged=converged,
            )

        monkeypatch.setattr(
            regenie2_binary_firth_null,
            "fit_covariate_only_firth_null_model_once",
            probe_attempt_policy,
        )
        observed = regenie2_binary_firth_null.fit_covariate_only_firth_null_model(
            covariate_matrix,
            phenotype_vector,
            loco_offset,
            initial_coefficients,
            kernel_config,
        )
        expected_attempt_index = min(first_converged_attempt_index, 4)
        expected_convergence = first_converged_attempt_index <= 4

        tests.numerical.assert_absolute_difference_less_than(
            observed.coefficients,
            np.full_like(fixture.coefficients, expected_attempt_index),
            1.0e-15,
        )
        assert bool(np.asarray(observed.converged)) is expected_convergence
        if expected_convergence:
            assert float(np.asarray(observed.penalized_log_likelihood)) == expected_attempt_index
        else:
            assert bool(np.isnan(np.asarray(observed.penalized_log_likelihood)))
