"""Correctness tests for scalar approximate-Firth primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

import tests.numerical
from g.compute.regenie2_binary import candidates as regenie2_binary_candidates
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import logistic as regenie2_binary_logistic
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary.firth import scalar_approx as regenie2_binary_firth_scalar_approx
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types
from g.compute.regenie2_binary.firth.batch import compute as regenie2_binary_firth_batch_compute
from g.compute.regenie2_binary.firth.batch import prepare as regenie2_binary_firth_batch_prepare

PSEUDO_RECURRENCE_ABSOLUTE_TOLERANCE = 5.0e-7


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


@dataclass(frozen=True)
class IndependentPseudoFirthResult:
    """Independent upstream-style pseudo-Firth recurrence result."""

    beta: float
    standard_error: float
    chi_squared: float
    outer_iteration_count: int
    converged: bool


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
    pseudo_maximum_iterations: int = 20,
    gradient_tolerance: float = 1.0e-8,
    pseudo_inner_maximum_iterations: int = 30,
    sparse_carrier_dosage_threshold: float = 0.5,
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
            gradient_tolerance=gradient_tolerance,
            maximum_step_size=5.0,
            pseudo_maximum_iterations=pseudo_maximum_iterations,
            pseudo_inner_maximum_iterations=pseudo_inner_maximum_iterations,
            line_search_maximum_attempts=20,
            sparse_carrier_dosage_threshold=sparse_carrier_dosage_threshold,
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
) -> ScalarFirthComponentReference:
    """Evaluate the scalar penalized likelihood and adjusted score in NumPy."""
    linear_predictor = fixture.offset_vector + fixture.genotype_vector * fixture.beta
    probability = np.reciprocal(1.0 + np.exp(-linear_predictor))
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
    return ScalarFirthComponentReference(
        genotype_information=genotype_information,
        score_adjustment=score_adjustment,
        penalized_deviance=fixture.non_active_deviance + active_deviance - math.log(genotype_information),
        score=score,
    )


def fit_independent_upstream_pseudo_firth(
    *,
    fixture: ScalarFirthFixture,
    maximum_iterations: int,
    tolerance: float,
    inner_maximum_iterations: int,
    maximum_step_size: float,
) -> IndependentPseudoFirthResult:
    """Run the upstream scalar pseudo-Firth recurrence independently in NumPy."""
    initial_fixture = ScalarFirthFixture(
        phenotype_vector=fixture.phenotype_vector,
        genotype_vector=fixture.genotype_vector,
        offset_vector=fixture.offset_vector,
        active_sample_mask=fixture.active_sample_mask,
        non_active_deviance=fixture.non_active_deviance,
        beta=0.0,
    )
    initial_components = compute_scalar_firth_component_reference(initial_fixture)
    null_probability = np.reciprocal(1.0 + np.exp(-fixture.offset_vector))
    negative_log_likelihood = -np.where(
        fixture.phenotype_vector > 0.5,
        np.log(null_probability),
        np.log1p(-null_probability),
    )
    full_null_deviance = 2.0 * float(np.sum(negative_log_likelihood, dtype=np.float64))
    deviance_null = full_null_deviance - math.log(initial_components.genotype_information)
    beta = 0.0
    beta_iteration_14 = 0.0
    components = initial_components
    converged = False
    outer_iteration_count = 0
    genotype_vector_compute = fixture.genotype_vector.astype(np.float32)
    phenotype_vector_compute = fixture.phenotype_vector.astype(np.float32)
    offset_vector_compute = fixture.offset_vector.astype(np.float32)

    for outer_iteration_count in range(1, maximum_iterations + 1):
        if abs(components.score) < tolerance and outer_iteration_count >= 2:
            converged = True
            break
        if outer_iteration_count == 14:
            beta_iteration_14 = beta
        if outer_iteration_count == 15 and abs(beta - beta_iteration_14) > 0.1:
            raise AssertionError("Independent pseudo-Firth recurrence failed the iteration-15 guard.")

        score = components.score
        genotype_information = components.genotype_information
        previous_step_size = math.inf
        for _inner_iteration in range(inner_maximum_iterations):
            step_size = score / genotype_information
            absolute_step_size = abs(step_size)
            if absolute_step_size > previous_step_size:
                raise AssertionError("Independent pseudo-logistic step increased.")
            beta += step_size / max(absolute_step_size / maximum_step_size, 1.0)
            linear_predictor_compute = offset_vector_compute + genotype_vector_compute * np.asarray(
                beta,
                dtype=np.float32,
            )
            probability_vector_compute = np.reciprocal(
                np.asarray(1.0, dtype=np.float32) + np.exp(-linear_predictor_compute)
            )
            score = float(
                np.sum(
                    np.where(
                        fixture.active_sample_mask,
                        genotype_vector_compute * (phenotype_vector_compute - probability_vector_compute),
                        np.asarray(0.0, dtype=np.float32),
                    ).astype(np.float64),
                    dtype=np.float64,
                )
                + components.score_adjustment
            )
            weight_vector_compute = probability_vector_compute * (
                np.asarray(1.0, dtype=np.float32) - probability_vector_compute
            )
            genotype_information = float(
                np.sum(
                    np.where(
                        fixture.active_sample_mask,
                        genotype_vector_compute * genotype_vector_compute * weight_vector_compute,
                        np.asarray(0.0, dtype=np.float32),
                    ).astype(np.float64),
                    dtype=np.float64,
                )
            )
            if abs(score) < tolerance:
                break
            previous_step_size = absolute_step_size

        components = compute_scalar_firth_component_reference(
            ScalarFirthFixture(
                phenotype_vector=fixture.phenotype_vector,
                genotype_vector=fixture.genotype_vector,
                offset_vector=fixture.offset_vector,
                active_sample_mask=fixture.active_sample_mask,
                non_active_deviance=fixture.non_active_deviance,
                beta=beta,
            )
        )

    return IndependentPseudoFirthResult(
        beta=beta,
        standard_error=math.sqrt(1.0 / components.genotype_information),
        chi_squared=deviance_null - components.penalized_deviance,
        outer_iteration_count=outer_iteration_count,
        converged=converged,
    )


def build_test_firth_chromosome_state(
    *,
    phenotype_vector: npt.NDArray[np.float64],
    offset_vector: npt.NDArray[np.float64],
    square_root_weight: npt.NDArray[np.float32],
    weighted_genotype_projection_matrix: npt.NDArray[np.float32],
) -> regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState:
    """Build explicit production-shaped state for raw candidate preparation."""
    sample_count = phenotype_vector.size
    null_probability = regenie2_binary_logistic.compute_regenie_logistic_probability(jnp.asarray(offset_vector))
    full_null_deviance = regenie2_binary_logistic.compute_logistic_deviance(
        jnp.asarray(phenotype_vector),
        null_probability,
        jnp.ones((sample_count,), dtype=jnp.bool_),
    )
    return regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState(
        score_state=regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState(
            score_right_hand_matrix=jnp.zeros((1, sample_count), dtype=jnp.float32),
            bernoulli_weight=jnp.square(jnp.asarray(square_root_weight[None, :], dtype=jnp.float32)),
            null_logistic_converged=jnp.ones((1,), dtype=jnp.bool_),
        ),
        phenotype_matrix=jnp.asarray(phenotype_vector[None, :], dtype=jnp.float32),
        null_firth_offset_matrix=jnp.asarray(offset_vector[None, :], dtype=jnp.float64),
        square_root_weight=jnp.asarray(square_root_weight[None, :], dtype=jnp.float32),
        weighted_genotype_projection_matrix=jnp.asarray(
            weighted_genotype_projection_matrix[None, :, :],
            dtype=jnp.float32,
        ),
        full_null_deviance=full_null_deviance[None],
        null_firth_penalized_log_likelihood=jnp.zeros((1,), dtype=jnp.float64),
    )


def prepare_raw_scalar_firth_candidates(
    *,
    raw_genotype_matrix_by_variant: npt.NDArray[np.float32],
    phenotype_vector: npt.NDArray[np.float64],
    offset_vector: npt.NDArray[np.float64],
    square_root_weight: npt.NDArray[np.float32],
    weighted_genotype_projection_matrix: npt.NDArray[np.float32],
    sparse_candidate_mask: npt.NDArray[np.bool_],
    native_genotype_mean: npt.NDArray[np.float32] | None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> regenie2_binary_candidates.ScalarFirthCandidateBatchInputs:
    """Prepare raw dosages through production orientation and residualization."""
    variant_count = raw_genotype_matrix_by_variant.shape[0]
    chromosome_state = build_test_firth_chromosome_state(
        phenotype_vector=phenotype_vector,
        offset_vector=offset_vector,
        square_root_weight=square_root_weight,
        weighted_genotype_projection_matrix=weighted_genotype_projection_matrix,
    )
    selected_rows = regenie2_binary_firth_batch_prepare.SelectedMultiFirthCandidateRows(
        flat_active_mask=jnp.ones((variant_count,), dtype=jnp.bool_),
        flat_trait_indices=jnp.zeros((variant_count,), dtype=jnp.int32),
        flat_variant_indices=jnp.arange(variant_count, dtype=jnp.int32),
        genotype_matrix_by_variant=jnp.asarray(raw_genotype_matrix_by_variant),
    )
    return regenie2_binary_firth_batch_prepare.prepare_scalar_firth_candidate_batch(
        chromosome_state=chromosome_state,
        selected_rows=selected_rows,
        sparse_candidate_mask=jnp.asarray(sparse_candidate_mask),
        order_candidates=False,
        kernel_config=kernel_config,
        native_genotype_mean=None if native_genotype_mean is None else jnp.asarray(native_genotype_mean),
    )


def scalar_fixture_from_prepared_candidate(
    *,
    candidate_inputs: regenie2_binary_candidates.ScalarFirthCandidateBatchInputs,
    variant_index: int,
) -> ScalarFirthFixture:
    """Build an independent recurrence fixture from production-prepared arrays."""
    phenotype_vector = np.asarray(candidate_inputs.lanes.phenotype_matrix[0], dtype=np.float64)
    genotype_vector = np.asarray(candidate_inputs.genotype_matrix_by_variant[variant_index], dtype=np.float64)
    offset_vector = np.asarray(candidate_inputs.null_firth_offset_matrix[0], dtype=np.float64)
    active_sample_mask = np.asarray(candidate_inputs.carrier_sample_mask[variant_index], dtype=np.bool_)
    null_probability = np.reciprocal(1.0 + np.exp(-offset_vector))
    negative_log_likelihood = -np.where(
        phenotype_vector > 0.5,
        np.log(null_probability),
        np.log1p(-null_probability),
    )
    full_null_deviance = 2.0 * float(np.sum(negative_log_likelihood, dtype=np.float64))
    active_null_deviance = 2.0 * float(
        np.sum(
            np.where(active_sample_mask, negative_log_likelihood, 0.0),
            dtype=np.float64,
        )
    )
    return ScalarFirthFixture(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=full_null_deviance - active_null_deviance,
        beta=0.0,
    )


def assert_pseudo_terminal_matches_independent_recurrence(
    *,
    observed_beta: jax.Array,
    observed_standard_error: jax.Array,
    observed_chi_squared: jax.Array,
    reference: IndependentPseudoFirthResult,
) -> None:
    """Compare every numerical pseudo-Firth terminal field strictly."""
    tests.numerical.assert_absolute_difference_less_than(
        observed_beta,
        reference.beta,
        PSEUDO_RECURRENCE_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed_standard_error,
        reference.standard_error,
        PSEUDO_RECURRENCE_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed_chi_squared,
        reference.chi_squared,
        PSEUDO_RECURRENCE_ABSOLUTE_TOLERANCE,
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
    assert int(np.asarray(observed.sparse_pseudo_maximum_iterations)) == 15
    assert int(np.asarray(observed.newton_raphson_maximum_iterations)) == 15
    assert int(np.asarray(observed.pseudo_inner_maximum_iterations)) == 30
    assert int(np.asarray(observed.line_search_maximum_attempts)) == 20
    assert not observed.use_cuda_components


def test_scalar_solver_parameter_budget_retains_sparse_half_above_dense_cap() -> None:
    """Match upstream's 50 dense versus 125 sparse pseudo split for total 250."""
    observed = regenie2_binary_firth_scalar_approx.build_scalar_approximate_firth_solver_parameters(
        build_binary_kernel_config(
            approximate_firth_maximum_iterations=250,
            pseudo_maximum_iterations=50,
        )
    )

    assert int(np.asarray(observed.pseudo_maximum_iterations)) == 50
    assert int(np.asarray(observed.sparse_pseudo_maximum_iterations)) == 125
    assert int(np.asarray(observed.newton_raphson_maximum_iterations)) == 125


def test_raw_dosage_preparation_applies_flip_and_strict_carrier_threshold() -> None:
    """Classify carriers from oriented mean-imputed dosage at the strict boundary."""
    threshold = np.float32(1.0e-4)
    above_threshold = np.nextafter(threshold, np.float32(np.inf))
    raw_genotype_matrix_by_variant = np.asarray(
        [
            [threshold, above_threshold, 0.75, 0.0],
            [2.0, 1.5, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    kernel_config = build_binary_kernel_config(sparse_carrier_dosage_threshold=float(threshold))
    observed = prepare_raw_scalar_firth_candidates(
        raw_genotype_matrix_by_variant=raw_genotype_matrix_by_variant,
        phenotype_vector=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
        offset_vector=np.zeros((4,), dtype=np.float64),
        square_root_weight=np.full((4,), 0.5, dtype=np.float32),
        weighted_genotype_projection_matrix=np.empty((0, 4), dtype=np.float32),
        sparse_candidate_mask=np.ones((2,), dtype=np.bool_),
        native_genotype_mean=np.asarray([0.2, 1.5], dtype=np.float32),
        kernel_config=kernel_config,
    )

    np.testing.assert_array_equal(np.asarray(observed.genotype_flip_mask), np.asarray([False, True]))
    np.testing.assert_array_equal(
        np.asarray(observed.carrier_sample_mask),
        np.asarray(
            [
                [False, True, True, False],
                [False, True, True, False],
            ]
        ),
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.genotype_matrix_by_variant,
        np.asarray(
            [
                [threshold, above_threshold, 0.75, 0.0],
                [0.0, 0.5, 2.0, 0.0],
            ],
            dtype=np.float32,
        ),
        1.0e-7,
    )


def test_raw_sparse_dispatch_uses_uncapped_budget_at_compact_boundary() -> None:
    """Route 64/65 raw-dosage carriers to pseudo convergence after iteration 50."""
    sample_count = 130
    raw_genotype_matrix_by_variant = np.zeros((2, sample_count), dtype=np.float32)
    raw_genotype_matrix_by_variant[0, :64] = 0.75
    raw_genotype_matrix_by_variant[1, :65] = 0.75
    phenotype_vector = np.zeros((sample_count,), dtype=np.float64)
    phenotype_vector[1:65] = np.arange(1, 65, dtype=np.int64) % 2
    offset_vector = np.full((sample_count,), -1.0, dtype=np.float64)
    square_root_weight = np.full((sample_count,), 0.5, dtype=np.float32)
    square_root_weight[0] = 0.001
    leading_projection = np.float32(-0.00428205128205128)
    trailing_projection = np.float32(
        math.sqrt((1.0 - float(leading_projection * leading_projection)) / (sample_count - 1))
    )
    weighted_genotype_projection_matrix = np.full(
        (1, sample_count),
        trailing_projection,
        dtype=np.float32,
    )
    weighted_genotype_projection_matrix[0, 0] = leading_projection
    kernel_config = build_binary_kernel_config(
        approximate_firth_maximum_iterations=250,
        pseudo_maximum_iterations=50,
        gradient_tolerance=2.5e-4,
        pseudo_inner_maximum_iterations=25,
        sparse_carrier_dosage_threshold=1.0e-4,
    )
    candidate_inputs = prepare_raw_scalar_firth_candidates(
        raw_genotype_matrix_by_variant=raw_genotype_matrix_by_variant,
        phenotype_vector=phenotype_vector,
        offset_vector=offset_vector,
        square_root_weight=square_root_weight,
        weighted_genotype_projection_matrix=weighted_genotype_projection_matrix,
        sparse_candidate_mask=np.ones((2,), dtype=np.bool_),
        native_genotype_mean=None,
        kernel_config=kernel_config,
    )
    carrier_counts = np.sum(np.asarray(candidate_inputs.carrier_sample_mask), axis=1)
    raw_minor_allele_counts = np.sum(raw_genotype_matrix_by_variant, axis=1, dtype=np.float64)
    raw_zero_counts = np.sum(raw_genotype_matrix_by_variant <= np.float32(1.0e-4), axis=1)
    solver_parameters = regenie2_binary_firth_scalar_approx.build_scalar_approximate_firth_solver_parameters(
        kernel_config
    )

    np.testing.assert_array_equal(carrier_counts, np.asarray([64, 65]))
    np.testing.assert_array_equal(raw_minor_allele_counts, np.asarray([48.0, 48.75]))
    np.testing.assert_array_equal(raw_zero_counts, np.asarray([66, 65]))
    assert bool(np.all(raw_minor_allele_counts < 50.0))
    assert bool(np.all(2 * raw_zero_counts >= sample_count))
    np.testing.assert_array_equal(np.asarray(candidate_inputs.genotype_flip_mask), np.asarray([False, False]))
    np.testing.assert_array_equal(np.asarray(candidate_inputs.sparse_correction_mask), np.asarray([True, True]))
    assert carrier_counts[0] == regenie2_binary_firth_batch_compute.SPARSE_FIRTH_CARRIER_CAPACITY
    assert carrier_counts[1] == regenie2_binary_firth_batch_compute.SPARSE_FIRTH_CARRIER_CAPACITY + 1

    compact_carrier_indices = np.flatnonzero(np.asarray(candidate_inputs.carrier_sample_mask[0]))
    compact_initial_state = regenie2_binary_firth_scalar_approx.initialize_compact_carrier_regenie_approximate_firth(
        phenotype_vector=jnp.take(candidate_inputs.lanes.phenotype_matrix[0], compact_carrier_indices),
        genotype_vector=jnp.take(candidate_inputs.genotype_matrix_by_variant[0], compact_carrier_indices),
        offset_vector=jnp.take(candidate_inputs.null_firth_offset_matrix[0], compact_carrier_indices),
        active_carrier_slot_mask=jnp.ones((64,), dtype=jnp.bool_),
        full_null_deviance=candidate_inputs.full_null_deviance[0],
        solver_parameters=solver_parameters,
    )
    masked_initial_state = regenie2_binary_firth_scalar_approx.initialize_single_variant_regenie_approximate_firth(
        phenotype_vector=candidate_inputs.lanes.phenotype_matrix[0],
        genotype_vector=candidate_inputs.genotype_matrix_by_variant[1],
        offset_vector=candidate_inputs.null_firth_offset_matrix[0],
        carrier_sample_mask=candidate_inputs.carrier_sample_mask[1],
        full_null_deviance=candidate_inputs.full_null_deviance[1],
        sparse_correction=jnp.asarray(1, dtype=jnp.bool_),
        solver_parameters=solver_parameters,
    )
    dense_initial_state = regenie2_binary_firth_scalar_approx.initialize_single_variant_regenie_approximate_firth(
        phenotype_vector=candidate_inputs.lanes.phenotype_matrix[0],
        genotype_vector=candidate_inputs.genotype_matrix_by_variant[1],
        offset_vector=candidate_inputs.null_firth_offset_matrix[0],
        carrier_sample_mask=candidate_inputs.carrier_sample_mask[1],
        full_null_deviance=candidate_inputs.full_null_deviance[1],
        sparse_correction=jnp.asarray(0, dtype=jnp.bool_),
        solver_parameters=solver_parameters,
    )

    assert int(np.asarray(compact_initial_state.solver_parameters.pseudo_maximum_iterations)) == 125
    assert int(np.asarray(masked_initial_state.solver_parameters.pseudo_maximum_iterations)) == 125
    assert int(np.asarray(dense_initial_state.solver_parameters.pseudo_maximum_iterations)) == 50

    observed = regenie2_binary_firth_batch_compute.compute_scalar_firth_multi_variantwise_fixed_batches(
        candidate_inputs=candidate_inputs,
        fallback_count=jnp.asarray(2, dtype=jnp.int32),
        firth_batch_size=1,
        kernel_config=kernel_config,
    )
    for variant_index in range(2):
        fixture = scalar_fixture_from_prepared_candidate(
            candidate_inputs=candidate_inputs,
            variant_index=variant_index,
        )
        sparse_reference = fit_independent_upstream_pseudo_firth(
            fixture=fixture,
            maximum_iterations=125,
            tolerance=kernel_config.approximate_firth.gradient_tolerance,
            inner_maximum_iterations=kernel_config.approximate_firth.pseudo_inner_maximum_iterations,
            maximum_step_size=kernel_config.approximate_firth.maximum_step_size,
        )
        dense_capped_reference = fit_independent_upstream_pseudo_firth(
            fixture=fixture,
            maximum_iterations=50,
            tolerance=kernel_config.approximate_firth.gradient_tolerance,
            inner_maximum_iterations=kernel_config.approximate_firth.pseudo_inner_maximum_iterations,
            maximum_step_size=kernel_config.approximate_firth.maximum_step_size,
        )
        capped_initial_state = (
            regenie2_binary_firth_scalar_approx.initialize_scalar_approximate_firth_with_active_samples(
                phenotype_vector=jnp.asarray(fixture.phenotype_vector),
                genotype_vector=jnp.asarray(fixture.genotype_vector),
                offset_vector=jnp.asarray(fixture.offset_vector),
                active_sample_mask=jnp.asarray(fixture.active_sample_mask),
                full_null_deviance=candidate_inputs.full_null_deviance[variant_index],
                non_active_deviance=jnp.asarray(fixture.non_active_deviance),
                solver_parameters=solver_parameters,
            )
        )
        capped_observed = regenie2_binary_firth_scalar_approx.run_initialized_scalar_pseudo_firth_solver(
            capped_initial_state
        )

        assert 50 < sparse_reference.outer_iteration_count <= 125
        assert sparse_reference.converged
        assert dense_capped_reference.outer_iteration_count == 50
        assert not dense_capped_reference.converged
        assert bool(np.asarray(observed.valid_mask[variant_index]))
        assert not bool(np.asarray(capped_observed.valid_mask))
        assert_pseudo_terminal_matches_independent_recurrence(
            observed_beta=observed.beta[variant_index],
            observed_standard_error=observed.standard_error[variant_index],
            observed_chi_squared=observed.chi_squared[variant_index],
            reference=sparse_reference,
        )
        assert_pseudo_terminal_matches_independent_recurrence(
            observed_beta=capped_observed.beta,
            observed_standard_error=capped_observed.standard_error,
            observed_chi_squared=capped_observed.chi_squared,
            reference=dense_capped_reference,
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
