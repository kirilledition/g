"""Correctness tests for binary score and candidate-planning kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

import tests.numerical
from g import types
from g.compute.common import genotype
from g.compute.regenie2_binary import candidates as regenie2_binary_candidates
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import correction as regenie2_binary_correction
from g.compute.regenie2_binary import logistic as regenie2_binary_logistic
from g.compute.regenie2_binary import null_logistic as regenie2_binary_null_logistic
from g.compute.regenie2_binary import score as regenie2_binary_score
from g.compute.regenie2_binary import state as regenie2_binary_state

# The oracle evaluates the null fit and score algebra in float64, while the
# production path performs its null fit and sample reductions in float32.
# These exclusive bounds are approximately twice the largest observed errors
# (5.94e-6, 1.26e-6, 1.05e-5, and 2.74e-6 respectively), leaving architecture
# headroom at the expected float32 forward-error scale without masking a change
# in an individual statistic.
BINARY_BETA_ABSOLUTE_TOLERANCE = 1.2e-5
BINARY_STANDARD_ERROR_ABSOLUTE_TOLERANCE = 2.6e-6
BINARY_CHI_SQUARED_ABSOLUTE_TOLERANCE = 2.2e-5
BINARY_LOG10_P_VALUE_ABSOLUTE_TOLERANCE = 5.6e-6

# Packed delivery and explicit decoding feed the same float32 score kernel.
# Separate bounds keep a regression in one statistic from borrowing the wider
# chi-square allowance.
BINARY_PACKED_BETA_ABSOLUTE_TOLERANCE = 6.0e-6
BINARY_PACKED_STANDARD_ERROR_ABSOLUTE_TOLERANCE = 2.0e-6
BINARY_PACKED_CHI_SQUARED_ABSOLUTE_TOLERANCE = 6.2e-6
BINARY_PACKED_LOG10_P_VALUE_ABSOLUTE_TOLERANCE = 2.0e-6


@dataclass(frozen=True)
class BinaryFixture:
    """Deterministic binary-score inputs with two independent traits."""

    covariate_matrix: npt.NDArray[np.float64]
    phenotype_matrix: npt.NDArray[np.float64]
    loco_offset_matrix: npt.NDArray[np.float64]
    genotype_matrix_by_variant: npt.NDArray[np.float64]


@dataclass(frozen=True)
class NullLogisticReference:
    """Independent null-logistic fit quantities used by score references."""

    coefficients: npt.NDArray[np.float64]
    probability: npt.NDArray[np.float64]
    weight: npt.NDArray[np.float64]
    score_residual: npt.NDArray[np.float64]
    weighted_projection_matrix: npt.NDArray[np.float64]


@dataclass(frozen=True)
class BinaryReferenceResult:
    """Independent NumPy binary score-test result."""

    beta: npt.NDArray[np.float64]
    standard_error: npt.NDArray[np.float64]
    chi_squared: npt.NDArray[np.float64]
    log10_p_value: npt.NDArray[np.float64]


def build_binary_score_config() -> regenie2_binary_config.BinaryScoreConfig:
    """Build one stable numerical policy shared by binary tests."""
    return regenie2_binary_config.BinaryScoreConfig(
        numerical=regenie2_binary_config.BinaryNumericalConfig(
            minimum_probability=1.0e-7,
            minimum_variance=1.0e-8,
            relative_variance_tolerance=1.0e-7,
        ),
        null_logistic=regenie2_binary_config.BinaryNullLogisticConfig(
            maximum_iterations=100,
            coefficient_tolerance=1.0e-6,
        ),
    )


def build_binary_fixture() -> BinaryFixture:
    """Build binary traits and variants spanning both allele orientations."""
    return BinaryFixture(
        covariate_matrix=np.asarray(
            [
                [1.0, -1.8],
                [1.0, -1.4],
                [1.0, -1.0],
                [1.0, -0.6],
                [1.0, -0.2],
                [1.0, 0.2],
                [1.0, 0.6],
                [1.0, 1.0],
                [1.0, 1.4],
                [1.0, 1.8],
            ],
            dtype=np.float64,
        ),
        phenotype_matrix=np.asarray(
            [
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        loco_offset_matrix=np.asarray(
            [
                [0.05, -0.04, 0.02, 0.00, -0.03, 0.04, -0.02, 0.03, -0.01, 0.01],
                [-0.02, 0.01, 0.03, -0.04, 0.05, -0.01, 0.00, 0.02, -0.03, 0.04],
            ],
            dtype=np.float64,
        ),
        genotype_matrix_by_variant=np.asarray(
            [
                [0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0],
                [2.0, 2.0, 1.5, 2.0, 1.0, 1.5, 1.0, 0.0, 2.0, 1.0],
                [0.2, 1.1, 0.4, 1.8, 0.7, 1.5, 0.1, 1.2, 0.6, 1.9],
            ],
            dtype=np.float64,
        ),
    )


def compute_numpy_null_logistic(
    covariate_matrix: npt.NDArray[np.float64],
    phenotype_vector: npt.NDArray[np.float64],
    loco_offset: npt.NDArray[np.float64],
    kernel_config: regenie2_binary_config.BinaryScoreConfig,
) -> NullLogisticReference:
    """Fit the null model independently with NumPy IRLS."""
    coefficients = np.zeros(covariate_matrix.shape[1], dtype=np.float64)
    minimum_probability = kernel_config.numerical.minimum_probability
    minimum_variance = kernel_config.numerical.minimum_variance
    for _iteration_index in range(kernel_config.null_logistic.maximum_iterations):
        linear_predictor = covariate_matrix @ coefficients + loco_offset
        probability = np.reciprocal(1.0 + np.exp(-linear_predictor))
        fitted_probability = np.clip(probability, minimum_probability, 1.0 - minimum_probability)
        weight = np.maximum(fitted_probability * (1.0 - fitted_probability), minimum_variance)
        score = covariate_matrix.T @ (phenotype_vector - fitted_probability)
        information = (covariate_matrix.T * weight) @ covariate_matrix
        coefficient_delta = np.linalg.solve(
            information + np.eye(information.shape[0], dtype=np.float64) * minimum_variance,
            score,
        )
        coefficients += coefficient_delta
        if float(np.max(np.abs(coefficient_delta))) <= kernel_config.null_logistic.coefficient_tolerance:
            break

    linear_predictor = covariate_matrix @ coefficients + loco_offset
    probability = np.clip(
        np.reciprocal(1.0 + np.exp(-linear_predictor)),
        minimum_probability,
        1.0 - minimum_probability,
    )
    weight = np.maximum(probability * (1.0 - probability), minimum_variance)
    square_root_weight = np.sqrt(weight)
    weighted_covariate_matrix = square_root_weight[:, None] * covariate_matrix
    information = weighted_covariate_matrix.T @ weighted_covariate_matrix
    cholesky_factor = np.linalg.cholesky(
        information + np.eye(information.shape[0], dtype=np.float64) * minimum_variance
    )
    weighted_projection_matrix = np.linalg.solve(cholesky_factor, weighted_covariate_matrix.T)
    return NullLogisticReference(
        coefficients=coefficients,
        probability=probability,
        weight=weight,
        score_residual=phenotype_vector - probability,
        weighted_projection_matrix=weighted_projection_matrix,
    )


def compute_negative_log10_chi_square_probability(
    chi_squared: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Evaluate one-degree-of-freedom chi-square tails analytically."""
    flat_probabilities = [math.erfc(math.sqrt(float(value) / 2.0)) for value in chi_squared.ravel()]
    probabilities = np.asarray(flat_probabilities, dtype=np.float64).reshape(chi_squared.shape)
    return -np.log10(probabilities)


def compute_binary_score_reference(
    fixture: BinaryFixture,
    kernel_config: regenie2_binary_config.BinaryScoreConfig,
) -> BinaryReferenceResult:
    """Compute weighted binary score statistics without JAX production helpers."""
    trait_count = fixture.phenotype_matrix.shape[0]
    variant_count = fixture.genotype_matrix_by_variant.shape[0]
    beta = np.empty((trait_count, variant_count), dtype=np.float64)
    standard_error = np.empty_like(beta)
    chi_squared = np.empty_like(beta)
    genotype_means = np.mean(fixture.genotype_matrix_by_variant, axis=1)

    for trait_index in range(trait_count):
        null_reference = compute_numpy_null_logistic(
            fixture.covariate_matrix,
            fixture.phenotype_matrix[trait_index],
            fixture.loco_offset_matrix[trait_index],
            kernel_config,
        )
        square_root_weight = np.sqrt(null_reference.weight)
        for variant_index in range(variant_count):
            raw_genotype = fixture.genotype_matrix_by_variant[variant_index]
            flipped = bool(genotype_means[variant_index] > 1.0)
            coded_genotype = 2.0 - raw_genotype if flipped else raw_genotype
            projection_coordinates = null_reference.weighted_projection_matrix @ (square_root_weight * coded_genotype)
            variance = np.sum(null_reference.weight * coded_genotype**2) - np.sum(projection_coordinates**2)
            score = float(coded_genotype @ null_reference.score_residual)
            inverse_variance = 1.0 / variance
            beta[trait_index, variant_index] = (-score if flipped else score) * inverse_variance
            standard_error[trait_index, variant_index] = math.sqrt(inverse_variance)
            chi_squared[trait_index, variant_index] = score * score * inverse_variance

    return BinaryReferenceResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=compute_negative_log10_chi_square_probability(chi_squared),
    )


def build_binary_chromosome_state(
    fixture: BinaryFixture,
    kernel_config: regenie2_binary_config.BinaryScoreConfig,
) -> regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState:
    """Build the production score state for the deterministic fixture."""
    state = regenie2_binary_state.build_multi_binary_state(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_matrix=jnp.asarray(fixture.phenotype_matrix),
    )
    return regenie2_binary_state.build_multi_binary_score_chromosome_state(
        state=state,
        loco_offset_matrix=jnp.asarray(fixture.loco_offset_matrix),
        kernel_config=kernel_config,
    )


def test_regenie_logistic_probability_uses_documented_endpoint_clipping() -> None:
    """Exercise both strict eta endpoints and the ordinary sigmoid branch."""
    linear_predictor = jnp.asarray([-31.0, -30.0, 0.0, 30.0, 31.0], dtype=jnp.float64)
    observed = np.asarray(regenie2_binary_logistic.compute_regenie_logistic_probability(linear_predictor))
    epsilon = regenie2_binary_config.REGENIE_NUMERICAL_EPSILON_MULTIPLIER * np.finfo(np.float64).eps
    reference = np.asarray(
        [
            epsilon / (1.0 + epsilon),
            1.0 / (1.0 + math.exp(30.0)),
            0.5,
            1.0 / (1.0 + math.exp(-30.0)),
            1.0 / (1.0 + epsilon),
        ],
        dtype=np.float64,
    )

    tests.numerical.assert_absolute_difference_less_than(observed, reference, 1.0e-15)


def test_logistic_deviance_matches_masked_bernoulli_likelihood() -> None:
    """Include only active samples and use the binary case threshold."""
    phenotype = jnp.asarray([0.0, 1.0, 1.0], dtype=jnp.float64)
    probability = jnp.asarray([0.25, 0.75, 0.5], dtype=jnp.float64)
    active_mask = jnp.asarray([True, True, False])
    reference = -2.0 * (math.log1p(-0.25) + math.log(0.75))

    observed = regenie2_binary_logistic.compute_logistic_deviance(phenotype, probability, active_mask)

    tests.numerical.assert_absolute_difference_less_than(observed, reference, 1.0e-12)


def test_intercept_only_null_logistic_matches_balanced_analytic_solution() -> None:
    """Converge at a zero intercept for a balanced phenotype."""
    kernel_config = build_binary_score_config()
    observed = regenie2_binary_null_logistic.fit_null_logistic_coefficients(
        covariate_matrix=jnp.ones((6, 1), dtype=jnp.float32),
        phenotype_vector=jnp.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=jnp.float32),
        loco_offset=jnp.zeros((6,), dtype=jnp.float32),
        kernel_config=kernel_config,
    )

    assert bool(np.asarray(observed.converged))
    assert int(np.asarray(observed.iteration_count)) == 1
    tests.numerical.assert_absolute_difference_less_than(observed.coefficients, np.asarray([0.0]), 1.0e-12)


def test_null_logistic_zero_iteration_budget_is_explicit_failure() -> None:
    """Preserve the maximum-iteration failure signal without hidden work."""
    base_config = build_binary_score_config()
    zero_iteration_config = regenie2_binary_config.BinaryScoreConfig(
        numerical=base_config.numerical,
        null_logistic=regenie2_binary_config.BinaryNullLogisticConfig(
            maximum_iterations=0,
            coefficient_tolerance=base_config.null_logistic.coefficient_tolerance,
        ),
    )
    observed = regenie2_binary_null_logistic.fit_null_logistic_coefficients(
        covariate_matrix=jnp.ones((4, 1), dtype=jnp.float32),
        phenotype_vector=jnp.asarray([0.0, 1.0, 0.0, 1.0], dtype=jnp.float32),
        loco_offset=jnp.zeros((4,), dtype=jnp.float32),
        kernel_config=zero_iteration_config,
    )

    assert not bool(np.asarray(observed.converged))
    assert int(np.asarray(observed.iteration_count)) == 0
    tests.numerical.assert_absolute_difference_less_than(observed.coefficients, np.asarray([0.0]), 1.0e-12)


def test_binary_score_matches_independent_weighted_numpy_reference() -> None:
    """Validate multi-trait score statistics, including high-frequency flipping."""
    fixture = build_binary_fixture()
    kernel_config = build_binary_score_config()
    chromosome_state = build_binary_chromosome_state(fixture, kernel_config)
    reference = compute_binary_score_reference(fixture, kernel_config)

    observed = regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=jnp.asarray(fixture.genotype_matrix_by_variant),
        firth_candidate_p_threshold=None,
        minimum_variance=kernel_config.numerical.minimum_variance,
        relative_variance_tolerance=kernel_config.numerical.relative_variance_tolerance,
        native_genotype_mean=None,
    )

    tests.numerical.assert_absolute_difference_less_than(
        observed.beta,
        reference.beta,
        BINARY_BETA_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.standard_error,
        reference.standard_error,
        BINARY_STANDARD_ERROR_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.chi_squared,
        reference.chi_squared,
        BINARY_CHI_SQUARED_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.log10_p_value,
        reference.log10_p_value,
        BINARY_LOG10_P_VALUE_ABSOLUTE_TOLERANCE,
    )
    np.testing.assert_array_equal(
        np.asarray(observed.correction_code),
        np.full(reference.beta.shape, types.BinaryCorrectionCode.SCORE_SUCCESS.value, dtype=np.uint8),
    )


def test_ultra_rare_flipped_score_uses_stable_minor_allele_reductions() -> None:
    """Keep high-frequency score statistics and Firth classification stable."""
    sample_count = 2_504
    variant_count = 3
    random_generator = np.random.default_rng(192)
    score_residual = random_generator.normal(0.0, 0.45, size=sample_count).astype(np.float32)
    score_residual -= np.float32(score_residual.mean(dtype=np.float64))
    bernoulli_weight = random_generator.uniform(0.08, 0.25, size=sample_count).astype(np.float32)
    raw_genotype_matrix_by_variant = np.full((variant_count, sample_count), 2.0, dtype=np.float32)
    for variant_index in range(variant_count):
        carrier_indices = random_generator.choice(
            sample_count,
            size=variant_index + 1,
            replace=False,
        )
        raw_genotype_matrix_by_variant[variant_index, carrier_indices] = random_generator.choice(
            np.asarray([0.0, 1.0], dtype=np.float32),
            size=carrier_indices.size,
        )

    chromosome_state = regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState(
        score_right_hand_matrix=jnp.asarray(
            np.concatenate(
                [
                    np.zeros((1, sample_count), dtype=np.float32),
                    score_residual[None, :],
                ],
                axis=0,
            )
        ),
        bernoulli_weight=jnp.asarray(bernoulli_weight[None, :]),
        null_logistic_converged=jnp.asarray([True]),
    )
    observed = regenie2_binary_score.compute_multi_binary_score_test_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=jnp.asarray(raw_genotype_matrix_by_variant),
        firth_candidate_p_threshold=0.05,
        minimum_variance=1.0e-10,
        relative_variance_tolerance=1.0e-7,
        native_genotype_mean=None,
    )

    score_genotype_matrix_by_variant = 2.0 - raw_genotype_matrix_by_variant.astype(np.float64)
    reference_score = score_genotype_matrix_by_variant @ score_residual.astype(np.float64)
    reference_variance = (score_genotype_matrix_by_variant * score_genotype_matrix_by_variant) @ (
        bernoulli_weight.astype(np.float64)
    )
    reference_beta = -reference_score / reference_variance
    reference_standard_error = np.sqrt(np.reciprocal(reference_variance))
    reference_chi_squared = reference_score * reference_score / reference_variance
    reference_log10_p_value = compute_negative_log10_chi_square_probability(reference_chi_squared)

    assert bool(np.all(np.mean(raw_genotype_matrix_by_variant, axis=1) > 1.0))
    tests.numerical.assert_absolute_difference_less_than(
        observed.beta[0],
        reference_beta,
        BINARY_BETA_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.standard_error[0],
        reference_standard_error,
        BINARY_STANDARD_ERROR_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.chi_squared[0],
        reference_chi_squared,
        BINARY_CHI_SQUARED_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.log10_p_value[0],
        reference_log10_p_value,
        BINARY_LOG10_P_VALUE_ABSOLUTE_TOLERANCE,
    )
    np.testing.assert_array_equal(
        np.asarray(observed.correction_code[0]),
        np.asarray(
            [
                types.BinaryCorrectionCode.FIRTH_SUCCESS.value,
                types.BinaryCorrectionCode.SCORE_SUCCESS.value,
                types.BinaryCorrectionCode.SCORE_SUCCESS.value,
            ],
            dtype=np.uint8,
        ),
    )


@pytest.mark.parametrize("sample_count", [256, 257])
def test_decoded_score_reduction_handles_full_and_tail_tiles(sample_count: int) -> None:
    """Reduce exact tile boundaries and one-sample tails without complement sums."""
    sample_indices = np.arange(sample_count)
    raw_genotype_matrix_by_variant = np.stack(
        [
            (sample_indices % 3).astype(np.float32),
            np.where(sample_indices % 64 == 0, 1.0, 2.0).astype(np.float32),
        ]
    )
    score_right_hand_matrix = np.stack(
        [
            np.zeros(sample_count, dtype=np.float32),
            np.where(sample_indices % 2 == 0, 1.0, -1.0).astype(np.float32),
        ]
    )
    bernoulli_weight = np.full((1, sample_count), 0.25, dtype=np.float32)
    chromosome_state = regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState(
        score_right_hand_matrix=jnp.asarray(score_right_hand_matrix),
        bernoulli_weight=jnp.asarray(bernoulli_weight),
        null_logistic_converged=jnp.asarray([True]),
    )
    genotype_flip_mask = np.mean(raw_genotype_matrix_by_variant, axis=1) > 1.0

    observed = regenie2_binary_score.reduce_tiled_score_genotypes(
        chromosome_state,
        jnp.asarray(raw_genotype_matrix_by_variant),
        jnp.asarray(genotype_flip_mask),
    )

    score_genotype_matrix_by_variant = np.where(
        genotype_flip_mask[:, None],
        2.0 - raw_genotype_matrix_by_variant,
        raw_genotype_matrix_by_variant,
    )
    expected_stacked_product = score_genotype_matrix_by_variant @ score_right_hand_matrix.T
    expected_weighted_sum_squares = (score_genotype_matrix_by_variant**2) @ bernoulli_weight.T
    np.testing.assert_array_equal(
        np.asarray(observed.stacked_product_by_variant),
        expected_stacked_product,
    )
    np.testing.assert_array_equal(
        np.asarray(observed.weighted_genotype_sum_squares).T,
        expected_weighted_sum_squares,
    )


@pytest.mark.parametrize("sample_count", [256, 257])
def test_packed8_score_uses_exact_minor_dosage_numerators(sample_count: int) -> None:
    """Decode flipped and unflipped byte rows across full and tail geometries."""
    sample_indices = np.arange(sample_count)
    rare_sample_mask = sample_indices % 31 == 0
    low_frequency_first_probability = np.where(rare_sample_mask, 5, 230).astype(np.uint8)
    low_frequency_second_probability = np.where(rare_sample_mask, 10, 20).astype(np.uint8)
    packed_probability_pairs_by_variant = np.stack(
        [
            np.stack(
                [low_frequency_first_probability, low_frequency_second_probability],
                axis=1,
            ),
            np.stack(
                [230 - low_frequency_first_probability + 5, 30 - low_frequency_second_probability],
                axis=1,
            ),
        ]
    ).astype(np.uint8)
    probability_values = packed_probability_pairs_by_variant.astype(np.float64)
    native_dosage_numerator = (
        genotype.PACKED8_DIPLOID_NUMERATOR
        - genotype.ALLELE_COUNT_MULTIPLIER * probability_values[:, :, 0]
        - probability_values[:, :, 1]
    )
    native_genotype_mean = np.mean(
        native_dosage_numerator / genotype.EIGHT_BIT_PROBABILITY_DENOMINATOR,
        axis=1,
    ).astype(np.float32)
    genotype_flip_mask = native_genotype_mean > 1.0
    expected_score_dosage_numerator = np.where(
        genotype_flip_mask[:, None],
        genotype.ALLELE_COUNT_MULTIPLIER * probability_values[:, :, 0] + probability_values[:, :, 1],
        native_dosage_numerator,
    )
    expected_score_genotype_matrix_by_variant = expected_score_dosage_numerator.astype(np.float32) / np.float32(
        genotype.EIGHT_BIT_PROBABILITY_DENOMINATOR
    )
    random_generator = np.random.default_rng(2_100 + sample_count)
    score_residual = random_generator.normal(0.0, 0.45, size=sample_count).astype(np.float32)
    score_residual -= np.float32(score_residual.mean(dtype=np.float64))
    bernoulli_weight = random_generator.uniform(0.08, 0.25, size=sample_count).astype(np.float32)
    chromosome_state = regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState(
        score_right_hand_matrix=jnp.asarray(
            np.stack(
                [
                    np.zeros(sample_count, dtype=np.float32),
                    score_residual,
                ]
            )
        ),
        bernoulli_weight=jnp.asarray(bernoulli_weight[None, :]),
        null_logistic_converged=jnp.asarray([True]),
    )

    observed_score_genotype_matrix_by_variant = genotype.decode_packed8_probability_pairs_to_regenie_score_genotypes(
        jnp.asarray(packed_probability_pairs_by_variant),
        jnp.asarray(genotype_flip_mask),
    )
    observed = regenie2_binary_score.compute_multi_binary_score_test_packed8_core(
        chromosome_state=chromosome_state,
        packed_probability_pairs_by_variant=jnp.asarray(packed_probability_pairs_by_variant),
        firth_candidate_p_threshold=None,
        minimum_variance=1.0e-10,
        relative_variance_tolerance=1.0e-7,
        native_genotype_mean=jnp.asarray(native_genotype_mean),
    )

    expected_score_genotype_float64 = expected_score_genotype_matrix_by_variant.astype(np.float64)
    expected_score = expected_score_genotype_float64 @ score_residual.astype(np.float64)
    expected_variance = (expected_score_genotype_float64**2) @ bernoulli_weight.astype(np.float64)
    expected_beta = np.where(
        genotype_flip_mask, -expected_score / expected_variance, expected_score / expected_variance
    )
    expected_standard_error = np.sqrt(np.reciprocal(expected_variance))
    expected_chi_squared = expected_score * expected_score / expected_variance
    expected_log10_p_value = compute_negative_log10_chi_square_probability(expected_chi_squared)

    assert genotype_flip_mask.tolist() == [False, True]
    tests.numerical.assert_absolute_difference_less_than(
        observed_score_genotype_matrix_by_variant,
        expected_score_genotype_matrix_by_variant,
        1.3e-7,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.beta[0],
        expected_beta,
        BINARY_BETA_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.standard_error[0],
        expected_standard_error,
        BINARY_STANDARD_ERROR_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.chi_squared[0],
        expected_chi_squared,
        BINARY_CHI_SQUARED_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.log10_p_value[0],
        expected_log10_p_value,
        BINARY_LOG10_P_VALUE_ABSOLUTE_TOLERANCE,
    )
    np.testing.assert_array_equal(
        np.asarray(observed.correction_code[0]),
        np.full(2, types.BinaryCorrectionCode.SCORE_SUCCESS.value, dtype=np.uint8),
    )


def test_binary_monomorphic_variant_is_score_failure() -> None:
    """Reject zero residual variance and label the row explicitly."""
    fixture = build_binary_fixture()
    kernel_config = build_binary_score_config()
    chromosome_state = build_binary_chromosome_state(fixture, kernel_config)
    observed = regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=jnp.ones((1, fixture.covariate_matrix.shape[0]), dtype=jnp.float32),
        firth_candidate_p_threshold=None,
        minimum_variance=kernel_config.numerical.minimum_variance,
        relative_variance_tolerance=kernel_config.numerical.relative_variance_tolerance,
        native_genotype_mean=None,
    )

    assert bool(np.all(np.isnan(np.asarray(observed.beta))))
    assert bool(np.all(np.isnan(np.asarray(observed.standard_error))))
    np.testing.assert_array_equal(
        np.asarray(observed.correction_code),
        np.full((fixture.phenotype_matrix.shape[0], 1), types.BinaryCorrectionCode.SCORE_FAILED.value, dtype=np.uint8),
    )


def test_correction_threshold_is_strict_and_invalid_rows_take_precedence() -> None:
    """Select Firth only above the negative-log10 score threshold."""
    threshold = 0.05
    log10_threshold = -math.log10(threshold)
    observed = regenie2_binary_correction.build_correction_code(
        log10_p_value=jnp.asarray([log10_threshold, log10_threshold + 1.0e-6, log10_threshold + 2.0]),
        valid_mask=jnp.asarray([True, True, False]),
        firth_candidate_p_threshold=threshold,
    )

    np.testing.assert_array_equal(
        np.asarray(observed),
        np.asarray(
            [
                types.BinaryCorrectionCode.SCORE_SUCCESS.value,
                types.BinaryCorrectionCode.FIRTH_SUCCESS.value,
                types.BinaryCorrectionCode.SCORE_FAILED.value,
            ],
            dtype=np.uint8,
        ),
    )


def test_separation_heuristic_uses_case_and_control_allele_counts() -> None:
    """Detect absent alternate or reference alleles in either outcome group."""
    genotype_matrix = jnp.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 2.0, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    phenotype = jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jnp.float32)

    observed = regenie2_binary_candidates.compute_firth_pre_dispatch_mask_without_mask(
        genotype_matrix,
        phenotype,
    )

    np.testing.assert_array_equal(np.asarray(observed), np.asarray([True, False, True]))


def test_fixed_capacity_batch_plan_preserves_candidate_order_and_padding() -> None:
    """Build deterministic candidate batches without data-dependent shapes."""
    observed = regenie2_binary_candidates.build_device_firth_batch_plan(
        fallback_mask=jnp.asarray([False, True, False, True, True]),
        candidate_capacity=4,
        firth_batch_size=2,
    )

    np.testing.assert_array_equal(np.asarray(observed.fallback_index_matrix), np.asarray([[1, 3], [4, 0]]))
    np.testing.assert_array_equal(
        np.asarray(observed.fallback_active_mask_matrix),
        np.asarray([[True, True], [True, False]]),
    )


def test_candidate_bucket_order_is_stable_within_each_class() -> None:
    """Group regular, heuristic, and inactive lanes without reordering peers."""
    observed = regenie2_binary_candidates.build_firth_candidate_bucket_order(
        flat_active_mask=jnp.asarray([True, True, False, True, False]),
        heuristic_firth_mask=jnp.asarray([True, False, False, True, False]),
    )

    np.testing.assert_array_equal(np.asarray(observed), np.asarray([1, 0, 3, 2, 4], dtype=np.int32))


def test_candidate_planning_rejects_invalid_static_geometry() -> None:
    """Fail before tracing when capacity or batch size cannot form an executable."""
    fallback_mask = jnp.asarray([True, False])
    with pytest.raises(ValueError, match="capacity must be positive"):
        regenie2_binary_candidates.build_device_firth_batch_plan(fallback_mask, 0, 1)
    with pytest.raises(ValueError, match="batch size must be positive"):
        regenie2_binary_candidates.build_device_firth_batch_plan(fallback_mask, 1, 0)


def test_binary_packed8_path_matches_explicit_decode() -> None:
    """Keep packed delivery equivalent to the dosage score kernel."""
    fixture = build_binary_fixture()
    kernel_config = build_binary_score_config()
    chromosome_state = build_binary_chromosome_state(fixture, kernel_config)
    packed_probabilities = jnp.asarray(
        [
            [[255, 0], [220, 30], [150, 80], [90, 100], [40, 150], [0, 210], [120, 100], [20, 40], [180, 30], [70, 80]],
            [[0, 0], [20, 10], [60, 40], [90, 50], [110, 90], [150, 50], [180, 40], [240, 10], [30, 20], [100, 120]],
        ],
        dtype=jnp.uint8,
    )
    decoded = (
        510.0
        - 2.0 * np.asarray(packed_probabilities[:, :, 0], dtype=np.float32)
        - np.asarray(packed_probabilities[:, :, 1], dtype=np.float32)
    ) / 255.0

    packed_result = regenie2_binary_score.compute_multi_binary_score_test_packed8_core(
        chromosome_state=chromosome_state,
        packed_probability_pairs_by_variant=packed_probabilities,
        firth_candidate_p_threshold=None,
        minimum_variance=kernel_config.numerical.minimum_variance,
        relative_variance_tolerance=kernel_config.numerical.relative_variance_tolerance,
        native_genotype_mean=None,
    )
    decoded_result = regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=jnp.asarray(decoded),
        firth_candidate_p_threshold=None,
        minimum_variance=kernel_config.numerical.minimum_variance,
        relative_variance_tolerance=kernel_config.numerical.relative_variance_tolerance,
        native_genotype_mean=None,
    )

    tests.numerical.assert_absolute_difference_less_than(
        packed_result.beta,
        decoded_result.beta,
        BINARY_PACKED_BETA_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        packed_result.standard_error,
        decoded_result.standard_error,
        BINARY_PACKED_STANDARD_ERROR_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        packed_result.chi_squared,
        decoded_result.chi_squared,
        BINARY_PACKED_CHI_SQUARED_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        packed_result.log10_p_value,
        decoded_result.log10_p_value,
        BINARY_PACKED_LOG10_P_VALUE_ABSOLUTE_TOLERANCE,
    )
    np.testing.assert_array_equal(packed_result.correction_code, decoded_result.correction_code)
