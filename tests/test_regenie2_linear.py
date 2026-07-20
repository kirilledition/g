"""Correctness tests for shared and quantitative REGENIE kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

import tests.numerical
from g.compute.common import genotype, linalg, pvalue
from g.compute.regenie2_linear import score as regenie2_linear_score
from g.compute.regenie2_linear import state as regenie2_linear_state

# The independent oracle uses float64 and production uses float32 reductions.
# These exclusive per-statistic bounds exceed twice the largest measured
# cross-host maxima (5.48e-7, 1.03e-7, 4.52e-6, and 1.24e-6 respectively).
# That headroom covers architecture-dependent float32 lowering of this
# eight-sample fixture while keeping each statistic independently constrained.
LINEAR_BETA_ABSOLUTE_TOLERANCE = 1.2e-6
LINEAR_STANDARD_ERROR_ABSOLUTE_TOLERANCE = 2.2e-7
LINEAR_CHI_SQUARED_ABSOLUTE_TOLERANCE = 1.0e-5
LINEAR_LOG10_P_VALUE_ABSOLUTE_TOLERANCE = 2.6e-6


@dataclass(frozen=True)
class LinearFixture:
    """Deterministic inputs spanning allele orientation and LOCO projection."""

    covariate_matrix: npt.NDArray[np.float64]
    phenotype_matrix: npt.NDArray[np.float64]
    loco_prediction_matrix: npt.NDArray[np.float64]
    genotype_matrix_by_variant: npt.NDArray[np.float64]


@dataclass(frozen=True)
class LinearReferenceResult:
    """Independent NumPy result for one multi-trait linear chunk."""

    beta: npt.NDArray[np.float64]
    standard_error: npt.NDArray[np.float64]
    chi_squared: npt.NDArray[np.float64]
    log10_p_value: npt.NDArray[np.float64]


def build_linear_fixture() -> LinearFixture:
    """Build a full-rank fixture with ordinary and high-frequency variants."""
    return LinearFixture(
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
        phenotype_matrix=np.asarray(
            [
                [1.2, 0.7, 1.6, 2.4, 2.0, 3.1, 2.8, 4.2],
                [-0.4, 0.6, -0.1, 1.1, 0.8, 0.2, 1.8, 1.4],
            ],
            dtype=np.float64,
        ),
        loco_prediction_matrix=np.asarray(
            [
                [0.10, -0.05, 0.00, 0.08, -0.04, 0.02, -0.03, 0.06],
                [-0.03, 0.04, -0.02, 0.01, 0.05, -0.06, 0.02, -0.01],
            ],
            dtype=np.float64,
        ),
        genotype_matrix_by_variant=np.asarray(
            [
                [0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0],
                [2.0, 2.0, 1.0, 2.0, 1.5, 1.0, 1.0, 0.0],
                [0.2, 1.1, 0.4, 1.8, 0.7, 1.5, 0.1, 1.2],
            ],
            dtype=np.float64,
        ),
    )


def compute_negative_log10_chi_square_probability(
    chi_squared: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Evaluate the one-degree-of-freedom chi-square tail independently."""
    flat_probabilities = [math.erfc(math.sqrt(float(value) / 2.0)) for value in chi_squared.ravel()]
    probabilities = np.asarray(flat_probabilities, dtype=np.float64).reshape(chi_squared.shape)
    return -np.log10(probabilities)


def compute_linear_reference(fixture: LinearFixture) -> LinearReferenceResult:
    """Compute the current quantitative contract with independent NumPy algebra."""
    return compute_linear_reference_with_genotype_statistics(
        fixture=fixture,
        genotype_means=np.mean(fixture.genotype_matrix_by_variant, axis=1),
        imputed_dosage_square_sum=None,
    )


def compute_linear_reference_with_genotype_statistics(
    *,
    fixture: LinearFixture,
    genotype_means: npt.NDArray[np.float64],
    imputed_dosage_square_sum: npt.NDArray[np.float64] | None,
) -> LinearReferenceResult:
    """Compute quantitative statistics with explicit native genotype moments."""
    covariates = fixture.covariate_matrix
    phenotypes = fixture.phenotype_matrix
    covariate_crossproduct = covariates.T @ covariates
    phenotype_coefficients = np.linalg.solve(covariate_crossproduct, covariates.T @ phenotypes.T)
    phenotype_residual_matrix = phenotypes - (covariates @ phenotype_coefficients).T

    cholesky_factor = np.linalg.cholesky(covariate_crossproduct)
    whitened_covariate_transpose = np.linalg.solve(cholesky_factor, covariates.T)
    adjusted_residual_matrix = phenotype_residual_matrix - fixture.loco_prediction_matrix
    residual_projection_coordinates = adjusted_residual_matrix @ whitened_covariate_transpose.T
    adjusted_residual_sum_squares = np.sum(adjusted_residual_matrix**2, axis=1) - np.sum(
        residual_projection_coordinates**2,
        axis=1,
    )

    genotype_offsets = np.where(genotype_means > 1.0, 2.0, 0.0)
    normalized_genotypes = fixture.genotype_matrix_by_variant - genotype_offsets[:, None]
    if imputed_dosage_square_sum is None:
        genotype_sum_squares = np.sum(normalized_genotypes**2, axis=1)
    else:
        sample_count = fixture.genotype_matrix_by_variant.shape[1]
        imputed_dosage_sum = genotype_means * sample_count
        genotype_sum_squares = (
            imputed_dosage_square_sum - 2.0 * genotype_offsets * imputed_dosage_sum + sample_count * genotype_offsets**2
        )
    genotype_projection_coordinates = whitened_covariate_transpose @ normalized_genotypes.T
    genotype_residual_sum_squares = genotype_sum_squares - np.sum(genotype_projection_coordinates**2, axis=0)
    covariance = adjusted_residual_matrix @ normalized_genotypes.T - (
        residual_projection_coordinates @ genotype_projection_coordinates
    )
    degrees_of_freedom = covariates.shape[0] - covariates.shape[1]
    null_mean_squared_error = adjusted_residual_sum_squares / degrees_of_freedom

    beta = covariance / genotype_residual_sum_squares[None, :]
    standard_error = np.sqrt(null_mean_squared_error[:, None] / genotype_residual_sum_squares[None, :])
    chi_squared = covariance**2 / genotype_residual_sum_squares[None, :] / null_mean_squared_error[:, None]
    return LinearReferenceResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=compute_negative_log10_chi_square_probability(chi_squared),
    )


def build_linear_chromosome_state(
    fixture: LinearFixture,
) -> regenie2_linear_state.Regenie2MultiLinearChromosomeState:
    """Build the production state for the deterministic fixture."""
    shared_state = regenie2_linear_state.build_multi_linear_state(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_matrix=jnp.asarray(fixture.phenotype_matrix),
    )
    return regenie2_linear_state.build_multi_linear_chromosome_state(
        state=shared_state,
        loco_prediction_matrix=jnp.asarray(fixture.loco_prediction_matrix),
    )


def test_packed8_decode_matches_probability_definition() -> None:
    """Decode probability bytes through the BGEN eight-bit dosage equation."""
    packed_probabilities = jnp.asarray(
        [[[255, 0], [0, 0], [0, 255], [64, 127]]],
        dtype=jnp.uint8,
    )

    observed = genotype.decode_packed8_probability_pairs_to_variant_major_dosage(packed_probabilities)

    tests.numerical.assert_absolute_difference_less_than(
        observed,
        np.asarray([[0.0, 2.0, 1.0, 1.0]], dtype=np.float64),
        1.0e-7,
    )


def test_regenie_allele_flip_is_strictly_above_mean_one() -> None:
    """Preserve REGENIE's strict high-frequency allele orientation boundary."""
    genotype_matrix = jnp.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.5, 2.0],
        ],
        dtype=jnp.float32,
    )

    result = genotype.build_regenie_flipped_genotypes(genotype_matrix, native_genotype_mean=None)

    np.testing.assert_array_equal(np.asarray(result.flip_mask), np.asarray([False, False, True]))
    tests.numerical.assert_absolute_difference_less_than(
        result.genotype_matrix_by_variant,
        np.asarray(
            [
                [0.0, 1.0, 2.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.5, 0.0],
            ],
            dtype=np.float64,
        ),
        1.0e-7,
    )


def test_positive_variance_mask_uses_exclusive_absolute_and_relative_floors() -> None:
    """Reject values equal to either numerical variance floor."""
    observed = linalg.compute_positive_residual_variance_mask(
        variance=jnp.asarray([1.0e-6, 1.1e-6, 2.0e-3, 2.1e-3]),
        reference_sum_squares=jnp.asarray([1.0e-4, 1.0e-4, 2.0, 2.0]),
        minimum_variance=1.0e-6,
        relative_variance_tolerance=1.0e-3,
    )

    np.testing.assert_array_equal(np.asarray(observed), np.asarray([False, True, False, True]))


def test_positive_definite_solve_matches_numpy() -> None:
    """Solve vector and matrix right-hand sides from one Cholesky factor."""
    coefficient_matrix = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    right_hand_side = np.asarray([[1.0, 2.0], [3.0, -1.0]], dtype=np.float64)
    observed = linalg.solve_positive_definite_system(
        jnp.asarray(np.linalg.cholesky(coefficient_matrix)),
        jnp.asarray(right_hand_side),
    )

    tests.numerical.assert_absolute_difference_less_than(
        observed,
        np.linalg.solve(coefficient_matrix, right_hand_side),
        1.0e-12,
    )


def test_chi_squared_tail_matches_erfc_reference_and_clamps_negative_input() -> None:
    """Match the analytic one-degree-of-freedom survival function."""
    chi_squared = np.asarray([-1.0, 0.0, 1.0, 4.0, 9.0], dtype=np.float64)
    safe_chi_squared = np.maximum(chi_squared, 0.0)
    reference = compute_negative_log10_chi_square_probability(safe_chi_squared)

    observed = pvalue.chi_squared_to_log10_p_value(jnp.asarray(chi_squared))

    tests.numerical.assert_absolute_difference_less_than(observed, reference, 1.0e-12)


def test_linear_state_residualizes_phenotypes_against_covariates() -> None:
    """Keep reusable phenotype residuals orthogonal to the covariate space."""
    fixture = build_linear_fixture()
    shared_state = regenie2_linear_state.build_multi_linear_state(
        covariate_matrix=jnp.asarray(fixture.covariate_matrix),
        phenotype_matrix=jnp.asarray(fixture.phenotype_matrix),
    )
    crossproduct = np.asarray(shared_state.phenotype_residual_matrix) @ fixture.covariate_matrix

    tests.numerical.assert_absolute_difference_less_than(crossproduct, np.zeros_like(crossproduct), 2.0e-5)
    assert float(np.asarray(shared_state.degrees_of_freedom)) == 6.0


def test_linear_chunk_matches_independent_numpy_reference() -> None:
    """Validate beta, uncertainty, chi-square, and tail probability together."""
    fixture = build_linear_fixture()
    chromosome_state = build_linear_chromosome_state(fixture)
    reference = compute_linear_reference(fixture)

    observed = regenie2_linear_score.compute_regenie2_linear_chunk_trait_major_variant_major_core(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=jnp.asarray(fixture.genotype_matrix_by_variant),
        native_genotype_mean=None,
        genotype_imputed_dosage_square_sum=None,
        linear_minimum_variance=1.0e-8,
        linear_relative_variance_tolerance=1.0e-7,
    )

    tests.numerical.assert_absolute_difference_less_than(
        observed.beta,
        reference.beta,
        LINEAR_BETA_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.standard_error,
        reference.standard_error,
        LINEAR_STANDARD_ERROR_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.chi_squared,
        reference.chi_squared,
        LINEAR_CHI_SQUARED_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.log10_p_value,
        reference.log10_p_value,
        LINEAR_LOG10_P_VALUE_ABSOLUTE_TOLERANCE,
    )
    assert observed.correction_code is None


def test_linear_native_statistics_path_uses_supplied_moments() -> None:
    """Use native BGEN summaries instead of silently reducing delivered dosages."""
    fixture = build_linear_fixture()
    chromosome_state = build_linear_chromosome_state(fixture)
    direct_genotype_means = np.mean(fixture.genotype_matrix_by_variant, axis=1)
    direct_genotype_square_sums = np.sum(fixture.genotype_matrix_by_variant**2, axis=1)
    native_genotype_means = direct_genotype_means + np.asarray([0.03, -0.02, 0.04], dtype=np.float64)
    native_genotype_square_sums = direct_genotype_square_sums + np.asarray(
        [0.75, 1.25, 0.50],
        dtype=np.float64,
    )
    reference = compute_linear_reference_with_genotype_statistics(
        fixture=fixture,
        genotype_means=native_genotype_means,
        imputed_dosage_square_sum=native_genotype_square_sums,
    )

    direct = regenie2_linear_score.compute_regenie2_linear_chunk_trait_major_variant_major_core(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=jnp.asarray(fixture.genotype_matrix_by_variant),
        native_genotype_mean=None,
        genotype_imputed_dosage_square_sum=None,
        linear_minimum_variance=1.0e-8,
        linear_relative_variance_tolerance=1.0e-7,
    )
    summarized = regenie2_linear_score.compute_regenie2_linear_chunk_trait_major_variant_major_core(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=jnp.asarray(fixture.genotype_matrix_by_variant),
        native_genotype_mean=jnp.asarray(native_genotype_means),
        genotype_imputed_dosage_square_sum=jnp.asarray(native_genotype_square_sums),
        linear_minimum_variance=1.0e-8,
        linear_relative_variance_tolerance=1.0e-7,
    )

    assert float(np.max(np.abs(np.asarray(summarized.beta) - np.asarray(direct.beta)))) > 1.0e-3
    tests.numerical.assert_absolute_difference_less_than(
        summarized.beta,
        reference.beta,
        LINEAR_BETA_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        summarized.standard_error,
        reference.standard_error,
        LINEAR_STANDARD_ERROR_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        summarized.chi_squared,
        reference.chi_squared,
        LINEAR_CHI_SQUARED_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        summarized.log10_p_value,
        reference.log10_p_value,
        LINEAR_LOG10_P_VALUE_ABSOLUTE_TOLERANCE,
    )


def test_linear_monomorphic_variant_produces_invalid_statistics() -> None:
    """Reject a variant with no residual variance after covariate projection."""
    fixture = build_linear_fixture()
    chromosome_state = build_linear_chromosome_state(fixture)
    observed = regenie2_linear_score.compute_regenie2_linear_chunk_trait_major_variant_major_core(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=jnp.ones((1, fixture.covariate_matrix.shape[0]), dtype=jnp.float32),
        native_genotype_mean=None,
        genotype_imputed_dosage_square_sum=None,
        linear_minimum_variance=1.0e-8,
        linear_relative_variance_tolerance=1.0e-7,
    )

    assert bool(np.all(np.isnan(np.asarray(observed.beta))))
    assert bool(np.all(np.isnan(np.asarray(observed.standard_error))))
    assert bool(np.all(np.isnan(np.asarray(observed.chi_squared))))
    assert bool(np.all(np.isnan(np.asarray(observed.log10_p_value))))


def test_linear_packed8_path_matches_explicit_decode() -> None:
    """Match both delivery paths to a well-conditioned independent oracle."""
    fixture = build_linear_fixture()
    chromosome_state = build_linear_chromosome_state(fixture)
    packed_probabilities = jnp.asarray(
        [
            [[255, 0], [0, 0], [255, 0], [0, 0], [255, 0], [0, 0], [255, 0], [0, 0]],
            [[255, 0], [0, 255], [0, 0], [255, 0], [0, 0], [0, 255], [255, 0], [0, 0]],
        ],
        dtype=jnp.uint8,
    )
    decoded_genotypes = genotype.decode_packed8_probability_pairs_to_variant_major_dosage(packed_probabilities)
    decoded_fixture = LinearFixture(
        covariate_matrix=fixture.covariate_matrix,
        phenotype_matrix=fixture.phenotype_matrix,
        loco_prediction_matrix=fixture.loco_prediction_matrix,
        genotype_matrix_by_variant=np.asarray(decoded_genotypes, dtype=np.float64),
    )
    reference = compute_linear_reference(decoded_fixture)

    packed_result = regenie2_linear_score.compute_multi_linear_chunk_packed8_donating_inputs(
        chromosome_state=chromosome_state,
        packed_probability_pairs_by_variant=packed_probabilities,
        native_genotype_mean=None,
        genotype_imputed_dosage_square_sum=None,
        linear_minimum_variance=1.0e-8,
        linear_relative_variance_tolerance=1.0e-6,
    )
    decoded_result = regenie2_linear_score.compute_regenie2_linear_chunk_trait_major_variant_major_core(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=decoded_genotypes,
        native_genotype_mean=None,
        genotype_imputed_dosage_square_sum=None,
        linear_minimum_variance=1.0e-8,
        linear_relative_variance_tolerance=1.0e-6,
    )

    result_fields = (
        (packed_result.beta, decoded_result.beta, reference.beta, LINEAR_BETA_ABSOLUTE_TOLERANCE),
        (
            packed_result.standard_error,
            decoded_result.standard_error,
            reference.standard_error,
            LINEAR_STANDARD_ERROR_ABSOLUTE_TOLERANCE,
        ),
        (
            packed_result.chi_squared,
            decoded_result.chi_squared,
            reference.chi_squared,
            LINEAR_CHI_SQUARED_ABSOLUTE_TOLERANCE,
        ),
        (
            packed_result.log10_p_value,
            decoded_result.log10_p_value,
            reference.log10_p_value,
            LINEAR_LOG10_P_VALUE_ABSOLUTE_TOLERANCE,
        ),
    )
    for packed_values, decoded_values, reference_values, tolerance in result_fields:
        tests.numerical.assert_absolute_difference_less_than(packed_values, reference_values, tolerance)
        tests.numerical.assert_absolute_difference_less_than(decoded_values, reference_values, tolerance)
