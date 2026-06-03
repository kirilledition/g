"""Unit tests for REGENIE step 2 linear association kernel."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing
import numpy.typing as npt

from g import types
from g.compute.common import genotype, linalg, pvalue
from g.compute.regenie2_linear import api as regenie2_linear
from g.compute.regenie2_linear import score as regenie2_linear_score
from g.compute.regenie2_linear import state as regenie2_linear_state


@dataclass(frozen=True)
class ReferenceRegenie2LinearChunkResult:
    """Reference result from an unoptimized score-statistic formula."""

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    valid_mask: jax.Array


@dataclass(frozen=True)
class LinearLocoCovariateFixture:
    """Tiny REGENIE parity fixture with covariate-correlated LOCO predictions."""

    covariate_matrix: npt.NDArray[np.float64]
    phenotype_vector: npt.NDArray[np.float64]
    genotype_matrix: npt.NDArray[np.float64]
    loco_predictions: npt.NDArray[np.float64]
    expected_beta: npt.NDArray[np.float64]
    expected_standard_error: npt.NDArray[np.float64]
    expected_chi_squared: npt.NDArray[np.float64]
    expected_log10_p_value: npt.NDArray[np.float64]


@dataclass(frozen=True)
class LinearFormulaResult:
    """Linear association statistics for one candidate LOCO residualization formula."""

    beta: npt.NDArray[np.float64]
    standard_error: npt.NDArray[np.float64]
    chi_squared: npt.NDArray[np.float64]
    log10_p_value: npt.NDArray[np.float64]


def compute_score_reference_chunk(
    state: regenie2_linear_state.Regenie2LinearState,
    covariate_matrix: jax.Array,
    genotype_matrix: jax.Array,
    loco_predictions: jax.Array,
) -> ReferenceRegenie2LinearChunkResult:
    """Compute the unoptimized score-statistic formula for regression-test comparison."""
    normalized_genotype_matrix = genotype.normalize_high_frequency_diploid_genotypes_sample_major(
        genotype_matrix,
    )
    covariate_matrix_compute = jnp.asarray(covariate_matrix, dtype=jnp.float32)
    covariate_matrix_transpose = covariate_matrix_compute.T
    covariate_crossproduct_cholesky_factor = jnp.linalg.cholesky(covariate_matrix_transpose @ covariate_matrix_compute)
    adjusted_residual = state.phenotype_residual - loco_predictions
    covariate_genotype_crossproduct = covariate_matrix_transpose @ normalized_genotype_matrix
    covariate_adjusted_residual_crossproduct = covariate_matrix_transpose @ adjusted_residual
    genotype_projection = linalg.solve_positive_definite_system(
        covariate_crossproduct_cholesky_factor,
        covariate_genotype_crossproduct,
    )
    adjusted_residual_projection = linalg.solve_positive_definite_system(
        covariate_crossproduct_cholesky_factor,
        covariate_adjusted_residual_crossproduct,
    )
    raw_adjusted_residual_sum_squares = jnp.dot(adjusted_residual, adjusted_residual)
    adjusted_residual_projection_sum_squares = covariate_adjusted_residual_crossproduct @ adjusted_residual_projection
    adjusted_residual_sum_squares = jnp.maximum(
        raw_adjusted_residual_sum_squares - adjusted_residual_projection_sum_squares,
        0.0,
    )
    genotype_sum_squares = jnp.einsum("ij,ij->j", normalized_genotype_matrix, normalized_genotype_matrix)
    projection_sum_squares = jnp.einsum("ij,ij->j", covariate_genotype_crossproduct, genotype_projection)
    genotype_residual_sum_squares = jnp.maximum(genotype_sum_squares - projection_sum_squares, 0.0)
    raw_covariance_with_phenotype = normalized_genotype_matrix.T @ adjusted_residual
    covariance_projection = covariate_genotype_crossproduct.T @ adjusted_residual_projection
    covariance_with_phenotype = raw_covariance_with_phenotype - covariance_projection
    covariance_squared = covariance_with_phenotype * covariance_with_phenotype
    positive_genotype_residual_mask = genotype_residual_sum_squares > 0.0
    genotype_residual_sum_squares_inverse = jnp.where(
        positive_genotype_residual_mask,
        jnp.reciprocal(genotype_residual_sum_squares),
        0.0,
    )
    beta = jnp.where(
        positive_genotype_residual_mask,
        covariance_with_phenotype * genotype_residual_sum_squares_inverse,
        jnp.nan,
    )
    null_mean_squared_error = adjusted_residual_sum_squares / state.degrees_of_freedom
    positive_null_mean_squared_error_mask = null_mean_squared_error > 0.0
    standard_error = jnp.where(
        positive_genotype_residual_mask & positive_null_mean_squared_error_mask,
        jnp.sqrt(null_mean_squared_error * genotype_residual_sum_squares_inverse),
        jnp.nan,
    )
    valid_statistic_mask = positive_genotype_residual_mask & positive_null_mean_squared_error_mask
    chi_squared = jnp.where(
        valid_statistic_mask,
        covariance_squared * genotype_residual_sum_squares_inverse / null_mean_squared_error,
        jnp.nan,
    )
    log10_p_value = jnp.where(
        valid_statistic_mask,
        pvalue.chi_squared_to_log10_p_value(chi_squared),
        jnp.nan,
    )
    valid_mask = jnp.isfinite(beta) & jnp.isfinite(standard_error) & (standard_error > 0.0)
    return ReferenceRegenie2LinearChunkResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        valid_mask=valid_mask,
    )


def build_loco_covariate_fixture() -> LinearLocoCovariateFixture:
    """Build a tiny fixture generated against REGENIE v4.1 step 2 output."""
    sample_count = 12
    correlated_covariate = np.asarray(
        [-1.6, -1.3, -0.9, -0.5, -0.2, 0.0, 0.3, 0.5, 0.8, 1.1, 1.4, 1.7],
        dtype=np.float64,
    )
    covariate_matrix = np.column_stack([np.ones(sample_count, dtype=np.float64), correlated_covariate])

    base_residual = np.asarray(
        [1.2, -0.7, 0.4, -1.1, 0.9, -0.2, 1.5, -1.4, 0.6, -0.8, 1.0, -1.3],
        dtype=np.float64,
    )
    residual_projection = covariate_matrix @ np.linalg.solve(
        covariate_matrix.T @ covariate_matrix,
        covariate_matrix.T @ base_residual,
    )
    phenotype_residual = base_residual - residual_projection
    phenotype_residual *= np.sqrt(sample_count - covariate_matrix.shape[1]) / np.linalg.norm(phenotype_residual)
    phenotype_vector = 1.25 + 0.4 * correlated_covariate + phenotype_residual

    loco_predictions = 0.65 * correlated_covariate + 0.15 * phenotype_residual
    genotype_matrix = 2.0 - np.asarray(
        [
            [0.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 2.0, 2.0],
            [2.0, 1.0, 1.0],
            [2.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return LinearLocoCovariateFixture(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        genotype_matrix=genotype_matrix,
        loco_predictions=loco_predictions,
        expected_beta=np.asarray([0.0502609, 0.29938, -0.0627165], dtype=np.float64),
        expected_standard_error=np.asarray([0.325455, 0.306214, 0.354042], dtype=np.float64),
        expected_chi_squared=np.asarray([0.0238496, 0.955865, 0.0313802], dtype=np.float64),
        expected_log10_p_value=np.asarray([0.0568676, 0.483821, 0.0658072], dtype=np.float64),
    )


def residualize_against_covariates(
    covariate_matrix: npt.NDArray[np.float64],
    value_array: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Residualize one vector or matrix against the covariate matrix."""
    covariate_coefficients = np.linalg.solve(covariate_matrix.T @ covariate_matrix, covariate_matrix.T @ value_array)
    return value_array - covariate_matrix @ covariate_coefficients


def compute_regenie_null_mse_formula(
    *,
    covariate_matrix: npt.NDArray[np.float64],
    adjusted_residual: npt.NDArray[np.float64],
    genotype_matrix: npt.NDArray[np.float64],
) -> LinearFormulaResult:
    """Compute REGENIE default QT score statistics for a chosen adjusted residual."""
    genotype_residual_matrix = residualize_against_covariates(covariate_matrix, genotype_matrix)
    genotype_residual_sum_squares = np.einsum("ij,ij->j", genotype_residual_matrix, genotype_residual_matrix)
    covariance_with_phenotype = genotype_residual_matrix.T @ adjusted_residual
    residualized_adjusted_residual = residualize_against_covariates(covariate_matrix, adjusted_residual)
    adjusted_residual_sum_squares = float(residualized_adjusted_residual @ residualized_adjusted_residual)
    null_degrees_of_freedom = covariate_matrix.shape[0] - covariate_matrix.shape[1]

    beta = covariance_with_phenotype / genotype_residual_sum_squares
    standard_error = np.sqrt(adjusted_residual_sum_squares / null_degrees_of_freedom / genotype_residual_sum_squares)
    chi_squared = np.square(beta / standard_error)
    log10_p_value = np.asarray(
        pvalue.chi_squared_to_log10_p_value(jnp.asarray(chi_squared, dtype=jnp.float32)),
        dtype=np.float64,
    )
    return LinearFormulaResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
    )


class TestPrepareRegenie2LinearState:
    """Tests for prepare_regenie2_linear_state."""

    def test_creates_valid_state(self) -> None:
        """Ensure state preparation creates valid projection components."""
        sample_count = 100
        covariate_count = 3

        covariate_matrix = jnp.ones((sample_count, covariate_count), dtype=jnp.float32)
        covariate_matrix = covariate_matrix.at[:, 1].set(jnp.arange(sample_count, dtype=jnp.float32))
        covariate_matrix = covariate_matrix.at[:, 2].set(jnp.arange(sample_count, dtype=jnp.float32) ** 2)

        phenotype_vector = jnp.arange(sample_count, dtype=jnp.float32) + 0.5

        state = regenie2_linear.prepare_regenie2_linear_state(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
        )

        assert state.whitened_covariate_transpose.shape == (covariate_count, sample_count)
        assert state.phenotype_residual.shape == (sample_count,)
        assert float(state.degrees_of_freedom) == sample_count - covariate_count

    def test_phenotype_residual_orthogonal_to_covariates(self) -> None:
        """Ensure phenotype residual is orthogonal to covariate space."""
        sample_count = 100
        covariate_count = 2

        rng = np.random.default_rng(42)
        covariate_matrix = jnp.array(rng.standard_normal((sample_count, covariate_count)), dtype=jnp.float32)
        phenotype_vector = jnp.array(rng.standard_normal(sample_count), dtype=jnp.float32)

        state = regenie2_linear.prepare_regenie2_linear_state(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
        )

        crossproduct = covariate_matrix.T @ state.phenotype_residual
        numpy.testing.assert_allclose(crossproduct, jnp.zeros(covariate_count), atol=1e-4)


class TestChiSquaredToLog10PValue:
    """Tests for chi_squared_to_log10_p_value."""

    def test_known_values(self) -> None:
        """Validate against known chi-squared to p-value conversions."""
        chi_squared = jnp.array([3.841, 6.635, 10.828], dtype=jnp.float32)

        log10_p = pvalue.chi_squared_to_log10_p_value(chi_squared)

        numpy.testing.assert_allclose(log10_p[0], 1.30103, atol=0.01)
        numpy.testing.assert_allclose(log10_p[1], 2.0, atol=0.01)
        numpy.testing.assert_allclose(log10_p[2], 3.0, atol=0.01)

    def test_zero_chi_squared(self) -> None:
        """Ensure zero chi-squared gives zero log10 p-value."""
        chi_squared = jnp.array([0.0], dtype=jnp.float32)

        log10_p = pvalue.chi_squared_to_log10_p_value(chi_squared)

        numpy.testing.assert_allclose(log10_p[0], 0.0, atol=1e-6)

    def test_large_chi_squared(self) -> None:
        """Ensure large chi-squared values don't overflow."""
        chi_squared = jnp.array([100.0, 200.0], dtype=jnp.float32)

        log10_p = pvalue.chi_squared_to_log10_p_value(chi_squared)

        assert jnp.all(jnp.isfinite(log10_p))
        assert log10_p[1] > log10_p[0]


class TestComputeRegenie2LinearChunk:
    """Tests for compute_regenie2_linear_chunk."""

    def test_variant_major_kernel_matches_native_sum_square_stats_path(self) -> None:
        """Ensure native genotype statistics reproduce normalized genotype sum squares."""
        sample_count = 5
        variant_count = 3
        genotype_matrix_by_variant = jnp.asarray(
            [
                [0.0, 0.0, 1.0, 2.0, 0.0],
                [2.0, 2.0, 1.0, 2.0, 2.0],
                [2.0, 2.0, 1.75, 2.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        native_dosage_sum = jnp.asarray([3.0, 9.0, 7.0], dtype=jnp.float32)
        native_observation_count = jnp.asarray([5, 5, 4], dtype=jnp.int32)
        native_imputed_dosage_square_sum = jnp.asarray([5.0, 17.0, 16.0625], dtype=jnp.float32)
        normalized_genotype_matrix_by_variant = genotype.normalize_high_frequency_diploid_genotypes_variant_major(
            genotype_matrix_by_variant
        )
        expected_sum_squares = jnp.einsum(
            "ij,ij->i",
            normalized_genotype_matrix_by_variant,
            normalized_genotype_matrix_by_variant,
        )
        observed_sum_squares = regenie2_linear_score.compute_normalized_genotype_sum_squares_from_stats(
            genotype_dosage_sum=native_dosage_sum,
            genotype_observation_count=native_observation_count,
            genotype_imputed_dosage_square_sum=native_imputed_dosage_square_sum,
            sample_count=genotype_matrix_by_variant.shape[1],
        )

        numpy.testing.assert_allclose(observed_sum_squares, expected_sum_squares, rtol=1e-6, atol=1e-6)

        whitened_covariate_transpose = jnp.asarray([[0.0] * sample_count], dtype=jnp.float32)
        adjusted_residual_matrix = jnp.asarray([[0.2, -0.1, 0.4, -0.3, 0.5]], dtype=jnp.float32)
        adjusted_residual_projection_coordinate_matrix = jnp.asarray([[0.0]], dtype=jnp.float32)
        adjusted_residual_sum_squares = jnp.asarray([0.55], dtype=jnp.float32)
        degrees_of_freedom = jnp.asarray(sample_count - 1, dtype=jnp.float32)
        fallback_result = regenie2_linear_score.compute_regenie2_linear_chunk_trait_major_variant_major(
            whitened_covariate_transpose=whitened_covariate_transpose,
            adjusted_residual_matrix=adjusted_residual_matrix,
            adjusted_residual_projection_coordinate_matrix=adjusted_residual_projection_coordinate_matrix,
            adjusted_residual_sum_squares=adjusted_residual_sum_squares,
            degrees_of_freedom=degrees_of_freedom,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
        )
        stats_result = regenie2_linear_score.compute_regenie2_linear_chunk_trait_major_variant_major(
            whitened_covariate_transpose=whitened_covariate_transpose,
            adjusted_residual_matrix=adjusted_residual_matrix,
            adjusted_residual_projection_coordinate_matrix=adjusted_residual_projection_coordinate_matrix,
            adjusted_residual_sum_squares=adjusted_residual_sum_squares,
            degrees_of_freedom=degrees_of_freedom,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            genotype_dosage_sum=native_dosage_sum,
            genotype_observation_count=native_observation_count,
            genotype_imputed_dosage_square_sum=native_imputed_dosage_square_sum,
        )

        assert fallback_result.beta.shape == (1, variant_count)
        numpy.testing.assert_allclose(stats_result.beta, fallback_result.beta, rtol=1e-6, atol=1e-6)
        numpy.testing.assert_allclose(
            stats_result.standard_error,
            fallback_result.standard_error,
            rtol=1e-6,
            atol=1e-6,
        )
        numpy.testing.assert_allclose(stats_result.chi_squared, fallback_result.chi_squared, rtol=1e-6, atol=1e-6)
        numpy.testing.assert_allclose(stats_result.log10_p_value, fallback_result.log10_p_value, rtol=1e-6, atol=1e-6)
        numpy.testing.assert_array_equal(stats_result.valid_mask, fallback_result.valid_mask)

    def test_matches_manual_calculation(self) -> None:
        """Validate chunk computation against manual numpy calculation."""
        sample_count = 100
        variant_count = 5
        covariate_count = 2

        rng = np.random.default_rng(42)

        covariate_matrix = np.zeros((sample_count, covariate_count), dtype=np.float32)
        covariate_matrix[:, 0] = 1.0
        covariate_matrix[:, 1] = rng.standard_normal(sample_count).astype(np.float32)
        covariate_matrix = jnp.array(covariate_matrix)

        phenotype_vector = jnp.array(rng.standard_normal(sample_count), dtype=jnp.float32)

        genotype_matrix = jnp.array(rng.choice([0, 1, 2], size=(sample_count, variant_count)).astype(np.float32))

        loco_predictions = jnp.array(rng.standard_normal(sample_count) * 0.1, dtype=jnp.float32)

        state = regenie2_linear.prepare_regenie2_linear_state(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
        )

        result = regenie2_linear.compute_regenie2_linear_chunk(
            state=state,
            genotype_matrix=genotype_matrix,
            loco_predictions=loco_predictions,
        )

        assert result.beta.shape == (variant_count,)
        assert result.standard_error.shape == (variant_count,)
        assert result.chi_squared.shape == (variant_count,)
        assert result.log10_p_value.shape == (variant_count,)
        assert result.valid_mask.shape == (variant_count,)

        assert jnp.all(result.valid_mask)
        assert jnp.all(result.chi_squared >= 0)
        assert jnp.all(result.log10_p_value >= 0)

    def test_optimized_kernel_matches_score_reference_formula(self) -> None:
        """Ensure stacked-score optimization preserves the score statistic formula."""
        sample_count = 128
        variant_count = 8
        covariate_count = 3

        rng = np.random.default_rng(19)
        covariate_matrix = np.ones((sample_count, covariate_count), dtype=np.float32)
        covariate_matrix[:, 1] = rng.standard_normal(sample_count).astype(np.float32)
        covariate_matrix[:, 2] = rng.standard_normal(sample_count).astype(np.float32)
        phenotype_vector = jnp.array(rng.standard_normal(sample_count), dtype=jnp.float32)
        genotype_matrix = rng.choice([0, 1, 2], size=(sample_count, variant_count)).astype(np.float32)
        genotype_matrix[:, 0] = 0.0
        genotype_matrix = jnp.array(genotype_matrix)
        loco_predictions = jnp.array(rng.standard_normal(sample_count) * 0.2, dtype=jnp.float32)
        state = regenie2_linear.prepare_regenie2_linear_state(
            covariate_matrix=jnp.array(covariate_matrix),
            phenotype_vector=phenotype_vector,
        )
        optimized_result = regenie2_linear.compute_regenie2_linear_chunk(
            state=state,
            genotype_matrix=genotype_matrix,
            loco_predictions=loco_predictions,
        )
        reference_result = compute_score_reference_chunk(
            state=state,
            covariate_matrix=jnp.asarray(covariate_matrix),
            genotype_matrix=genotype_matrix,
            loco_predictions=loco_predictions,
        )

        numpy.testing.assert_allclose(optimized_result.beta, reference_result.beta, rtol=1e-4, atol=1e-5)
        numpy.testing.assert_allclose(
            optimized_result.standard_error,
            reference_result.standard_error,
            rtol=1e-4,
            atol=1e-5,
        )
        numpy.testing.assert_allclose(optimized_result.chi_squared, reference_result.chi_squared, rtol=1e-4, atol=1e-5)
        numpy.testing.assert_allclose(
            optimized_result.log10_p_value,
            reference_result.log10_p_value,
            rtol=1e-4,
            atol=1e-5,
        )
        numpy.testing.assert_array_equal(optimized_result.valid_mask, reference_result.valid_mask)

    def test_score_dtype_float64_controls_linear_score_kernel_dtype(self) -> None:
        """Ensure the linear score path honors an explicit float64 policy."""
        covariate_matrix = jnp.asarray(
            [
                [1.0, -1.0],
                [1.0, -0.5],
                [1.0, 0.5],
                [1.0, 1.0],
            ],
            dtype=jnp.float64,
        )
        phenotype_vector = jnp.asarray([0.1, -0.2, 0.3, 0.7], dtype=jnp.float64)
        genotype_matrix = jnp.asarray(
            [
                [0.0, 2.0],
                [1.0, 2.0],
                [1.0, 1.0],
                [2.0, 0.0],
            ],
            dtype=jnp.float64,
        )
        loco_predictions = jnp.asarray([0.01, -0.02, 0.03, -0.01], dtype=jnp.float64)

        state = regenie2_linear.prepare_regenie2_linear_state(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            score_dtype=types.FloatingPointDtype.FLOAT64,
        )
        result = regenie2_linear.compute_regenie2_linear_chunk(
            state=state,
            genotype_matrix=genotype_matrix,
            loco_predictions=loco_predictions,
            score_dtype=types.FloatingPointDtype.FLOAT64,
        )

        assert state.phenotype_residual.dtype == jnp.float64
        assert result.beta.dtype == jnp.float64
        assert result.chi_squared.dtype == jnp.float64

    def test_chromosome_state_matches_direct_chunk_api(self) -> None:
        """Ensure chromosome-cached computation matches the compatibility wrapper."""
        sample_count = 64
        variant_count = 4
        covariate_count = 2

        rng = np.random.default_rng(7)
        covariate_matrix = np.ones((sample_count, covariate_count), dtype=np.float32)
        covariate_matrix[:, 1] = rng.standard_normal(sample_count).astype(np.float32)
        phenotype_vector = jnp.array(rng.standard_normal(sample_count), dtype=jnp.float32)
        genotype_matrix = jnp.array(rng.choice([0, 1, 2], size=(sample_count, variant_count)).astype(np.float32))
        loco_predictions = jnp.array(rng.standard_normal(sample_count) * 0.2, dtype=jnp.float32)

        state = regenie2_linear.prepare_regenie2_linear_state(
            covariate_matrix=jnp.array(covariate_matrix),
            phenotype_vector=phenotype_vector,
        )
        chromosome_state = regenie2_linear.prepare_regenie2_linear_chromosome_state(state, loco_predictions)

        direct_result = regenie2_linear.compute_regenie2_linear_chunk(
            state=state,
            genotype_matrix=genotype_matrix,
            loco_predictions=loco_predictions,
        )
        cached_result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state(
            chromosome_state=chromosome_state,
            genotype_matrix=genotype_matrix,
        )

        numpy.testing.assert_allclose(direct_result.beta, cached_result.beta)
        numpy.testing.assert_allclose(direct_result.standard_error, cached_result.standard_error)
        numpy.testing.assert_allclose(direct_result.chi_squared, cached_result.chi_squared)
        numpy.testing.assert_allclose(direct_result.log10_p_value, cached_result.log10_p_value)
        numpy.testing.assert_array_equal(direct_result.valid_mask, cached_result.valid_mask)

    def test_variant_major_kernel_matches_sample_major_with_native_square_sums(self) -> None:
        """Ensure native-square-sum variant-major computation matches sample-major results."""
        sample_count = 96
        variant_count = 6
        covariate_count = 3

        rng = np.random.default_rng(23)
        covariate_matrix = np.ones((sample_count, covariate_count), dtype=np.float32)
        covariate_matrix[:, 1] = rng.standard_normal(sample_count).astype(np.float32)
        covariate_matrix[:, 2] = rng.standard_normal(sample_count).astype(np.float32)
        phenotype_vector = jnp.asarray(rng.standard_normal(sample_count), dtype=jnp.float32)
        genotype_matrix = jnp.asarray(
            rng.choice([0, 1, 2], size=(sample_count, variant_count)).astype(np.float32),
        )
        loco_predictions = jnp.asarray(rng.standard_normal(sample_count) * 0.1, dtype=jnp.float32)
        state = regenie2_linear.prepare_regenie2_linear_state(
            covariate_matrix=jnp.asarray(covariate_matrix),
            phenotype_vector=phenotype_vector,
        )
        chromosome_state = regenie2_linear.prepare_regenie2_linear_chromosome_state(state, loco_predictions)
        sample_major_result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state(
            chromosome_state=chromosome_state,
            genotype_matrix=genotype_matrix,
        )
        variant_major_result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state_variant_major(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=genotype_matrix.T,
        )

        numpy.testing.assert_allclose(variant_major_result.beta, sample_major_result.beta, rtol=1e-5, atol=1e-5)
        numpy.testing.assert_allclose(
            variant_major_result.standard_error,
            sample_major_result.standard_error,
            rtol=1e-5,
            atol=1e-5,
        )
        numpy.testing.assert_allclose(
            variant_major_result.chi_squared,
            sample_major_result.chi_squared,
            rtol=1e-5,
            atol=1e-5,
        )
        numpy.testing.assert_allclose(
            variant_major_result.log10_p_value,
            sample_major_result.log10_p_value,
            rtol=1e-5,
            atol=1e-5,
        )
        numpy.testing.assert_array_equal(variant_major_result.valid_mask, sample_major_result.valid_mask)

    def test_high_frequency_diploid_dosages_match_float64_reference(self) -> None:
        """Guard REGENIE parity for mostly-homozygous alternate dosage columns."""
        sample_count = 2504
        rng = np.random.default_rng(547528741)
        covariate_matrix = np.ones((sample_count, 3), dtype=np.float64)
        covariate_matrix[:, 1] = np.linspace(-1.0, 1.0, sample_count, dtype=np.float64)
        covariate_matrix[:, 2] = rng.normal(size=sample_count)
        phenotype_vector = rng.normal(size=sample_count)
        loco_predictions = rng.normal(scale=0.05, size=sample_count)
        genotype_matrix = np.full((sample_count, 1), 2.0, dtype=np.float64)
        genotype_matrix[:5, 0] = 1.0

        phenotype_residual = residualize_against_covariates(covariate_matrix, phenotype_vector)
        reference_result = compute_regenie_null_mse_formula(
            covariate_matrix=covariate_matrix,
            adjusted_residual=phenotype_residual - loco_predictions,
            genotype_matrix=genotype_matrix,
        )
        state = regenie2_linear.prepare_regenie2_linear_state(
            covariate_matrix=jnp.asarray(covariate_matrix, dtype=jnp.float32),
            phenotype_vector=jnp.asarray(phenotype_vector, dtype=jnp.float32),
        )
        chromosome_state = regenie2_linear.prepare_regenie2_linear_chromosome_state(
            state,
            jnp.asarray(loco_predictions, dtype=jnp.float32),
        )
        genotype_matrix_float32 = jnp.asarray(genotype_matrix, dtype=jnp.float32)

        sample_major_result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state(
            chromosome_state=chromosome_state,
            genotype_matrix=genotype_matrix_float32,
        )
        variant_major_result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state_variant_major(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=genotype_matrix_float32.T,
        )

        numpy.testing.assert_allclose(sample_major_result.beta, reference_result.beta, rtol=1e-5, atol=1e-6)
        numpy.testing.assert_allclose(variant_major_result.beta, reference_result.beta, rtol=1e-5, atol=1e-6)
        numpy.testing.assert_allclose(
            sample_major_result.standard_error,
            reference_result.standard_error,
            rtol=1e-5,
            atol=1e-6,
        )
        numpy.testing.assert_allclose(
            variant_major_result.standard_error,
            reference_result.standard_error,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_handles_zero_variance_genotypes(self) -> None:
        """Ensure monomorphic variants are marked invalid."""
        sample_count = 50
        covariate_count = 2

        rng = np.random.default_rng(42)

        covariate_matrix = np.ones((sample_count, covariate_count), dtype=np.float32)
        covariate_matrix[:, 1] = rng.standard_normal(sample_count).astype(np.float32)
        covariate_matrix = jnp.array(covariate_matrix)

        phenotype_vector = jnp.array(rng.standard_normal(sample_count), dtype=jnp.float32)

        genotype_matrix = jnp.zeros((sample_count, 2), dtype=jnp.float32)
        genotype_matrix = genotype_matrix.at[:, 0].set(0.0)
        genotype_matrix = genotype_matrix.at[:, 1].set(rng.choice([0, 1, 2], size=sample_count).astype(np.float32))

        loco_predictions = jnp.zeros(sample_count, dtype=jnp.float32)

        state = regenie2_linear.prepare_regenie2_linear_state(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
        )

        result = regenie2_linear.compute_regenie2_linear_chunk(
            state=state,
            genotype_matrix=genotype_matrix,
            loco_predictions=loco_predictions,
        )

        assert not result.valid_mask[0]
        assert result.valid_mask[1]
        assert jnp.isnan(result.chi_squared[0])
        assert jnp.isnan(result.log10_p_value[0])
        assert jnp.isfinite(result.chi_squared[1])
        assert jnp.isfinite(result.log10_p_value[1])

    def test_invalid_variants_emit_nan_statistics_for_sample_and_variant_major_paths(self) -> None:
        """Ensure invalid quantitative rows do not look like valid null associations."""
        sample_count = 32
        covariate_count = 2

        rng = np.random.default_rng(17)
        covariate_matrix = np.ones((sample_count, covariate_count), dtype=np.float32)
        covariate_matrix[:, 1] = rng.standard_normal(sample_count).astype(np.float32)
        phenotype_vector = jnp.asarray(rng.standard_normal(sample_count), dtype=jnp.float32)
        genotype_matrix = jnp.asarray(
            np.column_stack(
                [
                    np.zeros(sample_count, dtype=np.float32),
                    rng.choice([0, 1, 2], size=sample_count).astype(np.float32),
                ]
            ),
            dtype=jnp.float32,
        )
        loco_predictions = jnp.zeros(sample_count, dtype=jnp.float32)

        state = regenie2_linear.prepare_regenie2_linear_state(
            covariate_matrix=jnp.asarray(covariate_matrix),
            phenotype_vector=phenotype_vector,
        )
        chromosome_state = regenie2_linear.prepare_regenie2_linear_chromosome_state(state, loco_predictions)
        sample_major_result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state(
            chromosome_state=chromosome_state,
            genotype_matrix=genotype_matrix,
        )
        variant_major_result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state_variant_major(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=genotype_matrix.T,
        )

        numpy.testing.assert_array_equal(np.asarray(sample_major_result.valid_mask), np.asarray([False, True]))
        numpy.testing.assert_array_equal(np.asarray(variant_major_result.valid_mask), np.asarray([False, True]))
        assert jnp.isnan(sample_major_result.chi_squared[0])
        assert jnp.isnan(sample_major_result.log10_p_value[0])
        assert jnp.isnan(variant_major_result.chi_squared[0])
        assert jnp.isnan(variant_major_result.log10_p_value[0])
        assert jnp.isfinite(sample_major_result.chi_squared[1])
        assert jnp.isfinite(sample_major_result.log10_p_value[1])
        assert jnp.isfinite(variant_major_result.chi_squared[1])
        assert jnp.isfinite(variant_major_result.log10_p_value[1])

    def test_invalid_multi_trait_variants_emit_nan_statistics(self) -> None:
        """Ensure multi-trait invalid quantitative rows emit NaN statistics."""
        sample_count = 40
        covariate_count = 2
        trait_count = 2

        rng = np.random.default_rng(23)
        covariate_matrix = np.ones((sample_count, covariate_count), dtype=np.float32)
        covariate_matrix[:, 1] = rng.standard_normal(sample_count).astype(np.float32)
        phenotype_matrix = jnp.asarray(rng.standard_normal((trait_count, sample_count)), dtype=jnp.float32)
        genotype_matrix = jnp.asarray(
            np.column_stack(
                [
                    np.zeros(sample_count, dtype=np.float32),
                    rng.choice([0, 1, 2], size=sample_count).astype(np.float32),
                ]
            ),
            dtype=jnp.float32,
        )
        loco_prediction_matrix = jnp.zeros((trait_count, sample_count), dtype=jnp.float32)

        multi_state = regenie2_linear.prepare_regenie2_multi_linear_state(
            covariate_matrix=jnp.asarray(covariate_matrix),
            phenotype_matrix=phenotype_matrix,
        )
        multi_chromosome_state = regenie2_linear.prepare_regenie2_multi_linear_chromosome_state(
            multi_state,
            loco_prediction_matrix,
        )
        sample_major_result = regenie2_linear.compute_regenie2_multi_linear_chunk_from_chromosome_state(
            multi_chromosome_state,
            genotype_matrix,
        )
        variant_major_result = regenie2_linear.compute_regenie2_multi_linear_chunk_from_chromosome_state_variant_major(
            chromosome_state=multi_chromosome_state,
            genotype_matrix_by_variant=genotype_matrix.T,
        )

        numpy.testing.assert_array_equal(
            np.asarray(sample_major_result.valid_mask),
            np.asarray([[False, True], [False, True]]),
        )
        numpy.testing.assert_array_equal(
            np.asarray(variant_major_result.valid_mask),
            np.asarray([[False, True], [False, True]]),
        )
        assert jnp.all(jnp.isnan(sample_major_result.chi_squared[:, 0]))
        assert jnp.all(jnp.isnan(sample_major_result.log10_p_value[:, 0]))
        assert jnp.all(jnp.isnan(variant_major_result.chi_squared[:, 0]))
        assert jnp.all(jnp.isnan(variant_major_result.log10_p_value[:, 0]))
        assert jnp.all(jnp.isfinite(sample_major_result.chi_squared[:, 1]))
        assert jnp.all(jnp.isfinite(sample_major_result.log10_p_value[:, 1]))
        assert jnp.all(jnp.isfinite(variant_major_result.chi_squared[:, 1]))
        assert jnp.all(jnp.isfinite(variant_major_result.log10_p_value[:, 1]))

    def test_loco_adjustment_affects_results(self) -> None:
        """Ensure LOCO predictions affect the association statistics."""
        sample_count = 100
        covariate_count = 2
        variant_count = 3

        rng = np.random.default_rng(42)

        covariate_matrix = np.ones((sample_count, covariate_count), dtype=np.float32)
        covariate_matrix[:, 1] = rng.standard_normal(sample_count).astype(np.float32)
        covariate_matrix = jnp.array(covariate_matrix)

        phenotype_vector = jnp.array(rng.standard_normal(sample_count), dtype=jnp.float32)

        genotype_matrix = jnp.array(rng.choice([0, 1, 2], size=(sample_count, variant_count)).astype(np.float32))

        state = regenie2_linear.prepare_regenie2_linear_state(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
        )

        result_no_loco = regenie2_linear.compute_regenie2_linear_chunk(
            state=state,
            genotype_matrix=genotype_matrix,
            loco_predictions=jnp.zeros(sample_count, dtype=jnp.float32),
        )

        loco_predictions = jnp.array(rng.standard_normal(sample_count), dtype=jnp.float32)
        result_with_loco = regenie2_linear.compute_regenie2_linear_chunk(
            state=state,
            genotype_matrix=genotype_matrix,
            loco_predictions=loco_predictions,
        )

        assert not jnp.allclose(result_no_loco.beta, result_with_loco.beta)
        assert not jnp.allclose(result_no_loco.chi_squared, result_with_loco.chi_squared)

    def test_loco_predictions_with_covariate_signal_residualize_null_mse(self) -> None:
        """Lock down null MSE when LOCO predictions are not covariate-orthogonal."""
        fixture = build_loco_covariate_fixture()
        phenotype_residual = residualize_against_covariates(
            fixture.covariate_matrix,
            fixture.phenotype_vector,
        )
        current_order_residual = phenotype_residual - fixture.loco_predictions
        alternative_order_residual = residualize_against_covariates(
            fixture.covariate_matrix,
            fixture.phenotype_vector - fixture.loco_predictions,
        )

        current_order_result = compute_regenie_null_mse_formula(
            covariate_matrix=fixture.covariate_matrix,
            adjusted_residual=current_order_residual,
            genotype_matrix=fixture.genotype_matrix,
        )
        alternative_order_result = compute_regenie_null_mse_formula(
            covariate_matrix=fixture.covariate_matrix,
            adjusted_residual=alternative_order_residual,
            genotype_matrix=fixture.genotype_matrix,
        )

        numpy.testing.assert_allclose(current_order_result.beta, fixture.expected_beta, rtol=1e-5, atol=1e-7)
        numpy.testing.assert_allclose(
            current_order_result.standard_error,
            fixture.expected_standard_error,
            rtol=1e-5,
            atol=1e-7,
        )
        numpy.testing.assert_allclose(
            current_order_result.chi_squared,
            fixture.expected_chi_squared,
            rtol=1e-5,
            atol=1e-7,
        )
        numpy.testing.assert_allclose(
            current_order_result.log10_p_value,
            fixture.expected_log10_p_value,
            rtol=1e-5,
            atol=1e-7,
        )
        numpy.testing.assert_allclose(
            alternative_order_result.standard_error,
            fixture.expected_standard_error,
            rtol=1e-5,
            atol=1e-7,
        )
        numpy.testing.assert_allclose(
            alternative_order_result.chi_squared,
            fixture.expected_chi_squared,
            rtol=1e-5,
            atol=1e-7,
        )

        state = regenie2_linear.prepare_regenie2_linear_state(
            covariate_matrix=jnp.asarray(fixture.covariate_matrix, dtype=jnp.float32),
            phenotype_vector=jnp.asarray(fixture.phenotype_vector, dtype=jnp.float32),
        )
        observed_result = regenie2_linear.compute_regenie2_linear_chunk(
            state=state,
            genotype_matrix=jnp.asarray(fixture.genotype_matrix, dtype=jnp.float32),
            loco_predictions=jnp.asarray(fixture.loco_predictions, dtype=jnp.float32),
        )

        numpy.testing.assert_allclose(observed_result.beta, fixture.expected_beta, rtol=1e-5, atol=1e-6)
        numpy.testing.assert_allclose(
            observed_result.standard_error,
            fixture.expected_standard_error,
            rtol=1e-5,
            atol=1e-6,
        )
        numpy.testing.assert_allclose(
            observed_result.chi_squared,
            fixture.expected_chi_squared,
            rtol=1e-5,
            atol=1e-6,
        )
        numpy.testing.assert_allclose(
            observed_result.log10_p_value,
            fixture.expected_log10_p_value,
            rtol=1e-5,
            atol=1e-6,
        )

        chromosome_state = regenie2_linear.prepare_regenie2_linear_chromosome_state(
            state,
            jnp.asarray(fixture.loco_predictions, dtype=jnp.float32),
        )
        genotype_matrix = jnp.asarray(fixture.genotype_matrix, dtype=jnp.float32)
        variant_major_result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state_variant_major(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=genotype_matrix.T,
        )
        numpy.testing.assert_allclose(variant_major_result.beta, fixture.expected_beta, rtol=1e-5, atol=1e-6)
        numpy.testing.assert_allclose(
            variant_major_result.standard_error,
            fixture.expected_standard_error,
            rtol=1e-5,
            atol=1e-6,
        )
        numpy.testing.assert_allclose(
            variant_major_result.chi_squared,
            fixture.expected_chi_squared,
            rtol=1e-5,
            atol=1e-6,
        )
        numpy.testing.assert_allclose(
            variant_major_result.log10_p_value,
            fixture.expected_log10_p_value,
            rtol=1e-5,
            atol=1e-6,
        )

        multi_state = regenie2_linear.prepare_regenie2_multi_linear_state(
            covariate_matrix=jnp.asarray(fixture.covariate_matrix, dtype=jnp.float32),
            phenotype_matrix=jnp.asarray(fixture.phenotype_vector[None, :], dtype=jnp.float32),
        )
        multi_chromosome_state = regenie2_linear.prepare_regenie2_multi_linear_chromosome_state(
            multi_state,
            jnp.asarray(fixture.loco_predictions[None, :], dtype=jnp.float32),
        )
        multi_result = regenie2_linear.compute_regenie2_multi_linear_chunk_from_chromosome_state(
            multi_chromosome_state,
            genotype_matrix,
        )
        numpy.testing.assert_allclose(multi_result.beta[0], fixture.expected_beta, rtol=1e-5, atol=1e-6)
        numpy.testing.assert_allclose(
            multi_result.standard_error[0],
            fixture.expected_standard_error,
            rtol=1e-5,
            atol=1e-6,
        )
        numpy.testing.assert_allclose(
            multi_result.chi_squared[0],
            fixture.expected_chi_squared,
            rtol=1e-5,
            atol=1e-6,
        )
        numpy.testing.assert_allclose(
            multi_result.log10_p_value[0],
            fixture.expected_log10_p_value,
            rtol=1e-5,
            atol=1e-6,
        )
        multi_variant_major_result = (
            regenie2_linear.compute_regenie2_multi_linear_chunk_from_chromosome_state_variant_major(
                chromosome_state=multi_chromosome_state,
                genotype_matrix_by_variant=genotype_matrix.T,
            )
        )
        numpy.testing.assert_allclose(multi_variant_major_result.beta[0], fixture.expected_beta, rtol=1e-5, atol=1e-6)
        numpy.testing.assert_allclose(
            multi_variant_major_result.standard_error[0],
            fixture.expected_standard_error,
            rtol=1e-5,
            atol=1e-6,
        )
        numpy.testing.assert_allclose(
            multi_variant_major_result.chi_squared[0],
            fixture.expected_chi_squared,
            rtol=1e-5,
            atol=1e-6,
        )
        numpy.testing.assert_allclose(
            multi_variant_major_result.log10_p_value[0],
            fixture.expected_log10_p_value,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_multi_trait_kernel_matches_stacked_single_trait_results(self) -> None:
        """Ensure multi-trait computation matches stacked single-trait computation."""
        sample_count = 96
        variant_count = 6
        covariate_count = 3
        trait_count = 2

        rng = np.random.default_rng(31)
        covariate_matrix = np.ones((sample_count, covariate_count), dtype=np.float32)
        covariate_matrix[:, 1] = rng.standard_normal(sample_count).astype(np.float32)
        covariate_matrix[:, 2] = rng.standard_normal(sample_count).astype(np.float32)
        phenotype_matrix = jnp.asarray(rng.standard_normal((trait_count, sample_count)), dtype=jnp.float32)
        genotype_matrix = jnp.asarray(
            rng.choice([0, 1, 2], size=(sample_count, variant_count)).astype(np.float32),
            dtype=jnp.float32,
        )
        loco_prediction_matrix = jnp.asarray(
            rng.standard_normal((trait_count, sample_count)) * 0.1,
            dtype=jnp.float32,
        )

        multi_state = regenie2_linear.prepare_regenie2_multi_linear_state(
            covariate_matrix=jnp.asarray(covariate_matrix),
            phenotype_matrix=phenotype_matrix,
        )
        multi_chromosome_state = regenie2_linear.prepare_regenie2_multi_linear_chromosome_state(
            multi_state,
            loco_prediction_matrix,
        )
        multi_result = regenie2_linear.compute_regenie2_multi_linear_chunk_from_chromosome_state(
            multi_chromosome_state,
            genotype_matrix,
        )

        single_results = []
        for trait_index in range(trait_count):
            single_state = regenie2_linear.prepare_regenie2_linear_state(
                covariate_matrix=jnp.asarray(covariate_matrix),
                phenotype_vector=phenotype_matrix[trait_index],
            )
            single_results.append(
                regenie2_linear.compute_regenie2_linear_chunk(
                    state=single_state,
                    genotype_matrix=genotype_matrix,
                    loco_predictions=loco_prediction_matrix[trait_index],
                )
            )

        numpy.testing.assert_allclose(
            np.asarray(multi_result.beta),
            np.stack([np.asarray(result.beta) for result in single_results], axis=0),
            rtol=1e-5,
            atol=1e-5,
        )
        numpy.testing.assert_allclose(
            np.asarray(multi_result.standard_error),
            np.stack([np.asarray(result.standard_error) for result in single_results], axis=0),
            rtol=1e-5,
            atol=1e-5,
        )
        numpy.testing.assert_allclose(
            np.asarray(multi_result.chi_squared),
            np.stack([np.asarray(result.chi_squared) for result in single_results], axis=0),
            rtol=1e-5,
            atol=1e-5,
        )
        numpy.testing.assert_allclose(
            np.asarray(multi_result.log10_p_value),
            np.stack([np.asarray(result.log10_p_value) for result in single_results], axis=0),
            rtol=1e-5,
            atol=1e-5,
        )


class TestSolvePositiveDefiniteSystem:
    """Tests for solve_positive_definite_system."""

    def test_solves_correctly(self) -> None:
        """Ensure the solver returns correct solutions."""
        rng = np.random.default_rng(42)
        matrix_a = rng.standard_normal((5, 5)).astype(np.float32)
        positive_definite = jnp.array(matrix_a.T @ matrix_a + 0.1 * np.eye(5), dtype=jnp.float32)
        right_hand_side = jnp.array(rng.standard_normal(5), dtype=jnp.float32)

        cholesky_factor = jnp.linalg.cholesky(positive_definite)
        solution = linalg.solve_positive_definite_system(cholesky_factor, right_hand_side)

        reconstructed = positive_definite @ solution
        numpy.testing.assert_allclose(reconstructed, right_hand_side, atol=1e-4)
