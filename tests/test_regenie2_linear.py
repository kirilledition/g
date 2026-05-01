"""Unit tests for REGENIE step 2 linear association kernel."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing

from g.compute import regenie2_linear

if typing.TYPE_CHECKING:
    from g import models


@dataclass(frozen=True)
class ReferenceRegenie2LinearChunkResult:
    """Reference result from the pre-optimization formula."""

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    valid_mask: jax.Array


def compute_legacy_reference_chunk(
    state: models.Regenie2LinearState,
    genotype_matrix: jax.Array,
    loco_predictions: jax.Array,
) -> ReferenceRegenie2LinearChunkResult:
    """Compute the pre-optimization formula for regression-test comparison."""
    adjusted_residual = state.phenotype_residual - loco_predictions
    adjusted_residual_sum_squares = jnp.dot(adjusted_residual, adjusted_residual)
    covariate_genotype_crossproduct = state.covariate_matrix_transpose @ genotype_matrix
    genotype_projection = regenie2_linear.solve_positive_definite_system(
        state.covariate_crossproduct_cholesky_factor,
        covariate_genotype_crossproduct,
    )
    genotype_sum_squares = jnp.einsum("ij,ij->j", genotype_matrix, genotype_matrix)
    projection_sum_squares = jnp.einsum("ij,ij->j", covariate_genotype_crossproduct, genotype_projection)
    genotype_residual_sum_squares = jnp.maximum(genotype_sum_squares - projection_sum_squares, 0.0)
    covariance_with_phenotype = genotype_matrix.T @ adjusted_residual
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
    residual_sum_squares_after = adjusted_residual_sum_squares - (
        covariance_squared * genotype_residual_sum_squares_inverse
    )
    residual_sum_squares_after = jnp.maximum(residual_sum_squares_after, 0.0)
    positive_residual_sum_squares_mask = residual_sum_squares_after > 0.0
    standard_error = jnp.where(
        positive_genotype_residual_mask & positive_residual_sum_squares_mask,
        jnp.sqrt(residual_sum_squares_after * genotype_residual_sum_squares_inverse / state.degrees_of_freedom),
        jnp.nan,
    )
    chi_squared = jnp.where(
        positive_genotype_residual_mask & positive_residual_sum_squares_mask,
        (
            covariance_squared
            * genotype_residual_sum_squares_inverse
            * state.degrees_of_freedom
            / residual_sum_squares_after
        ),
        0.0,
    )
    log10_p_value = regenie2_linear.chi_squared_to_log10_p_value(chi_squared)
    valid_mask = jnp.isfinite(beta) & jnp.isfinite(standard_error) & (standard_error > 0.0)
    return ReferenceRegenie2LinearChunkResult(
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        valid_mask=valid_mask,
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

        assert state.covariate_matrix.shape == (sample_count, covariate_count)
        assert state.covariate_matrix_transpose.shape == (covariate_count, sample_count)
        assert state.covariate_crossproduct_cholesky_factor.shape == (covariate_count, covariate_count)
        assert state.whitened_covariate_transpose.shape == (covariate_count, sample_count)
        assert state.phenotype_residual.shape == (sample_count,)
        assert int(state.sample_count) == sample_count
        assert float(state.degrees_of_freedom) == sample_count - covariate_count - 1

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

        crossproduct = state.covariate_matrix.T @ state.phenotype_residual
        numpy.testing.assert_allclose(crossproduct, jnp.zeros(covariate_count), atol=1e-4)


class TestChiSquaredToLog10PValue:
    """Tests for chi_squared_to_log10_p_value."""

    def test_known_values(self) -> None:
        """Validate against known chi-squared to p-value conversions."""
        chi_squared = jnp.array([3.841, 6.635, 10.828], dtype=jnp.float32)

        log10_p = regenie2_linear.chi_squared_to_log10_p_value(chi_squared)

        numpy.testing.assert_allclose(log10_p[0], 1.30103, atol=0.01)
        numpy.testing.assert_allclose(log10_p[1], 2.0, atol=0.01)
        numpy.testing.assert_allclose(log10_p[2], 3.0, atol=0.01)

    def test_zero_chi_squared(self) -> None:
        """Ensure zero chi-squared gives zero log10 p-value."""
        chi_squared = jnp.array([0.0], dtype=jnp.float32)

        log10_p = regenie2_linear.chi_squared_to_log10_p_value(chi_squared)

        numpy.testing.assert_allclose(log10_p[0], 0.0, atol=1e-6)

    def test_large_chi_squared(self) -> None:
        """Ensure large chi-squared values don't overflow."""
        chi_squared = jnp.array([100.0, 200.0], dtype=jnp.float32)

        log10_p = regenie2_linear.chi_squared_to_log10_p_value(chi_squared)

        assert jnp.all(jnp.isfinite(log10_p))
        assert log10_p[1] > log10_p[0]


class TestComputeRegenie2LinearChunk:
    """Tests for compute_regenie2_linear_chunk."""

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

    def test_optimized_kernel_matches_legacy_reference_formula(self) -> None:
        """Ensure stacked-score optimization preserves the previous formula."""
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
        reference_result = compute_legacy_reference_chunk(
            state=state,
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


class TestSolvePositiveDefiniteSystem:
    """Tests for solve_positive_definite_system."""

    def test_solves_correctly(self) -> None:
        """Ensure the solver returns correct solutions."""
        rng = np.random.default_rng(42)
        matrix_a = rng.standard_normal((5, 5)).astype(np.float32)
        positive_definite = jnp.array(matrix_a.T @ matrix_a + 0.1 * np.eye(5), dtype=jnp.float32)
        right_hand_side = jnp.array(rng.standard_normal(5), dtype=jnp.float32)

        cholesky_factor = jnp.linalg.cholesky(positive_definite)
        solution = regenie2_linear.solve_positive_definite_system(cholesky_factor, right_hand_side)

        reconstructed = positive_definite @ solution
        numpy.testing.assert_allclose(reconstructed, right_hand_side, atol=1e-4)
