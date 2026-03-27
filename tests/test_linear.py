from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from g.compute.linear import compute_linear_association_chunk, prepare_linear_association_state


def compute_expected_linear_statistics(
    covariate_matrix: np.ndarray,
    phenotype_vector: np.ndarray,
    genotype_vector: np.ndarray,
) -> tuple[float, float, float]:
    """Compute linear association statistics with NumPy residualization."""
    covariate_projection, *_ = np.linalg.lstsq(covariate_matrix, phenotype_vector, rcond=None)
    phenotype_residual = phenotype_vector - covariate_matrix @ covariate_projection

    genotype_projection, *_ = np.linalg.lstsq(covariate_matrix, genotype_vector, rcond=None)
    genotype_residual = genotype_vector - covariate_matrix @ genotype_projection

    genotype_residual_sum_squares = float(genotype_residual @ genotype_residual)
    beta = float((genotype_residual @ phenotype_residual) / genotype_residual_sum_squares)
    residual_sum_squares = float(phenotype_residual @ phenotype_residual) - beta * float(
        genotype_residual @ phenotype_residual
    )
    degrees_of_freedom = covariate_matrix.shape[0] - covariate_matrix.shape[1] - 1
    residual_variance = residual_sum_squares / degrees_of_freedom
    standard_error = float(np.sqrt(residual_variance / genotype_residual_sum_squares))
    test_statistic = beta / standard_error
    return beta, standard_error, test_statistic


def test_prepare_linear_association_state_basic() -> None:
    """Test basic linear association state preparation."""
    covariate_matrix = jnp.array(
        [
            [1.0, 0.5],
            [1.0, -0.5],
            [1.0, 2.0],
        ]
    )
    phenotype_vector = jnp.array([1.0, 2.0, 3.0])

    state = prepare_linear_association_state(covariate_matrix, phenotype_vector)

    assert state.covariate_matrix.shape == (3, 2)
    assert state.covariate_crossproduct_inverse.shape == (2, 2)
    assert state.phenotype_residual.shape == (3,)
    assert float(state.phenotype_residual_sum_squares) >= 0.0


def test_prepare_linear_association_state_perfect_fit() -> None:
    """Test state preparation when phenotype is perfectly predicted by covariates."""
    covariate_matrix = jnp.array(
        [
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )
    phenotype_vector = jnp.array([2.0, 4.0, 6.0])

    state = prepare_linear_association_state(covariate_matrix, phenotype_vector)

    assert float(state.phenotype_residual_sum_squares) < 1e-10


def test_compute_linear_association_chunk_basic() -> None:
    """Test basic linear association computation."""
    covariate_matrix = jnp.array(
        [
            [1.0, 0.5],
            [1.0, -0.5],
            [1.0, 2.0],
        ]
    )
    phenotype_vector = jnp.array([1.0, 2.0, 3.0])

    state = prepare_linear_association_state(covariate_matrix, phenotype_vector)

    genotype_matrix = jnp.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 0.0],
        ]
    )

    result = compute_linear_association_chunk(state, genotype_matrix)

    assert result.beta.shape == (2,)
    assert result.standard_error.shape == (2,)
    assert result.test_statistic.shape == (2,)
    assert result.p_value.shape == (2,)
    assert result.valid_mask.shape == (2,)


def test_compute_linear_association_chunk_zero_variance() -> None:
    """Test that constant genotypes are marked invalid."""
    covariate_matrix = jnp.array(
        [
            [1.0, 0.5],
            [1.0, -0.5],
            [1.0, 2.0],
        ]
    )
    phenotype_vector = jnp.array([1.0, 2.0, 3.0])

    state = prepare_linear_association_state(covariate_matrix, phenotype_vector)

    genotype_matrix = jnp.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    result = compute_linear_association_chunk(state, genotype_matrix)

    np.testing.assert_array_equal(result.valid_mask, jnp.array([False, False]))


def test_compute_linear_association_chunk_matches_numpy_residualization() -> None:
    """Ensure linear association statistics match a NumPy reference implementation."""
    covariate_matrix = np.array(
        [
            [1.0, 20.0],
            [1.0, 25.0],
            [1.0, 30.0],
            [1.0, 35.0],
            [1.0, 40.0],
        ]
    )
    phenotype_vector = np.array([2.0, 2.6, 3.8, 4.1, 5.2])
    genotype_matrix = np.array(
        [
            [0.0, 2.0],
            [1.0, 1.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [1.0, 1.0],
        ]
    )

    state = prepare_linear_association_state(jnp.asarray(covariate_matrix), jnp.asarray(phenotype_vector))
    result = compute_linear_association_chunk(state, jnp.asarray(genotype_matrix))

    expected_statistics = [
        compute_expected_linear_statistics(covariate_matrix, phenotype_vector, genotype_matrix[:, variant_index])
        for variant_index in range(genotype_matrix.shape[1])
    ]

    np.testing.assert_allclose(result.beta, np.array([item[0] for item in expected_statistics]), atol=1e-6)
    np.testing.assert_allclose(result.standard_error, np.array([item[1] for item in expected_statistics]), atol=1e-6)
    np.testing.assert_allclose(result.test_statistic, np.array([item[2] for item in expected_statistics]), atol=1e-6)
    np.testing.assert_array_equal(result.valid_mask, np.array([True, True]))
