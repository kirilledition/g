from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from g.compute.linear import compute_linear_association_chunk, prepare_linear_association_state


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
