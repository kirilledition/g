import jax.numpy as jnp
import numpy as np
import pytest

from g.compute.logistic import compute_covariate_only_probability_matrix, MINIMUM_PROBABILITY

def test_compute_covariate_only_probability_matrix():
    """Test batched covariate-only logistic probabilities computation."""
    # N = 3 samples, P = 2 covariates
    covariate_matrix = jnp.array([
        [1.0, 0.5],
        [1.0, -0.5],
        [1.0, 2.0],
    ])

    # M = 2 sets of coefficients, P = 2 covariates
    coefficients = jnp.array([
        [0.0, 0.0],   # Should result in 0.5 probability (sigmoid of 0)
        [10.0, 20.0], # Should result in probabilities close to 1, testing clipping
    ])

    result = compute_covariate_only_probability_matrix(covariate_matrix, coefficients)

    # Check shape is (M, N) = (2, 3)
    assert result.shape == (2, 3)

    # First set of coefficients is all zeros, so linear predictor is 0
    # sigmoid(0) = 0.5
    expected_prob_0 = jnp.array([0.5, 0.5, 0.5])
    np.testing.assert_allclose(result[0], expected_prob_0, rtol=1e-5)

    # Second set of coefficients:
    # [1.0, 0.5] @ [10.0, 20.0] = 10 + 10 = 20
    # [1.0, -0.5] @ [10.0, 20.0] = 10 - 10 = 0
    # [1.0, 2.0] @ [10.0, 20.0] = 10 + 40 = 50
    # sigmoid(20) ~ 1.0
    # sigmoid(0) = 0.5
    # sigmoid(50) ~ 1.0

    # Test clipping: since sigmoid(20) and sigmoid(50) are extremely close to 1,
    # they should be clipped to 1.0 - MINIMUM_PROBABILITY
    expected_prob_1 = jnp.array([
        1.0 - MINIMUM_PROBABILITY,
        0.5,
        1.0 - MINIMUM_PROBABILITY
    ])
    np.testing.assert_allclose(result[1], expected_prob_1, rtol=1e-5)

    # Test negative clipping
    coefficients_neg = jnp.array([
        [-100.0, -100.0]
    ])
    result_neg = compute_covariate_only_probability_matrix(covariate_matrix, coefficients_neg)

    # linear predictors: [-150, -50, -300]
    # probabilities should all clip to MINIMUM_PROBABILITY
    expected_prob_neg = jnp.array([
        MINIMUM_PROBABILITY,
        MINIMUM_PROBABILITY,
        MINIMUM_PROBABILITY
    ])
    np.testing.assert_allclose(result_neg[0], expected_prob_neg, rtol=1e-5)
