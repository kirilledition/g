"""Regressions for binary nuisance designs with realistic units and correlations."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

import tests.numerical
import tests.test_regenie2_binary as binary_reference
import tests.test_regenie2_binary_pipeline as firth_reference
from g.compute.regenie2_binary import score as regenie2_binary_score
from g.compute.regenie2_binary import state as regenie2_binary_state


@pytest.mark.parametrize("include_correlated_covariate", [False, True])
def test_large_binary_design_matches_independent_float64_reference(
    *,
    include_correlated_covariate: bool,
) -> None:
    """Retain age and a nearly redundant measured covariate in 100,000 samples."""
    sample_count = 100_000
    random_generator = np.random.default_rng(104729)
    age_coordinate = np.tile(np.asarray([-1.0, -1.0, 1.0, 1.0]), sample_count // 4)
    second_coordinate = np.tile(np.asarray([-1.0, 1.0, -1.0, 1.0]), sample_count // 4)
    ages = 50.0 + 10.0 * age_coordinate
    covariate_columns = [np.ones(sample_count), ages]
    reference_columns = [np.ones(sample_count), age_coordinate]
    if include_correlated_covariate:
        # The small difference is exactly representable in float32, but its
        # direction is lost by float32 normal equations on the original design.
        covariate_columns.append(ages + second_coordinate / 4096.0)
        reference_columns.append(second_coordinate)
    covariate_matrix = np.column_stack(covariate_columns).astype(np.float32)
    reference_covariates = np.column_stack(reference_columns)
    linear_predictor = -0.3 + 0.7 * age_coordinate
    if include_correlated_covariate:
        linear_predictor += 0.4 * second_coordinate
    probability = np.reciprocal(1.0 + np.exp(-linear_predictor))
    phenotype_matrix = random_generator.binomial(1, probability).astype(np.float64)[None, :]
    genotype_matrix = random_generator.binomial(
        2,
        np.asarray([0.05, 0.3, 0.8])[:, None],
        size=(3, sample_count),
    ).astype(np.float64)
    loco_offset_matrix = np.zeros_like(phenotype_matrix)
    kernel_config = binary_reference.build_binary_score_config()
    reference_fixture = binary_reference.BinaryFixture(
        covariate_matrix=reference_covariates,
        phenotype_matrix=phenotype_matrix,
        loco_offset_matrix=loco_offset_matrix,
        genotype_matrix_by_variant=genotype_matrix,
    )
    null_reference = binary_reference.compute_numpy_null_logistic(
        reference_covariates,
        phenotype_matrix[0],
        loco_offset_matrix[0],
        kernel_config,
    )
    score_reference = binary_reference.compute_binary_score_reference(reference_fixture, kernel_config)
    shared_state = regenie2_binary_state.build_multi_binary_state(
        jnp.asarray(covariate_matrix),
        jnp.asarray(phenotype_matrix),
    )
    trait_state = regenie2_binary_state.prepare_binary_trait_state(
        shared_state.covariate_matrix,
        shared_state.phenotype_matrix[0],
        jnp.asarray(loco_offset_matrix[0]),
        kernel_config,
    )
    chromosome_state = regenie2_binary_state.build_multi_binary_score_chromosome_state(
        shared_state,
        jnp.asarray(loco_offset_matrix),
        kernel_config,
    )
    observed = regenie2_binary_score.compute_multi_binary_score_test_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=jnp.asarray(genotype_matrix, dtype=jnp.float32),
        firth_candidate_p_threshold=None,
        minimum_variance=kernel_config.numerical.minimum_variance,
        relative_variance_tolerance=kernel_config.numerical.relative_variance_tolerance,
        native_genotype_mean=None,
    )

    assert bool(trait_state.null_logistic_converged)
    assert shared_state.covariate_matrix.dtype == jnp.float32
    assert trait_state.null_logistic_coefficients.dtype == jnp.float64
    assert trait_state.score_residual.dtype == jnp.float32
    assert trait_state.score_projection_matrix.dtype == jnp.float32
    assert chromosome_state.score_right_hand_matrix.dtype == jnp.float32
    assert chromosome_state.bernoulli_weight.dtype == jnp.float32
    np.testing.assert_array_equal(shared_state.covariate_matrix[:, 0], np.ones(sample_count))
    tests.numerical.assert_absolute_difference_less_than(
        np.asarray(trait_state.score_residual),
        null_reference.score_residual,
        2.0e-6,
    )
    tests.numerical.assert_absolute_difference_less_than(
        np.asarray(trait_state.bernoulli_weight),
        null_reference.weight,
        5.0e-7,
    )
    # Sample reductions and final statistics remain float32; allow their
    # measured accumulation error without masking the original basis failure.
    tests.numerical.assert_absolute_difference_less_than(observed.beta, score_reference.beta, 2.0e-6)
    tests.numerical.assert_absolute_difference_less_than(
        observed.standard_error,
        score_reference.standard_error,
        2.0e-6,
    )
    tests.numerical.assert_absolute_difference_less_than(observed.chi_squared, score_reference.chi_squared, 2.0e-4)
    tests.numerical.assert_absolute_difference_less_than(observed.log10_p_value, score_reference.log10_p_value, 1.0e-4)


def test_binary_conditioning_preserves_firth_predictor_and_likelihood_ratio() -> None:
    """Share one nuisance basis through null logistic, Firth, and score preparation."""
    original_fixture = firth_reference.build_firth_pipeline_fixture()
    covariate_matrix = original_fixture.covariate_matrix.copy()
    covariate_matrix[:, 1] = 50.0 + 10.0 * covariate_matrix[:, 1]
    fixture = dataclasses.replace(original_fixture, covariate_matrix=covariate_matrix)
    kernel_config = firth_reference.build_binary_kernel_config(candidate_capacity=2, batch_size=2)
    kernel_config = dataclasses.replace(
        kernel_config,
        null_firth=dataclasses.replace(kernel_config.null_firth, gradient_tolerance=1.0e-6),
    )
    prepared = firth_reference.prepare_firth_pipeline(fixture=fixture, kernel_config=kernel_config)
    observed = firth_reference.run_production_firth_pipeline(
        prepared=prepared,
        firth_se=False,
        p_threshold=1.0,
        kernel_config=kernel_config,
        chromosome_state=prepared.chromosome_state,
    )
    reference_state = prepared.independent_trait_states[0]
    tests.numerical.assert_absolute_difference_less_than(
        prepared.chromosome_state.null_firth_offset_matrix[0],
        reference_state.null_firth_offset,
        2.0e-6,
    )
    for variant_index in range(fixture.genotype_matrix_by_variant.shape[0]):
        reference = firth_reference.compute_firth_reference(
            prepared=prepared,
            trait_index=0,
            variant_index=variant_index,
            sparse_correction=bool(fixture.sparse_candidate_mask[variant_index]),
        )
        tests.numerical.assert_absolute_difference_less_than(
            observed.association.beta[0, variant_index],
            reference.beta,
            firth_reference.FIRTH_BETA_ABSOLUTE_TOLERANCE,
        )
        tests.numerical.assert_absolute_difference_less_than(
            observed.association.chi_squared[0, variant_index],
            reference.chi_squared,
            firth_reference.FIRTH_CHI_SQUARED_ABSOLUTE_TOLERANCE,
        )


def test_binary_score_and_firth_preserve_large_constant_offset_invariance() -> None:
    """Retain fitted logits when a large LOCO intercept is absorbed by the design."""
    original_fixture = firth_reference.build_firth_pipeline_fixture()
    fixture = dataclasses.replace(
        original_fixture,
        phenotype_matrix=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
        loco_offset_matrix=np.zeros_like(original_fixture.loco_offset_matrix),
    )
    kernel_config = firth_reference.build_binary_kernel_config(candidate_capacity=2, batch_size=2)
    prepared = firth_reference.prepare_firth_pipeline(fixture=fixture, kernel_config=kernel_config)
    shared_state = regenie2_binary_state.build_multi_binary_state(
        jnp.asarray(fixture.covariate_matrix),
        jnp.asarray(fixture.phenotype_matrix),
    )
    shifted_offsets = jnp.full(fixture.loco_offset_matrix.shape, 1.0e8, dtype=jnp.float32)
    shifted_traits = regenie2_binary_state.prepare_binary_traits(shared_state, shifted_offsets, kernel_config)
    shifted_state = regenie2_binary_state.build_multi_binary_firth_chromosome_state(
        shared_state,
        shifted_offsets,
        kernel_config,
    )

    assert bool(shifted_traits.null_logistic_converged[0])
    np.testing.assert_array_equal(shifted_traits.loco_offset, np.zeros_like(fixture.loco_offset_matrix))
    np.testing.assert_array_equal(
        shifted_state.score_state.score_right_hand_matrix,
        prepared.chromosome_state.score_state.score_right_hand_matrix,
    )
    assert bool(jnp.all(jnp.isfinite(shifted_state.null_firth_offset_matrix)))
    np.testing.assert_array_equal(
        shifted_state.null_firth_offset_matrix,
        prepared.chromosome_state.null_firth_offset_matrix,
    )
    baseline_result = firth_reference.run_production_firth_pipeline(
        prepared=prepared,
        firth_se=False,
        p_threshold=1.0,
        kernel_config=kernel_config,
        chromosome_state=prepared.chromosome_state,
    )
    shifted_result = firth_reference.run_production_firth_pipeline(
        prepared=prepared,
        firth_se=False,
        p_threshold=1.0,
        kernel_config=kernel_config,
        chromosome_state=shifted_state,
    )
    np.testing.assert_array_equal(shifted_result.association.beta, baseline_result.association.beta)
    np.testing.assert_array_equal(shifted_result.association.chi_squared, baseline_result.association.chi_squared)


def test_binary_conditioning_preserves_intercept_only_design() -> None:
    """Keep the analytical intercept-only null probability unchanged."""
    phenotype_matrix = jnp.asarray([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    shared_state = regenie2_binary_state.build_multi_binary_state(jnp.ones((8, 1)), phenotype_matrix)
    trait_state = regenie2_binary_state.prepare_binary_trait_state(
        shared_state.covariate_matrix,
        shared_state.phenotype_matrix[0],
        jnp.zeros(8),
        binary_reference.build_binary_score_config(),
    )

    np.testing.assert_array_equal(shared_state.covariate_matrix, np.ones((8, 1)))
    assert bool(trait_state.null_logistic_converged)
    np.testing.assert_array_equal(trait_state.score_residual, np.asarray(phenotype_matrix[0]) - 0.375)


@pytest.mark.parametrize(
    "covariate_matrix",
    [
        np.column_stack([np.ones(8), np.arange(8), 10.0 + 2.0 * np.arange(8)]),
        np.ones((8, 2)),
        np.ones((8, 0)),
        np.ones((2, 2)),
        np.column_stack([np.full(8, 2.0), np.arange(8)]),
    ],
    ids=["affine-dependence", "duplicate-intercept", "missing-intercept", "saturated", "nonunit-intercept"],
)
def test_binary_conditioning_rejects_invalid_design(covariate_matrix: npt.NDArray[np.float64]) -> None:
    """Reject invalid designs before QR can create an arbitrary nuisance direction."""
    with pytest.raises(ValueError, match="Binary covariate design"):
        regenie2_binary_state.build_multi_binary_state(
            jnp.asarray(covariate_matrix),
            jnp.zeros((1, covariate_matrix.shape[0])),
        )
