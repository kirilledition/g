"""Exact dosage-invariance guards for binary score reductions."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import tests.numerical
import tests.test_regenie2_binary
from g import types
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import score as regenie2_binary_score
from g.compute.regenie2_binary import state as regenie2_binary_state


def assert_constant_dosages_are_untestable(
    observed: regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult,
    constant_variant_count: int,
) -> None:
    """Require invalid statistics and prevent constant variants becoming Firth candidates."""
    for values in (observed.beta, observed.standard_error, observed.chi_squared, observed.log10_p_value):
        assert bool(np.all(np.isnan(np.asarray(values)[:, :constant_variant_count])))
        assert bool(np.all(np.isfinite(np.asarray(values)[:, constant_variant_count:])))
    np.testing.assert_array_equal(
        np.asarray(observed.correction_code)[:, :constant_variant_count],
        np.full(
            (observed.beta.shape[0], constant_variant_count),
            types.BinaryCorrectionCode.SCORE_FAILED.value,
            dtype=np.uint8,
        ),
    )


@pytest.mark.parametrize("firth_candidate_p_threshold", [None, 1.0])
def test_binary_uniform_dense_dosages_are_untestable(firth_candidate_p_threshold: float | None) -> None:
    """Reject uniform integer and fractional dosages while retaining rare variants."""
    fixture = tests.test_regenie2_binary.build_binary_fixture()
    kernel_config = tests.test_regenie2_binary.build_binary_score_config()
    chromosome_state = tests.test_regenie2_binary.build_binary_chromosome_state(fixture, kernel_config)
    sample_count = fixture.covariate_matrix.shape[0]
    constant_dosages = np.asarray([0.0, 0.125, 0.5, 1.0, 1.5, 1.999, 2.0], dtype=np.float32)
    constant_variants = np.broadcast_to(constant_dosages[:, None], (constant_dosages.size, sample_count))
    rare_variants = np.zeros((2, sample_count), dtype=np.float32)
    rare_variants[0, -1] = 1.0
    rare_variants[1] = 2.0 - rare_variants[0]
    raw_genotypes = jnp.asarray(np.concatenate([constant_variants, rare_variants], axis=0))
    genotype_flip_mask = jnp.mean(raw_genotypes, axis=1) > 1.0
    materialized_genotypes = jnp.where(genotype_flip_mask[:, None], 2.0 - raw_genotypes, raw_genotypes)
    tiled_result = regenie2_binary_score.compute_multi_binary_score_test_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=raw_genotypes,
        firth_candidate_p_threshold=firth_candidate_p_threshold,
        minimum_variance=kernel_config.numerical.minimum_variance,
        relative_variance_tolerance=kernel_config.numerical.relative_variance_tolerance,
        native_genotype_mean=None,
    )
    materialized_result = regenie2_binary_score.build_multi_binary_score_result(
        chromosome_state=chromosome_state,
        score_reduction=regenie2_binary_score.reduce_materialized_score_genotypes(
            chromosome_state,
            materialized_genotypes,
        ),
        genotype_flip_mask=genotype_flip_mask,
        firth_candidate_p_threshold=firth_candidate_p_threshold,
        minimum_variance=kernel_config.numerical.minimum_variance,
        relative_variance_tolerance=kernel_config.numerical.relative_variance_tolerance,
    )
    reference_fixture = tests.test_regenie2_binary.BinaryFixture(
        covariate_matrix=fixture.covariate_matrix,
        phenotype_matrix=fixture.phenotype_matrix,
        loco_offset_matrix=fixture.loco_offset_matrix,
        genotype_matrix_by_variant=rare_variants.astype(np.float64),
    )
    reference = tests.test_regenie2_binary.compute_binary_score_reference(reference_fixture, kernel_config)

    for observed in (tiled_result, materialized_result):
        assert_constant_dosages_are_untestable(observed, constant_dosages.size)
        tests.numerical.assert_absolute_difference_less_than(
            observed.beta[:, constant_dosages.size :],
            reference.beta,
            tests.test_regenie2_binary.BINARY_BETA_ABSOLUTE_TOLERANCE,
        )
        tests.numerical.assert_absolute_difference_less_than(
            observed.chi_squared[:, constant_dosages.size :],
            reference.chi_squared,
            tests.test_regenie2_binary.BINARY_CHI_SQUARED_ABSOLUTE_TOLERANCE,
        )


@pytest.mark.parametrize("firth_candidate_p_threshold", [None, 1.0])
def test_binary_uniform_packed8_dosages_are_untestable(firth_candidate_p_threshold: float | None) -> None:
    """Test decoded dosage equality even when packed probability pairs differ."""
    fixture = tests.test_regenie2_binary.build_binary_fixture()
    kernel_config = tests.test_regenie2_binary.build_binary_score_config()
    chromosome_state = tests.test_regenie2_binary.build_binary_chromosome_state(fixture, kernel_config)
    sample_count = fixture.covariate_matrix.shape[0]
    constant_pairs = np.asarray([[255, 0], [223, 0], [0, 255], [64, 0], [64, 127], [0, 0]], dtype=np.uint8)
    constant_variant_count = constant_pairs.shape[0]
    packed_probabilities = np.zeros((constant_variant_count + 2, sample_count, 2), dtype=np.uint8)
    packed_probabilities[:constant_variant_count] = constant_pairs[:, None, :]
    packed_probabilities[4, ::2] = [0, 255]
    packed_probabilities[-2, :, 0] = 255
    packed_probabilities[-2, -1] = [0, 255]
    packed_probabilities[-1, -1] = [0, 255]
    observed = regenie2_binary_score.compute_multi_binary_score_test_packed8_donating_inputs(
        chromosome_state=chromosome_state,
        packed_probability_pairs_by_variant=jnp.asarray(packed_probabilities),
        firth_candidate_p_threshold=firth_candidate_p_threshold,
        minimum_variance=kernel_config.numerical.minimum_variance,
        relative_variance_tolerance=kernel_config.numerical.relative_variance_tolerance,
        native_genotype_mean=None,
    )

    assert_constant_dosages_are_untestable(observed, constant_variant_count)


@pytest.mark.parametrize("sample_count", [256, 257, 512])
def test_binary_variation_guard_spans_full_tiles_and_tail(sample_count: int) -> None:
    """Detect rare tail carriers and differences between individually constant tiles."""
    sample_indices = np.arange(sample_count)
    raw_genotypes = np.zeros((4, sample_count), dtype=np.float32)
    raw_genotypes[0] = 0.25
    raw_genotypes[1, -1] = 1.0
    raw_genotypes[2] = 2.0 - raw_genotypes[1]
    raw_genotypes[3] = sample_indices >= min(256, sample_count // 2)
    chromosome_state = regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState(
        score_right_hand_matrix=jnp.asarray(
            np.stack(
                [
                    np.full(sample_count, 0.5 / np.sqrt(sample_count), dtype=np.float32),
                    np.where(sample_indices % 2 == 0, -0.5, 0.5).astype(np.float32),
                ],
            ),
        ),
        bernoulli_weight=jnp.full((1, sample_count), 0.25, dtype=jnp.float32),
        null_logistic_converged=jnp.asarray([True]),
    )
    genotype_flip_mask = jnp.asarray(np.mean(raw_genotypes, axis=1) > 1.0)
    reduction = regenie2_binary_score.reduce_tiled_score_genotypes(
        chromosome_state,
        jnp.asarray(raw_genotypes),
        genotype_flip_mask,
    )
    observed = regenie2_binary_score.build_multi_binary_score_result(
        chromosome_state=chromosome_state,
        score_reduction=reduction,
        genotype_flip_mask=genotype_flip_mask,
        firth_candidate_p_threshold=1.0,
        minimum_variance=1.0e-8,
        relative_variance_tolerance=1.0e-7,
    )

    np.testing.assert_array_equal(np.asarray(reduction.variable_genotype_mask), [False, True, True, True])
    assert_constant_dosages_are_untestable(observed, 1)


def test_binary_constant_dosage_rejects_positive_projection_roundoff() -> None:
    """Reject a uniform variant even when rounded projection leaves positive variance."""
    sample_count = 16
    rounded_projection_value = np.nextafter(np.float32(0.125), np.float32(0.0))
    chromosome_state = regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState(
        score_right_hand_matrix=jnp.asarray(
            np.stack(
                [
                    np.full(sample_count, rounded_projection_value),
                    np.full(sample_count, np.float32(1.0e-4)),
                ],
            ),
        ),
        bernoulli_weight=jnp.full((1, sample_count), 0.25, dtype=jnp.float32),
        null_logistic_converged=jnp.asarray([True]),
    )
    raw_genotypes = jnp.ones((1, sample_count), dtype=jnp.float32)
    reduction = regenie2_binary_score.reduce_materialized_score_genotypes(chromosome_state, raw_genotypes)
    projection_square = reduction.stacked_product_by_variant[:, 0] ** 2
    assert bool(jnp.all(reduction.weighted_genotype_sum_squares - projection_square > 0.0))
    observed = regenie2_binary_score.build_multi_binary_score_result(
        chromosome_state=chromosome_state,
        score_reduction=reduction,
        genotype_flip_mask=jnp.asarray([False]),
        firth_candidate_p_threshold=1.0,
        minimum_variance=1.0e-8,
        relative_variance_tolerance=1.0e-7,
    )

    assert_constant_dosages_are_untestable(observed, 1)
