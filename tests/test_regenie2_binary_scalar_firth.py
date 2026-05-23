from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import g.compute.regenie2_binary as regenie2_binary
from g import types
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary import types as regenie2_binary_types
from g.compute.regenie2_binary.firth import batch as regenie2_binary_firth_batch
from g.compute.regenie2_binary.firth import scalar as regenie2_binary_firth_scalar


def build_scalar_fixture() -> tuple[regenie2_binary_types.Regenie2BinaryChromosomeState, jax.Array, jax.Array]:
    """Build a deterministic separation fixture for scalar Firth tests."""
    covariate_matrix = jnp.asarray(
        [
            [1.0, 20.0],
            [1.0, 25.0],
            [1.0, 30.0],
            [1.0, 35.0],
            [1.0, 40.0],
            [1.0, 45.0],
            [1.0, 50.0],
            [1.0, 55.0],
        ],
        dtype=jnp.float32,
    )
    phenotype_vector = jnp.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
    genotype_vector = jnp.asarray([0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0], dtype=jnp.float32)
    state = regenie2_binary_state.prepare_regenie2_binary_state(covariate_matrix, phenotype_vector)
    chromosome_state = regenie2_binary_state.prepare_regenie2_binary_chromosome_state(
        state,
        jnp.zeros_like(phenotype_vector),
    )
    residualized_genotype_vector = regenie2_binary_firth_batch.residualize_and_scale_genotypes_for_approximate_firth(
        chromosome_state,
        genotype_vector[None, :],
    )[0]
    return chromosome_state, genotype_vector, residualized_genotype_vector


def test_regenie_logistic_deviance_matches_manual_formula() -> None:
    phenotype_vector = jnp.asarray([0.0, 1.0, 1.0], dtype=jnp.float32)
    probability_vector = jnp.asarray([0.25, 0.75, 0.50], dtype=jnp.float32)
    active_sample_mask = jnp.asarray([True, True, False], dtype=jnp.bool_)

    deviance = regenie2_binary.compute_logistic_deviance(
        phenotype_vector,
        probability_vector,
        active_sample_mask,
    )

    expected = -2.0 * (np.log1p(-0.25) + np.log(0.75))
    np.testing.assert_allclose(np.asarray(deviance), expected, rtol=1.0e-6)


def test_scalar_pseudo_firth_components_match_formula() -> None:
    phenotype_vector = jnp.asarray([0.0, 1.0, 1.0], dtype=jnp.float32)
    genotype_vector = jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32)
    offset_vector = jnp.asarray([-0.2, 0.1, 0.3], dtype=jnp.float32)
    active_sample_mask = jnp.asarray([True, True, True], dtype=jnp.bool_)

    components = regenie2_binary_firth_scalar.compute_scalar_firth_components(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=jnp.asarray(0.0, dtype=jnp.float32),
        beta=jnp.asarray(0.4, dtype=jnp.float32),
    )

    probability_vector = regenie2_binary.compute_regenie_logistic_probability(offset_vector + genotype_vector * 0.4)
    weight_vector = probability_vector * (1.0 - probability_vector)
    genotype_information_diagonal = genotype_vector * genotype_vector * weight_vector
    genotype_information = jnp.sum(genotype_information_diagonal)
    leverage_vector = genotype_information_diagonal / genotype_information
    adjusted_response = phenotype_vector + leverage_vector * (0.5 - probability_vector)
    expected_score = jnp.sum(genotype_vector * (adjusted_response - probability_vector))
    expected_deviance = regenie2_binary.compute_logistic_deviance(
        phenotype_vector, probability_vector, active_sample_mask
    ) - jnp.log(genotype_information)

    np.testing.assert_allclose(np.asarray(components.score), np.asarray(expected_score), rtol=1.0e-6)
    np.testing.assert_allclose(
        np.asarray(components.penalized_deviance),
        np.asarray(expected_deviance),
        rtol=1.0e-6,
    )
    assert bool(np.asarray(components.valid))


def test_scalar_approximate_firth_uses_nr_fallback_after_pseudo_attempt() -> None:
    chromosome_state, raw_genotype_vector, genotype_vector = build_scalar_fixture()
    offset_vector = chromosome_state.null_firth_offset

    result = regenie2_binary_firth_scalar.fit_single_variant_regenie_approximate_firth(
        phenotype_vector=chromosome_state.phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        carrier_sample_mask=raw_genotype_vector > regenie2_binary_firth_batch.SPARSE_CARRIER_DOSAGE_THRESHOLD,
        sparse_correction=jnp.asarray(1, dtype=jnp.bool_),
        warm_start_beta=jnp.asarray(0.0, dtype=jnp.float32),
        skip_firth=jnp.asarray(0, dtype=jnp.bool_),
        null_failed=jnp.asarray(0, dtype=jnp.bool_),
        kernel_config=regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    )

    assert bool(np.asarray(result.valid_mask))
    assert int(np.asarray(result.pseudo_firth_iteration_count)) > 0
    assert int(np.asarray(result.correction_code)) == types.FirthCorrectionCode.NEWTON_RAPHSON_WARM_START.value
    assert int(np.asarray(result.nr_warm_start_iteration_count)) > 0


def test_sparse_carrier_only_flag_is_recorded_for_sparse_candidate() -> None:
    chromosome_state, raw_genotype_vector, genotype_vector = build_scalar_fixture()
    offset_vector = chromosome_state.null_firth_offset

    result = regenie2_binary_firth_scalar.fit_single_variant_regenie_approximate_firth(
        phenotype_vector=chromosome_state.phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        carrier_sample_mask=raw_genotype_vector > regenie2_binary_firth_batch.SPARSE_CARRIER_DOSAGE_THRESHOLD,
        sparse_correction=jnp.asarray(1, dtype=jnp.bool_),
        warm_start_beta=jnp.asarray(0.0, dtype=jnp.float32),
        skip_firth=jnp.asarray(0, dtype=jnp.bool_),
        null_failed=jnp.asarray(0, dtype=jnp.bool_),
        kernel_config=regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    )

    assert bool(np.asarray(result.sparse_correction_mask))
    assert np.isfinite(np.asarray(result.beta))
    assert np.isfinite(np.asarray(result.chi_squared))


def test_collinear_scalar_candidate_gets_numerical_failure_label() -> None:
    covariate_matrix = jnp.asarray(
        [[1.0, 20.0], [1.0, 25.0], [1.0, 30.0], [1.0, 35.0], [1.0, 40.0], [1.0, 45.0]],
        dtype=jnp.float32,
    )
    phenotype_vector = jnp.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
    state = regenie2_binary_state.prepare_regenie2_binary_state(covariate_matrix, phenotype_vector)
    chromosome_state = regenie2_binary_state.prepare_regenie2_binary_chromosome_state(
        state, jnp.zeros_like(phenotype_vector)
    )
    raw_genotype_vector = covariate_matrix[:, 1]
    genotype_vector = regenie2_binary_firth_batch.residualize_and_scale_genotypes_for_approximate_firth(
        chromosome_state,
        raw_genotype_vector[None, :],
    )[0]

    result = regenie2_binary_firth_scalar.fit_single_variant_regenie_approximate_firth(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=chromosome_state.null_firth_offset,
        carrier_sample_mask=raw_genotype_vector > regenie2_binary_firth_batch.SPARSE_CARRIER_DOSAGE_THRESHOLD,
        sparse_correction=jnp.asarray(0, dtype=jnp.bool_),
        warm_start_beta=jnp.asarray(0.0, dtype=jnp.float32),
        skip_firth=jnp.asarray(0, dtype=jnp.bool_),
        null_failed=jnp.asarray(0, dtype=jnp.bool_),
        kernel_config=regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
    )

    assert not bool(np.asarray(result.valid_mask))
    assert int(np.asarray(result.failure_code)) == types.FirthFailureCode.NUMERICAL.value
