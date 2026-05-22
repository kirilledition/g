from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from g import types
from g.compute import regenie2_binary, regenie2_binary_types


def build_scalar_fixture(
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_types.BinaryKernelConfig = regenie2_binary.DEFAULT_BINARY_KERNEL_CONFIG,
) -> tuple[regenie2_binary_types.Regenie2BinaryChromosomeState, jax.Array, jax.Array]:
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
    state = regenie2_binary.prepare_regenie2_binary_state(covariate_matrix, phenotype_vector)
    chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
        state,
        jnp.zeros_like(phenotype_vector),
        correction_plan,
        kernel_config,
    )
    residualized_genotype_vector = regenie2_binary.residualize_and_scale_genotypes_for_approximate_firth(
        chromosome_state,
        genotype_vector[None, :],
    )[0]
    return chromosome_state, genotype_vector, residualized_genotype_vector


def compute_single_lane_firth_result(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    raw_genotype_vector: jax.Array,
    genotype_vector: jax.Array,
    kernel_config: regenie2_binary_types.BinaryKernelConfig,
) -> regenie2_binary.FirthVariantResult:
    """Run the vectorized Firth lane directly so tests can inspect internal result dtype."""
    coefficient_count = chromosome_state.covariate_matrix.shape[1] + 1
    initial_coefficients = jnp.zeros((1, coefficient_count), dtype=regenie2_binary.BINARY_SCORE_DTYPE)
    return regenie2_binary.compute_firth_variantwise(
        covariate_matrix=chromosome_state.covariate_matrix,
        null_logistic_coefficients=chromosome_state.null_logistic_coefficients,
        null_firth_offset=chromosome_state.null_firth_offset,
        phenotype_vector=chromosome_state.phenotype_vector,
        genotype_matrix_by_variant=genotype_vector[None, :],
        raw_genotype_matrix_by_variant=raw_genotype_vector[None, :],
        loco_offset=chromosome_state.loco_offset,
        initial_coefficients=initial_coefficients,
        skip_firth_mask=jnp.asarray([False], dtype=jnp.bool_),
        sparse_correction_mask=jnp.asarray([True], dtype=jnp.bool_),
        null_penalized_log_likelihood=chromosome_state.null_firth_penalized_log_likelihood,
        kernel_config=kernel_config,
    )


def test_default_firth_internal_dtype_remains_float64() -> None:
    chromosome_state, raw_genotype_vector, genotype_vector = build_scalar_fixture(
        types.BinaryCorrectionPlan(method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE)
    )

    result = compute_single_lane_firth_result(
        chromosome_state,
        raw_genotype_vector,
        genotype_vector,
        regenie2_binary.DEFAULT_BINARY_KERNEL_CONFIG,
    )

    assert chromosome_state.covariate_matrix.dtype == jnp.dtype(regenie2_binary.BINARY_SCORE_DTYPE)
    assert chromosome_state.null_firth_offset.dtype == jnp.dtype(regenie2_binary.BINARY_FIRTH_REFERENCE_DTYPE)
    assert result.beta.dtype == jnp.dtype(regenie2_binary.BINARY_FIRTH_REFERENCE_DTYPE)
    assert result.log10_p_value.dtype == jnp.dtype(regenie2_binary.BINARY_FIRTH_REFERENCE_DTYPE)


def test_float32_firth_debug_config_demotes_null_and_scalar_firth_internal_dtype() -> None:
    kernel_config = dataclasses.replace(regenie2_binary.DEFAULT_BINARY_KERNEL_CONFIG, use_float32_firth_math=True)
    chromosome_state, raw_genotype_vector, genotype_vector = build_scalar_fixture(
        types.BinaryCorrectionPlan(method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE),
        kernel_config,
    )

    result = compute_single_lane_firth_result(
        chromosome_state,
        raw_genotype_vector,
        genotype_vector,
        kernel_config,
    )

    assert chromosome_state.null_firth_coefficients.dtype == jnp.dtype(regenie2_binary.BINARY_SCORE_DTYPE)
    assert chromosome_state.null_firth_offset.dtype == jnp.dtype(regenie2_binary.BINARY_SCORE_DTYPE)
    assert result.beta.dtype == jnp.dtype(regenie2_binary.BINARY_SCORE_DTYPE)
    assert result.log10_p_value.dtype == jnp.dtype(regenie2_binary.BINARY_SCORE_DTYPE)


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

    components = regenie2_binary.compute_scalar_firth_components(
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

    result = regenie2_binary.fit_single_variant_regenie_approximate_firth(
        phenotype_vector=chromosome_state.phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        carrier_sample_mask=raw_genotype_vector > regenie2_binary.SPARSE_CARRIER_DOSAGE_THRESHOLD,
        sparse_correction=jnp.asarray(1, dtype=jnp.bool_),
        warm_start_beta=jnp.asarray(0.0, dtype=jnp.float32),
        skip_firth=jnp.asarray(0, dtype=jnp.bool_),
        null_failed=jnp.asarray(0, dtype=jnp.bool_),
    )

    assert bool(np.asarray(result.valid_mask))
    assert int(np.asarray(result.pseudo_firth_iteration_count)) > 0
    assert int(np.asarray(result.correction_code)) == types.FirthCorrectionCode.NEWTON_RAPHSON_WARM_START.value
    assert int(np.asarray(result.nr_warm_start_iteration_count)) > 0


def test_sparse_carrier_only_flag_is_recorded_for_sparse_candidate() -> None:
    chromosome_state, raw_genotype_vector, genotype_vector = build_scalar_fixture()
    offset_vector = chromosome_state.null_firth_offset

    result = regenie2_binary.fit_single_variant_regenie_approximate_firth(
        phenotype_vector=chromosome_state.phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        carrier_sample_mask=raw_genotype_vector > regenie2_binary.SPARSE_CARRIER_DOSAGE_THRESHOLD,
        sparse_correction=jnp.asarray(1, dtype=jnp.bool_),
        warm_start_beta=jnp.asarray(0.0, dtype=jnp.float32),
        skip_firth=jnp.asarray(0, dtype=jnp.bool_),
        null_failed=jnp.asarray(0, dtype=jnp.bool_),
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
    state = regenie2_binary.prepare_regenie2_binary_state(covariate_matrix, phenotype_vector)
    chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(state, jnp.zeros_like(phenotype_vector))
    raw_genotype_vector = covariate_matrix[:, 1]
    genotype_vector = regenie2_binary.residualize_and_scale_genotypes_for_approximate_firth(
        chromosome_state,
        raw_genotype_vector[None, :],
    )[0]

    result = regenie2_binary.fit_single_variant_regenie_approximate_firth(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=chromosome_state.null_firth_offset,
        carrier_sample_mask=raw_genotype_vector > regenie2_binary.SPARSE_CARRIER_DOSAGE_THRESHOLD,
        sparse_correction=jnp.asarray(0, dtype=jnp.bool_),
        warm_start_beta=jnp.asarray(0.0, dtype=jnp.float32),
        skip_firth=jnp.asarray(0, dtype=jnp.bool_),
        null_failed=jnp.asarray(0, dtype=jnp.bool_),
    )

    assert not bool(np.asarray(result.valid_mask))
    assert int(np.asarray(result.failure_code)) == types.FirthFailureCode.NUMERICAL.value
