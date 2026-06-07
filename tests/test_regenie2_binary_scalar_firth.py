from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from g import execution_plan, types
from g.compute.regenie2_binary import api as regenie2_binary
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import logistic as regenie2_binary_logistic
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary.firth import scalar_approx as regenie2_binary_firth_scalar_approx
from g.interface import config as interface_config


def build_default_binary_kernel_config() -> regenie2_binary_config.BinaryKernelConfig:
    """Build the packaged-default kernel config for tests."""
    return execution_plan.build_binary_kernel_config(interface_config.load_packaged_config().g_compute)


def build_scalar_fixture() -> tuple[regenie2_binary_state.Regenie2BinaryChromosomeState, jax.Array, jax.Array]:
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
    kernel_config = build_default_binary_kernel_config()
    chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
        state,
        jnp.zeros_like(phenotype_vector),
        types.BinaryCorrectionPlan(),
        kernel_config,
    )
    residualize_genotypes = regenie2_binary_firth_scalar_approx.residualize_and_scale_genotypes_for_approximate_firth
    residualized_genotype_vector = residualize_genotypes(chromosome_state, genotype_vector[None, :])[0]
    return chromosome_state, genotype_vector, residualized_genotype_vector


def test_regenie_logistic_deviance_matches_manual_formula() -> None:
    phenotype_vector = jnp.asarray([0.0, 1.0, 1.0], dtype=jnp.float32)
    probability_vector = jnp.asarray([0.25, 0.75, 0.50], dtype=jnp.float32)
    active_sample_mask = jnp.asarray([True, True, False], dtype=jnp.bool_)

    deviance = regenie2_binary_logistic.compute_logistic_deviance(
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

    components = regenie2_binary_firth_scalar_approx.compute_scalar_firth_components(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=active_sample_mask,
        non_active_deviance=jnp.asarray(0.0, dtype=jnp.float32),
        beta=jnp.asarray(0.4, dtype=jnp.float32),
        kernel_config=build_default_binary_kernel_config(),
    )

    probability_vector = regenie2_binary_logistic.compute_regenie_logistic_probability(
        offset_vector + genotype_vector * 0.4
    )
    weight_vector = probability_vector * (1.0 - probability_vector)
    genotype_information_diagonal = genotype_vector * genotype_vector * weight_vector
    genotype_information = jnp.sum(genotype_information_diagonal)
    leverage_vector = genotype_information_diagonal / genotype_information
    adjusted_response = phenotype_vector + leverage_vector * (0.5 - probability_vector)
    expected_score = jnp.sum(genotype_vector * (adjusted_response - probability_vector))
    expected_deviance = regenie2_binary_logistic.compute_logistic_deviance(
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
    kernel_config = build_default_binary_kernel_config()

    result = regenie2_binary_firth_scalar_approx.fit_single_variant_regenie_approximate_firth(
        phenotype_vector=chromosome_state.phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        carrier_sample_mask=raw_genotype_vector > kernel_config.approximate_firth.sparse_carrier_dosage_threshold,
        sparse_correction=jnp.asarray(1, dtype=jnp.bool_),
        warm_start_beta=jnp.asarray(0.0, dtype=jnp.float32),
        skip_firth=jnp.asarray(0, dtype=jnp.bool_),
        null_failed=jnp.asarray(0, dtype=jnp.bool_),
        kernel_config=kernel_config,
    )

    assert bool(np.asarray(result.valid_mask))
    assert int(np.asarray(result.pseudo_firth_iteration_count)) > 0
    assert int(np.asarray(result.correction_code)) == types.FirthCorrectionCode.NEWTON_RAPHSON_WARM_START.value
    assert int(np.asarray(result.nr_zero_start_iteration_count)) == 0
    assert int(np.asarray(result.nr_warm_start_iteration_count)) > 0


def test_scalar_approximate_firth_reports_zero_counts_for_skipped_lane() -> None:
    chromosome_state, raw_genotype_vector, genotype_vector = build_scalar_fixture()
    kernel_config = build_default_binary_kernel_config()

    result = regenie2_binary_firth_scalar_approx.fit_single_variant_regenie_approximate_firth(
        phenotype_vector=chromosome_state.phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=chromosome_state.null_firth_offset,
        carrier_sample_mask=raw_genotype_vector > kernel_config.approximate_firth.sparse_carrier_dosage_threshold,
        sparse_correction=jnp.asarray(1, dtype=jnp.bool_),
        warm_start_beta=jnp.asarray(0.0, dtype=jnp.float32),
        skip_firth=jnp.asarray(1, dtype=jnp.bool_),
        null_failed=jnp.asarray(0, dtype=jnp.bool_),
        kernel_config=kernel_config,
    )

    assert not bool(np.asarray(result.valid_mask))
    assert int(np.asarray(result.pseudo_firth_iteration_count)) == 0
    assert int(np.asarray(result.nr_zero_start_iteration_count)) == 0
    assert int(np.asarray(result.nr_warm_start_iteration_count)) == 0


def test_scalar_newton_line_search_exhaustion_does_not_move_beta() -> None:
    phenotype_vector = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    genotype_vector = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    offset_vector = jnp.zeros_like(phenotype_vector)
    initial_beta = jnp.asarray(0.25, dtype=jnp.float32)
    initial_components = regenie2_binary_firth_scalar_approx.compute_scalar_firth_components(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=jnp.ones_like(phenotype_vector, dtype=jnp.bool_),
        non_active_deviance=jnp.asarray(0.0, dtype=jnp.float32),
        beta=initial_beta,
        kernel_config=build_default_binary_kernel_config(),
    )

    result = regenie2_binary_firth_scalar_approx.fit_scalar_newton_raphson_firth(
        deviance_null=initial_components.penalized_deviance,
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        active_sample_mask=jnp.ones_like(phenotype_vector, dtype=jnp.bool_),
        non_active_deviance=jnp.asarray(0.0, dtype=jnp.float32),
        initial_beta=initial_beta,
        maximum_iterations=1,
        tolerance=jnp.asarray(1.0e-8, dtype=jnp.float32),
        maximum_step_size=jnp.asarray(1.0, dtype=jnp.float32),
        line_search_maximum_attempts=0,
        kernel_config=build_default_binary_kernel_config(),
    )

    np.testing.assert_allclose(np.asarray(result.beta), np.asarray(initial_beta), rtol=0.0, atol=0.0)
    assert not bool(np.asarray(result.valid))
    assert not bool(np.asarray(result.converged))


def test_sparse_carrier_only_flag_is_recorded_for_sparse_candidate() -> None:
    chromosome_state, raw_genotype_vector, genotype_vector = build_scalar_fixture()
    offset_vector = chromosome_state.null_firth_offset
    kernel_config = build_default_binary_kernel_config()

    result = regenie2_binary_firth_scalar_approx.fit_single_variant_regenie_approximate_firth(
        phenotype_vector=chromosome_state.phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=offset_vector,
        carrier_sample_mask=raw_genotype_vector > kernel_config.approximate_firth.sparse_carrier_dosage_threshold,
        sparse_correction=jnp.asarray(1, dtype=jnp.bool_),
        warm_start_beta=jnp.asarray(0.0, dtype=jnp.float32),
        skip_firth=jnp.asarray(0, dtype=jnp.bool_),
        null_failed=jnp.asarray(0, dtype=jnp.bool_),
        kernel_config=kernel_config,
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
    kernel_config = build_default_binary_kernel_config()
    chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
        state,
        jnp.zeros_like(phenotype_vector),
        types.BinaryCorrectionPlan(),
        kernel_config,
    )
    raw_genotype_vector = covariate_matrix[:, 1]
    residualize_genotypes = regenie2_binary_firth_scalar_approx.residualize_and_scale_genotypes_for_approximate_firth
    genotype_vector = residualize_genotypes(chromosome_state, raw_genotype_vector[None, :])[0]

    result = regenie2_binary_firth_scalar_approx.fit_single_variant_regenie_approximate_firth(
        phenotype_vector=phenotype_vector,
        genotype_vector=genotype_vector,
        offset_vector=chromosome_state.null_firth_offset,
        carrier_sample_mask=raw_genotype_vector > kernel_config.approximate_firth.sparse_carrier_dosage_threshold,
        sparse_correction=jnp.asarray(0, dtype=jnp.bool_),
        warm_start_beta=jnp.asarray(0.0, dtype=jnp.float32),
        skip_firth=jnp.asarray(0, dtype=jnp.bool_),
        null_failed=jnp.asarray(0, dtype=jnp.bool_),
        kernel_config=kernel_config,
    )

    assert not bool(np.asarray(result.valid_mask))
    assert int(np.asarray(result.failure_code)) == types.FirthFailureCode.NUMERICAL.value
