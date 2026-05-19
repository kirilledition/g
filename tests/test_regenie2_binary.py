from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from g import types
from g.compute import regenie2_binary, regenie2_binary_types

APPROXIMATE_FIRTH_PLAN = types.BinaryCorrectionPlan(
    method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
    p_threshold=0.05,
    firth_se=False,
)

BinaryChunkComputeFunction = typing.Callable[
    [regenie2_binary_types.Regenie2BinaryChromosomeState, jax.Array, types.BinaryCorrectionPlan],
    regenie2_binary_types.Regenie2BinaryChunkResult,
]
compute_score_test_chunk = typing.cast(
    "BinaryChunkComputeFunction",
    regenie2_binary.compute_regenie2_binary_score_test_chunk_from_chromosome_state,
)
compute_binary_chunk = typing.cast(
    "BinaryChunkComputeFunction",
    regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state,
)
compute_score_test_chunk_variant_major = typing.cast(
    "BinaryChunkComputeFunction",
    regenie2_binary.compute_regenie2_binary_score_test_chunk_from_chromosome_state_variant_major,
)
compute_binary_chunk_variant_major = typing.cast(
    "BinaryChunkComputeFunction",
    regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state_variant_major,
)


def clear_binary_compute_caches() -> None:
    """Clear cached binary configuration and JAX traces."""
    regenie2_binary.get_firth_batch_size.cache_clear()
    regenie2_binary.get_firth_candidate_capacity.cache_clear()
    regenie2_binary.get_use_block_firth_math.cache_clear()
    jax.clear_caches()


def jax_backend_is_available(backend_name: str) -> bool:
    try:
        return bool(jax.devices(backend_name))
    except RuntimeError:
        return False


def build_binary_inputs() -> tuple[jax.Array, jax.Array, jax.Array]:
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
    genotype_matrix = jnp.asarray(
        [
            [0.0, 0.0, 20.0],
            [0.0, 0.0, 25.0],
            [0.0, 1.0, 30.0],
            [0.0, 1.0, 35.0],
            [2.0, 1.0, 40.0],
            [2.0, 1.0, 45.0],
            [2.0, 2.0, 50.0],
            [2.0, 2.0, 55.0],
        ],
        dtype=jnp.float32,
    )
    return covariate_matrix, phenotype_vector, genotype_matrix


def build_chromosome_state() -> tuple[
    jax.Array,
    regenie2_binary_types.Regenie2BinaryChromosomeState,
]:
    covariate_matrix, phenotype_vector, genotype_matrix = build_binary_inputs()
    state = regenie2_binary.prepare_regenie2_binary_state(covariate_matrix, phenotype_vector)
    chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
        state,
        jnp.zeros((phenotype_vector.shape[0],), dtype=jnp.float32),
    )
    return genotype_matrix, chromosome_state


def test_firth_candidate_capacity_uses_default() -> None:
    regenie2_binary.get_firth_candidate_capacity.cache_clear()

    assert regenie2_binary.get_firth_candidate_capacity() == regenie2_binary.DEFAULT_FIRTH_CANDIDATE_CAPACITY


def test_firth_candidate_capacity_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="Firth candidate capacity"):
        regenie2_binary.regenie2_binary_candidate_planning.configure_firth_candidate_planning(
            firth_batch_size=64,
            firth_candidate_capacity=0,
        )


def test_device_firth_batch_plan_uses_candidate_capacity() -> None:
    regenie2_binary.regenie2_binary_candidate_planning.configure_firth_candidate_planning(
        firth_batch_size=2,
        firth_candidate_capacity=1024,
    )
    clear_binary_compute_caches()
    fallback_mask = jnp.asarray([True, False, True, False, True], dtype=jnp.bool_)

    batch_plan = regenie2_binary.build_device_firth_batch_plan(fallback_mask, candidate_capacity=4)

    np.testing.assert_array_equal(np.asarray(batch_plan.fallback_index_matrix), [[0, 2], [4, 0]])
    np.testing.assert_array_equal(np.asarray(batch_plan.fallback_active_mask_matrix), [[True, True], [True, False]])
    np.testing.assert_array_equal(np.asarray(batch_plan.active_flat_position_vector), [0, 1, 2, 0])
    regenie2_binary.regenie2_binary_candidate_planning.configure_firth_candidate_planning(
        firth_batch_size=regenie2_binary.DEFAULT_FIRTH_BATCH_SIZE,
        firth_candidate_capacity=regenie2_binary.DEFAULT_FIRTH_CANDIDATE_CAPACITY,
    )


def test_group_firth_candidate_batch_inputs_places_heuristic_lanes_after_regular_lanes() -> None:
    ordered_inputs = regenie2_binary.group_firth_candidate_batch_inputs(
        flat_fallback_indices=jnp.asarray([10, 11, 12, 0], dtype=jnp.int32),
        flat_active_mask=jnp.asarray([True, True, True, False], dtype=jnp.bool_),
        genotype_matrix_by_variant=jnp.asarray(
            [
                [10.0, 10.0],
                [11.0, 11.0],
                [12.0, 12.0],
                [0.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
        heuristic_firth_mask=jnp.asarray([True, False, True, False], dtype=jnp.bool_),
    )

    np.testing.assert_array_equal(np.asarray(ordered_inputs.flat_fallback_indices), [11, 10, 12, 0])
    np.testing.assert_array_equal(np.asarray(ordered_inputs.flat_active_mask), [True, True, True, False])
    np.testing.assert_array_equal(np.asarray(ordered_inputs.heuristic_firth_mask), [False, True, True, False])
    np.testing.assert_array_equal(
        np.asarray(ordered_inputs.genotype_matrix_by_variant),
        [[11.0, 11.0], [10.0, 10.0], [12.0, 12.0], [0.0, 0.0]],
    )


def test_score_only_plan_produces_no_fallback_candidates() -> None:
    extra_code = regenie2_binary.build_extra_code(
        log10_p_value=jnp.asarray([0.5, 2.0, 8.0], dtype=jnp.float32),
        valid_mask=jnp.asarray([True, True, True], dtype=jnp.bool_),
        correction_plan=types.BinaryCorrectionPlan(),
    )

    np.testing.assert_array_equal(np.asarray(extra_code), [regenie2_binary.EXTRA_CODE_SCORE] * 3)


def test_score_only_chromosome_prep_skips_firth_null_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    covariate_matrix, phenotype_vector, _ = build_binary_inputs()
    state = regenie2_binary.prepare_regenie2_binary_state(covariate_matrix, phenotype_vector)

    def fail_firth_null_fit(*args: object, **kwargs: object) -> jax.Array:
        del args, kwargs
        raise AssertionError("score-only chromosome prep must not fit the Firth null model")

    monkeypatch.setattr(regenie2_binary, "fit_covariate_only_firth_null_model", fail_firth_null_fit)
    chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
        state,
        jnp.zeros((phenotype_vector.shape[0],), dtype=jnp.float32),
        types.BinaryCorrectionPlan(),
    )

    assert float(np.asarray(chromosome_state.null_firth_penalized_log_likelihood)) == 0.0
    assert int(np.asarray(chromosome_state.null_logistic_iteration_count)) <= (
        regenie2_binary.DEFAULT_MAXIMUM_NULL_ITERATIONS
    )


def test_p_threshold_controls_fallback_candidate_selection() -> None:
    valid_mask = jnp.asarray([True, True, True], dtype=jnp.bool_)
    log10_p_value = jnp.asarray([1.1, 1.5, 2.1], dtype=jnp.float32)
    relaxed_plan = types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
        p_threshold=0.05,
    )
    strict_plan = types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
        p_threshold=0.01,
    )

    relaxed_extra_code = regenie2_binary.build_extra_code(log10_p_value, valid_mask, relaxed_plan)
    strict_extra_code = regenie2_binary.build_extra_code(log10_p_value, valid_mask, strict_plan)

    assert np.count_nonzero(np.asarray(relaxed_extra_code) == regenie2_binary.EXTRA_CODE_FIRTH) == 2
    assert np.count_nonzero(np.asarray(strict_extra_code) == regenie2_binary.EXTRA_CODE_FIRTH) == 1


@pytest.mark.parametrize(
    "unsupported_plan",
    [
        types.BinaryCorrectionPlan(method=types.BinaryFallbackMethod.FIRTH),
        types.BinaryCorrectionPlan(method=types.BinaryFallbackMethod.SPA),
    ],
)
def test_unsupported_direct_binary_compute_paths_fail_loudly(
    unsupported_plan: types.BinaryCorrectionPlan,
) -> None:
    genotype_matrix, chromosome_state = build_chromosome_state()

    with pytest.raises(NotImplementedError):
        compute_score_test_chunk(chromosome_state, genotype_matrix[:, :1], unsupported_plan)


def test_full_model_adjusted_weight_components_match_design_matrix_path() -> None:
    genotype_matrix, chromosome_state = build_chromosome_state()
    genotype_vector = genotype_matrix[:, 1]
    coefficients = jnp.asarray([0.1, -0.01, 0.25], dtype=jnp.float32)
    linear_predictor = (
        chromosome_state.covariate_matrix @ coefficients[:-1]
        + genotype_vector * coefficients[-1]
        + chromosome_state.loco_offset
    )
    probability_vector = regenie2_binary.compute_logistic_probability(linear_predictor)
    information_components = regenie2_binary.compute_information_components(
        chromosome_state.covariate_matrix,
        genotype_vector,
        probability_vector,
    )
    full_design_matrix = jnp.concatenate([chromosome_state.covariate_matrix, genotype_vector[:, None]], axis=1)

    existing_components = regenie2_binary.compute_full_model_adjusted_weight_components(
        full_design_matrix=full_design_matrix,
        probability_vector=probability_vector,
        information_matrix=information_components.information_matrix,
        phenotype_vector=chromosome_state.phenotype_vector,
    )
    block_components = regenie2_binary.compute_full_model_adjusted_weight_components_from_parts(
        covariate_matrix=chromosome_state.covariate_matrix,
        genotype_vector=genotype_vector,
        probability_vector=probability_vector,
        information_matrix=information_components.information_matrix,
        phenotype_vector=chromosome_state.phenotype_vector,
    )

    np.testing.assert_allclose(
        np.asarray(block_components.leverage_vector),
        np.asarray(existing_components.leverage_vector),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(block_components.adjusted_weight_vector),
        np.asarray(existing_components.adjusted_weight_vector),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(block_components.second_weight_vector),
        np.asarray(existing_components.second_weight_vector),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_device_firth_candidate_correction_returns_finite_statistics() -> None:
    genotype_matrix, chromosome_state = build_chromosome_state()
    candidate_genotype_matrix = genotype_matrix[:, :1]
    score_result = compute_score_test_chunk(
        chromosome_state,
        candidate_genotype_matrix,
        APPROXIMATE_FIRTH_PLAN,
    )
    forced_candidate_result = regenie2_binary_types.Regenie2BinaryChunkResult(
        beta=score_result.beta,
        standard_error=score_result.standard_error,
        chi_squared=score_result.chi_squared,
        log10_p_value=score_result.log10_p_value,
        extra_code=jnp.asarray([regenie2_binary.EXTRA_CODE_FIRTH], dtype=jnp.int32),
        valid_mask=jnp.asarray([True]),
        firth_iteration_count=jnp.asarray([0], dtype=jnp.int32),
        firth_failure_code=jnp.asarray([0], dtype=jnp.int32),
    )

    result = regenie2_binary.apply_device_candidate_corrections(
        chromosome_state=chromosome_state,
        genotype_matrix=candidate_genotype_matrix,
        result=forced_candidate_result,
        correction_plan=APPROXIMATE_FIRTH_PLAN,
    )

    assert np.isfinite(np.asarray(result.beta[0]))
    assert np.isfinite(np.asarray(result.standard_error[0]))
    assert np.isfinite(np.asarray(result.chi_squared[0]))
    assert np.isfinite(np.asarray(result.log10_p_value[0]))
    assert int(np.asarray(result.extra_code[0])) == regenie2_binary.EXTRA_CODE_FIRTH
    assert bool(np.asarray(result.valid_mask[0]))


def test_firth_se_changes_only_successful_firth_standard_error() -> None:
    genotype_matrix, chromosome_state = build_chromosome_state()
    candidate_genotype_matrix = genotype_matrix[:, :1]
    score_result = compute_score_test_chunk(
        chromosome_state,
        candidate_genotype_matrix,
        APPROXIMATE_FIRTH_PLAN,
    )
    forced_candidate_result = regenie2_binary_types.Regenie2BinaryChunkResult(
        beta=score_result.beta,
        standard_error=score_result.standard_error,
        chi_squared=score_result.chi_squared,
        log10_p_value=score_result.log10_p_value,
        extra_code=jnp.asarray([regenie2_binary.EXTRA_CODE_FIRTH], dtype=jnp.int32),
        valid_mask=jnp.asarray([True]),
        firth_iteration_count=jnp.asarray([0], dtype=jnp.int32),
        firth_failure_code=jnp.asarray([0], dtype=jnp.int32),
    )
    firth_se_plan = types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
        p_threshold=0.05,
        firth_se=True,
    )

    default_result = regenie2_binary.apply_device_candidate_corrections(
        chromosome_state=chromosome_state,
        genotype_matrix=candidate_genotype_matrix,
        result=forced_candidate_result,
        correction_plan=APPROXIMATE_FIRTH_PLAN,
    )
    firth_se_result = regenie2_binary.apply_device_candidate_corrections(
        chromosome_state=chromosome_state,
        genotype_matrix=candidate_genotype_matrix,
        result=forced_candidate_result,
        correction_plan=firth_se_plan,
    )

    assert int(np.asarray(firth_se_result.extra_code[0])) == regenie2_binary.EXTRA_CODE_FIRTH
    np.testing.assert_allclose(np.asarray(firth_se_result.beta), np.asarray(default_result.beta))
    np.testing.assert_allclose(np.asarray(firth_se_result.chi_squared), np.asarray(default_result.chi_squared))
    expected_standard_error = np.abs(np.asarray(firth_se_result.beta)) / np.sqrt(
        np.asarray(firth_se_result.chi_squared)
    )
    np.testing.assert_allclose(np.asarray(firth_se_result.standard_error), expected_standard_error)


def test_sparse_candidate_mask_does_not_expand_score_candidates() -> None:
    genotype_matrix, chromosome_state = build_chromosome_state()
    low_score_genotype_matrix = genotype_matrix[:, :1]
    score_result = compute_score_test_chunk(
        chromosome_state,
        low_score_genotype_matrix,
        APPROXIMATE_FIRTH_PLAN,
    )
    assert int(np.asarray(score_result.extra_code[0])) == regenie2_binary.EXTRA_CODE_SCORE

    sparse_result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state(
        chromosome_state,
        low_score_genotype_matrix,
        APPROXIMATE_FIRTH_PLAN,
        jnp.asarray([True], dtype=jnp.bool_),
    )

    assert int(np.asarray(sparse_result.extra_code[0])) == regenie2_binary.EXTRA_CODE_SCORE
    assert int(np.asarray(sparse_result.firth_iteration_count[0])) == 0


def test_firth_candidate_capacity_overflow_matches_full_chunk_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    genotype_matrix, chromosome_state = build_chromosome_state()
    score_result = compute_score_test_chunk(
        chromosome_state,
        genotype_matrix,
        APPROXIMATE_FIRTH_PLAN,
    )
    forced_candidate_result = regenie2_binary_types.Regenie2BinaryChunkResult(
        beta=score_result.beta,
        standard_error=score_result.standard_error,
        chi_squared=score_result.chi_squared,
        log10_p_value=score_result.log10_p_value,
        extra_code=jnp.full((genotype_matrix.shape[1],), regenie2_binary.EXTRA_CODE_FIRTH, dtype=jnp.int32),
        valid_mask=jnp.ones((genotype_matrix.shape[1],), dtype=jnp.bool_),
        firth_iteration_count=jnp.zeros((genotype_matrix.shape[1],), dtype=jnp.int32),
        firth_failure_code=jnp.zeros((genotype_matrix.shape[1],), dtype=jnp.int32),
    )

    regenie2_binary.regenie2_binary_candidate_planning.configure_firth_candidate_planning(
        firth_batch_size=regenie2_binary.DEFAULT_FIRTH_BATCH_SIZE,
        firth_candidate_capacity=1,
    )
    clear_binary_compute_caches()
    overflow_result = regenie2_binary.apply_device_candidate_corrections(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        result=forced_candidate_result,
        correction_plan=APPROXIMATE_FIRTH_PLAN,
    )

    regenie2_binary.regenie2_binary_candidate_planning.configure_firth_candidate_planning(
        firth_batch_size=regenie2_binary.DEFAULT_FIRTH_BATCH_SIZE,
        firth_candidate_capacity=8,
    )
    clear_binary_compute_caches()
    bounded_result = regenie2_binary.apply_device_candidate_corrections(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        result=forced_candidate_result,
        correction_plan=APPROXIMATE_FIRTH_PLAN,
    )
    regenie2_binary.regenie2_binary_candidate_planning.configure_firth_candidate_planning(
        firth_batch_size=regenie2_binary.DEFAULT_FIRTH_BATCH_SIZE,
        firth_candidate_capacity=regenie2_binary.DEFAULT_FIRTH_CANDIDATE_CAPACITY,
    )

    np.testing.assert_allclose(np.asarray(overflow_result.beta), np.asarray(bounded_result.beta), equal_nan=True)
    np.testing.assert_allclose(
        np.asarray(overflow_result.standard_error),
        np.asarray(bounded_result.standard_error),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray(overflow_result.chi_squared),
        np.asarray(bounded_result.chi_squared),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray(overflow_result.log10_p_value),
        np.asarray(bounded_result.log10_p_value),
        equal_nan=True,
    )
    np.testing.assert_array_equal(np.asarray(overflow_result.extra_code), np.asarray(bounded_result.extra_code))


def test_non_candidate_score_rows_remain_unchanged_after_device_correction() -> None:
    genotype_matrix, chromosome_state = build_chromosome_state()

    score_test_result = compute_score_test_chunk(
        chromosome_state,
        genotype_matrix,
        APPROXIMATE_FIRTH_PLAN,
    )
    corrected_result = compute_binary_chunk(
        chromosome_state,
        genotype_matrix,
        APPROXIMATE_FIRTH_PLAN,
    )

    non_candidate_mask = np.asarray(score_test_result.extra_code) == regenie2_binary.EXTRA_CODE_SCORE
    np.testing.assert_allclose(
        np.asarray(corrected_result.beta)[non_candidate_mask],
        np.asarray(score_test_result.beta)[non_candidate_mask],
    )
    np.testing.assert_allclose(
        np.asarray(corrected_result.standard_error)[non_candidate_mask],
        np.asarray(score_test_result.standard_error)[non_candidate_mask],
    )
    np.testing.assert_allclose(
        np.asarray(corrected_result.chi_squared)[non_candidate_mask],
        np.asarray(score_test_result.chi_squared)[non_candidate_mask],
    )
    np.testing.assert_allclose(
        np.asarray(corrected_result.log10_p_value)[non_candidate_mask],
        np.asarray(score_test_result.log10_p_value)[non_candidate_mask],
    )


def test_variant_major_score_test_matches_sample_major() -> None:
    genotype_matrix, chromosome_state = build_chromosome_state()

    sample_major_result = compute_score_test_chunk(
        chromosome_state,
        genotype_matrix,
        APPROXIMATE_FIRTH_PLAN,
    )
    variant_major_result = compute_score_test_chunk_variant_major(
        chromosome_state,
        jnp.transpose(genotype_matrix),
        APPROXIMATE_FIRTH_PLAN,
    )

    np.testing.assert_allclose(np.asarray(variant_major_result.beta), np.asarray(sample_major_result.beta))
    np.testing.assert_allclose(
        np.asarray(variant_major_result.standard_error),
        np.asarray(sample_major_result.standard_error),
    )
    np.testing.assert_allclose(
        np.asarray(variant_major_result.chi_squared),
        np.asarray(sample_major_result.chi_squared),
    )
    np.testing.assert_allclose(
        np.asarray(variant_major_result.log10_p_value),
        np.asarray(sample_major_result.log10_p_value),
    )
    np.testing.assert_array_equal(
        np.asarray(variant_major_result.extra_code), np.asarray(sample_major_result.extra_code)
    )


def test_variant_major_binary_chunk_matches_sample_major() -> None:
    genotype_matrix, chromosome_state = build_chromosome_state()

    sample_major_result = compute_binary_chunk(
        chromosome_state,
        genotype_matrix,
        APPROXIMATE_FIRTH_PLAN,
    )
    variant_major_result = compute_binary_chunk_variant_major(
        chromosome_state,
        jnp.transpose(genotype_matrix),
        APPROXIMATE_FIRTH_PLAN,
    )

    np.testing.assert_allclose(
        np.asarray(variant_major_result.beta), np.asarray(sample_major_result.beta), equal_nan=True
    )
    np.testing.assert_allclose(
        np.asarray(variant_major_result.standard_error),
        np.asarray(sample_major_result.standard_error),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray(variant_major_result.chi_squared),
        np.asarray(sample_major_result.chi_squared),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray(variant_major_result.log10_p_value),
        np.asarray(sample_major_result.log10_p_value),
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        np.asarray(variant_major_result.extra_code), np.asarray(sample_major_result.extra_code)
    )


def test_failed_firth_lanes_become_test_fail() -> None:
    covariate_matrix = jnp.asarray(
        [
            [1.0, 20.0],
            [1.0, 25.0],
            [1.0, 30.0],
            [1.0, 35.0],
            [1.0, 40.0],
            [1.0, 45.0],
        ],
        dtype=jnp.float32,
    )
    phenotype_vector = jnp.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
    state = regenie2_binary.prepare_regenie2_binary_state(covariate_matrix, phenotype_vector)
    chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
        state,
        jnp.zeros((phenotype_vector.shape[0],), dtype=jnp.float32),
    )
    genotype_matrix = covariate_matrix[:, 1:2]
    score_result = compute_score_test_chunk(
        chromosome_state,
        genotype_matrix,
        APPROXIMATE_FIRTH_PLAN,
    )
    forced_candidate_result = regenie2_binary_types.Regenie2BinaryChunkResult(
        beta=score_result.beta,
        standard_error=score_result.standard_error,
        chi_squared=score_result.chi_squared,
        log10_p_value=score_result.log10_p_value,
        extra_code=jnp.asarray([regenie2_binary.EXTRA_CODE_FIRTH], dtype=jnp.int32),
        valid_mask=jnp.asarray([True]),
        firth_iteration_count=jnp.asarray([0], dtype=jnp.int32),
        firth_failure_code=jnp.asarray([0], dtype=jnp.int32),
    )

    corrected_result = regenie2_binary.apply_device_candidate_corrections(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        result=forced_candidate_result,
        correction_plan=APPROXIMATE_FIRTH_PLAN,
    )

    assert int(np.asarray(corrected_result.extra_code[0])) == regenie2_binary.EXTRA_CODE_TEST_FAIL
    assert not bool(np.asarray(corrected_result.valid_mask[0]))


@pytest.mark.skipif(
    not jax_backend_is_available("gpu") or not jax_backend_is_available("cpu"),
    reason="CPU or GPU backend unavailable",
)
def test_cpu_and_gpu_jax_outputs_match_on_toy_chunk() -> None:
    covariate_matrix, phenotype_vector, genotype_matrix = build_binary_inputs()
    cpu_device = jax.devices("cpu")[0]
    gpu_device = jax.devices("gpu")[0]

    cpu_covariates = jax.device_put(covariate_matrix, cpu_device)
    cpu_phenotype = jax.device_put(phenotype_vector, cpu_device)
    cpu_genotypes = jax.device_put(genotype_matrix, cpu_device)
    cpu_state = regenie2_binary.prepare_regenie2_binary_state(cpu_covariates, cpu_phenotype)
    cpu_chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
        cpu_state,
        jax.device_put(jnp.zeros((phenotype_vector.shape[0],), dtype=jnp.float32), cpu_device),
    )
    cpu_result = compute_binary_chunk(
        cpu_chromosome_state,
        cpu_genotypes,
        APPROXIMATE_FIRTH_PLAN,
    )

    gpu_covariates = jax.device_put(covariate_matrix, gpu_device)
    gpu_phenotype = jax.device_put(phenotype_vector, gpu_device)
    gpu_genotypes = jax.device_put(genotype_matrix, gpu_device)
    gpu_state = regenie2_binary.prepare_regenie2_binary_state(gpu_covariates, gpu_phenotype)
    gpu_chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
        gpu_state,
        jax.device_put(jnp.zeros((phenotype_vector.shape[0],), dtype=jnp.float32), gpu_device),
    )
    gpu_result = compute_binary_chunk(
        gpu_chromosome_state,
        gpu_genotypes,
        APPROXIMATE_FIRTH_PLAN,
    )

    np.testing.assert_allclose(np.asarray(cpu_result.beta), np.asarray(gpu_result.beta), rtol=1.0e-4, atol=1.0e-4)
    np.testing.assert_allclose(
        np.asarray(cpu_result.standard_error),
        np.asarray(gpu_result.standard_error),
        rtol=1.0e-4,
        atol=1.0e-4,
    )
    np.testing.assert_allclose(
        np.asarray(cpu_result.chi_squared),
        np.asarray(gpu_result.chi_squared),
        rtol=1.0e-4,
        atol=1.0e-4,
    )
    np.testing.assert_allclose(
        np.asarray(cpu_result.log10_p_value),
        np.asarray(gpu_result.log10_p_value),
        rtol=1.0e-4,
        atol=1.0e-4,
    )
