from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from g.compute import regenie2_binary, regenie2_binary_types
from g.types import RegenieBinaryCorrection

BinaryChunkComputeFunction = typing.Callable[
    [regenie2_binary_types.Regenie2BinaryChromosomeState, jax.Array, RegenieBinaryCorrection],
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


def test_firth_candidate_capacity_rejects_invalid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G_REGENIE2_BINARY_FIRTH_CANDIDATE_CAPACITY", "0")
    regenie2_binary.get_firth_candidate_capacity.cache_clear()

    with pytest.raises(ValueError, match="G_REGENIE2_BINARY_FIRTH_CANDIDATE_CAPACITY"):
        regenie2_binary.get_firth_candidate_capacity()


def test_device_firth_batch_plan_uses_candidate_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G_REGENIE2_BINARY_FIRTH_BATCH_SIZE", "2")
    clear_binary_compute_caches()
    fallback_mask = jnp.asarray([True, False, True, False, True], dtype=jnp.bool_)

    batch_plan = regenie2_binary.build_device_firth_batch_plan(fallback_mask, candidate_capacity=4)

    np.testing.assert_array_equal(np.asarray(batch_plan.fallback_index_matrix), [[0, 2], [4, 0]])
    np.testing.assert_array_equal(np.asarray(batch_plan.fallback_active_mask_matrix), [[True, True], [True, False]])
    np.testing.assert_array_equal(np.asarray(batch_plan.active_flat_position_vector), [0, 1, 2, 0])


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


def test_device_firth_candidate_correction_returns_finite_statistics() -> None:
    genotype_matrix, chromosome_state = build_chromosome_state()
    candidate_genotype_matrix = genotype_matrix[:, :1]
    score_result = compute_score_test_chunk(
        chromosome_state,
        candidate_genotype_matrix,
        RegenieBinaryCorrection.FIRTH_APPROXIMATE,
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
        correction=RegenieBinaryCorrection.FIRTH_APPROXIMATE,
    )

    assert np.isfinite(np.asarray(result.beta[0]))
    assert np.isfinite(np.asarray(result.standard_error[0]))
    assert np.isfinite(np.asarray(result.chi_squared[0]))
    assert np.isfinite(np.asarray(result.log10_p_value[0]))
    assert int(np.asarray(result.extra_code[0])) == regenie2_binary.EXTRA_CODE_FIRTH
    assert bool(np.asarray(result.valid_mask[0]))


def test_firth_candidate_capacity_overflow_matches_full_chunk_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    genotype_matrix, chromosome_state = build_chromosome_state()
    score_result = compute_score_test_chunk(
        chromosome_state,
        genotype_matrix,
        RegenieBinaryCorrection.FIRTH_APPROXIMATE,
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

    monkeypatch.setenv("G_REGENIE2_BINARY_FIRTH_CANDIDATE_CAPACITY", "1")
    clear_binary_compute_caches()
    overflow_result = regenie2_binary.apply_device_candidate_corrections(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        result=forced_candidate_result,
        correction=RegenieBinaryCorrection.FIRTH_APPROXIMATE,
    )

    monkeypatch.setenv("G_REGENIE2_BINARY_FIRTH_CANDIDATE_CAPACITY", "8")
    clear_binary_compute_caches()
    bounded_result = regenie2_binary.apply_device_candidate_corrections(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        result=forced_candidate_result,
        correction=RegenieBinaryCorrection.FIRTH_APPROXIMATE,
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
        RegenieBinaryCorrection.FIRTH_APPROXIMATE,
    )
    corrected_result = compute_binary_chunk(
        chromosome_state,
        genotype_matrix,
        RegenieBinaryCorrection.FIRTH_APPROXIMATE,
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
        RegenieBinaryCorrection.FIRTH_APPROXIMATE,
    )
    variant_major_result = compute_score_test_chunk_variant_major(
        chromosome_state,
        jnp.transpose(genotype_matrix),
        RegenieBinaryCorrection.FIRTH_APPROXIMATE,
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
        RegenieBinaryCorrection.FIRTH_APPROXIMATE,
    )
    variant_major_result = compute_binary_chunk_variant_major(
        chromosome_state,
        jnp.transpose(genotype_matrix),
        RegenieBinaryCorrection.FIRTH_APPROXIMATE,
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
        RegenieBinaryCorrection.FIRTH_APPROXIMATE,
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
        correction=RegenieBinaryCorrection.FIRTH_APPROXIMATE,
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
        RegenieBinaryCorrection.FIRTH_APPROXIMATE,
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
        RegenieBinaryCorrection.FIRTH_APPROXIMATE,
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
