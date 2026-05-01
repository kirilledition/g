from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from g import models
from g.compute import regenie2_binary
from g.types import RegenieBinaryCorrection


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
    models.Regenie2BinaryChromosomeState,
]:
    covariate_matrix, phenotype_vector, genotype_matrix = build_binary_inputs()
    state = regenie2_binary.prepare_regenie2_binary_state(covariate_matrix, phenotype_vector)
    chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
        state,
        jnp.zeros((phenotype_vector.shape[0],), dtype=jnp.float32),
    )
    return genotype_matrix, chromosome_state


def test_device_firth_candidate_correction_returns_finite_statistics() -> None:
    genotype_matrix, chromosome_state = build_chromosome_state()
    candidate_genotype_matrix = genotype_matrix[:, :1]
    score_result = regenie2_binary.compute_regenie2_binary_score_test_chunk_from_chromosome_state(
        chromosome_state=chromosome_state,
        genotype_matrix=candidate_genotype_matrix,
        correction=RegenieBinaryCorrection.FIRTH_APPROXIMATE,
    )
    forced_candidate_result = models.Regenie2BinaryChunkResult(
        beta=score_result.beta,
        standard_error=score_result.standard_error,
        chi_squared=score_result.chi_squared,
        log10_p_value=score_result.log10_p_value,
        extra_code=jnp.asarray([regenie2_binary.EXTRA_CODE_FIRTH], dtype=jnp.int32),
        valid_mask=jnp.asarray([True]),
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


def test_non_candidate_score_rows_remain_unchanged_after_device_correction() -> None:
    genotype_matrix, chromosome_state = build_chromosome_state()

    score_test_result = regenie2_binary.compute_regenie2_binary_score_test_chunk_from_chromosome_state(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        correction=RegenieBinaryCorrection.FIRTH_APPROXIMATE,
    )
    corrected_result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        correction=RegenieBinaryCorrection.FIRTH_APPROXIMATE,
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
    score_result = regenie2_binary.compute_regenie2_binary_score_test_chunk_from_chromosome_state(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        correction=RegenieBinaryCorrection.FIRTH_APPROXIMATE,
    )
    forced_candidate_result = models.Regenie2BinaryChunkResult(
        beta=score_result.beta,
        standard_error=score_result.standard_error,
        chi_squared=score_result.chi_squared,
        log10_p_value=score_result.log10_p_value,
        extra_code=jnp.asarray([regenie2_binary.EXTRA_CODE_FIRTH], dtype=jnp.int32),
        valid_mask=jnp.asarray([True]),
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
    not jax.devices("gpu") or not jax.devices("cpu"),
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
    cpu_result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state(
        chromosome_state=cpu_chromosome_state,
        genotype_matrix=cpu_genotypes,
        correction=RegenieBinaryCorrection.FIRTH_APPROXIMATE,
    )

    gpu_covariates = jax.device_put(covariate_matrix, gpu_device)
    gpu_phenotype = jax.device_put(phenotype_vector, gpu_device)
    gpu_genotypes = jax.device_put(genotype_matrix, gpu_device)
    gpu_state = regenie2_binary.prepare_regenie2_binary_state(gpu_covariates, gpu_phenotype)
    gpu_chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
        gpu_state,
        jax.device_put(jnp.zeros((phenotype_vector.shape[0],), dtype=jnp.float32), gpu_device),
    )
    gpu_result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state(
        chromosome_state=gpu_chromosome_state,
        genotype_matrix=gpu_genotypes,
        correction=RegenieBinaryCorrection.FIRTH_APPROXIMATE,
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
