"""REGENIE step 2 binary score-test kernel with device-resident Firth fallback."""

from __future__ import annotations

import functools
import typing

import jax
import jax.numpy as jnp

from g import types
from g.compute import (
    regenie2_binary_config,
    regenie2_binary_correction,
    regenie2_binary_score,
    regenie2_binary_state,
    regenie2_binary_types,
    regenie2_binary_variant_major,
)

BinaryScoreTestChunkComputeFunction = typing.Callable[
    [regenie2_binary_types.Regenie2BinaryChromosomeState, jax.Array, types.BinaryCorrectionPlan],
    regenie2_binary_types.Regenie2BinaryChunkResult,
]
BinaryChunkComputeFunction = typing.Callable[
    [
        regenie2_binary_types.Regenie2BinaryChromosomeState,
        jax.Array,
        types.BinaryCorrectionPlan,
        jax.Array | None,
        regenie2_binary_types.BinaryKernelConfig,
    ],
    regenie2_binary_types.Regenie2BinaryChunkResult,
]
BinaryVariantMajorChunkComputeFunction = typing.Callable[
    [
        regenie2_binary_types.Regenie2BinaryChromosomeState,
        jax.Array,
        types.BinaryCorrectionPlan,
        jax.Array | None,
        regenie2_binary_types.BinaryKernelConfig,
    ],
    regenie2_binary_types.Regenie2BinaryChunkResult,
]


def compute_regenie_logistic_probability(linear_predictor: jax.Array) -> jax.Array:
    """Compute probabilities with REGENIE's glm-style endpoint clipping."""
    epsilon = jnp.asarray(regenie2_binary_config.REGENIE_NUMERICAL_EPSILON, dtype=linear_predictor.dtype)
    lower_probability = epsilon / (1.0 + epsilon)
    upper_probability = jnp.reciprocal(1.0 + epsilon)
    return jnp.where(
        linear_predictor > regenie2_binary_config.REGENIE_LOGISTIC_MAXIMUM_ETA,
        upper_probability,
        jnp.where(
            linear_predictor < regenie2_binary_config.REGENIE_LOGISTIC_MINIMUM_ETA,
            lower_probability,
            jax.nn.sigmoid(linear_predictor),
        ),
    )


def compute_logistic_deviance(
    phenotype_vector: jax.Array,
    probability_vector: jax.Array,
    active_sample_mask: jax.Array,
) -> jax.Array:
    """Compute REGENIE's Bernoulli deviance over active samples."""
    epsilon = jnp.asarray(regenie2_binary_config.REGENIE_NUMERICAL_EPSILON, dtype=probability_vector.dtype)
    clipped_probability = jnp.clip(
        probability_vector,
        epsilon / (1.0 + epsilon),
        jnp.reciprocal(1.0 + epsilon),
    )
    negative_log_likelihood = -jnp.where(
        phenotype_vector > regenie2_binary_config.BINARY_CASE_THRESHOLD,
        jnp.log(clipped_probability),
        jnp.log1p(-clipped_probability),
    )
    return 2.0 * jnp.sum(jnp.where(active_sample_mask, negative_log_likelihood, 0.0))


@functools.partial(jax.jit, static_argnames=("correction_plan",))
def compute_regenie2_binary_score_test_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Compute the uncorrected score-test result for one binary chunk."""
    return regenie2_binary_score.compute_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=jnp.asarray(genotype_matrix, dtype=jnp.float32).T,
        correction_plan=correction_plan,
    )


def build_single_binary_chromosome_state_from_multi(
    chromosome_state: regenie2_binary_types.Regenie2MultiBinaryChromosomeState,
    trait_index: jax.Array,
) -> regenie2_binary_types.Regenie2BinaryChromosomeState:
    """Build a single-trait chromosome state view from a multi-trait state."""
    return regenie2_binary_types.Regenie2BinaryChromosomeState(
        covariate_matrix=chromosome_state.covariate_matrix,
        phenotype_vector=chromosome_state.phenotype_matrix[trait_index],
        null_logistic_coefficients=chromosome_state.null_logistic_coefficients[trait_index],
        null_firth_coefficients=chromosome_state.null_firth_coefficients[trait_index],
        null_firth_offset=chromosome_state.null_firth_offset_matrix[trait_index],
        fitted_probability=chromosome_state.fitted_probability[trait_index],
        score_residual=chromosome_state.score_residual[trait_index],
        loco_offset=chromosome_state.loco_offset_matrix[trait_index],
        standardized_residual=chromosome_state.standardized_residual[trait_index],
        square_root_weight=chromosome_state.square_root_weight[trait_index],
        weighted_genotype_projection_matrix=chromosome_state.weighted_genotype_projection_matrix[trait_index],
        null_firth_penalized_log_likelihood=chromosome_state.null_firth_penalized_log_likelihood[trait_index],
        null_firth_iteration_count=chromosome_state.null_firth_iteration_count[trait_index],
        null_firth_convergence_reason_code=chromosome_state.null_firth_convergence_reason_code[trait_index],
        null_logistic_iteration_count=chromosome_state.null_logistic_iteration_count[trait_index],
        null_logistic_converged=chromosome_state.null_logistic_converged[trait_index],
    )


def build_multi_binary_chunk_result(
    result: regenie2_binary_types.Regenie2BinaryChunkResult,
) -> regenie2_binary_types.Regenie2MultiBinaryChunkResult:
    """Rewrap a vmapped single-trait binary result as a multi-trait result."""
    return regenie2_binary_types.Regenie2MultiBinaryChunkResult(
        beta=result.beta,
        standard_error=result.standard_error,
        chi_squared=result.chi_squared,
        log10_p_value=result.log10_p_value,
        extra_code=result.extra_code,
        valid_mask=result.valid_mask,
        firth_iteration_count=result.firth_iteration_count,
        firth_failure_code=result.firth_failure_code,
        firth_convergence_reason_code=result.firth_convergence_reason_code,
        firth_correction_code=result.firth_correction_code,
        firth_sparse_correction_mask=result.firth_sparse_correction_mask,
        pseudo_firth_iteration_count=result.pseudo_firth_iteration_count,
        nr_zero_start_iteration_count=result.nr_zero_start_iteration_count,
        nr_warm_start_iteration_count=result.nr_warm_start_iteration_count,
    )


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config"))
def compute_regenie2_multi_binary_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_types.Regenie2MultiBinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2MultiBinaryChunkResult:
    """Compute multi-trait binary REGENIE step 2 association using one genotype chunk."""
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        return regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=jnp.asarray(genotype_matrix, dtype=jnp.float32).T,
            correction_plan=correction_plan,
        )

    def compute_one_trait(trait_index: jax.Array) -> regenie2_binary_types.Regenie2BinaryChunkResult:
        single_chromosome_state = build_single_binary_chromosome_state_from_multi(chromosome_state, trait_index)
        return compute_regenie2_binary_chunk_from_chromosome_state(
            chromosome_state=single_chromosome_state,
            genotype_matrix=genotype_matrix,
            correction_plan=correction_plan,
            sparse_candidate_mask=sparse_candidate_mask,
            kernel_config=kernel_config,
        )

    trait_count = chromosome_state.phenotype_matrix.shape[0]
    return build_multi_binary_chunk_result(jax.vmap(compute_one_trait)(jnp.arange(trait_count, dtype=jnp.int32)))


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config"))
def compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_binary_types.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2MultiBinaryChunkResult:
    """Compute multi-trait binary association from variant-major genotypes."""
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        return regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            correction_plan=correction_plan,
        )

    def compute_one_trait(trait_index: jax.Array) -> regenie2_binary_types.Regenie2BinaryChunkResult:
        single_chromosome_state = build_single_binary_chromosome_state_from_multi(chromosome_state, trait_index)
        compute_variant_major_chunk = (
            regenie2_binary_variant_major.compute_regenie2_binary_chunk_from_chromosome_state_variant_major
        )
        return compute_variant_major_chunk(
            chromosome_state=single_chromosome_state,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            correction_plan=correction_plan,
            sparse_candidate_mask=sparse_candidate_mask,
            kernel_config=kernel_config,
        )

    trait_count = chromosome_state.phenotype_matrix.shape[0]
    return build_multi_binary_chunk_result(jax.vmap(compute_one_trait)(jnp.arange(trait_count, dtype=jnp.int32)))


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config"))
def compute_regenie2_binary_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Compute REGENIE step 2 binary association using cached null state."""
    score_test_result = compute_regenie2_binary_score_test_chunk_from_chromosome_state(
        chromosome_state,
        genotype_matrix,
        correction_plan,
    )
    return regenie2_binary_correction.apply_device_candidate_corrections(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        result=score_test_result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
    )


def compute_regenie2_binary_chunk(
    state: regenie2_binary_types.Regenie2BinaryState,
    genotype_matrix: jax.Array,
    loco_offset: jax.Array,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_types.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_types.Regenie2BinaryChunkResult:
    """Compute REGENIE step 2 binary association for a genotype chunk."""
    chromosome_state = regenie2_binary_state.prepare_regenie2_binary_chromosome_state(
        state, loco_offset, correction_plan, kernel_config
    )
    compute_regenie2_binary_chunk_from_state = typing.cast(
        "BinaryChunkComputeFunction",
        compute_regenie2_binary_chunk_from_chromosome_state,
    )
    return compute_regenie2_binary_chunk_from_state(
        chromosome_state,
        genotype_matrix,
        correction_plan,
        sparse_candidate_mask,
        kernel_config,
    )
