"""Public binary REGENIE step 2 compute API."""

from __future__ import annotations

import functools
import typing

import jax
import jax.numpy as jnp

from g import types as g_types
from g.compute.common import genotype
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import diagnostics as regenie2_binary_diagnostics
from g.compute.regenie2_binary import logistic as regenie2_binary_logistic
from g.compute.regenie2_binary import null_logistic as regenie2_binary_null_logistic
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import score as regenie2_binary_score
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_binary import variant_major_correction as regenie2_binary_variant_major_correction
from g.compute.regenie2_binary.firth import null as regenie2_binary_firth_null
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types

BinaryChunkDiagnostics = regenie2_binary_diagnostics.BinaryChunkDiagnostics
Regenie2BinaryState = regenie2_binary_state.Regenie2BinaryState
Regenie2BinaryChromosomeState = regenie2_binary_state.Regenie2BinaryChromosomeState
Regenie2MultiBinaryState = regenie2_binary_state.Regenie2MultiBinaryState
Regenie2MultiBinaryChromosomeState = regenie2_binary_state.Regenie2MultiBinaryChromosomeState
Regenie2BinaryScoreChunkResult = regenie2_binary_result.Regenie2BinaryScoreChunkResult
Regenie2BinaryChunkResult = regenie2_binary_result.Regenie2BinaryChunkResult
Regenie2MultiBinaryScoreChunkResult = regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult
Regenie2MultiBinaryChunkResult = regenie2_binary_result.Regenie2MultiBinaryChunkResult
count_binary_chunk_diagnostics = regenie2_binary_diagnostics.count_binary_chunk_diagnostics


def prepare_regenie2_binary_state(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
) -> regenie2_binary_state.Regenie2BinaryState:
    """Prepare reusable binary step 2 state."""
    covariate_matrix_float32 = jnp.asarray(covariate_matrix, dtype=jnp.float32)
    phenotype_vector_float32 = jnp.asarray(phenotype_vector, dtype=jnp.float32)
    return regenie2_binary_state.Regenie2BinaryState(
        covariate_matrix=covariate_matrix_float32,
        phenotype_vector=phenotype_vector_float32,
        sample_count=jnp.asarray(covariate_matrix_float32.shape[0], dtype=jnp.int32),
    )


def prepare_regenie2_multi_binary_state(
    covariate_matrix: jax.Array,
    phenotype_matrix: jax.Array,
) -> regenie2_binary_state.Regenie2MultiBinaryState:
    """Prepare reusable multi-trait binary step 2 state."""
    covariate_matrix_float32 = jnp.asarray(covariate_matrix, dtype=jnp.float32)
    phenotype_matrix_float32 = jnp.asarray(phenotype_matrix, dtype=jnp.float32)
    return regenie2_binary_state.Regenie2MultiBinaryState(
        covariate_matrix=covariate_matrix_float32,
        phenotype_matrix=phenotype_matrix_float32,
        sample_count=jnp.asarray(covariate_matrix_float32.shape[0], dtype=jnp.int32),
    )


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config"))
def prepare_regenie2_binary_chromosome_state(
    state: regenie2_binary_state.Regenie2BinaryState,
    loco_offset: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_state.Regenie2BinaryChromosomeState:
    """Prepare chromosome-specific null logistic state reused across chunks."""
    loco_offset_float32 = jnp.asarray(loco_offset, dtype=jnp.float32)
    null_logistic_fit_state = regenie2_binary_null_logistic.fit_null_logistic_coefficients(
        covariate_matrix=state.covariate_matrix,
        phenotype_vector=state.phenotype_vector,
        loco_offset=loco_offset_float32,
        maximum_iterations=None,
        kernel_config=kernel_config,
    )
    null_logistic_coefficients = null_logistic_fit_state.coefficients
    fitted_probability = regenie2_binary_logistic.compute_clipped_logistic_probability(
        state.covariate_matrix @ null_logistic_coefficients + loco_offset_float32,
        kernel_config,
    )
    bernoulli_variance = jnp.maximum(
        fitted_probability * (1.0 - fitted_probability),
        kernel_config.numerical.minimum_variance,
    )
    square_root_weight = jnp.sqrt(bernoulli_variance)
    score_residual = state.phenotype_vector - fitted_probability
    weighted_covariate_matrix = square_root_weight[:, None] * state.covariate_matrix
    weighted_covariate_transpose = weighted_covariate_matrix.T
    weighted_covariate_crossproduct = weighted_covariate_transpose @ weighted_covariate_matrix
    cholesky_factor = jnp.linalg.cholesky(
        weighted_covariate_crossproduct
        + jnp.eye(weighted_covariate_crossproduct.shape[0], dtype=jnp.float32)
        * kernel_config.numerical.minimum_variance
    )
    weighted_genotype_projection_matrix = jax.lax.linalg.triangular_solve(
        cholesky_factor,
        weighted_covariate_transpose,
        left_side=True,
        lower=True,
    )
    if correction_plan.method == g_types.BinaryFallbackMethod.SCORE_ONLY:
        null_firth_coefficients = jnp.asarray(null_logistic_coefficients, dtype=jnp.float64)
        null_firth_offset = state.covariate_matrix.astype(jnp.float64) @ null_firth_coefficients + jnp.asarray(
            loco_offset_float32, dtype=jnp.float64
        )
        null_firth_result = regenie2_binary_firth_types.NullFirthFitResult(
            coefficients=null_firth_coefficients,
            penalized_log_likelihood=jnp.asarray(0.0, dtype=jnp.float64),
            iteration_count=jnp.asarray(0, dtype=jnp.int32),
            convergence_reason_code=jnp.asarray(
                regenie2_binary_firth_types.FirthConvergenceReason.NONE.value,
                dtype=jnp.int32,
            ),
            converged=jnp.asarray(1, dtype=jnp.bool_),
        )
    else:
        null_firth_result = regenie2_binary_firth_null.fit_covariate_only_firth_null_model(
            covariate_matrix=state.covariate_matrix,
            phenotype_vector=state.phenotype_vector,
            loco_offset=loco_offset_float32,
            initial_coefficients=null_logistic_coefficients,
            kernel_config=kernel_config,
        )
        null_firth_offset = state.covariate_matrix.astype(jnp.float64) @ null_firth_result.coefficients + jnp.asarray(
            loco_offset_float32, dtype=jnp.float64
        )
    return regenie2_binary_state.Regenie2BinaryChromosomeState(
        covariate_matrix=state.covariate_matrix,
        phenotype_vector=state.phenotype_vector,
        null_logistic_coefficients=null_logistic_coefficients,
        null_firth_offset=null_firth_offset,
        score_residual=score_residual,
        loco_offset=loco_offset_float32,
        square_root_weight=square_root_weight,
        weighted_genotype_projection_matrix=weighted_genotype_projection_matrix,
        null_firth_penalized_log_likelihood=null_firth_result.penalized_log_likelihood,
        null_firth_iteration_count=null_firth_result.iteration_count,
        null_firth_convergence_reason_code=null_firth_result.convergence_reason_code,
        null_logistic_iteration_count=null_logistic_fit_state.iteration_count,
        null_logistic_converged=null_logistic_fit_state.converged,
    )


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config"))
def prepare_regenie2_multi_binary_chromosome_state(
    state: regenie2_binary_state.Regenie2MultiBinaryState,
    loco_offset_matrix: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_state.Regenie2MultiBinaryChromosomeState:
    """Prepare chromosome-specific null logistic state for all requested binary traits."""
    loco_offset_matrix_float32 = jnp.asarray(loco_offset_matrix, dtype=jnp.float32)

    def prepare_one_trait(
        phenotype_vector: jax.Array,
        loco_offset: jax.Array,
    ) -> regenie2_binary_state.Regenie2BinaryChromosomeState:
        trait_state = regenie2_binary_state.Regenie2BinaryState(
            covariate_matrix=state.covariate_matrix,
            phenotype_vector=phenotype_vector,
            sample_count=state.sample_count,
        )
        return prepare_regenie2_binary_chromosome_state(trait_state, loco_offset, correction_plan, kernel_config)

    chromosome_states = jax.vmap(prepare_one_trait)(state.phenotype_matrix, loco_offset_matrix_float32)
    return regenie2_binary_state.Regenie2MultiBinaryChromosomeState(
        covariate_matrix=state.covariate_matrix,
        phenotype_matrix=state.phenotype_matrix,
        null_logistic_coefficients=chromosome_states.null_logistic_coefficients,
        null_firth_offset_matrix=chromosome_states.null_firth_offset,
        score_residual=chromosome_states.score_residual,
        loco_offset_matrix=chromosome_states.loco_offset,
        square_root_weight=chromosome_states.square_root_weight,
        weighted_genotype_projection_matrix=chromosome_states.weighted_genotype_projection_matrix,
        null_firth_penalized_log_likelihood=chromosome_states.null_firth_penalized_log_likelihood,
        null_firth_iteration_count=chromosome_states.null_firth_iteration_count,
        null_firth_convergence_reason_code=chromosome_states.null_firth_convergence_reason_code,
        null_logistic_iteration_count=chromosome_states.null_logistic_iteration_count,
        null_logistic_converged=chromosome_states.null_logistic_converged,
    )


@functools.partial(jax.jit, static_argnames=("correction_plan", "kernel_config"))
def compute_regenie2_binary_score_test_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult:
    """Compute the uncorrected score-test result for one binary chunk."""
    return regenie2_binary_score.compute_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype.convert_sample_major_to_variant_major(genotype_matrix),
        correction_plan=correction_plan,
        kernel_config=kernel_config,
    )


def compute_regenie2_multi_binary_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult | regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Compute multi-trait binary REGENIE step 2 association using one genotype chunk."""
    return compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype.convert_sample_major_to_variant_major(genotype_matrix),
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
    )


def compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult | regenie2_binary_result.Regenie2MultiBinaryChunkResult:
    """Compute multi-trait binary association from variant-major genotypes."""
    if correction_plan.method == g_types.BinaryFallbackMethod.SCORE_ONLY:
        return regenie2_binary_score.compute_multi_binary_score_test_chunk_variant_major(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            correction_plan=correction_plan,
            kernel_config=kernel_config,
        )

    def compute_one_trait(trait_index: int) -> regenie2_binary_result.Regenie2BinaryChunkResult:
        single_chromosome_state = regenie2_binary_state.build_single_binary_chromosome_state_from_multi(
            chromosome_state,
            trait_index,
        )
        result = compute_regenie2_binary_chunk_from_chromosome_state_variant_major(
            chromosome_state=single_chromosome_state,
            genotype_matrix_by_variant=genotype_matrix_by_variant,
            correction_plan=correction_plan,
            sparse_candidate_mask=sparse_candidate_mask,
            kernel_config=kernel_config,
        )
        return typing.cast("regenie2_binary_result.Regenie2BinaryChunkResult", result)

    trait_count = chromosome_state.phenotype_matrix.shape[0]
    return regenie2_binary_result.stack_binary_chunk_results(
        [compute_one_trait(trait_index) for trait_index in range(trait_count)]
    )


def compute_regenie2_binary_chunk_from_chromosome_state_variant_major(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix_by_variant: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult | regenie2_binary_result.Regenie2BinaryChunkResult:
    """Compute binary association from a variant-major chunk."""
    score_test_result = regenie2_binary_score.compute_binary_score_test_chunk_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        correction_plan=correction_plan,
        kernel_config=kernel_config,
    )
    return regenie2_binary_variant_major_correction.apply_device_candidate_corrections_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype_matrix_by_variant,
        result=score_test_result,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
    )


def compute_regenie2_binary_chunk_from_chromosome_state(
    chromosome_state: regenie2_binary_state.Regenie2BinaryChromosomeState,
    genotype_matrix: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult | regenie2_binary_result.Regenie2BinaryChunkResult:
    """Compute REGENIE step 2 binary association using cached null state."""
    return compute_regenie2_binary_chunk_from_chromosome_state_variant_major(
        chromosome_state=chromosome_state,
        genotype_matrix_by_variant=genotype.convert_sample_major_to_variant_major(genotype_matrix),
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
    )


def compute_regenie2_binary_chunk(
    state: regenie2_binary_state.Regenie2BinaryState,
    genotype_matrix: jax.Array,
    loco_offset: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan = g_types.BinaryCorrectionPlan(),
    sparse_candidate_mask: jax.Array | None = None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig = regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG,
) -> regenie2_binary_result.Regenie2BinaryScoreChunkResult | regenie2_binary_result.Regenie2BinaryChunkResult:
    """Compute REGENIE step 2 binary association for a genotype chunk."""
    chromosome_state = prepare_regenie2_binary_chromosome_state(state, loco_offset, correction_plan, kernel_config)
    return compute_regenie2_binary_chunk_from_chromosome_state(
        chromosome_state=chromosome_state,
        genotype_matrix=genotype_matrix,
        correction_plan=correction_plan,
        sparse_candidate_mask=sparse_candidate_mask,
        kernel_config=kernel_config,
    )
