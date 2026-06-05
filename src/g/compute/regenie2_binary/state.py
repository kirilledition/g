"""Binary state preparation for REGENIE step 2."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g import types as g_types
from g.compute.common import dtype as compute_dtype
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import logistic as regenie2_binary_logistic
from g.compute.regenie2_binary import null_logistic as regenie2_binary_null_logistic
from g.compute.regenie2_binary.firth import null as regenie2_binary_firth_null
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2BinaryState:
    """Reusable state for REGENIE step 2 binary association.

    Attributes:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.

    """

    covariate_matrix: jax.Array
    phenotype_vector: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2BinaryChromosomeState:
    """Chromosome-specific binary null model state.

    Attributes:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.
        null_logistic_coefficients: Covariate-only null logistic coefficients.
        null_firth_offset: Covariate-only null Firth linear predictor plus LOCO offset.
        score_residual: Raw score residual, ``phenotype - fitted_probability``.
        loco_offset: LOCO offset in the logistic linear predictor.
        square_root_weight: Square root of Bernoulli variance.
        weighted_genotype_projection_matrix: Cholesky-whitened weighted covariate transpose.
        null_firth_penalized_log_likelihood: Covariate-only Firth null penalized log-likelihood.
        null_firth_iteration_count: Number of covariate-only Firth iterations.
        null_firth_convergence_reason_code: Internal covariate-only Firth termination-reason code.
        null_logistic_iteration_count: Number of IRLS updates used for the null logistic fit.
        null_logistic_converged: Whether the null logistic IRLS fit converged.

    """

    covariate_matrix: jax.Array
    phenotype_vector: jax.Array
    null_logistic_coefficients: jax.Array
    null_firth_offset: jax.Array
    score_residual: jax.Array
    loco_offset: jax.Array
    square_root_weight: jax.Array
    weighted_genotype_projection_matrix: jax.Array
    null_firth_penalized_log_likelihood: jax.Array
    null_firth_iteration_count: jax.Array
    null_firth_convergence_reason_code: jax.Array
    null_logistic_iteration_count: jax.Array
    null_logistic_converged: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiBinaryState:
    """Reusable state for multi-trait binary REGENIE step 2 association.

    Attributes:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_matrix: Binary phenotype matrix with shape ``traits x samples``.

    """

    covariate_matrix: jax.Array
    phenotype_matrix: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiBinaryChromosomeState:
    """Trait-major chromosome-specific binary null model state.

    Attributes:
        covariate_matrix: Shared covariate design matrix including intercept.
        phenotype_matrix: Binary phenotype matrix with shape ``traits x samples``.
        null_logistic_coefficients: Per-trait null logistic coefficients.
        null_firth_offset_matrix: Per-trait null Firth linear predictors plus LOCO offsets.
        score_residual: Per-trait raw score residuals.
        loco_offset_matrix: Per-trait LOCO offsets.
        square_root_weight: Per-trait square root Bernoulli variance.
        weighted_genotype_projection_matrix: Per-trait weighted covariate projection matrix.
        null_firth_penalized_log_likelihood: Per-trait Firth null penalized log-likelihood.
        null_firth_iteration_count: Per-trait covariate-only Firth iteration counts.
        null_firth_convergence_reason_code: Per-trait covariate-only Firth termination-reason codes.
        null_logistic_iteration_count: Per-trait null IRLS iteration counts.
        null_logistic_converged: Per-trait null IRLS convergence flags.

    """

    covariate_matrix: jax.Array
    phenotype_matrix: jax.Array
    null_logistic_coefficients: jax.Array
    null_firth_offset_matrix: jax.Array
    score_residual: jax.Array
    loco_offset_matrix: jax.Array
    square_root_weight: jax.Array
    weighted_genotype_projection_matrix: jax.Array
    null_firth_penalized_log_likelihood: jax.Array
    null_firth_iteration_count: jax.Array
    null_firth_convergence_reason_code: jax.Array
    null_logistic_iteration_count: jax.Array
    null_logistic_converged: jax.Array


def build_binary_state(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
) -> Regenie2BinaryState:
    """Build reusable binary step 2 state."""
    jax_dtype = compute_dtype.resolve_jax_dtype(score_dtype)
    covariate_matrix_compute = jnp.asarray(covariate_matrix, dtype=jax_dtype)
    phenotype_vector_compute = jnp.asarray(phenotype_vector, dtype=jax_dtype)
    return Regenie2BinaryState(
        covariate_matrix=covariate_matrix_compute,
        phenotype_vector=phenotype_vector_compute,
    )


def build_multi_binary_state(
    covariate_matrix: jax.Array,
    phenotype_matrix: jax.Array,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
) -> Regenie2MultiBinaryState:
    """Build reusable multi-trait binary step 2 state."""
    jax_dtype = compute_dtype.resolve_jax_dtype(score_dtype)
    covariate_matrix_compute = jnp.asarray(covariate_matrix, dtype=jax_dtype)
    phenotype_matrix_compute = jnp.asarray(phenotype_matrix, dtype=jax_dtype)
    return Regenie2MultiBinaryState(
        covariate_matrix=covariate_matrix_compute,
        phenotype_matrix=phenotype_matrix_compute,
    )


def build_binary_chromosome_state(
    state: Regenie2BinaryState,
    loco_offset: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
) -> Regenie2BinaryChromosomeState:
    """Build chromosome-specific binary null model state reused across chunks."""
    jax_dtype = compute_dtype.resolve_jax_dtype(score_dtype)
    loco_offset_compute = jnp.asarray(loco_offset, dtype=jax_dtype)
    null_logistic_fit_state = regenie2_binary_null_logistic.fit_null_logistic_coefficients(
        covariate_matrix=state.covariate_matrix,
        phenotype_vector=state.phenotype_vector,
        loco_offset=loco_offset_compute,
        maximum_iterations=None,
        kernel_config=kernel_config,
    )
    null_logistic_coefficients = null_logistic_fit_state.coefficients
    fitted_probability = regenie2_binary_logistic.compute_clipped_logistic_probability(
        state.covariate_matrix @ null_logistic_coefficients + loco_offset_compute,
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
        + jnp.eye(weighted_covariate_crossproduct.shape[0], dtype=jax_dtype) * kernel_config.numerical.minimum_variance
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
            loco_offset_compute, dtype=jnp.float64
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
            loco_offset=loco_offset_compute,
            initial_coefficients=null_logistic_coefficients,
            kernel_config=kernel_config,
        )
        null_firth_offset = state.covariate_matrix.astype(jnp.float64) @ null_firth_result.coefficients + jnp.asarray(
            loco_offset_compute, dtype=jnp.float64
        )
    return Regenie2BinaryChromosomeState(
        covariate_matrix=state.covariate_matrix,
        phenotype_vector=state.phenotype_vector,
        null_logistic_coefficients=null_logistic_coefficients,
        null_firth_offset=null_firth_offset,
        score_residual=score_residual,
        loco_offset=loco_offset_compute,
        square_root_weight=square_root_weight,
        weighted_genotype_projection_matrix=weighted_genotype_projection_matrix,
        null_firth_penalized_log_likelihood=null_firth_result.penalized_log_likelihood,
        null_firth_iteration_count=null_firth_result.iteration_count,
        null_firth_convergence_reason_code=null_firth_result.convergence_reason_code,
        null_logistic_iteration_count=null_logistic_fit_state.iteration_count,
        null_logistic_converged=null_logistic_fit_state.converged,
    )


def build_multi_binary_chromosome_state(
    state: Regenie2MultiBinaryState,
    loco_offset_matrix: jax.Array,
    correction_plan: g_types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    score_dtype: g_types.FloatingPointDtype = g_types.FloatingPointDtype.FLOAT32,
) -> Regenie2MultiBinaryChromosomeState:
    """Build chromosome-specific null logistic state for all requested binary traits."""
    loco_offset_matrix_compute = jnp.asarray(loco_offset_matrix, dtype=compute_dtype.resolve_jax_dtype(score_dtype))

    def prepare_one_trait(
        phenotype_vector: jax.Array,
        loco_offset: jax.Array,
    ) -> Regenie2BinaryChromosomeState:
        trait_state = Regenie2BinaryState(
            covariate_matrix=state.covariate_matrix,
            phenotype_vector=phenotype_vector,
        )
        return build_binary_chromosome_state(trait_state, loco_offset, correction_plan, kernel_config, score_dtype)

    chromosome_states = jax.vmap(prepare_one_trait)(state.phenotype_matrix, loco_offset_matrix_compute)
    return Regenie2MultiBinaryChromosomeState(
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


def build_single_binary_chromosome_state_from_multi(
    chromosome_state: Regenie2MultiBinaryChromosomeState,
    trait_index: int | jax.Array,
) -> Regenie2BinaryChromosomeState:
    """Build a single-trait chromosome state view from a multi-trait state."""
    return Regenie2BinaryChromosomeState(
        covariate_matrix=chromosome_state.covariate_matrix,
        phenotype_vector=chromosome_state.phenotype_matrix[trait_index],
        null_logistic_coefficients=chromosome_state.null_logistic_coefficients[trait_index],
        null_firth_offset=chromosome_state.null_firth_offset_matrix[trait_index],
        score_residual=chromosome_state.score_residual[trait_index],
        loco_offset=chromosome_state.loco_offset_matrix[trait_index],
        square_root_weight=chromosome_state.square_root_weight[trait_index],
        weighted_genotype_projection_matrix=chromosome_state.weighted_genotype_projection_matrix[trait_index],
        null_firth_penalized_log_likelihood=chromosome_state.null_firth_penalized_log_likelihood[trait_index],
        null_firth_iteration_count=chromosome_state.null_firth_iteration_count[trait_index],
        null_firth_convergence_reason_code=chromosome_state.null_firth_convergence_reason_code[trait_index],
        null_logistic_iteration_count=chromosome_state.null_logistic_iteration_count[trait_index],
        null_logistic_converged=chromosome_state.null_logistic_converged[trait_index],
    )


def build_multi_binary_chromosome_state_from_single(
    chromosome_state: Regenie2BinaryChromosomeState,
) -> Regenie2MultiBinaryChromosomeState:
    """Build a one-trait binary chromosome state view from a single-trait state."""
    return Regenie2MultiBinaryChromosomeState(
        covariate_matrix=chromosome_state.covariate_matrix,
        phenotype_matrix=chromosome_state.phenotype_vector[None, :],
        null_logistic_coefficients=chromosome_state.null_logistic_coefficients[None, :],
        null_firth_offset_matrix=chromosome_state.null_firth_offset[None, :],
        score_residual=chromosome_state.score_residual[None, :],
        loco_offset_matrix=chromosome_state.loco_offset[None, :],
        square_root_weight=chromosome_state.square_root_weight[None, :],
        weighted_genotype_projection_matrix=chromosome_state.weighted_genotype_projection_matrix[None, :, :],
        null_firth_penalized_log_likelihood=chromosome_state.null_firth_penalized_log_likelihood[None],
        null_firth_iteration_count=chromosome_state.null_firth_iteration_count[None],
        null_firth_convergence_reason_code=chromosome_state.null_firth_convergence_reason_code[None],
        null_logistic_iteration_count=chromosome_state.null_logistic_iteration_count[None],
        null_logistic_converged=chromosome_state.null_logistic_converged[None],
    )
