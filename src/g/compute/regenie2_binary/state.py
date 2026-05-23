"""Binary state preparation for REGENIE step 2."""

from __future__ import annotations

from dataclasses import dataclass

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2BinaryState:
    """Reusable state for REGENIE step 2 binary association.

    Attributes:
        covariate_matrix: Covariate design matrix including intercept.
        phenotype_vector: Binary phenotype vector in 0/1 encoding.
        sample_count: Number of samples.

    """

    covariate_matrix: jax.Array
    phenotype_vector: jax.Array
    sample_count: jax.Array


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
        sample_count: Number of samples.

    """

    covariate_matrix: jax.Array
    phenotype_matrix: jax.Array
    sample_count: jax.Array


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
