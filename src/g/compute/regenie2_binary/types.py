"""JAX pytree types for binary REGENIE step 2 compute."""

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
class Regenie2BinaryScoreChunkResult:
    """Score-test association outputs for a REGENIE step 2 binary chunk.

    Attributes:
        beta: Estimated effect sizes.
        standard_error: Standard errors of estimates.
        chi_squared: Chi-squared statistics.
        log10_p_value: Negative log10 p-values.
        extra_code: Integer value from `types.BinaryExtraCode` for output rendering.
        valid_mask: Boolean mask for valid statistics.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array
    valid_mask: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2BinaryChunkResult:
    """Association outputs for a REGENIE step 2 binary chunk.

    Attributes:
        beta: Estimated effect sizes.
        standard_error: Standard errors of estimates.
        chi_squared: Chi-squared statistics.
        log10_p_value: Negative log10 p-values.
        extra_code: Integer value from `types.BinaryExtraCode` for output rendering.
        valid_mask: Boolean mask for valid statistics.
        firth_iteration_count: Number of Firth iterations per variant, or zero for non-Firth rows.
        firth_failure_code: Integer value from `types.FirthFailureCode`, or zero for non-failed rows.
        firth_convergence_reason_code: Internal Firth termination-reason integer.
        firth_correction_code: Integer value from `types.FirthCorrectionCode`.
        firth_sparse_correction_mask: Whether the approximate correction used carrier-only sparse inputs.
        pseudo_firth_iteration_count: Scalar pseudo-Firth iterations per variant.
        nr_zero_start_iteration_count: Scalar Newton-Raphson zero-start iterations per variant.
        nr_warm_start_iteration_count: Scalar Newton-Raphson warm-start iterations per variant.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array
    valid_mask: jax.Array
    firth_iteration_count: jax.Array
    firth_failure_code: jax.Array
    firth_convergence_reason_code: jax.Array
    firth_correction_code: jax.Array
    firth_sparse_correction_mask: jax.Array
    pseudo_firth_iteration_count: jax.Array
    nr_zero_start_iteration_count: jax.Array
    nr_warm_start_iteration_count: jax.Array


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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiBinaryScoreChunkResult:
    """Trait-major score-test outputs for a multi-trait binary chunk.

    Attributes:
        beta: Estimated effect sizes with shape ``traits x variants``.
        standard_error: Standard errors with shape ``traits x variants``.
        chi_squared: Chi-squared statistics with shape ``traits x variants``.
        log10_p_value: Negative log10 p-values with shape ``traits x variants``.
        extra_code: Integer values from `types.BinaryExtraCode` with shape ``traits x variants``.
        valid_mask: Boolean mask for valid statistics with shape ``traits x variants``.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array
    valid_mask: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiBinaryChunkResult:
    """Trait-major association outputs for a multi-trait binary chunk.

    Attributes:
        beta: Estimated effect sizes with shape ``traits x variants``.
        standard_error: Standard errors with shape ``traits x variants``.
        chi_squared: Chi-squared statistics with shape ``traits x variants``.
        log10_p_value: Negative log10 p-values with shape ``traits x variants``.
        extra_code: Integer values from `types.BinaryExtraCode` with shape ``traits x variants``.
        valid_mask: Boolean mask for valid statistics with shape ``traits x variants``.
        firth_iteration_count: Firth iteration counts with shape ``traits x variants``.
        firth_failure_code: Values from `types.FirthFailureCode` with shape ``traits x variants``.
        firth_convergence_reason_code: Internal Firth termination-reason integers with shape ``traits x variants``.
        firth_correction_code: Values from `types.FirthCorrectionCode` with shape ``traits x variants``.
        firth_sparse_correction_mask: Sparse carrier-only correction flags with shape ``traits x variants``.
        pseudo_firth_iteration_count: Scalar pseudo-Firth iteration counts with shape ``traits x variants``.
        nr_zero_start_iteration_count: Scalar zero-start NR iteration counts with shape ``traits x variants``.
        nr_warm_start_iteration_count: Scalar warm-start NR iteration counts with shape ``traits x variants``.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array
    valid_mask: jax.Array
    firth_iteration_count: jax.Array
    firth_failure_code: jax.Array
    firth_convergence_reason_code: jax.Array
    firth_correction_code: jax.Array
    firth_sparse_correction_mask: jax.Array
    pseudo_firth_iteration_count: jax.Array
    nr_zero_start_iteration_count: jax.Array
    nr_warm_start_iteration_count: jax.Array
