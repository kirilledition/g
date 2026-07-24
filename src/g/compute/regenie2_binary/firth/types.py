"""JAX pytree containers for binary Firth correction kernels."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthFitResult:
    """Result of the covariate-only Firth null fit.

    Attributes:
        coefficients: Final covariate coefficients, or the last attempted coefficients on failure.
        penalized_log_likelihood: Final trusted penalized log-likelihood, or NaN on failure.
        converged: Whether the null fit converged.

    """

    coefficients: jax.Array
    penalized_log_likelihood: jax.Array
    converged: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthComponents:
    """Intermediate quantities for REGENIE-style null Firth Newton-Raphson."""

    information_cholesky_factor: jax.Array
    deviance: jax.Array
    modified_score: jax.Array
    valid: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthScoreHistoryState:
    """Consecutive score-increase state for null Firth convergence checks.

    Attributes:
        previous_score_maximum: Maximum absolute score from the immediately previous iterate.
        score_increase_count: Number of consecutive score increases.
        failed: Whether the enabled consecutive-increase limit was exceeded.

    """

    previous_score_maximum: jax.Array
    score_increase_count: jax.Array
    failed: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthNewtonRaphsonState:
    """Loop state for covariate-only null Firth Newton-Raphson."""

    coefficients: jax.Array
    deviance: jax.Array
    converged: jax.Array
    failed: jax.Array
    iteration_count: jax.Array
    previous_score_maximum: jax.Array
    score_increase_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthLineSearchState:
    """Line-search state for covariate-only null Firth Newton-Raphson."""

    attempt_count: jax.Array
    next_coefficient_step: jax.Array
    accepted_coefficients: jax.Array
    accepted_deviance: jax.Array
    accepted: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthLineSearchResult:
    """Result of null Firth deviance-decreasing step-halving."""

    coefficients: jax.Array
    deviance: jax.Array
    accepted: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NullFirthFallbackState:
    """Mutable state for lazy null Firth fallback attempts."""

    selected_result: NullFirthFitResult
    next_attempt_index: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FirthVariantResult:
    """Firth outputs for one genotype lane.

    Attributes:
        beta: Corrected genotype effect.
        standard_error: Standard error of the corrected effect.
        chi_squared: Likelihood-ratio chi-squared statistic.
        log10_p_value: Negative log10 p-value.
        valid_mask: Whether corrected statistics are valid.

    """

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    valid_mask: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarFirthTerminalResult:
    """Terminal scalar solver quantities awaiting shared finalization."""

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    valid_mask: jax.Array


def build_empty_firth_variant_result(batch_size: int) -> FirthVariantResult:
    """Build a placeholder Firth result for skipped padded batches."""
    return FirthVariantResult(
        beta=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        standard_error=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        chi_squared=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        log10_p_value=jnp.full((batch_size,), jnp.nan, dtype=jnp.float64),
        valid_mask=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


def flatten_batched_firth_variant_result(result: FirthVariantResult) -> FirthVariantResult:
    """Flatten batched Firth outputs into candidate-lane order."""
    return FirthVariantResult(
        beta=result.beta.reshape((-1,)),
        standard_error=result.standard_error.reshape((-1,)),
        chi_squared=result.chi_squared.reshape((-1,)),
        log10_p_value=result.log10_p_value.reshape((-1,)),
        valid_mask=result.valid_mask.reshape((-1,)),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarFirthComponents:
    """Scalar approximate-Firth quantities for one beta value.

    Attributes:
        genotype_information: Scalar genotype information.
        score_adjustment: Scalar leverage adjustment to the logistic score.
        penalized_deviance: REGENIE approximate penalized deviance.
        score: Scalar modified score.
        valid: Whether probabilities, the score, and information are finite and usable.

    """

    genotype_information: jax.Array
    score_adjustment: jax.Array
    penalized_deviance: jax.Array
    score: jax.Array
    valid: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarPseudoFirthState:
    """Loop state for REGENIE scalar pseudo-Firth."""

    beta: jax.Array
    components: ScalarFirthComponents
    outer_iteration_count: jax.Array
    beta_iteration_14: jax.Array
    converged: jax.Array
    failed: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarPseudoLogisticState:
    """Inner pseudo-response logistic state for one scalar beta update."""

    beta: jax.Array
    score: jax.Array
    genotype_information: jax.Array
    previous_step_size: jax.Array
    iteration_count: jax.Array
    converged: jax.Array
    failed: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarNewtonRaphsonState:
    """Loop state for REGENIE scalar Newton-Raphson Firth fallback."""

    beta: jax.Array
    penalized_deviance: jax.Array
    genotype_information: jax.Array
    score: jax.Array
    iteration_count: jax.Array
    converged: jax.Array
    failed: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarLineSearchState:
    """Line-search state for scalar Newton-Raphson Firth."""

    beta: jax.Array
    step_size: jax.Array
    penalized_deviance: jax.Array
    genotype_information: jax.Array
    score: jax.Array
    attempt_count: jax.Array
    accepted: jax.Array
    valid: jax.Array


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=(
        "minimum_variance",
        "tolerance",
        "maximum_step_size",
        "pseudo_maximum_iterations",
        "pseudo_inner_maximum_iterations",
        "newton_raphson_maximum_iterations",
        "line_search_maximum_attempts",
    ),
    meta_fields=("use_cuda_components",),
)
@dataclass(frozen=True)
class ScalarApproximateFirthSolverParameters:
    """Scalar approximate-Firth policy values carried through JAX branches."""

    minimum_variance: jax.Array
    tolerance: jax.Array
    maximum_step_size: jax.Array
    pseudo_maximum_iterations: jax.Array
    pseudo_inner_maximum_iterations: jax.Array
    newton_raphson_maximum_iterations: jax.Array
    line_search_maximum_attempts: jax.Array
    use_cuda_components: bool

    @property
    def sparse_pseudo_maximum_iterations(self) -> jax.Array:
        """Return the uncapped half-budget shared with Newton-Raphson."""
        return self.newton_raphson_maximum_iterations


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScalarApproximateFirthInitialState:
    """Shared initial state for pseudo-Firth and Newton-Raphson fits."""

    phenotype_vector: jax.Array
    genotype_vector: jax.Array
    offset_vector: jax.Array
    active_sample_mask: jax.Array
    non_active_deviance: jax.Array
    solver_parameters: ScalarApproximateFirthSolverParameters
    beta: jax.Array
    components: ScalarFirthComponents
    deviance_null: jax.Array
