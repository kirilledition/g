"""Binary state preparation for REGENIE step 2."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import logistic as regenie2_binary_logistic
from g.compute.regenie2_binary import null_logistic as regenie2_binary_null_logistic


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
class PreparedBinaryTraitState:
    """Prepared null-logistic quantities shared by score and Firth execution.

    Attributes:
        phenotype_vector: Binary phenotype vector.
        null_logistic_coefficients: Covariate-only null logistic coefficients.
        score_residual: Raw score residual.
        loco_offset: LOCO offset in the logistic linear predictor.
        square_root_weight: Square root of Bernoulli variance.
        bernoulli_weight: Bernoulli variance.
        weighted_genotype_projection_matrix: Cholesky-whitened weighted covariate transpose.
        score_projection_matrix: Cholesky-whitened score projection matrix.
        null_logistic_converged: Whether null logistic IRLS converged.

    """

    phenotype_vector: jax.Array
    null_logistic_coefficients: jax.Array
    score_residual: jax.Array
    loco_offset: jax.Array
    square_root_weight: jax.Array
    bernoulli_weight: jax.Array
    weighted_genotype_projection_matrix: jax.Array
    score_projection_matrix: jax.Array
    null_logistic_converged: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiBinaryScoreChromosomeState:
    """Minimal trait-major chromosome state retained by score-only execution.

    Attributes:
        score_right_hand_matrix: Stacked matrix multiplied by genotype chunks.
        bernoulli_weight: Per-trait Bernoulli variance.
        null_logistic_converged: Per-trait null IRLS convergence flags.

    """

    score_right_hand_matrix: jax.Array
    bernoulli_weight: jax.Array
    null_logistic_converged: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PreparedBinaryFirthTraitState:
    """Prepared per-trait state required only by approximate Firth.

    Attributes:
        score_state: Null-logistic quantities shared with score execution.
        null_firth_offset: Covariate-only null Firth predictor plus LOCO offset.
        full_null_deviance: Full-sample null deviance.
        null_firth_penalized_log_likelihood: Covariate-only Firth null likelihood.

    """

    score_state: PreparedBinaryTraitState
    null_firth_offset: jax.Array
    full_null_deviance: jax.Array
    null_firth_penalized_log_likelihood: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Regenie2MultiBinaryFirthChromosomeState:
    """Trait-major chromosome state retained by approximate-Firth execution.

    Attributes:
        score_state: Minimal state consumed by the shared score kernel.
        phenotype_matrix: Binary phenotype matrix with shape ``traits x samples``.
        null_firth_offset_matrix: Per-trait null Firth predictors plus LOCO offsets.
        square_root_weight: Per-trait square root Bernoulli variance.
        weighted_genotype_projection_matrix: Per-trait weighted covariate projection matrix.
        full_null_deviance: Per-trait full-sample null deviance.
        null_firth_penalized_log_likelihood: Per-trait Firth null likelihood.

    """

    score_state: Regenie2MultiBinaryScoreChromosomeState
    phenotype_matrix: jax.Array
    null_firth_offset_matrix: jax.Array
    square_root_weight: jax.Array
    weighted_genotype_projection_matrix: jax.Array
    full_null_deviance: jax.Array
    null_firth_penalized_log_likelihood: jax.Array


def build_multi_binary_state(
    covariate_matrix: jax.Array,
    phenotype_matrix: jax.Array,
) -> Regenie2MultiBinaryState:
    """Build reusable multi-trait binary step 2 state."""
    return Regenie2MultiBinaryState(
        covariate_matrix=jnp.asarray(covariate_matrix, dtype=jnp.float32),
        phenotype_matrix=jnp.asarray(phenotype_matrix, dtype=jnp.float32),
    )


def prepare_binary_trait_state(
    covariate_matrix: jax.Array,
    phenotype_vector: jax.Array,
    loco_offset: jax.Array,
    kernel_config: regenie2_binary_config.BinaryScoreConfig,
) -> PreparedBinaryTraitState:
    """Prepare shared null-logistic and score quantities for one trait."""
    loco_offset_compute = jnp.asarray(loco_offset, dtype=jnp.float32)
    null_logistic_fit_state = regenie2_binary_null_logistic.fit_null_logistic_coefficients(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        loco_offset=loco_offset_compute,
        kernel_config=kernel_config,
    )
    null_logistic_coefficients = null_logistic_fit_state.coefficients
    fitted_probability = regenie2_binary_logistic.compute_clipped_logistic_probability(
        covariate_matrix @ null_logistic_coefficients + loco_offset_compute,
        kernel_config,
    )
    bernoulli_weight = jnp.maximum(
        fitted_probability * (1.0 - fitted_probability),
        kernel_config.numerical.minimum_variance,
    )
    square_root_weight = jnp.sqrt(bernoulli_weight)
    weighted_covariate_matrix = square_root_weight[:, None] * covariate_matrix
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
    score_residual = phenotype_vector - fitted_probability
    return PreparedBinaryTraitState(
        phenotype_vector=phenotype_vector,
        null_logistic_coefficients=null_logistic_coefficients,
        score_residual=score_residual,
        loco_offset=loco_offset_compute,
        square_root_weight=square_root_weight,
        bernoulli_weight=bernoulli_weight,
        weighted_genotype_projection_matrix=weighted_genotype_projection_matrix,
        score_projection_matrix=weighted_genotype_projection_matrix * square_root_weight[None, :],
        null_logistic_converged=null_logistic_fit_state.converged,
    )


def build_multi_binary_score_chromosome_state_from_traits(
    trait_states: PreparedBinaryTraitState,
) -> Regenie2MultiBinaryScoreChromosomeState:
    """Assemble the minimal trait-major score state from prepared traits."""
    trait_count = trait_states.score_residual.shape[0]
    covariate_count = trait_states.score_projection_matrix.shape[1]
    sample_count = trait_states.score_residual.shape[1]
    flattened_projection_matrix = jnp.reshape(
        trait_states.score_projection_matrix,
        (trait_count * covariate_count, sample_count),
    )
    return Regenie2MultiBinaryScoreChromosomeState(
        score_right_hand_matrix=jnp.concatenate(
            [flattened_projection_matrix, trait_states.score_residual],
            axis=0,
        ),
        bernoulli_weight=trait_states.bernoulli_weight,
        null_logistic_converged=trait_states.null_logistic_converged,
    )


def prepare_binary_traits(
    state: Regenie2MultiBinaryState,
    loco_offset_matrix: jax.Array,
    kernel_config: regenie2_binary_config.BinaryScoreConfig,
) -> PreparedBinaryTraitState:
    """Prepare shared null-logistic quantities for every requested trait."""
    loco_offset_matrix_compute = jnp.asarray(loco_offset_matrix, dtype=jnp.float32)
    return jax.vmap(
        lambda phenotype_vector, loco_offset: prepare_binary_trait_state(
            state.covariate_matrix,
            phenotype_vector,
            loco_offset,
            kernel_config,
        )
    )(state.phenotype_matrix, loco_offset_matrix_compute)


@functools.partial(jax.jit, static_argnames=("kernel_config",))
def build_multi_binary_score_chromosome_state(
    state: Regenie2MultiBinaryState,
    loco_offset_matrix: jax.Array,
    kernel_config: regenie2_binary_config.BinaryScoreConfig,
) -> Regenie2MultiBinaryScoreChromosomeState:
    """Build chromosome state containing only binary score-kernel operands."""
    trait_states = prepare_binary_traits(state, loco_offset_matrix, kernel_config)
    return build_multi_binary_score_chromosome_state_from_traits(trait_states)


def compute_full_null_deviance(phenotype_vector: jax.Array, null_firth_offset: jax.Array) -> jax.Array:
    """Compute full-sample null deviance from a prepared null Firth offset."""
    scalar_offset_vector = jnp.asarray(null_firth_offset, dtype=jnp.float64)
    scalar_phenotype_vector = jnp.asarray(phenotype_vector, dtype=jnp.float64)
    null_probability_vector = regenie2_binary_logistic.compute_regenie_logistic_probability(scalar_offset_vector)
    return regenie2_binary_logistic.compute_logistic_deviance(
        scalar_phenotype_vector,
        null_probability_vector,
        jnp.ones_like(scalar_phenotype_vector, dtype=jnp.bool_),
    )


@functools.partial(jax.jit, static_argnames=("kernel_config",))
def build_multi_binary_firth_chromosome_state(
    state: Regenie2MultiBinaryState,
    loco_offset_matrix: jax.Array,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> Regenie2MultiBinaryFirthChromosomeState:
    """Build chromosome state with score operands and approximate-Firth null state."""
    from g.compute.regenie2_binary.firth import null as regenie2_binary_firth_null

    trait_states = prepare_binary_traits(state, loco_offset_matrix, kernel_config)

    def prepare_firth_trait(score_state: PreparedBinaryTraitState) -> PreparedBinaryFirthTraitState:
        null_firth_result = regenie2_binary_firth_null.fit_covariate_only_firth_null_model(
            covariate_matrix=state.covariate_matrix,
            phenotype_vector=score_state.phenotype_vector,
            loco_offset=score_state.loco_offset,
            initial_coefficients=score_state.null_logistic_coefficients,
            kernel_config=kernel_config,
        )
        null_firth_offset = state.covariate_matrix.astype(jnp.float64) @ null_firth_result.coefficients + jnp.asarray(
            score_state.loco_offset, dtype=jnp.float64
        )
        return PreparedBinaryFirthTraitState(
            score_state=score_state,
            null_firth_offset=null_firth_offset,
            full_null_deviance=compute_full_null_deviance(score_state.phenotype_vector, null_firth_offset),
            null_firth_penalized_log_likelihood=null_firth_result.penalized_log_likelihood,
        )

    firth_trait_states = jax.vmap(prepare_firth_trait)(trait_states)
    score_trait_states = firth_trait_states.score_state
    return Regenie2MultiBinaryFirthChromosomeState(
        score_state=build_multi_binary_score_chromosome_state_from_traits(score_trait_states),
        phenotype_matrix=state.phenotype_matrix,
        null_firth_offset_matrix=firth_trait_states.null_firth_offset,
        square_root_weight=score_trait_states.square_root_weight,
        weighted_genotype_projection_matrix=score_trait_states.weighted_genotype_projection_matrix,
        full_null_deviance=firth_trait_states.full_null_deviance,
        null_firth_penalized_log_likelihood=firth_trait_states.null_firth_penalized_log_likelihood,
    )
