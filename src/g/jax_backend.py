"""Coarse native-to-JAX association backend."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

from g import _core, types
from g.compute.regenie2_binary import api as regenie2_binary
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_linear import api as regenie2_linear
from g.compute.regenie2_linear import config as regenie2_linear_config
from g.compute.regenie2_linear import score as regenie2_linear_score
from g.compute.regenie2_linear import state as regenie2_linear_state


@dataclass(frozen=True)
class DeviceAssociationResult:
    """Opaque device-resident result returned to the native scheduler."""

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    correction_code: jax.Array | None


@dataclass(frozen=True)
class HostAssociationResult:
    """Trait-major host result consumed by the native engine."""

    beta: npt.NDArray[np.float32]
    standard_error: npt.NDArray[np.float32]
    chi_squared: npt.NDArray[np.float32]
    log10_p_value: npt.NDArray[np.float32]
    correction_code: npt.NDArray[np.int32] | None


@dataclass(frozen=True)
class DeviceGenotypeBatch:
    """One native genotype batch transferred to the JAX device."""

    dosage_matrix: jax.Array | None
    packed8_probabilities: jax.Array | None
    dosage_sum: jax.Array
    observation_count: jax.Array
    imputed_dosage_square_sum: jax.Array | None
    rare_sparse_mask: jax.Array | None


def prepare_device_genotype_batch(genotype_batch: _core.engine.JaxGenotypeBatch) -> DeviceGenotypeBatch:
    """Validate and transfer native genotype inputs for one compute batch.

    Args:
        genotype_batch: Genotypes and per-variant statistics owned by the native scheduler.

    Raises:
        ValueError: If the batch does not contain exactly one genotype representation.

    """
    dosage_matrix = genotype_batch.dosage_matrix
    packed8_probabilities = genotype_batch.packed8_probabilities
    if (dosage_matrix is None) == (packed8_probabilities is None):
        message = "A genotype batch must contain exactly one of dosage_matrix or packed8_probabilities."
        raise ValueError(message)
    return DeviceGenotypeBatch(
        dosage_matrix=None if dosage_matrix is None else jax.device_put(dosage_matrix),
        packed8_probabilities=(None if packed8_probabilities is None else jax.device_put(packed8_probabilities)),
        dosage_sum=jax.device_put(genotype_batch.dosage_sum),
        observation_count=jax.device_put(genotype_batch.observation_count),
        imputed_dosage_square_sum=(
            None
            if genotype_batch.imputed_dosage_square_sum is None
            else jax.device_put(genotype_batch.imputed_dosage_square_sum)
        ),
        rare_sparse_mask=(
            None if genotype_batch.rare_sparse_mask is None else jax.device_put(genotype_batch.rare_sparse_mask)
        ),
    )


def compute_linear_device_batch(
    chromosome_state: regenie2_linear_state.Regenie2MultiLinearChromosomeState,
    device_genotype_batch: DeviceGenotypeBatch,
    score_dtype: types.FloatingPointDtype,
    linear_config: regenie2_linear_config.LinearNumericalConfig,
) -> DeviceAssociationResult:
    """Run one linear association batch with already-transferred inputs."""
    if device_genotype_batch.dosage_matrix is not None:
        linear_result = regenie2_linear_score.compute_regenie2_linear_chunk_trait_major_variant_major(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=device_genotype_batch.dosage_matrix,
            genotype_dosage_sum=device_genotype_batch.dosage_sum,
            genotype_observation_count=device_genotype_batch.observation_count,
            genotype_imputed_dosage_square_sum=device_genotype_batch.imputed_dosage_square_sum,
            score_dtype=score_dtype,
            linear_minimum_variance=linear_config.minimum_variance,
            linear_relative_variance_tolerance=linear_config.relative_variance_tolerance,
        )
    else:
        assert device_genotype_batch.packed8_probabilities is not None
        linear_result = regenie2_linear.compute_multi_linear_chunk_packed8_donating_inputs(
            chromosome_state=chromosome_state,
            packed_probability_pairs_by_variant=device_genotype_batch.packed8_probabilities,
            genotype_dosage_sum=device_genotype_batch.dosage_sum,
            genotype_observation_count=device_genotype_batch.observation_count,
            genotype_imputed_dosage_square_sum=device_genotype_batch.imputed_dosage_square_sum,
            score_dtype=score_dtype,
            linear_minimum_variance=linear_config.minimum_variance,
            linear_relative_variance_tolerance=linear_config.relative_variance_tolerance,
        )
    return DeviceAssociationResult(
        beta=linear_result.beta,
        standard_error=linear_result.standard_error,
        chi_squared=linear_result.chi_squared,
        log10_p_value=linear_result.log10_p_value,
        correction_code=None,
    )


def compute_binary_variant_major_device_batch(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    device_genotype_batch: DeviceGenotypeBatch,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    score_dtype: types.FloatingPointDtype,
) -> DeviceAssociationResult:
    """Run one variant-major binary association batch with device-resident inputs."""
    assert device_genotype_batch.dosage_matrix is not None
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        binary_result = regenie2_binary.compute_multi_binary_score_test_variant_major_donating_inputs(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=device_genotype_batch.dosage_matrix,
            correction_plan=correction_plan,
            kernel_config=kernel_config,
            dosage_sum=device_genotype_batch.dosage_sum,
            observation_count=device_genotype_batch.observation_count,
            score_dtype=score_dtype,
        )
    else:
        binary_result = regenie2_binary.compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=device_genotype_batch.dosage_matrix,
            correction_plan=correction_plan,
            kernel_config=kernel_config,
            sparse_candidate_mask=device_genotype_batch.rare_sparse_mask,
            dosage_sum=device_genotype_batch.dosage_sum,
            observation_count=device_genotype_batch.observation_count,
            score_dtype=score_dtype,
        )
    return DeviceAssociationResult(
        beta=binary_result.beta,
        standard_error=binary_result.standard_error,
        chi_squared=binary_result.chi_squared,
        log10_p_value=binary_result.log10_p_value,
        correction_code=binary_result.correction_code,
    )


def compute_binary_packed8_device_batch(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    device_genotype_batch: DeviceGenotypeBatch,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    score_dtype: types.FloatingPointDtype,
) -> DeviceAssociationResult:
    """Run one packed8 binary association batch with device-resident inputs."""
    assert device_genotype_batch.packed8_probabilities is not None
    if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
        binary_result = regenie2_binary.compute_multi_binary_score_test_packed8_donating_inputs(
            chromosome_state=chromosome_state,
            packed_probability_pairs_by_variant=device_genotype_batch.packed8_probabilities,
            correction_plan=correction_plan,
            kernel_config=kernel_config,
            dosage_sum=device_genotype_batch.dosage_sum,
            observation_count=device_genotype_batch.observation_count,
            score_dtype=score_dtype,
        )
    else:
        binary_result = regenie2_binary.compute_regenie2_multi_binary_chunk_from_chromosome_state_packed8(
            chromosome_state=chromosome_state,
            packed_probability_pairs_by_variant=device_genotype_batch.packed8_probabilities,
            correction_plan=correction_plan,
            kernel_config=kernel_config,
            sparse_candidate_mask=device_genotype_batch.rare_sparse_mask,
            dosage_sum=device_genotype_batch.dosage_sum,
            observation_count=device_genotype_batch.observation_count,
            score_dtype=score_dtype,
        )
    return DeviceAssociationResult(
        beta=binary_result.beta,
        standard_error=binary_result.standard_error,
        chi_squared=binary_result.chi_squared,
        log10_p_value=binary_result.log10_p_value,
        correction_code=binary_result.correction_code,
    )


def compute_binary_device_batch(
    chromosome_state: regenie2_binary_state.Regenie2MultiBinaryChromosomeState,
    device_genotype_batch: DeviceGenotypeBatch,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    score_dtype: types.FloatingPointDtype,
) -> DeviceAssociationResult:
    """Dispatch one binary association batch by its genotype representation."""
    if device_genotype_batch.dosage_matrix is not None:
        return compute_binary_variant_major_device_batch(
            chromosome_state,
            device_genotype_batch,
            correction_plan,
            kernel_config,
            score_dtype,
        )
    return compute_binary_packed8_device_batch(
        chromosome_state,
        device_genotype_batch,
        correction_plan,
        kernel_config,
        score_dtype,
    )


class JaxAssociationBackend:
    """Execute REGENIE association kernels behind four coarse operations."""

    def __init__(self, config: _core.engine.JaxBackendConfig) -> None:
        """Initialize the JAX kernel policy from native configuration.

        Args:
            config: Validated native JAX backend configuration.

        """
        self.association_mode = types.AssociationMode(config.association_mode)
        self.score_dtype = types.FloatingPointDtype.FLOAT32
        correction_config = config.correction
        linear_config = config.linear
        binary_config = config.binary
        binary_numerical_config = binary_config.numerical
        binary_null_logistic_config = binary_config.null_logistic
        firth_candidate_config = binary_config.firth_candidate
        approximate_firth_config = binary_config.approximate_firth
        null_firth_config = binary_config.null_firth
        self.linear_config = regenie2_linear_config.LinearNumericalConfig(
            minimum_variance=linear_config.minimum_variance,
            relative_variance_tolerance=linear_config.relative_variance_tolerance,
        )
        self.binary_config = regenie2_binary_config.BinaryKernelConfig(
            numerical=regenie2_binary_config.BinaryNumericalConfig(
                minimum_probability=binary_numerical_config.minimum_probability,
                minimum_variance=binary_numerical_config.minimum_variance,
                relative_variance_tolerance=binary_numerical_config.relative_variance_tolerance,
            ),
            null_logistic=regenie2_binary_config.BinaryNullLogisticConfig(
                maximum_iterations=binary_null_logistic_config.maximum_iterations,
                coefficient_tolerance=binary_null_logistic_config.coefficient_tolerance,
            ),
            firth_candidate=regenie2_binary_config.FirthCandidateConfig(
                batch_size=firth_candidate_config.batch_size,
                candidate_capacity=firth_candidate_config.candidate_capacity,
            ),
            approximate_firth=regenie2_binary_config.ApproximateFirthConfig(
                maximum_iterations=approximate_firth_config.maximum_iterations,
                gradient_tolerance=approximate_firth_config.gradient_tolerance,
                coefficient_tolerance=approximate_firth_config.coefficient_tolerance,
                likelihood_tolerance=approximate_firth_config.likelihood_tolerance,
                maximum_step_size=approximate_firth_config.maximum_step_size,
                pseudo_maximum_iterations=approximate_firth_config.pseudo_maximum_iterations,
                pseudo_inner_maximum_iterations=approximate_firth_config.pseudo_inner_maximum_iterations,
                newton_raphson_zero_start_iterations=approximate_firth_config.newton_raphson_zero_start_iterations,
                line_search_maximum_attempts=approximate_firth_config.line_search_maximum_attempts,
                step_halving_maximum_attempts=approximate_firth_config.step_halving_maximum_attempts,
                initial_response_scale=approximate_firth_config.initial_response_scale,
                sparse_carrier_dosage_threshold=approximate_firth_config.sparse_carrier_dosage_threshold,
                step_halving_scale=approximate_firth_config.step_halving_scale,
                use_block_math=approximate_firth_config.use_block_math,
            ),
            null_firth=regenie2_binary_config.NullFirthConfig(
                maximum_iterations=null_firth_config.maximum_iterations,
                gradient_tolerance=null_firth_config.gradient_tolerance,
                maximum_step_size=null_firth_config.maximum_step_size,
                fallback_iteration_multiplier=null_firth_config.fallback_iteration_multiplier,
                fallback_step_divisor=null_firth_config.fallback_step_divisor,
                line_search_maximum_attempts=null_firth_config.line_search_maximum_attempts,
                step_halving_scale=null_firth_config.step_halving_scale,
            ),
        )
        self.binary_correction_plan = (
            types.BinaryCorrectionPlan(
                method=types.BinaryFallbackMethod(correction_config.method),
                p_threshold=correction_config.p_threshold,
                firth_se=correction_config.firth_se,
            )
            if self.association_mode == types.AssociationMode.REGENIE2_BINARY
            else None
        )

    def prepare_group(
        self,
        group_input: _core.engine.JaxGroupInput,
    ) -> regenie2_linear_state.Regenie2MultiLinearState | regenie2_binary_state.Regenie2MultiBinaryState:
        """Prepare reusable device state for one aligned phenotype group.

        Args:
            group_input: Trait-major phenotypes and their shared covariate matrix.

        Returns:
            Opaque group state for subsequent chromosome preparation.

        """
        phenotype_matrix = jax.device_put(group_input.phenotype_matrix)
        covariate_matrix = jax.device_put(group_input.covariate_matrix)
        if self.association_mode == types.AssociationMode.REGENIE2_LINEAR:
            return regenie2_linear_state.build_multi_linear_state(
                covariate_matrix=covariate_matrix,
                phenotype_matrix=phenotype_matrix,
                score_dtype=self.score_dtype,
            )
        return regenie2_binary_state.build_multi_binary_state(
            covariate_matrix=covariate_matrix,
            phenotype_matrix=phenotype_matrix,
            score_dtype=self.score_dtype,
        )

    def prepare_chromosome(
        self,
        group_state: regenie2_linear_state.Regenie2MultiLinearState | regenie2_binary_state.Regenie2MultiBinaryState,
        prediction_matrix: npt.NDArray[np.float32],
    ) -> _core.engine.JaxPreparedChromosome:
        """Prepare chromosome-specific state and convergence policy input.

        Args:
            group_state: Opaque state returned by :meth:`prepare_group`.
            prediction_matrix: Trait-major LOCO predictions.

        Returns:
            Opaque chromosome state with host null-logistic convergence flags.

        Raises:
            TypeError: If the supplied state does not match the configured mode.

        """
        prediction_matrix_device = jax.device_put(prediction_matrix)
        if self.association_mode == types.AssociationMode.REGENIE2_LINEAR:
            if not isinstance(group_state, regenie2_linear_state.Regenie2MultiLinearState):
                message = "Quantitative chromosome preparation requires quantitative group state."
                raise TypeError(message)
            chromosome_state = regenie2_linear_state.build_multi_linear_chromosome_state(
                state=group_state,
                loco_prediction_matrix=prediction_matrix_device,
                score_dtype=self.score_dtype,
            )
            jax.block_until_ready(chromosome_state.adjusted_residual_matrix)
            return _core.engine.JaxPreparedChromosome(state=chromosome_state, null_logistic_converged=None)

        if not isinstance(group_state, regenie2_binary_state.Regenie2MultiBinaryState):
            message = "Binary chromosome preparation requires binary group state."
            raise TypeError(message)
        correction_plan = self.binary_correction_plan
        if correction_plan is None:
            message = "Binary correction plan was not initialized for binary association."
            raise RuntimeError(message)
        chromosome_state = regenie2_binary_state.build_multi_binary_chromosome_state(
            state=group_state,
            loco_offset_matrix=prediction_matrix_device,
            correction_plan=correction_plan,
            kernel_config=self.binary_config,
            score_dtype=self.score_dtype,
        )
        null_logistic_converged = jax.device_get(chromosome_state.null_logistic_converged)
        return _core.engine.JaxPreparedChromosome(
            state=chromosome_state,
            null_logistic_converged=null_logistic_converged,
        )

    def compute_batch(
        self,
        chromosome_state: (
            regenie2_linear_state.Regenie2MultiLinearChromosomeState
            | regenie2_binary_state.Regenie2MultiBinaryChromosomeState
        ),
        genotype_batch: _core.engine.JaxGenotypeBatch,
    ) -> DeviceAssociationResult:
        """Dispatch one validated native batch to the configured JAX kernel."""
        device_genotype_batch = prepare_device_genotype_batch(genotype_batch)

        if self.association_mode == types.AssociationMode.REGENIE2_LINEAR:
            if not isinstance(chromosome_state, regenie2_linear_state.Regenie2MultiLinearChromosomeState):
                message = "Quantitative batch computation requires quantitative chromosome state."
                raise TypeError(message)
            return compute_linear_device_batch(
                chromosome_state,
                device_genotype_batch,
                self.score_dtype,
                self.linear_config,
            )

        if not isinstance(chromosome_state, regenie2_binary_state.Regenie2MultiBinaryChromosomeState):
            message = "Binary batch computation requires binary chromosome state."
            raise TypeError(message)
        correction_plan = self.binary_correction_plan
        if correction_plan is None:
            message = "Binary correction plan was not initialized for binary association."
            raise RuntimeError(message)
        return compute_binary_device_batch(
            chromosome_state,
            device_genotype_batch,
            correction_plan,
            self.binary_config,
            self.score_dtype,
        )

    def materialize_batch(
        self,
        device_result: DeviceAssociationResult,
        request: _core.engine.JaxMaterializationRequest,
    ) -> HostAssociationResult:
        """Select active traits and transfer one association result to the host.

        Args:
            device_result: Opaque device result returned by :meth:`compute_batch`.
            request: Active trait rows and native output statistic dtype.

        Returns:
            Trait-major host arrays and correction codes.

        """
        active_trait_indices = tuple(request.active_trait_indices)
        total_trait_count = device_result.beta.shape[0]
        if active_trait_indices == tuple(range(total_trait_count)):
            beta = device_result.beta
            standard_error = device_result.standard_error
            chi_squared = device_result.chi_squared
            log10_p_value = device_result.log10_p_value
            correction_code = device_result.correction_code
        else:
            active_trait_index_array = jnp.asarray(active_trait_indices, dtype=jnp.int32)
            beta = jnp.take(device_result.beta, active_trait_index_array, axis=0)
            standard_error = jnp.take(device_result.standard_error, active_trait_index_array, axis=0)
            chi_squared = jnp.take(device_result.chi_squared, active_trait_index_array, axis=0)
            log10_p_value = jnp.take(device_result.log10_p_value, active_trait_index_array, axis=0)
            correction_code = (
                None
                if device_result.correction_code is None
                else jnp.take(device_result.correction_code, active_trait_index_array, axis=0)
            )

        transfer_payload: dict[str, object] = {
            "beta": jnp.asarray(beta, dtype=jnp.float32),
            "standard_error": jnp.asarray(standard_error, dtype=jnp.float32),
            "chi_squared": jnp.asarray(chi_squared, dtype=jnp.float32),
            "log10_p_value": jnp.asarray(log10_p_value, dtype=jnp.float32),
            "correction_code": (None if correction_code is None else jnp.asarray(correction_code, dtype=jnp.int32)),
        }
        host_payload = jax.device_get(transfer_payload)
        return HostAssociationResult(
            beta=host_payload["beta"],
            standard_error=host_payload["standard_error"],
            chi_squared=host_payload["chi_squared"],
            log10_p_value=host_payload["log10_p_value"],
            correction_code=host_payload["correction_code"],
        )
