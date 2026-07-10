"""Coarse native-to-JAX association backend."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from g import _core, types
from g.compute.regenie2_binary import api as regenie2_binary
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_binary import diagnostics as regenie2_binary_diagnostics
from g.compute.regenie2_binary import result as regenie2_binary_result
from g.compute.regenie2_binary import state as regenie2_binary_state
from g.compute.regenie2_linear import api as regenie2_linear
from g.compute.regenie2_linear import config as regenie2_linear_config
from g.compute.regenie2_linear import score as regenie2_linear_score
from g.compute.regenie2_linear import state as regenie2_linear_state

type HostIntegerArray = npt.NDArray[np.int32]
type HostBooleanArray = npt.NDArray[np.bool_]
type HostStatisticArray = npt.NDArray[np.float32] | npt.NDArray[np.float64]
type BinaryDeviceResult = (
    regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult | regenie2_binary_result.Regenie2MultiBinaryChunkResult
)
type GroupState = regenie2_linear_state.Regenie2MultiLinearState | regenie2_binary_state.Regenie2MultiBinaryState
type ChromosomeState = (
    regenie2_linear_state.Regenie2MultiLinearChromosomeState | regenie2_binary_state.Regenie2MultiBinaryChromosomeState
)


@dataclass(frozen=True)
class DeviceAssociationResult:
    """Opaque device-resident result returned to the native scheduler."""

    beta: jax.Array
    standard_error: jax.Array
    chi_squared: jax.Array
    log10_p_value: jax.Array
    extra_code: jax.Array | None
    binary_diagnostics: regenie2_binary_diagnostics.BinaryChunkDiagnostics | None


class JaxAssociationBackend:
    """Execute REGENIE association kernels behind four coarse operations."""

    def __init__(self, config: _core.engine.JaxBackendConfig) -> None:
        """Initialize the JAX kernel policy from native configuration.

        Args:
            config: Validated native JAX backend configuration.

        """
        self.association_mode = types.AssociationMode(config.association_mode)
        self.score_dtype = types.FloatingPointDtype(config.score_dtype)
        self.linear_config = regenie2_linear_config.LinearNumericalConfig(
            minimum_variance=config.linear_minimum_variance,
            relative_variance_tolerance=config.linear_relative_variance_tolerance,
        )
        self.binary_config = regenie2_binary_config.BinaryKernelConfig(
            numerical=regenie2_binary_config.BinaryNumericalConfig(
                minimum_probability=config.binary_minimum_probability,
                minimum_variance=config.binary_minimum_variance,
                relative_variance_tolerance=config.binary_relative_variance_tolerance,
            ),
            null_logistic=regenie2_binary_config.BinaryNullLogisticConfig(
                maximum_iterations=config.binary_null_maximum_iterations,
                coefficient_tolerance=config.binary_null_coefficient_tolerance,
            ),
            firth_candidate=regenie2_binary_config.FirthCandidateConfig(
                batch_size=config.firth_batch_size,
                candidate_capacity=config.firth_candidate_capacity,
            ),
            approximate_firth=regenie2_binary_config.ApproximateFirthConfig(
                maximum_iterations=config.firth_maximum_iterations,
                gradient_tolerance=config.firth_gradient_tolerance,
                coefficient_tolerance=config.firth_coefficient_tolerance,
                likelihood_tolerance=config.firth_likelihood_tolerance,
                maximum_step_size=config.firth_maximum_step_size,
                pseudo_maximum_iterations=config.firth_pseudo_maximum_iterations,
                pseudo_inner_maximum_iterations=config.firth_pseudo_inner_maximum_iterations,
                newton_raphson_zero_start_iterations=config.firth_newton_raphson_zero_start_iterations,
                line_search_maximum_attempts=config.firth_line_search_maximum_attempts,
                step_halving_maximum_attempts=config.firth_step_halving_maximum_attempts,
                initial_response_scale=config.firth_initial_response_scale,
                sparse_carrier_dosage_threshold=config.firth_sparse_carrier_dosage_threshold,
                step_halving_scale=config.firth_step_halving_scale,
                use_block_math=config.use_block_firth_math,
            ),
            null_firth=regenie2_binary_config.NullFirthConfig(
                maximum_iterations=config.null_firth_maximum_iterations,
                gradient_tolerance=config.null_firth_gradient_tolerance,
                maximum_step_size=config.null_firth_maximum_step_size,
                fallback_iteration_multiplier=config.null_firth_fallback_iteration_multiplier,
                fallback_step_divisor=config.null_firth_fallback_step_divisor,
                line_search_maximum_attempts=config.null_firth_line_search_maximum_attempts,
                step_halving_scale=config.null_firth_step_halving_scale,
            ),
        )
        self.binary_correction_plan = (
            types.BinaryCorrectionPlan(
                method=types.BinaryFallbackMethod(config.correction_method),
                p_threshold=config.correction_p_threshold,
                firth_se=config.firth_se,
            )
            if self.association_mode == types.AssociationMode.REGENIE2_BINARY
            else None
        )

    def prepare_group(
        self,
        group_input: _core.engine.JaxGroupInput,
    ) -> GroupState:
        """Prepare reusable device state for one aligned phenotype group.

        Args:
            group_input: Trait-major phenotypes and their shared covariate matrix.

        Returns:
            Opaque group state for subsequent chromosome preparation.

        """
        phenotype_matrix = typing.cast("jax.Array", jax.device_put(group_input.phenotype_matrix))
        covariate_matrix = typing.cast("jax.Array", jax.device_put(group_input.covariate_matrix))
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
        group_state: GroupState,
        chromosome_input: _core.engine.JaxChromosomeInput,
    ) -> _core.engine.JaxPreparedChromosome:
        """Prepare chromosome-specific state and null-model diagnostics.

        Args:
            group_state: Opaque state returned by :meth:`prepare_group`.
            chromosome_input: Chromosome label and trait-major LOCO predictions.

        Returns:
            Opaque chromosome state with host null-model diagnostics.

        Raises:
            TypeError: If the supplied state does not match the configured mode.

        """
        prediction_matrix = typing.cast("jax.Array", jax.device_put(chromosome_input.prediction_matrix))
        if self.association_mode == types.AssociationMode.REGENIE2_LINEAR:
            if not isinstance(group_state, regenie2_linear_state.Regenie2MultiLinearState):
                message = "Quantitative chromosome preparation requires quantitative group state."
                raise TypeError(message)
            chromosome_state = regenie2_linear_state.build_multi_linear_chromosome_state(
                state=group_state,
                loco_prediction_matrix=prediction_matrix,
                score_dtype=self.score_dtype,
            )
            jax.block_until_ready(chromosome_state.adjusted_residual_matrix)
            return _core.engine.JaxPreparedChromosome(state=chromosome_state, diagnostics=None)

        if not isinstance(group_state, regenie2_binary_state.Regenie2MultiBinaryState):
            message = "Binary chromosome preparation requires binary group state."
            raise TypeError(message)
        correction_plan = self.binary_correction_plan
        if correction_plan is None:
            message = "Binary correction plan was not initialized for binary association."
            raise RuntimeError(message)
        chromosome_state = regenie2_binary_state.build_multi_binary_chromosome_state(
            state=group_state,
            loco_offset_matrix=prediction_matrix,
            correction_plan=correction_plan,
            kernel_config=self.binary_config,
            score_dtype=self.score_dtype,
        )
        host_diagnostics = typing.cast(
            "dict[str, object]",
            jax.device_get(
                {
                    "logistic_converged": chromosome_state.null_logistic_converged,
                    "logistic_iteration_count": chromosome_state.null_logistic_iteration_count,
                    "firth_iteration_count": chromosome_state.null_firth_iteration_count,
                    "firth_convergence_reason_code": chromosome_state.null_firth_convergence_reason_code,
                }
            ),
        )
        diagnostics = _core.engine.JaxNullModelDiagnostics(
            logistic_converged=typing.cast("HostBooleanArray", host_diagnostics["logistic_converged"]),
            logistic_iteration_count=typing.cast(
                "HostIntegerArray",
                host_diagnostics["logistic_iteration_count"],
            ),
            firth_iteration_count=typing.cast(
                "HostIntegerArray",
                host_diagnostics["firth_iteration_count"],
            ),
            firth_convergence_reason_code=typing.cast(
                "HostIntegerArray",
                host_diagnostics["firth_convergence_reason_code"],
            ),
        )
        return _core.engine.JaxPreparedChromosome(state=chromosome_state, diagnostics=diagnostics)

    def compute_batch(
        self,
        chromosome_state: ChromosomeState,
        genotype_batch: _core.engine.JaxGenotypeBatch,
    ) -> DeviceAssociationResult:
        """Dispatch one variant-major dosage or packed8 batch to JAX.

        Args:
            chromosome_state: Opaque state returned by :meth:`prepare_chromosome`.
            genotype_batch: Genotypes and native per-variant compute statistics.

        Returns:
            Device-resident association result.

        Raises:
            ValueError: If the batch does not contain exactly one genotype representation.
            TypeError: If the supplied state does not match the configured mode.

        """
        dosage_matrix = genotype_batch.dosage_matrix
        packed8_probabilities = genotype_batch.packed8_probabilities
        if (dosage_matrix is None) == (packed8_probabilities is None):
            message = "A genotype batch must contain exactly one of dosage_matrix or packed8_probabilities."
            raise ValueError(message)

        dosage_sum = typing.cast("jax.Array", jax.device_put(genotype_batch.dosage_sum))
        observation_count = typing.cast("jax.Array", jax.device_put(genotype_batch.observation_count))
        imputed_dosage_square_sum = (
            None
            if genotype_batch.imputed_dosage_square_sum is None
            else typing.cast("jax.Array", jax.device_put(genotype_batch.imputed_dosage_square_sum))
        )
        rare_sparse_mask = (
            None
            if genotype_batch.rare_sparse_mask is None
            else typing.cast("jax.Array", jax.device_put(genotype_batch.rare_sparse_mask))
        )

        if self.association_mode == types.AssociationMode.REGENIE2_LINEAR:
            if not isinstance(chromosome_state, regenie2_linear_state.Regenie2MultiLinearChromosomeState):
                message = "Quantitative batch computation requires quantitative chromosome state."
                raise TypeError(message)
            if dosage_matrix is not None:
                genotype_device_array = typing.cast("jax.Array", jax.device_put(dosage_matrix))
                linear_result = regenie2_linear_score.compute_regenie2_linear_chunk_trait_major_variant_major(
                    chromosome_state=chromosome_state,
                    genotype_matrix_by_variant=genotype_device_array,
                    genotype_dosage_sum=dosage_sum,
                    genotype_observation_count=observation_count,
                    genotype_imputed_dosage_square_sum=imputed_dosage_square_sum,
                    score_dtype=self.score_dtype,
                    linear_minimum_variance=self.linear_config.minimum_variance,
                    linear_relative_variance_tolerance=self.linear_config.relative_variance_tolerance,
                )
            else:
                packed_device_array = typing.cast("jax.Array", jax.device_put(packed8_probabilities))
                linear_result = regenie2_linear.compute_multi_linear_chunk_packed8_donating_inputs(
                    chromosome_state=chromosome_state,
                    packed_probability_pairs_by_variant=packed_device_array,
                    genotype_dosage_sum=dosage_sum,
                    genotype_observation_count=observation_count,
                    genotype_imputed_dosage_square_sum=imputed_dosage_square_sum,
                    score_dtype=self.score_dtype,
                    linear_minimum_variance=self.linear_config.minimum_variance,
                    linear_relative_variance_tolerance=self.linear_config.relative_variance_tolerance,
                )
            return DeviceAssociationResult(
                beta=linear_result.beta,
                standard_error=linear_result.standard_error,
                chi_squared=linear_result.chi_squared,
                log10_p_value=linear_result.log10_p_value,
                extra_code=None,
                binary_diagnostics=None,
            )

        if not isinstance(chromosome_state, regenie2_binary_state.Regenie2MultiBinaryChromosomeState):
            message = "Binary batch computation requires binary chromosome state."
            raise TypeError(message)
        correction_plan = self.binary_correction_plan
        if correction_plan is None:
            message = "Binary correction plan was not initialized for binary association."
            raise RuntimeError(message)
        if dosage_matrix is not None:
            genotype_device_array = typing.cast("jax.Array", jax.device_put(dosage_matrix))
            if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
                binary_result: BinaryDeviceResult = (
                    regenie2_binary.compute_multi_binary_score_test_variant_major_donating_inputs(
                        chromosome_state=chromosome_state,
                        genotype_matrix_by_variant=genotype_device_array,
                        correction_plan=correction_plan,
                        kernel_config=self.binary_config,
                        dosage_sum=dosage_sum,
                        observation_count=observation_count,
                        score_dtype=self.score_dtype,
                    )
                )
            else:
                binary_result = regenie2_binary.compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
                    chromosome_state=chromosome_state,
                    genotype_matrix_by_variant=genotype_device_array,
                    correction_plan=correction_plan,
                    kernel_config=self.binary_config,
                    sparse_candidate_mask=rare_sparse_mask,
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                    score_dtype=self.score_dtype,
                    stage_duration_recorder=None,
                )
        else:
            packed_device_array = typing.cast("jax.Array", jax.device_put(packed8_probabilities))
            if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
                binary_result = regenie2_binary.compute_multi_binary_score_test_packed8_donating_inputs(
                    chromosome_state=chromosome_state,
                    packed_probability_pairs_by_variant=packed_device_array,
                    correction_plan=correction_plan,
                    kernel_config=self.binary_config,
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                    score_dtype=self.score_dtype,
                )
            else:
                binary_result = regenie2_binary.compute_regenie2_multi_binary_chunk_from_chromosome_state_packed8(
                    chromosome_state=chromosome_state,
                    packed_probability_pairs_by_variant=packed_device_array,
                    correction_plan=correction_plan,
                    kernel_config=self.binary_config,
                    sparse_candidate_mask=rare_sparse_mask,
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                    score_dtype=self.score_dtype,
                    stage_duration_recorder=None,
                )
        return DeviceAssociationResult(
            beta=binary_result.beta,
            standard_error=binary_result.standard_error,
            chi_squared=binary_result.chi_squared,
            log10_p_value=binary_result.log10_p_value,
            extra_code=binary_result.extra_code,
            binary_diagnostics=regenie2_binary_diagnostics.count_binary_chunk_diagnostics(binary_result),
        )

    def materialize_batch(
        self,
        device_result: DeviceAssociationResult,
        request: _core.engine.JaxMaterializationRequest,
    ) -> _core.engine.JaxHostAssociationBatch:
        """Select active traits and transfer one association result to the host.

        Args:
            device_result: Opaque device result returned by :meth:`compute_batch`.
            request: Active trait rows and native output statistic dtype.

        Returns:
            Trait-major host arrays and aggregate binary diagnostics.

        """
        active_trait_indices = tuple(request.active_trait_indices)
        total_trait_count = device_result.beta.shape[0]
        if active_trait_indices == tuple(range(total_trait_count)):
            beta = device_result.beta
            standard_error = device_result.standard_error
            chi_squared = device_result.chi_squared
            log10_p_value = device_result.log10_p_value
            extra_code = device_result.extra_code
        else:
            active_trait_index_array = jnp.asarray(active_trait_indices, dtype=jnp.int32)
            beta = jnp.take(device_result.beta, active_trait_index_array, axis=0)
            standard_error = jnp.take(device_result.standard_error, active_trait_index_array, axis=0)
            chi_squared = jnp.take(device_result.chi_squared, active_trait_index_array, axis=0)
            log10_p_value = jnp.take(device_result.log10_p_value, active_trait_index_array, axis=0)
            extra_code = (
                None
                if device_result.extra_code is None
                else jnp.take(device_result.extra_code, active_trait_index_array, axis=0)
            )

        output_statistic_dtype = types.FloatingPointDtype(request.output_statistic_dtype)
        statistic_jax_dtype = jnp.float32 if output_statistic_dtype == types.FloatingPointDtype.FLOAT32 else jnp.float64
        transfer_payload: dict[str, object] = {
            "beta": jnp.asarray(beta, dtype=statistic_jax_dtype),
            "standard_error": jnp.asarray(standard_error, dtype=statistic_jax_dtype),
            "chi_squared": jnp.asarray(chi_squared, dtype=statistic_jax_dtype),
            "log10_p_value": jnp.asarray(log10_p_value, dtype=statistic_jax_dtype),
            "extra_code": None if extra_code is None else jnp.asarray(extra_code, dtype=jnp.int32),
            "binary_diagnostics": device_result.binary_diagnostics,
        }
        host_payload = typing.cast("dict[str, object]", jax.device_get(transfer_payload))
        binary_diagnostics = host_payload["binary_diagnostics"]
        host_binary_diagnostics = None
        if binary_diagnostics is not None:
            diagnostics = typing.cast("regenie2_binary_diagnostics.BinaryChunkDiagnostics", binary_diagnostics)
            host_binary_diagnostics = _core.engine.JaxBinaryDiagnostics(
                score_only_count=int(diagnostics.score_only_count),
                score_test_candidate_count=int(diagnostics.score_test_candidate_count),
                firth_candidate_count=int(diagnostics.firth_candidate_count),
                firth_iteration_min=int(diagnostics.firth_iteration_min),
                firth_iteration_median=float(diagnostics.firth_iteration_median),
                firth_iteration_max=int(diagnostics.firth_iteration_max),
                firth_converged_count=int(diagnostics.firth_converged_count),
                firth_failed_count=int(diagnostics.firth_failed_count),
                firth_numerical_failure_count=int(diagnostics.firth_numerical_failure_count),
                firth_max_iteration_failure_count=int(diagnostics.firth_max_iteration_failure_count),
                firth_invalid_statistic_failure_count=int(diagnostics.firth_invalid_statistic_failure_count),
                firth_step_halving_failure_count=int(diagnostics.firth_step_halving_failure_count),
                pseudo_firth_attempt_count=int(diagnostics.pseudo_firth_attempt_count),
                pseudo_firth_success_count=int(diagnostics.pseudo_firth_success_count),
                newton_raphson_zero_start_attempt_count=int(diagnostics.nr_zero_start_attempt_count),
                newton_raphson_zero_start_success_count=int(diagnostics.nr_zero_start_success_count),
                newton_raphson_warm_start_attempt_count=int(diagnostics.nr_warm_start_attempt_count),
                newton_raphson_warm_start_success_count=int(diagnostics.nr_warm_start_success_count),
                sparse_correction_count=int(diagnostics.sparse_correction_count),
                dense_correction_count=int(diagnostics.dense_correction_count),
            )
        return _core.engine.JaxHostAssociationBatch(
            beta=typing.cast("HostStatisticArray", host_payload["beta"]),
            standard_error=typing.cast("HostStatisticArray", host_payload["standard_error"]),
            chi_squared=typing.cast("HostStatisticArray", host_payload["chi_squared"]),
            log10_p_value=typing.cast("HostStatisticArray", host_payload["log10_p_value"]),
            extra_code=typing.cast("HostIntegerArray | None", host_payload["extra_code"]),
            binary_diagnostics=host_binary_diagnostics,
        )
