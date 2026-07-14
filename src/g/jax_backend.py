"""Coarse native-to-JAX association backend."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from g import types
from g.compute.common import result as association_result

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import state as regenie2_binary_state
    from g.compute.regenie2_linear import state as regenie2_linear_state

type DeviceAssociationResult = (
    association_result.AssociationResult[jax.Array, jax.Array] | association_result.AssociationResult[jax.Array, None]
)
type HostAssociationResult = (
    association_result.AssociationResult[npt.NDArray[np.float32], npt.NDArray[np.uint8]]
    | association_result.AssociationResult[npt.NDArray[np.float32], None]
)


class JaxBackendBase:
    """Shared host materialization for concrete association backends."""

    def materialize_batch(
        self,
        device_result: DeviceAssociationResult,
        active_trait_indices: npt.NDArray[np.int32] | None,
        logical_variant_count: int,
    ) -> HostAssociationResult:
        """Select active traits on device and compact a partial tail on the host."""
        if active_trait_indices is None:
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

        host_result = jax.device_get(
            association_result.AssociationResult(
                beta=jnp.asarray(beta, dtype=jnp.float32),
                standard_error=jnp.asarray(standard_error, dtype=jnp.float32),
                chi_squared=jnp.asarray(chi_squared, dtype=jnp.float32),
                log10_p_value=jnp.asarray(log10_p_value, dtype=jnp.float32),
                correction_code=(None if correction_code is None else jnp.asarray(correction_code, dtype=jnp.uint8)),
            )
        )
        if logical_variant_count == host_result.beta.shape[1]:
            return host_result
        return association_result.AssociationResult(
            beta=host_result.beta[:, :logical_variant_count],
            standard_error=host_result.standard_error[:, :logical_variant_count],
            chi_squared=host_result.chi_squared[:, :logical_variant_count],
            log10_p_value=host_result.log10_p_value[:, :logical_variant_count],
            correction_code=(
                None if host_result.correction_code is None else host_result.correction_code[:, :logical_variant_count]
            ),
        )


class LinearJaxBackend(JaxBackendBase):
    """Execute linear REGENIE kernels without runtime mode dispatch."""

    def __init__(
        self,
        *,
        minimum_variance: float,
        relative_variance_tolerance: float,
    ) -> None:
        """Initialize the linear numerical policy."""
        from g.compute.regenie2_linear import score as regenie2_linear_score
        from g.compute.regenie2_linear import state as regenie2_linear_state

        self.minimum_variance = minimum_variance
        self.relative_variance_tolerance = relative_variance_tolerance
        self._linear_score = regenie2_linear_score
        self._linear_state = regenie2_linear_state

    def prepare_group(
        self,
        phenotype_matrix: npt.NDArray[np.float32],
        covariate_matrix: npt.NDArray[np.float32],
    ) -> regenie2_linear_state.Regenie2MultiLinearState:
        """Prepare reusable device state for one aligned phenotype group."""
        group_state = self._linear_state.build_multi_linear_state(
            covariate_matrix=jax.device_put(covariate_matrix),
            phenotype_matrix=jax.device_put(phenotype_matrix),
        )
        jax.block_until_ready(group_state.phenotype_residual_matrix)
        return group_state

    def prepare_chromosome(
        self,
        group_state: regenie2_linear_state.Regenie2MultiLinearState,
        prediction_matrix: npt.NDArray[np.float32],
    ) -> regenie2_linear_state.Regenie2MultiLinearChromosomeState:
        """Prepare reusable device state for one chromosome."""
        chromosome_state = self._linear_state.build_multi_linear_chromosome_state(
            state=group_state,
            loco_prediction_matrix=jax.device_put(prediction_matrix),
        )
        jax.block_until_ready(chromosome_state.score_left_hand_matrix)
        return chromosome_state

    def compute_dosage_batch(
        self,
        chromosome_state: regenie2_linear_state.Regenie2MultiLinearChromosomeState,
        dosage_matrix: npt.NDArray[np.float32],
        genotype_mean: npt.NDArray[np.float32],
        imputed_dosage_square_sum: npt.NDArray[np.float32],
    ) -> DeviceAssociationResult:
        """Transfer and submit one dosage batch to the linear kernel."""
        return self._linear_score.compute_regenie2_linear_chunk_trait_major_variant_major_donating_inputs(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=jax.device_put(dosage_matrix),
            native_genotype_mean=jax.device_put(genotype_mean),
            genotype_imputed_dosage_square_sum=jax.device_put(imputed_dosage_square_sum),
            linear_minimum_variance=self.minimum_variance,
            linear_relative_variance_tolerance=self.relative_variance_tolerance,
        )

    def compute_packed8_batch(
        self,
        chromosome_state: regenie2_linear_state.Regenie2MultiLinearChromosomeState,
        packed8_probabilities: npt.NDArray[np.uint8],
        genotype_mean: npt.NDArray[np.float32],
        imputed_dosage_square_sum: npt.NDArray[np.float32],
    ) -> DeviceAssociationResult:
        """Transfer and submit one packed8 batch to the linear kernel."""
        return self._linear_score.compute_multi_linear_chunk_packed8_donating_inputs(
            chromosome_state=chromosome_state,
            packed_probability_pairs_by_variant=jax.device_put(packed8_probabilities, may_alias=False),
            native_genotype_mean=jax.device_put(genotype_mean),
            genotype_imputed_dosage_square_sum=jax.device_put(imputed_dosage_square_sum),
            linear_minimum_variance=self.minimum_variance,
            linear_relative_variance_tolerance=self.relative_variance_tolerance,
        )


class BinaryJaxBackendBase(JaxBackendBase):
    """Shared binary score configuration and group-state preparation."""

    def __init__(
        self,
        *,
        minimum_probability: float,
        minimum_variance: float,
        relative_variance_tolerance: float,
        null_logistic_maximum_iterations: int,
        null_logistic_coefficient_tolerance: float,
    ) -> None:
        """Initialize policy required by every binary score kernel."""
        from g.compute.regenie2_binary import config as regenie2_binary_config
        from g.compute.regenie2_binary import score as regenie2_binary_score
        from g.compute.regenie2_binary import state as regenie2_binary_state

        self._binary_score = regenie2_binary_score
        self._binary_state = regenie2_binary_state
        self.score_config = regenie2_binary_config.BinaryScoreConfig(
            numerical=regenie2_binary_config.BinaryNumericalConfig(
                minimum_probability=minimum_probability,
                minimum_variance=minimum_variance,
                relative_variance_tolerance=relative_variance_tolerance,
            ),
            null_logistic=regenie2_binary_config.BinaryNullLogisticConfig(
                maximum_iterations=null_logistic_maximum_iterations,
                coefficient_tolerance=null_logistic_coefficient_tolerance,
            ),
        )

    def prepare_group(
        self,
        phenotype_matrix: npt.NDArray[np.float32],
        covariate_matrix: npt.NDArray[np.float32],
    ) -> regenie2_binary_state.Regenie2MultiBinaryState:
        """Prepare reusable device state for one aligned phenotype group."""
        return self._binary_state.build_multi_binary_state(
            covariate_matrix=jax.device_put(covariate_matrix),
            phenotype_matrix=jax.device_put(phenotype_matrix),
        )


class BinaryScoreJaxBackend(BinaryJaxBackendBase):
    """Execute binary score kernels without correction dispatch."""

    def prepare_chromosome(
        self,
        group_state: regenie2_binary_state.Regenie2MultiBinaryState,
        prediction_matrix: npt.NDArray[np.float32],
    ) -> regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState:
        """Prepare only the chromosome operands consumed by score kernels."""
        return self._binary_state.build_multi_binary_score_chromosome_state(
            state=group_state,
            loco_offset_matrix=jax.device_put(prediction_matrix),
            kernel_config=self.score_config,
        )

    def compute_dosage_batch(
        self,
        chromosome_state: regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState,
        dosage_matrix: npt.NDArray[np.float32],
        genotype_mean: npt.NDArray[np.float32],
    ) -> DeviceAssociationResult:
        """Transfer and submit one dosage batch to the binary score kernel."""
        return self._binary_score.compute_multi_binary_score_test_variant_major_donating_inputs(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=jax.device_put(dosage_matrix),
            firth_candidate_p_threshold=None,
            minimum_variance=self.score_config.numerical.minimum_variance,
            relative_variance_tolerance=self.score_config.numerical.relative_variance_tolerance,
            native_genotype_mean=jax.device_put(genotype_mean),
        )

    def compute_packed8_batch(
        self,
        chromosome_state: regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState,
        packed8_probabilities: npt.NDArray[np.uint8],
        genotype_mean: npt.NDArray[np.float32],
    ) -> DeviceAssociationResult:
        """Transfer and submit one packed8 batch to the binary score kernel."""
        return self._binary_score.compute_multi_binary_score_test_packed8_donating_inputs(
            chromosome_state=chromosome_state,
            packed_probability_pairs_by_variant=jax.device_put(packed8_probabilities, may_alias=False),
            firth_candidate_p_threshold=None,
            minimum_variance=self.score_config.numerical.minimum_variance,
            relative_variance_tolerance=self.score_config.numerical.relative_variance_tolerance,
            native_genotype_mean=jax.device_put(genotype_mean),
        )


class BinaryFirthJaxBackend(BinaryJaxBackendBase):
    """Execute binary score kernels with approximate-Firth correction."""

    def __init__(
        self,
        *,
        p_threshold: float,
        firth_se: bool,
        minimum_probability: float,
        minimum_variance: float,
        relative_variance_tolerance: float,
        null_logistic_maximum_iterations: int,
        null_logistic_coefficient_tolerance: float,
        firth_batch_size: int,
        firth_candidate_capacity: int,
        firth_maximum_iterations: int,
        firth_gradient_tolerance: float,
        firth_coefficient_tolerance: float,
        firth_likelihood_tolerance: float,
        firth_maximum_step_size: float,
        firth_pseudo_maximum_iterations: int,
        firth_pseudo_inner_maximum_iterations: int,
        firth_newton_raphson_zero_start_iterations: int,
        firth_line_search_maximum_attempts: int,
        firth_step_halving_maximum_attempts: int,
        firth_initial_response_scale: float,
        firth_sparse_carrier_dosage_threshold: float,
        firth_step_halving_scale: float,
        firth_use_block_math: bool,
        null_firth_maximum_iterations: int,
        null_firth_gradient_tolerance: float,
        null_firth_maximum_step_size: float,
        null_firth_fallback_iteration_multiplier: int,
        null_firth_fallback_step_divisor: float,
        null_firth_line_search_maximum_attempts: int,
        null_firth_step_halving_scale: float,
    ) -> None:
        """Initialize score and approximate-Firth policy."""
        from g.compute.regenie2_binary import api as regenie2_binary
        from g.compute.regenie2_binary import config as regenie2_binary_config

        super().__init__(
            minimum_probability=minimum_probability,
            minimum_variance=minimum_variance,
            relative_variance_tolerance=relative_variance_tolerance,
            null_logistic_maximum_iterations=null_logistic_maximum_iterations,
            null_logistic_coefficient_tolerance=null_logistic_coefficient_tolerance,
        )
        self._binary_api = regenie2_binary
        self.correction_plan = types.BinaryCorrectionPlan(
            p_threshold=p_threshold,
            firth_se=firth_se,
        )
        self.binary_config = regenie2_binary_config.BinaryKernelConfig(
            numerical=self.score_config.numerical,
            null_logistic=self.score_config.null_logistic,
            firth_candidate=regenie2_binary_config.FirthCandidateConfig(
                batch_size=firth_batch_size,
                candidate_capacity=firth_candidate_capacity,
            ),
            approximate_firth=regenie2_binary_config.ApproximateFirthConfig(
                maximum_iterations=firth_maximum_iterations,
                gradient_tolerance=firth_gradient_tolerance,
                coefficient_tolerance=firth_coefficient_tolerance,
                likelihood_tolerance=firth_likelihood_tolerance,
                maximum_step_size=firth_maximum_step_size,
                pseudo_maximum_iterations=firth_pseudo_maximum_iterations,
                pseudo_inner_maximum_iterations=firth_pseudo_inner_maximum_iterations,
                newton_raphson_zero_start_iterations=firth_newton_raphson_zero_start_iterations,
                line_search_maximum_attempts=firth_line_search_maximum_attempts,
                step_halving_maximum_attempts=firth_step_halving_maximum_attempts,
                initial_response_scale=firth_initial_response_scale,
                sparse_carrier_dosage_threshold=firth_sparse_carrier_dosage_threshold,
                step_halving_scale=firth_step_halving_scale,
                use_block_math=firth_use_block_math,
            ),
            null_firth=regenie2_binary_config.NullFirthConfig(
                maximum_iterations=null_firth_maximum_iterations,
                gradient_tolerance=null_firth_gradient_tolerance,
                maximum_step_size=null_firth_maximum_step_size,
                fallback_iteration_multiplier=null_firth_fallback_iteration_multiplier,
                fallback_step_divisor=null_firth_fallback_step_divisor,
                line_search_maximum_attempts=null_firth_line_search_maximum_attempts,
                step_halving_scale=null_firth_step_halving_scale,
            ),
        )

    def prepare_chromosome(
        self,
        group_state: regenie2_binary_state.Regenie2MultiBinaryState,
        prediction_matrix: npt.NDArray[np.float32],
    ) -> regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState:
        """Prepare score operands and approximate-Firth null state."""
        return self._binary_state.build_multi_binary_firth_chromosome_state(
            state=group_state,
            loco_offset_matrix=jax.device_put(prediction_matrix),
            kernel_config=self.binary_config,
        )

    def compute_dosage_batch(
        self,
        chromosome_state: regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState,
        dosage_matrix: npt.NDArray[np.float32],
        genotype_mean: npt.NDArray[np.float32],
        sparse_candidate_mask: npt.NDArray[np.bool_],
    ) -> DeviceAssociationResult:
        """Transfer and submit one dosage batch to score and Firth kernels."""
        return self._binary_api.compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=jax.device_put(dosage_matrix),
            correction_plan=self.correction_plan,
            kernel_config=self.binary_config,
            sparse_candidate_mask=jax.device_put(sparse_candidate_mask),
            native_genotype_mean=jax.device_put(genotype_mean),
        )

    def compute_packed8_batch(
        self,
        chromosome_state: regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState,
        packed8_probabilities: npt.NDArray[np.uint8],
        genotype_mean: npt.NDArray[np.float32],
        sparse_candidate_mask: npt.NDArray[np.bool_],
    ) -> DeviceAssociationResult:
        """Transfer and submit one packed8 batch to score and Firth kernels."""
        return self._binary_api.compute_regenie2_multi_binary_chunk_from_chromosome_state_packed8(
            chromosome_state=chromosome_state,
            packed_probability_pairs_by_variant=jax.device_put(packed8_probabilities, may_alias=False),
            correction_plan=self.correction_plan,
            kernel_config=self.binary_config,
            sparse_candidate_mask=jax.device_put(sparse_candidate_mask),
            native_genotype_mean=jax.device_put(genotype_mean),
        )
