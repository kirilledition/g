"""Concrete numerical backends consumed by the native association host."""

from __future__ import annotations

import typing

import jax

from g import types
from g.backend import contracts, transport

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from g.compute.regenie2_binary import state as regenie2_binary_state
    from g.compute.regenie2_linear import state as regenie2_linear_state


class LinearJaxBackend(transport.BackendTransport):
    """Execute linear REGENIE kernels without runtime mode dispatch."""

    retain_compressed_imputed_dosage_square_sum = True
    collect_compressed_sparse_candidate_mask = False

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
        source_sample_count: int | None,
        selected_sample_count: int | None,
        selection_start: int | None,
        selected_sample_indices: npt.NDArray[np.uint32] | None,
    ) -> contracts.DeviceGroupState[regenie2_linear_state.Regenie2MultiLinearState]:
        """Prepare reusable device state for one aligned phenotype group."""
        association_state = self._linear_state.build_multi_linear_state(
            covariate_matrix=jax.device_put(covariate_matrix),
            phenotype_matrix=jax.device_put(phenotype_matrix),
        )
        compressed_transfer_selection = transport.prepare_compressed_transfer_selection(
            source_sample_count=source_sample_count,
            selected_sample_count=selected_sample_count,
            selection_start=selection_start,
            selected_sample_indices=selected_sample_indices,
        )
        jax.block_until_ready(association_state.phenotype_residual_matrix)
        return contracts.DeviceGroupState(
            association_state=association_state,
            compressed_transfer_selection=compressed_transfer_selection,
        )

    def prepare_chromosome(
        self,
        group_state: contracts.DeviceGroupState[regenie2_linear_state.Regenie2MultiLinearState],
        prediction_matrix: npt.NDArray[np.float32],
    ) -> regenie2_linear_state.Regenie2MultiLinearChromosomeState:
        """Prepare reusable device state for one chromosome."""
        chromosome_state = self._linear_state.build_multi_linear_chromosome_state(
            state=group_state.association_state,
            loco_prediction_matrix=jax.device_put(prediction_matrix),
        )
        jax.block_until_ready(chromosome_state.score_left_hand_matrix)
        return chromosome_state

    def compute_batch(
        self,
        chromosome_state: regenie2_linear_state.Regenie2MultiLinearChromosomeState,
        batch: contracts.DeviceGenotypeBatch,
    ) -> contracts.DeviceAssociationBatch:
        """Submit one transferred batch to the matching linear kernel."""
        if batch.imputed_dosage_square_sum is None:
            raise ValueError("Linear association requires imputed dosage square sums.")
        if batch.packed8:
            association = self._linear_score.compute_multi_linear_chunk_packed8_donating_inputs(
                chromosome_state=chromosome_state,
                packed_probability_pairs_by_variant=batch.genotype_values,
                native_genotype_mean=batch.genotype_mean,
                genotype_imputed_dosage_square_sum=batch.imputed_dosage_square_sum,
                linear_minimum_variance=self.minimum_variance,
                linear_relative_variance_tolerance=self.relative_variance_tolerance,
            )
        else:
            association = self._linear_score.compute_regenie2_linear_chunk_trait_major_variant_major_donating_inputs(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=batch.genotype_values,
                native_genotype_mean=batch.genotype_mean,
                genotype_imputed_dosage_square_sum=batch.imputed_dosage_square_sum,
                linear_minimum_variance=self.minimum_variance,
                linear_relative_variance_tolerance=self.relative_variance_tolerance,
            )
        return contracts.AssociationBatch(
            association=association,
            raw_packed8_statistics=batch.raw_packed8_statistics,
            firth_candidate_count=None,
            firth_candidate_capacity=None,
        )


class BinaryJaxBackendBase(transport.BackendTransport):
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
        source_sample_count: int | None,
        selected_sample_count: int | None,
        selection_start: int | None,
        selected_sample_indices: npt.NDArray[np.uint32] | None,
    ) -> contracts.DeviceGroupState[regenie2_binary_state.Regenie2MultiBinaryState]:
        """Prepare reusable device state for one aligned phenotype group."""
        association_state = self._binary_state.build_multi_binary_state(
            covariate_matrix=jax.device_put(covariate_matrix),
            phenotype_matrix=jax.device_put(phenotype_matrix),
        )
        return contracts.DeviceGroupState(
            association_state=association_state,
            compressed_transfer_selection=transport.prepare_compressed_transfer_selection(
                source_sample_count=source_sample_count,
                selected_sample_count=selected_sample_count,
                selection_start=selection_start,
                selected_sample_indices=selected_sample_indices,
            ),
        )


class BinaryScoreJaxBackend(BinaryJaxBackendBase):
    """Execute binary score kernels without correction dispatch."""

    retain_compressed_imputed_dosage_square_sum = False
    collect_compressed_sparse_candidate_mask = False

    def prepare_chromosome(
        self,
        group_state: contracts.DeviceGroupState[regenie2_binary_state.Regenie2MultiBinaryState],
        prediction_matrix: npt.NDArray[np.float32],
    ) -> regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState:
        """Prepare only the chromosome operands consumed by score kernels."""
        return self._binary_state.build_multi_binary_score_chromosome_state(
            state=group_state.association_state,
            loco_offset_matrix=jax.device_put(prediction_matrix),
            kernel_config=self.score_config,
        )

    def compute_batch(
        self,
        chromosome_state: regenie2_binary_state.Regenie2MultiBinaryScoreChromosomeState,
        batch: contracts.DeviceGenotypeBatch,
    ) -> contracts.DeviceAssociationBatch:
        """Submit one transferred batch to the matching binary score kernel."""
        if batch.packed8:
            association = self._binary_score.compute_multi_binary_score_test_packed8_donating_inputs(
                chromosome_state=chromosome_state,
                packed_probability_pairs_by_variant=batch.genotype_values,
                firth_candidate_p_threshold=None,
                minimum_variance=self.score_config.numerical.minimum_variance,
                relative_variance_tolerance=self.score_config.numerical.relative_variance_tolerance,
                native_genotype_mean=batch.genotype_mean,
            )
        else:
            association = self._binary_score.compute_multi_binary_score_test_variant_major_donating_inputs(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=batch.genotype_values,
                firth_candidate_p_threshold=None,
                minimum_variance=self.score_config.numerical.minimum_variance,
                relative_variance_tolerance=self.score_config.numerical.relative_variance_tolerance,
                native_genotype_mean=batch.genotype_mean,
            )
        return contracts.AssociationBatch(
            association=association,
            raw_packed8_statistics=batch.raw_packed8_statistics,
            firth_candidate_count=None,
            firth_candidate_capacity=None,
        )


class BinaryFirthJaxBackend(BinaryJaxBackendBase):
    """Execute binary score kernels with approximate-Firth correction."""

    retain_compressed_imputed_dosage_square_sum = False
    collect_compressed_sparse_candidate_mask = True

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
        firth_maximum_step_size: float,
        firth_pseudo_maximum_iterations: int,
        firth_pseudo_inner_maximum_iterations: int,
        firth_line_search_maximum_attempts: int,
        firth_sparse_carrier_dosage_threshold: float,
        use_cuda_firth_components: bool,
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
                maximum_step_size=firth_maximum_step_size,
                pseudo_maximum_iterations=firth_pseudo_maximum_iterations,
                pseudo_inner_maximum_iterations=firth_pseudo_inner_maximum_iterations,
                line_search_maximum_attempts=firth_line_search_maximum_attempts,
                sparse_carrier_dosage_threshold=firth_sparse_carrier_dosage_threshold,
                use_cuda_components=use_cuda_firth_components,
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
        group_state: contracts.DeviceGroupState[regenie2_binary_state.Regenie2MultiBinaryState],
        prediction_matrix: npt.NDArray[np.float32],
    ) -> regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState:
        """Prepare score operands and approximate-Firth null state."""
        return self._binary_state.build_multi_binary_firth_chromosome_state(
            state=group_state.association_state,
            loco_offset_matrix=jax.device_put(prediction_matrix),
            kernel_config=self.binary_config,
        )

    def compute_batch(
        self,
        chromosome_state: regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState,
        batch: contracts.DeviceGenotypeBatch,
    ) -> contracts.DeviceAssociationBatch:
        """Submit one transferred batch to the matching score and Firth kernels."""
        if batch.sparse_candidate_mask is None:
            raise ValueError("Binary Firth association requires a sparse candidate mask.")
        if batch.packed8:
            corrected_result = self._binary_api.compute_regenie2_multi_binary_chunk_from_chromosome_state_packed8(
                chromosome_state=chromosome_state,
                packed_probability_pairs_by_variant=batch.genotype_values,
                correction_plan=self.correction_plan,
                kernel_config=self.binary_config,
                sparse_candidate_mask=batch.sparse_candidate_mask,
                native_genotype_mean=batch.genotype_mean,
            )
        else:
            corrected_result = self._binary_api.compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=batch.genotype_values,
                correction_plan=self.correction_plan,
                kernel_config=self.binary_config,
                sparse_candidate_mask=batch.sparse_candidate_mask,
                native_genotype_mean=batch.genotype_mean,
            )
        return contracts.AssociationBatch(
            association=corrected_result.association,
            raw_packed8_statistics=batch.raw_packed8_statistics,
            firth_candidate_count=corrected_result.firth_candidate_count,
            firth_candidate_capacity=corrected_result.firth_candidate_capacity,
        )
