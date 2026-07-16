"""Coarse native-to-JAX association backend."""

from __future__ import annotations

import functools
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from g import types
from g.compute.common import compressed_genotype
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
type DevicePacked8RawStatistics = compressed_genotype.Packed8RawStatistics[jax.Array, jax.Array]
type HostPacked8RawStatistics = compressed_genotype.Packed8RawStatistics[npt.NDArray[np.uint64], npt.NDArray[np.uint32]]


@dataclass(frozen=True, slots=True)
class DeviceCompressedTransferSelection:
    """Persistent device selection and static compressed-transfer geometry.

    Attributes:
        selected_sample_indices: Indexed source samples, or an empty contiguous operand.
        source_sample_count: Number of samples encoded in each source row.
        selected_sample_count: Number of samples consumed by association kernels.
        selection_start: Contiguous source offset, or ``-1`` for indexed selection.

    """

    selected_sample_indices: jax.Array
    source_sample_count: int
    selected_sample_count: int
    selection_start: int


@dataclass(frozen=True, slots=True)
class DeviceGroupState[AssociationState]:
    """Association state with an optional persistent compressed transfer.

    Attributes:
        association_state: Mode-specific reusable association state.
        compressed_transfer_selection: Persistent compressed-transfer selection.

    """

    association_state: AssociationState
    compressed_transfer_selection: DeviceCompressedTransferSelection | None


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=("association", "raw_packed8_statistics", "firth_candidate_count"),
    meta_fields=("firth_candidate_capacity",),
)
@dataclass(frozen=True, slots=True)
class AssociationBatch[AssociationValue, RawStatistics]:
    """Association values and optional packed8 summaries at one residency.

    Attributes:
        association: Device or host association statistics.
        raw_packed8_statistics: Exact compressed-input summaries when applicable.
        firth_candidate_count: Device count used to detect hard-capacity overflow after materialization.
        firth_candidate_capacity: Static capacity matching the device count.

    """

    association: AssociationValue
    raw_packed8_statistics: RawStatistics | None
    firth_candidate_count: jax.Array | npt.NDArray[np.int32] | None
    firth_candidate_capacity: int | None


type DeviceAssociationBatch = AssociationBatch[
    DeviceAssociationResult,
    DevicePacked8RawStatistics,
]
type HostMaterializedAssociationBatch = AssociationBatch[
    HostAssociationResult,
    HostPacked8RawStatistics,
]


@dataclass(frozen=True, slots=True)
class DeviceGenotypeBatch:
    """Genotype operands transferred for one association batch.

    Attributes:
        genotype_values: Dosages or packed probability pairs on the device.
        genotype_mean: Native genotype means on the device.
        imputed_dosage_square_sum: Linear-test square sums when required.
        sparse_candidate_mask: Binary-Firth sparse candidates when required.
        packed8: Whether genotype values contain packed probability pairs.
        raw_packed8_statistics: Exact compressed-input summaries when applicable.

    """

    genotype_values: jax.Array
    genotype_mean: jax.Array
    imputed_dosage_square_sum: jax.Array | None
    sparse_candidate_mask: jax.Array | None
    packed8: bool
    raw_packed8_statistics: DevicePacked8RawStatistics | None


def prepare_compressed_transfer_selection(
    source_sample_count: int | None,
    selected_sample_count: int | None,
    selection_start: int | None,
    selected_sample_indices: npt.NDArray[np.uint32] | None,
) -> DeviceCompressedTransferSelection | None:
    """Upload one persistent compressed-transfer selection for a group."""
    if source_sample_count is None:
        if selected_sample_count is not None or selection_start is not None or selected_sample_indices is not None:
            raise ValueError("Host transfer requires every compressed selection value to be None.")
        return None
    if selected_sample_count is None:
        raise ValueError("Compressed transfer requires source and selected sample counts.")
    if source_sample_count <= 0 or selected_sample_count <= 0:
        raise ValueError("Compressed source and selected sample counts must be positive.")
    if selection_start is not None and selected_sample_indices is None:
        if selection_start < 0 or selected_sample_count > source_sample_count - selection_start:
            raise ValueError("Contiguous compressed selection exceeds the source sample count.")
        host_selected_sample_indices = np.empty((0,), dtype=np.uint32)
        native_selection_start = selection_start
    elif selection_start is None and selected_sample_indices is not None:
        if selected_sample_indices.ndim != 1 or selected_sample_indices.dtype != np.dtype(np.uint32):
            raise ValueError("Compressed selected sample indices must be a one-dimensional uint32 array.")
        if selected_sample_indices.size != selected_sample_count:
            raise ValueError("Indexed compressed selection requires one index per selected sample.")
        host_selected_sample_indices = selected_sample_indices
        native_selection_start = -1
    else:
        raise ValueError("Compressed selection must be either contiguous or indexed.")
    return DeviceCompressedTransferSelection(
        selected_sample_indices=jax.device_put(host_selected_sample_indices, may_alias=False),
        source_sample_count=source_sample_count,
        selected_sample_count=selected_sample_count,
        selection_start=native_selection_start,
    )


class JaxBackendBase:
    """Shared host materialization for concrete association backends."""

    retain_compressed_imputed_dosage_square_sum: bool
    collect_compressed_sparse_candidate_mask: bool

    def transfer_batch(
        self,
        genotype_values: npt.NDArray[np.float32] | npt.NDArray[np.uint8],
        genotype_mean: npt.NDArray[np.float32],
        imputed_dosage_square_sum: npt.NDArray[np.float32] | None,
        sparse_candidate_mask: npt.NDArray[np.bool_] | None,
    ) -> DeviceGenotypeBatch:
        """Initiate asynchronous host-to-device transfer for one batch."""
        packed8 = genotype_values.dtype == np.dtype(np.uint8)
        device_genotype_values = (
            jax.device_put(genotype_values, may_alias=False) if packed8 else jax.device_put(genotype_values)
        )
        return DeviceGenotypeBatch(
            genotype_values=device_genotype_values,
            genotype_mean=jax.device_put(genotype_mean),
            imputed_dosage_square_sum=(
                None if imputed_dosage_square_sum is None else jax.device_put(imputed_dosage_square_sum)
            ),
            sparse_candidate_mask=(None if sparse_candidate_mask is None else jax.device_put(sparse_candidate_mask)),
            packed8=packed8,
            raw_packed8_statistics=None,
        )

    def transfer_compressed_batch[AssociationState](
        self,
        group_state: DeviceGroupState[AssociationState],
        compressed_slab: npt.NDArray[np.uint8],
        compressed_metadata: npt.NDArray[np.uint32],
        compute_variant_count: int,
    ) -> DeviceGenotypeBatch:
        """Transfer and decode one trusted raw-DEFLATE packed8 batch."""
        transfer_selection = group_state.compressed_transfer_selection
        if transfer_selection is None:
            raise ValueError("Compressed transfer requires a prepared compressed group selection.")
        decoded_batch = compressed_genotype.decode_packed8_deflate_batch(
            compressed_slab=jax.device_put(compressed_slab, may_alias=False),
            compressed_metadata=jax.device_put(compressed_metadata, may_alias=False),
            selected_sample_indices=transfer_selection.selected_sample_indices,
            source_sample_count=transfer_selection.source_sample_count,
            selected_sample_count=transfer_selection.selected_sample_count,
            selection_start=transfer_selection.selection_start,
            compute_variant_count=compute_variant_count,
            retain_imputed_dosage_square_sum=self.retain_compressed_imputed_dosage_square_sum,
            collect_sparse_candidate_mask=self.collect_compressed_sparse_candidate_mask,
        )
        return DeviceGenotypeBatch(
            genotype_values=decoded_batch.packed_probability_pairs_by_variant,
            genotype_mean=decoded_batch.genotype_mean,
            imputed_dosage_square_sum=decoded_batch.imputed_dosage_square_sum,
            sparse_candidate_mask=decoded_batch.sparse_candidate_mask,
            packed8=True,
            raw_packed8_statistics=decoded_batch.raw_packed8_statistics,
        )

    def materialize_batch(
        self,
        device_result: DeviceAssociationBatch,
        active_trait_indices: npt.NDArray[np.int32] | None,
        logical_variant_count: int,
    ) -> HostMaterializedAssociationBatch:
        """Materialize selected association and packed8 arrays in one transfer."""
        association = device_result.association
        if active_trait_indices is None:
            beta = association.beta
            standard_error = association.standard_error
            chi_squared = association.chi_squared
            log10_p_value = association.log10_p_value
            correction_code = association.correction_code
        else:
            active_trait_index_array = jnp.asarray(active_trait_indices, dtype=jnp.int32)
            beta = jnp.take(association.beta, active_trait_index_array, axis=0)
            standard_error = jnp.take(association.standard_error, active_trait_index_array, axis=0)
            chi_squared = jnp.take(association.chi_squared, active_trait_index_array, axis=0)
            log10_p_value = jnp.take(association.log10_p_value, active_trait_index_array, axis=0)
            correction_code = (
                None
                if association.correction_code is None
                else jnp.take(association.correction_code, active_trait_index_array, axis=0)
            )

        if correction_code is None:
            selected_association = association_result.AssociationResult(
                beta=jnp.asarray(beta[:, :logical_variant_count], dtype=jnp.float32),
                standard_error=jnp.asarray(standard_error[:, :logical_variant_count], dtype=jnp.float32),
                chi_squared=jnp.asarray(chi_squared[:, :logical_variant_count], dtype=jnp.float32),
                log10_p_value=jnp.asarray(log10_p_value[:, :logical_variant_count], dtype=jnp.float32),
                correction_code=None,
            )
        else:
            selected_association = association_result.AssociationResult(
                beta=jnp.asarray(beta[:, :logical_variant_count], dtype=jnp.float32),
                standard_error=jnp.asarray(standard_error[:, :logical_variant_count], dtype=jnp.float32),
                chi_squared=jnp.asarray(chi_squared[:, :logical_variant_count], dtype=jnp.float32),
                log10_p_value=jnp.asarray(log10_p_value[:, :logical_variant_count], dtype=jnp.float32),
                correction_code=jnp.asarray(correction_code[:, :logical_variant_count], dtype=jnp.uint8),
            )

        raw_packed8_statistics = device_result.raw_packed8_statistics
        materializable_raw_statistics = (
            None
            if raw_packed8_statistics is None
            else compressed_genotype.Packed8RawStatistics(
                dosage_sums=raw_packed8_statistics.dosage_sums[:logical_variant_count],
                dosage_square_sums=raw_packed8_statistics.dosage_square_sums[:logical_variant_count],
                statuses=raw_packed8_statistics.statuses[:logical_variant_count],
                selected_sample_count=raw_packed8_statistics.selected_sample_count,
            )
        )
        materialized_batch = jax.device_get(
            AssociationBatch(
                association=selected_association,
                raw_packed8_statistics=materializable_raw_statistics,
                firth_candidate_count=device_result.firth_candidate_count,
                firth_candidate_capacity=device_result.firth_candidate_capacity,
            )
        )
        materialized_firth_candidate_count = materialized_batch.firth_candidate_count
        materialized_firth_candidate_capacity = materialized_batch.firth_candidate_capacity
        if (materialized_firth_candidate_count is None) != (materialized_firth_candidate_capacity is None):
            raise ValueError("Firth candidate count and capacity must be materialized together.")
        if materialized_firth_candidate_count is not None and materialized_firth_candidate_capacity is not None:
            host_firth_candidate_count = int(materialized_firth_candidate_count)
            if host_firth_candidate_count > materialized_firth_candidate_capacity:
                message = (
                    f"Aggregate Firth candidate count {host_firth_candidate_count} exceeded the static aggregate "
                    f"capacity of {materialized_firth_candidate_capacity}. Increase [compute] firth_candidate_capacity "
                    "(the per-trait capacity scaling value) and rerun."
                )
                raise ValueError(message)
        return materialized_batch


class LinearJaxBackend(JaxBackendBase):
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
    ) -> DeviceGroupState[regenie2_linear_state.Regenie2MultiLinearState]:
        """Prepare reusable device state for one aligned phenotype group."""
        association_state = self._linear_state.build_multi_linear_state(
            covariate_matrix=jax.device_put(covariate_matrix),
            phenotype_matrix=jax.device_put(phenotype_matrix),
        )
        compressed_transfer_selection = prepare_compressed_transfer_selection(
            source_sample_count=source_sample_count,
            selected_sample_count=selected_sample_count,
            selection_start=selection_start,
            selected_sample_indices=selected_sample_indices,
        )
        jax.block_until_ready(association_state.phenotype_residual_matrix)
        return DeviceGroupState(
            association_state=association_state,
            compressed_transfer_selection=compressed_transfer_selection,
        )

    def prepare_chromosome(
        self,
        group_state: DeviceGroupState[regenie2_linear_state.Regenie2MultiLinearState],
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
        batch: DeviceGenotypeBatch,
    ) -> DeviceAssociationBatch:
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
        return AssociationBatch(
            association=association,
            raw_packed8_statistics=batch.raw_packed8_statistics,
            firth_candidate_count=None,
            firth_candidate_capacity=None,
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
        source_sample_count: int | None,
        selected_sample_count: int | None,
        selection_start: int | None,
        selected_sample_indices: npt.NDArray[np.uint32] | None,
    ) -> DeviceGroupState[regenie2_binary_state.Regenie2MultiBinaryState]:
        """Prepare reusable device state for one aligned phenotype group."""
        association_state = self._binary_state.build_multi_binary_state(
            covariate_matrix=jax.device_put(covariate_matrix),
            phenotype_matrix=jax.device_put(phenotype_matrix),
        )
        return DeviceGroupState(
            association_state=association_state,
            compressed_transfer_selection=prepare_compressed_transfer_selection(
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
        group_state: DeviceGroupState[regenie2_binary_state.Regenie2MultiBinaryState],
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
        batch: DeviceGenotypeBatch,
    ) -> DeviceAssociationBatch:
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
        return AssociationBatch(
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
        group_state: DeviceGroupState[regenie2_binary_state.Regenie2MultiBinaryState],
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
        batch: DeviceGenotypeBatch,
    ) -> DeviceAssociationBatch:
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
        return AssociationBatch(
            association=corrected_result.association,
            raw_packed8_statistics=batch.raw_packed8_statistics,
            firth_candidate_count=corrected_result.firth_candidate_count,
            firth_candidate_capacity=corrected_result.firth_candidate_capacity,
        )
