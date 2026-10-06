"""Persistent selections and genotype transfers for native association backends."""

from __future__ import annotations

import jax
import numpy as np
import numpy.typing as npt

from g.backend import contracts, materialization
from g.compute.common import compressed_genotype


def device_genotype_batch_from_decoded(
    decoded_batch: compressed_genotype.DecodedPacked8DeflateBatch,
) -> contracts.DeviceGenotypeBatch:
    """Adapt decoded values and private statistics to the compute boundary."""
    return contracts.DeviceGenotypeBatch(
        genotype_values=decoded_batch.packed_probability_pairs_by_variant,
        genotype_mean=decoded_batch.genotype_mean,
        imputed_dosage_square_sum=decoded_batch.imputed_dosage_square_sum,
        sparse_candidate_mask=decoded_batch.sparse_candidate_mask,
        packed8=True,
        raw_packed8_statistics=decoded_batch.raw_packed8_statistics,
    )


def resolve_host_compressed_transfer_selection(
    source_sample_count: int,
    selected_sample_count: int,
    selection_start: int | None,
    selected_sample_indices: npt.NDArray[np.uint32] | None,
) -> contracts.HostCompressedTransferSelection:
    """Validate and normalize one compressed-transfer selection on the host."""
    if selection_start is not None and selected_sample_indices is None:
        if selection_start < 0 or selected_sample_count > source_sample_count - selection_start:
            raise ValueError("Contiguous compressed selection exceeds the source sample count.")
        return contracts.HostCompressedTransferSelection(
            selected_sample_indices=np.empty((0,), dtype=np.uint32),
            selection_start=selection_start,
        )
    if selection_start is None and selected_sample_indices is not None:
        if selected_sample_indices.ndim != 1 or selected_sample_indices.dtype != np.dtype(np.uint32):
            raise ValueError("Compressed selected sample indices must be a one-dimensional uint32 array.")
        if selected_sample_indices.size != selected_sample_count:
            raise ValueError("Indexed compressed selection requires one index per selected sample.")
        return contracts.HostCompressedTransferSelection(
            selected_sample_indices=selected_sample_indices,
            selection_start=-1,
        )
    raise ValueError("Compressed selection must be either contiguous or indexed.")


def prepare_compressed_transfer_selection(
    source_sample_count: int | None,
    selected_sample_count: int | None,
    selection_start: int | None,
    selected_sample_indices: npt.NDArray[np.uint32] | None,
) -> contracts.DeviceCompressedTransferSelection | None:
    """Upload one persistent compressed-transfer selection for a group."""
    if source_sample_count is None:
        if selected_sample_count is not None or selection_start is not None or selected_sample_indices is not None:
            raise ValueError("Host transfer requires every compressed selection value to be None.")
        return None
    if selected_sample_count is None:
        raise ValueError("Compressed transfer requires source and selected sample counts.")
    if source_sample_count <= 0 or selected_sample_count <= 0:
        raise ValueError("Compressed source and selected sample counts must be positive.")
    host_selection = resolve_host_compressed_transfer_selection(
        source_sample_count,
        selected_sample_count,
        selection_start,
        selected_sample_indices,
    )
    return contracts.DeviceCompressedTransferSelection(
        selected_sample_indices=jax.device_put(host_selection.selected_sample_indices, may_alias=False),
        source_sample_count=source_sample_count,
        selected_sample_count=selected_sample_count,
        selection_start=host_selection.selection_start,
    )


class BackendTransport:
    """Shared device transport lifecycle for concrete association backends."""

    retain_compressed_imputed_dosage_square_sum: bool
    collect_compressed_sparse_candidate_mask: bool

    def transfer_batch(
        self,
        genotype_values: npt.NDArray[np.float32] | npt.NDArray[np.uint8],
        genotype_mean: npt.NDArray[np.float32],
        imputed_dosage_square_sum: npt.NDArray[np.float32] | None,
        sparse_candidate_mask: npt.NDArray[np.bool_] | None,
    ) -> contracts.DeviceGenotypeBatch:
        """Initiate asynchronous host-to-device transfer for one batch."""
        packed8 = genotype_values.dtype == np.dtype(np.uint8)
        device_genotype_values = (
            jax.device_put(genotype_values, may_alias=False) if packed8 else jax.device_put(genotype_values)
        )
        return contracts.DeviceGenotypeBatch(
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
        group_state: contracts.DeviceGroupState[AssociationState],
        compressed_slab: npt.NDArray[np.uint8],
        compressed_metadata: npt.NDArray[np.uint32],
        compute_variant_count: int,
    ) -> contracts.DeviceGenotypeBatch:
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
        return device_genotype_batch_from_decoded(decoded_batch)

    def prepare_shared_source(
        self,
        compressed_slab: npt.NDArray[np.uint8],
        compressed_metadata: npt.NDArray[np.uint32],
        compute_variant_count: int,
        source_sample_count: int,
    ) -> contracts.DeviceSharedSourceBatch:
        """Decode full-source packed pairs while retaining only reusable buffers."""
        if compressed_metadata.ndim != 2 or compressed_metadata.shape[1] != 3:
            raise ValueError("Shared compressed metadata requires three columns per logical variant.")
        logical_variant_count = compressed_metadata.shape[0]
        if source_sample_count <= 0 or not 0 < logical_variant_count <= compute_variant_count:
            raise ValueError("Shared compressed input requires positive, consistent source geometry.")
        decoded_source = compressed_genotype.decode_packed8_deflate_source(
            compressed_slab=jax.device_put(compressed_slab, may_alias=False),
            compressed_metadata=jax.device_put(compressed_metadata, may_alias=False),
            source_sample_count=source_sample_count,
            compute_variant_count=compute_variant_count,
        )
        return contracts.DeviceSharedSourceBatch(
            packed_probability_pairs_by_variant=decoded_source.packed_probability_pairs_by_variant,
            statuses=decoded_source.statuses,
            logical_variant_count=logical_variant_count,
        )

    def select_shared_source[AssociationState](
        self,
        group_state: contracts.DeviceGroupState[AssociationState],
        source: contracts.DeviceSharedSourceBatch,
    ) -> contracts.DeviceGenotypeBatch:
        """Select one group without donating or mutating the shared source."""
        transfer_selection = group_state.compressed_transfer_selection
        if transfer_selection is None:
            raise ValueError("Shared source selection requires a prepared compressed group selection.")
        if transfer_selection.source_sample_count != source.packed_probability_pairs_by_variant.shape[1]:
            raise ValueError("Shared source sample count differs from the prepared group selection.")
        decoded_batch = compressed_genotype.select_shared_packed8_batch(
            packed_probability_pairs_by_variant=source.packed_probability_pairs_by_variant,
            source_statuses=source.statuses,
            selected_sample_indices=transfer_selection.selected_sample_indices,
            logical_variant_count=source.logical_variant_count,
            selected_sample_count=transfer_selection.selected_sample_count,
            selection_start=transfer_selection.selection_start,
            retain_imputed_dosage_square_sum=self.retain_compressed_imputed_dosage_square_sum,
            collect_sparse_candidate_mask=self.collect_compressed_sparse_candidate_mask,
        )
        return device_genotype_batch_from_decoded(decoded_batch)

    def materialize_batch(
        self,
        device_result: contracts.DeviceAssociationBatch,
        active_trait_indices: npt.NDArray[np.int32] | None,
        logical_variant_count: int,
    ) -> contracts.HostMaterializedAssociationBatch:
        """Materialize selected statistics through the common residency boundary."""
        return materialization.materialize_batch(
            device_result=device_result,
            active_trait_indices=active_trait_indices,
            logical_variant_count=logical_variant_count,
        )
