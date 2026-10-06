"""Typed payloads retained by the coarse native-to-JAX boundary."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import numpy as np
import numpy.typing as npt

from g.compute.common import compressed_genotype
from g.compute.common import result as association_result

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
class HostCompressedTransferSelection:
    """Validated host operands for one compressed-transfer selection.

    Attributes:
        selected_sample_indices: Indexed source samples, or an empty contiguous operand.
        selection_start: Contiguous source offset, or ``-1`` for indexed selection.

    """

    selected_sample_indices: npt.NDArray[np.uint32]
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


@dataclass(frozen=True, slots=True)
class DeviceSharedSourceBatch:
    """Immutable full-source packed values retained across selected consumers.

    Attributes:
        packed_probability_pairs_by_variant: Full-source pairs, including padded rows.
        statuses: Original source validation statuses, including padded rows.
        logical_variant_count: Number of source rows before compute padding.

    """

    packed_probability_pairs_by_variant: jax.Array
    statuses: jax.Array
    logical_variant_count: int
