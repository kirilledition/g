"""Native BGEN dispatch data models."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    from g import _core, execution_plan, types


class SampleAlignmentConfigProtocol(typing.Protocol):
    """Sample identity alignment settings accepted by native dispatch."""

    sample_key_mode: types.SampleKeyMode


class MultiRegeniePredictionSourceProtocol(typing.Protocol):
    """Prediction source interface used by grouped native run inputs."""

    def get_chromosome_predictions(self, chromosome: str) -> npt.NDArray[np.float32]:
        """Return trait-major LOCO predictions for one chromosome."""
        ...


class BgenDeliveryRunInputProtocol(typing.Protocol):
    """Sample selection input accepted by native BGEN chunk delivery."""

    sample_indices: npt.NDArray[np.int64]

    @property
    def native_aligned_sample_data(self) -> _core.NativeAlignedSampleData | None:
        """Return the native single-trait alignment handle when available."""
        ...

    @property
    def native_multi_aligned_sample_data(self) -> _core.NativeMultiAlignedSampleData | None:
        """Return the native multi-trait alignment handle when available."""
        ...


class BgenDeliveryBatchSizeProtocol(typing.Protocol):
    """Callback batch-size contract accepted by native BGEN chunk delivery planning."""

    @property
    def native_callback_batch_size(self) -> int:
        """Return the native callback batch size configured for delivery."""
        ...


class BgenDeliveryCallbackProtocol(BgenDeliveryBatchSizeProtocol, typing.Protocol):
    """Callback lifecycle contract accepted by native BGEN chunk delivery."""

    def start(self) -> None:
        """Start callback worker resources."""
        ...

    def finish(self) -> None:
        """Drain callback worker resources."""
        ...

    def abort(self) -> None:
        """Abort callback worker resources."""
        ...


@dataclass(frozen=True)
class NativeBgenRunInput:
    """Sample-aligned inputs retained in native form for BGEN REGENIE step 2.

    Attributes:
        native_aligned_sample_data: Rust-owned aligned sample identifiers and matrices.
        sample_indices: BGEN sample indices for native chunk delivery.
        phenotype_vector: Host phenotype vector.
        covariate_matrix: Host design matrix.
        is_binary_trait: Whether the run is for a binary trait.

    """

    native_aligned_sample_data: _core.NativeAlignedSampleData
    sample_indices: npt.NDArray[np.int64]
    phenotype_vector: npt.NDArray[np.float32]
    covariate_matrix: npt.NDArray[np.float32]
    is_binary_trait: bool

    @property
    def native_multi_aligned_sample_data(self) -> None:
        """Return no multi-trait native alignment handle for single-trait runs."""
        return None

    @property
    def family_identifiers(self) -> tuple[str, ...]:
        """Expose family identifiers lazily for diagnostics and tests."""
        return tuple(self.native_aligned_sample_data.family_identifiers)

    @property
    def individual_identifiers(self) -> tuple[str, ...]:
        """Expose individual identifiers lazily for diagnostics and tests."""
        return tuple(self.native_aligned_sample_data.individual_identifiers)

    @property
    def covariate_names(self) -> tuple[str, ...]:
        """Expose covariate names lazily for diagnostics and tests."""
        return tuple(self.native_aligned_sample_data.covariate_names)


@dataclass(frozen=True)
class NativeBgenMultiRunInput:
    """Sample-aligned inputs for an opt-in complete-case multi-phenotype native BGEN run.

    Attributes:
        native_multi_aligned_sample_data: Rust-owned complete-case aligned multi-phenotype data.
        phenotype_names: Phenotype names in trait-major matrix order.
        sample_indices: BGEN sample indices for native chunk delivery.
        phenotype_matrix: Host trait-major phenotype matrix.
        covariate_matrix: Host shared design matrix.
        is_binary_trait: Whether the run is for binary traits.

    """

    native_multi_aligned_sample_data: _core.NativeMultiAlignedSampleData
    phenotype_names: tuple[str, ...]
    sample_indices: npt.NDArray[np.int64]
    phenotype_matrix: npt.NDArray[np.float32]
    covariate_matrix: npt.NDArray[np.float32]
    is_binary_trait: bool

    @property
    def native_aligned_sample_data(self) -> None:
        """Return no single-trait native alignment handle for multi-trait runs."""
        return None

    @property
    def family_identifiers(self) -> tuple[str, ...]:
        """Expose family identifiers lazily for diagnostics and tests."""
        return tuple(self.native_multi_aligned_sample_data.family_identifiers)

    @property
    def individual_identifiers(self) -> tuple[str, ...]:
        """Expose individual identifiers lazily for diagnostics and tests."""
        return tuple(self.native_multi_aligned_sample_data.individual_identifiers)

    @property
    def covariate_names(self) -> tuple[str, ...]:
        """Expose covariate names lazily for diagnostics and tests."""
        return tuple(self.native_multi_aligned_sample_data.covariate_names)


@dataclass(frozen=True)
class NativeBgenUnionRunInput:
    """Union sample selection used to decode one BGEN pass for several phenotype groups.

    Attributes:
        sample_indices: Ordered union of compatible phenotype-group sample indices.

    """

    sample_indices: npt.NDArray[np.int64]

    @property
    def native_aligned_sample_data(self) -> None:
        """Return no single-trait native alignment handle for union delivery."""
        return None

    @property
    def native_multi_aligned_sample_data(self) -> None:
        """Return no multi-trait native alignment handle for union delivery."""
        return None


@dataclass(frozen=True)
class NativeBgenGroupedRunInput:
    """One native-planned group of compatible per-phenotype run inputs.

    Attributes:
        compute_group: Planned phenotype group with resolved compatibility fingerprints.
        phenotype_indices: Original phenotype indices included in this group.
        run_input: Multi-trait run input for the compatible phenotype group.
        prediction_source: Native multi-trait prediction source aligned to the group.

    """

    compute_group: execution_plan.PhenotypeComputeGroup
    phenotype_indices: tuple[int, ...]
    run_input: NativeBgenMultiRunInput
    prediction_source: MultiRegeniePredictionSourceProtocol


def build_native_bgen_run_input(
    native_aligned_sample_data: _core.NativeAlignedSampleData,
) -> NativeBgenRunInput:
    """Build host Python views over Rust-owned aligned sample data."""
    return NativeBgenRunInput(
        native_aligned_sample_data=native_aligned_sample_data,
        sample_indices=np.ascontiguousarray(native_aligned_sample_data.sample_indices, dtype=np.int64),
        phenotype_vector=np.ascontiguousarray(native_aligned_sample_data.phenotype_vector, dtype=np.float32),
        covariate_matrix=np.ascontiguousarray(native_aligned_sample_data.covariate_matrix, dtype=np.float32),
        is_binary_trait=native_aligned_sample_data.is_binary_trait,
    )


def build_native_bgen_multi_run_input(
    native_multi_aligned_sample_data: _core.NativeMultiAlignedSampleData,
) -> NativeBgenMultiRunInput:
    """Build host Python views over Rust-owned complete-case multi-phenotype data."""
    return NativeBgenMultiRunInput(
        native_multi_aligned_sample_data=native_multi_aligned_sample_data,
        phenotype_names=tuple(native_multi_aligned_sample_data.phenotype_names),
        sample_indices=np.ascontiguousarray(native_multi_aligned_sample_data.sample_indices, dtype=np.int64),
        phenotype_matrix=np.ascontiguousarray(native_multi_aligned_sample_data.phenotype_matrix, dtype=np.float32),
        covariate_matrix=np.ascontiguousarray(native_multi_aligned_sample_data.covariate_matrix, dtype=np.float32),
        is_binary_trait=native_multi_aligned_sample_data.is_binary_trait,
    )
