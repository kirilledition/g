"""Grouped multi-phenotype callback fanout helpers."""

from __future__ import annotations

import contextlib

import numpy as np
import numpy.typing as npt

import g.engine.callbacks.shared as shared
import g.engine.callbacks.transfers as transfers
from g import _core


class GroupedMultiPhenotypeFanoutCallback:
    """Fan out one union-sample native decode to compatible phenotype-group callbacks."""

    def __init__(self, group_fanouts: tuple[shared.MultiPhenotypeGroupFanout, ...]) -> None:
        """Initialize fanout callback state."""
        if not group_fanouts:
            message = "At least one phenotype group callback is required for fanout delivery."
            raise ValueError(message)
        self.group_fanouts = group_fanouts

    @property
    def native_callback_batch_size(self) -> int:
        """Return the native callback batch size shared by grouped callbacks."""
        return self.group_fanouts[0].callback.native_callback_batch_size

    def start(self) -> None:
        """Start all group callbacks before native chunk delivery."""
        for group_fanout in self.group_fanouts:
            group_fanout.callback.start()

    def finish(self) -> None:
        """Drain all group callbacks after native chunk delivery."""
        first_error: BaseException | None = None
        for group_fanout in self.group_fanouts:
            try:
                group_fanout.callback.finish()
            except BaseException as error:  # noqa: BLE001
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def abort(self) -> None:
        """Abort all group callbacks after a native delivery failure."""
        for group_fanout in self.group_fanouts:
            with contextlib.suppress(Exception):
                group_fanout.callback.abort()

    def acquire_variant_major_dosage_buffer(
        self,
        variant_count: int,
        sample_count: int,
    ) -> npt.NDArray[np.float32]:
        """Return a union-sample host dosage buffer for native decode."""
        return np.empty((variant_count, sample_count), dtype=np.float32, order="C")

    def acquire_variant_major_packed8_probability_pair_buffer(
        self,
        variant_count: int,
        sample_count: int,
    ) -> npt.NDArray[np.uint8]:
        """Return a union-sample host packed8 buffer for native decode."""
        return np.empty((variant_count, sample_count, 2), dtype=np.uint8, order="C")

    def compute_preprocessed_variant_major_dosage_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Slice one union-sample dosage chunk and forward it to each group callback."""
        del chunk_stats
        variant_count = int(genotype_matrix_by_variant.shape[0])
        for group_fanout in self.group_fanouts:
            group_callback = group_fanout.callback
            group_sample_count = int(group_fanout.sample_position_array.shape[0])
            group_genotype_matrix = group_callback.acquire_variant_major_dosage_buffer(
                variant_count,
                group_sample_count,
            )
            np.take(
                genotype_matrix_by_variant,
                group_fanout.sample_position_array,
                axis=1,
                out=group_genotype_matrix,
            )
            group_chunk_stats = transfers.build_projected_variant_major_dosage_chunk_stats(group_genotype_matrix)
            group_callback.compute_preprocessed_variant_major_dosage_chunk(
                metadata,
                group_genotype_matrix,
                group_chunk_stats,
            )

    def compute_preprocessed_variant_major_packed8_probability_pair_chunk(
        self,
        metadata: _core.VariantMetadata,
        packed_probability_pairs_by_variant: npt.NDArray[np.uint8],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Reject packed8 fanout until projected packed statistics are available."""
        del metadata, packed_probability_pairs_by_variant, chunk_stats
        message = "Union grouped packed8 delivery requires projected packed8 chunk statistics."
        raise RuntimeError(message)


__all__ = ["GroupedMultiPhenotypeFanoutCallback"]
