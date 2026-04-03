from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from g.io.prefetch import prefetch_iterator_values
from g.models import GenotypeChunk, VariantMetadata


def build_chunk(variant_start_index: int) -> GenotypeChunk:
    """Build a small chunk fixture for prefetch tests."""
    return GenotypeChunk(
        genotypes=jnp.array([[0.0], [1.0]]),
        missing_mask=jnp.array([[False], [False]]),
        has_missing_values=False,
        metadata=VariantMetadata(
            variant_start_index=variant_start_index,
            variant_stop_index=variant_start_index + 1,
            chromosome=np.array(["1"]),
            variant_identifiers=np.array([f"variant{variant_start_index}"]),
            position=np.array([100 + variant_start_index], dtype=np.int64),
            allele_one=np.array(["A"]),
            allele_two=np.array(["G"]),
        ),
        allele_one_frequency=jnp.array([0.25], dtype=jnp.float32),
        observation_count=jnp.array([2], dtype=jnp.int32),
    )


def test_prefetch_iterator_values_preserves_order() -> None:
    """Ensure the background prefetch wrapper yields items in source order."""
    prefetched_chunks = list(prefetch_iterator_values(iter([build_chunk(0), build_chunk(1)]), prefetch_chunks=2))

    assert [chunk.metadata.variant_start_index for chunk in prefetched_chunks] == [0, 1]


def test_prefetch_iterator_values_surfaces_worker_errors() -> None:
    """Ensure iterator failures on the worker thread propagate to the consumer."""

    from collections.abc import Iterator
    def failing_iterator() -> Iterator[GenotypeChunk]:
        raise ValueError("broken iterator")
        yield build_chunk(0)

    with pytest.raises(RuntimeError, match="broken iterator"):
        list(prefetch_iterator_values(failing_iterator(), prefetch_chunks=1))
