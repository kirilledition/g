"""Exact selection contracts for immutable identity-decoded packed8 sources."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from g.compute.common import compressed_genotype


@dataclass(frozen=True)
class SelectedReference:
    """Independent native-contract reference values."""

    pairs: npt.NDArray[np.uint8]
    sums: npt.NDArray[np.uint64]
    square_sums: npt.NDArray[np.uint64]
    statuses: npt.NDArray[np.uint32]
    means: npt.NDArray[np.float32]
    floating_square_sums: npt.NDArray[np.float32]
    sparse: npt.NDArray[np.bool_]


def select_reference(
    source: npt.NDArray[np.uint8],
    statuses: npt.NDArray[np.uint32],
    indices: list[int],
    logical_count: int,
) -> SelectedReference:
    """Follow native row gates and unsigned integer arithmetic using Python ints."""
    count = len(indices)
    pairs = np.broadcast_to(np.asarray([255, 0], dtype=np.uint8), (source.shape[0], count, 2)).copy()
    sums = np.zeros(source.shape[0], dtype=np.uint64)
    square_sums = sums.copy()
    selected_statuses = np.zeros(source.shape[0], dtype=np.uint32)
    sparse = np.zeros(source.shape[0], dtype=np.bool_)
    for row in range(logical_count):
        status = int(statuses[row])
        selected_statuses[row] = status
        if status & (1 | 2 | 2048):
            continue
        dosages: list[int] = []
        for position, sample in enumerate(indices):
            if sample < source.shape[1]:
                pairs[row, position] = source[row, sample]
            else:
                status |= 1024
            first, second = (int(value) for value in pairs[row, position])
            dosages.append((510 - 2 * first - second) % (1 << 32))
        dosage_sum = sum(dosages) % (1 << 64)
        dosage_square_sum = sum(value * value for value in dosages) % (1 << 64)
        zero_count = sum(value == 0 for value in dosages) % (1 << 32)
        alternate_count = sum(value >= 383 for value in dosages) % (1 << 32)
        regenie_zero_count = alternate_count if dosage_sum > 255 * count else zero_count
        minor_count = min(dosage_sum, (510 * count - dosage_sum) % (1 << 64))
        sparse[row] = regenie_zero_count * 2 >= count and minor_count < 50 * 255
        sums[row] = dosage_sum
        square_sums[row] = dosage_square_sum
        selected_statuses[row] = status
    return SelectedReference(
        pairs=pairs,
        sums=sums,
        square_sums=square_sums,
        statuses=selected_statuses,
        means=(sums.astype(np.float64) / 255 / count).astype(np.float32),
        floating_square_sums=(square_sums.astype(np.float64) / (255 * 255)).astype(np.float32),
        sparse=sparse,
    )


@pytest.mark.parametrize(
    ("selection_start", "selected_count", "indices"),
    [
        (0, 5, []),
        (1, 3, []),
        (-1, 4, [4, 1, 4, 2]),
        (-1, 7, [4, 1, 4, 2, 999, (1 << 32) - 1, 3]),
    ],
)
@pytest.mark.parametrize(
    ("retain_square_sum", "collect_sparse"), [(False, False), (True, False), (False, True), (True, True)]
)
def test_shared_selection_preserves_native_statuses_statistics_and_source(
    selection_start: int,
    selected_count: int,
    indices: list[int],
    *,
    retain_square_sum: bool,
    collect_sparse: bool,
) -> None:
    """Cover source-wide errors, early gates, invalid indices, duplicates and tails."""
    source = np.tile(np.asarray([[255, 0], [0, 255], [0, 0], [127, 63], [84, 85]], dtype=np.uint8), (9, 1, 1))
    source[0] = [255, 0]
    source[5, 0] = [255, 255]
    source[7] = [0, 0]
    statuses = np.asarray([0, 1, 2, 2048, 4 | 512, 256, 512, 0, 0], dtype=np.uint32)
    expected_indices = (
        list(range(selection_start, selection_start + selected_count)) if selection_start >= 0 else indices
    )
    expected = select_reference(source, statuses, expected_indices, 8)
    device_source = jnp.asarray(source)
    device_statuses = jnp.asarray(statuses)
    observed = compressed_genotype.select_shared_packed8_batch(
        device_source,
        device_statuses,
        jnp.asarray(indices, dtype=jnp.uint32),
        logical_variant_count=8,
        selected_sample_count=selected_count,
        selection_start=selection_start,
        retain_imputed_dosage_square_sum=retain_square_sum,
        collect_sparse_candidate_mask=collect_sparse,
    )
    jax.block_until_ready(observed)
    np.testing.assert_array_equal(np.asarray(observed.packed_probability_pairs_by_variant), expected.pairs)
    np.testing.assert_array_equal(np.asarray(observed.raw_packed8_statistics.dosage_sums), expected.sums)
    np.testing.assert_array_equal(np.asarray(observed.raw_packed8_statistics.dosage_square_sums), expected.square_sums)
    np.testing.assert_array_equal(np.asarray(observed.raw_packed8_statistics.statuses), expected.statuses)
    np.testing.assert_array_equal(np.asarray(observed.genotype_mean).view(np.uint32), expected.means.view(np.uint32))
    assert observed.raw_packed8_statistics.selected_sample_count == selected_count
    assert observed.raw_packed8_statistics.dosage_sums.dtype == jnp.uint64
    assert observed.raw_packed8_statistics.dosage_square_sums.dtype == jnp.uint64
    assert observed.raw_packed8_statistics.statuses.dtype == jnp.uint32
    if retain_square_sum:
        assert observed.imputed_dosage_square_sum is not None
        np.testing.assert_array_equal(
            np.asarray(observed.imputed_dosage_square_sum).view(np.uint32),
            expected.floating_square_sums.view(np.uint32),
        )
    else:
        assert observed.imputed_dosage_square_sum is None
    if collect_sparse:
        assert observed.sparse_candidate_mask is not None
        np.testing.assert_array_equal(np.asarray(observed.sparse_candidate_mask), expected.sparse)
    else:
        assert observed.sparse_candidate_mask is None
    np.testing.assert_array_equal(np.asarray(device_source), source)
    np.testing.assert_array_equal(np.asarray(device_statuses), statuses)


def test_shared_selection_large_cohort_preserves_rare_and_fractional_moments() -> None:
    """Accumulate large integer moments before a single final float32 narrowing."""
    sample_count = 500_000
    source = np.full((3, sample_count, 2), [255, 0], dtype=np.uint8)
    source[0, 0] = [0, 255]
    source[1] = [0, 0]
    source[1, 0] = [0, 255]
    source[2] = [200, 0]
    observed = compressed_genotype.select_shared_packed8_batch(
        jnp.asarray(source),
        jnp.zeros(3, dtype=jnp.uint32),
        jnp.zeros(0, dtype=jnp.uint32),
        logical_variant_count=3,
        selected_sample_count=sample_count,
        selection_start=0,
        retain_imputed_dosage_square_sum=True,
        collect_sparse_candidate_mask=False,
    )
    np.testing.assert_array_equal(
        np.asarray(observed.genotype_mean),
        np.asarray([1 / sample_count, 2 - 1 / sample_count, 110 / 255], dtype=np.float32),
    )
    assert observed.imputed_dosage_square_sum is not None
    np.testing.assert_array_equal(
        np.asarray(observed.imputed_dosage_square_sum),
        np.asarray([1, 4 * (sample_count - 1) + 1, sample_count * (110 / 255) ** 2], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("selection_start", "selected_count", "indices"), [(4, 2, []), (0, 3, [0]), (-2, 3, []), (-1, 3, [0]), (0, 0, [])]
)
def test_shared_selection_rejects_invalid_static_selection(
    selection_start: int,
    selected_count: int,
    indices: list[int],
) -> None:
    """Reject invalid geometry before dispatching a gather."""
    with pytest.raises(ValueError):
        compressed_genotype.select_shared_packed8_batch(
            jnp.zeros((2, 5, 2), dtype=jnp.uint8),
            jnp.zeros(2, dtype=jnp.uint32),
            jnp.asarray(indices, dtype=jnp.uint32),
            logical_variant_count=2,
            selected_sample_count=selected_count,
            selection_start=selection_start,
            retain_imputed_dosage_square_sum=False,
            collect_sparse_candidate_mask=False,
        )
