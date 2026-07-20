"""CPU-safe correctness tests for compressed packed8 device decoding."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

import tests.numerical
from g.compute.common import compressed_genotype

type Packed8ForeignOutputs = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]


class Packed8ForeignCall(typing.Protocol):
    """Callable returned by the JAX FFI constructor in these tests."""

    def __call__(
        self,
        compressed_slab: jax.Array,
        compressed_metadata: jax.Array,
        selected_sample_indices: jax.Array,
        *,
        source_sample_count: int,
        selection_start: int,
    ) -> Packed8ForeignOutputs:
        """Return deterministic native outputs for one abstract FFI call."""


def build_foreign_outputs() -> Packed8ForeignOutputs:
    """Build exact output values spanning sparse-mask decision branches."""
    packed_probability_pairs = jnp.zeros((4, 100, 2), dtype=jnp.uint8)
    packed_probability_pairs = packed_probability_pairs.at[:, :2, :].set(
        jnp.asarray(
            [
                [[255, 0], [0, 255]],
                [[0, 0], [255, 0]],
                [[64, 127], [127, 64]],
                [[0, 255], [255, 0]],
            ],
            dtype=jnp.uint8,
        )
    )
    return (
        packed_probability_pairs,
        jnp.asarray([1_000, 50_000, 1_000, 12_750], dtype=jnp.uint64),
        jnp.asarray([65_025, 130_050, 32_512, 0], dtype=jnp.uint64),
        jnp.asarray([60, 2, 40, 60], dtype=jnp.uint32),
        jnp.asarray([2, 60, 1, 1], dtype=jnp.uint32),
        jnp.asarray([0, 1, 2, 7], dtype=jnp.uint32),
        jnp.asarray([0.25, 1.75, 0.5, 0.5], dtype=jnp.float32),
    )


def build_compressed_slab() -> npt.NDArray[np.uint8]:
    """Build one deterministic slab spanning four logical members."""
    return np.arange(12, dtype=np.uint8)


def build_compressed_metadata() -> npt.NDArray[np.uint32]:
    """Build production-shaped rows of offset, size, and Adler checksum."""
    return np.asarray(
        [
            [0, 3, 11],
            [3, 2, 12],
            [5, 4, 13],
            [9, 3, 14],
        ],
        dtype=np.uint32,
    )


def install_fake_packed8_ffi(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_selected_sample_indices: npt.NDArray[np.uint32],
    expected_selection_start: int,
) -> None:
    """Replace only the external JAX FFI boundary with deterministic outputs."""
    expected_outputs = build_foreign_outputs()

    def fake_ffi_call(
        target_name: str,
        result_shape_dtypes: tuple[jax.ShapeDtypeStruct, ...],
    ) -> Packed8ForeignCall:
        assert target_name == compressed_genotype.PACKED8_DEFLATE_FFI_TARGET
        assert tuple(output.shape for output in result_shape_dtypes) == (
            (4, 100, 2),
            (4,),
            (4,),
            (4,),
            (4,),
            (4,),
            (4,),
        )
        assert tuple(output.dtype for output in result_shape_dtypes) == (
            np.dtype(np.uint8),
            np.dtype(np.uint64),
            np.dtype(np.uint64),
            np.dtype(np.uint32),
            np.dtype(np.uint32),
            np.dtype(np.uint32),
            np.dtype(np.float32),
        )

        def foreign_call(
            compressed_slab: jax.Array,
            compressed_metadata: jax.Array,
            selected_sample_indices: jax.Array,
            *,
            source_sample_count: int,
            selection_start: int,
        ) -> Packed8ForeignOutputs:
            assert compressed_slab.shape == (12,)
            assert compressed_slab.dtype == jnp.uint8
            assert compressed_metadata.shape == (4, 3)
            assert compressed_metadata.dtype == jnp.uint32
            assert selected_sample_indices.shape == expected_selected_sample_indices.shape
            assert selected_sample_indices.dtype == jnp.uint32
            np.testing.assert_array_equal(np.asarray(compressed_slab), build_compressed_slab())
            np.testing.assert_array_equal(
                np.asarray(compressed_metadata),
                build_compressed_metadata(),
            )
            np.testing.assert_array_equal(
                np.asarray(selected_sample_indices),
                expected_selected_sample_indices,
            )
            assert source_sample_count == 120
            assert selection_start == expected_selection_start
            return expected_outputs

        return foreign_call

    monkeypatch.setattr(jax.ffi, "ffi_call", fake_ffi_call)


@pytest.mark.parametrize(
    ("retain_square_sum", "collect_sparse_mask"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_decode_packed8_deflate_batch_derives_optional_outputs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    retain_square_sum: bool,
    collect_sparse_mask: bool,
) -> None:
    """Derive floating moments and exact REGENIE sparse decisions from native counts."""
    install_fake_packed8_ffi(
        monkeypatch,
        expected_selected_sample_indices=np.arange(100, dtype=np.uint32),
        expected_selection_start=-1,
    )

    with jax.disable_jit():
        observed = compressed_genotype.decode_packed8_deflate_batch(
            compressed_slab=jnp.asarray(build_compressed_slab()),
            compressed_metadata=jnp.asarray(build_compressed_metadata()),
            selected_sample_indices=jnp.arange(100, dtype=jnp.uint32),
            source_sample_count=120,
            selected_sample_count=100,
            selection_start=-1,
            compute_variant_count=4,
            retain_imputed_dosage_square_sum=retain_square_sum,
            collect_sparse_candidate_mask=collect_sparse_mask,
        )

    expected_outputs = build_foreign_outputs()
    assert observed.packed_probability_pairs_by_variant.shape == (4, 100, 2)
    assert observed.packed_probability_pairs_by_variant.dtype == jnp.uint8
    assert observed.genotype_mean.shape == (4,)
    assert observed.genotype_mean.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(observed.packed_probability_pairs_by_variant),
        np.asarray(expected_outputs[0]),
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.genotype_mean,
        np.asarray([0.25, 1.75, 0.5, 0.5], dtype=np.float32),
        1.0e-7,
    )
    np.testing.assert_array_equal(
        np.asarray(observed.raw_packed8_statistics.dosage_sums),
        np.asarray([1_000, 50_000, 1_000, 12_750], dtype=np.uint64),
    )
    assert observed.raw_packed8_statistics.dosage_sums.dtype == jnp.uint64
    np.testing.assert_array_equal(
        np.asarray(observed.raw_packed8_statistics.dosage_square_sums),
        np.asarray([65_025, 130_050, 32_512, 0], dtype=np.uint64),
    )
    assert observed.raw_packed8_statistics.dosage_square_sums.dtype == jnp.uint64
    np.testing.assert_array_equal(
        np.asarray(observed.raw_packed8_statistics.statuses),
        np.asarray([0, 1, 2, 7], dtype=np.uint32),
    )
    assert observed.raw_packed8_statistics.statuses.dtype == jnp.uint32
    assert observed.raw_packed8_statistics.selected_sample_count == 100

    if retain_square_sum:
        assert observed.imputed_dosage_square_sum is not None
        assert observed.imputed_dosage_square_sum.dtype == jnp.float32
        tests.numerical.assert_absolute_difference_less_than(
            observed.imputed_dosage_square_sum,
            np.asarray([1.0, 2.0, 32_512.0 / 65_025.0, 0.0], dtype=np.float32),
            1.0e-7,
        )
    else:
        assert observed.imputed_dosage_square_sum is None

    if collect_sparse_mask:
        assert observed.sparse_candidate_mask is not None
        assert observed.sparse_candidate_mask.dtype == jnp.bool_
        np.testing.assert_array_equal(
            np.asarray(observed.sparse_candidate_mask),
            np.asarray([True, True, False, False]),
        )
    else:
        assert observed.sparse_candidate_mask is None
