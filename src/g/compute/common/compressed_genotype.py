"""Device decoding for trusted raw-DEFLATE packed8 BGEN batches."""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from g.compute.common import genotype

PACKED8_DEFLATE_FFI_TARGET = "g.bgen.packed8_deflate.v1"
RARE_SPARSE_FIRTH_MINOR_ALLELE_COUNT = 50


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class Packed8RawStatistics[SumArray, StatusArray]:
    """Exact packed8 integer summaries across device and host residency.

    Attributes:
        dosage_sums: Dosage sums in 1/255 units.
        dosage_square_sums: Dosage square sums in 1/65025 units.
        statuses: Per-variant native decode status bits.
        selected_sample_count: Observation count shared by every decoded row.

    """

    dosage_sums: SumArray
    dosage_square_sums: SumArray
    statuses: StatusArray
    selected_sample_count: int = dataclasses.field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class DecodedPacked8DeflateBatch:
    """Decoded packed8 operands and exact integer summaries.

    Attributes:
        packed_probability_pairs_by_variant: Variant-major probability pairs.
        genotype_mean: Per-variant mean dosage.
        imputed_dosage_square_sum: Per-variant dosage square sums when requested.
        sparse_candidate_mask: Exact REGENIE sparse-candidate decisions when requested.
        raw_packed8_statistics: Exact integer summaries retained for materialization.

    """

    packed_probability_pairs_by_variant: jax.Array
    genotype_mean: jax.Array
    imputed_dosage_square_sum: jax.Array | None
    sparse_candidate_mask: jax.Array | None
    raw_packed8_statistics: Packed8RawStatistics[jax.Array, jax.Array]


@functools.partial(
    jax.jit,
    static_argnames=(
        "source_sample_count",
        "selected_sample_count",
        "selection_start",
        "compute_variant_count",
        "retain_imputed_dosage_square_sum",
        "collect_sparse_candidate_mask",
    ),
)
def decode_packed8_deflate_batch(
    compressed_slab: jax.Array,
    compressed_metadata: jax.Array,
    selected_sample_indices: jax.Array,
    *,
    source_sample_count: int,
    selected_sample_count: int,
    selection_start: int,
    compute_variant_count: int,
    retain_imputed_dosage_square_sum: bool,
    collect_sparse_candidate_mask: bool,
) -> DecodedPacked8DeflateBatch:
    """Decode one fixed-geometry compressed batch with the CUDA FFI target.

    Args:
        compressed_slab: Aligned raw-DEFLATE members in one byte slab.
        compressed_metadata: Logical member offsets, sizes, and Adler checksums.
        selected_sample_indices: Indexed selection, or an empty contiguous operand.
        source_sample_count: Sample count encoded in each source BGEN row.
        selected_sample_count: Number of output samples.
        selection_start: Contiguous source offset, or ``-1`` for indexed selection.
        compute_variant_count: Padded variant count used by association kernels.
        retain_imputed_dosage_square_sum: Whether association needs floating-point square sums.
        collect_sparse_candidate_mask: Whether association needs sparse-candidate decisions.

    Returns:
        Decoded packed8 operands and summaries for association and materialization.

    """
    foreign_outputs = jax.ffi.ffi_call(
        PACKED8_DEFLATE_FFI_TARGET,
        (
            jax.ShapeDtypeStruct(
                (compute_variant_count, selected_sample_count, 2),
                np.uint8,
            ),
            jax.ShapeDtypeStruct((compute_variant_count,), np.uint64),
            jax.ShapeDtypeStruct((compute_variant_count,), np.uint64),
            jax.ShapeDtypeStruct((compute_variant_count,), np.uint32),
            jax.ShapeDtypeStruct((compute_variant_count,), np.uint32),
            jax.ShapeDtypeStruct((compute_variant_count,), np.uint32),
            jax.ShapeDtypeStruct((compute_variant_count,), np.float32),
        ),
    )(
        compressed_slab,
        compressed_metadata,
        selected_sample_indices,
        source_sample_count=source_sample_count,
        selection_start=selection_start,
    )
    (
        packed_probability_pairs_by_variant,
        raw_dosage_sums,
        raw_dosage_square_sums,
        zero_counts,
        homozygous_alternate_counts,
        statuses,
        genotype_mean,
    ) = foreign_outputs

    if retain_imputed_dosage_square_sum:
        packed8_probability_square_scale = jnp.asarray(
            1.0 / (genotype.EIGHT_BIT_PROBABILITY_DENOMINATOR * genotype.EIGHT_BIT_PROBABILITY_DENOMINATOR),
            dtype=jnp.float32,
        )
        imputed_dosage_square_sum = (
            jnp.asarray(raw_dosage_square_sums, dtype=jnp.float32) * packed8_probability_square_scale
        )
    else:
        imputed_dosage_square_sum = None

    if collect_sparse_candidate_mask:
        selected_sample_count_unsigned = jnp.asarray(selected_sample_count, dtype=jnp.uint64)
        allele_flip_mask = raw_dosage_sums > (
            genotype.EIGHT_BIT_PROBABILITY_DENOMINATOR * selected_sample_count_unsigned
        )
        regenie_zero_counts = jnp.where(
            allele_flip_mask,
            homozygous_alternate_counts,
            zero_counts,
        )
        dense_zero_mask = jnp.asarray(regenie_zero_counts, dtype=jnp.uint64) * 2 >= selected_sample_count_unsigned
        reference_allele_raw_counts = (
            genotype.PACKED8_DIPLOID_NUMERATOR * selected_sample_count_unsigned - raw_dosage_sums
        )
        minor_allele_raw_counts = jnp.minimum(
            raw_dosage_sums,
            reference_allele_raw_counts,
        )
        sparse_candidate_mask = dense_zero_mask & (
            minor_allele_raw_counts < RARE_SPARSE_FIRTH_MINOR_ALLELE_COUNT * genotype.EIGHT_BIT_PROBABILITY_DENOMINATOR
        )
    else:
        sparse_candidate_mask = None
    return DecodedPacked8DeflateBatch(
        packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
        genotype_mean=genotype_mean,
        imputed_dosage_square_sum=imputed_dosage_square_sum,
        sparse_candidate_mask=sparse_candidate_mask,
        raw_packed8_statistics=Packed8RawStatistics(
            dosage_sums=raw_dosage_sums,
            dosage_square_sums=raw_dosage_square_sums,
            statuses=statuses,
            selected_sample_count=selected_sample_count,
        ),
    )
