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
PACKED8_EARLY_FAILURE_STATUS_MASK = 1 | 2 | 2048
PACKED8_SAMPLE_INDEX_STATUS = 1024


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
        _legacy_genotype_mean,
    ) = foreign_outputs

    return build_decoded_packed8_batch(
        packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
        raw_dosage_sums=raw_dosage_sums,
        raw_dosage_square_sums=raw_dosage_square_sums,
        zero_counts=zero_counts,
        homozygous_alternate_counts=homozygous_alternate_counts,
        statuses=statuses,
        selected_sample_count=selected_sample_count,
        retain_imputed_dosage_square_sum=retain_imputed_dosage_square_sum,
        collect_sparse_candidate_mask=collect_sparse_candidate_mask,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "logical_variant_count",
        "selected_sample_count",
        "selection_start",
        "retain_imputed_dosage_square_sum",
        "collect_sparse_candidate_mask",
    ),
)
def select_shared_packed8_batch(
    packed_probability_pairs_by_variant: jax.Array,
    source_statuses: jax.Array,
    selected_sample_indices: jax.Array,
    *,
    logical_variant_count: int,
    selected_sample_count: int,
    selection_start: int,
    retain_imputed_dosage_square_sum: bool,
    collect_sparse_candidate_mask: bool,
) -> DecodedPacked8DeflateBatch:
    """Select a group from an immutable, identity-decoded source batch.

    The source must come from the trusted packed8 decoder with identity sample
    selection. Its source-wide row validation and statuses remain authoritative;
    this helper adds only selection-index failures. No source buffer is donated.

    Args:
        packed_probability_pairs_by_variant: Validated full-source pairs, including compute tails.
        source_statuses: Per-row status bits from identity decoding.
        selected_sample_indices: Ordered source indices, or an empty contiguous operand.
        logical_variant_count: Number of source rows preceding the compute tail.
        selected_sample_count: Number of samples in the group.
        selection_start: Contiguous source offset, or ``-1`` for indexed selection.
        retain_imputed_dosage_square_sum: Whether association needs floating-point square sums.
        collect_sparse_candidate_mask: Whether association needs sparse-candidate decisions.

    Returns:
        Selected packed8 operands and exact group statistics.

    Raises:
        ValueError: If the static geometry or operand dtypes violate the decoder contract.

    """
    if (
        packed_probability_pairs_by_variant.ndim != 3
        or packed_probability_pairs_by_variant.shape[2] != 2
        or packed_probability_pairs_by_variant.dtype != jnp.uint8
    ):
        raise ValueError("Shared packed8 probabilities must be uint8 [compute_variants, source_samples, 2].")
    compute_variant_count, source_sample_count, _ = packed_probability_pairs_by_variant.shape
    if source_sample_count <= 0 or not 0 < logical_variant_count <= compute_variant_count:
        raise ValueError("Shared packed8 source geometry must contain positive samples and logical variants.")
    if source_statuses.shape != (compute_variant_count,) or source_statuses.dtype != jnp.uint32:
        raise ValueError("Shared packed8 statuses must be one uint32 value per compute variant.")
    if selected_sample_count <= 0 or selected_sample_indices.ndim != 1 or selected_sample_indices.dtype != jnp.uint32:
        raise ValueError("Shared packed8 selection requires positive sample count and a uint32 index vector.")

    if selection_start >= 0:
        if selected_sample_indices.size != 0 or selected_sample_count > source_sample_count - selection_start:
            raise ValueError("Contiguous shared packed8 selection exceeds the source or includes index operands.")
        selected_pairs = packed_probability_pairs_by_variant[
            :, selection_start : selection_start + selected_sample_count, :
        ]
        selection_status = jnp.asarray(0, dtype=jnp.uint32)
    elif selection_start == -1:
        if selected_sample_indices.size != selected_sample_count:
            raise ValueError("Indexed shared packed8 selection requires one source index per selected sample.")
        valid_indices = selected_sample_indices < source_sample_count
        safe_indices = jnp.where(valid_indices, selected_sample_indices, jnp.uint32(0))
        gathered_pairs = jnp.take(packed_probability_pairs_by_variant, safe_indices, axis=1, mode="clip")
        selected_pairs = jnp.where(
            valid_indices[None, :, None],
            gathered_pairs,
            jnp.asarray([255, 0], dtype=jnp.uint8),
        )
        selection_status = jnp.where(
            jnp.all(valid_indices),
            jnp.uint32(0),
            jnp.uint32(PACKED8_SAMPLE_INDEX_STATUS),
        )
    else:
        raise ValueError("Shared packed8 selection_start must be -1 or nonnegative.")

    logical_rows = jnp.arange(compute_variant_count, dtype=jnp.uint64) < logical_variant_count
    # These gates precede row reads and selected-index validation in native code.
    # Neutral pairs alone cannot distinguish a gated row from genuine zero calls.
    selected_rows = logical_rows & ((source_statuses & PACKED8_EARLY_FAILURE_STATUS_MASK) == 0)
    selected_pairs = jnp.where(selected_rows[:, None, None], selected_pairs, jnp.asarray([255, 0], dtype=jnp.uint8))
    statuses = jnp.where(logical_rows, source_statuses | jnp.where(selected_rows, selection_status, jnp.uint32(0)), 0)

    # Native subtraction wraps in uint32 even on invalid probability pairs;
    # only then does it widen to uint64 for products and reductions.
    raw_dosage = (
        jnp.uint32(genotype.PACKED8_DIPLOID_NUMERATOR)
        - jnp.uint32(2) * selected_pairs[:, :, 0].astype(jnp.uint32)
        - selected_pairs[:, :, 1].astype(jnp.uint32)
    ).astype(jnp.uint64)
    raw_dosage_sums = jnp.sum(raw_dosage, axis=1, dtype=jnp.uint64)
    raw_dosage_square_sums = jnp.sum(raw_dosage * raw_dosage, axis=1, dtype=jnp.uint64)
    zero_counts = jnp.where(selected_rows, jnp.sum(raw_dosage == 0, axis=1, dtype=jnp.uint32), jnp.uint32(0))
    homozygous_alternate_counts = jnp.sum(raw_dosage >= 383, axis=1, dtype=jnp.uint32)
    return build_decoded_packed8_batch(
        packed_probability_pairs_by_variant=selected_pairs,
        raw_dosage_sums=raw_dosage_sums,
        raw_dosage_square_sums=raw_dosage_square_sums,
        zero_counts=zero_counts,
        homozygous_alternate_counts=homozygous_alternate_counts,
        statuses=statuses,
        selected_sample_count=selected_sample_count,
        retain_imputed_dosage_square_sum=retain_imputed_dosage_square_sum,
        collect_sparse_candidate_mask=collect_sparse_candidate_mask,
    )


def build_decoded_packed8_batch(
    *,
    packed_probability_pairs_by_variant: jax.Array,
    raw_dosage_sums: jax.Array,
    raw_dosage_square_sums: jax.Array,
    zero_counts: jax.Array,
    homozygous_alternate_counts: jax.Array,
    statuses: jax.Array,
    selected_sample_count: int,
    retain_imputed_dosage_square_sum: bool,
    collect_sparse_candidate_mask: bool,
) -> DecodedPacked8DeflateBatch:
    """Derive identical compute moments and sparse decisions from exact totals."""
    # Keep the existing FFI shape while deriving both compute moments from its
    # exact integers. Narrowing a raw total first biases means and square sums
    # in large cohorts; the native mean predates this precision policy.
    dosage_sums = jnp.asarray(raw_dosage_sums, dtype=jnp.float64) / genotype.EIGHT_BIT_PROBABILITY_DENOMINATOR
    genotype_mean = jnp.asarray(dosage_sums / selected_sample_count, dtype=jnp.float32)

    if retain_imputed_dosage_square_sum:
        imputed_dosage_square_sum = jnp.asarray(
            jnp.asarray(raw_dosage_square_sums, dtype=jnp.float64)
            / (genotype.EIGHT_BIT_PROBABILITY_DENOMINATOR * genotype.EIGHT_BIT_PROBABILITY_DENOMINATOR),
            dtype=jnp.float32,
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
