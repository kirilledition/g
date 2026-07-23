from __future__ import annotations

import importlib
import math
import os
import struct
import subprocess
import sys
import typing
import zlib
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

import tests.numerical
from g.compute import cuda_ffi
from g.compute.common import compressed_genotype
from g.compute.regenie2_binary.firth import scalar_approx as regenie2_binary_firth_scalar_approx
from g.compute.regenie2_binary.firth import types as regenie2_binary_firth_types

RUN_CUDA_NATIVE_TESTS = os.environ.get("GWAS_ENGINE_RUN_CUDA_NATIVE_TESTS") == "1"
FIRTH_LANE_CAPACITY = 1_024
FIRTH_ACTIVE_PREFIXES = (0, 16, 400, 900, 1_024)
FIRTH_SAMPLE_COUNTS = (31, 32, 33, 255, 256, 257)
PACKED8_DESCRIPTOR_STATUS = 0x0000_0800
PACKED8_ADLER_STATUS = 0x0000_0200
PACKED8_SAMPLE_INDEX_STATUS = 0x0000_0400
PACKED8_BOUNDARY_GEOMETRIES = (
    (31, 255, 256),
    (32, 256, 256),
    (33, 257, 512),
    (255, 255, 256),
    (256, 256, 256),
    (257, 257, 512),
)
PACKED8_PRODUCTION_GEOMETRIES = (
    (31, 16_384, 16_384),
    (257, 257, 16_384),
)

pytestmark = [
    pytest.mark.cuda_native,
    pytest.mark.skipif(
        not RUN_CUDA_NATIVE_TESTS,
        reason="set GWAS_ENGINE_RUN_CUDA_NATIVE_TESTS=1 on the exclusive Landau allocation",
    ),
]


class CudaFfiTestSupport(typing.Protocol):
    def register_firth_components_ffi(self) -> str: ...

    def register_packed8_deflate_ffi(self) -> str: ...

    def register_unqualified_packed8_deflate_ffi_for_test(self) -> str: ...

    def nvcomp_input_alignment(self) -> int: ...


@dataclass(frozen=True)
class FirthComponentFixture:
    phenotype: npt.NDArray[np.float64]
    genotype: npt.NDArray[np.float64]
    offset: npt.NDArray[np.float64]
    active_sample_mask: npt.NDArray[np.bool_]
    non_active_deviance: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    minimum_variance: npt.NDArray[np.float64]


@dataclass(frozen=True)
class RawDeflateFixture:
    compressed_slab: npt.NDArray[np.uint8]
    compressed_metadata: npt.NDArray[np.uint32]
    expected_probabilities: npt.NDArray[np.uint8]
    source_sample_count: int
    logical_variant_count: int


@dataclass(frozen=True)
class Packed8Expected:
    probabilities: npt.NDArray[np.uint8]
    dosage_sums: npt.NDArray[np.uint64]
    dosage_square_sums: npt.NDArray[np.uint64]
    genotype_means: npt.NDArray[np.float32]
    sparse_candidate_mask: npt.NDArray[np.bool_]


def load_cuda_test_support() -> CudaFfiTestSupport:
    try:
        module = importlib.import_module("g._core._testing")
    except ModuleNotFoundError:
        pytest.fail("CUDA native tests require an extension built with private-test-support.")
    return typing.cast("CudaFfiTestSupport", module)


@pytest.fixture(scope="module", autouse=True)
def configure_cuda_runtime() -> None:
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - Native Firth operands require f64.
    jax.config.update("jax_platforms", "cuda")


@pytest.fixture(scope="module")
def cuda_test_support() -> CudaFfiTestSupport:
    return load_cuda_test_support()


@jax.jit
def compute_portable_firth_components(
    phenotype: jax.Array,
    genotype: jax.Array,
    offset: jax.Array,
    active_sample_mask: jax.Array,
    non_active_deviance: jax.Array,
    beta: jax.Array,
    minimum_variance: jax.Array,
) -> regenie2_binary_firth_types.ScalarFirthComponents:
    def compute_lane(
        phenotype_vector: jax.Array,
        genotype_vector: jax.Array,
        offset_vector: jax.Array,
        active_sample_mask: jax.Array,
        non_active_deviance: jax.Array,
        beta: jax.Array,
        minimum_variance: jax.Array,
    ) -> regenie2_binary_firth_types.ScalarFirthComponents:
        return regenie2_binary_firth_scalar_approx.compute_scalar_firth_components_with_minimum_variance(
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            offset_vector=offset_vector,
            active_sample_mask=active_sample_mask,
            non_active_deviance=non_active_deviance,
            beta=beta,
            minimum_variance=minimum_variance,
            use_cuda_components=False,
        )

    return jax.vmap(compute_lane)(
        phenotype,
        genotype,
        offset,
        active_sample_mask,
        non_active_deviance,
        beta,
        minimum_variance,
    )


@jax.jit
def compute_raw_cuda_firth_components(
    phenotype: jax.Array,
    genotype: jax.Array,
    offset: jax.Array,
    active_sample_mask: jax.Array,
    non_active_deviance: jax.Array,
    beta: jax.Array,
    minimum_variance: jax.Array,
) -> regenie2_binary_firth_types.ScalarFirthComponents:
    def compute_lane(
        phenotype_vector: jax.Array,
        genotype_vector: jax.Array,
        offset_vector: jax.Array,
        active_sample_mask: jax.Array,
        non_active_deviance: jax.Array,
        beta: jax.Array,
        minimum_variance: jax.Array,
    ) -> regenie2_binary_firth_types.ScalarFirthComponents:
        return regenie2_binary_firth_scalar_approx.compute_scalar_firth_components_with_minimum_variance(
            phenotype_vector=phenotype_vector,
            genotype_vector=genotype_vector,
            offset_vector=offset_vector,
            active_sample_mask=active_sample_mask,
            non_active_deviance=non_active_deviance,
            beta=beta,
            minimum_variance=minimum_variance,
            use_cuda_components=True,
        )

    return jax.vmap(compute_lane)(
        phenotype,
        genotype,
        offset,
        active_sample_mask,
        non_active_deviance,
        beta,
        minimum_variance,
    )


def build_firth_component_fixture(sample_count: int, active_prefix: int) -> FirthComponentFixture:
    sample_indices = np.arange(sample_count, dtype=np.int64)[None, :]
    lane_indices = np.arange(FIRTH_LANE_CAPACITY, dtype=np.int64)[:, None]
    phenotype = (((lane_indices * 3 + sample_indices * 5 + 1) % 11) < 5).astype(np.float64)
    genotype = (((lane_indices * 7 + sample_indices * 13 + 3) % 9) / 4.0).astype(np.float64)
    offset = (((lane_indices * 11 + sample_indices * 17) % 13) - 6).astype(np.float64) * 0.075
    active_sample_mask = ((lane_indices + sample_indices * 3) % 5 != 0) & (lane_indices < active_prefix)
    non_active_deviance = (np.arange(FIRTH_LANE_CAPACITY, dtype=np.float64) % 17) * 0.125
    beta = ((np.arange(FIRTH_LANE_CAPACITY, dtype=np.float64) % 19) - 9) * 0.05
    minimum_variance = np.full(FIRTH_LANE_CAPACITY, 1.0e-8, dtype=np.float64)

    special_lane_count = min(active_prefix, 16)
    active_sample_mask[:special_lane_count, :] = True
    linear_predictors = np.asarray((-31.0, -30.0, -29.999, 0.0, 29.999, 30.0, 31.0), dtype=np.float64)
    genotype[: linear_predictors.size, :] = 1.0
    offset[: linear_predictors.size, :] = linear_predictors[:, None]
    beta[: linear_predictors.size] = 0.0

    genotype[7, :] = 0.0
    near_threshold_genotype = math.sqrt(4.0e-8 / sample_count)
    genotype[8:10, :] = near_threshold_genotype
    offset[8:10, :] = 0.0
    beta[8:10] = 0.0
    minimum_variance[8] = 1.01e-8
    minimum_variance[9] = 0.99e-8

    genotype[10, 0] = np.nan
    non_active_deviance[11] = np.inf
    phenotype[12, 0] = np.nan
    offset[13, 0] = np.nan
    minimum_variance[14] = np.nan
    active_sample_mask[15, ::3] = False

    return FirthComponentFixture(
        phenotype=phenotype,
        genotype=genotype,
        offset=offset,
        active_sample_mask=active_sample_mask,
        non_active_deviance=non_active_deviance,
        beta=beta,
        minimum_variance=minimum_variance,
    )


def assert_firth_component_parity(
    observed: regenie2_binary_firth_types.ScalarFirthComponents,
    expected: regenie2_binary_firth_types.ScalarFirthComponents,
) -> None:
    tests.numerical.assert_absolute_difference_less_than(
        observed.genotype_information,
        expected.genotype_information,
        5.0e-7,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.score_adjustment,
        expected.score_adjustment,
        5.0e-7,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.penalized_deviance,
        expected.penalized_deviance,
        2.0e-6,
    )
    tests.numerical.assert_absolute_difference_less_than(observed.score, expected.score, 5.0e-7)
    np.testing.assert_array_equal(np.asarray(observed.valid), np.asarray(expected.valid))


def test_native_registration_is_exact_and_idempotent(cuda_test_support: CudaFfiTestSupport) -> None:
    assert cuda_test_support.register_firth_components_ffi() == cuda_ffi.FIRTH_COMPONENTS_FFI_TARGET
    assert cuda_test_support.register_firth_components_ffi() == cuda_ffi.FIRTH_COMPONENTS_FFI_TARGET
    assert cuda_test_support.register_packed8_deflate_ffi() == cuda_ffi.PACKED8_DEFLATE_FFI_TARGET
    assert cuda_test_support.register_packed8_deflate_ffi() == cuda_ffi.PACKED8_DEFLATE_FFI_TARGET


def run_unqualified_packed8_handler_regression() -> None:
    """Assert the native handler rejects execution before device qualification."""
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - Native summary operands require u64.
    jax.config.update("jax_platforms", "cuda")
    target = load_cuda_test_support().register_unqualified_packed8_deflate_ffi_for_test()
    fixture = build_raw_deflate_fixture(source_sample_count=31, logical_variant_count=1)

    @jax.jit
    def invoke_unqualified_handler(
        compressed_slab: jax.Array,
        compressed_metadata: jax.Array,
        selected_sample_indices: jax.Array,
    ) -> typing.Sequence[jax.Array]:
        return jax.ffi.ffi_call(
            target,
            (
                jax.ShapeDtypeStruct((1, fixture.source_sample_count, 2), np.uint8),
                jax.ShapeDtypeStruct((1,), np.uint64),
                jax.ShapeDtypeStruct((1,), np.uint64),
                jax.ShapeDtypeStruct((1,), np.uint32),
                jax.ShapeDtypeStruct((1,), np.uint32),
                jax.ShapeDtypeStruct((1,), np.uint32),
                jax.ShapeDtypeStruct((1,), np.float32),
            ),
        )(
            compressed_slab,
            compressed_metadata,
            selected_sample_indices,
            source_sample_count=fixture.source_sample_count,
            selection_start=0,
        )

    try:
        observed = invoke_unqualified_handler(
            jnp.asarray(fixture.compressed_slab),
            jnp.asarray(fixture.compressed_metadata),
            jnp.asarray(np.empty((0,), dtype=np.uint32)),
        )
        jax.block_until_ready(observed)
    except jax.errors.JaxRuntimeError as error:
        expected_message = "before its execution device was qualified"
        if expected_message not in str(error):
            raise AssertionError(f"Unexpected unqualified-handler error: {error}") from error
    else:
        raise AssertionError("The packed8 handler executed without a qualified CUDA device.")


def test_packed8_handler_rejects_use_before_qualification_in_fresh_process() -> None:
    script = (
        "from tests.test_native_cuda_ffi_handlers import "
        "run_unqualified_packed8_handler_regression; "
        "run_unqualified_packed8_handler_regression()"
    )
    completed_process = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=180,
    )

    assert completed_process.returncode == 0, (
        f"Unqualified packed8 regression subprocess failed.\n"
        f"stdout:\n{completed_process.stdout}\n"
        f"stderr:\n{completed_process.stderr}"
    )


@pytest.mark.parametrize("sample_count", FIRTH_SAMPLE_COUNTS)
@pytest.mark.parametrize("active_prefix", FIRTH_ACTIVE_PREFIXES)
def test_registered_firth_handler_matches_portable_boundary_matrix(
    cuda_test_support: CudaFfiTestSupport,
    sample_count: int,
    active_prefix: int,
) -> None:
    assert cuda_test_support.register_firth_components_ffi() == cuda_ffi.FIRTH_COMPONENTS_FFI_TARGET
    fixture = build_firth_component_fixture(sample_count, active_prefix)

    observed = compute_raw_cuda_firth_components(
        fixture.phenotype,
        fixture.genotype,
        fixture.offset,
        fixture.active_sample_mask,
        fixture.non_active_deviance,
        fixture.beta,
        fixture.minimum_variance,
    )
    expected = compute_portable_firth_components(
        fixture.phenotype,
        fixture.genotype,
        fixture.offset,
        fixture.active_sample_mask,
        fixture.non_active_deviance,
        fixture.beta,
        fixture.minimum_variance,
    )
    jax.block_until_ready((observed, expected))

    assert_firth_component_parity(observed, expected)


@pytest.mark.parametrize(
    ("malformation", "expected_message"),
    (
        ("phenotype_dtype", "matching batch prefixes, f64 values, and boolean masks"),
        ("genotype_shape", "matching batch prefixes, f64 values, and boolean masks"),
        ("mask_dtype", "matching batch prefixes, f64 values, and boolean masks"),
        ("zero_samples", "nonempty final dimension"),
    ),
)
def test_registered_firth_handler_rejects_malformed_operands(
    cuda_test_support: CudaFfiTestSupport,
    malformation: str,
    expected_message: str,
) -> None:
    assert cuda_test_support.register_firth_components_ffi() == cuda_ffi.FIRTH_COMPONENTS_FFI_TARGET
    fixture = build_firth_component_fixture(33, 16)
    if malformation == "phenotype_dtype":
        fixture = FirthComponentFixture(
            phenotype=typing.cast("npt.NDArray[np.float64]", fixture.phenotype.astype(np.float32)),
            genotype=fixture.genotype,
            offset=fixture.offset,
            active_sample_mask=fixture.active_sample_mask,
            non_active_deviance=fixture.non_active_deviance,
            beta=fixture.beta,
            minimum_variance=fixture.minimum_variance,
        )
    elif malformation == "genotype_shape":
        fixture = FirthComponentFixture(
            phenotype=fixture.phenotype,
            genotype=fixture.genotype[:, :-1],
            offset=fixture.offset,
            active_sample_mask=fixture.active_sample_mask,
            non_active_deviance=fixture.non_active_deviance,
            beta=fixture.beta,
            minimum_variance=fixture.minimum_variance,
        )
    elif malformation == "mask_dtype":
        fixture = FirthComponentFixture(
            phenotype=fixture.phenotype,
            genotype=fixture.genotype,
            offset=fixture.offset,
            active_sample_mask=typing.cast(
                "npt.NDArray[np.bool_]",
                fixture.active_sample_mask.astype(np.uint8),
            ),
            non_active_deviance=fixture.non_active_deviance,
            beta=fixture.beta,
            minimum_variance=fixture.minimum_variance,
        )
    elif malformation == "zero_samples":
        fixture = FirthComponentFixture(
            phenotype=fixture.phenotype[:, :0],
            genotype=fixture.genotype[:, :0],
            offset=fixture.offset[:, :0],
            active_sample_mask=fixture.active_sample_mask[:, :0],
            non_active_deviance=fixture.non_active_deviance,
            beta=fixture.beta,
            minimum_variance=fixture.minimum_variance,
        )
    with pytest.raises(jax.errors.JaxRuntimeError, match=expected_message):
        result = compute_raw_cuda_firth_components(
            fixture.phenotype,
            fixture.genotype,
            fixture.offset,
            fixture.active_sample_mask,
            fixture.non_active_deviance,
            fixture.beta,
            fixture.minimum_variance,
        )
        jax.block_until_ready(result)


def build_raw_deflate_fixture(source_sample_count: int, logical_variant_count: int) -> RawDeflateFixture:
    probability_rows = np.empty((logical_variant_count, source_sample_count, 2), dtype=np.uint8)
    for variant_index in range(logical_variant_count):
        for sample_index in range(source_sample_count):
            first_probability = (variant_index * 29 + sample_index * 17 + 3) % 256
            second_probability = (variant_index * 11 + sample_index * 7 + 5) % (256 - first_probability)
            probability_rows[variant_index, sample_index] = (first_probability, second_probability)
    return build_raw_deflate_fixture_from_probabilities(probability_rows)


def build_raw_deflate_fixture_from_probabilities(
    probability_rows: npt.NDArray[np.uint8],
) -> RawDeflateFixture:
    """Encode explicit packed8 probability rows as aligned raw DEFLATE members."""
    if probability_rows.ndim != 3 or probability_rows.shape[2] != 2:
        raise ValueError("Packed8 fixture probabilities must have shape variants x samples x 2.")
    logical_variant_count, source_sample_count, _ = probability_rows.shape
    compressed_slab = bytearray()
    metadata_rows: list[tuple[int, int, int]] = []
    for variant_index in range(logical_variant_count):
        decompressed_row = (
            struct.pack("<I", source_sample_count)
            + struct.pack("<H", 2)
            + bytes((2, 2))
            + bytes([2]) * source_sample_count
            + bytes((0, 8))
            + probability_rows[variant_index].tobytes()
        )
        compressor = zlib.compressobj(level=1, wbits=-15)
        raw_deflate = compressor.compress(decompressed_row) + compressor.flush()
        aligned_offset = (len(compressed_slab) + 3) & ~3
        compressed_slab.extend(bytes(aligned_offset - len(compressed_slab)))
        compressed_slab.extend(raw_deflate)
        metadata_rows.append((aligned_offset, len(raw_deflate), zlib.adler32(decompressed_row)))
    return RawDeflateFixture(
        compressed_slab=np.frombuffer(bytes(compressed_slab), dtype=np.uint8),
        compressed_metadata=np.asarray(metadata_rows, dtype=np.uint32),
        expected_probabilities=probability_rows,
        source_sample_count=source_sample_count,
        logical_variant_count=logical_variant_count,
    )


def build_packed8_expected(
    fixture: RawDeflateFixture,
    selected_sample_indices: npt.NDArray[np.int64],
) -> Packed8Expected:
    probabilities = fixture.expected_probabilities[:, selected_sample_indices]
    first_probabilities = probabilities[:, :, 0].astype(np.uint64)
    second_probabilities = probabilities[:, :, 1].astype(np.uint64)
    raw_dosages = 510 - 2 * first_probabilities - second_probabilities
    dosage_sums = np.sum(raw_dosages, axis=1, dtype=np.uint64)
    dosage_square_sums = np.sum(raw_dosages * raw_dosages, axis=1, dtype=np.uint64)
    selected_sample_count = selected_sample_indices.size
    genotype_means = dosage_sums.astype(np.float32) * np.float32(1.0 / 255.0) / np.float32(selected_sample_count)
    zero_counts = np.sum(raw_dosages == 0, axis=1, dtype=np.uint64)
    homozygous_alternate_counts = np.sum(raw_dosages >= 383, axis=1, dtype=np.uint64)
    allele_flip_mask = dosage_sums > 255 * selected_sample_count
    regenie_zero_counts = np.where(allele_flip_mask, homozygous_alternate_counts, zero_counts)
    minor_allele_raw_counts = np.minimum(
        dosage_sums,
        510 * selected_sample_count - dosage_sums,
    )
    sparse_candidate_mask = (regenie_zero_counts * 2 >= selected_sample_count) & (minor_allele_raw_counts < 50 * 255)
    return Packed8Expected(
        probabilities=probabilities,
        dosage_sums=dosage_sums,
        dosage_square_sums=dosage_square_sums,
        genotype_means=genotype_means,
        sparse_candidate_mask=sparse_candidate_mask,
    )


def decode_packed8_fixture(
    fixture: RawDeflateFixture,
    selected_sample_indices: npt.NDArray[np.uint32],
    *,
    selection_start: int,
    selected_sample_count: int,
    compute_variant_count: int,
    compressed_metadata: npt.NDArray[np.uint32] | None = None,
) -> compressed_genotype.DecodedPacked8DeflateBatch:
    return compressed_genotype.decode_packed8_deflate_batch(
        compressed_slab=jnp.asarray(fixture.compressed_slab),
        compressed_metadata=jnp.asarray(
            fixture.compressed_metadata if compressed_metadata is None else compressed_metadata
        ),
        selected_sample_indices=jnp.asarray(selected_sample_indices),
        source_sample_count=fixture.source_sample_count,
        selected_sample_count=selected_sample_count,
        selection_start=selection_start,
        compute_variant_count=compute_variant_count,
        retain_imputed_dosage_square_sum=True,
        collect_sparse_candidate_mask=True,
    )


def assert_packed8_result_matches(
    observed: compressed_genotype.DecodedPacked8DeflateBatch,
    expected: Packed8Expected,
    *,
    logical_variant_count: int,
    compute_variant_count: int,
    selected_sample_count: int,
) -> None:
    expected_probabilities = np.full(
        (compute_variant_count, selected_sample_count, 2),
        (255, 0),
        dtype=np.uint8,
    )
    expected_probabilities[:logical_variant_count] = expected.probabilities
    np.testing.assert_array_equal(np.asarray(observed.packed_probability_pairs_by_variant), expected_probabilities)
    expected_dosage_sums = np.zeros(compute_variant_count, dtype=np.uint64)
    expected_dosage_sums[:logical_variant_count] = expected.dosage_sums
    np.testing.assert_array_equal(np.asarray(observed.raw_packed8_statistics.dosage_sums), expected_dosage_sums)
    expected_square_sums = np.zeros(compute_variant_count, dtype=np.uint64)
    expected_square_sums[:logical_variant_count] = expected.dosage_square_sums
    np.testing.assert_array_equal(
        np.asarray(observed.raw_packed8_statistics.dosage_square_sums),
        expected_square_sums,
    )
    np.testing.assert_array_equal(
        np.asarray(observed.raw_packed8_statistics.statuses),
        np.zeros(compute_variant_count, dtype=np.uint32),
    )
    expected_genotype_means = np.zeros(compute_variant_count, dtype=np.float32)
    expected_genotype_means[:logical_variant_count] = expected.genotype_means
    tests.numerical.assert_absolute_difference_less_than(
        observed.genotype_mean,
        expected_genotype_means,
        5.0e-7,
    )
    assert observed.imputed_dosage_square_sum is not None
    expected_imputed_square_sums = expected_square_sums.astype(np.float32) * np.float32(1.0 / 65_025.0)
    tests.numerical.assert_absolute_difference_less_than(
        observed.imputed_dosage_square_sum,
        expected_imputed_square_sums,
        5.0e-7,
    )
    assert observed.sparse_candidate_mask is not None
    expected_sparse_mask = np.zeros(compute_variant_count, dtype=np.bool_)
    expected_sparse_mask[:logical_variant_count] = expected.sparse_candidate_mask
    np.testing.assert_array_equal(np.asarray(observed.sparse_candidate_mask), expected_sparse_mask)


@pytest.mark.parametrize(("logical_variant_count", "compute_variant_count"), ((64, 64), (17, 64)))
@pytest.mark.parametrize("selection_mode", ("identity", "contiguous", "indexed"))
def test_registered_packed8_handler_covers_geometry_selection_and_padding(
    cuda_test_support: CudaFfiTestSupport,
    logical_variant_count: int,
    compute_variant_count: int,
    selection_mode: str,
) -> None:
    assert cuda_test_support.register_packed8_deflate_ffi() == cuda_ffi.PACKED8_DEFLATE_FFI_TARGET
    fixture = build_raw_deflate_fixture(source_sample_count=33, logical_variant_count=logical_variant_count)
    if selection_mode == "identity":
        selected_indices = np.arange(33, dtype=np.int64)
        selected_index_operand = np.empty((0,), dtype=np.uint32)
        selection_start = 0
    elif selection_mode == "contiguous":
        selected_indices = np.arange(3, 30, dtype=np.int64)
        selected_index_operand = np.empty((0,), dtype=np.uint32)
        selection_start = 3
    else:
        selected_indices = np.asarray((32, 0, 16, 3, 31, 7), dtype=np.int64)
        selected_index_operand = selected_indices.astype(np.uint32)
        selection_start = -1
    expected = build_packed8_expected(fixture, selected_indices)

    observed = decode_packed8_fixture(
        fixture,
        selected_index_operand,
        selection_start=selection_start,
        selected_sample_count=selected_indices.size,
        compute_variant_count=compute_variant_count,
    )
    jax.block_until_ready(observed)

    assert_packed8_result_matches(
        observed,
        expected,
        logical_variant_count=logical_variant_count,
        compute_variant_count=compute_variant_count,
        selected_sample_count=selected_indices.size,
    )


@pytest.mark.parametrize("selection_mode", ("contiguous", "indexed"))
def test_registered_packed8_handler_covers_nonidentity_selection_over_256_samples(
    cuda_test_support: CudaFfiTestSupport,
    selection_mode: str,
) -> None:
    assert cuda_test_support.register_packed8_deflate_ffi() == cuda_ffi.PACKED8_DEFLATE_FFI_TARGET
    fixture = build_raw_deflate_fixture(source_sample_count=513, logical_variant_count=5)
    if selection_mode == "contiguous":
        selected_indices = np.arange(71, 371, dtype=np.int64)
        selected_index_operand = np.empty((0,), dtype=np.uint32)
        selection_start = 71
    else:
        selected_indices = (np.arange(300, dtype=np.int64) * 37 + 11) % 513
        selected_index_operand = selected_indices.astype(np.uint32)
        selection_start = -1
    expected = build_packed8_expected(fixture, selected_indices)

    observed = decode_packed8_fixture(
        fixture,
        selected_index_operand,
        selection_start=selection_start,
        selected_sample_count=selected_indices.size,
        compute_variant_count=8,
    )
    jax.block_until_ready(observed)

    assert_packed8_result_matches(
        observed,
        expected,
        logical_variant_count=fixture.logical_variant_count,
        compute_variant_count=8,
        selected_sample_count=selected_indices.size,
    )


def test_registered_packed8_handler_covers_sparse_mask_positive_and_threshold_boundary(
    cuda_test_support: CudaFfiTestSupport,
) -> None:
    assert cuda_test_support.register_packed8_deflate_ffi() == cuda_ffi.PACKED8_DEFLATE_FFI_TARGET
    source_sample_count = 100
    probability_rows = np.full((3, source_sample_count, 2), (255, 0), dtype=np.uint8)
    probability_rows[0, -1] = (0, 255)
    probability_rows[1, -49:] = (0, 255)
    probability_rows[2, -50:] = (0, 255)
    fixture = build_raw_deflate_fixture_from_probabilities(probability_rows)
    selected_indices = np.arange(source_sample_count, dtype=np.int64)
    expected = build_packed8_expected(fixture, selected_indices)
    np.testing.assert_array_equal(expected.sparse_candidate_mask, np.asarray((True, True, False)))

    observed = decode_packed8_fixture(
        fixture,
        np.empty((0,), dtype=np.uint32),
        selection_start=0,
        selected_sample_count=source_sample_count,
        compute_variant_count=4,
    )
    jax.block_until_ready(observed)

    assert_packed8_result_matches(
        observed,
        expected,
        logical_variant_count=fixture.logical_variant_count,
        compute_variant_count=4,
        selected_sample_count=source_sample_count,
    )


@pytest.mark.parametrize(
    ("source_sample_count", "logical_variant_count", "compute_variant_count"),
    PACKED8_BOUNDARY_GEOMETRIES,
)
def test_registered_packed8_handler_covers_sample_and_block_boundaries(
    cuda_test_support: CudaFfiTestSupport,
    source_sample_count: int,
    logical_variant_count: int,
    compute_variant_count: int,
) -> None:
    assert cuda_test_support.register_packed8_deflate_ffi() == cuda_ffi.PACKED8_DEFLATE_FFI_TARGET
    fixture = build_raw_deflate_fixture(
        source_sample_count=source_sample_count,
        logical_variant_count=logical_variant_count,
    )
    selected_indices = np.arange(source_sample_count, dtype=np.int64)
    expected = build_packed8_expected(fixture, selected_indices)

    observed = decode_packed8_fixture(
        fixture,
        np.empty((0,), dtype=np.uint32),
        selection_start=0,
        selected_sample_count=source_sample_count,
        compute_variant_count=compute_variant_count,
    )
    jax.block_until_ready(observed)

    assert_packed8_result_matches(
        observed,
        expected,
        logical_variant_count=logical_variant_count,
        compute_variant_count=compute_variant_count,
        selected_sample_count=source_sample_count,
    )


@pytest.mark.parametrize(
    ("source_sample_count", "logical_variant_count", "compute_variant_count"),
    PACKED8_PRODUCTION_GEOMETRIES,
)
def test_registered_packed8_handler_covers_production_chunk_full_and_tail_geometry(
    cuda_test_support: CudaFfiTestSupport,
    source_sample_count: int,
    logical_variant_count: int,
    compute_variant_count: int,
) -> None:
    assert cuda_test_support.register_packed8_deflate_ffi() == cuda_ffi.PACKED8_DEFLATE_FFI_TARGET
    fixture = build_raw_deflate_fixture(
        source_sample_count=source_sample_count,
        logical_variant_count=logical_variant_count,
    )
    selected_indices = np.arange(source_sample_count, dtype=np.int64)
    expected = build_packed8_expected(fixture, selected_indices)

    observed = decode_packed8_fixture(
        fixture,
        np.empty((0,), dtype=np.uint32),
        selection_start=0,
        selected_sample_count=source_sample_count,
        compute_variant_count=compute_variant_count,
    )
    jax.block_until_ready(observed)

    assert_packed8_result_matches(
        observed,
        expected,
        logical_variant_count=logical_variant_count,
        compute_variant_count=compute_variant_count,
        selected_sample_count=source_sample_count,
    )


@pytest.mark.parametrize("descriptor_failure", ("unaligned_offset", "out_of_bounds", "zero_size"))
def test_registered_packed8_handler_reports_descriptor_failures_without_retry(
    cuda_test_support: CudaFfiTestSupport,
    descriptor_failure: str,
) -> None:
    assert cuda_test_support.register_packed8_deflate_ffi() == cuda_ffi.PACKED8_DEFLATE_FFI_TARGET
    fixture = build_raw_deflate_fixture(source_sample_count=33, logical_variant_count=2)
    invalid_metadata = fixture.compressed_metadata.copy()
    if descriptor_failure == "unaligned_offset":
        input_alignment = cuda_test_support.nvcomp_input_alignment()
        if input_alignment == 1:
            pytest.skip("the selected nvCOMP runtime accepts every byte offset")
        invalid_metadata[0, 0] = 1
    elif descriptor_failure == "out_of_bounds":
        invalid_metadata[0, 0] = fixture.compressed_slab.size + 4
    else:
        invalid_metadata[0, 1] = 0

    observed = decode_packed8_fixture(
        fixture,
        np.empty((0,), dtype=np.uint32),
        selection_start=0,
        selected_sample_count=fixture.source_sample_count,
        compute_variant_count=4,
        compressed_metadata=invalid_metadata,
    )
    jax.block_until_ready(observed)

    statuses = np.asarray(observed.raw_packed8_statistics.statuses)
    assert int(statuses[0]) == PACKED8_DESCRIPTOR_STATUS
    np.testing.assert_array_equal(statuses[1:], np.zeros(3, dtype=np.uint32))
    np.testing.assert_array_equal(
        np.asarray(observed.packed_probability_pairs_by_variant)[0],
        np.tile(np.asarray((255, 0), dtype=np.uint8), (fixture.source_sample_count, 1)),
    )


def test_registered_packed8_handler_reports_adler_status(
    cuda_test_support: CudaFfiTestSupport,
) -> None:
    assert cuda_test_support.register_packed8_deflate_ffi() == cuda_ffi.PACKED8_DEFLATE_FFI_TARGET
    fixture = build_raw_deflate_fixture(source_sample_count=33, logical_variant_count=2)
    invalid_metadata = fixture.compressed_metadata.copy()
    invalid_metadata[0, 2] ^= np.uint32(1)

    observed = decode_packed8_fixture(
        fixture,
        np.empty((0,), dtype=np.uint32),
        selection_start=0,
        selected_sample_count=fixture.source_sample_count,
        compute_variant_count=2,
        compressed_metadata=invalid_metadata,
    )
    jax.block_until_ready(observed)

    np.testing.assert_array_equal(
        np.asarray(observed.raw_packed8_statistics.statuses),
        np.asarray((PACKED8_ADLER_STATUS, 0), dtype=np.uint32),
    )


def test_registered_packed8_handler_reports_invalid_selected_index(
    cuda_test_support: CudaFfiTestSupport,
) -> None:
    assert cuda_test_support.register_packed8_deflate_ffi() == cuda_ffi.PACKED8_DEFLATE_FFI_TARGET
    fixture = build_raw_deflate_fixture(source_sample_count=33, logical_variant_count=2)

    observed = decode_packed8_fixture(
        fixture,
        np.asarray((0, fixture.source_sample_count), dtype=np.uint32),
        selection_start=-1,
        selected_sample_count=2,
        compute_variant_count=2,
    )
    jax.block_until_ready(observed)

    np.testing.assert_array_equal(
        np.asarray(observed.raw_packed8_statistics.statuses),
        np.full(2, PACKED8_SAMPLE_INDEX_STATUS, dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        np.asarray(observed.packed_probability_pairs_by_variant)[:, 1, :],
        np.tile(np.asarray((255, 0), dtype=np.uint8), (2, 1)),
    )


def test_registered_packed8_handler_rejects_capacity_overflow(
    cuda_test_support: CudaFfiTestSupport,
) -> None:
    assert cuda_test_support.register_packed8_deflate_ffi() == cuda_ffi.PACKED8_DEFLATE_FFI_TARGET
    fixture = build_raw_deflate_fixture(source_sample_count=33, logical_variant_count=3)

    with pytest.raises(jax.errors.JaxRuntimeError, match="compute variant count must cover all logical variants"):
        result = decode_packed8_fixture(
            fixture,
            np.empty((0,), dtype=np.uint32),
            selection_start=0,
            selected_sample_count=fixture.source_sample_count,
            compute_variant_count=2,
        )
        jax.block_until_ready(result)
