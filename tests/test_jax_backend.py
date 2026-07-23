"""CPU-safe contracts for native-to-JAX transfer and materialization."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

import tests.numerical
import tests.test_compressed_genotype
import tests.test_regenie2_binary
import tests.test_regenie2_binary_pipeline
import tests.test_regenie2_linear
from g import jax_backend, types
from g.compute.common import compressed_genotype
from g.compute.common import result as association_result

if typing.TYPE_CHECKING:
    from g.compute.regenie2_binary import result as regenie2_binary_result
    from g.compute.regenie2_binary import state as regenie2_binary_state
    from g.compute.regenie2_linear import state as regenie2_linear_state

# The adapter quantizes native inputs and genotype summaries to float32 before
# the production kernel. The measured beta difference from the independent
# float64 oracle is 3.17e-7; this exclusive bound remains below the 5e-7
# application correctness limit.
BACKEND_LINEAR_BETA_ABSOLUTE_TOLERANCE = 5.0e-7
# Keep materialization assertions aligned with the campaign's exclusive,
# statistic-specific whole-application acceptance ceilings.
MATERIALIZED_BETA_ABSOLUTE_TOLERANCE = 5.0e-7
MATERIALIZED_STANDARD_ERROR_ABSOLUTE_TOLERANCE = 2.5e-7
MATERIALIZED_CHI_SQUARED_ABSOLUTE_TOLERANCE = 2.0e-6
MATERIALIZED_LOG10_P_VALUE_ABSOLUTE_TOLERANCE = 5.0e-7


class CompressedTestBackend(jax_backend.JaxBackendBase):
    """Concrete policy used to exercise the shared compressed path."""

    retain_compressed_imputed_dosage_square_sum = True
    collect_compressed_sparse_candidate_mask = True


def build_device_association(
    *,
    include_correction_codes: bool,
) -> jax_backend.DeviceAssociationResult:
    """Build deterministic trait-major statistics with padded variant columns."""
    result = association_result.AssociationResult(
        beta=jnp.asarray(
            [
                [0.1, 0.2, 0.3, 9.0],
                [1.1, 1.2, 1.3, 9.0],
                [2.1, 2.2, 2.3, 9.0],
            ],
            dtype=jnp.float32,
        ),
        standard_error=jnp.asarray(
            [
                [0.01, 0.02, 0.03, 9.0],
                [0.11, 0.12, 0.13, 9.0],
                [0.21, 0.22, 0.23, 9.0],
            ],
            dtype=jnp.float32,
        ),
        chi_squared=jnp.asarray(
            [
                [1.0, 2.0, 3.0, 9.0],
                [11.0, 12.0, 13.0, 9.0],
                [21.0, 22.0, 23.0, 9.0],
            ],
            dtype=jnp.float32,
        ),
        log10_p_value=jnp.asarray(
            [
                [0.5, 0.6, 0.7, 9.0],
                [1.5, 1.6, 1.7, 9.0],
                [2.5, 2.6, 2.7, 9.0],
            ],
            dtype=jnp.float32,
        ),
        correction_code=(
            jnp.asarray(
                [
                    [0, 1, 2, 3],
                    [3, 2, 1, 0],
                    [1, 3, 0, 2],
                ],
                dtype=jnp.uint8,
            )
            if include_correction_codes
            else None
        ),
    )
    return typing.cast("jax_backend.DeviceAssociationResult", result)


def build_raw_statistics() -> compressed_genotype.Packed8RawStatistics[jax.Array, jax.Array]:
    """Build padded exact native statistics for materialization tests."""
    return compressed_genotype.Packed8RawStatistics(
        dosage_sums=jnp.asarray([100, 200, 300, 999], dtype=jnp.uint64),
        dosage_square_sums=jnp.asarray([1_000, 2_000, 3_000, 999], dtype=jnp.uint64),
        statuses=jnp.asarray([0, 1, 2, 7], dtype=jnp.uint32),
        selected_sample_count=8,
    )


def assert_host_association_statistics(
    association: jax_backend.HostAssociationResult,
    *,
    beta: npt.NDArray[np.float32],
    standard_error: npt.NDArray[np.float32],
    chi_squared: npt.NDArray[np.float32],
    log10_p_value: npt.NDArray[np.float32],
) -> None:
    """Assert every host statistic with its strict application tolerance."""
    tests.numerical.assert_absolute_difference_less_than(
        association.beta,
        beta,
        MATERIALIZED_BETA_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        association.standard_error,
        standard_error,
        MATERIALIZED_STANDARD_ERROR_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        association.chi_squared,
        chi_squared,
        MATERIALIZED_CHI_SQUARED_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        association.log10_p_value,
        log10_p_value,
        MATERIALIZED_LOG10_P_VALUE_ABSOLUTE_TOLERANCE,
    )
    assert association.beta.dtype == np.dtype(np.float32)
    assert association.standard_error.dtype == np.dtype(np.float32)
    assert association.chi_squared.dtype == np.dtype(np.float32)
    assert association.log10_p_value.dtype == np.dtype(np.float32)


def test_resolve_contiguous_compressed_selection_accepts_exact_tail() -> None:
    observed = jax_backend.resolve_host_compressed_transfer_selection(
        source_sample_count=12,
        selected_sample_count=5,
        selection_start=7,
        selected_sample_indices=None,
    )

    assert observed.selection_start == 7
    assert observed.selected_sample_indices.dtype == np.dtype(np.uint32)
    assert observed.selected_sample_indices.shape == (0,)


@pytest.mark.parametrize(
    ("selection_start", "selected_sample_count"),
    [(-1, 1), (8, 5), (13, 1)],
)
def test_resolve_contiguous_compressed_selection_rejects_invalid_range(
    selection_start: int,
    selected_sample_count: int,
) -> None:
    with pytest.raises(ValueError, match="exceeds the source sample count"):
        jax_backend.resolve_host_compressed_transfer_selection(
            source_sample_count=12,
            selected_sample_count=selected_sample_count,
            selection_start=selection_start,
            selected_sample_indices=None,
        )


def test_resolve_indexed_compressed_selection_preserves_nonmonotonic_indices() -> None:
    selected_sample_indices = np.asarray([7, 1, 7, 3], dtype=np.uint32)

    observed = jax_backend.resolve_host_compressed_transfer_selection(
        source_sample_count=12,
        selected_sample_count=4,
        selection_start=None,
        selected_sample_indices=selected_sample_indices,
    )

    assert observed.selection_start == -1
    assert observed.selected_sample_indices is selected_sample_indices


@pytest.mark.parametrize(
    "selected_sample_indices",
    [
        np.asarray([[1, 2]], dtype=np.uint32),
        np.asarray([1, 2], dtype=np.int32),
    ],
)
def test_resolve_indexed_compressed_selection_rejects_shape_or_dtype(
    selected_sample_indices: npt.NDArray[np.integer],
) -> None:
    with pytest.raises(ValueError, match="one-dimensional uint32"):
        jax_backend.resolve_host_compressed_transfer_selection(
            source_sample_count=12,
            selected_sample_count=selected_sample_indices.size,
            selection_start=None,
            selected_sample_indices=typing.cast("npt.NDArray[np.uint32]", selected_sample_indices),
        )


def test_resolve_indexed_compressed_selection_rejects_count_mismatch() -> None:
    with pytest.raises(ValueError, match="one index per selected sample"):
        jax_backend.resolve_host_compressed_transfer_selection(
            source_sample_count=12,
            selected_sample_count=3,
            selection_start=None,
            selected_sample_indices=np.asarray([1, 2], dtype=np.uint32),
        )


@pytest.mark.parametrize(
    ("selection_start", "selected_sample_indices"),
    [
        (None, None),
        (0, np.asarray([0], dtype=np.uint32)),
    ],
)
def test_resolve_compressed_selection_requires_exactly_one_mode(
    selection_start: int | None,
    selected_sample_indices: npt.NDArray[np.uint32] | None,
) -> None:
    with pytest.raises(ValueError, match="either contiguous or indexed"):
        jax_backend.resolve_host_compressed_transfer_selection(
            source_sample_count=12,
            selected_sample_count=1,
            selection_start=selection_start,
            selected_sample_indices=selected_sample_indices,
        )


def test_prepare_host_transfer_requires_all_compressed_values_absent() -> None:
    assert (
        jax_backend.prepare_compressed_transfer_selection(
            source_sample_count=None,
            selected_sample_count=None,
            selection_start=None,
            selected_sample_indices=None,
        )
        is None
    )

    with pytest.raises(ValueError, match="every compressed selection value to be None"):
        jax_backend.prepare_compressed_transfer_selection(
            source_sample_count=None,
            selected_sample_count=4,
            selection_start=None,
            selected_sample_indices=None,
        )


def test_prepare_compressed_transfer_requires_complete_positive_geometry() -> None:
    with pytest.raises(ValueError, match="source and selected sample counts"):
        jax_backend.prepare_compressed_transfer_selection(
            source_sample_count=12,
            selected_sample_count=None,
            selection_start=0,
            selected_sample_indices=None,
        )

    for source_sample_count, selected_sample_count in [(0, 1), (12, 0), (-1, 1), (12, -1)]:
        with pytest.raises(ValueError, match="counts must be positive"):
            jax_backend.prepare_compressed_transfer_selection(
                source_sample_count=source_sample_count,
                selected_sample_count=selected_sample_count,
                selection_start=0,
                selected_sample_indices=None,
            )


def test_prepare_compressed_transfer_uploads_indexed_selection() -> None:
    selected_sample_indices = np.asarray([5, 1, 3], dtype=np.uint32)

    observed = jax_backend.prepare_compressed_transfer_selection(
        source_sample_count=8,
        selected_sample_count=3,
        selection_start=None,
        selected_sample_indices=selected_sample_indices,
    )

    assert observed is not None
    assert observed.source_sample_count == 8
    assert observed.selected_sample_count == 3
    assert observed.selection_start == -1
    assert observed.selected_sample_indices.dtype == jnp.uint32
    np.testing.assert_array_equal(np.asarray(observed.selected_sample_indices), selected_sample_indices)


@pytest.mark.parametrize("packed8", [False, True])
def test_transfer_batch_preserves_values_and_optional_operands(*, packed8: bool) -> None:
    backend = CompressedTestBackend()
    genotype_values: npt.NDArray[np.float32] | npt.NDArray[np.uint8]
    if packed8:
        genotype_values = np.asarray([[[255, 0], [0, 255]]], dtype=np.uint8)
    else:
        genotype_values = np.asarray([[0.0, 1.0]], dtype=np.float32)
    genotype_mean = np.asarray([0.5], dtype=np.float32)
    imputed_dosage_square_sum = np.asarray([1.0], dtype=np.float32)
    sparse_candidate_mask = np.asarray([True], dtype=np.bool_)

    observed = backend.transfer_batch(
        genotype_values=genotype_values,
        genotype_mean=genotype_mean,
        imputed_dosage_square_sum=imputed_dosage_square_sum,
        sparse_candidate_mask=sparse_candidate_mask,
    )

    assert observed.packed8 is packed8
    np.testing.assert_array_equal(np.asarray(observed.genotype_values), genotype_values)
    tests.numerical.assert_absolute_difference_less_than(observed.genotype_mean, genotype_mean, 1.0e-7)
    assert observed.imputed_dosage_square_sum is not None
    tests.numerical.assert_absolute_difference_less_than(
        observed.imputed_dosage_square_sum,
        imputed_dosage_square_sum,
        1.0e-7,
    )
    assert observed.sparse_candidate_mask is not None
    np.testing.assert_array_equal(np.asarray(observed.sparse_candidate_mask), sparse_candidate_mask)
    assert observed.raw_packed8_statistics is None


def test_transfer_batch_preserves_absent_optional_operands() -> None:
    observed = CompressedTestBackend().transfer_batch(
        genotype_values=np.asarray([[0.0, 1.0]], dtype=np.float32),
        genotype_mean=np.asarray([0.5], dtype=np.float32),
        imputed_dosage_square_sum=None,
        sparse_candidate_mask=None,
    )

    assert observed.imputed_dosage_square_sum is None
    assert observed.sparse_candidate_mask is None


def test_transfer_compressed_batch_requires_prepared_selection() -> None:
    group_state = jax_backend.DeviceGroupState(
        association_state=object(),
        compressed_transfer_selection=None,
    )

    with pytest.raises(ValueError, match="requires a prepared compressed group selection"):
        CompressedTestBackend().transfer_compressed_batch(
            group_state=group_state,
            compressed_slab=np.asarray([1], dtype=np.uint8),
            compressed_metadata=np.asarray([[0, 1, 1]], dtype=np.uint32),
            compute_variant_count=1,
        )


def test_transfer_compressed_batch_maps_indexed_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    tests.test_compressed_genotype.install_fake_packed8_ffi(
        monkeypatch,
        expected_selected_sample_indices=np.arange(100, dtype=np.uint32),
        expected_selection_start=-1,
    )
    selection = jax_backend.prepare_compressed_transfer_selection(
        source_sample_count=120,
        selected_sample_count=100,
        selection_start=None,
        selected_sample_indices=np.arange(100, dtype=np.uint32),
    )
    assert selection is not None
    group_state = jax_backend.DeviceGroupState(
        association_state=object(),
        compressed_transfer_selection=selection,
    )

    with jax.disable_jit():
        observed = CompressedTestBackend().transfer_compressed_batch(
            group_state=group_state,
            compressed_slab=tests.test_compressed_genotype.build_compressed_slab(),
            compressed_metadata=tests.test_compressed_genotype.build_compressed_metadata(),
            compute_variant_count=4,
        )

    assert observed.packed8
    assert observed.genotype_values.shape == (4, 100, 2)
    assert observed.genotype_values.dtype == jnp.uint8
    assert observed.imputed_dosage_square_sum is not None
    assert observed.sparse_candidate_mask is not None
    np.testing.assert_array_equal(
        np.asarray(observed.sparse_candidate_mask),
        np.asarray([True, True, False, False]),
    )
    assert observed.raw_packed8_statistics is not None
    np.testing.assert_array_equal(
        np.asarray(observed.raw_packed8_statistics.statuses),
        np.asarray([0, 1, 2, 7], dtype=np.uint32),
    )


def test_transfer_compressed_batch_maps_contiguous_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_selected_sample_indices = np.empty((0,), dtype=np.uint32)
    tests.test_compressed_genotype.install_fake_packed8_ffi(
        monkeypatch,
        expected_selected_sample_indices=expected_selected_sample_indices,
        expected_selection_start=10,
    )
    selection = jax_backend.prepare_compressed_transfer_selection(
        source_sample_count=120,
        selected_sample_count=100,
        selection_start=10,
        selected_sample_indices=None,
    )
    assert selection is not None
    assert selection.selection_start == 10
    assert selection.selected_sample_indices.shape == (0,)
    group_state = jax_backend.DeviceGroupState(
        association_state=object(),
        compressed_transfer_selection=selection,
    )

    with jax.disable_jit():
        observed = CompressedTestBackend().transfer_compressed_batch(
            group_state=group_state,
            compressed_slab=tests.test_compressed_genotype.build_compressed_slab(),
            compressed_metadata=tests.test_compressed_genotype.build_compressed_metadata(),
            compute_variant_count=4,
        )

    assert observed.packed8
    assert observed.genotype_values.shape == (4, 100, 2)
    assert observed.raw_packed8_statistics is not None
    assert observed.raw_packed8_statistics.selected_sample_count == 100


def test_materialize_batch_reorders_traits_and_truncates_padded_variants() -> None:
    device_batch: jax_backend.DeviceAssociationBatch = jax_backend.AssociationBatch(
        association=build_device_association(include_correction_codes=True),
        raw_packed8_statistics=build_raw_statistics(),
        firth_candidate_count=jnp.asarray(4, dtype=jnp.int32),
        firth_candidate_capacity=4,
    )

    observed = jax_backend.JaxBackendBase().materialize_batch(
        device_result=device_batch,
        active_trait_indices=np.asarray([2, 0], dtype=np.int32),
        logical_variant_count=3,
    )

    assert_host_association_statistics(
        observed.association,
        beta=np.asarray([[2.1, 2.2, 2.3], [0.1, 0.2, 0.3]], dtype=np.float32),
        standard_error=np.asarray([[0.21, 0.22, 0.23], [0.01, 0.02, 0.03]], dtype=np.float32),
        chi_squared=np.asarray([[21.0, 22.0, 23.0], [1.0, 2.0, 3.0]], dtype=np.float32),
        log10_p_value=np.asarray([[2.5, 2.6, 2.7], [0.5, 0.6, 0.7]], dtype=np.float32),
    )
    assert observed.association.correction_code is not None
    assert observed.association.correction_code.dtype == np.dtype(np.uint8)
    np.testing.assert_array_equal(
        observed.association.correction_code,
        np.asarray([[1, 3, 0], [0, 1, 2]], dtype=np.uint8),
    )
    assert observed.raw_packed8_statistics is not None
    assert observed.raw_packed8_statistics.dosage_sums.dtype == np.dtype(np.uint64)
    assert observed.raw_packed8_statistics.dosage_square_sums.dtype == np.dtype(np.uint64)
    assert observed.raw_packed8_statistics.statuses.dtype == np.dtype(np.uint32)
    np.testing.assert_array_equal(
        observed.raw_packed8_statistics.dosage_sums,
        np.asarray([100, 200, 300], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        observed.raw_packed8_statistics.dosage_square_sums,
        np.asarray([1_000, 2_000, 3_000], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        observed.raw_packed8_statistics.statuses,
        np.asarray([0, 1, 2], dtype=np.uint32),
    )
    assert observed.raw_packed8_statistics.selected_sample_count == 8
    assert int(np.asarray(observed.firth_candidate_count)) == 4
    assert observed.firth_candidate_capacity == 4


def test_materialize_batch_supports_uncorrected_full_batch() -> None:
    device_batch: jax_backend.DeviceAssociationBatch = jax_backend.AssociationBatch(
        association=build_device_association(include_correction_codes=False),
        raw_packed8_statistics=build_raw_statistics(),
        firth_candidate_count=None,
        firth_candidate_capacity=None,
    )

    observed = jax_backend.JaxBackendBase().materialize_batch(
        device_result=device_batch,
        active_trait_indices=None,
        logical_variant_count=4,
    )

    assert_host_association_statistics(
        observed.association,
        beta=np.asarray(
            [
                [0.1, 0.2, 0.3, 9.0],
                [1.1, 1.2, 1.3, 9.0],
                [2.1, 2.2, 2.3, 9.0],
            ],
            dtype=np.float32,
        ),
        standard_error=np.asarray(
            [
                [0.01, 0.02, 0.03, 9.0],
                [0.11, 0.12, 0.13, 9.0],
                [0.21, 0.22, 0.23, 9.0],
            ],
            dtype=np.float32,
        ),
        chi_squared=np.asarray(
            [
                [1.0, 2.0, 3.0, 9.0],
                [11.0, 12.0, 13.0, 9.0],
                [21.0, 22.0, 23.0, 9.0],
            ],
            dtype=np.float32,
        ),
        log10_p_value=np.asarray(
            [
                [0.5, 0.6, 0.7, 9.0],
                [1.5, 1.6, 1.7, 9.0],
                [2.5, 2.6, 2.7, 9.0],
            ],
            dtype=np.float32,
        ),
    )
    assert observed.association.correction_code is None
    assert observed.raw_packed8_statistics is not None
    assert observed.raw_packed8_statistics.dosage_sums.dtype == np.dtype(np.uint64)
    assert observed.raw_packed8_statistics.dosage_square_sums.dtype == np.dtype(np.uint64)
    assert observed.raw_packed8_statistics.statuses.dtype == np.dtype(np.uint32)
    np.testing.assert_array_equal(
        observed.raw_packed8_statistics.dosage_sums,
        np.asarray([100, 200, 300, 999], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        observed.raw_packed8_statistics.dosage_square_sums,
        np.asarray([1_000, 2_000, 3_000, 999], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        observed.raw_packed8_statistics.statuses,
        np.asarray([0, 1, 2, 7], dtype=np.uint32),
    )
    assert observed.raw_packed8_statistics.selected_sample_count == 8
    assert observed.firth_candidate_count is None
    assert observed.firth_candidate_capacity is None


def test_materialize_batch_truncates_uncorrected_association_without_raw_statistics() -> None:
    device_batch: jax_backend.DeviceAssociationBatch = jax_backend.AssociationBatch(
        association=build_device_association(include_correction_codes=False),
        raw_packed8_statistics=None,
        firth_candidate_count=None,
        firth_candidate_capacity=None,
    )

    observed = jax_backend.JaxBackendBase().materialize_batch(
        device_result=device_batch,
        active_trait_indices=None,
        logical_variant_count=2,
    )

    assert_host_association_statistics(
        observed.association,
        beta=np.asarray([[0.1, 0.2], [1.1, 1.2], [2.1, 2.2]], dtype=np.float32),
        standard_error=np.asarray([[0.01, 0.02], [0.11, 0.12], [0.21, 0.22]], dtype=np.float32),
        chi_squared=np.asarray([[1.0, 2.0], [11.0, 12.0], [21.0, 22.0]], dtype=np.float32),
        log10_p_value=np.asarray([[0.5, 0.6], [1.5, 1.6], [2.5, 2.6]], dtype=np.float32),
    )
    assert observed.association.correction_code is None
    assert observed.raw_packed8_statistics is None
    assert observed.firth_candidate_count is None
    assert observed.firth_candidate_capacity is None


@pytest.mark.parametrize(
    ("candidate_count", "candidate_capacity"),
    [(jnp.asarray(1, dtype=jnp.int32), None), (None, 1)],
)
def test_materialize_batch_rejects_partial_candidate_capacity_contract(
    candidate_count: jax.Array | None,
    candidate_capacity: int | None,
) -> None:
    device_batch: jax_backend.DeviceAssociationBatch = jax_backend.AssociationBatch(
        association=build_device_association(include_correction_codes=False),
        raw_packed8_statistics=None,
        firth_candidate_count=candidate_count,
        firth_candidate_capacity=candidate_capacity,
    )

    with pytest.raises(ValueError, match="count and capacity must be materialized together"):
        jax_backend.JaxBackendBase().materialize_batch(
            device_result=device_batch,
            active_trait_indices=None,
            logical_variant_count=4,
        )


def test_materialize_batch_rejects_candidate_capacity_overflow() -> None:
    device_batch: jax_backend.DeviceAssociationBatch = jax_backend.AssociationBatch(
        association=build_device_association(include_correction_codes=False),
        raw_packed8_statistics=None,
        firth_candidate_count=jnp.asarray(5, dtype=jnp.int32),
        firth_candidate_capacity=4,
    )

    with pytest.raises(ValueError, match=r"candidate count 5 exceeded.*capacity of 4"):
        jax_backend.JaxBackendBase().materialize_batch(
            device_result=device_batch,
            active_trait_indices=None,
            logical_variant_count=4,
        )


def build_linear_backend() -> jax_backend.LinearJaxBackend:
    """Build the adapter with the policy used by the independent linear oracle."""
    return jax_backend.LinearJaxBackend(
        minimum_variance=1.0e-8,
        relative_variance_tolerance=1.0e-7,
    )


def test_linear_backend_runs_decoded_batch_against_independent_oracle() -> None:
    fixture = tests.test_regenie2_linear.build_linear_fixture()
    genotype_mean = np.asarray(np.mean(fixture.genotype_matrix_by_variant, axis=1), dtype=np.float32)
    dosage_square_sum = np.asarray(np.sum(fixture.genotype_matrix_by_variant**2, axis=1), dtype=np.float32)
    reference = tests.test_regenie2_linear.compute_linear_reference_with_genotype_statistics(
        fixture=fixture,
        genotype_means=np.asarray(genotype_mean, dtype=np.float64),
        imputed_dosage_square_sum=np.asarray(dosage_square_sum, dtype=np.float64),
    )
    backend = build_linear_backend()
    group_state = backend.prepare_group(
        phenotype_matrix=np.asarray(fixture.phenotype_matrix, dtype=np.float32),
        covariate_matrix=np.asarray(fixture.covariate_matrix, dtype=np.float32),
        source_sample_count=None,
        selected_sample_count=None,
        selection_start=None,
        selected_sample_indices=None,
    )
    chromosome_state = backend.prepare_chromosome(
        group_state=group_state,
        prediction_matrix=np.asarray(fixture.loco_prediction_matrix, dtype=np.float32),
    )
    batch = backend.transfer_batch(
        genotype_values=np.asarray(fixture.genotype_matrix_by_variant, dtype=np.float32),
        genotype_mean=genotype_mean,
        imputed_dosage_square_sum=dosage_square_sum,
        sparse_candidate_mask=None,
    )

    observed = backend.compute_batch(chromosome_state=chromosome_state, batch=batch)

    tests.numerical.assert_absolute_difference_less_than(
        observed.association.beta,
        reference.beta,
        BACKEND_LINEAR_BETA_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.association.standard_error,
        reference.standard_error,
        tests.test_regenie2_linear.LINEAR_STANDARD_ERROR_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.association.chi_squared,
        reference.chi_squared,
        tests.test_regenie2_linear.LINEAR_CHI_SQUARED_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.association.log10_p_value,
        reference.log10_p_value,
        tests.test_regenie2_linear.LINEAR_LOG10_P_VALUE_ABSOLUTE_TOLERANCE,
    )
    assert observed.association.correction_code is None
    assert observed.raw_packed8_statistics is None


def test_linear_backend_runs_packed8_batch_equivalent_to_decoded_probabilities() -> None:
    fixture = tests.test_regenie2_linear.build_linear_fixture()
    packed_probabilities = np.asarray(
        [
            [[255, 0], [0, 0], [255, 0], [0, 0], [255, 0], [0, 0], [255, 0], [0, 0]],
            [[255, 0], [0, 255], [0, 0], [255, 0], [0, 0], [0, 255], [255, 0], [0, 0]],
        ],
        dtype=np.uint8,
    )
    decoded_genotypes = (
        510.0
        - 2.0 * np.asarray(packed_probabilities[:, :, 0], dtype=np.float32)
        - np.asarray(packed_probabilities[:, :, 1], dtype=np.float32)
    ) / 255.0
    genotype_mean = np.asarray(np.mean(decoded_genotypes, axis=1), dtype=np.float32)
    dosage_square_sum = np.asarray(np.sum(decoded_genotypes**2, axis=1), dtype=np.float32)
    decoded_fixture = tests.test_regenie2_linear.LinearFixture(
        covariate_matrix=fixture.covariate_matrix,
        phenotype_matrix=fixture.phenotype_matrix,
        loco_prediction_matrix=fixture.loco_prediction_matrix,
        genotype_matrix_by_variant=np.asarray(decoded_genotypes, dtype=np.float64),
    )
    reference = tests.test_regenie2_linear.compute_linear_reference_with_genotype_statistics(
        fixture=decoded_fixture,
        genotype_means=np.asarray(genotype_mean, dtype=np.float64),
        imputed_dosage_square_sum=np.asarray(dosage_square_sum, dtype=np.float64),
    )
    backend = build_linear_backend()
    group_state = backend.prepare_group(
        phenotype_matrix=np.asarray(fixture.phenotype_matrix, dtype=np.float32),
        covariate_matrix=np.asarray(fixture.covariate_matrix, dtype=np.float32),
        source_sample_count=None,
        selected_sample_count=None,
        selection_start=None,
        selected_sample_indices=None,
    )
    chromosome_state = backend.prepare_chromosome(
        group_state=group_state,
        prediction_matrix=np.asarray(fixture.loco_prediction_matrix, dtype=np.float32),
    )
    batch = backend.transfer_batch(
        genotype_values=packed_probabilities,
        genotype_mean=genotype_mean,
        imputed_dosage_square_sum=dosage_square_sum,
        sparse_candidate_mask=None,
    )

    observed = backend.compute_batch(chromosome_state=chromosome_state, batch=batch)

    tests.numerical.assert_absolute_difference_less_than(
        observed.association.beta,
        reference.beta,
        tests.test_regenie2_linear.LINEAR_BETA_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.association.standard_error,
        reference.standard_error,
        tests.test_regenie2_linear.LINEAR_STANDARD_ERROR_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.association.chi_squared,
        reference.chi_squared,
        tests.test_regenie2_linear.LINEAR_CHI_SQUARED_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.association.log10_p_value,
        reference.log10_p_value,
        tests.test_regenie2_linear.LINEAR_LOG10_P_VALUE_ABSOLUTE_TOLERANCE,
    )


def test_linear_backend_rejects_missing_dosage_square_sum() -> None:
    batch = build_decoded_genotype_batch(sparse_candidate_mask=None)
    assert batch.imputed_dosage_square_sum is None

    with pytest.raises(ValueError, match="requires imputed dosage square sums"):
        build_linear_backend().compute_batch(
            chromosome_state=typing.cast("regenie2_linear_state.Regenie2MultiLinearChromosomeState", object()),
            batch=batch,
        )


def build_binary_score_backend() -> jax_backend.BinaryScoreJaxBackend:
    """Build the adapter with the policy used by the independent score oracle."""
    config = tests.test_regenie2_binary.build_binary_score_config()
    return jax_backend.BinaryScoreJaxBackend(
        minimum_probability=config.numerical.minimum_probability,
        minimum_variance=config.numerical.minimum_variance,
        relative_variance_tolerance=config.numerical.relative_variance_tolerance,
        null_logistic_maximum_iterations=config.null_logistic.maximum_iterations,
        null_logistic_coefficient_tolerance=config.null_logistic.coefficient_tolerance,
    )


def build_decoded_genotype_batch(
    *,
    sparse_candidate_mask: npt.NDArray[np.bool_] | None,
) -> jax_backend.DeviceGenotypeBatch:
    """Build a small decoded batch for early adapter validation branches."""
    return jax_backend.DeviceGenotypeBatch(
        genotype_values=jnp.asarray([[0.0, 1.0]], dtype=jnp.float32),
        genotype_mean=jnp.asarray([0.5], dtype=jnp.float32),
        imputed_dosage_square_sum=None,
        sparse_candidate_mask=(
            None if sparse_candidate_mask is None else jnp.asarray(sparse_candidate_mask, dtype=jnp.bool_)
        ),
        packed8=False,
        raw_packed8_statistics=None,
    )


def test_binary_score_backend_matches_independent_oracle() -> None:
    fixture = tests.test_regenie2_binary.build_binary_fixture()
    config = tests.test_regenie2_binary.build_binary_score_config()
    reference = tests.test_regenie2_binary.compute_binary_score_reference(fixture, config)
    backend = build_binary_score_backend()
    group_state = backend.prepare_group(
        phenotype_matrix=np.asarray(fixture.phenotype_matrix, dtype=np.float32),
        covariate_matrix=np.asarray(fixture.covariate_matrix, dtype=np.float32),
        source_sample_count=None,
        selected_sample_count=None,
        selection_start=None,
        selected_sample_indices=None,
    )
    chromosome_state = backend.prepare_chromosome(
        group_state=group_state,
        prediction_matrix=np.asarray(fixture.loco_offset_matrix, dtype=np.float32),
    )
    batch = backend.transfer_batch(
        genotype_values=np.asarray(fixture.genotype_matrix_by_variant, dtype=np.float32),
        genotype_mean=np.asarray(np.mean(fixture.genotype_matrix_by_variant, axis=1), dtype=np.float32),
        imputed_dosage_square_sum=None,
        sparse_candidate_mask=None,
    )

    observed = backend.compute_batch(chromosome_state=chromosome_state, batch=batch)

    tests.numerical.assert_absolute_difference_less_than(
        observed.association.beta,
        reference.beta,
        tests.test_regenie2_binary.BINARY_BETA_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.association.standard_error,
        reference.standard_error,
        tests.test_regenie2_binary.BINARY_STANDARD_ERROR_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.association.chi_squared,
        reference.chi_squared,
        tests.test_regenie2_binary.BINARY_CHI_SQUARED_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        observed.association.log10_p_value,
        reference.log10_p_value,
        tests.test_regenie2_binary.BINARY_LOG10_P_VALUE_ABSOLUTE_TOLERANCE,
    )
    assert observed.association.correction_code is not None
    np.testing.assert_array_equal(
        np.asarray(observed.association.correction_code),
        np.zeros(reference.beta.shape, dtype=np.uint8),
    )


def test_binary_score_backend_packed8_route_matches_decoded_route() -> None:
    fixture = tests.test_regenie2_binary.build_binary_fixture()
    packed_probabilities = np.asarray(
        [
            [[255, 0], [220, 30], [150, 80], [90, 100], [40, 150], [0, 210], [120, 100], [20, 40], [180, 30], [70, 80]],
            [[0, 0], [20, 10], [60, 40], [90, 50], [110, 90], [150, 50], [180, 40], [240, 10], [30, 20], [100, 120]],
        ],
        dtype=np.uint8,
    )
    decoded_genotypes = (
        510.0
        - 2.0 * np.asarray(packed_probabilities[:, :, 0], dtype=np.float32)
        - np.asarray(packed_probabilities[:, :, 1], dtype=np.float32)
    ) / 255.0
    backend = build_binary_score_backend()
    group_state = backend.prepare_group(
        phenotype_matrix=np.asarray(fixture.phenotype_matrix, dtype=np.float32),
        covariate_matrix=np.asarray(fixture.covariate_matrix, dtype=np.float32),
        source_sample_count=None,
        selected_sample_count=None,
        selection_start=None,
        selected_sample_indices=None,
    )
    chromosome_state = backend.prepare_chromosome(
        group_state=group_state,
        prediction_matrix=np.asarray(fixture.loco_offset_matrix, dtype=np.float32),
    )
    packed_batch = backend.transfer_batch(
        genotype_values=packed_probabilities,
        genotype_mean=np.asarray(np.mean(decoded_genotypes, axis=1), dtype=np.float32),
        imputed_dosage_square_sum=None,
        sparse_candidate_mask=None,
    )
    decoded_batch = backend.transfer_batch(
        genotype_values=np.asarray(decoded_genotypes, dtype=np.float32),
        genotype_mean=np.asarray(np.mean(decoded_genotypes, axis=1), dtype=np.float32),
        imputed_dosage_square_sum=None,
        sparse_candidate_mask=None,
    )

    packed_result = backend.compute_batch(chromosome_state=chromosome_state, batch=packed_batch)
    decoded_result = backend.compute_batch(chromosome_state=chromosome_state, batch=decoded_batch)

    tests.numerical.assert_absolute_difference_less_than(
        packed_result.association.beta,
        decoded_result.association.beta,
        tests.test_regenie2_binary.BINARY_PACKED_BETA_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        packed_result.association.standard_error,
        decoded_result.association.standard_error,
        tests.test_regenie2_binary.BINARY_PACKED_STANDARD_ERROR_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        packed_result.association.chi_squared,
        decoded_result.association.chi_squared,
        tests.test_regenie2_binary.BINARY_PACKED_CHI_SQUARED_ABSOLUTE_TOLERANCE,
    )
    tests.numerical.assert_absolute_difference_less_than(
        packed_result.association.log10_p_value,
        decoded_result.association.log10_p_value,
        tests.test_regenie2_binary.BINARY_PACKED_LOG10_P_VALUE_ABSOLUTE_TOLERANCE,
    )
    np.testing.assert_array_equal(
        np.asarray(packed_result.association.correction_code),
        np.asarray(decoded_result.association.correction_code),
    )


def build_firth_backend(
    *,
    firth_maximum_iterations: int = 100,
) -> jax_backend.BinaryFirthJaxBackend:
    """Build a bounded CPU Firth adapter matching the independent oracle policy."""
    config = tests.test_regenie2_binary_pipeline.build_binary_kernel_config(candidate_capacity=2, batch_size=2)
    return jax_backend.BinaryFirthJaxBackend(
        p_threshold=1.0,
        firth_se=False,
        minimum_probability=config.numerical.minimum_probability,
        minimum_variance=config.numerical.minimum_variance,
        relative_variance_tolerance=config.numerical.relative_variance_tolerance,
        null_logistic_maximum_iterations=config.null_logistic.maximum_iterations,
        null_logistic_coefficient_tolerance=config.null_logistic.coefficient_tolerance,
        firth_batch_size=config.firth_candidate.batch_size,
        firth_candidate_capacity=config.firth_candidate.candidate_capacity,
        firth_maximum_iterations=firth_maximum_iterations,
        firth_gradient_tolerance=config.approximate_firth.gradient_tolerance,
        firth_maximum_step_size=config.approximate_firth.maximum_step_size,
        firth_pseudo_maximum_iterations=config.approximate_firth.pseudo_maximum_iterations,
        firth_pseudo_inner_maximum_iterations=config.approximate_firth.pseudo_inner_maximum_iterations,
        firth_line_search_maximum_attempts=config.approximate_firth.line_search_maximum_attempts,
        firth_sparse_carrier_dosage_threshold=config.approximate_firth.sparse_carrier_dosage_threshold,
        use_cuda_firth_components=False,
        null_firth_maximum_iterations=config.null_firth.maximum_iterations,
        null_firth_gradient_tolerance=config.null_firth.gradient_tolerance,
        null_firth_maximum_step_size=config.null_firth.maximum_step_size,
        null_firth_fallback_iteration_multiplier=config.null_firth.fallback_iteration_multiplier,
        null_firth_fallback_step_divisor=config.null_firth.fallback_step_divisor,
        null_firth_line_search_maximum_attempts=config.null_firth.line_search_maximum_attempts,
        null_firth_step_halving_scale=config.null_firth.step_halving_scale,
    )


def test_firth_backend_rejects_total_iteration_budget_below_four() -> None:
    """Defend direct Python construction from an unusable phase split."""
    with pytest.raises(ValueError, match="must be at least 4"):
        build_firth_backend(firth_maximum_iterations=3)


def run_firth_backend_fixture(
    fixture: tests.test_regenie2_binary_pipeline.FirthPipelineFixture,
    *,
    packed8: bool,
) -> jax_backend.DeviceAssociationBatch:
    """Run one Firth fixture through the production adapter and selected delivery route."""
    backend = build_firth_backend()
    group_state = backend.prepare_group(
        phenotype_matrix=np.asarray(fixture.phenotype_matrix, dtype=np.float32),
        covariate_matrix=np.asarray(fixture.covariate_matrix, dtype=np.float32),
        source_sample_count=None,
        selected_sample_count=None,
        selection_start=None,
        selected_sample_indices=None,
    )
    chromosome_state = backend.prepare_chromosome(
        group_state=group_state,
        prediction_matrix=np.asarray(fixture.loco_offset_matrix, dtype=np.float32),
    )
    genotype_values: npt.NDArray[np.float32] | npt.NDArray[np.uint8]
    if packed8:
        genotype_values = tests.test_regenie2_binary_pipeline.encode_integer_dosages_as_packed8(
            fixture.genotype_matrix_by_variant
        )
    else:
        genotype_values = np.asarray(fixture.genotype_matrix_by_variant, dtype=np.float32)
    genotype_mean = (
        np.asarray(np.mean(fixture.genotype_matrix_by_variant, axis=1), dtype=np.float32)
        if fixture.native_genotype_mean is None
        else np.asarray(fixture.native_genotype_mean, dtype=np.float32)
    )
    batch = backend.transfer_batch(
        genotype_values=genotype_values,
        genotype_mean=genotype_mean,
        imputed_dosage_square_sum=None,
        sparse_candidate_mask=fixture.sparse_candidate_mask,
    )
    return backend.compute_batch(chromosome_state=chromosome_state, batch=batch)


@pytest.mark.parametrize("packed8", [False, True])
def test_firth_backend_matches_independent_dense_and_sparse_oracles(*, packed8: bool) -> None:
    fixture = (
        tests.test_regenie2_binary_pipeline.build_packed_firth_pipeline_fixture()
        if packed8
        else tests.test_regenie2_binary_pipeline.build_firth_pipeline_fixture()
    )
    kernel_config = tests.test_regenie2_binary_pipeline.build_binary_kernel_config(
        candidate_capacity=2,
        batch_size=2,
    )
    prepared = tests.test_regenie2_binary_pipeline.prepare_firth_pipeline(
        fixture=fixture,
        kernel_config=kernel_config,
    )
    references = [
        [
            tests.test_regenie2_binary_pipeline.compute_firth_reference(
                prepared=prepared,
                trait_index=0,
                variant_index=0,
                sparse_correction=False,
            ),
            tests.test_regenie2_binary_pipeline.compute_firth_reference(
                prepared=prepared,
                trait_index=0,
                variant_index=1,
                sparse_correction=True,
            ),
        ]
    ]

    observed = run_firth_backend_fixture(fixture, packed8=packed8)

    assert int(np.asarray(observed.firth_candidate_count)) == 2
    assert observed.firth_candidate_capacity == 2
    assert observed.association.correction_code is not None
    assert observed.association.correction_code.dtype == jnp.uint8
    np.testing.assert_array_equal(
        np.asarray(observed.association.correction_code),
        np.full((1, 2), types.BinaryCorrectionCode.FIRTH_SUCCESS.value, dtype=np.uint8),
    )
    tests.test_regenie2_binary_pipeline.assert_firth_association_matches_references(
        typing.cast("regenie2_binary_result.Regenie2MultiBinaryScoreChunkResult", observed.association),
        references,
    )


def test_firth_backend_rejects_missing_sparse_candidate_mask() -> None:
    batch = build_decoded_genotype_batch(sparse_candidate_mask=None)

    with pytest.raises(ValueError, match="requires a sparse candidate mask"):
        build_firth_backend().compute_batch(
            chromosome_state=typing.cast("regenie2_binary_state.Regenie2MultiBinaryFirthChromosomeState", object()),
            batch=batch,
        )
