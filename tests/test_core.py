from __future__ import annotations

from pathlib import Path

import numpy as np

from g import _core

TEST_DATA_DIRECTORY = Path(__file__).parent / "data" / "bgen"
HAPLOTYPES_BGEN_PATH = TEST_DATA_DIRECTORY / "haplotypes.bgen"


def test_hello_from_bin_returns_expected_message() -> None:
    """Ensure the extension module exports a simple health-check string."""
    assert _core.hello_from_bin() == "Hello from g!"


def test_plan_genotype_chunks_splits_by_boundaries_and_resume_state() -> None:
    """Ensure the native chunk planner returns chromosome-homogeneous work units."""
    chunks = _core.plan_genotype_chunks(
        variant_count=12,
        chunk_size=5,
        chromosome_boundary_indices=[0, 3, 9, 12],
        committed_chunk_identifiers=[5],
    )

    assert [(chunk.variant_start_index, chunk.variant_stop_index) for chunk in chunks] == [
        (0, 3),
        (3, 5),
        (9, 10),
        (10, 12),
    ]


def test_preprocessed_bgen_read_fills_output_and_returns_stats() -> None:
    reader = _core.PyBgenReader(str(HAPLOTYPES_BGEN_PATH))
    sample_indices = np.array([0, 2, 3], dtype=np.int64)
    output_array = np.empty((3, 3), dtype=np.float32, order="C")
    try:
        reader.prepare_sample_selection(sample_indices)
        stats = reader.read_preprocessed_dosage_f32_into_prepared(1, 4, output_array)
    finally:
        reader.clear_prepared_sample_selection()

    np.testing.assert_allclose(
        output_array,
        np.array(
            [
                [1.0, 1.0, 2.0],
                [2.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(stats.allele_one_frequency, output_array.mean(axis=0) / 2.0)
    np.testing.assert_array_equal(stats.observation_count, np.array([3, 3, 3], dtype=np.int32))
    assert stats.has_missing_values is False


def test_regenie2_run_engine_calls_callback_for_planned_bgen_chunks() -> None:
    class RecordingCallback:
        def __init__(self) -> None:
            self.chunk_shapes: list[tuple[int, int, int]] = []

        def compute_chunk(
            self,
            metadata: _core.PyVariantMetadata,
            genotype_matrix: np.ndarray,
            allele_one_frequency: np.ndarray,
            observation_count: np.ndarray,
        ) -> None:
            self.chunk_shapes.append(
                (
                    metadata.variant_start_index,
                    genotype_matrix.shape[0],
                    genotype_matrix.shape[1],
                )
            )
            assert allele_one_frequency.shape == (genotype_matrix.shape[1],)
            assert observation_count.shape == (genotype_matrix.shape[1],)

    callback = RecordingCallback()
    engine = _core.PyRegenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)

    processed_chunk_count = engine.run_bgen_chunks(
        np.arange(4, dtype=np.int64),
        callback,
        committed_chunk_identifiers=[0],
    )

    assert processed_chunk_count == 1
    assert callback.chunk_shapes == [(2, 4, 2)]
