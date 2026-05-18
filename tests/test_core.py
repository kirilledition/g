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
    reader = _core.BgenReader(str(HAPLOTYPES_BGEN_PATH))
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


def test_preprocessed_bgen_trusted_fast_path_matches_safe_stats() -> None:
    safe_reader = _core.BgenReader(str(HAPLOTYPES_BGEN_PATH))
    trusted_reader = _core.BgenReader(str(HAPLOTYPES_BGEN_PATH), trusted_no_missing_diploid=True)
    sample_indices = np.array([0, 2, 3], dtype=np.int64)
    safe_output_array = np.empty((3, 3), dtype=np.float32, order="C")
    trusted_output_array = np.empty((3, 3), dtype=np.float32, order="C")
    try:
        safe_reader.prepare_sample_selection(sample_indices)
        trusted_reader.prepare_sample_selection(sample_indices)
        safe_stats = safe_reader.read_preprocessed_dosage_f32_into_prepared(1, 4, safe_output_array)
        trusted_stats = trusted_reader.read_preprocessed_dosage_f32_into_prepared(1, 4, trusted_output_array)
    finally:
        safe_reader.clear_prepared_sample_selection()
        trusted_reader.clear_prepared_sample_selection()

    np.testing.assert_allclose(trusted_output_array, safe_output_array)
    np.testing.assert_allclose(trusted_stats.allele_one_frequency, safe_stats.allele_one_frequency)
    np.testing.assert_array_equal(trusted_stats.observation_count, safe_stats.observation_count)
    assert trusted_stats.has_missing_values == safe_stats.has_missing_values


def test_regenie2_run_engine_calls_callback_for_planned_bgen_chunks() -> None:
    class RecordingCallback:
        def __init__(self) -> None:
            self.chunk_shapes: list[tuple[int, int, int]] = []

        def compute_chunk(
            self,
            metadata: _core.VariantMetadata,
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
    engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)

    processed_chunk_count = engine.run_bgen_chunks(
        np.arange(4, dtype=np.int64),
        callback,
        committed_chunk_identifiers=[0],
    )

    assert processed_chunk_count == 1
    assert callback.chunk_shapes == [(2, 4, 2)]


def test_regenie2_run_engine_buffered_chunks_deliver_preprocessed_dosage_chunks() -> None:
    class RecordingCallback:
        def __init__(self) -> None:
            self.chunk_shapes: list[tuple[int, int, int]] = []
            self.free_buffers: list[np.ndarray] = []

        def acquire_dosage_buffer(self, sample_count: int, variant_count: int) -> np.ndarray:
            if self.free_buffers:
                return self.free_buffers.pop()
            return np.empty((sample_count, variant_count), dtype=np.float32, order="C")

        def compute_preprocessed_dosage_chunk(
            self,
            metadata: _core.VariantMetadata,
            genotype_matrix: np.ndarray,
            chunk_stats: _core.ChunkStats,
        ) -> None:
            self.chunk_shapes.append(
                (
                    metadata.variant_start_index,
                    genotype_matrix.shape[0],
                    genotype_matrix.shape[1],
                )
            )
            assert not np.isnan(genotype_matrix).any()
            np.testing.assert_allclose(chunk_stats.allele_one_frequency, genotype_matrix.mean(axis=0) / 2.0)
            np.testing.assert_array_equal(chunk_stats.observation_count, np.full(genotype_matrix.shape[1], 4))
            self.free_buffers.append(genotype_matrix)

    callback = RecordingCallback()
    engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)

    processed_chunk_count = engine.run_bgen_dosage_buffered_chunks(
        np.arange(4, dtype=np.int64),
        callback,
    )

    assert processed_chunk_count == 2
    assert callback.chunk_shapes == [(0, 4, 2), (2, 4, 2)]


def test_regenie_prediction_source_loads_aligned_loco_predictions(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID 0_A 0_B 0_C\nchr22 0.1 0.2 0.3\n01 1.0 2.0 3.0\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    prediction_source = _core.RegeniePredictionSource(
        str(prediction_list_path),
        "trait",
        ["0", "0"],
        ["C", "A"],
    )

    assert prediction_source.get_chromosome_predictions("22").dtype == np.float32
    np.testing.assert_allclose(prediction_source.get_chromosome_predictions("chr22"), [0.3, 0.1], atol=1e-6)
    np.testing.assert_allclose(prediction_source.get_chromosome_predictions("1"), [3.0, 1.0], atol=1e-6)


def test_regenie_prediction_source_loads_from_native_aligned_sample_data(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID 0_A 0_B 0_C\nchr22 0.1 0.2 0.3\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")
    phenotype_path = tmp_path / "phenotypes.tsv"
    phenotype_path.write_text("IID\ttrait\nC\t1.0\nA\t2.0\n")
    native_aligned_sample_data = _core.align_sample_data(
        np.asarray([0, 1], dtype=np.int64),
        ["0", "0"],
        ["C", "A"],
        str(phenotype_path),
        "trait",
    )

    prediction_source = _core.RegeniePredictionSource.from_native_aligned_sample_data(
        str(prediction_list_path),
        "trait",
        native_aligned_sample_data,
    )

    np.testing.assert_allclose(prediction_source.get_chromosome_predictions("chr22"), [0.3, 0.1], atol=1e-6)


def test_regenie_prediction_source_reports_missing_samples(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID 0_A\n22 0.1\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    with np.testing.assert_raises_regex(ValueError, "Target samples not found in LOCO file"):
        _core.RegeniePredictionSource(
            str(prediction_list_path),
            "trait",
            ["0"],
            ["missing"],
        )
