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


def test_multi_regenie_prediction_source_returns_trait_major_loco_matrix(tmp_path: Path) -> None:
    trait_a_loco_path = tmp_path / "trait_a.loco"
    trait_a_loco_path.write_text("FID_IID 0_A 0_B 0_C\nchr22 0.1 0.2 0.3\n")
    trait_b_loco_path = tmp_path / "trait_b.loco"
    trait_b_loco_path.write_text("FID_IID 0_A 0_B 0_C\nchr22 1.1 1.2 1.3\n")
    prediction_list_path = tmp_path / "pred.list"
    prediction_list_path.write_text(f"trait_a {trait_a_loco_path}\ntrait_b {trait_b_loco_path}\n")
    phenotype_path = tmp_path / "phenotypes.tsv"
    phenotype_path.write_text("IID\ttrait_a\ttrait_b\nC\t1.0\t2.0\nA\t3.0\t4.0\n")
    native_multi_aligned_sample_data = _core.align_multi_sample_data(
        np.asarray([0, 1], dtype=np.int64),
        ["0", "0"],
        ["C", "A"],
        str(phenotype_path),
        ["trait_a", "trait_b"],
    )

    prediction_source = _core.MultiRegeniePredictionSource.from_native_multi_aligned_sample_data(
        str(prediction_list_path),
        native_multi_aligned_sample_data,
    )

    np.testing.assert_allclose(
        prediction_source.get_chromosome_predictions("chr22"),
        np.asarray([[0.3, 0.1], [1.3, 1.1]], dtype=np.float32),
        atol=1e-6,
    )


def test_multi_regenie_prediction_source_reports_missing_phenotype(tmp_path: Path) -> None:
    trait_a_loco_path = tmp_path / "trait_a.loco"
    trait_a_loco_path.write_text("FID_IID 0_A\n22 0.1\n")
    prediction_list_path = tmp_path / "pred.list"
    prediction_list_path.write_text(f"trait_a {trait_a_loco_path}\n")

    with np.testing.assert_raises_regex(ValueError, "Phenotype 'trait_b' not found"):
        _core.MultiRegeniePredictionSource(
            str(prediction_list_path),
            ["trait_a", "trait_b"],
            ["0"],
            ["A"],
        )


def test_multi_regenie_prediction_source_reports_missing_chromosome(tmp_path: Path) -> None:
    trait_a_loco_path = tmp_path / "trait_a.loco"
    trait_a_loco_path.write_text("FID_IID 0_A\n22 0.1\n")
    trait_b_loco_path = tmp_path / "trait_b.loco"
    trait_b_loco_path.write_text("FID_IID 0_A\n22 1.1\n")
    prediction_list_path = tmp_path / "pred.list"
    prediction_list_path.write_text(f"trait_a {trait_a_loco_path}\ntrait_b {trait_b_loco_path}\n")
    prediction_source = _core.MultiRegeniePredictionSource(
        str(prediction_list_path),
        ["trait_a", "trait_b"],
        ["0"],
        ["A"],
    )

    with np.testing.assert_raises_regex(ValueError, "Chromosome '1'"):
        prediction_source.get_chromosome_predictions("1")


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


def test_regenie_prediction_source_rejects_duplicate_loco_iid_by_default(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID f1_s1 f2_s1\n22 0.1 0.2\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    with np.testing.assert_raises_regex(ValueError, "Duplicate LOCO IID 's1'"):
        _core.RegeniePredictionSource(
            str(prediction_list_path),
            "trait",
            ["f1"],
            ["s1"],
        )


def test_regenie_prediction_source_allows_duplicate_loco_iid_with_compatibility_flag(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID f1_s1 f2_s1\n22 0.1 0.2\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    prediction_source = _core.RegeniePredictionSource(
        str(prediction_list_path),
        "trait",
        ["f2"],
        ["s1"],
        allow_duplicate_iid_alignment=True,
    )

    np.testing.assert_allclose(prediction_source.get_chromosome_predictions("22"), [0.2], atol=1e-6)


def test_regenie_prediction_source_rejects_duplicate_exact_loco_key_even_with_flag(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID f1_s1 f1_s1\n22 0.1 0.2\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    with np.testing.assert_raises_regex(ValueError, "Duplicate LOCO sample key: f1_s1"):
        _core.RegeniePredictionSource(
            str(prediction_list_path),
            "trait",
            ["f1"],
            ["s1"],
            allow_duplicate_iid_alignment=True,
        )


def test_regenie_prediction_source_fid_iid_mode_aligns_repeated_iid(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID f1_s1 f2_s1\n22 0.1 0.2\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    prediction_source = _core.RegeniePredictionSource(
        str(prediction_list_path),
        "trait",
        ["f2", "f1"],
        ["s1", "s1"],
        sample_key_mode="fid_iid",
    )

    np.testing.assert_allclose(prediction_source.get_chromosome_predictions("22"), [0.2, 0.1], atol=1e-6)
