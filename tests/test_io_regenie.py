"""Unit tests for REGENIE LOCO file parsers."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from g.io import regenie


class TestParsePredictionListFile:
    """Tests for parse_prediction_list_file."""

    def test_parses_single_entry(self, tmp_path: Path) -> None:
        """Ensure single-entry prediction list is parsed correctly."""
        prediction_list_path = tmp_path / "test_pred.list"
        prediction_list_path.write_text("phenotype1 /path/to/phenotype1.loco\n")

        entries = regenie.parse_prediction_list_file(prediction_list_path)

        assert len(entries) == 1
        assert entries[0].phenotype_name == "phenotype1"
        assert entries[0].loco_file_path == Path("/path/to/phenotype1.loco")

    def test_parses_multiple_entries(self, tmp_path: Path) -> None:
        """Ensure multiple-entry prediction list is parsed correctly."""
        prediction_list_path = tmp_path / "test_pred.list"
        prediction_list_path.write_text("phenotype1 /path/to/phenotype1.loco\nphenotype2 /path/to/phenotype2.loco\n")

        entries = regenie.parse_prediction_list_file(prediction_list_path)

        assert len(entries) == 2
        assert entries[0].phenotype_name == "phenotype1"
        assert entries[1].phenotype_name == "phenotype2"

    def test_raises_on_invalid_line_format(self, tmp_path: Path) -> None:
        """Ensure ValueError is raised for invalid line format."""
        prediction_list_path = tmp_path / "test_pred.list"
        prediction_list_path.write_text("phenotype1\n")

        with pytest.raises(ValueError, match="expected 2 space-delimited fields"):
            regenie.parse_prediction_list_file(prediction_list_path)

    def test_raises_on_empty_file(self, tmp_path: Path) -> None:
        """Ensure ValueError is raised for empty file."""
        prediction_list_path = tmp_path / "test_pred.list"
        prediction_list_path.write_text("")

        with pytest.raises(ValueError, match="empty"):
            regenie.parse_prediction_list_file(prediction_list_path)

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Ensure FileNotFoundError is raised for missing file."""
        prediction_list_path = tmp_path / "nonexistent.list"

        with pytest.raises(FileNotFoundError):
            regenie.parse_prediction_list_file(prediction_list_path)


class TestParseLocoSampleIdentifiers:
    """Tests for parse_loco_sample_identifiers."""

    def test_parses_fid_iid_format(self) -> None:
        """Ensure FID_IID format is correctly parsed."""
        header = "FID_IID 0_HG00096 0_HG00097 0_HG00099"

        sample_index = regenie.parse_loco_sample_identifiers(header)

        assert sample_index.loco_sample_count == 3
        np.testing.assert_array_equal(
            sample_index.family_identifiers,
            np.array(["0", "0", "0"], dtype=np.str_),
        )
        np.testing.assert_array_equal(
            sample_index.individual_identifiers,
            np.array(["HG00096", "HG00097", "HG00099"], dtype=np.str_),
        )

    def test_raises_on_missing_fid_iid_marker(self) -> None:
        """Ensure ValueError is raised when FID_IID marker is missing."""
        header = "WRONG_HEADER 0_HG00096 0_HG00097"

        with pytest.raises(ValueError, match="must start with 'FID_IID'"):
            regenie.parse_loco_sample_identifiers(header)

    def test_raises_on_invalid_sample_format(self) -> None:
        """Ensure ValueError is raised for sample identifiers without underscore."""
        header = "FID_IID HG00096 HG00097"

        with pytest.raises(ValueError, match="does not contain underscore separator"):
            regenie.parse_loco_sample_identifiers(header)


class TestNormalizeChromosome:
    """Tests for normalize_chromosome."""

    def test_strips_chr_prefix(self) -> None:
        """Ensure 'chr' prefix is stripped."""
        assert regenie.normalize_chromosome("chr22") == "22"
        assert regenie.normalize_chromosome("CHR22") == "22"
        assert regenie.normalize_chromosome("Chr1") == "1"

    def test_strips_leading_zeros(self) -> None:
        """Ensure leading zeros are stripped for numeric chromosomes."""
        assert regenie.normalize_chromosome("01") == "1"
        assert regenie.normalize_chromosome("02") == "2"

    def test_preserves_non_numeric(self) -> None:
        """Ensure non-numeric chromosomes are preserved."""
        assert regenie.normalize_chromosome("X") == "x"
        assert regenie.normalize_chromosome("chrX") == "x"


class TestParseLocoFile:
    """Tests for parse_loco_file."""

    def test_parses_valid_loco_file(self, tmp_path: Path) -> None:
        """Ensure valid LOCO file is parsed correctly."""
        loco_path = tmp_path / "test.loco"
        loco_path.write_text("FID_IID 0_SAMPLE1 0_SAMPLE2 0_SAMPLE3\n1 0.1 0.2 0.3\n22 0.4 0.5 0.6\n")

        predictions = regenie.parse_loco_file(loco_path)

        assert predictions.sample_index.loco_sample_count == 3
        assert "1" in predictions.chromosome_predictions
        assert "22" in predictions.chromosome_predictions
        np.testing.assert_allclose(
            predictions.chromosome_predictions["1"],
            np.array([0.1, 0.2, 0.3]),
        )
        np.testing.assert_allclose(
            predictions.chromosome_predictions["22"],
            np.array([0.4, 0.5, 0.6]),
        )

    def test_raises_on_duplicate_chromosome(self, tmp_path: Path) -> None:
        """Ensure ValueError is raised for duplicate chromosomes."""
        loco_path = tmp_path / "test.loco"
        loco_path.write_text("FID_IID 0_SAMPLE1 0_SAMPLE2\n1 0.1 0.2\n1 0.3 0.4\n")

        with pytest.raises(ValueError, match="duplicate chromosome"):
            regenie.parse_loco_file(loco_path)

    def test_raises_on_mismatched_prediction_count(self, tmp_path: Path) -> None:
        """Ensure ValueError is raised when prediction count doesn't match sample count."""
        loco_path = tmp_path / "test.loco"
        loco_path.write_text("FID_IID 0_SAMPLE1 0_SAMPLE2 0_SAMPLE3\n1 0.1 0.2\n")

        with pytest.raises(ValueError, match="expected 3 predictions"):
            regenie.parse_loco_file(loco_path)


class TestBuildSampleAlignmentIndices:
    """Tests for build_sample_alignment_indices."""

    def test_handles_same_order(self) -> None:
        """Ensure alignment works when orders match."""
        sample_index = regenie.LocoSampleIndex(
            family_identifiers=np.array(["0", "0", "0"], dtype=np.str_),
            individual_identifiers=np.array(["A", "B", "C"], dtype=np.str_),
            loco_sample_count=3,
        )

        indices = regenie.build_sample_alignment_indices(
            loco_sample_index=sample_index,
            target_family_identifiers=np.array(["0", "0", "0"], dtype=np.str_),
            target_individual_identifiers=np.array(["A", "B", "C"], dtype=np.str_),
        )

        np.testing.assert_array_equal(indices, np.array([0, 1, 2]))

    def test_handles_reordering(self) -> None:
        """Ensure alignment works when orders differ."""
        sample_index = regenie.LocoSampleIndex(
            family_identifiers=np.array(["0", "0", "0"], dtype=np.str_),
            individual_identifiers=np.array(["A", "B", "C"], dtype=np.str_),
            loco_sample_count=3,
        )

        indices = regenie.build_sample_alignment_indices(
            loco_sample_index=sample_index,
            target_family_identifiers=np.array(["0", "0", "0"], dtype=np.str_),
            target_individual_identifiers=np.array(["C", "A", "B"], dtype=np.str_),
        )

        np.testing.assert_array_equal(indices, np.array([2, 0, 1]))

    def test_raises_on_missing_samples(self) -> None:
        """Ensure ValueError is raised when target samples are missing from LOCO."""
        sample_index = regenie.LocoSampleIndex(
            family_identifiers=np.array(["0", "0"], dtype=np.str_),
            individual_identifiers=np.array(["A", "B"], dtype=np.str_),
            loco_sample_count=2,
        )

        with pytest.raises(ValueError, match="not found in LOCO file"):
            regenie.build_sample_alignment_indices(
                loco_sample_index=sample_index,
                target_family_identifiers=np.array(["0", "0"], dtype=np.str_),
                target_individual_identifiers=np.array(["A", "MISSING"], dtype=np.str_),
            )


class TestLoadAlignedChromosomePredictions:
    """Tests for load_aligned_chromosome_predictions."""

    def test_returns_jax_array(self, tmp_path: Path) -> None:
        """Ensure aligned predictions are returned as JAX array."""
        loco_path = tmp_path / "test.loco"
        loco_path.write_text("FID_IID 0_A 0_B 0_C\n22 0.1 0.2 0.3\n")
        predictions = regenie.parse_loco_file(loco_path)

        aligned = regenie.load_aligned_chromosome_predictions(
            loco_predictions=predictions,
            chromosome="22",
            target_family_identifiers=np.array(["0", "0"], dtype=np.str_),
            target_individual_identifiers=np.array(["C", "A"], dtype=np.str_),
        )

        assert isinstance(aligned, jnp.ndarray)
        np.testing.assert_allclose(aligned, np.array([0.3, 0.1]), atol=1e-6)

    def test_raises_on_missing_chromosome(self, tmp_path: Path) -> None:
        """Ensure KeyError is raised for missing chromosome."""
        loco_path = tmp_path / "test.loco"
        loco_path.write_text("FID_IID 0_A 0_B\n22 0.1 0.2\n")
        predictions = regenie.parse_loco_file(loco_path)

        with pytest.raises(KeyError, match="not found in LOCO file"):
            regenie.load_aligned_chromosome_predictions(
                loco_predictions=predictions,
                chromosome="1",
                target_family_identifiers=np.array(["0"], dtype=np.str_),
                target_individual_identifiers=np.array(["A"], dtype=np.str_),
            )


class TestLoadPredictionSource:
    """Tests for load_prediction_source."""

    def test_loads_prediction_source(self, tmp_path: Path) -> None:
        """Ensure prediction source is loaded correctly."""
        loco_path = tmp_path / "test.loco"
        loco_path.write_text("FID_IID 0_A 0_B\n22 0.1 0.2\n")

        prediction_list_path = tmp_path / "test_pred.list"
        prediction_list_path.write_text(f"my_phenotype {loco_path}\n")

        source = regenie.load_prediction_source(
            prediction_list_path=prediction_list_path,
            phenotype_name="my_phenotype",
        )

        predictions = source.get_chromosome_predictions(
            chromosome="22",
            sample_family_identifiers=np.array(["0", "0"], dtype=np.str_),
            sample_individual_identifiers=np.array(["B", "A"], dtype=np.str_),
        )

        np.testing.assert_allclose(predictions, np.array([0.2, 0.1]), atol=1e-6)

    def test_raises_on_missing_phenotype(self, tmp_path: Path) -> None:
        """Ensure ValueError is raised for missing phenotype."""
        loco_path = tmp_path / "test.loco"
        loco_path.write_text("FID_IID 0_A\n22 0.1\n")

        prediction_list_path = tmp_path / "test_pred.list"
        prediction_list_path.write_text(f"other_phenotype {loco_path}\n")

        with pytest.raises(ValueError, match="not found in prediction list"):
            regenie.load_prediction_source(
                prediction_list_path=prediction_list_path,
                phenotype_name="missing_phenotype",
            )


class TestWithRegenieBaselines:
    """Integration tests using actual REGENIE baseline files."""

    @pytest.fixture
    def baseline_paths(self) -> dict[str, Path]:
        """Return paths to baseline files if they exist."""
        baselines_dir = Path("/home/kirill/Projects/g/data/baselines")
        prediction_list = baselines_dir / "regenie_step1_qt_pred.list"
        if not prediction_list.exists():
            pytest.skip("REGENIE baseline files not available")
        return {
            "prediction_list": prediction_list,
            "loco_file": baselines_dir / "regenie_step1_qt_1.loco",
        }

    def test_parses_baseline_prediction_list(self, baseline_paths: dict[str, Path]) -> None:
        """Ensure baseline _pred.list file is parsed correctly."""
        entries = regenie.parse_prediction_list_file(baseline_paths["prediction_list"])

        assert len(entries) == 1
        assert entries[0].phenotype_name == "phenotype_continuous"

    def test_parses_baseline_loco_file(self, baseline_paths: dict[str, Path]) -> None:
        """Ensure baseline .loco file is parsed correctly."""
        predictions = regenie.parse_loco_file(baseline_paths["loco_file"])

        assert predictions.sample_index.loco_sample_count == 2504
        assert "22" in predictions.chromosome_predictions
        assert predictions.chromosome_predictions["22"].shape == (2504,)

    def test_loads_prediction_source_from_baselines(self, baseline_paths: dict[str, Path]) -> None:
        """Ensure prediction source loads correctly from baselines."""
        source = regenie.load_prediction_source(
            prediction_list_path=baseline_paths["prediction_list"],
            phenotype_name="phenotype_continuous",
        )

        first_three_fids = np.array(["0", "0", "0"], dtype=np.str_)
        first_three_iids = np.array(["HG00096", "HG00097", "HG00099"], dtype=np.str_)

        predictions = source.get_chromosome_predictions(
            chromosome="22",
            sample_family_identifiers=first_three_fids,
            sample_individual_identifiers=first_three_iids,
        )

        assert predictions.shape == (3,)
        assert predictions.dtype == jnp.float32
