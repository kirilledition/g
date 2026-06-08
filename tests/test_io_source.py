from __future__ import annotations

from pathlib import Path

import pytest

from g.io import source


def test_build_bgen_source_config_rejects_non_bgen_suffix() -> None:
    """Ensure source configs fail fast for non-BGEN paths."""
    with pytest.raises(ValueError, match=r"Expected a \.bgen source path"):
        source.build_bgen_source_config(Path("study.vcf"))


def test_build_bgen_source_config_prefers_explicit_sample_path(tmp_path: Path) -> None:
    """Ensure BGEN source configs keep the optional sample-file path."""
    bgen_path = tmp_path / "study.bgen"
    explicit_sample_path = tmp_path / "explicit.sample"
    adjacent_sample_path = tmp_path / "study.sample"
    adjacent_sample_path.write_text("", encoding="utf-8")

    genotype_source_config = source.build_bgen_source_config(bgen_path, sample_path=explicit_sample_path)

    assert genotype_source_config.source_path == bgen_path
    assert genotype_source_config.sample_path == explicit_sample_path
    assert genotype_source_config.resolved_sample_path == explicit_sample_path


def test_build_bgen_source_config_resolves_adjacent_sample_path(tmp_path: Path) -> None:
    """Ensure BGEN source configs use an adjacent sample file when present."""
    bgen_path = tmp_path / "study.bgen"
    adjacent_sample_path = tmp_path / "study.sample"
    adjacent_sample_path.write_text("", encoding="utf-8")

    genotype_source_config = source.build_bgen_source_config(bgen_path)

    assert genotype_source_config.sample_path is None
    assert genotype_source_config.resolved_sample_path == adjacent_sample_path


def test_build_bgen_source_config_allows_embedded_bgen_samples(tmp_path: Path) -> None:
    """Ensure BGEN source configs allow embedded samples without a sample file."""
    genotype_source_config = source.build_bgen_source_config(tmp_path / "study.bgen")

    assert genotype_source_config.sample_path is None
    assert genotype_source_config.resolved_sample_path is None
