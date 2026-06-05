from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from g.io import source


def test_resolve_genotype_source_config_requires_bgen_source() -> None:
    """Ensure the source resolver requires a BGEN input."""
    with pytest.raises(ValueError, match="BGEN source"):
        source.resolve_genotype_source_config(None)


def test_build_genotype_source_signature_paths_uses_bgen_and_sample() -> None:
    """Ensure reproducibility signatures include BGEN and optional sample files."""
    bgen_paths = source.build_genotype_source_signature_paths(source.build_bgen_source_config(Path("dataset.bgen")))

    with patch("g.io.source.resolve_bgen_sample_path", return_value=Path("dataset.sample")):
        bgen_sample_paths = source.build_genotype_source_signature_paths(
            source.build_bgen_source_config(Path("dataset.bgen"), sample_path=Path("dataset.sample"))
        )

    assert bgen_paths == (Path("dataset.bgen"),)
    assert bgen_sample_paths == (Path("dataset.bgen"), Path("dataset.sample"))


def test_validate_genotype_source_config_rejects_non_bgen_suffix() -> None:
    """Ensure source configs fail fast for non-BGEN paths."""
    with pytest.raises(ValueError, match=r"Expected a \.bgen source path"):
        source.validate_genotype_source_config(source.GenotypeSourceConfig(source_path=Path("study.vcf")))


def test_build_bgen_source_config_preserves_sample_path() -> None:
    """Ensure BGEN source configs keep the optional sample-file path."""
    genotype_source_config = source.build_bgen_source_config(Path("study.bgen"), sample_path=Path("study.sample"))

    assert genotype_source_config.sample_path == Path("study.sample")
