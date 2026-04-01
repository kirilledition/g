from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from g.io.source import (
    build_bgen_source_config,
    build_genotype_source_signature_paths,
    build_plink_source_config,
    iter_genotype_chunks_from_source,
    resolve_genotype_source_config,
)
from g.models import GenotypeChunk, VariantMetadata


def build_chunk(variant_start_index: int) -> GenotypeChunk:
    """Build a small chunk fixture for source tests."""
    return GenotypeChunk(
        genotypes=jnp.array([[0.0], [1.0]]),
        missing_mask=jnp.array([[False], [False]]),
        has_missing_values=False,
        metadata=VariantMetadata(
            variant_start_index=variant_start_index,
            variant_stop_index=variant_start_index + 1,
            chromosome=np.array(["1"]),
            variant_identifiers=np.array([f"variant{variant_start_index}"]),
            position=np.array([100 + variant_start_index], dtype=np.int64),
            allele_one=np.array(["A"]),
            allele_two=np.array(["G"]),
        ),
        allele_one_frequency=jnp.array([0.25], dtype=jnp.float32),
        observation_count=jnp.array([2], dtype=jnp.int32),
    )


def test_resolve_genotype_source_config_requires_exactly_one_source() -> None:
    """Ensure the public source resolver rejects ambiguous inputs."""
    with pytest.raises(ValueError, match="Exactly one genotype source"):
        resolve_genotype_source_config(None, None)
    with pytest.raises(ValueError, match="Exactly one genotype source"):
        resolve_genotype_source_config("dataset", "dataset.bgen")


def test_build_genotype_source_signature_paths_supports_both_formats() -> None:
    """Ensure reproducibility signatures include the right source files."""
    plink_paths = build_genotype_source_signature_paths(build_plink_source_config(Path("dataset")))
    bgen_paths = build_genotype_source_signature_paths(build_bgen_source_config(Path("dataset.bgen")))

    assert plink_paths == (Path("dataset.bed"), Path("dataset.bim"), Path("dataset.fam"))
    assert bgen_paths == (Path("dataset.bgen"),)

def test_iter_genotype_chunks_from_source_dispatches_to_bgen_reader() -> None:
    """Ensure the shared source iterator dispatches through the BGEN backend."""
    bgen_source_config = build_bgen_source_config(Path("study.bgen"))
    expected_chunk = build_chunk(0)

    with patch("g.io.source.iter_bgen_genotype_chunks", return_value=iter([expected_chunk])) as mock_iter_bgen:
        chunks = list(
            iter_genotype_chunks_from_source(
                genotype_source_config=bgen_source_config,
                sample_indices=np.array([0, 1], dtype=np.int64),
                expected_individual_identifiers=np.array(["sample0", "sample1"]),
                chunk_size=64,
                variant_limit=1,
            )
        )

    assert [chunk.metadata.variant_identifiers.tolist() for chunk in chunks] == [["variant0"]]
    mock_iter_bgen.assert_called_once()
