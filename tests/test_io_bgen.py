from pathlib import Path

import numpy as np
import pytest
from cbgen import example

from g.io.bgen import iter_genotype_chunks, iter_linear_genotype_chunks, validate_bgen_sample_order


def test_iter_genotype_chunks():
    bgen_path = Path(example.get("haplotypes.bgen"))

    # haplotypes.bgen has 4 samples
    sample_indices = np.arange(4)
    # The sample names are sample_0, sample_1, ...
    expected_ids = np.array([f"sample_{i}" for i in range(4)], dtype=object)

    chunks = list(
        iter_genotype_chunks(
            bgen_path=bgen_path,
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_ids,
            chunk_size=2,
            variant_limit=10,
        )
    )

    # 4 variants total, chunk size 2 -> 2 chunks
    assert len(chunks) == 2
    for chunk in chunks:
        # shape is (num_samples, chunk_size) -> (4, 2)
        assert chunk.genotypes.shape == (4, 2)
        assert chunk.metadata.chromosome.shape == (2,)


def test_iter_linear_genotype_chunks():
    bgen_path = Path(example.get("haplotypes.bgen"))

    sample_indices = np.arange(4)
    expected_ids = np.array([f"sample_{i}" for i in range(4)], dtype=object)

    chunks = list(
        iter_linear_genotype_chunks(
            bgen_path=bgen_path,
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_ids,
            chunk_size=3,
            variant_limit=7,
        )
    )

    assert len(chunks) == 2
    assert chunks[0].genotypes.shape == (4, 3)
    assert chunks[1].genotypes.shape == (4, 1)


def test_validate_bgen_sample_order_failure():
    bgen_path = Path(example.get("haplotypes.bgen"))
    sample_indices = np.arange(4)
    # Give wrong IDs to test validation error
    expected_ids = np.array(["wrong1", "wrong2", "wrong3", "wrong4"], dtype=object)

    with pytest.raises(ValueError, match="BGEN sample order does not match"):
        # We need to call it inside a context that opens the bgen file
        from bgen_reader import open_bgen

        with open_bgen(str(bgen_path), verbose=False) as bgen_handle:
            validate_bgen_sample_order(bgen_handle, sample_indices, expected_ids, bgen_path)
