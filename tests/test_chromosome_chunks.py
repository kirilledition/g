import jax.numpy as jnp
import numpy as np

from g import models
from g.engine import chromosome_chunks


def test_split_dosage_genotype_chunk_by_chromosome_splits_transition() -> None:
    chunk = models.DosageGenotypeChunk(
        genotypes=jnp.asarray([[0.0, 1.0, 2.0]], dtype=jnp.float32),
        metadata=models.VariantMetadata(
            variant_start_index=10,
            variant_stop_index=13,
            chromosome=np.asarray(["1", "1", "2"]),
            variant_identifiers=np.asarray(["a", "b", "c"]),
            position=np.asarray([100, 101, 102], dtype=np.int64),
            allele_one=np.asarray(["A", "A", "C"]),
            allele_two=np.asarray(["G", "G", "T"]),
        ),
        allele_one_frequency=jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float32),
        observation_count=jnp.asarray([10, 10, 10], dtype=jnp.int32),
    )

    subchunks = chromosome_chunks.split_dosage_genotype_chunk_by_chromosome(chunk)

    assert len(subchunks) == 2
    assert subchunks[0].metadata.variant_start_index == 10
    assert subchunks[0].metadata.variant_stop_index == 12
    assert subchunks[1].metadata.variant_start_index == 12
    assert subchunks[1].metadata.variant_stop_index == 13
