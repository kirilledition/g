
from __future__ import annotations
import itertools

import numpy as np

from g import models
from g.io import reader as genotype_reader_protocols


def split_dosage_genotype_chunk_by_chromosome(
    genotype_chunk: models.DosageGenotypeChunk,
) -> tuple[models.DosageGenotypeChunk, ...]:
    """Split one dosage chunk into chromosome-homogeneous subchunks."""
    chromosome_values = genotype_chunk.metadata.chromosome
    if chromosome_values.size == 0:
        return (genotype_chunk,)
    if np.all(chromosome_values == chromosome_values[0]):
        return (genotype_chunk,)

    chromosome_start_indices = [0]
    for variant_index in range(1, len(chromosome_values)):
        if chromosome_values[variant_index] != chromosome_values[variant_index - 1]:
            chromosome_start_indices.append(variant_index)
    chromosome_start_indices.append(len(chromosome_values))

    chromosome_subchunks: list[models.DosageGenotypeChunk] = []
    for start_index, stop_index in itertools.pairwise(chromosome_start_indices):
        chromosome_subchunks.append(
            models.DosageGenotypeChunk(
                genotypes=genotype_chunk.genotypes[:, start_index:stop_index],
                metadata=models.VariantMetadata(
                    variant_start_index=genotype_chunk.metadata.variant_start_index + start_index,
                    variant_stop_index=genotype_chunk.metadata.variant_start_index + stop_index,
                    chromosome=genotype_chunk.metadata.chromosome[start_index:stop_index],
                    variant_identifiers=genotype_chunk.metadata.variant_identifiers[start_index:stop_index],
                    position=genotype_chunk.metadata.position[start_index:stop_index],
                    allele_one=genotype_chunk.metadata.allele_one[start_index:stop_index],
                    allele_two=genotype_chunk.metadata.allele_two[start_index:stop_index],
                ),
                allele_one_frequency=genotype_chunk.allele_one_frequency[start_index:stop_index],
                observation_count=genotype_chunk.observation_count[start_index:stop_index],
            )
        )
    return tuple(chromosome_subchunks)


def split_dosage_genotype_chunk_by_absolute_variant_slices(
    genotype_chunk: models.DosageGenotypeChunk,
    variant_slices: tuple[tuple[int, int], ...],
) -> tuple[models.DosageGenotypeChunk, ...]:
    """Split one dosage chunk using absolute variant slice boundaries."""
    if not variant_slices:
        return ()
    if len(variant_slices) == 1:
        only_variant_start, only_variant_stop = variant_slices[0]
        if (
            only_variant_start == genotype_chunk.metadata.variant_start_index
            and only_variant_stop == genotype_chunk.metadata.variant_stop_index
        ):
            return (genotype_chunk,)

    chromosome_subchunks: list[models.DosageGenotypeChunk] = []
    for variant_start, variant_stop in variant_slices:
        relative_variant_start = variant_start - genotype_chunk.metadata.variant_start_index
        relative_variant_stop = variant_stop - genotype_chunk.metadata.variant_start_index
        chromosome_subchunks.append(
            models.DosageGenotypeChunk(
                genotypes=genotype_chunk.genotypes[:, relative_variant_start:relative_variant_stop],
                metadata=models.VariantMetadata(
                    variant_start_index=variant_start,
                    variant_stop_index=variant_stop,
                    chromosome=genotype_chunk.metadata.chromosome[relative_variant_start:relative_variant_stop],
                    variant_identifiers=genotype_chunk.metadata.variant_identifiers[
                        relative_variant_start:relative_variant_stop
                    ],
                    position=genotype_chunk.metadata.position[relative_variant_start:relative_variant_stop],
                    allele_one=genotype_chunk.metadata.allele_one[relative_variant_start:relative_variant_stop],
                    allele_two=genotype_chunk.metadata.allele_two[relative_variant_start:relative_variant_stop],
                ),
                allele_one_frequency=genotype_chunk.allele_one_frequency[relative_variant_start:relative_variant_stop],
                observation_count=genotype_chunk.observation_count[relative_variant_start:relative_variant_stop],
            )
        )
    return tuple(chromosome_subchunks)


def split_dosage_genotype_chunk_with_reader_metadata(
    genotype_chunk: models.DosageGenotypeChunk,
    genotype_reader: object,
) -> tuple[models.DosageGenotypeChunk, ...]:
    """Split one dosage chunk by chromosome, using reader metadata when available."""
    if isinstance(genotype_reader, genotype_reader_protocols.ChromosomePartitionReader):
        chromosome_variant_slices = genotype_reader.split_variant_slice_by_chromosome(
            genotype_chunk.metadata.variant_start_index,
            genotype_chunk.metadata.variant_stop_index,
        )
        return split_dosage_genotype_chunk_by_absolute_variant_slices(genotype_chunk, chromosome_variant_slices)
    return split_dosage_genotype_chunk_by_chromosome(genotype_chunk)
