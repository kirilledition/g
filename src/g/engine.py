"""High-level orchestration for REGENIE step 2 association runs."""

from __future__ import annotations

import itertools
import typing

import jax
import jax.profiler
import numpy as np

from g import models, types
from g.compute import regenie2_binary, regenie2_linear
from g.io import reader as genotype_reader_protocols
from g.io import regenie, source
from g.output import schema

if typing.TYPE_CHECKING:
    import collections.abc
    from pathlib import Path



BinaryChunkComputeFunction = typing.Callable[
    [models.Regenie2BinaryChromosomeState, jax.Array, types.RegenieBinaryCorrection],
    models.Regenie2BinaryChunkResult,
]


open_genotype_reader = source.open_genotype_reader
load_aligned_sample_data_from_source = source.load_aligned_sample_data_from_source
iter_dosage_genotype_chunks_from_source = source.iter_dosage_genotype_chunks_from_source
load_prediction_source = regenie.load_prediction_source
prepare_regenie2_linear_state = regenie2_linear.prepare_regenie2_linear_state
prepare_regenie2_linear_chromosome_state = regenie2_linear.prepare_regenie2_linear_chromosome_state
compute_regenie2_linear_chunk = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state
prepare_regenie2_binary_state = regenie2_binary.prepare_regenie2_binary_state
prepare_regenie2_binary_chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state
ChunkAccumulator = schema.AssociationChunkResult

compute_regenie2_binary_chunk = typing.cast(
    "BinaryChunkComputeFunction",
    regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state,
)


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


def iter_regenie2_linear_output_frames(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    prefetch_chunks: int = 0,
    committed_chunk_identifiers: set[int] | None = None,
) -> collections.abc.Iterator[ChunkAccumulator]:
    """Yield REGENIE step 2 linear chunk accumulators."""
    if genotype_source_config.source_format != types.GenotypeSourceFormat.BGEN:
        message = "REGENIE step 2 linear association requires a BGEN genotype source."
        raise ValueError(message)

    genotype_reader = open_genotype_reader(genotype_source_config)
    committed_identifier_set = committed_chunk_identifiers or set()

    with genotype_reader:
        with jax.profiler.TraceAnnotation("regenie2_linear.load_aligned_sample_data"):
            aligned_sample_data = load_aligned_sample_data_from_source(
                genotype_source_config=genotype_source_config,
                phenotype_path=phenotype_path,
                phenotype_name=phenotype_name,
                covariate_path=covariate_path,
                covariate_names=covariate_names,
                is_binary_trait=False,
                genotype_reader=genotype_reader,
            )
        with jax.profiler.TraceAnnotation("regenie2_linear.prepare_state"):
            regenie2_linear_state = prepare_regenie2_linear_state(
                covariate_matrix=aligned_sample_data.covariate_matrix,
                phenotype_vector=aligned_sample_data.phenotype_vector,
            )
        with jax.profiler.TraceAnnotation("regenie2_linear.load_prediction_source"):
            prediction_source = load_prediction_source(prediction_list_path, phenotype_name)

        chunk_iterator = iter_dosage_genotype_chunks_from_source(
            genotype_source_config=genotype_source_config,
            sample_indices=aligned_sample_data.sample_indices,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            prefetch_chunks=prefetch_chunks,
            genotype_reader=genotype_reader,
        )

        current_chromosome: str | None = None
        current_loco_predictions: jax.Array | None = None
        current_regenie2_linear_chromosome_state: models.Regenie2LinearChromosomeState | None = None
        chunk_number = 0
        for source_chunk in chunk_iterator:
            for current_chunk in split_dosage_genotype_chunk_with_reader_metadata(source_chunk, genotype_reader):
                chunk_identifier = current_chunk.metadata.variant_start_index
                if chunk_identifier in committed_identifier_set:
                    continue

                chromosome = str(current_chunk.metadata.chromosome[0])
                if chromosome != current_chromosome:
                    current_loco_predictions = prediction_source.get_chromosome_predictions(
                        chromosome=chromosome,
                        sample_family_identifiers=aligned_sample_data.family_identifiers,
                        sample_individual_identifiers=aligned_sample_data.individual_identifiers,
                    )
                    current_regenie2_linear_chromosome_state = prepare_regenie2_linear_chromosome_state(
                        regenie2_linear_state,
                        current_loco_predictions,
                    )
                    current_chromosome = chromosome

                assert current_loco_predictions is not None
                assert current_regenie2_linear_chromosome_state is not None
                with jax.profiler.StepTraceAnnotation("regenie2_linear_chunk", step_num=chunk_number):
                    with jax.profiler.TraceAnnotation("regenie2_linear.compute"):
                        regenie2_linear_result = compute_regenie2_linear_chunk(
                            chromosome_state=current_regenie2_linear_chromosome_state,
                            genotype_matrix=current_chunk.genotypes,
                        )
                    with jax.profiler.TraceAnnotation("regenie2_linear.accumulate"):
                        yield ChunkAccumulator(
                            metadata=current_chunk.metadata,
                            allele_one_frequency=current_chunk.allele_one_frequency,
                            observation_count=current_chunk.observation_count,
                            beta=regenie2_linear_result.beta,
                            standard_error=regenie2_linear_result.standard_error,
                            chi_squared=regenie2_linear_result.chi_squared,
                            log10_p_value=regenie2_linear_result.log10_p_value,
                            extra_code=None,
                        )
                chunk_number += 1


def iter_regenie2_binary_output_frames(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    prefetch_chunks: int = 0,
    committed_chunk_identifiers: set[int] | None = None,
    correction: types.RegenieBinaryCorrection = types.RegenieBinaryCorrection.FIRTH_APPROXIMATE,
) -> collections.abc.Iterator[ChunkAccumulator]:
    """Yield REGENIE step 2 binary chunk accumulators."""
    if genotype_source_config.source_format != types.GenotypeSourceFormat.BGEN:
        message = "REGENIE step 2 binary association requires a BGEN genotype source."
        raise ValueError(message)

    genotype_reader = open_genotype_reader(genotype_source_config)
    committed_identifier_set = committed_chunk_identifiers or set()

    with genotype_reader:
        with jax.profiler.TraceAnnotation("regenie2_binary.load_aligned_sample_data"):
            aligned_sample_data = load_aligned_sample_data_from_source(
                genotype_source_config=genotype_source_config,
                phenotype_path=phenotype_path,
                phenotype_name=phenotype_name,
                covariate_path=covariate_path,
                covariate_names=covariate_names,
                is_binary_trait=True,
                genotype_reader=genotype_reader,
            )
        with jax.profiler.TraceAnnotation("regenie2_binary.prepare_state"):
            regenie2_binary_state = prepare_regenie2_binary_state(
                covariate_matrix=aligned_sample_data.covariate_matrix,
                phenotype_vector=aligned_sample_data.phenotype_vector,
            )
        with jax.profiler.TraceAnnotation("regenie2_binary.load_prediction_source"):
            prediction_source = load_prediction_source(prediction_list_path, phenotype_name)

        chunk_iterator = iter_dosage_genotype_chunks_from_source(
            genotype_source_config=genotype_source_config,
            sample_indices=aligned_sample_data.sample_indices,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            prefetch_chunks=prefetch_chunks,
            genotype_reader=genotype_reader,
        )

        current_chromosome: str | None = None
        current_regenie2_binary_chromosome_state: models.Regenie2BinaryChromosomeState | None = None
        chunk_number = 0
        for source_chunk in chunk_iterator:
            for current_chunk in split_dosage_genotype_chunk_with_reader_metadata(source_chunk, genotype_reader):
                chunk_identifier = current_chunk.metadata.variant_start_index
                if chunk_identifier in committed_identifier_set:
                    continue

                chromosome = str(current_chunk.metadata.chromosome[0])
                if chromosome != current_chromosome:
                    loco_offset = prediction_source.get_chromosome_predictions(
                        chromosome=chromosome,
                        sample_family_identifiers=aligned_sample_data.family_identifiers,
                        sample_individual_identifiers=aligned_sample_data.individual_identifiers,
                    )
                    current_regenie2_binary_chromosome_state = prepare_regenie2_binary_chromosome_state(
                        regenie2_binary_state,
                        loco_offset,
                    )
                    current_chromosome = chromosome

                assert current_regenie2_binary_chromosome_state is not None
                with jax.profiler.StepTraceAnnotation("regenie2_binary_chunk", step_num=chunk_number):
                    with jax.profiler.TraceAnnotation("regenie2_binary.compute"):
                        regenie2_binary_result = compute_regenie2_binary_chunk(
                            current_regenie2_binary_chromosome_state,
                            current_chunk.genotypes,
                            correction,
                        )
                    with jax.profiler.TraceAnnotation("regenie2_binary.accumulate"):
                        yield ChunkAccumulator(
                            metadata=current_chunk.metadata,
                            allele_one_frequency=current_chunk.allele_one_frequency,
                            observation_count=current_chunk.observation_count,
                            beta=regenie2_binary_result.beta,
                            standard_error=regenie2_binary_result.standard_error,
                            chi_squared=regenie2_binary_result.chi_squared,
                            log10_p_value=regenie2_binary_result.log10_p_value,
                            extra_code=regenie2_binary_result.extra_code,
                        )
                chunk_number += 1
