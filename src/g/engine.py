"""High-level orchestration for REGENIE step 2 association runs."""

from __future__ import annotations

import itertools
import typing
from dataclasses import dataclass

import jax
import jax.profiler
import numpy as np
import polars as pl

from g import models, types
from g.compute import regenie2_linear
from g.io import reader as genotype_reader_protocols
from g.io import regenie, source

if typing.TYPE_CHECKING:
    import collections.abc
    from pathlib import Path

    import numpy.typing as npt


@dataclass(frozen=True)
class Regenie2LinearChunkAccumulator:
    """Accumulator for REGENIE step 2 linear chunk results (JAX arrays, device memory)."""

    metadata: models.VariantMetadata
    allele_one_frequency: jax.Array
    observation_count: jax.Array
    regenie2_linear_result: models.Regenie2LinearChunkResult


@dataclass(frozen=True)
class Regenie2LinearChunkPayload:
    """Host-side REGENIE step 2 linear association payload ready for persistence."""

    chunk_identifier: int
    variant_start_index: int
    variant_stop_index: int
    chromosome: npt.NDArray[np.str_]
    position: npt.NDArray[np.int64]
    variant_identifier: npt.NDArray[np.str_]
    allele_one: npt.NDArray[np.str_]
    allele_two: npt.NDArray[np.str_]
    allele_one_frequency: npt.NDArray[np.float32]
    observation_count: npt.NDArray[np.int32]
    beta: npt.NDArray[np.float32]
    standard_error: npt.NDArray[np.float32]
    chi_squared: npt.NDArray[np.float32]
    log10_p_value: npt.NDArray[np.float32]
    is_valid: npt.NDArray[np.bool_]


ChunkPayload = Regenie2LinearChunkPayload


open_genotype_reader = source.open_genotype_reader
load_aligned_sample_data_from_source = source.load_aligned_sample_data_from_source
iter_dosage_genotype_chunks_from_source = source.iter_dosage_genotype_chunks_from_source
load_prediction_source = regenie.load_prediction_source
prepare_regenie2_linear_state = regenie2_linear.prepare_regenie2_linear_state
compute_regenie2_linear_chunk = regenie2_linear.compute_regenie2_linear_chunk

DEFAULT_TSV_FRAME_BATCH_SIZE = 2


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


def build_regenie2_linear_output_frame(
    metadata: models.VariantMetadata,
    allele_one_frequency: jax.Array,
    observation_count: jax.Array,
    regenie2_linear_result: models.Regenie2LinearChunkResult,
) -> pl.DataFrame:
    """Build a tabular REGENIE step 2 linear result frame."""
    host_values = jax.device_get(
        {
            "allele_one_frequency": allele_one_frequency,
            "observation_count": observation_count,
            "beta": regenie2_linear_result.beta,
            "standard_error": regenie2_linear_result.standard_error,
            "chi_squared": regenie2_linear_result.chi_squared,
            "log10_p_value": regenie2_linear_result.log10_p_value,
            "valid_mask": regenie2_linear_result.valid_mask,
        }
    )
    return pl.DataFrame(
        {
            "chromosome": metadata.chromosome,
            "position": metadata.position,
            "variant_identifier": metadata.variant_identifiers,
            "allele_one": metadata.allele_one,
            "allele_two": metadata.allele_two,
            "allele_one_frequency": host_values["allele_one_frequency"],
            "observation_count": host_values["observation_count"],
            "beta": host_values["beta"],
            "standard_error": host_values["standard_error"],
            "chi_squared": host_values["chi_squared"],
            "log10_p_value": host_values["log10_p_value"],
            "is_valid": host_values["valid_mask"],
        }
    )


def build_regenie2_linear_chunk_payload(
    metadata: models.VariantMetadata,
    allele_one_frequency: jax.Array,
    observation_count: jax.Array,
    regenie2_linear_result: models.Regenie2LinearChunkResult,
) -> Regenie2LinearChunkPayload:
    """Build a host-side REGENIE step 2 payload for background persistence."""
    host_values = jax.device_get(
        {
            "allele_one_frequency": allele_one_frequency,
            "observation_count": observation_count,
            "beta": regenie2_linear_result.beta,
            "standard_error": regenie2_linear_result.standard_error,
            "chi_squared": regenie2_linear_result.chi_squared,
            "log10_p_value": regenie2_linear_result.log10_p_value,
            "valid_mask": regenie2_linear_result.valid_mask,
        }
    )
    return Regenie2LinearChunkPayload(
        chunk_identifier=metadata.variant_start_index,
        variant_start_index=metadata.variant_start_index,
        variant_stop_index=metadata.variant_stop_index,
        chromosome=metadata.chromosome,
        position=metadata.position,
        variant_identifier=metadata.variant_identifiers,
        allele_one=metadata.allele_one,
        allele_two=metadata.allele_two,
        allele_one_frequency=host_values["allele_one_frequency"],
        observation_count=host_values["observation_count"],
        beta=host_values["beta"],
        standard_error=host_values["standard_error"],
        chi_squared=host_values["chi_squared"],
        log10_p_value=host_values["log10_p_value"],
        is_valid=host_values["valid_mask"],
    )


def build_regenie2_linear_chunk_payload_batch(
    chunk_accumulators: collections.abc.Sequence[Regenie2LinearChunkAccumulator],
) -> list[Regenie2LinearChunkPayload]:
    """Build host-side REGENIE step 2 payloads with one device transfer for the batch."""
    if not chunk_accumulators:
        return []
    host_value_lists = jax.device_get(
        {
            "allele_one_frequency": [
                chunk_accumulator.allele_one_frequency for chunk_accumulator in chunk_accumulators
            ],
            "observation_count": [chunk_accumulator.observation_count for chunk_accumulator in chunk_accumulators],
            "beta": [chunk_accumulator.regenie2_linear_result.beta for chunk_accumulator in chunk_accumulators],
            "standard_error": [
                chunk_accumulator.regenie2_linear_result.standard_error for chunk_accumulator in chunk_accumulators
            ],
            "chi_squared": [
                chunk_accumulator.regenie2_linear_result.chi_squared for chunk_accumulator in chunk_accumulators
            ],
            "log10_p_value": [
                chunk_accumulator.regenie2_linear_result.log10_p_value for chunk_accumulator in chunk_accumulators
            ],
            "valid_mask": [
                chunk_accumulator.regenie2_linear_result.valid_mask for chunk_accumulator in chunk_accumulators
            ],
        }
    )
    host_values = {key: np.concatenate(value_list) for key, value_list in host_value_lists.items()}

    payloads: list[Regenie2LinearChunkPayload] = []
    variant_offset = 0
    for chunk_accumulator in chunk_accumulators:
        metadata = chunk_accumulator.metadata
        variant_count = len(metadata.position)
        next_variant_offset = variant_offset + variant_count
        payloads.append(
            Regenie2LinearChunkPayload(
                chunk_identifier=metadata.variant_start_index,
                variant_start_index=metadata.variant_start_index,
                variant_stop_index=metadata.variant_stop_index,
                chromosome=metadata.chromosome,
                position=metadata.position,
                variant_identifier=metadata.variant_identifiers,
                allele_one=metadata.allele_one,
                allele_two=metadata.allele_two,
                allele_one_frequency=np.ascontiguousarray(
                    host_values["allele_one_frequency"][variant_offset:next_variant_offset]
                ),
                observation_count=np.ascontiguousarray(
                    host_values["observation_count"][variant_offset:next_variant_offset]
                ),
                beta=np.ascontiguousarray(host_values["beta"][variant_offset:next_variant_offset]),
                standard_error=np.ascontiguousarray(host_values["standard_error"][variant_offset:next_variant_offset]),
                chi_squared=np.ascontiguousarray(host_values["chi_squared"][variant_offset:next_variant_offset]),
                log10_p_value=np.ascontiguousarray(host_values["log10_p_value"][variant_offset:next_variant_offset]),
                is_valid=np.ascontiguousarray(host_values["valid_mask"][variant_offset:next_variant_offset]),
            )
        )
        variant_offset = next_variant_offset
    return payloads


def build_chunk_payload(chunk_accumulator: Regenie2LinearChunkAccumulator) -> ChunkPayload:
    """Build a host-side chunk payload from a device-resident accumulator."""
    return build_regenie2_linear_chunk_payload(
        metadata=chunk_accumulator.metadata,
        allele_one_frequency=chunk_accumulator.allele_one_frequency,
        observation_count=chunk_accumulator.observation_count,
        regenie2_linear_result=chunk_accumulator.regenie2_linear_result,
    )


def build_chunk_payload_batch(
    chunk_accumulators: collections.abc.Sequence[Regenie2LinearChunkAccumulator],
) -> list[ChunkPayload]:
    """Build host-side payloads from a same-mode accumulator batch."""
    return build_regenie2_linear_chunk_payload_batch(chunk_accumulators)


def concatenate_regenie2_linear_results(
    accumulators: list[Regenie2LinearChunkAccumulator],
) -> pl.DataFrame:
    """Concatenate REGENIE step 2 linear chunk results into one DataFrame."""
    if not accumulators:
        return pl.DataFrame()

    all_chromosomes = np.concatenate([accumulator.metadata.chromosome for accumulator in accumulators])
    all_positions = np.concatenate([accumulator.metadata.position for accumulator in accumulators])
    all_variant_identifiers = np.concatenate([accumulator.metadata.variant_identifiers for accumulator in accumulators])
    all_allele_one = np.concatenate([accumulator.metadata.allele_one for accumulator in accumulators])
    all_allele_two = np.concatenate([accumulator.metadata.allele_two for accumulator in accumulators])

    host_value_lists = jax.device_get(
        {
            "allele_one_frequency": [accumulator.allele_one_frequency for accumulator in accumulators],
            "observation_count": [accumulator.observation_count for accumulator in accumulators],
            "beta": [accumulator.regenie2_linear_result.beta for accumulator in accumulators],
            "standard_error": [accumulator.regenie2_linear_result.standard_error for accumulator in accumulators],
            "chi_squared": [accumulator.regenie2_linear_result.chi_squared for accumulator in accumulators],
            "log10_p_value": [accumulator.regenie2_linear_result.log10_p_value for accumulator in accumulators],
            "valid_mask": [accumulator.regenie2_linear_result.valid_mask for accumulator in accumulators],
        }
    )
    host_values = {key: np.concatenate(value_list) for key, value_list in host_value_lists.items()}

    return pl.DataFrame(
        {
            "chromosome": all_chromosomes,
            "position": all_positions,
            "variant_identifier": all_variant_identifiers,
            "allele_one": all_allele_one,
            "allele_two": all_allele_two,
            "allele_one_frequency": host_values["allele_one_frequency"],
            "observation_count": host_values["observation_count"],
            "beta": host_values["beta"],
            "standard_error": host_values["standard_error"],
            "chi_squared": host_values["chi_squared"],
            "log10_p_value": host_values["log10_p_value"],
            "is_valid": host_values["valid_mask"],
        }
    )


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
) -> collections.abc.Iterator[Regenie2LinearChunkAccumulator]:
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
                    current_chromosome = chromosome

                assert current_loco_predictions is not None
                with jax.profiler.StepTraceAnnotation("regenie2_linear_chunk", step_num=chunk_number):
                    with jax.profiler.TraceAnnotation("regenie2_linear.compute"):
                        regenie2_linear_result = compute_regenie2_linear_chunk(
                            state=regenie2_linear_state,
                            genotype_matrix=current_chunk.genotypes,
                            loco_predictions=current_loco_predictions,
                        )
                    with jax.profiler.TraceAnnotation("regenie2_linear.accumulate"):
                        yield Regenie2LinearChunkAccumulator(
                            metadata=current_chunk.metadata,
                            allele_one_frequency=current_chunk.allele_one_frequency,
                            observation_count=current_chunk.observation_count,
                            regenie2_linear_result=regenie2_linear_result,
                        )
                chunk_number += 1


def write_frame_iterator_to_tsv(
    frame_iterator: collections.abc.Iterator[Regenie2LinearChunkAccumulator],
    output_path: Path,
    *,
    frame_batch_size: int = DEFAULT_TSV_FRAME_BATCH_SIZE,
) -> None:
    """Write accumulated REGENIE chunk results to a TSV file."""
    if frame_batch_size <= 0:
        message = "TSV frame batch size must be positive."
        raise ValueError(message)

    def flush_accumulator_batch(
        accumulator_batch: list[Regenie2LinearChunkAccumulator],
        output_file: typing.TextIO,
        *,
        include_header: bool,
    ) -> bool:
        if not accumulator_batch:
            return False
        output_frame = concatenate_regenie2_linear_results(accumulator_batch)
        output_frame.write_csv(
            output_file,
            separator="\t",
            include_header=include_header,
        )
        accumulator_batch.clear()
        return True

    first_chunk_written = False
    accumulator_batch: list[Regenie2LinearChunkAccumulator] = []
    with output_path.open("w", encoding="utf-8") as output_file:
        for accumulator in frame_iterator:
            accumulator_batch.append(accumulator)
            if len(accumulator_batch) >= frame_batch_size:
                batch_was_written = flush_accumulator_batch(
                    accumulator_batch,
                    output_file,
                    include_header=not first_chunk_written,
                )
                first_chunk_written = first_chunk_written or batch_was_written
        batch_was_written = flush_accumulator_batch(
            accumulator_batch,
            output_file,
            include_header=not first_chunk_written,
        )
        first_chunk_written = first_chunk_written or batch_was_written
    if not first_chunk_written:
        pl.DataFrame().write_csv(output_path, separator="\t")
