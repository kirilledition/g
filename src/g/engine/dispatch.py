
from __future__ import annotations
import typing
import jax
import jax.profiler
from g import models, types
from g.engine import types as engine_types
from g.compute import regenie2_binary, regenie2_linear
from g.engine import chromosome_chunks, profiling
from g.io import regenie, source
if typing.TYPE_CHECKING:
    import collections.abc
    from pathlib import Path

BinaryChunkComputeFunction = typing.Callable[[models.Regenie2BinaryChromosomeState, jax.Array, types.RegenieBinaryCorrection], models.Regenie2BinaryChunkResult]
compute_regenie2_binary_chunk = typing.cast("BinaryChunkComputeFunction", regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state)

def iter_regenie2_linear_output_frames(*, genotype_source_config: source.GenotypeSourceConfig, phenotype_path: Path, phenotype_name: str, prediction_list_path: Path, covariate_path: Path | None, covariate_names: tuple[str, ...] | None, chunk_size: int, variant_limit: int | None, prefetch_chunks: int = 0, committed_chunk_identifiers: set[int] | None = None) -> collections.abc.Iterator[engine_types.Regenie2ChunkAccumulator]:
    if genotype_source_config.source_format != types.GenotypeSourceFormat.BGEN:
        raise ValueError("REGENIE step 2 linear association requires a BGEN genotype source.")
    genotype_reader = source.open_genotype_reader(genotype_source_config)
    committed_identifier_set = committed_chunk_identifiers or set()
    with genotype_reader:
        with jax.profiler.TraceAnnotation("regenie2_linear.load_aligned_sample_data"):
            aligned_sample_data = source.load_aligned_sample_data_from_source(genotype_source_config=genotype_source_config, phenotype_path=phenotype_path, phenotype_name=phenotype_name, covariate_path=covariate_path, covariate_names=covariate_names, is_binary_trait=False, genotype_reader=genotype_reader)
        regenie2_linear_state = regenie2_linear.prepare_regenie2_linear_state(covariate_matrix=aligned_sample_data.covariate_matrix, phenotype_vector=aligned_sample_data.phenotype_vector)
        prediction_source = regenie.load_prediction_source(prediction_list_path, phenotype_name)
        chunk_iterator = source.iter_dosage_genotype_chunks_from_source(genotype_source_config=genotype_source_config, sample_indices=aligned_sample_data.sample_indices, expected_individual_identifiers=aligned_sample_data.individual_identifiers, chunk_size=chunk_size, variant_limit=variant_limit, prefetch_chunks=prefetch_chunks, genotype_reader=genotype_reader)
        current_chromosome: str | None = None
        current_chromosome_state: models.Regenie2LinearChromosomeState | None = None
        chunk_number = 0
        for source_chunk in chunk_iterator:
            for current_chunk in chromosome_chunks.split_dosage_genotype_chunk_with_reader_metadata(source_chunk, genotype_reader):
                if current_chunk.metadata.variant_start_index in committed_identifier_set:
                    continue
                chromosome = str(current_chunk.metadata.chromosome[0])
                if chromosome != current_chromosome:
                    current_loco_predictions = prediction_source.get_chromosome_predictions(chromosome=chromosome, sample_family_identifiers=aligned_sample_data.family_identifiers, sample_individual_identifiers=aligned_sample_data.individual_identifiers)
                    current_chromosome_state = regenie2_linear.prepare_regenie2_linear_chromosome_state(regenie2_linear_state, current_loco_predictions)
                    current_chromosome = chromosome
                assert current_chromosome_state is not None
                with profiling.profiled_regenie2_linear_chunk_step(chunk_number):
                    regenie2_linear_result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state(chromosome_state=current_chromosome_state, genotype_matrix=current_chunk.genotypes)
                    yield engine_types.Regenie2ChunkAccumulator(metadata=current_chunk.metadata, allele_one_frequency=current_chunk.allele_one_frequency, observation_count=current_chunk.observation_count, beta=regenie2_linear_result.beta, standard_error=regenie2_linear_result.standard_error, chi_squared=regenie2_linear_result.chi_squared, log10_p_value=regenie2_linear_result.log10_p_value, extra_code=None)
                chunk_number += 1

def iter_regenie2_binary_output_frames(*, genotype_source_config: source.GenotypeSourceConfig, phenotype_path: Path, phenotype_name: str, prediction_list_path: Path, covariate_path: Path | None, covariate_names: tuple[str, ...] | None, chunk_size: int, variant_limit: int | None, prefetch_chunks: int = 0, committed_chunk_identifiers: set[int] | None = None, correction: types.RegenieBinaryCorrection = types.RegenieBinaryCorrection.FIRTH_APPROXIMATE) -> collections.abc.Iterator[engine_types.Regenie2ChunkAccumulator]:
    if genotype_source_config.source_format != types.GenotypeSourceFormat.BGEN:
        raise ValueError("REGENIE step 2 binary association requires a BGEN genotype source.")
    genotype_reader = source.open_genotype_reader(genotype_source_config)
    committed_identifier_set = committed_chunk_identifiers or set()
    with genotype_reader:
        aligned_sample_data = source.load_aligned_sample_data_from_source(genotype_source_config=genotype_source_config, phenotype_path=phenotype_path, phenotype_name=phenotype_name, covariate_path=covariate_path, covariate_names=covariate_names, is_binary_trait=True, genotype_reader=genotype_reader)
        regenie2_binary_state = regenie2_binary.prepare_regenie2_binary_state(covariate_matrix=aligned_sample_data.covariate_matrix, phenotype_vector=aligned_sample_data.phenotype_vector)
        prediction_source = regenie.load_prediction_source(prediction_list_path, phenotype_name)
        chunk_iterator = source.iter_dosage_genotype_chunks_from_source(genotype_source_config=genotype_source_config, sample_indices=aligned_sample_data.sample_indices, expected_individual_identifiers=aligned_sample_data.individual_identifiers, chunk_size=chunk_size, variant_limit=variant_limit, prefetch_chunks=prefetch_chunks, genotype_reader=genotype_reader)
        current_chromosome: str | None = None
        current_chromosome_state: models.Regenie2BinaryChromosomeState | None = None
        chunk_number = 0
        for source_chunk in chunk_iterator:
            for current_chunk in chromosome_chunks.split_dosage_genotype_chunk_with_reader_metadata(source_chunk, genotype_reader):
                if current_chunk.metadata.variant_start_index in committed_identifier_set:
                    continue
                chromosome = str(current_chunk.metadata.chromosome[0])
                if chromosome != current_chromosome:
                    loco_offset = prediction_source.get_chromosome_predictions(chromosome=chromosome, sample_family_identifiers=aligned_sample_data.family_identifiers, sample_individual_identifiers=aligned_sample_data.individual_identifiers)
                    current_chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(regenie2_binary_state, loco_offset)
                    current_chromosome = chromosome
                assert current_chromosome_state is not None
                with profiling.profiled_regenie2_binary_chunk_step(chunk_number):
                    result = compute_regenie2_binary_chunk(current_chromosome_state, current_chunk.genotypes, correction)
                    yield engine_types.Regenie2ChunkAccumulator(metadata=current_chunk.metadata, allele_one_frequency=current_chunk.allele_one_frequency, observation_count=current_chunk.observation_count, beta=result.beta, standard_error=result.standard_error, chi_squared=result.chi_squared, log10_p_value=result.log10_p_value, extra_code=result.extra_code)
                chunk_number += 1
