"""JAX compilation-cache warming helpers for native REGENIE step 2 runs."""

from __future__ import annotations

import enum
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

import g.engine.callbacks.diagnostics as callback_diagnostics
import g.engine.callbacks.transfers as callback_transfers
from g import _core, types
from g.compute.regenie2_binary import api as regenie2_binary
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_linear import api as regenie2_linear
from g.engine.native_dispatch import engine as native_dispatch_engine
from g.engine.native_dispatch import loaders as native_dispatch_loaders
from g.engine.native_dispatch import models as native_dispatch_models

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g.io import source


@dataclass(frozen=True)
class WarmCacheShape:
    """One genotype matrix shape warmed for the JAX compilation cache."""

    sample_count: int
    variant_count: int


class WarmCacheGenotypePath(enum.StrEnum):
    """JAX entrypoint family warmed for one cache signature."""

    LINEAR_DOSAGE = "linear_dosage"
    LINEAR_PACKED8 = "linear_packed8"
    BINARY_DOSAGE_SCORE = "binary_dosage_score"
    BINARY_DOSAGE_CORRECTION = "binary_dosage_correction"
    BINARY_PACKED8_SCORE = "binary_packed8_score"
    BINARY_PACKED8_CORRECTION = "binary_packed8_correction"


@dataclass(frozen=True)
class WarmCacheSignature:
    """One exact production JAX signature warmed for the compilation cache.

    Attributes:
        shape: Genotype chunk sample and variant counts.
        association_mode: Association mode warmed by the signature.
        genotype_format: Host-to-device genotype representation.
        genotype_path: Concrete JAX entrypoint family.
        trait_count: Number of traits represented by the warmed state.
        score_dtype: Score-kernel floating point dtype.
        correction_method: Binary correction method for binary signatures.
        correction_p_threshold: Binary correction p-value threshold.
        correction_firth_se: Whether Firth rows use LRT-derived standard errors.
        firth_candidate_batch_size: Fixed Firth batch size for binary correction.
        firth_candidate_capacity: Preferred Firth candidate capacity for binary correction.

    """

    shape: WarmCacheShape
    association_mode: types.AssociationMode
    genotype_format: types.GpuGenotypeFormat
    genotype_path: WarmCacheGenotypePath
    trait_count: int
    score_dtype: types.FloatingPointDtype
    correction_method: types.BinaryFallbackMethod | None
    correction_p_threshold: float | None
    correction_firth_se: bool | None
    firth_candidate_batch_size: int | None
    firth_candidate_capacity: int | None


@dataclass(frozen=True)
class WarmCacheReport:
    """Summary of warmed REGENIE step 2 JAX cache entries."""

    warmed_shapes: tuple[WarmCacheShape, ...]
    warmed_signatures: tuple[WarmCacheSignature, ...]


@dataclass(frozen=True)
class WarmCacheNativeStats:
    """Synthetic native statistic arrays for production cache warming.

    Attributes:
        dosage_sum: Per-variant dosage sums.
        observation_count: Per-variant observation counts.
        imputed_dosage_square_sum: Per-variant imputed dosage square sums.

    """

    dosage_sum: jax.Array
    observation_count: jax.Array
    imputed_dosage_square_sum: jax.Array


def build_warm_cache_shapes(
    *,
    engine: _core.Regenie2RunEngine,
    chunk_size: int,
    variant_limit: int | None,
    sample_count: int,
) -> tuple[WarmCacheShape, ...]:
    """Build unique production chunk shapes that should be warmed."""
    chunk_specs = _core.plan_genotype_chunks(
        engine.variant_count,
        chunk_size,
        engine.chromosome_boundary_indices(),
        variant_limit=variant_limit,
        committed_chunk_identifiers=None,
    )
    shapes: list[WarmCacheShape] = []
    seen_shapes: set[WarmCacheShape] = set()
    for chunk_spec in chunk_specs:
        variant_count = int(chunk_spec.variant_stop_index - chunk_spec.variant_start_index)
        shape = WarmCacheShape(sample_count=sample_count, variant_count=variant_count)
        if variant_count > 0 and shape not in seen_shapes:
            shapes.append(shape)
            seen_shapes.add(shape)
    return tuple(shapes)


def build_linear_warm_cache_signature(
    *,
    shape: WarmCacheShape,
    gpu_genotype_format: types.GpuGenotypeFormat,
    score_dtype: types.FloatingPointDtype,
) -> WarmCacheSignature:
    """Build the warmed signature report entry for a linear chunk."""
    genotype_path = (
        WarmCacheGenotypePath.LINEAR_PACKED8
        if gpu_genotype_format == types.GpuGenotypeFormat.PACKED8
        else WarmCacheGenotypePath.LINEAR_DOSAGE
    )
    return WarmCacheSignature(
        shape=shape,
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        genotype_format=gpu_genotype_format,
        genotype_path=genotype_path,
        trait_count=1,
        score_dtype=score_dtype,
        correction_method=None,
        correction_p_threshold=None,
        correction_firth_se=None,
        firth_candidate_batch_size=None,
        firth_candidate_capacity=None,
    )


def build_binary_warm_cache_signature(
    *,
    shape: WarmCacheShape,
    gpu_genotype_format: types.GpuGenotypeFormat,
    score_dtype: types.FloatingPointDtype,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
) -> WarmCacheSignature:
    """Build the warmed signature report entry for a binary chunk."""
    score_only = correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY
    if gpu_genotype_format == types.GpuGenotypeFormat.PACKED8:
        genotype_path = (
            WarmCacheGenotypePath.BINARY_PACKED8_SCORE
            if score_only
            else WarmCacheGenotypePath.BINARY_PACKED8_CORRECTION
        )
    else:
        genotype_path = (
            WarmCacheGenotypePath.BINARY_DOSAGE_SCORE if score_only else WarmCacheGenotypePath.BINARY_DOSAGE_CORRECTION
        )
    return WarmCacheSignature(
        shape=shape,
        association_mode=types.AssociationMode.REGENIE2_BINARY,
        genotype_format=gpu_genotype_format,
        genotype_path=genotype_path,
        trait_count=1,
        score_dtype=score_dtype,
        correction_method=correction_plan.method,
        correction_p_threshold=correction_plan.p_threshold,
        correction_firth_se=correction_plan.firth_se,
        firth_candidate_batch_size=None if score_only else kernel_config.firth_candidate.batch_size,
        firth_candidate_capacity=None if score_only else kernel_config.firth_candidate.candidate_capacity,
    )


def build_synthetic_genotype_matrix(
    *,
    phenotype_vector: jax.Array,
    variant_count: int,
    is_binary_trait: bool,
) -> jax.Array:
    """Build deterministic genotype inputs for cache warming."""
    if is_binary_trait:
        genotype_vector = jnp.asarray(phenotype_vector, dtype=jnp.float32) * 2.0
    else:
        sample_index = jnp.arange(phenotype_vector.shape[0], dtype=jnp.float32)
        genotype_vector = jnp.mod(sample_index, 3.0)
        genotype_vector = genotype_vector - jnp.mean(genotype_vector)
    return jnp.tile(genotype_vector[:, None], (1, variant_count))


def build_synthetic_integer_genotype_matrix(
    *,
    phenotype_vector: jax.Array,
    variant_count: int,
) -> jax.Array:
    """Build exact 0/1/2 dosage inputs for packed8 cache warming."""
    sample_index = jnp.arange(phenotype_vector.shape[0], dtype=jnp.float32)
    genotype_vector = jnp.mod(sample_index, 3.0)
    return jnp.tile(genotype_vector[:, None], (1, variant_count))


def build_synthetic_variant_major_genotype_matrix(
    *,
    phenotype_vector: jax.Array,
    variant_count: int,
    is_binary_trait: bool,
    exact_integer_dosage: bool,
) -> jax.Array:
    """Build deterministic variant-major genotype inputs for cache warming."""
    if exact_integer_dosage:
        genotype_matrix = build_synthetic_integer_genotype_matrix(
            phenotype_vector=phenotype_vector,
            variant_count=variant_count,
        )
    else:
        genotype_matrix = build_synthetic_genotype_matrix(
            phenotype_vector=phenotype_vector,
            variant_count=variant_count,
            is_binary_trait=is_binary_trait,
        )
    return genotype_matrix.T


def encode_variant_major_dosage_to_packed8_probability_pairs(genotype_matrix_by_variant: jax.Array) -> jax.Array:
    """Encode exact 0/1/2 variant-major dosage as trusted packed8 probability pairs."""
    homozygous_reference_probability = jnp.where(
        genotype_matrix_by_variant == jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(255, dtype=jnp.uint8),
        jnp.asarray(0, dtype=jnp.uint8),
    )
    heterozygous_probability = jnp.where(
        genotype_matrix_by_variant == jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(255, dtype=jnp.uint8),
        jnp.asarray(0, dtype=jnp.uint8),
    )
    return jnp.stack((homozygous_reference_probability, heterozygous_probability), axis=2)


def build_synthetic_native_stats(genotype_matrix_by_variant: jax.Array) -> WarmCacheNativeStats:
    """Build synthetic native stats matching a variant-major dosage matrix."""
    return WarmCacheNativeStats(
        dosage_sum=jnp.sum(genotype_matrix_by_variant, axis=1, dtype=jnp.float32),
        observation_count=jnp.full(
            (genotype_matrix_by_variant.shape[0],),
            genotype_matrix_by_variant.shape[1],
            dtype=jnp.int32,
        ),
        imputed_dosage_square_sum=jnp.sum(genotype_matrix_by_variant * genotype_matrix_by_variant, axis=1),
    )


def warm_regenie2_linear_bgen_cache(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    alignment_config: native_dispatch_models.SampleAlignmentConfigProtocol | None,
    gpu_genotype_format: types.GpuGenotypeFormat,
    score_dtype: types.FloatingPointDtype,
) -> WarmCacheReport:
    """Warm full and tail JAX compilation-cache shapes for quantitative REGENIE step 2."""
    engine = native_dispatch_engine.build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        trusted_bgen_validator=None,
    )
    run_input = native_dispatch_loaders.load_native_bgen_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=False,
        alignment_config=alignment_config,
        build_native_bgen_run_input_callable=None,
        load_aligned_sample_data_callable=None,
    )
    prediction_source = native_dispatch_loaders.build_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
        alignment_config=alignment_config,
    )
    chromosome = first_engine_chromosome(engine)
    covariate_matrix = callback_transfers.put_compute_array_on_device(run_input.covariate_matrix)
    phenotype_vector = callback_transfers.put_compute_array_on_device(run_input.phenotype_vector)
    regenie_state = regenie2_linear.prepare_regenie2_linear_state(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        score_dtype=score_dtype,
    )
    chromosome_state = regenie2_linear.prepare_regenie2_linear_chromosome_state(
        state=regenie_state,
        loco_predictions=jax.device_put(prediction_source.get_chromosome_predictions(chromosome)),
        score_dtype=score_dtype,
    )
    shapes = build_warm_cache_shapes(
        engine=engine,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        sample_count=int(run_input.sample_indices.shape[0]),
    )
    for shape in shapes:
        genotype_matrix_by_variant = build_synthetic_variant_major_genotype_matrix(
            phenotype_vector=phenotype_vector,
            variant_count=shape.variant_count,
            is_binary_trait=False,
            exact_integer_dosage=gpu_genotype_format == types.GpuGenotypeFormat.PACKED8,
        )
        native_stats = build_synthetic_native_stats(genotype_matrix_by_variant)
        if gpu_genotype_format == types.GpuGenotypeFormat.PACKED8:
            result = regenie2_linear.compute_linear_chunk_packed8_donating_inputs(
                chromosome_state=chromosome_state,
                packed_probability_pairs_by_variant=encode_variant_major_dosage_to_packed8_probability_pairs(
                    genotype_matrix_by_variant
                ),
                genotype_dosage_sum=native_stats.dosage_sum,
                genotype_observation_count=native_stats.observation_count,
                genotype_imputed_dosage_square_sum=native_stats.imputed_dosage_square_sum,
                score_dtype=score_dtype,
            )
        else:
            result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state_variant_major(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                genotype_dosage_sum=native_stats.dosage_sum,
                genotype_observation_count=native_stats.observation_count,
                genotype_imputed_dosage_square_sum=native_stats.imputed_dosage_square_sum,
                score_dtype=score_dtype,
            )
        callback_diagnostics.block_until_ready(result.log10_p_value)
    signatures = tuple(
        build_linear_warm_cache_signature(
            shape=shape,
            gpu_genotype_format=gpu_genotype_format,
            score_dtype=score_dtype,
        )
        for shape in shapes
    )
    return WarmCacheReport(warmed_shapes=shapes, warmed_signatures=signatures)


def warm_regenie2_binary_bgen_cache(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    correction_plan: types.BinaryCorrectionPlan,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    alignment_config: native_dispatch_models.SampleAlignmentConfigProtocol | None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    gpu_genotype_format: types.GpuGenotypeFormat,
    score_dtype: types.FloatingPointDtype,
) -> WarmCacheReport:
    """Warm full and tail JAX compilation-cache shapes for binary REGENIE step 2."""
    engine = native_dispatch_engine.build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        trusted_bgen_validator=None,
    )
    run_input = native_dispatch_loaders.load_native_bgen_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=True,
        alignment_config=alignment_config,
        build_native_bgen_run_input_callable=None,
        load_aligned_sample_data_callable=None,
    )
    prediction_source = native_dispatch_loaders.build_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
        alignment_config=alignment_config,
    )
    chromosome = first_engine_chromosome(engine)
    covariate_matrix = callback_transfers.put_compute_array_on_device(run_input.covariate_matrix)
    phenotype_vector = callback_transfers.put_compute_array_on_device(run_input.phenotype_vector)
    regenie_state = regenie2_binary.prepare_regenie2_binary_state(
        covariate_matrix=covariate_matrix,
        phenotype_vector=phenotype_vector,
        score_dtype=score_dtype,
    )
    chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
        state=regenie_state,
        loco_offset=jax.device_put(prediction_source.get_chromosome_predictions(chromosome)),
        correction_plan=correction_plan,
        kernel_config=kernel_config,
        score_dtype=score_dtype,
    )
    shapes = build_warm_cache_shapes(
        engine=engine,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        sample_count=int(run_input.sample_indices.shape[0]),
    )
    for shape in shapes:
        genotype_matrix_by_variant = build_synthetic_variant_major_genotype_matrix(
            phenotype_vector=phenotype_vector,
            variant_count=shape.variant_count,
            is_binary_trait=True,
            exact_integer_dosage=gpu_genotype_format == types.GpuGenotypeFormat.PACKED8,
        )
        native_stats = build_synthetic_native_stats(genotype_matrix_by_variant)
        if gpu_genotype_format == types.GpuGenotypeFormat.PACKED8:
            packed_probability_pairs_by_variant = encode_variant_major_dosage_to_packed8_probability_pairs(
                genotype_matrix_by_variant
            )
            if correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
                result = regenie2_binary.compute_binary_score_test_packed8_donating_inputs(
                    chromosome_state=chromosome_state,
                    packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
                    correction_plan=correction_plan,
                    kernel_config=kernel_config,
                    dosage_sum=native_stats.dosage_sum,
                    observation_count=native_stats.observation_count,
                    score_dtype=score_dtype,
                )
            else:
                result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state_packed8(
                    chromosome_state=chromosome_state,
                    packed_probability_pairs_by_variant=packed_probability_pairs_by_variant,
                    correction_plan=correction_plan,
                    kernel_config=kernel_config,
                    sparse_candidate_mask=None,
                    stage_duration_recorder=None,
                    dosage_sum=native_stats.dosage_sum,
                    observation_count=native_stats.observation_count,
                    score_dtype=score_dtype,
                )
        elif correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
            result = regenie2_binary.compute_binary_score_test_variant_major_donating_inputs(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                correction_plan=correction_plan,
                kernel_config=kernel_config,
                dosage_sum=native_stats.dosage_sum,
                observation_count=native_stats.observation_count,
                score_dtype=score_dtype,
            )
        else:
            result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state_variant_major(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_matrix_by_variant,
                correction_plan=correction_plan,
                kernel_config=kernel_config,
                sparse_candidate_mask=None,
                stage_duration_recorder=None,
                dosage_sum=native_stats.dosage_sum,
                observation_count=native_stats.observation_count,
                score_dtype=score_dtype,
            )
        callback_diagnostics.block_until_ready(result.log10_p_value)
    signatures = tuple(
        build_binary_warm_cache_signature(
            shape=shape,
            gpu_genotype_format=gpu_genotype_format,
            score_dtype=score_dtype,
            correction_plan=correction_plan,
            kernel_config=kernel_config,
        )
        for shape in shapes
    )
    return WarmCacheReport(warmed_shapes=shapes, warmed_signatures=signatures)


def first_engine_chromosome(engine: _core.Regenie2RunEngine) -> str:
    """Return the first chromosome label from the native BGEN engine."""
    chromosome_values, _, _, _, _ = engine.variant_metadata_slice(0, 1)
    if not chromosome_values:
        message = "Cannot warm REGENIE step 2 cache for an empty BGEN dataset."
        raise ValueError(message)
    return chromosome_values[0]
