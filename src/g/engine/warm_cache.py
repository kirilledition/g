"""JAX compilation-cache warming helpers for native REGENIE step 2 runs."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from g import _core, types
from g.compute.regenie2_binary import api as regenie2_binary
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_linear import api as regenie2_linear
from g.engine import callbacks, native_dispatch

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g.io import source


@dataclass(frozen=True)
class WarmCacheShape:
    """One genotype matrix shape warmed for the JAX compilation cache."""

    sample_count: int
    variant_count: int


@dataclass(frozen=True)
class WarmCacheReport:
    """Summary of warmed REGENIE step 2 JAX cache entries."""

    warmed_shapes: tuple[WarmCacheShape, ...]


def build_warm_cache_shapes(
    *,
    engine: _core.Regenie2RunEngine,
    chunk_size: int,
    variant_limit: int | None,
    sample_count: int,
) -> tuple[WarmCacheShape, ...]:
    """Build the full and tail chunk shapes that should be warmed."""
    chunk_specs = _core.plan_genotype_chunks(
        engine.variant_count,
        chunk_size,
        engine.chromosome_boundary_indices(),
        variant_limit=variant_limit,
        committed_chunk_identifiers=None,
    )
    variant_counts = []
    for chunk_spec in chunk_specs:
        variant_count = int(chunk_spec.variant_stop_index - chunk_spec.variant_start_index)
        if variant_count > 0 and variant_count not in variant_counts:
            variant_counts.append(variant_count)
    variant_counts.sort(reverse=True)
    return tuple(
        WarmCacheShape(sample_count=sample_count, variant_count=variant_count) for variant_count in variant_counts[:2]
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
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None = None,
) -> WarmCacheReport:
    """Warm full and tail JAX compilation-cache shapes for quantitative REGENIE step 2."""
    engine = native_dispatch.build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
    )
    run_input = native_dispatch.load_native_bgen_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=False,
        alignment_config=alignment_config,
    )
    prediction_source = native_dispatch.build_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
        alignment_config=alignment_config,
    )
    chromosome = first_engine_chromosome(engine)
    regenie_state = regenie2_linear.prepare_regenie2_linear_state(
        covariate_matrix=run_input.covariate_matrix,
        phenotype_vector=run_input.phenotype_vector,
    )
    chromosome_state = regenie2_linear.prepare_regenie2_linear_chromosome_state(
        regenie_state,
        jax.device_put(prediction_source.get_chromosome_predictions(chromosome)),
    )
    shapes = build_warm_cache_shapes(
        engine=engine,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        sample_count=int(run_input.sample_indices.shape[0]),
    )
    for shape in shapes:
        genotype_matrix = build_synthetic_genotype_matrix(
            phenotype_vector=run_input.phenotype_vector,
            variant_count=shape.variant_count,
            is_binary_trait=False,
        )
        result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state(
            chromosome_state=chromosome_state,
            genotype_matrix=genotype_matrix,
        )
        callbacks.block_until_ready(result.log10_p_value)
    return WarmCacheReport(warmed_shapes=shapes)


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
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None = None,
    kernel_config: regenie2_binary_config.BinaryKernelConfig | None = None,
) -> WarmCacheReport:
    """Warm full and tail JAX compilation-cache shapes for binary REGENIE step 2."""
    resolved_kernel_config = kernel_config or regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG
    engine = native_dispatch.build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
    )
    run_input = native_dispatch.load_native_bgen_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=True,
        alignment_config=alignment_config,
    )
    prediction_source = native_dispatch.build_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
        alignment_config=alignment_config,
    )
    chromosome = first_engine_chromosome(engine)
    regenie_state = regenie2_binary.prepare_regenie2_binary_state(
        covariate_matrix=run_input.covariate_matrix,
        phenotype_vector=run_input.phenotype_vector,
    )
    chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
        state=regenie_state,
        loco_offset=jax.device_put(prediction_source.get_chromosome_predictions(chromosome)),
        correction_plan=correction_plan,
        kernel_config=resolved_kernel_config,
    )
    shapes = build_warm_cache_shapes(
        engine=engine,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        sample_count=int(run_input.sample_indices.shape[0]),
    )
    for shape in shapes:
        genotype_matrix = build_synthetic_genotype_matrix(
            phenotype_vector=run_input.phenotype_vector,
            variant_count=shape.variant_count,
            is_binary_trait=True,
        )
        result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state(
            chromosome_state=chromosome_state,
            genotype_matrix=genotype_matrix,
            correction_plan=correction_plan,
            kernel_config=resolved_kernel_config,
        )
        callbacks.block_until_ready(result.log10_p_value)
    return WarmCacheReport(warmed_shapes=shapes)


def first_engine_chromosome(engine: _core.Regenie2RunEngine) -> str:
    """Return the first chromosome label from the native BGEN engine."""
    chromosome_values, _, _, _, _ = engine.variant_metadata_slice(0, 1)
    if not chromosome_values:
        message = "Cannot warm REGENIE step 2 cache for an empty BGEN dataset."
        raise ValueError(message)
    return chromosome_values[0]
