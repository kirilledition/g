"""Immutable execution plans for REGENIE-compatible runs."""

from __future__ import annotations

import re
import typing
from dataclasses import dataclass

from g import types
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_linear import config as regenie2_linear_config
from g.io import output, source

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g.interface import config


PHENOTYPE_DIRECTORY_SAFE_CHARACTER_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
PHENOTYPE_DIRECTORY_MAXIMUM_SLUG_LENGTH = 80


@dataclass(frozen=True)
class KernelConfig:
    """Engine kernel and batching settings.

    Attributes:
        chunk_size: Variant block size.
        device: JAX device requested for execution.
        staging_depth: Native callback staging depth.
        result_in_flight_limit: Optional cap for result chunks awaiting materialization.
        dosage_buffer_limit: Optional cap for reusable native dosage decode buffers.
        variant_limit: Optional debug cap on variants.
        thread_count: Requested CPU thread count.
        bgen_decode_tile_variant_count: Native BGEN decode tile variant count.
        gpu_genotype_format: Host-to-device genotype representation for GPU kernels.
        trusted_no_missing_diploid: Whether BGEN records can use the trusted diploid fast path.
        trusted_bgen_validation_mode: Validation policy for trusted BGEN decoding.
        alignment_config: Sample alignment settings consumed by the native dispatcher.
        multi_phenotype_sample_mode: Sample handling for multi-phenotype requests.
        binary_kernel_config: Static binary JAX kernel settings.
        linear_numerical_config: Static linear JAX numerical settings.

    """

    chunk_size: int
    device: types.Device
    staging_depth: int
    result_in_flight_limit: int | None
    dosage_buffer_limit: int | None
    variant_limit: int | None
    thread_count: int | None
    bgen_decode_tile_variant_count: int
    gpu_genotype_format: types.GpuGenotypeFormat
    trusted_no_missing_diploid: bool
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode
    alignment_config: config.GComputeConfig
    multi_phenotype_sample_mode: types.MultiPhenotypeSampleMode
    binary_kernel_config: regenie2_binary_config.BinaryKernelConfig | None = None
    linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None = None


@dataclass(frozen=True)
class OutputPlan:
    """Output materialization and writer settings.

    Attributes:
        output_prefix: User-facing output prefix.
        output_run_root: Root directory for per-phenotype chunked runs.
        output_format: Requested final output format.
        finalize_parquet: Whether the native writer should finalize Parquet.
        resume: Whether to resume a previous run.
        resume_mode: Resume validation mode.
        writer_threads: Number of output writer threads.
        writer_queue_depth: Output writer queue depth.
        chunks_per_arrow_file: Number of engine chunks grouped into one output file.
        arrow_compression: Arrow IPC compression codec.
        parquet_compression: Parquet dataset part compression codec.

    """

    output_prefix: Path
    output_run_root: Path
    output_format: types.OutputFormat
    finalize_parquet: bool
    resume: bool
    resume_mode: types.ResumeMode
    writer_threads: int
    writer_queue_depth: int
    chunks_per_arrow_file: int
    arrow_compression: types.ArrowCompression
    parquet_compression: types.ParquetCompression


@dataclass(frozen=True)
class PhenotypeRunPlan:
    """Prepared run state for one phenotype.

    Attributes:
        phenotype_name: Phenotype column name.
        output_run_paths: Chunked output paths for the phenotype.
        existing_manifest: Existing manifest loaded for resume, if present.
        effective_config_path: Path where the effective TOML config is written.

    """

    phenotype_name: str
    output_run_paths: output.OutputRunPaths
    existing_manifest: dict[str, typing.Any] | None
    effective_config_path: Path


@dataclass(frozen=True)
class PhenotypeComputeGroup:
    """Planned phenotype group that can share one compute delivery.

    Attributes:
        group_mode: Grouping mode selected by planning or alignment.
        phenotype_indices: Zero-based phenotype indices in run/output order.
        phenotype_names: Phenotype names in compute order.
        sample_mode: Sample handling semantics for this group.
        sample_set_fingerprint: Stable sample-set fingerprint when alignment is available.
        covariate_design_fingerprint: Stable covariate design fingerprint when alignment is available.
        prediction_alignment_fingerprint: Stable prediction alignment fingerprint when available.

    """

    group_mode: types.PhenotypeComputeGroupMode
    phenotype_indices: tuple[int, ...]
    phenotype_names: tuple[str, ...]
    sample_mode: types.MultiPhenotypeSampleMode
    sample_set_fingerprint: str | None = None
    covariate_design_fingerprint: str | None = None
    prediction_alignment_fingerprint: str | None = None


@dataclass(frozen=True)
class RegenieExecutionPlan:
    """Complete immutable execution plan for one REGENIE-compatible request.

    Attributes:
        association_mode: Statistical association engine to run.
        genotype_source_config: BGEN source configuration.
        phenotype_path: Phenotype table path.
        prediction_list_path: REGENIE step 1 prediction list.
        covariate_path: Optional covariate table path.
        covariate_names: Optional covariate column names.
        phenotype_run_plans: Per-phenotype output and manifest plans.
        phenotype_compute_groups: Planned phenotype compute groups.
        binary_correction_plan: Normalized binary fallback settings.
        kernel_config: Engine kernel and batching settings.
        output_plan: Output materialization settings.
        stage_timings_json: Optional stage timing diagnostics path.

    """

    association_mode: types.AssociationMode
    genotype_source_config: source.GenotypeSourceConfig
    phenotype_path: Path
    prediction_list_path: Path
    covariate_path: Path | None
    covariate_names: tuple[str, ...] | None
    phenotype_run_plans: tuple[PhenotypeRunPlan, ...]
    phenotype_compute_groups: tuple[PhenotypeComputeGroup, ...]
    binary_correction_plan: types.BinaryCorrectionPlan
    kernel_config: KernelConfig
    output_plan: OutputPlan
    stage_timings_json: Path | None


def normalize_binary_correction_config(binary_config: config.BinaryConfig) -> types.BinaryCorrectionPlan:
    """Normalize REGENIE-style binary correction flags into an internal plan."""
    if not (0.0 < binary_config.p_threshold < 1.0):
        message = "pThresh must be in (0, 1)."
        raise ValueError(message)
    if binary_config.spa:
        message = "SPA fallback is not implemented yet. Omit --spa for score-test-only output."
        raise NotImplementedError(message)
    if binary_config.approx and not binary_config.firth:
        message = "--approx requires --firth."
        raise ValueError(message)
    if binary_config.firth and binary_config.approx:
        return types.BinaryCorrectionPlan(
            method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
            p_threshold=binary_config.p_threshold,
            firth_se=binary_config.firth_se,
        )
    if binary_config.firth:
        message = "Exact REGENIE --firth without --approx is not implemented yet. Use --firth --approx."
        raise NotImplementedError(message)
    return types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod.SCORE_ONLY,
        p_threshold=binary_config.p_threshold,
        firth_se=False,
    )


def build_binary_kernel_config(compute_config: config.GComputeConfig) -> regenie2_binary_config.BinaryKernelConfig:
    """Build immutable binary JAX kernel settings from public compute config."""
    return regenie2_binary_config.BinaryKernelConfig(
        numerical=regenie2_binary_config.BinaryNumericalConfig(
            minimum_probability=compute_config.binary_minimum_probability,
            minimum_variance=compute_config.binary_minimum_variance,
            relative_variance_tolerance=compute_config.binary_relative_variance_tolerance,
        ),
        null_logistic=regenie2_binary_config.BinaryNullLogisticConfig(
            maximum_iterations=compute_config.binary_null_maximum_iterations,
            coefficient_tolerance=compute_config.binary_null_coefficient_tolerance,
        ),
        firth_candidate=regenie2_binary_config.FirthCandidateConfig(
            batch_size=compute_config.firth_batch_size,
            candidate_capacity=compute_config.firth_candidate_capacity,
        ),
        approximate_firth=regenie2_binary_config.ApproximateFirthConfig(
            maximum_iterations=compute_config.firth_maximum_iterations,
            gradient_tolerance=compute_config.firth_gradient_tolerance,
            coefficient_tolerance=compute_config.firth_coefficient_tolerance,
            likelihood_tolerance=compute_config.firth_likelihood_tolerance,
            maximum_step_size=compute_config.firth_maximum_step_size,
            pseudo_maximum_iterations=compute_config.firth_pseudo_maximum_iterations,
            pseudo_inner_maximum_iterations=compute_config.firth_pseudo_inner_maximum_iterations,
            newton_raphson_zero_start_iterations=compute_config.firth_newton_raphson_zero_start_iterations,
            line_search_maximum_attempts=compute_config.firth_line_search_maximum_attempts,
            step_halving_maximum_attempts=compute_config.firth_step_halving_maximum_attempts,
            initial_response_scale=compute_config.firth_initial_response_scale,
            sparse_carrier_dosage_threshold=compute_config.firth_sparse_carrier_dosage_threshold,
            step_halving_scale=compute_config.firth_step_halving_scale,
            use_block_math=compute_config.use_block_firth_math,
        ),
        null_firth=regenie2_binary_config.NullFirthConfig(
            maximum_iterations=compute_config.null_firth_maximum_iterations,
            gradient_tolerance=compute_config.null_firth_gradient_tolerance,
            maximum_step_size=compute_config.null_firth_maximum_step_size,
            fallback_iteration_multiplier=compute_config.null_firth_fallback_iteration_multiplier,
            fallback_step_divisor=compute_config.null_firth_fallback_step_divisor,
            line_search_maximum_attempts=compute_config.null_firth_line_search_maximum_attempts,
            step_halving_scale=compute_config.null_firth_step_halving_scale,
        ),
    )


def build_linear_numerical_config(
    compute_config: config.GComputeConfig,
) -> regenie2_linear_config.LinearNumericalConfig:
    """Build immutable linear JAX numerical settings from public compute config."""
    return regenie2_linear_config.LinearNumericalConfig(
        minimum_variance=compute_config.linear_minimum_variance,
        relative_variance_tolerance=compute_config.linear_relative_variance_tolerance,
    )


def build_regenie_execution_plan(regenie_config: config.RegenieConfig) -> RegenieExecutionPlan:
    """Build a complete execution plan from a validated public config."""
    output_prefix = typing.cast("Path", regenie_config.g_output.out)
    output_run_root = regenie_config.g_output.output_run_directory or output_prefix.with_name(f"{output_prefix.name}.g")
    association_mode = resolve_association_mode(regenie_config.trait.trait_type)
    output_plan = build_output_plan(regenie_config, output_prefix, output_run_root)
    kernel_config = build_kernel_config(regenie_config)
    phenotype_run_plans = tuple(
        build_phenotype_run_plan(
            phenotype_index=phenotype_index,
            phenotype_name=phenotype_name,
            association_mode=association_mode,
            output_plan=output_plan,
        )
        for phenotype_index, phenotype_name in enumerate(regenie_config.input.pheno_columns, start=1)
    )
    phenotype_compute_groups = build_phenotype_compute_groups(
        phenotype_names=tuple(phenotype_run_plan.phenotype_name for phenotype_run_plan in phenotype_run_plans),
        multi_phenotype_sample_mode=kernel_config.multi_phenotype_sample_mode,
    )
    return RegenieExecutionPlan(
        association_mode=association_mode,
        genotype_source_config=source.build_bgen_source_config(
            typing.cast("Path", regenie_config.input.bgen),
            regenie_config.input.sample,
        ),
        phenotype_path=typing.cast("Path", regenie_config.input.pheno_file),
        prediction_list_path=typing.cast("Path", regenie_config.input.pred),
        covariate_path=regenie_config.input.covar_file,
        covariate_names=regenie_config.input.covar_columns or None,
        phenotype_run_plans=phenotype_run_plans,
        phenotype_compute_groups=phenotype_compute_groups,
        binary_correction_plan=(
            normalize_binary_correction_config(regenie_config.binary)
            if regenie_config.trait.trait_type == types.RegenieTraitType.BINARY
            else types.BinaryCorrectionPlan()
        ),
        kernel_config=kernel_config,
        output_plan=output_plan,
        stage_timings_json=regenie_config.g_diagnostics.stage_timings_json,
    )


def build_phenotype_compute_groups(
    *,
    phenotype_names: tuple[str, ...],
    multi_phenotype_sample_mode: types.MultiPhenotypeSampleMode,
) -> tuple[PhenotypeComputeGroup, ...]:
    """Build config-time phenotype compute groups."""
    if not phenotype_names:
        message = "At least one phenotype is required for execution planning."
        raise ValueError(message)
    if len(phenotype_names) == 1:
        return (
            PhenotypeComputeGroup(
                group_mode=types.PhenotypeComputeGroupMode.SINGLE_PHENOTYPE,
                phenotype_indices=(0,),
                phenotype_names=phenotype_names,
                sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
            ),
        )
    phenotype_indices = tuple(range(len(phenotype_names)))
    if multi_phenotype_sample_mode == types.MultiPhenotypeSampleMode.COMPLETE_CASE:
        return (
            PhenotypeComputeGroup(
                group_mode=types.PhenotypeComputeGroupMode.COMPLETE_CASE,
                phenotype_indices=phenotype_indices,
                phenotype_names=phenotype_names,
                sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE,
            ),
        )
    return tuple(
        PhenotypeComputeGroup(
            group_mode=types.PhenotypeComputeGroupMode.PER_PHENOTYPE_COMPATIBLE,
            phenotype_indices=(phenotype_index,),
            phenotype_names=(phenotype_name,),
            sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        )
        for phenotype_index, phenotype_name in enumerate(phenotype_names)
    )


def resolve_association_mode(trait_type: types.RegenieTraitType) -> types.AssociationMode:
    """Resolve a trait family to the native association mode."""
    if trait_type == types.RegenieTraitType.BINARY:
        return types.AssociationMode.REGENIE2_BINARY
    return types.AssociationMode.REGENIE2_LINEAR


def build_output_plan(
    regenie_config: config.RegenieConfig,
    output_prefix: Path,
    output_run_root: Path,
) -> OutputPlan:
    """Build output settings from a public config."""
    return OutputPlan(
        output_prefix=output_prefix,
        output_run_root=output_run_root,
        output_format=regenie_config.g_output.format,
        finalize_parquet=regenie_config.g_output.finalize_parquet,
        resume=regenie_config.g_output.resume,
        resume_mode=regenie_config.g_output.resume_mode,
        writer_threads=regenie_config.g_output.writer_threads,
        writer_queue_depth=regenie_config.g_output.writer_queue_depth,
        chunks_per_arrow_file=regenie_config.g_output.chunks_per_arrow_file,
        arrow_compression=regenie_config.g_output.arrow_compression,
        parquet_compression=regenie_config.g_output.parquet_compression,
    )


def build_kernel_config(regenie_config: config.RegenieConfig) -> KernelConfig:
    """Build engine kernel settings from a public config."""
    return KernelConfig(
        chunk_size=regenie_config.trait.bsize,
        device=regenie_config.g_compute.device,
        staging_depth=regenie_config.g_compute.staging_depth,
        result_in_flight_limit=regenie_config.g_compute.result_in_flight_limit,
        dosage_buffer_limit=regenie_config.g_compute.dosage_buffer_limit,
        variant_limit=regenie_config.g_compute.variant_limit,
        thread_count=regenie_config.trait.threads,
        bgen_decode_tile_variant_count=regenie_config.g_compute.bgen_decode_tile_variant_count,
        gpu_genotype_format=regenie_config.g_compute.gpu_genotype_format,
        trusted_no_missing_diploid=regenie_config.g_compute.trusted_no_missing_diploid,
        trusted_bgen_validation_mode=regenie_config.g_compute.trusted_bgen_validation_mode,
        alignment_config=regenie_config.g_compute,
        multi_phenotype_sample_mode=regenie_config.g_compute.multi_phenotype_sample_mode,
        binary_kernel_config=(
            build_binary_kernel_config(regenie_config.g_compute)
            if regenie_config.trait.trait_type == types.RegenieTraitType.BINARY
            else None
        ),
        linear_numerical_config=(
            build_linear_numerical_config(regenie_config.g_compute)
            if regenie_config.trait.trait_type == types.RegenieTraitType.QUANTITATIVE
            else None
        ),
    )


def build_phenotype_run_plan(
    *,
    phenotype_index: int,
    phenotype_name: str,
    association_mode: types.AssociationMode,
    output_plan: OutputPlan,
) -> PhenotypeRunPlan:
    """Prepare output paths and resume manifest state for one phenotype."""
    output_directory_name = build_phenotype_output_directory_name(phenotype_index, phenotype_name)
    prepared_output_run = output.prepare_output_run(
        output_root=output_plan.output_run_root / output_directory_name,
        association_mode=association_mode,
        output_format=output_plan.output_format,
        resume=output_plan.resume,
        resume_mode=output_plan.resume_mode,
    )
    return PhenotypeRunPlan(
        phenotype_name=phenotype_name,
        output_run_paths=prepared_output_run.output_run_paths,
        existing_manifest=prepared_output_run.existing_manifest,
        effective_config_path=prepared_output_run.output_run_paths.run_directory / "effective_config.toml",
    )


def build_phenotype_output_directory_name(phenotype_index: int, phenotype_name: str) -> str:
    """Build a deterministic safe directory name for one phenotype output."""
    sanitized_slug = PHENOTYPE_DIRECTORY_SAFE_CHARACTER_PATTERN.sub("_", phenotype_name).strip("._-")
    if not sanitized_slug:
        sanitized_slug = "phenotype"
    return f"trait_{phenotype_index:04d}_{sanitized_slug[:PHENOTYPE_DIRECTORY_MAXIMUM_SLUG_LENGTH]}"
