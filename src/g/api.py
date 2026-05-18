"""Public Python API for GWAS execution."""

from __future__ import annotations

import dataclasses
import time
import warnings
from pathlib import Path

from g import engine, jax_setup, types
from g.io import output, source

configure_jax_device = jax_setup.configure_jax_device
run_regenie2_linear_bgen_pipeline = engine.run_regenie2_linear_bgen_pipeline
run_regenie2_binary_bgen_pipeline = engine.run_regenie2_binary_bgen_pipeline
warm_regenie2_linear_bgen_cache = engine.warm_regenie2_linear_bgen_cache
warm_regenie2_binary_bgen_cache = engine.warm_regenie2_binary_bgen_cache
prepare_output_run = output.prepare_output_run
finalize_chunks_to_parquet = output.finalize_chunks_to_parquet
WarmCacheReport = engine.WarmCacheReport

DEFAULT_REGENIE2_LINEAR_CHUNK_SIZE = 8192
DEFAULT_OUTPUT_WRITER_QUEUE_DEPTH = output.DEFAULT_WRITER_QUEUE_DEPTH


@dataclasses.dataclass(frozen=True)
class ComputeConfig:
    """Hardware and batching settings for REGENIE step 2 execution."""

    chunk_size: int = DEFAULT_REGENIE2_LINEAR_CHUNK_SIZE
    device: types.Device = types.Device.CPU
    variant_limit: int | None = None
    prefetch_chunks: int = 1
    output_run_directory: Path | None = None
    resume: bool = False
    finalize_parquet: bool = True
    output_writer_thread_count: int = output.DEFAULT_WRITER_THREAD_COUNT
    output_writer_queue_depth: int = DEFAULT_OUTPUT_WRITER_QUEUE_DEPTH
    trusted_no_missing_diploid: bool = False
    warm_cache_first: bool = False


@dataclasses.dataclass(frozen=True)
class Regenie2LinearConfig:
    """Configuration for REGENIE step 2 linear association."""


@dataclasses.dataclass(frozen=True)
class Regenie2BinaryConfig:
    """Configuration for REGENIE step 2 binary association."""

    firth: bool = False
    approx: bool = False
    spa: bool = False
    p_threshold: float = 0.05
    firth_se: bool = False


@dataclasses.dataclass(frozen=True)
class RunArtifacts:
    """Immutable pointers to generated output files."""

    output_run_directory: Path | None = None
    final_parquet: Path | None = None


def parse_covariate_name_list(raw_covariate_names: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Normalize covariate names into a tuple."""
    if raw_covariate_names is None:
        return None
    if isinstance(raw_covariate_names, str):
        covariate_names = tuple(
            stripped_name for name in raw_covariate_names.split(",") if (stripped_name := name.strip())
        )
        return covariate_names or None
    covariate_names = tuple(name.strip() for name in raw_covariate_names if name.strip())
    return covariate_names or None


def validate_compute_config(compute_config: ComputeConfig) -> None:
    """Validate a compute configuration."""
    if compute_config.chunk_size <= 0:
        message = "Chunk size must be positive."
        raise ValueError(message)
    if compute_config.variant_limit is not None and compute_config.variant_limit <= 0:
        message = "Variant limit must be positive when provided."
        raise ValueError(message)
    if compute_config.prefetch_chunks < 0:
        message = "Prefetch chunk count must be zero or positive."
        raise ValueError(message)
    if compute_config.output_writer_thread_count <= 0:
        message = "Output writer thread count must be positive."
        raise ValueError(message)
    if compute_config.output_writer_queue_depth <= 0:
        message = "Output writer queue depth must be positive."
        raise ValueError(message)


def normalize_binary_correction_config(binary_config: Regenie2BinaryConfig) -> types.BinaryCorrectionPlan:
    """Normalize REGENIE-style binary correction flags into an internal plan."""
    if not (0.0 < binary_config.p_threshold < 1.0):
        message = "pThresh must be in (0, 1)."
        raise ValueError(message)

    firth = binary_config.firth
    approx = binary_config.approx
    spa = binary_config.spa
    if firth and spa:
        warnings.warn(
            "Only one of --firth/--spa can be used. Mirroring REGENIE, Firth will be used.",
            stacklevel=2,
        )
        spa = False
    if approx and not firth:
        warnings.warn(
            "--approx only works with --firth. Mirroring REGENIE, --approx is ignored.",
            stacklevel=2,
        )
        approx = False

    if spa:
        message = "SPA fallback is not implemented yet. Omit --spa for score-test-only output."
        raise NotImplementedError(message)
    if firth and approx:
        return types.BinaryCorrectionPlan(
            method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
            p_threshold=binary_config.p_threshold,
            firth_se=binary_config.firth_se,
        )
    if firth:
        message = "Exact REGENIE --firth without --approx is not implemented yet. Use --firth --approx."
        raise NotImplementedError(message)
    return types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod.SCORE_ONLY,
        p_threshold=binary_config.p_threshold,
        firth_se=False,
    )


def regenie2_linear(
    *,
    bgen: Path | str,
    sample: Path | str | None = None,
    pheno: Path | str,
    pheno_name: str,
    out: Path | str,
    covar: Path | str | None = None,
    covar_names: str | list[str] | tuple[str, ...] | None = None,
    pred: Path | str,
    compute: ComputeConfig | None = None,
    solver: Regenie2LinearConfig | None = None,
) -> RunArtifacts:
    """Run a REGENIE step 2 linear association scan and write results to disk."""
    del solver
    return regenie2(
        bgen=bgen,
        sample=sample,
        pheno=pheno,
        pheno_name=pheno_name,
        out=out,
        covar=covar,
        covar_names=covar_names,
        pred=pred,
        trait_type=types.RegenieTraitType.QUANTITATIVE,
        compute=compute,
    )


def regenie2(
    *,
    bgen: Path | str,
    sample: Path | str | None = None,
    pheno: Path | str,
    pheno_name: str,
    out: Path | str,
    covar: Path | str | None = None,
    covar_names: str | list[str] | tuple[str, ...] | None = None,
    pred: Path | str,
    trait_type: types.RegenieTraitType = types.RegenieTraitType.QUANTITATIVE,
    compute: ComputeConfig | None = None,
    binary: Regenie2BinaryConfig | None = None,
) -> RunArtifacts:
    """Run a REGENIE step 2 association scan and write results to disk."""
    api_entry_start_time = time.perf_counter()
    stage_timing_recorder = engine.build_stage_timing_recorder_from_environment()
    compute_config = compute or ComputeConfig()
    try:
        validate_compute_config(compute_config)
        device_start_time = time.perf_counter()
        configure_jax_device(compute_config.device)
        engine.record_stage_duration(stage_timing_recorder, "jax_device_configuration_backend_init", device_start_time)
        covariate_name_list = parse_covariate_name_list(covar_names)
        genotype_source_config = source.build_bgen_source_config(bgen, sample)
        binary_config = binary or Regenie2BinaryConfig()
        binary_correction_plan = (
            normalize_binary_correction_config(binary_config)
            if trait_type == types.RegenieTraitType.BINARY
            else types.BinaryCorrectionPlan()
        )
        if compute_config.warm_cache_first:
            warm_cache_start_time = time.perf_counter()
            if trait_type == types.RegenieTraitType.BINARY:
                warm_regenie2_binary_bgen_cache(
                    genotype_source_config=genotype_source_config,
                    phenotype_path=Path(pheno),
                    phenotype_name=pheno_name,
                    prediction_list_path=Path(pred),
                    covariate_path=Path(covar) if covar is not None else None,
                    covariate_names=covariate_name_list,
                    chunk_size=compute_config.chunk_size,
                    variant_limit=compute_config.variant_limit,
                    correction_plan=binary_correction_plan,
                    trusted_no_missing_diploid=compute_config.trusted_no_missing_diploid,
                )
            else:
                warm_regenie2_linear_bgen_cache(
                    genotype_source_config=genotype_source_config,
                    phenotype_path=Path(pheno),
                    phenotype_name=pheno_name,
                    prediction_list_path=Path(pred),
                    covariate_path=Path(covar) if covar is not None else None,
                    covariate_names=covariate_name_list,
                    chunk_size=compute_config.chunk_size,
                    variant_limit=compute_config.variant_limit,
                    trusted_no_missing_diploid=compute_config.trusted_no_missing_diploid,
                )
            engine.record_stage_duration(stage_timing_recorder, "jax_cache_warmup", warm_cache_start_time)
        output_run_directory = compute_config.output_run_directory or Path(out)
        association_mode = (
            types.AssociationMode.REGENIE2_BINARY
            if trait_type == types.RegenieTraitType.BINARY
            else types.AssociationMode.REGENIE2_LINEAR
        )
        output_start_time = time.perf_counter()
        prepared_output_run = prepare_output_run(
            output_root=output_run_directory,
            association_mode=association_mode,
            resume=compute_config.resume,
        )
        engine.record_stage_duration(stage_timing_recorder, "output_run_preparation", output_start_time)
        output_run_paths = prepared_output_run.output_run_paths
        committed_chunk_identifiers = set(prepared_output_run.committed_chunk_identifiers)

        if trait_type == types.RegenieTraitType.BINARY:
            final_parquet_path = run_regenie2_binary_bgen_pipeline(
                genotype_source_config=genotype_source_config,
                phenotype_path=Path(pheno),
                phenotype_name=pheno_name,
                prediction_list_path=Path(pred),
                covariate_path=Path(covar) if covar is not None else None,
                covariate_names=covariate_name_list,
                chunk_size=compute_config.chunk_size,
                variant_limit=compute_config.variant_limit,
                prefetch_chunks=compute_config.prefetch_chunks,
                output_run_paths=output_run_paths,
                committed_chunk_identifiers=committed_chunk_identifiers,
                finalize_parquet=compute_config.finalize_parquet,
                writer_thread_count=compute_config.output_writer_thread_count,
                writer_queue_depth=compute_config.output_writer_queue_depth,
                trusted_no_missing_diploid=compute_config.trusted_no_missing_diploid,
                correction_plan=binary_correction_plan,
                stage_timing_recorder=stage_timing_recorder,
            )
        else:
            final_parquet_path = run_regenie2_linear_bgen_pipeline(
                genotype_source_config=genotype_source_config,
                phenotype_path=Path(pheno),
                phenotype_name=pheno_name,
                prediction_list_path=Path(pred),
                covariate_path=Path(covar) if covar is not None else None,
                covariate_names=covariate_name_list,
                chunk_size=compute_config.chunk_size,
                variant_limit=compute_config.variant_limit,
                prefetch_chunks=compute_config.prefetch_chunks,
                output_run_paths=output_run_paths,
                committed_chunk_identifiers=committed_chunk_identifiers,
                finalize_parquet=compute_config.finalize_parquet,
                writer_thread_count=compute_config.output_writer_thread_count,
                writer_queue_depth=compute_config.output_writer_queue_depth,
                trusted_no_missing_diploid=compute_config.trusted_no_missing_diploid,
                stage_timing_recorder=stage_timing_recorder,
            )

        return RunArtifacts(
            output_run_directory=output_run_paths.run_directory,
            final_parquet=final_parquet_path,
        )
    finally:
        engine.record_stage_duration(stage_timing_recorder, "python_api_entry", api_entry_start_time)
        engine.write_stage_timing_snapshot_from_environment(stage_timing_recorder)


def regenie2_warm_cache(
    *,
    bgen: Path | str,
    sample: Path | str | None = None,
    pheno: Path | str,
    pheno_name: str,
    covar: Path | str | None = None,
    covar_names: str | list[str] | tuple[str, ...] | None = None,
    pred: Path | str,
    trait_type: types.RegenieTraitType = types.RegenieTraitType.QUANTITATIVE,
    compute: ComputeConfig | None = None,
    binary: Regenie2BinaryConfig | None = None,
) -> WarmCacheReport:
    """Warm JAX compilation-cache entries for a REGENIE step 2 CLI run."""
    compute_config = compute or ComputeConfig()
    validate_compute_config(compute_config)
    configure_jax_device(compute_config.device)
    covariate_name_list = parse_covariate_name_list(covar_names)
    genotype_source_config = source.build_bgen_source_config(bgen, sample)
    if trait_type == types.RegenieTraitType.BINARY:
        binary_config = binary or Regenie2BinaryConfig()
        binary_correction_plan = normalize_binary_correction_config(binary_config)
        return warm_regenie2_binary_bgen_cache(
            genotype_source_config=genotype_source_config,
            phenotype_path=Path(pheno),
            phenotype_name=pheno_name,
            prediction_list_path=Path(pred),
            covariate_path=Path(covar) if covar is not None else None,
            covariate_names=covariate_name_list,
            chunk_size=compute_config.chunk_size,
            variant_limit=compute_config.variant_limit,
            correction_plan=binary_correction_plan,
            trusted_no_missing_diploid=compute_config.trusted_no_missing_diploid,
        )
    return warm_regenie2_linear_bgen_cache(
        genotype_source_config=genotype_source_config,
        phenotype_path=Path(pheno),
        phenotype_name=pheno_name,
        prediction_list_path=Path(pred),
        covariate_path=Path(covar) if covar is not None else None,
        covariate_names=covariate_name_list,
        chunk_size=compute_config.chunk_size,
        variant_limit=compute_config.variant_limit,
        trusted_no_missing_diploid=compute_config.trusted_no_missing_diploid,
    )
