"""Native sample and prediction-source loaders."""

from __future__ import annotations

import typing

from g import _core, execution_plan
from g.engine.native_dispatch import groups, models

if typing.TYPE_CHECKING:
    from pathlib import Path


def load_native_aligned_sample_data(
    *,
    engine: _core.Regenie2RunEngine,
    sample_path: Path | None,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
    alignment_config: models.SampleAlignmentConfigProtocol | None,
) -> _core.NativeAlignedSampleData:
    """Load Rust-owned aligned sample data from a sample file or embedded BGEN samples."""
    return engine.align_sample_data(
        str(sample_path) if sample_path is not None else None,
        str(phenotype_path),
        phenotype_name,
        str(covariate_path) if covariate_path is not None else None,
        list(covariate_names) if covariate_names is not None else None,
        is_binary_trait,
        sample_key_mode=groups.resolve_sample_key_mode(alignment_config).value,
    )


def load_native_multi_aligned_sample_data(
    *,
    engine: _core.Regenie2RunEngine,
    sample_path: Path | None,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
    alignment_config: models.SampleAlignmentConfigProtocol | None,
) -> _core.NativeMultiAlignedSampleData:
    """Load Rust-owned complete-case multi-phenotype sample data."""
    return engine.align_multi_sample_data(
        str(sample_path) if sample_path is not None else None,
        str(phenotype_path),
        list(phenotype_names),
        str(covariate_path) if covariate_path is not None else None,
        list(covariate_names) if covariate_names is not None else None,
        is_binary_trait,
        sample_key_mode=groups.resolve_sample_key_mode(alignment_config).value,
    )


def load_native_bgen_run_input(
    *,
    genotype_source_config: execution_plan.GenotypeSourceConfig,
    engine: _core.Regenie2RunEngine,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
    alignment_config: models.SampleAlignmentConfigProtocol | None,
) -> models.NativeBgenRunInput:
    """Load native-aligned samples and JAX compute inputs for a native BGEN run."""
    native_aligned_sample_data = load_native_aligned_sample_data(
        engine=engine,
        sample_path=genotype_source_config.sample_path,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=is_binary_trait,
        alignment_config=alignment_config,
    )
    return models.build_native_bgen_run_input(native_aligned_sample_data)


def load_native_bgen_multi_run_input(
    *,
    genotype_source_config: execution_plan.GenotypeSourceConfig,
    engine: _core.Regenie2RunEngine,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
    alignment_config: models.SampleAlignmentConfigProtocol | None,
) -> models.NativeBgenMultiRunInput:
    """Load native complete-case multi-phenotype samples and JAX compute inputs."""
    native_multi_aligned_sample_data = load_native_multi_aligned_sample_data(
        engine=engine,
        sample_path=genotype_source_config.sample_path,
        phenotype_path=phenotype_path,
        phenotype_names=phenotype_names,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=is_binary_trait,
        alignment_config=alignment_config,
    )
    return models.build_native_bgen_multi_run_input(native_multi_aligned_sample_data)


def load_native_bgen_grouped_run_inputs(
    *,
    genotype_source_config: execution_plan.GenotypeSourceConfig,
    engine: _core.Regenie2RunEngine,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
    alignment_config: models.SampleAlignmentConfigProtocol | None,
    planned_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...] | None,
) -> tuple[models.NativeBgenGroupedRunInput, ...]:
    """Load native grouped per-phenotype samples and JAX compute inputs."""
    native_grouped_aligned_sample_data = engine.align_grouped_sample_data(
        str(genotype_source_config.sample_path) if genotype_source_config.sample_path is not None else None,
        str(phenotype_path),
        list(phenotype_names),
        str(covariate_path) if covariate_path is not None else None,
        list(covariate_names) if covariate_names is not None else None,
        is_binary_trait,
        sample_key_mode=groups.resolve_sample_key_mode(alignment_config).value,
    )
    prediction_sources = _core.MultiRegeniePredictionSource.from_native_grouped_aligned_sample_data(
        str(prediction_list_path),
        native_grouped_aligned_sample_data,
        sample_key_mode=groups.resolve_sample_key_mode(alignment_config).value,
    )
    return groups.build_native_bgen_grouped_run_inputs(
        native_grouped_aligned_sample_data,
        prediction_sources,
        prediction_list_path=prediction_list_path,
        planned_compute_groups=planned_compute_groups,
        alignment_config=alignment_config,
    )


def build_regenie_prediction_source(
    *,
    prediction_list_path: Path,
    phenotype_name: str,
    run_input: models.NativeBgenRunInput,
    alignment_config: models.SampleAlignmentConfigProtocol | None,
) -> _core.RegeniePredictionSource:
    """Load Rust-owned REGENIE step 1 predictions aligned to the run samples."""
    return _core.RegeniePredictionSource.from_native_aligned_sample_data(
        str(prediction_list_path),
        phenotype_name,
        run_input.native_aligned_sample_data,
        sample_key_mode=groups.resolve_sample_key_mode(alignment_config).value,
    )


def build_multi_regenie_prediction_source(
    *,
    prediction_list_path: Path,
    run_input: models.NativeBgenMultiRunInput,
    alignment_config: models.SampleAlignmentConfigProtocol | None,
) -> _core.MultiRegeniePredictionSource:
    """Load native multi-trait REGENIE step 1 predictions aligned to shared samples."""
    return _core.MultiRegeniePredictionSource.from_native_multi_aligned_sample_data(
        str(prediction_list_path),
        run_input.native_multi_aligned_sample_data,
        sample_key_mode=groups.resolve_sample_key_mode(alignment_config).value,
    )
