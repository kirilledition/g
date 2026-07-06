"""Native sample and prediction-source loaders."""

from __future__ import annotations

import typing

from g import _core, execution_plan
from g.engine.native_dispatch import groups, models

if typing.TYPE_CHECKING:
    from pathlib import Path


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
    native_aligned_sample_data = engine.align_sample_data(
        str(genotype_source_config.sample_path) if genotype_source_config.sample_path is not None else None,
        str(phenotype_path),
        phenotype_name,
        str(covariate_path) if covariate_path is not None else None,
        list(covariate_names) if covariate_names is not None else None,
        is_binary_trait,
        sample_key_mode=groups.resolve_sample_key_mode(alignment_config).value,
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
    native_multi_aligned_sample_data = engine.align_multi_sample_data(
        str(genotype_source_config.sample_path) if genotype_source_config.sample_path is not None else None,
        str(phenotype_path),
        list(phenotype_names),
        str(covariate_path) if covariate_path is not None else None,
        list(covariate_names) if covariate_names is not None else None,
        is_binary_trait,
        sample_key_mode=groups.resolve_sample_key_mode(alignment_config).value,
    )
    return models.build_native_bgen_multi_run_input(native_multi_aligned_sample_data)
