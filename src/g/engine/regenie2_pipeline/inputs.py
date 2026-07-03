"""Native input loading helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import typing

from g.engine.native_dispatch import groups as native_dispatch_groups
from g.engine.native_dispatch import loaders as native_dispatch_loaders
from g.engine.native_dispatch import models as native_dispatch_models

if typing.TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from g import _core, execution_plan, types

type BgenDeliveryCallbackProtocol = native_dispatch_models.BgenDeliveryCallbackProtocol
type NativeBgenRunInput = native_dispatch_models.NativeBgenRunInput
type NativeBgenMultiRunInput = native_dispatch_models.NativeBgenMultiRunInput
type NativeBgenGroupedRunInput = native_dispatch_models.NativeBgenGroupedRunInput
type NativeBgenUnionRunInput = native_dispatch_models.NativeBgenUnionRunInput
type SampleAlignmentConfigProtocol = native_dispatch_models.SampleAlignmentConfigProtocol
type RegeniePredictionSourceProtocol = native_dispatch_models.RegeniePredictionSourceProtocol
type MultiRegeniePredictionSourceProtocol = native_dispatch_models.MultiRegeniePredictionSourceProtocol


def load_native_bgen_run_input(
    *,
    genotype_source_config: execution_plan.GenotypeSourceConfig,
    engine: _core.Regenie2RunEngine,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
    alignment_config: SampleAlignmentConfigProtocol | None,
) -> NativeBgenRunInput:
    """Load native-aligned samples and JAX compute inputs for a native BGEN run."""
    return native_dispatch_loaders.load_native_bgen_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=is_binary_trait,
        alignment_config=alignment_config,
    )


def load_native_bgen_multi_run_input(
    *,
    genotype_source_config: execution_plan.GenotypeSourceConfig,
    engine: _core.Regenie2RunEngine,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
    alignment_config: SampleAlignmentConfigProtocol | None,
) -> NativeBgenMultiRunInput:
    """Load native complete-case multi-phenotype samples and JAX compute inputs."""
    return native_dispatch_loaders.load_native_bgen_multi_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_names=phenotype_names,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=is_binary_trait,
        alignment_config=alignment_config,
    )


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
    alignment_config: SampleAlignmentConfigProtocol | None,
    planned_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...] | None,
) -> tuple[NativeBgenGroupedRunInput, ...]:
    """Load native grouped per-phenotype samples and JAX compute inputs."""
    return native_dispatch_loaders.load_native_bgen_grouped_run_inputs(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_names=phenotype_names,
        prediction_list_path=prediction_list_path,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=is_binary_trait,
        alignment_config=alignment_config,
        planned_compute_groups=planned_compute_groups,
    )


def build_regenie_prediction_source(
    *,
    prediction_list_path: Path,
    phenotype_name: str,
    run_input: NativeBgenRunInput,
    alignment_config: SampleAlignmentConfigProtocol | None,
) -> _core.RegeniePredictionSource:
    """Load Rust-owned REGENIE step 1 predictions aligned to the run samples."""
    return native_dispatch_loaders.build_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
        alignment_config=alignment_config,
    )


def build_multi_regenie_prediction_source(
    *,
    prediction_list_path: Path,
    run_input: NativeBgenMultiRunInput,
    alignment_config: SampleAlignmentConfigProtocol | None,
) -> _core.MultiRegeniePredictionSource:
    """Load native multi-trait REGENIE step 1 predictions aligned to shared samples."""
    return native_dispatch_loaders.build_multi_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        run_input=run_input,
        alignment_config=alignment_config,
    )


def build_resolved_single_phenotype_compute_group(
    *,
    phenotype_name: str,
    run_input: NativeBgenRunInput,
    prediction_list_path: Path,
    alignment_config: SampleAlignmentConfigProtocol | None,
) -> execution_plan.PhenotypeComputeGroup:
    """Build the alignment-resolved single-phenotype compute group."""
    return native_dispatch_groups.build_resolved_single_phenotype_compute_group(
        phenotype_name=phenotype_name,
        run_input=run_input,
        prediction_list_path=prediction_list_path,
        alignment_config=alignment_config,
    )


def build_resolved_complete_case_phenotype_compute_group(
    *,
    run_input: NativeBgenMultiRunInput,
    prediction_list_path: Path,
    planned_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...],
    alignment_config: SampleAlignmentConfigProtocol | None,
) -> execution_plan.PhenotypeComputeGroup:
    """Build the alignment-resolved complete-case compute group."""
    return native_dispatch_groups.build_resolved_complete_case_phenotype_compute_group(
        run_input=run_input,
        prediction_list_path=prediction_list_path,
        planned_compute_groups=planned_compute_groups,
        alignment_config=alignment_config,
    )


def build_native_bgen_union_run_input(
    *,
    sample_indices: npt.NDArray[np.int64],
) -> NativeBgenUnionRunInput:
    """Build a union sample-selection input for grouped native BGEN delivery."""
    return native_dispatch_models.NativeBgenUnionRunInput(sample_indices=sample_indices)


def resolve_sample_key_mode(alignment_config: SampleAlignmentConfigProtocol | None) -> types.SampleKeyMode:
    """Resolve the sample key mode for native calls."""
    return native_dispatch_groups.resolve_sample_key_mode(alignment_config)
