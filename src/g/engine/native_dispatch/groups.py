"""Native sample-group fingerprints and compute-group resolution."""

from __future__ import annotations

import typing

from g import _core, execution_plan, types
from g.engine.native_dispatch import models

if typing.TYPE_CHECKING:
    from pathlib import Path


def build_native_bgen_grouped_run_inputs(
    native_grouped_aligned_sample_data: _core.NativeGroupedAlignedSampleData,
    prediction_sources: list[_core.MultiRegeniePredictionSource],
    *,
    prediction_list_path: Path | None,
    planned_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...] | None,
    alignment_config: models.SampleAlignmentConfigProtocol | None,
) -> tuple[models.NativeBgenGroupedRunInput, ...]:
    """Build Python/JAX views over native grouped per-phenotype alignment data."""
    if len(native_grouped_aligned_sample_data.groups) != len(prediction_sources):
        message = (
            "Grouped prediction source count does not match grouped aligned sample data count: "
            f"{len(prediction_sources)} prediction source(s), "
            f"{len(native_grouped_aligned_sample_data.groups)} aligned group(s)."
        )
        raise ValueError(message)
    grouped_run_inputs: list[models.NativeBgenGroupedRunInput] = []
    for native_group, prediction_source in zip(
        native_grouped_aligned_sample_data.groups,
        prediction_sources,
        strict=True,
    ):
        phenotype_indices = tuple(int(phenotype_index) for phenotype_index in native_group.phenotype_indices)
        run_input = models.build_native_bgen_multi_run_input(native_group.aligned_sample_data)
        compute_group = build_resolved_phenotype_compute_group(
            phenotype_indices=phenotype_indices,
            run_input=run_input,
            prediction_list_path=prediction_list_path,
            planned_compute_groups=planned_compute_groups,
            alignment_config=alignment_config,
        )
        grouped_run_inputs.append(
            models.NativeBgenGroupedRunInput(
                compute_group=compute_group,
                phenotype_indices=compute_group.phenotype_indices,
                run_input=run_input,
                prediction_source=prediction_source,
            )
        )
    return tuple(grouped_run_inputs)


def build_resolved_phenotype_compute_group(
    *,
    phenotype_indices: tuple[int, ...],
    run_input: models.NativeBgenMultiRunInput,
    prediction_list_path: Path | None,
    planned_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...] | None,
    alignment_config: models.SampleAlignmentConfigProtocol | None,
) -> execution_plan.PhenotypeComputeGroup:
    """Build one alignment-resolved per-phenotype compute group."""
    planned_names_by_index = build_planned_phenotype_names_by_index(planned_compute_groups)
    if planned_names_by_index:
        phenotype_names = tuple(planned_names_by_index[phenotype_index] for phenotype_index in phenotype_indices)
    else:
        phenotype_names = run_input.phenotype_names
    native_compute_group = resolve_native_per_phenotype_compute_group(
        phenotype_indices=phenotype_indices,
        phenotype_names=phenotype_names,
        run_input=run_input,
        prediction_list_path=prediction_list_path,
        alignment_config=alignment_config,
    )
    return adapt_native_phenotype_compute_group(native_compute_group)


def build_resolved_single_phenotype_compute_group(
    *,
    phenotype_name: str,
    run_input: models.NativeBgenRunInput,
    prediction_list_path: Path,
    alignment_config: models.SampleAlignmentConfigProtocol | None,
) -> execution_plan.PhenotypeComputeGroup:
    """Build the alignment-resolved single-phenotype compute group."""
    native_compute_group = resolve_native_single_phenotype_compute_group(
        phenotype_name=phenotype_name,
        run_input=run_input,
        prediction_list_path=prediction_list_path,
        alignment_config=alignment_config,
    )
    return adapt_native_phenotype_compute_group(native_compute_group)


def build_resolved_complete_case_phenotype_compute_group(
    *,
    run_input: models.NativeBgenMultiRunInput,
    prediction_list_path: Path,
    planned_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...],
    alignment_config: models.SampleAlignmentConfigProtocol | None,
) -> execution_plan.PhenotypeComputeGroup:
    """Build the alignment-resolved complete-case compute group."""
    planned_compute_group = find_complete_case_compute_group(planned_compute_groups)
    native_compute_group = resolve_native_complete_case_compute_group(
        run_input=run_input,
        prediction_list_path=prediction_list_path,
        planned_compute_group=planned_compute_group,
        alignment_config=alignment_config,
    )
    return adapt_native_phenotype_compute_group(native_compute_group)


def find_complete_case_compute_group(
    planned_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...],
) -> execution_plan.PhenotypeComputeGroup:
    """Return the planned complete-case compute group."""
    for planned_compute_group in planned_compute_groups:
        if planned_compute_group.group_mode == types.PhenotypeComputeGroupMode.COMPLETE_CASE:
            return planned_compute_group
    message = "A complete-case phenotype compute group is required for complete-case execution."
    raise ValueError(message)


def build_planned_phenotype_names_by_index(
    planned_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...] | None,
) -> dict[int, str]:
    """Build a lookup from planned phenotype indices to names."""
    if planned_compute_groups is None:
        return {}
    planned_names_by_index: dict[int, str] = {}
    for planned_compute_group in planned_compute_groups:
        for phenotype_index, phenotype_name in zip(
            planned_compute_group.phenotype_indices,
            planned_compute_group.phenotype_names,
            strict=True,
        ):
            planned_names_by_index[phenotype_index] = phenotype_name
    return planned_names_by_index


def resolve_sample_key_mode(alignment_config: models.SampleAlignmentConfigProtocol | None) -> types.SampleKeyMode:
    """Resolve the sample key mode for native calls."""
    if alignment_config is None:
        return types.SampleKeyMode.IID
    return alignment_config.sample_key_mode


def resolve_native_single_phenotype_compute_group(
    *,
    phenotype_name: str,
    run_input: models.NativeBgenRunInput,
    prediction_list_path: Path,
    alignment_config: models.SampleAlignmentConfigProtocol | None,
) -> typing.Any:
    """Resolve a single-phenotype compute group through native code."""
    return _core.resolve_single_phenotype_compute_group(
        run_input.native_aligned_sample_data,
        phenotype_name,
        str(prediction_list_path),
        resolve_sample_key_mode(alignment_config).value,
    )


def resolve_native_per_phenotype_compute_group(
    *,
    phenotype_indices: tuple[int, ...],
    phenotype_names: tuple[str, ...],
    run_input: models.NativeBgenMultiRunInput,
    prediction_list_path: Path | None,
    alignment_config: models.SampleAlignmentConfigProtocol | None,
) -> typing.Any:
    """Resolve a grouped per-phenotype compute group through native code."""
    return _core.resolve_per_phenotype_compute_group(
        run_input.native_multi_aligned_sample_data,
        list(phenotype_indices),
        list(phenotype_names),
        None if prediction_list_path is None else str(prediction_list_path),
        resolve_sample_key_mode(alignment_config).value,
    )


def resolve_native_complete_case_compute_group(
    *,
    run_input: models.NativeBgenMultiRunInput,
    prediction_list_path: Path,
    planned_compute_group: execution_plan.PhenotypeComputeGroup,
    alignment_config: models.SampleAlignmentConfigProtocol | None,
) -> typing.Any:
    """Resolve a complete-case compute group through native code."""
    return _core.resolve_complete_case_compute_group(
        run_input.native_multi_aligned_sample_data,
        list(planned_compute_group.phenotype_indices),
        list(planned_compute_group.phenotype_names),
        str(prediction_list_path),
        resolve_sample_key_mode(alignment_config).value,
    )


def adapt_native_phenotype_compute_group(native_compute_group: typing.Any) -> execution_plan.PhenotypeComputeGroup:
    """Convert a native resolved compute-group DTO to the public Python dataclass."""
    return execution_plan.PhenotypeComputeGroup(
        group_mode=types.PhenotypeComputeGroupMode(native_compute_group.group_mode),
        phenotype_indices=tuple(int(phenotype_index) for phenotype_index in native_compute_group.phenotype_indices),
        phenotype_names=tuple(native_compute_group.phenotype_names),
        sample_mode=types.MultiPhenotypeSampleMode(native_compute_group.sample_mode),
        sample_set_fingerprint=native_compute_group.sample_set_fingerprint,
        covariate_design_fingerprint=native_compute_group.covariate_design_fingerprint,
        prediction_alignment_fingerprint=native_compute_group.prediction_alignment_fingerprint,
    )
