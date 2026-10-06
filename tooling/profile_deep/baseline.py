"""Matched REGENIE baseline scope and setup commands."""

from __future__ import annotations

import typing

from tooling.benchmark import benchmark as baseline_benchmark
from tooling.profile_deep import execution as profile_deep_execution
from tooling.profile_deep import models as profile_deep_models

if typing.TYPE_CHECKING:
    from pathlib import Path


def build_baseline_paths(arguments: profile_deep_models.ProfileArguments) -> baseline_benchmark.BaselinePaths:
    """Build baseline paths from Hydra-resolved profile arguments."""
    return baseline_benchmark.BaselinePaths(
        data_directory=arguments.data_directory,
        baseline_directory=arguments.baseline_directory,
        bed_prefix=arguments.bed_prefix,
        bgen_path=arguments.bgen_path,
        sample_path=arguments.sample_path,
        continuous_phenotype_path=arguments.continuous_phenotype_path,
        binary_phenotype_path=arguments.binary_phenotype_path,
        covariate_path=arguments.covariate_path,
        hail_directory=arguments.data_directory / "hail",
        hail_matrix_table_path=arguments.data_directory / "hail" / f"{arguments.bed_prefix.name}.mt",
        hail_suite_report_path=arguments.baseline_directory / "hail_suite_report.json",
        regenie_prediction_list_path=arguments.regenie_prediction_list_path,
        regenie_qt_prediction_list_path=arguments.regenie_qt_prediction_list_path,
    )


def ensure_prediction_lists(
    *,
    baseline_paths: typing.Any,
    regenie_executable: str,
    log_directory: Path,
) -> list[profile_deep_models.TrialResult]:
    """Generate missing REGENIE step 1 prediction lists before profiling."""
    setup_results: list[profile_deep_models.TrialResult] = []
    prediction_specs = [
        (
            baseline_paths.regenie_prediction_list_path,
            "regenie_step1_binary_setup",
            baseline_benchmark.build_regenie_step1_command(regenie_executable, baseline_paths),
        ),
        (
            baseline_paths.regenie_qt_prediction_list_path,
            "regenie_step1_quantitative_setup",
            baseline_benchmark.build_regenie_step1_continuous_command(regenie_executable, baseline_paths),
        ),
    ]
    for prediction_path, name, command_arguments in prediction_specs:
        if prediction_path is not None and prediction_path.exists():
            continue
        setup_results.append(
            profile_deep_execution.run_logged_command(
                name=name,
                implementation="regenie",
                trait_type="setup",
                device="external_cpu",
                command_arguments=command_arguments,
                environment_overrides={},
                log_directory=log_directory,
            )
        )
    return setup_results


def replace_command_output_prefix(command_arguments: list[str], output_prefix: Path) -> list[str]:
    """Return a command with its --out value replaced."""
    updated_arguments = list(command_arguments)
    output_index = updated_arguments.index("--out")
    updated_arguments[output_index + 1] = str(output_prefix)
    return updated_arguments


def variant_metadata_candidate_paths(baseline_paths: baseline_benchmark.BaselinePaths) -> list[Path]:
    """Return metadata files that can provide BGEN-order variant identifiers."""
    return [
        baseline_paths.bgen_path.with_suffix(".pvar"),
        baseline_paths.bed_prefix.with_suffix(".bim"),
    ]


def read_pvar_variant_identifiers(metadata_path: Path, variant_limit: int) -> tuple[str, ...]:
    """Read the first variant identifiers from a PVAR file."""
    variant_identifiers: list[str] = []
    identifier_index = 2
    with metadata_path.open(encoding="utf-8") as metadata_file:
        for raw_line in metadata_file:
            line = raw_line.strip()
            if not line or line.startswith("##"):
                continue
            columns = line.split()
            if columns[0].startswith("#"):
                header_columns = [column.lstrip("#") for column in columns]
                if "ID" in header_columns:
                    identifier_index = header_columns.index("ID")
                continue
            if len(columns) <= identifier_index:
                continue
            variant_identifier = columns[identifier_index]
            if variant_identifier and variant_identifier != ".":
                variant_identifiers.append(variant_identifier)
            if len(variant_identifiers) >= variant_limit:
                break
    return tuple(variant_identifiers)


def read_bim_variant_identifiers(metadata_path: Path, variant_limit: int) -> tuple[str, ...]:
    """Read the first variant identifiers from a BIM file."""
    variant_identifiers: list[str] = []
    with metadata_path.open(encoding="utf-8") as metadata_file:
        for raw_line in metadata_file:
            columns = raw_line.strip().split()
            if len(columns) < 2:
                continue
            variant_identifier = columns[1]
            if variant_identifier and variant_identifier != ".":
                variant_identifiers.append(variant_identifier)
            if len(variant_identifiers) >= variant_limit:
                break
    return tuple(variant_identifiers)


def read_variant_identifiers(metadata_path: Path, variant_limit: int) -> tuple[str, ...]:
    """Read first variant identifiers from a supported metadata file."""
    if metadata_path.suffix == ".pvar":
        return read_pvar_variant_identifiers(metadata_path, variant_limit)
    if metadata_path.suffix == ".bim":
        return read_bim_variant_identifiers(metadata_path, variant_limit)
    message = f"Unsupported variant metadata file: {metadata_path}"
    raise ValueError(message)


def build_regenie_baseline_scope(
    *,
    arguments: profile_deep_models.ProfileArguments,
    baseline_paths: baseline_benchmark.BaselinePaths,
    output_directory: Path,
) -> profile_deep_models.RegenieBaselineScope:
    """Build original REGENIE workload scope for direct paired comparisons."""
    variant_limit = arguments.regenie_baseline_variant_limit
    if variant_limit is None:
        return profile_deep_models.RegenieBaselineScope(
            status=profile_deep_models.RegenieBaselineScopeStatus.FULL,
            variant_limit=None,
            extract_path=None,
            metadata_path=None,
            selected_variant_count=None,
            variant_identifiers=(),
            notes="Original REGENIE baseline uses the full configured BGEN workload.",
        )
    if variant_limit <= 0:
        return profile_deep_models.RegenieBaselineScope(
            status=profile_deep_models.RegenieBaselineScopeStatus.UNSUPPORTED,
            variant_limit=variant_limit,
            extract_path=None,
            metadata_path=None,
            selected_variant_count=None,
            variant_identifiers=(),
            notes="Bounded REGENIE baseline requires a positive variant limit.",
        )
    for metadata_path in variant_metadata_candidate_paths(baseline_paths):
        if not metadata_path.exists():
            continue
        variant_identifiers = read_variant_identifiers(metadata_path, variant_limit)
        if variant_identifiers:
            extract_path = output_directory / "headline_runs" / f"regenie_first_{len(variant_identifiers)}_variants.txt"
            return profile_deep_models.RegenieBaselineScope(
                status=profile_deep_models.RegenieBaselineScopeStatus.BOUNDED,
                variant_limit=variant_limit,
                extract_path=extract_path,
                metadata_path=metadata_path,
                selected_variant_count=len(variant_identifiers),
                variant_identifiers=variant_identifiers,
                notes=(
                    "Original REGENIE baseline is bounded with an --extract list derived from the first "
                    f"{len(variant_identifiers)} variants in {metadata_path}."
                ),
            )
    metadata_paths = ", ".join(str(path) for path in variant_metadata_candidate_paths(baseline_paths))
    return profile_deep_models.RegenieBaselineScope(
        status=profile_deep_models.RegenieBaselineScopeStatus.UNSUPPORTED,
        variant_limit=variant_limit,
        extract_path=None,
        metadata_path=None,
        selected_variant_count=None,
        variant_identifiers=(),
        notes=f"Bounded REGENIE baseline needs a .pvar or .bim metadata file; checked {metadata_paths}.",
    )


def write_regenie_baseline_extract_file(scope: profile_deep_models.RegenieBaselineScope) -> None:
    """Write the REGENIE extract list for a bounded baseline scope."""
    if scope.status != profile_deep_models.RegenieBaselineScopeStatus.BOUNDED or scope.extract_path is None:
        return
    scope.extract_path.parent.mkdir(parents=True, exist_ok=True)
    scope.extract_path.write_text("\n".join(scope.variant_identifiers) + "\n", encoding="utf-8")


def apply_regenie_baseline_scope(
    command_arguments: list[str],
    baseline_scope: profile_deep_models.RegenieBaselineScope,
) -> list[str]:
    """Apply bounded baseline filters to a REGENIE command."""
    updated_arguments = list(command_arguments)
    if baseline_scope.extract_path is not None:
        updated_arguments.extend(["--extract", str(baseline_scope.extract_path)])
    return updated_arguments


def build_regenie_step2_command(
    *,
    trait_type: str,
    regenie_executable: str,
    baseline_paths: typing.Any,
    output_prefix: Path,
    baseline_scope: profile_deep_models.RegenieBaselineScope,
) -> list[str]:
    """Build one original REGENIE step 2 command with an isolated output prefix."""
    if trait_type == "binary":
        base_command = baseline_benchmark.build_regenie_step2_command(regenie_executable, baseline_paths)
    else:
        base_command = baseline_benchmark.build_regenie_step2_continuous_command(regenie_executable, baseline_paths)
    return apply_regenie_baseline_scope(
        replace_command_output_prefix(base_command, output_prefix),
        baseline_scope,
    )
