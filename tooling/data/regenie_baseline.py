"""Prepare and verify REGENIE step 1 baseline inputs."""

from __future__ import annotations

import enum
import typing
from dataclasses import dataclass
from pathlib import Path

from tooling.benchmark import benchmark as baseline_benchmark
from tooling.common import hydra_arguments as tooling_hydra_arguments

if typing.TYPE_CHECKING:
    import omegaconf


class RegenieBaselineTrait(enum.StrEnum):
    """Trait types supported by the REGENIE baseline data tool."""

    BINARY = "binary"
    QUANTITATIVE = "quantitative"


@dataclass(frozen=True)
class RegenieBaselineArguments:
    """Resolved arguments for REGENIE baseline input preparation.

    Attributes:
        data_directory: Data directory containing prepared local fixture inputs.
        trait: Trait type to prepare or verify.
        verify_only: Whether to only verify expected inputs and outputs.

    """

    data_directory: Path
    trait: RegenieBaselineTrait
    verify_only: bool


def required_input_paths(baseline_paths: baseline_benchmark.BaselinePaths) -> tuple[Path, ...]:
    """Return shared local input paths required by REGENIE step 1."""
    return (
        baseline_paths.bed_prefix.with_suffix(".bed"),
        baseline_paths.bed_prefix.with_suffix(".bim"),
        baseline_paths.bed_prefix.with_suffix(".fam"),
        baseline_paths.bgen_path,
        baseline_paths.sample_path,
        baseline_paths.covariate_path,
    )


def trait_input_paths(
    baseline_paths: baseline_benchmark.BaselinePaths,
    trait: RegenieBaselineTrait,
) -> tuple[Path, ...]:
    """Return trait-specific local input paths required by REGENIE step 1."""
    if trait == RegenieBaselineTrait.BINARY:
        return (baseline_paths.binary_phenotype_path,)
    if trait == RegenieBaselineTrait.QUANTITATIVE:
        return (baseline_paths.continuous_phenotype_path,)
    typing.assert_never(trait)


def trait_prediction_list_path(
    baseline_paths: baseline_benchmark.BaselinePaths,
    trait: RegenieBaselineTrait,
) -> Path:
    """Return the expected REGENIE step 1 prediction list path for a trait."""
    if trait == RegenieBaselineTrait.BINARY:
        return baseline_paths.regenie_prediction_list_path
    if trait == RegenieBaselineTrait.QUANTITATIVE:
        quantitative_prediction_list_path = baseline_paths.regenie_qt_prediction_list_path
        if quantitative_prediction_list_path is None:
            message = "Quantitative REGENIE prediction list path is not configured."
            raise ValueError(message)
        return quantitative_prediction_list_path
    typing.assert_never(trait)


def collect_missing_paths(paths: typing.Iterable[Path]) -> tuple[Path, ...]:
    """Return missing or empty paths from a required path collection."""
    missing_paths: list[Path] = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            missing_paths.append(path)
    return tuple(missing_paths)


def verify_baseline_inputs(
    baseline_paths: baseline_benchmark.BaselinePaths,
    trait: RegenieBaselineTrait,
) -> None:
    """Verify required local inputs and generated prediction lists."""
    expected_paths = (
        *required_input_paths(baseline_paths),
        *trait_input_paths(baseline_paths, trait),
        trait_prediction_list_path(baseline_paths, trait),
    )
    missing_paths = collect_missing_paths(expected_paths)
    if missing_paths:
        missing_lines = "\n".join(str(path) for path in missing_paths)
        message = f"Required REGENIE baseline paths are missing or empty:\n{missing_lines}"
        raise FileNotFoundError(message)
    print(f"REGENIE {trait.value} baseline inputs are present under {baseline_paths.data_directory}.")


def run_baseline_command(
    regenie_executable: str,
    baseline_paths: baseline_benchmark.BaselinePaths,
    trait: RegenieBaselineTrait,
) -> None:
    """Run the REGENIE step 1 baseline command for one trait."""
    baseline_paths.baseline_directory.mkdir(parents=True, exist_ok=True)
    expected_input_paths = (*required_input_paths(baseline_paths), *trait_input_paths(baseline_paths, trait))
    missing_input_paths = collect_missing_paths(expected_input_paths)
    if missing_input_paths:
        missing_lines = "\n".join(str(path) for path in missing_input_paths)
        message = f"Required REGENIE step 1 inputs are missing or empty:\n{missing_lines}"
        raise FileNotFoundError(message)

    if trait == RegenieBaselineTrait.BINARY:
        command_arguments = baseline_benchmark.build_regenie_step1_command(regenie_executable, baseline_paths)
        output_prefix = baseline_paths.baseline_directory / "regenie_step1"
        command_name = "Regenie Step 1 Binary"
    elif trait == RegenieBaselineTrait.QUANTITATIVE:
        command_arguments = baseline_benchmark.build_regenie_step1_continuous_command(
            regenie_executable,
            baseline_paths,
        )
        output_prefix = baseline_paths.baseline_directory / "regenie_step1_qt"
        command_name = "Regenie Step 1 Quantitative"
    else:
        typing.assert_never(trait)

    command_result = baseline_benchmark.run_command(command_name, command_arguments, output_prefix)
    prediction_list_path = trait_prediction_list_path(baseline_paths, trait)
    if not command_result.success or not prediction_list_path.exists() or prediction_list_path.stat().st_size == 0:
        message = f"REGENIE step 1 failed to produce {prediction_list_path}."
        raise RuntimeError(message)


def run_tool(arguments: RegenieBaselineArguments) -> None:
    """Prepare or verify REGENIE step 1 baseline inputs."""
    baseline_paths = baseline_benchmark.build_baseline_paths(arguments.data_directory)
    if arguments.verify_only:
        verify_baseline_inputs(baseline_paths, arguments.trait)
        return

    regenie_executable = baseline_benchmark.resolve_required_executable("REGENIE_BIN", "regenie")
    run_baseline_command(regenie_executable, baseline_paths, arguments.trait)
    verify_baseline_inputs(baseline_paths, arguments.trait)


def build_arguments_from_config(config: omegaconf.DictConfig) -> RegenieBaselineArguments:
    """Resolve REGENIE baseline arguments from Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return RegenieBaselineArguments(
        data_directory=tooling_hydra_arguments.path_or_none(tool_values["data_directory"]) or Path("data"),
        trait=RegenieBaselineTrait(str(tool_values["trait"])),
        verify_only=tooling_hydra_arguments.boolean_value(tool_values["verify_only"]),
    )
