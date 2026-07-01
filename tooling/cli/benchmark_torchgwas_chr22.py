#!/usr/bin/env python3
"""Benchmark single-trait chr22 linear GWAS against TorchGWAS."""

from __future__ import annotations

import dataclasses
import enum
import gzip
import logging
import os
import shlex
import time
import typing
from pathlib import Path

import hydra
import numpy as np

import tooling.configuration as tooling_configuration
from tooling.common import artifact_format as tooling_artifact_format
from tooling.common import commands as tooling_commands
from tooling.common import g_regenie as tooling_g_regenie
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import logging as tooling_logging
from tooling.common import paths as tooling_paths
from tooling.common import reports as tooling_reports

if typing.TYPE_CHECKING:
    import collections.abc

    import omegaconf


logger = logging.getLogger(__name__)
REPOSITORY_ROOT = tooling_paths.find_repository_root(Path(__file__))
TOOL_NAME = "benchmark_torchgwas_chr22"
TORCHGWAS_REPOSITORY_URL = "https://github.com/ZhiGroup/TorchGWAS.git"
TORCHGWAS_MAIN_COMMIT = "9e0070c8ca1e8fc862b98e6d3077652577cfceb5"
DEFAULT_OUTPUT_PARENT = Path("data/benchmarks")
DEFAULT_RUN_DIRECTORY_PREFIX = "torchgwas_chr22"


class BenchmarkTool(enum.StrEnum):
    """Tool exercised by one benchmark case."""

    G = "g"
    TORCHGWAS = "torchgwas"


class CacheState(enum.StrEnum):
    """Cache state exercised by one benchmark case."""

    COLD = "cold"
    WARM = "warm"


class RunStatus(enum.StrEnum):
    """Benchmark case status."""

    DRY_RUN = "dry_run"
    SUCCESS = "success"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    MISSING_EXECUTABLE = "missing_executable"


class TorchgwasGenotypeFormat(enum.StrEnum):
    """TorchGWAS genotype input format."""

    BGEN = "bgen"
    NPY = "npy"
    PLINK = "plink"


@dataclasses.dataclass(frozen=True)
class BenchmarkArguments:
    """Resolved TorchGWAS chr22 benchmark settings.

    Attributes:
        data_directory: Directory containing chr22 inputs.
        bgen_path: chr22 BGEN input.
        sample_path: chr22 sample input.
        bim_path: chr22 BIM input used to construct bounded smoke subsets.
        torchgwas_full_genotype_path: Full chr22 genotype input used by TorchGWAS.
        torchgwas_full_genotype_format: Full chr22 genotype format used by TorchGWAS.
        torchgwas_full_bim_path: Optional full chr22 BIM input used by TorchGWAS PLINK mode.
        torchgwas_full_fam_path: Optional full chr22 FAM input used by TorchGWAS PLINK mode.
        phenotype_path: Quantitative phenotype table.
        phenotype_column: Quantitative phenotype column.
        covariate_path: Covariate table.
        covariate_columns: Covariate column list.
        prediction_list_path: REGENIE Step 1 quantitative prediction list for g.
        chromosome_label: Chromosome label used in reports.
        output_parent: Parent directory for timestamped output directories.
        output_directory: Output directory for this benchmark run.
        dry_run: Whether to render commands without executing them.
        validate_inputs: Whether to validate required local input paths.
        variant_limit: Optional first-N variant cap used for smoke runs.
        chunk_size: Variant chunk size passed to both tools where supported.
        cpu_threads: Optional thread count passed to g.
        torchgwas_repository_url: TorchGWAS Git repository URL.
        torchgwas_commit: Pinned TorchGWAS commit.
        torchgwas_tools_directory: Repo-local tools directory for checkout and venv.
        torchgwas_python: Python executable used to create the TorchGWAS venv.
        torch_package: Torch package spec installed for TorchGWAS.
        torch_package_index_url: Optional package index URL for the Torch package.
        plink2_binary: plink2 executable used by TorchGWAS and subset preparation.
        command_timeout_seconds: Optional timeout for each executed command.
        g_runner_prefix: Command prefix used to invoke g.

    """

    data_directory: Path
    bgen_path: Path
    sample_path: Path
    bim_path: Path
    torchgwas_full_genotype_path: Path
    torchgwas_full_genotype_format: TorchgwasGenotypeFormat
    torchgwas_full_bim_path: Path | None
    torchgwas_full_fam_path: Path | None
    phenotype_path: Path
    phenotype_column: str
    covariate_path: Path
    covariate_columns: tuple[str, ...]
    prediction_list_path: Path
    chromosome_label: str
    output_parent: Path
    output_directory: Path
    dry_run: bool
    validate_inputs: bool
    variant_limit: int | None
    chunk_size: int
    cpu_threads: int | None
    torchgwas_repository_url: str
    torchgwas_commit: str
    torchgwas_tools_directory: Path
    torchgwas_python: str
    torch_package: str
    torch_package_index_url: str | None
    plink2_binary: str | None
    command_timeout_seconds: float | None
    g_runner_prefix: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class BenchmarkCase:
    """One concrete benchmark case."""

    case_id: str
    tool: BenchmarkTool
    cache_state: CacheState
    command_arguments: tuple[str, ...]
    output_directory: Path
    cache_directory: Path
    stdout_path: Path
    stderr_path: Path
    stage_timing_path: Path | None
    profile_summary_path: Path | None
    environment_overrides: dict[str, str]


@dataclasses.dataclass(frozen=True)
class TimedCommandResult:
    """Command result with wall timing."""

    command_id: str
    phase: str
    spec: tooling_commands.CommandSpec
    result: tooling_commands.CommandResult
    started_at: str | None
    finished_at: str | None
    wall_time_seconds: float | None


@dataclasses.dataclass(frozen=True)
class CaseResult:
    """Measured result for one benchmark case."""

    case_id: str
    tool: BenchmarkTool
    cache_state: CacheState
    status: RunStatus
    return_code: int | None
    wall_time_seconds: float | None
    command_arguments: tuple[str, ...]
    output_directory: str
    cache_directory: str
    output_row_count: int | None
    output_total_bytes: int | None
    cache_total_bytes: int | None
    stage_seconds: dict[str, float]


@dataclasses.dataclass(frozen=True)
class TorchgwasInputSpec:
    """TorchGWAS genotype inputs for one run."""

    genotype_path: Path
    genotype_format: TorchgwasGenotypeFormat
    sample_file: Path | None
    sample_ids_path: Path | None
    marker_ids_path: Path | None
    bim_path: Path | None
    fam_path: Path | None


def timestamped_output_directory(output_parent: Path, run_directory_prefix: str) -> Path:
    """Build the default timestamped output directory."""
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return output_parent / f"{run_directory_prefix}_{timestamp}"


def resolve_repo_path(value: typing.Any) -> Path:
    """Resolve a path relative to the repository root."""
    return tooling_paths.resolve_repo_relative_path(Path(str(value)), REPOSITORY_ROOT)


def resolve_data_path(data_directory: Path, value: typing.Any) -> Path:
    """Resolve one data path relative to the data directory."""
    return tooling_paths.resolve_data_path(data_directory, Path(str(value)))


def resolve_optional_data_path(data_directory: Path, value: typing.Any) -> Path | None:
    """Resolve an optional data path relative to the data directory."""
    if value is None:
        return None
    return resolve_data_path(data_directory, value)


def split_columns(raw_columns: str) -> tuple[str, ...]:
    """Split a comma-separated column list."""
    return tuple(column.strip() for column in raw_columns.split(",") if column.strip())


def optional_float(value: typing.Any) -> float | None:
    """Convert an optional config value to float."""
    if value is None:
        return None
    return float(value)


def parse_torchgwas_genotype_format(value: typing.Any) -> TorchgwasGenotypeFormat:
    """Parse a TorchGWAS genotype format from config."""
    try:
        return TorchgwasGenotypeFormat(str(value))
    except ValueError as error:
        supported_values = ", ".join(genotype_format.value for genotype_format in TorchgwasGenotypeFormat)
        message = f"Unsupported TorchGWAS genotype format {value!r}. Expected one of: {supported_values}."
        raise ValueError(message) from error


def build_arguments_from_config(config: omegaconf.DictConfig) -> BenchmarkArguments:
    """Build benchmark arguments from Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    data_directory = resolve_repo_path(tool_values["data_dir"])
    output_parent = resolve_repo_path(tool_values.get("output_parent", DEFAULT_OUTPUT_PARENT))
    explicit_output_directory = tooling_hydra_arguments.path_or_none(tool_values.get("output_dir"))
    if explicit_output_directory is not None:
        output_directory = tooling_paths.resolve_repo_relative_path(explicit_output_directory, REPOSITORY_ROOT)
        output_parent = output_directory.parent
    else:
        output_directory = timestamped_output_directory(
            output_parent,
            str(tool_values.get("run_directory_prefix", DEFAULT_RUN_DIRECTORY_PREFIX)),
        )
    runner_prefix = tuple(str(value) for value in typing.cast("list[typing.Any]", tool_values["g_runner_prefix"]))
    return BenchmarkArguments(
        data_directory=data_directory,
        bgen_path=resolve_data_path(data_directory, tool_values["bgen"]),
        sample_path=resolve_data_path(data_directory, tool_values["sample"]),
        bim_path=resolve_data_path(data_directory, tool_values["bim"]),
        torchgwas_full_genotype_path=resolve_data_path(data_directory, tool_values["torchgwas_full_genotype"]),
        torchgwas_full_genotype_format=parse_torchgwas_genotype_format(tool_values["torchgwas_full_genotype_format"]),
        torchgwas_full_bim_path=resolve_optional_data_path(data_directory, tool_values.get("torchgwas_full_bim")),
        torchgwas_full_fam_path=resolve_optional_data_path(data_directory, tool_values.get("torchgwas_full_fam")),
        phenotype_path=resolve_data_path(data_directory, tool_values["phenotype_file"]),
        phenotype_column=str(tool_values["phenotype_column"]),
        covariate_path=resolve_data_path(data_directory, tool_values["covariate_file"]),
        covariate_columns=split_columns(str(tool_values["covariate_columns"])),
        prediction_list_path=resolve_data_path(data_directory, tool_values["prediction_list"]),
        chromosome_label=str(tool_values["chromosome_label"]),
        output_parent=output_parent,
        output_directory=output_directory,
        dry_run=bool(tool_values["dry_run"]),
        validate_inputs=bool(tool_values["validate_inputs"]),
        variant_limit=tooling_hydra_arguments.integer_or_none(tool_values.get("variant_limit")),
        chunk_size=int(tool_values["chunk_size"]),
        cpu_threads=tooling_hydra_arguments.integer_or_none(tool_values.get("cpu_threads")),
        torchgwas_repository_url=str(tool_values["torchgwas_repository_url"]),
        torchgwas_commit=str(tool_values["torchgwas_commit"]),
        torchgwas_tools_directory=resolve_repo_path(tool_values["torchgwas_tools_dir"]),
        torchgwas_python=str(tool_values["torchgwas_python"]),
        torch_package=str(tool_values["torch_package"]),
        torch_package_index_url=(
            str(tool_values["torch_package_index_url"])
            if tool_values.get("torch_package_index_url") is not None
            else None
        ),
        plink2_binary=(str(tool_values["plink2_binary"]) if tool_values.get("plink2_binary") is not None else None),
        command_timeout_seconds=optional_float(tool_values.get("command_timeout_seconds")),
        g_runner_prefix=runner_prefix,
    )


def build_arguments_from_overrides(overrides: typing.Sequence[str] | None = None) -> BenchmarkArguments:
    """Build benchmark arguments from Hydra overrides."""
    config = tooling_configuration.compose_config(config_name="benchmark_torchgwas_chr22", overrides=overrides)
    return build_arguments_from_config(config)


def arguments_to_json_dict(arguments: BenchmarkArguments) -> dict[str, object]:
    """Convert benchmark arguments into a JSON-ready dictionary."""
    return {
        "chromosome_label": arguments.chromosome_label,
        "data_directory": str(arguments.data_directory),
        "bgen_path": str(arguments.bgen_path),
        "sample_path": str(arguments.sample_path),
        "bim_path": str(arguments.bim_path),
        "torchgwas_full_genotype_path": str(arguments.torchgwas_full_genotype_path),
        "torchgwas_full_genotype_format": arguments.torchgwas_full_genotype_format.value,
        "torchgwas_full_bim_path": (
            str(arguments.torchgwas_full_bim_path) if arguments.torchgwas_full_bim_path is not None else None
        ),
        "torchgwas_full_fam_path": (
            str(arguments.torchgwas_full_fam_path) if arguments.torchgwas_full_fam_path is not None else None
        ),
        "phenotype_path": str(arguments.phenotype_path),
        "phenotype_column": arguments.phenotype_column,
        "covariate_path": str(arguments.covariate_path),
        "covariate_columns": list(arguments.covariate_columns),
        "prediction_list_path": str(arguments.prediction_list_path),
        "output_parent": str(arguments.output_parent),
        "output_directory": str(arguments.output_directory),
        "dry_run": arguments.dry_run,
        "variant_limit": arguments.variant_limit,
        "chunk_size": arguments.chunk_size,
        "cpu_threads": arguments.cpu_threads,
        "torchgwas_repository_url": arguments.torchgwas_repository_url,
        "torchgwas_commit": arguments.torchgwas_commit,
        "torchgwas_tools_directory": str(arguments.torchgwas_tools_directory),
        "torchgwas_python": arguments.torchgwas_python,
        "torch_package": arguments.torch_package,
        "torch_package_index_url": arguments.torch_package_index_url,
        "plink2_binary": arguments.plink2_binary,
        "command_timeout_seconds": arguments.command_timeout_seconds,
        "g_runner_prefix": list(arguments.g_runner_prefix),
    }


def validate_input_paths(arguments: BenchmarkArguments) -> None:
    """Validate required benchmark inputs."""
    required_paths = [
        arguments.bgen_path,
        arguments.sample_path,
        arguments.bim_path,
        arguments.torchgwas_full_genotype_path,
        arguments.phenotype_path,
        arguments.covariate_path,
        arguments.prediction_list_path,
    ]
    if arguments.torchgwas_full_genotype_format == TorchgwasGenotypeFormat.PLINK:
        required_paths.extend(
            path for path in (arguments.torchgwas_full_bim_path, arguments.torchgwas_full_fam_path) if path is not None
        )
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        formatted_paths = "\n".join(f"- {path}" for path in missing_paths)
        message = f"Required {arguments.chromosome_label} TorchGWAS benchmark inputs are missing:\n{formatted_paths}"
        raise FileNotFoundError(message)


def torchgwas_source_directory(arguments: BenchmarkArguments) -> Path:
    """Return the pinned TorchGWAS source checkout path."""
    return arguments.torchgwas_tools_directory / arguments.torchgwas_commit / "source"


def torchgwas_python_environment_slug(python_executable: str) -> str:
    """Return a filesystem-safe Python interpreter slug."""
    executable_name = Path(python_executable).name or "python"
    slug = "".join(character if character.isalnum() else "_" for character in executable_name).strip("_")
    return slug or "python"


def torchgwas_venv_directory(arguments: BenchmarkArguments) -> Path:
    """Return the pinned TorchGWAS virtual environment path."""
    return (
        arguments.torchgwas_tools_directory
        / arguments.torchgwas_commit
        / f"venv-{torchgwas_python_environment_slug(arguments.torchgwas_python)}"
    )


def torchgwas_python_executable(arguments: BenchmarkArguments) -> Path:
    """Return the Python executable in the TorchGWAS venv."""
    return torchgwas_venv_directory(arguments) / "bin" / "python"


def torchgwas_console_script(arguments: BenchmarkArguments) -> Path:
    """Return the TorchGWAS console script in the TorchGWAS venv."""
    return torchgwas_venv_directory(arguments) / "bin" / "torchgwas"


def torchgwas_setup_specs(arguments: BenchmarkArguments) -> list[tuple[str, tooling_commands.CommandSpec]]:
    """Build setup commands for the pinned TorchGWAS checkout and venv."""
    source_directory = torchgwas_source_directory(arguments)
    venv_directory = torchgwas_venv_directory(arguments)
    setup_log_directory = arguments.output_directory / "logs" / "setup"
    clone_script = (
        "set -euo pipefail\n"
        f"mkdir -p {shlex.quote(str(source_directory.parent))}\n"
        f"if [ ! -d {shlex.quote(str(source_directory / '.git'))} ]; then\n"
        f"  git clone {shlex.quote(arguments.torchgwas_repository_url)} {shlex.quote(str(source_directory))}\n"
        "fi\n"
        f"git -C {shlex.quote(str(source_directory))} fetch --quiet origin\n"
        f"git -C {shlex.quote(str(source_directory))} checkout --quiet {shlex.quote(arguments.torchgwas_commit)}\n"
    )
    venv_script = (
        "set -euo pipefail\n"
        f"if [ ! -x {shlex.quote(str(venv_directory / 'bin' / 'python'))} ] "
        f"|| [ ! -x {shlex.quote(str(venv_directory / 'bin' / 'pip'))} ]; then\n"
        "  uv venv --no-project --seed --clear "
        f"--python {shlex.quote(arguments.torchgwas_python)} {shlex.quote(str(venv_directory))}\n"
        "fi\n"
        "uv pip install "
        f"--python {shlex.quote(str(venv_directory / 'bin' / 'python'))} "
        "--upgrade "
        + (
            f"--index {shlex.quote(arguments.torch_package_index_url)} "
            if arguments.torch_package_index_url is not None
            else ""
        )
        + f"{shlex.quote(arguments.torch_package)} "
        f"-e "
        f"{shlex.quote(str(source_directory))} pandas-plink\n"
    )
    cuda_probe_script = (
        "import json\n"
        "import torch\n"
        "cuda_available = torch.cuda.is_available()\n"
        "payload = {\n"
        "    'torch_version': torch.__version__,\n"
        "    'torch_cuda_version': torch.version.cuda,\n"
        "    'cuda_available': cuda_available,\n"
        "    'cuda_device_count': torch.cuda.device_count() if cuda_available else 0,\n"
        "    'cuda_device_name': torch.cuda.get_device_name(0) if cuda_available else None,\n"
        "}\n"
        "print(json.dumps(payload, sort_keys=True))\n"
        "if not cuda_available:\n"
        "    raise SystemExit('Torch CUDA is unavailable for the TorchGWAS benchmark.')\n"
    )
    return [
        (
            "torchgwas_checkout",
            tooling_commands.build_command_spec(
                ["bash", "-lc", clone_script],
                cwd=REPOSITORY_ROOT,
                timeout_seconds=arguments.command_timeout_seconds,
                stdout_path=setup_log_directory / "torchgwas_checkout.stdout.log",
                stderr_path=setup_log_directory / "torchgwas_checkout.stderr.log",
                stream=True,
            ),
        ),
        (
            "torchgwas_install",
            tooling_commands.build_command_spec(
                ["bash", "-lc", venv_script],
                cwd=REPOSITORY_ROOT,
                timeout_seconds=arguments.command_timeout_seconds,
                stdout_path=setup_log_directory / "torchgwas_install.stdout.log",
                stderr_path=setup_log_directory / "torchgwas_install.stderr.log",
                stream=True,
            ),
        ),
        (
            "torchgwas_cuda_probe",
            tooling_commands.build_command_spec(
                [str(venv_directory / "bin" / "python"), "-c", cuda_probe_script],
                cwd=REPOSITORY_ROOT,
                timeout_seconds=arguments.command_timeout_seconds,
                stdout_path=setup_log_directory / "torchgwas_cuda_probe.stdout.log",
                stderr_path=setup_log_directory / "torchgwas_cuda_probe.stderr.log",
                stream=True,
            ),
        ),
    ]


def read_first_bim_marker_identifiers(bim_path: Path, variant_limit: int) -> list[str]:
    """Read the first ``variant_limit`` marker IDs from a BIM file."""
    marker_identifiers: list[str] = []
    with bim_path.open("r", encoding="utf-8") as bim_file:
        for line in bim_file:
            fields = line.split()
            if len(fields) >= 2:
                marker_identifiers.append(fields[1])
            if len(marker_identifiers) >= variant_limit:
                break
    if len(marker_identifiers) < variant_limit:
        message = f"BIM file {bim_path} contains only {len(marker_identifiers)} markers, requested {variant_limit}."
        raise ValueError(message)
    return marker_identifiers


def write_lines(path: Path, values: collections.abc.Sequence[object]) -> None:
    """Write one value per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(value) for value in values) + "\n", encoding="utf-8")


def plink_raw_value_to_float(value: str) -> float:
    """Convert one PLINK2 raw genotype value."""
    if value == "NA":
        return float("nan")
    return float(value)


def convert_plink2_raw_to_numpy(
    *,
    raw_path: Path,
    genotype_path: Path,
    sample_ids_path: Path,
    marker_ids_path: Path,
) -> None:
    """Convert a PLINK2 additive raw export into TorchGWAS NumPy inputs."""
    with raw_path.open("r", encoding="utf-8") as raw_file:
        header = raw_file.readline().split()
        if not header:
            message = f"PLINK2 raw export is empty: {raw_path}"
            raise ValueError(message)
        marker_identifiers = header[6:]
        sample_identifiers: list[str] = []
        genotype_rows: list[list[float]] = []
        for line_number, line in enumerate(raw_file, start=2):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != len(header):
                message = (
                    f"PLINK2 raw export row {line_number} has {len(fields)} fields; "
                    f"expected {len(header)} fields from the header."
                )
                raise ValueError(message)
            sample_identifiers.append(fields[1])
            genotype_rows.append([plink_raw_value_to_float(value) for value in fields[6:]])
    genotype_array = np.asarray(genotype_rows, dtype=np.float32)
    genotype_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(genotype_path, genotype_array)
    write_lines(sample_ids_path, sample_identifiers)
    write_lines(marker_ids_path, marker_identifiers)


def prepare_variant_limited_torchgwas_inputs(
    *,
    arguments: BenchmarkArguments,
    command_results: list[TimedCommandResult],
) -> TorchgwasInputSpec:
    """Prepare a NumPy genotype subset when a smoke variant limit is requested."""
    if arguments.variant_limit is None:
        return TorchgwasInputSpec(
            genotype_path=arguments.torchgwas_full_genotype_path,
            genotype_format=arguments.torchgwas_full_genotype_format,
            sample_file=arguments.sample_path
            if arguments.torchgwas_full_genotype_format == TorchgwasGenotypeFormat.BGEN
            else None,
            sample_ids_path=None,
            marker_ids_path=None,
            bim_path=arguments.torchgwas_full_bim_path
            if arguments.torchgwas_full_genotype_format == TorchgwasGenotypeFormat.PLINK
            else None,
            fam_path=arguments.torchgwas_full_fam_path
            if arguments.torchgwas_full_genotype_format == TorchgwasGenotypeFormat.PLINK
            else None,
        )
    subset_directory = arguments.output_directory / "torchgwas_subset"
    genotype_path = subset_directory / "genotype.npy"
    sample_ids_path = subset_directory / "sample_ids.txt"
    marker_ids_path = subset_directory / "marker_ids.txt"
    if arguments.dry_run:
        return TorchgwasInputSpec(
            genotype_path=genotype_path,
            genotype_format=TorchgwasGenotypeFormat.NPY,
            sample_file=None,
            sample_ids_path=sample_ids_path,
            marker_ids_path=marker_ids_path,
            bim_path=None,
            fam_path=None,
        )
    raw_path = subset_directory / "plink2_subset.raw"
    if genotype_path.is_file() and sample_ids_path.is_file() and marker_ids_path.is_file():
        return TorchgwasInputSpec(
            genotype_path=genotype_path,
            genotype_format=TorchgwasGenotypeFormat.NPY,
            sample_file=None,
            sample_ids_path=sample_ids_path,
            marker_ids_path=marker_ids_path,
            bim_path=None,
            fam_path=None,
        )
    marker_list_path = subset_directory / "extract_markers.txt"
    write_lines(marker_list_path, read_first_bim_marker_identifiers(arguments.bim_path, arguments.variant_limit))
    plink2_binary = arguments.plink2_binary or "plink2"
    plink2_command = [
        plink2_binary,
        "--bgen",
        str(arguments.bgen_path),
        "ref-first",
        "--sample",
        str(arguments.sample_path),
        "--extract",
        str(marker_list_path),
        "--export",
        "A",
        "--out",
        str(subset_directory / "plink2_subset"),
    ]
    subset_log_directory = arguments.output_directory / "logs" / "setup"
    spec = tooling_commands.build_command_spec(
        plink2_command,
        cwd=REPOSITORY_ROOT,
        timeout_seconds=arguments.command_timeout_seconds,
        stdout_path=subset_log_directory / "torchgwas_subset_plink2.stdout.log",
        stderr_path=subset_log_directory / "torchgwas_subset_plink2.stderr.log",
        stream=True,
    )
    timed_result = run_timed_command("torchgwas_subset_plink2", "setup", spec)
    command_results.append(timed_result)
    if timed_result.result.return_code != 0:
        message = "Failed to prepare TorchGWAS variant-limited subset with plink2."
        raise RuntimeError(message)
    convert_plink2_raw_to_numpy(
        raw_path=raw_path,
        genotype_path=genotype_path,
        sample_ids_path=sample_ids_path,
        marker_ids_path=marker_ids_path,
    )
    return TorchgwasInputSpec(
        genotype_path=genotype_path,
        genotype_format=TorchgwasGenotypeFormat.NPY,
        sample_file=None,
        sample_ids_path=sample_ids_path,
        marker_ids_path=marker_ids_path,
        bim_path=None,
        fam_path=None,
    )


def build_environment_overrides() -> dict[str, str]:
    """Build child-process environment overrides."""
    python_path_entries = [str(REPOSITORY_ROOT)]
    existing_python_path = os.environ.get("PYTHONPATH")
    if existing_python_path:
        python_path_entries.append(existing_python_path)
    return {
        "PYTHONPATH": os.pathsep.join(python_path_entries),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".50",
    }


def append_option(command_arguments: list[str], option_name: str, value: object) -> None:
    """Append one CLI option and value."""
    command_arguments.extend([option_name, str(value)])


def append_optional_option(command_arguments: list[str], option_name: str, value: object | None) -> None:
    """Append one CLI option and value when the value is present."""
    if value is not None:
        append_option(command_arguments, option_name, value)


def build_g_command(
    arguments: BenchmarkArguments,
    case_id: str,
    cache_directory: Path,
    output_prefix: Path,
) -> list[str]:
    """Build the g quantitative chr22 command."""
    log_directory = arguments.output_directory / "logs" / case_id
    run_spec = tooling_g_regenie.RegenieRunSpec(
        trait_kind=tooling_g_regenie.RegenieTraitKind.QUANTITATIVE,
        command_prefix=arguments.g_runner_prefix,
        inputs=tooling_g_regenie.RegenieInputSpec(
            bgen_path=arguments.bgen_path,
            sample_path=arguments.sample_path,
            phenotype_path=arguments.phenotype_path,
            phenotype_columns=(arguments.phenotype_column,),
            covariate_path=arguments.covariate_path,
            covariate_columns=arguments.covariate_columns,
            prediction_list_path=arguments.prediction_list_path,
            output_prefix=output_prefix,
        ),
        compute=tooling_g_regenie.RegenieComputeOptions(
            device=tooling_g_regenie.RegenieDevice.GPU,
            bsize=arguments.chunk_size,
            threads=arguments.cpu_threads,
            staging_depth=1,
            native_callback_batch_size=None,
            result_in_flight_limit=None,
            dosage_buffer_limit=None,
            variant_limit=arguments.variant_limit,
            trusted_no_missing_diploid=True,
            trusted_bgen_validation_mode="cache_on_miss",
            bgen_decode_tile_variant_count=None,
            firth_batch_size=None,
            firth_candidate_capacity=None,
            gpu_genotype_format="dosage",
            jax_cache_dir=cache_directory,
            jax_persistent_cache=True,
            jax_persistent_cache_min_entry_size_bytes=-1,
            jax_persistent_cache_min_compile_time_seconds=0,
            jax_xla_autotune_cache=False,
        ),
        output=tooling_g_regenie.RegenieOutputOptions(
            output_format="parquet",
            output_run_directory=None,
            writer_threads=8,
            writer_queue_depth=8,
            chunks_per_arrow_file=None,
            arrow_compression=None,
            parquet_compression=None,
            output_statistic_dtype=None,
            finalize_parquet=False,
        ),
        diagnostics=tooling_g_regenie.RegenieDiagnosticsOptions(
            telemetry="progress",
            log_dir=log_directory,
            stage_timings_json=log_directory / "stage_timings.json",
            profile_summary_json=log_directory / "profile_summary.json",
            log_file=log_directory / "events.jsonl",
            log_filter="info",
            log_stderr=True,
            progress_interval_seconds=5.0,
            progress_interval_chunks=10,
        ),
        binary=None,
    )
    return tooling_g_regenie.render_g_regenie_cli(run_spec)


def build_torchgwas_command(
    *,
    arguments: BenchmarkArguments,
    cache_directory: Path,
    output_directory: Path,
    input_spec: TorchgwasInputSpec,
) -> list[str]:
    """Build the TorchGWAS linear command."""
    command_arguments = [
        str(torchgwas_console_script(arguments)),
        "linear",
    ]
    append_option(command_arguments, "--genotype", input_spec.genotype_path)
    append_option(command_arguments, "--genotype-format", input_spec.genotype_format.value)
    append_optional_option(command_arguments, "--sample-file", input_spec.sample_file)
    append_optional_option(command_arguments, "--sample-ids", input_spec.sample_ids_path)
    append_optional_option(command_arguments, "--marker-ids", input_spec.marker_ids_path)
    append_optional_option(command_arguments, "--bim", input_spec.bim_path)
    append_optional_option(command_arguments, "--fam", input_spec.fam_path)
    append_option(command_arguments, "--genotype-cache-dir", cache_directory)
    append_optional_option(command_arguments, "--plink2-binary", arguments.plink2_binary)
    append_option(command_arguments, "--compute-dtype", "float32")
    append_option(command_arguments, "--chunk-size", arguments.chunk_size)
    append_option(command_arguments, "--phenotype-table", arguments.phenotype_path)
    append_option(command_arguments, "--trait-columns", arguments.phenotype_column)
    append_option(command_arguments, "--covariates-table", arguments.covariate_path)
    append_option(command_arguments, "--covariate-columns", ",".join(arguments.covariate_columns))
    append_option(command_arguments, "--output-dir", output_directory)
    return command_arguments


def build_cases(arguments: BenchmarkArguments, input_spec: TorchgwasInputSpec) -> list[BenchmarkCase]:
    """Build the cold and warm benchmark cases."""
    cases: list[BenchmarkCase] = []
    environment_overrides = build_environment_overrides()
    g_cache_directory = arguments.output_directory / "caches" / "g"
    torchgwas_cache_directory = arguments.output_directory / "caches" / "torchgwas"
    for cache_state in (CacheState.COLD, CacheState.WARM):
        g_case_id = f"g_linear_gpu_{cache_state.value}"
        g_output_directory = arguments.output_directory / "runs" / g_case_id
        g_output_prefix = g_output_directory / "linear"
        g_log_directory = arguments.output_directory / "logs" / g_case_id
        cases.append(
            BenchmarkCase(
                case_id=g_case_id,
                tool=BenchmarkTool.G,
                cache_state=cache_state,
                command_arguments=tuple(build_g_command(arguments, g_case_id, g_cache_directory, g_output_prefix)),
                output_directory=g_output_directory,
                cache_directory=g_cache_directory,
                stdout_path=g_log_directory / "stdout.log",
                stderr_path=g_log_directory / "stderr.log",
                stage_timing_path=g_log_directory / "stage_timings.json",
                profile_summary_path=g_log_directory / "profile_summary.json",
                environment_overrides=environment_overrides,
            )
        )
        torchgwas_case_id = f"torchgwas_linear_gpu_{cache_state.value}"
        torchgwas_output_directory = arguments.output_directory / "runs" / torchgwas_case_id
        torchgwas_log_directory = arguments.output_directory / "logs" / torchgwas_case_id
        cases.append(
            BenchmarkCase(
                case_id=torchgwas_case_id,
                tool=BenchmarkTool.TORCHGWAS,
                cache_state=cache_state,
                command_arguments=tuple(
                    build_torchgwas_command(
                        arguments=arguments,
                        cache_directory=torchgwas_cache_directory,
                        output_directory=torchgwas_output_directory,
                        input_spec=input_spec,
                    )
                ),
                output_directory=torchgwas_output_directory,
                cache_directory=torchgwas_cache_directory,
                stdout_path=torchgwas_log_directory / "stdout.log",
                stderr_path=torchgwas_log_directory / "stderr.log",
                stage_timing_path=None,
                profile_summary_path=None,
                environment_overrides={},
            )
        )
    return cases


def run_timed_command(command_id: str, phase: str, spec: tooling_commands.CommandSpec) -> TimedCommandResult:
    """Run one command and capture wall timing."""
    started_at = tooling_artifact_format.utc_now()
    start_time = time.perf_counter()
    result = tooling_commands.run_command(spec)
    wall_time_seconds = time.perf_counter() - start_time
    finished_at = tooling_artifact_format.utc_now()
    return TimedCommandResult(
        command_id=command_id,
        phase=phase,
        spec=spec,
        result=result,
        started_at=started_at,
        finished_at=finished_at,
        wall_time_seconds=wall_time_seconds,
    )


def case_status_from_command_result(result: tooling_commands.CommandResult) -> RunStatus:
    """Map a command result to benchmark case status."""
    if result.timed_out:
        return RunStatus.TIMED_OUT
    if result.missing_executable:
        return RunStatus.MISSING_EXECUTABLE
    if result.return_code == 0:
        return RunStatus.SUCCESS
    return RunStatus.FAILED


def directory_size_bytes(path: Path) -> int | None:
    """Return total bytes under a directory if it exists."""
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    return sum(child_path.stat().st_size for child_path in path.rglob("*") if child_path.is_file())


def load_json_mapping(path: Path) -> dict[str, typing.Any] | None:
    """Load a JSON object if present."""
    if not path.is_file():
        return None
    return tooling_reports.read_json_report(path)


def load_stage_seconds(path: Path | None) -> dict[str, float]:
    """Load stage totals from a g stage timing file."""
    if path is None:
        return {}
    payload = load_json_mapping(path)
    if payload is None:
        return {}
    raw_stage_seconds = payload.get("stage_totals_seconds")
    if not isinstance(raw_stage_seconds, dict):
        return {}
    return {str(key): float(value) for key, value in raw_stage_seconds.items()}


def discover_g_output_run_directory(case: BenchmarkCase) -> Path:
    """Discover the g output run directory for one completed case."""
    expected_root = Path(f"{case.output_directory / 'linear'}.g")
    discovered_directories = sorted(
        path
        for path in expected_root.glob("*.regenie2_linear.run")
        if path.is_dir() and (path / "run_manifest.json").is_file()
    )
    if len(discovered_directories) == 1:
        return discovered_directories[0]
    return expected_root


def measure_g_outputs(case: BenchmarkCase) -> tuple[int | None, int | None]:
    """Measure g output rows and bytes."""
    output_run_directory = discover_g_output_run_directory(case)
    manifest_payload = load_json_mapping(output_run_directory / "run_manifest.json")
    committed_chunks = []
    if manifest_payload is not None and isinstance(manifest_payload.get("committed_chunks"), list):
        committed_chunks = typing.cast("list[typing.Any]", manifest_payload["committed_chunks"])
    output_row_count = 0
    for chunk_payload in committed_chunks:
        if isinstance(chunk_payload, dict) and chunk_payload.get("row_count") is not None:
            output_row_count += int(chunk_payload["row_count"])
    return output_row_count if committed_chunks else None, directory_size_bytes(output_run_directory)


def count_gzip_table_rows(path: Path) -> int | None:
    """Count data rows in a gzipped TSV table."""
    if not path.is_file():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as table_file:
        line_count = sum(1 for _ in table_file)
    return max(line_count - 1, 0)


def measure_case_outputs(case: BenchmarkCase) -> tuple[int | None, int | None, dict[str, float]]:
    """Measure case output rows, bytes, and stage timings."""
    if case.tool == BenchmarkTool.G:
        output_row_count, output_total_bytes = measure_g_outputs(case)
        return output_row_count, output_total_bytes, load_stage_seconds(case.stage_timing_path)
    return (
        count_gzip_table_rows(case.output_directory / "results.tsv.gz"),
        directory_size_bytes(case.output_directory),
        {},
    )


def run_one_case(arguments: BenchmarkArguments, case: BenchmarkCase) -> tuple[CaseResult, TimedCommandResult | None]:
    """Run one case or materialize a dry-run result."""
    if arguments.dry_run:
        logger.info("Dry-run %s: %s", case.case_id, shlex.join(case.command_arguments))
        return (
            CaseResult(
                case_id=case.case_id,
                tool=case.tool,
                cache_state=case.cache_state,
                status=RunStatus.DRY_RUN,
                return_code=None,
                wall_time_seconds=None,
                command_arguments=case.command_arguments,
                output_directory=str(case.output_directory),
                cache_directory=str(case.cache_directory),
                output_row_count=None,
                output_total_bytes=None,
                cache_total_bytes=None,
                stage_seconds={},
            ),
            None,
        )
    command_spec = tooling_commands.build_command_spec(
        case.command_arguments,
        cwd=REPOSITORY_ROOT,
        env=case.environment_overrides,
        timeout_seconds=arguments.command_timeout_seconds,
        stdout_path=case.stdout_path,
        stderr_path=case.stderr_path,
        stream=True,
    )
    timed_result = run_timed_command(case.case_id, "benchmark", command_spec)
    status = case_status_from_command_result(timed_result.result)
    output_row_count: int | None = None
    output_total_bytes: int | None = None
    stage_seconds: dict[str, float] = {}
    if status == RunStatus.SUCCESS:
        output_row_count, output_total_bytes, stage_seconds = measure_case_outputs(case)
    case_result = CaseResult(
        case_id=case.case_id,
        tool=case.tool,
        cache_state=case.cache_state,
        status=status,
        return_code=timed_result.result.return_code,
        wall_time_seconds=timed_result.wall_time_seconds,
        command_arguments=case.command_arguments,
        output_directory=str(case.output_directory),
        cache_directory=str(case.cache_directory),
        output_row_count=output_row_count,
        output_total_bytes=output_total_bytes,
        cache_total_bytes=directory_size_bytes(case.cache_directory),
        stage_seconds=stage_seconds,
    )
    logger.info("Finished %s with status=%s.", case.case_id, status.value)
    return case_result, timed_result


def case_result_to_json_dict(case_result: CaseResult) -> dict[str, typing.Any]:
    """Convert a case result into a JSON-ready dictionary."""
    return {
        "case_id": case_result.case_id,
        "tool": case_result.tool.value,
        "cache_state": case_result.cache_state.value,
        "status": case_result.status.value,
        "return_code": case_result.return_code,
        "wall_time_seconds": case_result.wall_time_seconds,
        "command_arguments": list(case_result.command_arguments),
        "output_directory": case_result.output_directory,
        "cache_directory": case_result.cache_directory,
        "output_row_count": case_result.output_row_count,
        "output_total_bytes": case_result.output_total_bytes,
        "cache_total_bytes": case_result.cache_total_bytes,
        "stage_seconds": case_result.stage_seconds,
    }


def status_to_artifact_status(status: RunStatus) -> tooling_artifact_format.ToolArtifactStatus:
    """Map a benchmark run status to an artifact status."""
    if status == RunStatus.DRY_RUN:
        return tooling_artifact_format.ToolArtifactStatus.DRY_RUN
    if status == RunStatus.TIMED_OUT:
        return tooling_artifact_format.ToolArtifactStatus.TIMED_OUT
    if status == RunStatus.SUCCESS:
        return tooling_artifact_format.ToolArtifactStatus.SUCCESS
    return tooling_artifact_format.ToolArtifactStatus.FAILED


def benchmark_artifact_status(
    arguments: BenchmarkArguments,
    case_results: list[CaseResult],
) -> tooling_artifact_format.ToolArtifactStatus:
    """Return the overall artifact status."""
    if arguments.dry_run:
        return tooling_artifact_format.ToolArtifactStatus.DRY_RUN
    if any(case_result.status != RunStatus.SUCCESS for case_result in case_results):
        return tooling_artifact_format.ToolArtifactStatus.FAILED
    return tooling_artifact_format.ToolArtifactStatus.SUCCESS


def build_metric_dimensions(arguments: BenchmarkArguments, case_result: CaseResult) -> dict[str, object]:
    """Build common dimensions for one metric."""
    if case_result.tool == BenchmarkTool.G:
        genotype_format = TorchgwasGenotypeFormat.BGEN.value
    elif arguments.variant_limit is None:
        genotype_format = arguments.torchgwas_full_genotype_format.value
    else:
        genotype_format = TorchgwasGenotypeFormat.NPY.value
    return {
        "chromosome_label": arguments.chromosome_label,
        "trait_type": "quantitative",
        "phenotype_column": arguments.phenotype_column,
        "tool": case_result.tool.value,
        "genotype_format": genotype_format,
        "cache_state": case_result.cache_state.value,
        "device": "gpu",
        "chunk_size": arguments.chunk_size,
        "variant_limit": arguments.variant_limit,
        "torchgwas_commit": arguments.torchgwas_commit if case_result.tool == BenchmarkTool.TORCHGWAS else None,
        "comparison_scope": "workflow_runtime_not_statistical_parity",
    }


def metric_unit(metric_name: str) -> str:
    """Return the unit for a normalized metric."""
    if metric_name == "wall_time_seconds" or metric_name.startswith("stage."):
        return tooling_artifact_format.MetricUnit.SECONDS.value
    if metric_name.endswith("_bytes"):
        return tooling_artifact_format.MetricUnit.BYTES.value
    if metric_name == "output_row_count":
        return tooling_artifact_format.MetricUnit.ROW.value
    if metric_name.endswith("_count"):
        return tooling_artifact_format.MetricUnit.COUNT.value
    return tooling_artifact_format.MetricUnit.COUNT.value


def build_metrics(
    *,
    arguments: BenchmarkArguments,
    run_id: str,
    case_results: list[CaseResult],
) -> list[tooling_artifact_format.MetricRecord]:
    """Build normalized benchmark metrics."""
    metric_records: list[tooling_artifact_format.MetricRecord] = []
    for case_index, case_result in enumerate(case_results):
        case_metrics: dict[str, float | int | None] = {
            "wall_time_seconds": case_result.wall_time_seconds,
            "output_row_count": case_result.output_row_count,
            "output_total_bytes": case_result.output_total_bytes,
            "cache_total_bytes": case_result.cache_total_bytes,
        }
        for stage_name, stage_seconds in case_result.stage_seconds.items():
            case_metrics[f"stage.{stage_name}"] = stage_seconds
        for metric_name, metric_value in case_metrics.items():
            metric_records.append(
                tooling_artifact_format.build_metric_record(
                    run_id=run_id,
                    case_id=case_result.case_id,
                    trial_id=case_result.case_id,
                    phase=case_result.cache_state.value,
                    metric_name=metric_name,
                    value=metric_value,
                    unit=metric_unit(metric_name),
                    aggregation=tooling_artifact_format.MetricAggregation.EXACT.value,
                    higher_is_better=False if metric_name == "wall_time_seconds" else None,
                    dimensions=build_metric_dimensions(arguments, case_result),
                    source=tooling_artifact_format.MetricSource(
                        artifact_path="report.json",
                        json_pointer=f"/trials/{case_index}/{metric_name.replace('.', '/')}",
                    ),
                )
            )
    return metric_records


def build_failure_records(case_results: list[CaseResult]) -> list[tooling_artifact_format.FailureRecord]:
    """Build structured failure records."""
    failures: list[tooling_artifact_format.FailureRecord] = []
    for failure_index, case_result in enumerate(
        (result for result in case_results if result.status not in {RunStatus.DRY_RUN, RunStatus.SUCCESS}),
        start=1,
    ):
        failures.append(
            tooling_artifact_format.FailureRecord(
                failure_id=f"F{failure_index:03d}",
                phase=case_result.cache_state.value,
                status=status_to_artifact_status(case_result.status),
                message=f"Benchmark case {case_result.case_id} failed with status {case_result.status.value}.",
                exception_type=None,
                stderr_excerpt=None,
                stdout_log=f"logs/{case_result.case_id}/stdout.log",
                stderr_log=f"logs/{case_result.case_id}/stderr.log",
                command_id=case_result.case_id,
            )
        )
    return failures


def build_command_records(
    *,
    run_id: str,
    output_directory: Path,
    timed_results: list[TimedCommandResult],
) -> list[tooling_artifact_format.CommandRecord]:
    """Build command ledger records from timed command results."""
    return [
        tooling_commands.command_record_from_result(
            command_id=timed_result.command_id,
            tool_name=TOOL_NAME,
            run_id=run_id,
            phase=timed_result.phase,
            spec=timed_result.spec,
            result=timed_result.result,
            output_directory=output_directory,
            started_at=timed_result.started_at,
            finished_at=timed_result.finished_at,
            wall_time_seconds=timed_result.wall_time_seconds,
        )
        for timed_result in timed_results
    ]


def build_input_records(arguments: BenchmarkArguments) -> list[tooling_artifact_format.InputFileRecord]:
    """Build input-file records."""
    input_records: list[tooling_artifact_format.InputFileRecord] = []
    seen_paths: set[Path] = set()

    def append_input_record(path: Path | None, kind: str) -> None:
        if path is None or path in seen_paths:
            return
        seen_paths.add(path)
        input_records.append(tooling_artifact_format.build_input_file_record(path=path, kind=kind))

    append_input_record(arguments.bgen_path, "g_bgen")
    append_input_record(arguments.sample_path, "sample")
    append_input_record(arguments.bim_path, "subset_bim")
    append_input_record(arguments.torchgwas_full_genotype_path, "torchgwas_full_genotype")
    append_input_record(arguments.torchgwas_full_bim_path, "torchgwas_full_bim")
    append_input_record(arguments.torchgwas_full_fam_path, "torchgwas_full_fam")
    append_input_record(arguments.phenotype_path, "phenotype")
    append_input_record(arguments.covariate_path, "covariates")
    append_input_record(arguments.prediction_list_path, "regenie_prediction_list")
    return input_records


def build_events(
    *,
    arguments: BenchmarkArguments,
    run_id: str,
    case_results: list[CaseResult],
) -> list[tooling_artifact_format.ToolEventRecord]:
    """Build structured event records."""
    status = benchmark_artifact_status(arguments, case_results)
    return [
        tooling_artifact_format.build_tool_event(
            tool_name=TOOL_NAME,
            run_id=run_id,
            phase="benchmark",
            event="benchmark_completed",
            message=f"{arguments.chromosome_label} TorchGWAS benchmark completed.",
            fields={
                "status": status.value,
                "case_count": len(case_results),
                "dry_run": arguments.dry_run,
            },
        )
    ]


def build_agent_summary(case_results: list[CaseResult]) -> dict[str, object]:
    """Build compact agent-readable summary payload."""
    successful_cases = [case_result for case_result in case_results if case_result.status == RunStatus.SUCCESS]
    key_observations = [
        f"{case_result.case_id}: {case_result.status.value}"
        + (f", wall={case_result.wall_time_seconds:.3f}s" if case_result.wall_time_seconds is not None else "")
        for case_result in case_results
    ]
    return {
        "one_sentence": f"Completed {len(successful_cases)}/{len(case_results)} TorchGWAS chr22 benchmark cases.",
        "key_observations": key_observations,
        "risks": [
            "TorchGWAS v0.1 linear GWAS is not a strict statistical parity target for g REGENIE Step 2 LOCO output.",
            (
                "Full-size TorchGWAS runs use the local PLINK triplet because the pinned BGEN path stalls in "
                "raw-table parsing."
            ),
            (
                "TorchGWAS PLINK runs do not emit a persistent genotype cache; warm cases are repeated-process "
                "measurements with possible filesystem cache effects."
            ),
        ],
        "next_actions": [],
    }


def build_markdown_report(arguments: BenchmarkArguments, case_results: list[CaseResult]) -> str:
    """Render a Markdown summary."""
    lines = [
        f"# {arguments.chromosome_label} TorchGWAS Benchmark",
        "",
        f"- Output directory: `{arguments.output_directory}`",
        f"- TorchGWAS commit: `{arguments.torchgwas_commit}`",
        f"- Variant limit: `{arguments.variant_limit}`",
        f"- Chunk size: `{arguments.chunk_size}`",
        f"- `g` genotype input: `{arguments.bgen_path}` (`bgen`)",
        "- TorchGWAS full genotype input: "
        f"`{arguments.torchgwas_full_genotype_path}` (`{arguments.torchgwas_full_genotype_format.value}`)",
        "",
        "This is a workflow/runtime comparison, not strict statistical parity: `g` runs REGENIE Step 2 with LOCO",
        "predictions, while TorchGWAS runs covariate-adjusted quantitative linear GWAS.",
        "Full TorchGWAS runs use the local PLINK triplet because the pinned BGEN path stalls while parsing the",
        "intermediate PLINK2 raw table at chr22 scale.",
        "TorchGWAS PLINK runs do not emit a persistent genotype cache, so their warm cases are repeated-process",
        "measurements with possible filesystem cache effects rather than explicit genotype-cache reuse.",
        "",
        "## Cases",
        "",
        "| Case | Tool | Cache | Status | Wall seconds | Rows | Output bytes | Cache bytes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for case_result in case_results:
        lines.append(
            "| "
            f"{case_result.case_id} | "
            f"{case_result.tool.value} | "
            f"{case_result.cache_state.value} | "
            f"{case_result.status.value} | "
            f"{case_result.wall_time_seconds if case_result.wall_time_seconds is not None else ''} | "
            f"{case_result.output_row_count if case_result.output_row_count is not None else ''} | "
            f"{case_result.output_total_bytes if case_result.output_total_bytes is not None else ''} | "
            f"{case_result.cache_total_bytes if case_result.cache_total_bytes is not None else ''} |"
        )
    lines.extend(["", "## Commands", ""])
    for case_result in case_results:
        lines.extend(
            [
                "### `" + case_result.case_id + "`",
                "",
                "```bash",
                shlex.join(case_result.command_arguments),
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def write_reports(
    *,
    arguments: BenchmarkArguments,
    case_results: list[CaseResult],
    timed_results: list[TimedCommandResult],
    hydra_config: omegaconf.DictConfig | None,
) -> None:
    """Write standard benchmark artifacts."""
    producer = tooling_artifact_format.build_producer(tool_name=TOOL_NAME, repository_root=REPOSITORY_ROOT)
    status = benchmark_artifact_status(arguments, case_results)
    run = tooling_artifact_format.build_run_identity(
        tool_name=TOOL_NAME,
        output_directory=arguments.output_directory,
        status=status,
        status_reason=(
            "One or more benchmark cases failed."
            if status == tooling_artifact_format.ToolArtifactStatus.FAILED
            else None
        ),
    )
    context_snapshot = tooling_artifact_format.build_context_snapshot(
        output_directory=arguments.output_directory,
        repository_root=REPOSITORY_ROOT,
    )
    report = tooling_artifact_format.build_report_envelope(
        producer=producer,
        run=run,
        context=context_snapshot,
        title=f"{arguments.chromosome_label} TorchGWAS Benchmark",
        configuration=arguments_to_json_dict(arguments),
        summary={
            "headline": f"{arguments.chromosome_label} TorchGWAS benchmark finished with status {status.value}.",
            "agent_summary": build_agent_summary(case_results),
            "comparison_scope": "single_trait_quantitative_workflow_runtime",
            "input_boundary": {
                "g_genotype_format": "bgen",
                "g_genotype_path": str(arguments.bgen_path),
                "torchgwas_full_genotype_format": arguments.torchgwas_full_genotype_format.value,
                "torchgwas_full_genotype_path": str(arguments.torchgwas_full_genotype_path),
                "torchgwas_variant_limited_genotype_format": TorchgwasGenotypeFormat.NPY.value,
            },
        },
        cases=[
            {
                "case_id": case_result.case_id,
                "tool": case_result.tool.value,
                "cache_state": case_result.cache_state.value,
            }
            for case_result in case_results
        ],
        trials=[case_result_to_json_dict(case_result) for case_result in case_results],
        metrics=build_metrics(arguments=arguments, run_id=run.run_id, case_results=case_results),
        diagnostics={
            "torchgwas_source_directory": str(torchgwas_source_directory(arguments)),
            "torchgwas_venv_directory": str(torchgwas_venv_directory(arguments)),
        },
        failures=build_failure_records(case_results),
    )
    tooling_artifact_format.write_standard_artifact_bundle(
        output_directory=arguments.output_directory,
        report=report,
        events=build_events(arguments=arguments, run_id=run.run_id, case_results=case_results),
        commands=build_command_records(
            run_id=run.run_id,
            output_directory=arguments.output_directory,
            timed_results=timed_results,
        ),
        input_files=build_input_records(arguments),
        summary_markdown=build_markdown_report(arguments, case_results),
        hydra_config=hydra_config,
        tool_payload=arguments_to_json_dict(arguments),
        notes=[
            "TorchGWAS v0.1 is quantitative-only for the documented linear path.",
            "Benchmark compares workflow/runtime shape, not strict statistical parity with REGENIE Step 2 LOCO output.",
            (
                "Full-size TorchGWAS cases use PLINK input because the pinned TorchGWAS BGEN path stalls in "
                "raw-table parsing."
            ),
            (
                "TorchGWAS PLINK cases do not emit a persistent genotype cache; warm timings are repeated-process "
                "measurements with possible filesystem cache effects."
            ),
        ],
    )
    tooling_reports.write_json_report(
        arguments.output_directory / "legacy_summary.json",
        {
            "schema_version": 1,
            "tool": f"tooling.cli.{TOOL_NAME}",
            "configuration": arguments_to_json_dict(arguments),
            "cases": [case_result_to_json_dict(case_result) for case_result in case_results],
        },
        sort_keys=True,
    )


def run_setup(arguments: BenchmarkArguments) -> list[TimedCommandResult]:
    """Run setup commands for TorchGWAS unless this is a dry run."""
    if arguments.dry_run:
        return []
    timed_results: list[TimedCommandResult] = []
    for command_id, command_spec in torchgwas_setup_specs(arguments):
        timed_result = run_timed_command(command_id, "setup", command_spec)
        timed_results.append(timed_result)
        if timed_result.result.return_code != 0:
            message = f"Setup command {command_id} failed."
            raise RuntimeError(message)
    return timed_results


def run_benchmark(
    arguments: BenchmarkArguments,
    hydra_config: omegaconf.DictConfig | None = None,
) -> list[CaseResult]:
    """Run the benchmark and write reports."""
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    tooling_logging.configure_tool_logging(arguments.output_directory / "tooling.log")
    if arguments.validate_inputs and not arguments.dry_run:
        validate_input_paths(arguments)
    timed_results = run_setup(arguments)
    input_spec = prepare_variant_limited_torchgwas_inputs(arguments=arguments, command_results=timed_results)
    case_results: list[CaseResult] = []
    for case in build_cases(arguments, input_spec):
        case_result, timed_result = run_one_case(arguments, case)
        case_results.append(case_result)
        if timed_result is not None:
            timed_results.append(timed_result)
        if case_result.status not in {RunStatus.DRY_RUN, RunStatus.SUCCESS}:
            logger.error("Stopping benchmark after failed case %s.", case.case_id)
            break
    write_reports(
        arguments=arguments,
        case_results=case_results,
        timed_results=timed_results,
        hydra_config=hydra_config,
    )
    if any(case_result.status not in {RunStatus.DRY_RUN, RunStatus.SUCCESS} for case_result in case_results):
        message = f"One or more {arguments.chromosome_label} TorchGWAS benchmark cases failed."
        raise SystemExit(message)
    return case_results


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark_torchgwas_chr22")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Hydra entrypoint for the TorchGWAS chr22 benchmark."""
    run_benchmark(build_arguments_from_config(config), hydra_config=config)


def main() -> None:
    """Run the Hydra entrypoint."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
