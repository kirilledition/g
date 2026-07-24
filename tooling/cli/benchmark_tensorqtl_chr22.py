#!/usr/bin/env python3
"""Benchmark single-trait chr22 workflow runtime against tensorQTL."""

from __future__ import annotations

import csv
import dataclasses
import enum
import logging
import os
import shlex
import time
import typing
from pathlib import Path

import hydra
import pyarrow.parquet

import tooling.configuration as tooling_configuration
from tooling.benchmark import native_lifecycle
from tooling.common import artifact_format as tooling_artifact_format
from tooling.common import commands as tooling_commands
from tooling.common import g_regenie as tooling_g_regenie
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import logging as tooling_logging
from tooling.common import paths as tooling_paths
from tooling.common import reports as tooling_reports

if typing.TYPE_CHECKING:
    import omegaconf


logger = logging.getLogger(__name__)
REPOSITORY_ROOT = tooling_paths.find_repository_root(Path(__file__))
TOOL_NAME = "benchmark_tensorqtl_chr22"
TENSORQTL_REPOSITORY_URL = "https://github.com/broadinstitute/tensorqtl.git"
TENSORQTL_MAIN_COMMIT = "0c4db65a0cdc47f3b824ae530b89d270ef5e0096"
DEFAULT_OUTPUT_PARENT = Path("data/benchmarks")
DEFAULT_RUN_DIRECTORY_PREFIX = "tensorqtl_chr22"


class BenchmarkTool(enum.StrEnum):
    """Tool exercised by one benchmark case."""

    G = "g"
    TENSORQTL = "tensorqtl"


class CacheState(enum.StrEnum):
    """Repeated-run state exercised by one benchmark case."""

    COLD = "cold"
    WARM = "warm"
    FIRST_PROCESS = "first_process"
    REPEAT_PROCESS = "repeat_process"


class RunStatus(enum.StrEnum):
    """Benchmark case status."""

    DRY_RUN = "dry_run"
    SUCCESS = "success"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    MISSING_EXECUTABLE = "missing_executable"


@dataclasses.dataclass(frozen=True)
class BenchmarkArguments:
    """Resolved tensorQTL chr22 benchmark settings.

    Attributes:
        data_directory: Directory containing chr22 inputs.
        bgen_path: chr22 BGEN input used by g.
        sample_path: chr22 sample input used by g.
        plink_prefix_path: chr22 PLINK prefix used by tensorQTL.
        bed_path: chr22 BED input used by tensorQTL.
        bim_path: chr22 BIM input used by tensorQTL.
        fam_path: chr22 FAM input used by tensorQTL.
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
        chunk_size: Variant chunk size passed to g.
        cpu_threads: Optional thread count passed to g.
        tensorqtl_batch_size: GPU batch size passed to tensorQTL trans mode.
        tensorqtl_repository_url: tensorQTL Git repository URL.
        tensorqtl_commit: Pinned tensorQTL commit.
        tensorqtl_tools_directory: Repo-local tools directory for checkout and venv.
        tensorqtl_python: Python executable used to create the tensorQTL venv.
        torch_package: Torch package spec installed for tensorQTL.
        torch_package_index_url: Optional package index URL for the Torch package.
        command_timeout_seconds: Optional timeout for each executed command.
        g_runner_prefix: Command prefix used to invoke g.

    """

    data_directory: Path
    bgen_path: Path
    sample_path: Path
    plink_prefix_path: Path
    bed_path: Path
    bim_path: Path
    fam_path: Path
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
    chunk_size: int
    cpu_threads: int | None
    tensorqtl_batch_size: int
    tensorqtl_repository_url: str
    tensorqtl_commit: str
    tensorqtl_tools_directory: Path
    tensorqtl_python: str
    torch_package: str
    torch_package_index_url: str | None
    command_timeout_seconds: float | None
    g_runner_prefix: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class TensorqtlInputSpec:
    """tensorQTL input files for one benchmark run."""

    genotype_prefix_path: Path
    phenotype_matrix_path: Path
    covariate_matrix_path: Path


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
class OutputMeasurement:
    """Measured output properties for one case."""

    row_count: int | None
    total_bytes: int | None
    stage_seconds: dict[str, float]


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
    cache_before: native_lifecycle.CacheSnapshot | None
    cache_after: native_lifecycle.CacheSnapshot | None
    stage_seconds: dict[str, float]


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


def split_columns(raw_columns: str) -> tuple[str, ...]:
    """Split a comma-separated column list."""
    return tuple(column.strip() for column in raw_columns.split(",") if column.strip())


def optional_float(value: typing.Any) -> float | None:
    """Convert an optional config value to float."""
    if value is None:
        return None
    return float(value)


def plink_triplet_from_prefix(prefix_path: Path) -> tuple[Path, Path, Path]:
    """Return BED, BIM, and FAM paths for a PLINK prefix."""
    return prefix_path.with_suffix(".bed"), prefix_path.with_suffix(".bim"), prefix_path.with_suffix(".fam")


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
    plink_prefix_path = resolve_data_path(data_directory, tool_values["plink_prefix"])
    bed_path, bim_path, fam_path = plink_triplet_from_prefix(plink_prefix_path)
    runner_prefix = tuple(str(value) for value in typing.cast("list[typing.Any]", tool_values["g_runner_prefix"]))
    return BenchmarkArguments(
        data_directory=data_directory,
        bgen_path=resolve_data_path(data_directory, tool_values["bgen"]),
        sample_path=resolve_data_path(data_directory, tool_values["sample"]),
        plink_prefix_path=plink_prefix_path,
        bed_path=bed_path,
        bim_path=bim_path,
        fam_path=fam_path,
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
        chunk_size=int(tool_values["chunk_size"]),
        cpu_threads=tooling_hydra_arguments.integer_or_none(tool_values.get("cpu_threads")),
        tensorqtl_batch_size=int(tool_values["tensorqtl_batch_size"]),
        tensorqtl_repository_url=str(tool_values["tensorqtl_repository_url"]),
        tensorqtl_commit=str(tool_values["tensorqtl_commit"]),
        tensorqtl_tools_directory=resolve_repo_path(tool_values["tensorqtl_tools_dir"]),
        tensorqtl_python=str(tool_values["tensorqtl_python"]),
        torch_package=str(tool_values["torch_package"]),
        torch_package_index_url=(
            str(tool_values["torch_package_index_url"])
            if tool_values.get("torch_package_index_url") is not None
            else None
        ),
        command_timeout_seconds=optional_float(tool_values.get("command_timeout_seconds")),
        g_runner_prefix=runner_prefix,
    )


def build_arguments_from_overrides(overrides: typing.Sequence[str] | None = None) -> BenchmarkArguments:
    """Build benchmark arguments from Hydra overrides."""
    config = tooling_configuration.compose_config(config_name="benchmark_tensorqtl_chr22", overrides=overrides)
    return build_arguments_from_config(config)


def arguments_to_json_dict(arguments: BenchmarkArguments) -> dict[str, object]:
    """Convert benchmark arguments into a JSON-ready dictionary."""
    return {
        "chromosome_label": arguments.chromosome_label,
        "data_directory": str(arguments.data_directory),
        "bgen_path": str(arguments.bgen_path),
        "sample_path": str(arguments.sample_path),
        "plink_prefix_path": str(arguments.plink_prefix_path),
        "bed_path": str(arguments.bed_path),
        "bim_path": str(arguments.bim_path),
        "fam_path": str(arguments.fam_path),
        "phenotype_path": str(arguments.phenotype_path),
        "phenotype_column": arguments.phenotype_column,
        "covariate_path": str(arguments.covariate_path),
        "covariate_columns": list(arguments.covariate_columns),
        "prediction_list_path": str(arguments.prediction_list_path),
        "output_parent": str(arguments.output_parent),
        "output_directory": str(arguments.output_directory),
        "dry_run": arguments.dry_run,
        "chunk_size": arguments.chunk_size,
        "cpu_threads": arguments.cpu_threads,
        "tensorqtl_batch_size": arguments.tensorqtl_batch_size,
        "tensorqtl_repository_url": arguments.tensorqtl_repository_url,
        "tensorqtl_commit": arguments.tensorqtl_commit,
        "tensorqtl_tools_directory": str(arguments.tensorqtl_tools_directory),
        "tensorqtl_python": arguments.tensorqtl_python,
        "torch_package": arguments.torch_package,
        "torch_package_index_url": arguments.torch_package_index_url,
        "command_timeout_seconds": arguments.command_timeout_seconds,
        "g_runner_prefix": list(arguments.g_runner_prefix),
    }


def validate_input_paths(arguments: BenchmarkArguments) -> None:
    """Validate required benchmark inputs."""
    required_paths = [
        arguments.bgen_path,
        arguments.sample_path,
        arguments.bed_path,
        arguments.bim_path,
        arguments.fam_path,
        arguments.phenotype_path,
        arguments.covariate_path,
        arguments.prediction_list_path,
    ]
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        formatted_paths = "\n".join(f"- {path}" for path in missing_paths)
        message = f"Required {arguments.chromosome_label} tensorQTL benchmark inputs are missing:\n{formatted_paths}"
        raise FileNotFoundError(message)


def tensorqtl_source_directory(arguments: BenchmarkArguments) -> Path:
    """Return the pinned tensorQTL source checkout path."""
    return arguments.tensorqtl_tools_directory / arguments.tensorqtl_commit / "source"


def tensorqtl_python_environment_slug(python_executable: str) -> str:
    """Return a filesystem-safe Python interpreter slug."""
    executable_name = Path(python_executable).name or "python"
    slug = "".join(character if character.isalnum() else "_" for character in executable_name).strip("_")
    return slug or "python"


def tensorqtl_venv_directory(arguments: BenchmarkArguments) -> Path:
    """Return the pinned tensorQTL virtual environment path."""
    return (
        arguments.tensorqtl_tools_directory
        / arguments.tensorqtl_commit
        / f"venv-plink-{tensorqtl_python_environment_slug(arguments.tensorqtl_python)}"
    )


def tensorqtl_python_executable(arguments: BenchmarkArguments) -> Path:
    """Return the Python executable in the tensorQTL venv."""
    return tensorqtl_venv_directory(arguments) / "bin" / "python"


def tensorqtl_setup_specs(arguments: BenchmarkArguments) -> list[tuple[str, tooling_commands.CommandSpec]]:
    """Build setup commands for the pinned tensorQTL checkout and venv."""
    source_directory = tensorqtl_source_directory(arguments)
    venv_directory = tensorqtl_venv_directory(arguments)
    setup_log_directory = arguments.output_directory / "logs" / "setup"
    clone_script = (
        "set -euo pipefail\n"
        f"mkdir -p {shlex.quote(str(source_directory.parent))}\n"
        f"if [ ! -d {shlex.quote(str(source_directory / '.git'))} ]; then\n"
        f"  git clone {shlex.quote(arguments.tensorqtl_repository_url)} {shlex.quote(str(source_directory))}\n"
        "fi\n"
        f"if ! git -C {shlex.quote(str(source_directory))} cat-file -e "
        f"{shlex.quote(arguments.tensorqtl_commit + '^{commit}')} 2>/dev/null; then\n"
        f"  git -C {shlex.quote(str(source_directory))} fetch --quiet origin\n"
        "fi\n"
        f"git -C {shlex.quote(str(source_directory))} checkout --quiet {shlex.quote(arguments.tensorqtl_commit)}\n"
    )
    install_script = (
        "set -euo pipefail\n"
        f"if [ ! -x {shlex.quote(str(venv_directory / 'bin' / 'python'))} ] "
        f"|| [ ! -x {shlex.quote(str(venv_directory / 'bin' / 'pip'))} ]; then\n"
        "  uv venv --no-project --seed --clear "
        f"--python {shlex.quote(arguments.tensorqtl_python)} {shlex.quote(str(venv_directory))}\n"
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
        f"-e {shlex.quote(str(source_directory))} "
        f"{shlex.quote('pandas<3')} "
        f"{shlex.quote('pandas-plink==2.2.9')} "
        "pyarrow\n"
    )
    cuda_probe_script = (
        "import json\n"
        "import torch\n"
        "import tensorqtl\n"
        "import pandas_plink\n"
        "cuda_available = torch.cuda.is_available()\n"
        "payload = {\n"
        "    'tensorqtl_version': tensorqtl.__version__,\n"
        "    'torch_version': torch.__version__,\n"
        "    'torch_cuda_version': torch.version.cuda,\n"
        "    'cuda_available': cuda_available,\n"
        "    'cuda_device_count': torch.cuda.device_count() if cuda_available else 0,\n"
        "    'cuda_device_name': torch.cuda.get_device_name(0) if cuda_available else None,\n"
        "    'pandas_plink_version': getattr(pandas_plink, '__version__', None),\n"
        "}\n"
        "print(json.dumps(payload, sort_keys=True))\n"
        "if not cuda_available:\n"
        "    raise SystemExit('Torch CUDA is unavailable for the tensorQTL benchmark.')\n"
    )
    return [
        (
            "tensorqtl_checkout",
            tooling_commands.build_command_spec(
                ["bash", "-lc", clone_script],
                cwd=REPOSITORY_ROOT,
                timeout_seconds=arguments.command_timeout_seconds,
                stdout_path=setup_log_directory / "tensorqtl_checkout.stdout.log",
                stderr_path=setup_log_directory / "tensorqtl_checkout.stderr.log",
                stream=True,
            ),
        ),
        (
            "tensorqtl_install",
            tooling_commands.build_command_spec(
                ["bash", "-lc", install_script],
                cwd=REPOSITORY_ROOT,
                timeout_seconds=arguments.command_timeout_seconds,
                stdout_path=setup_log_directory / "tensorqtl_install.stdout.log",
                stderr_path=setup_log_directory / "tensorqtl_install.stderr.log",
                stream=True,
            ),
        ),
        (
            "tensorqtl_cuda_probe",
            tooling_commands.build_command_spec(
                [str(venv_directory / "bin" / "python"), "-c", cuda_probe_script],
                cwd=REPOSITORY_ROOT,
                timeout_seconds=arguments.command_timeout_seconds,
                stdout_path=setup_log_directory / "tensorqtl_cuda_probe.stdout.log",
                stderr_path=setup_log_directory / "tensorqtl_cuda_probe.stderr.log",
                stream=True,
            ),
        ),
    ]


def read_fam_sample_identifiers(fam_path: Path) -> list[str]:
    """Read sample identifiers in tensorQTL PLINK genotype order from a FAM file."""
    sample_identifiers: list[str] = []
    with fam_path.open("r", encoding="utf-8") as fam_file:
        for line_number, line in enumerate(fam_file, start=1):
            fields = line.split()
            if len(fields) < 2:
                message = f"FAM file {fam_path} row {line_number} does not contain FID and IID columns."
                raise ValueError(message)
            sample_identifier = fields[1]
            sample_identifiers.append(sample_identifier)
    if not sample_identifiers:
        message = f"FAM file {fam_path} does not contain samples."
        raise ValueError(message)
    return sample_identifiers


def read_table_by_iid(path: Path) -> dict[str, dict[str, str]]:
    """Read a tabular sample-indexed file keyed by IID."""
    rows_by_iid: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as table_file:
        reader = csv.DictReader(table_file, delimiter="\t")
        for row in reader:
            sample_identifier = row.get("IID")
            if sample_identifier is None:
                message = f"Input table {path} does not contain an IID column."
                raise ValueError(message)
            rows_by_iid[sample_identifier] = row
    return rows_by_iid


def write_matrix_rows(
    path: Path, identifier_column: str, sample_identifiers: list[str], rows: dict[str, list[str]]
) -> None:
    """Write a tensorQTL row-oriented matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as matrix_file:
        writer = csv.writer(matrix_file, delimiter="\t", lineterminator="\n")
        writer.writerow([identifier_column, *sample_identifiers])
        for row_identifier, row_values in rows.items():
            writer.writerow([row_identifier, *row_values])


def prepare_tensorqtl_matrices(
    *,
    arguments: BenchmarkArguments,
) -> TensorqtlInputSpec:
    """Prepare tensorQTL phenotype and covariate matrices."""
    prepared_directory = arguments.output_directory / "tensorqtl_inputs"
    phenotype_matrix_path = prepared_directory / "phenotypes.tsv"
    covariate_matrix_path = prepared_directory / "covariates.tsv"
    if arguments.dry_run:
        return TensorqtlInputSpec(
            genotype_prefix_path=prepare_full_plink_alias_prefix(arguments),
            phenotype_matrix_path=phenotype_matrix_path,
            covariate_matrix_path=covariate_matrix_path,
        )
    sample_identifiers = read_fam_sample_identifiers(arguments.fam_path)
    phenotype_rows = read_table_by_iid(arguments.phenotype_path)
    covariate_rows = read_table_by_iid(arguments.covariate_path)
    phenotype_values = [
        phenotype_rows[sample_identifier][arguments.phenotype_column] for sample_identifier in sample_identifiers
    ]
    covariate_matrix_rows = {
        covariate_column: [
            covariate_rows[sample_identifier][covariate_column] for sample_identifier in sample_identifiers
        ]
        for covariate_column in arguments.covariate_columns
    }
    write_matrix_rows(
        phenotype_matrix_path,
        "phenotype_id",
        sample_identifiers,
        {arguments.phenotype_column: phenotype_values},
    )
    write_matrix_rows(covariate_matrix_path, "covariate_id", sample_identifiers, covariate_matrix_rows)
    return TensorqtlInputSpec(
        genotype_prefix_path=prepare_full_plink_alias_prefix(arguments),
        phenotype_matrix_path=phenotype_matrix_path,
        covariate_matrix_path=covariate_matrix_path,
    )


def link_plink_file(source_path: Path, alias_path: Path) -> None:
    """Create one PLINK alias symlink unless an equivalent path already exists."""
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    if alias_path.exists() or alias_path.is_symlink():
        if alias_path.resolve() == source_path.resolve():
            return
        message = f"Cannot create PLINK alias {alias_path}; path already points to a different file."
        raise FileExistsError(message)
    alias_path.symlink_to(source_path)


def prepare_full_plink_alias_prefix(arguments: BenchmarkArguments) -> Path:
    """Prepare a BED-only alias prefix so tensorQTL does not auto-select PGEN."""
    alias_prefix = arguments.output_directory / "tensorqtl_inputs" / "plink_genotypes"
    if arguments.dry_run:
        return alias_prefix
    alias_bed_path, alias_bim_path, alias_fam_path = plink_triplet_from_prefix(alias_prefix)
    link_plink_file(arguments.bed_path, alias_bed_path)
    link_plink_file(arguments.bim_path, alias_bim_path)
    link_plink_file(arguments.fam_path, alias_fam_path)
    return alias_prefix


def build_environment_overrides() -> dict[str, str]:
    """Build child-process environment overrides."""
    python_path_entries = [str(REPOSITORY_ROOT)]
    existing_python_path = os.environ.get("PYTHONPATH")
    if existing_python_path:
        python_path_entries.append(existing_python_path)
    return {
        "PYTHONPATH": os.pathsep.join(python_path_entries),
    }


def append_option(command_arguments: list[str], option_name: str, value: object) -> None:
    """Append one CLI option and value."""
    command_arguments.extend([option_name, str(value)])


def build_g_command(
    arguments: BenchmarkArguments,
    case_id: str,
    cache_directory: Path,
    output_prefix: Path,
) -> list[str]:
    """Build the g quantitative chr22 command."""
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
            cpu_threads=arguments.cpu_threads,
            jax_cache_dir=cache_directory,
        ),
        output=tooling_g_regenie.RegenieOutputOptions(
            output_run_directory=None,
            writer_threads=8,
            resume=False,
        ),
        diagnostics=tooling_g_regenie.RegenieDiagnosticsOptions(telemetry=tooling_g_regenie.RegenieTelemetry.OFF),
        binary=None,
    )
    config_path = arguments.output_directory / "configs" / f"{case_id}.toml"
    tooling_g_regenie.write_regenie_toml(run_spec, config_path)
    return tooling_g_regenie.render_g_regenie_command(run_spec, config_path)


def build_tensorqtl_command(
    *,
    arguments: BenchmarkArguments,
    output_directory: Path,
    input_spec: TensorqtlInputSpec,
) -> list[str]:
    """Build the tensorQTL trans dense command."""
    prefix = "tensorqtl"
    command_arguments = [
        str(tensorqtl_python_executable(arguments)),
        "-m",
        "tensorqtl",
        str(input_spec.genotype_prefix_path),
        str(input_spec.phenotype_matrix_path),
        prefix,
    ]
    append_option(command_arguments, "--covariates", input_spec.covariate_matrix_path)
    append_option(command_arguments, "--mode", "trans")
    command_arguments.append("--return_dense")
    append_option(command_arguments, "--batch_size", arguments.tensorqtl_batch_size)
    append_option(command_arguments, "--maf_threshold", 0)
    append_option(command_arguments, "--output_dir", output_directory)
    return command_arguments


def build_cases(arguments: BenchmarkArguments, input_spec: TensorqtlInputSpec) -> list[BenchmarkCase]:
    """Build cache-qualified g cases and repeated-process tensorQTL cases."""
    cases: list[BenchmarkCase] = []
    environment_overrides = build_environment_overrides()
    g_cache_directory = arguments.output_directory / "caches" / "g"
    tensorqtl_cache_directory = arguments.output_directory / "caches" / "tensorqtl"
    paired_states = (
        (CacheState.COLD, CacheState.FIRST_PROCESS),
        (CacheState.WARM, CacheState.REPEAT_PROCESS),
    )
    for g_cache_state, tensorqtl_process_state in paired_states:
        g_case_id = f"g_linear_gpu_{g_cache_state.value}"
        g_output_directory = arguments.output_directory / "runs" / g_case_id
        g_output_prefix = g_output_directory / "linear"
        g_log_directory = arguments.output_directory / "logs" / g_case_id
        cases.append(
            BenchmarkCase(
                case_id=g_case_id,
                tool=BenchmarkTool.G,
                cache_state=g_cache_state,
                command_arguments=tuple(build_g_command(arguments, g_case_id, g_cache_directory, g_output_prefix)),
                output_directory=g_output_directory,
                cache_directory=g_cache_directory,
                stdout_path=g_log_directory / "stdout.log",
                stderr_path=g_log_directory / "stderr.log",
                stage_timing_path=None,
                profile_summary_path=None,
                environment_overrides=environment_overrides,
            )
        )
        tensorqtl_case_id = f"tensorqtl_trans_dense_gpu_{tensorqtl_process_state.value}"
        tensorqtl_output_directory = arguments.output_directory / "runs" / tensorqtl_case_id
        tensorqtl_log_directory = arguments.output_directory / "logs" / tensorqtl_case_id
        cases.append(
            BenchmarkCase(
                case_id=tensorqtl_case_id,
                tool=BenchmarkTool.TENSORQTL,
                cache_state=tensorqtl_process_state,
                command_arguments=tuple(
                    build_tensorqtl_command(
                        arguments=arguments,
                        output_directory=tensorqtl_output_directory,
                        input_spec=input_spec,
                    )
                ),
                output_directory=tensorqtl_output_directory,
                cache_directory=tensorqtl_cache_directory,
                stdout_path=tensorqtl_log_directory / "stdout.log",
                stderr_path=tensorqtl_log_directory / "stderr.log",
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


def measure_g_outputs(case: BenchmarkCase, stdout_chunks: typing.Sequence[str]) -> OutputMeasurement:
    """Measure g output rows and bytes."""
    output_root = Path(f"{case.output_directory / 'linear'}.g")
    verified_outputs = native_lifecycle.collect_completed_output_evidence(
        stdout_chunks,
        output_root=output_root,
        expected_phenotype_count=1,
        run_label=case.case_id,
    )
    output_measurement = verified_outputs.runs[0]
    output_run_directory = Path(output_measurement.run_directory)
    return OutputMeasurement(
        row_count=output_measurement.row_count,
        total_bytes=directory_size_bytes(output_run_directory),
        stage_seconds=load_stage_seconds(case.stage_timing_path),
    )


def parquet_row_count(path: Path) -> int:
    """Read the positive row count from required Parquet metadata."""
    if not path.is_file():
        raise RuntimeError(f"Completed tensorQTL run has no p-value output: {path}")
    row_count = pyarrow.parquet.ParquetFile(path).metadata.num_rows
    if row_count == 0:
        raise RuntimeError(f"Completed tensorQTL run has an empty p-value output: {path}")
    return row_count


def measure_tensorqtl_outputs(case: BenchmarkCase) -> OutputMeasurement:
    """Measure tensorQTL dense trans output rows and bytes."""
    p_value_path = case.output_directory / "tensorqtl.trans_qtl_pval.parquet"
    return OutputMeasurement(
        row_count=parquet_row_count(p_value_path),
        total_bytes=directory_size_bytes(case.output_directory),
        stage_seconds={},
    )


def measure_case_outputs(case: BenchmarkCase, command_stdout: str) -> OutputMeasurement:
    """Measure case output rows, bytes, and stage timings."""
    if case.tool == BenchmarkTool.G:
        return measure_g_outputs(case, (command_stdout,))
    return measure_tensorqtl_outputs(case)


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
                cache_before=None,
                cache_after=None,
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
    case.output_directory.mkdir(parents=True, exist_ok=True)
    uses_persistent_cache = case.cache_state in {CacheState.COLD, CacheState.WARM}
    cache_before = native_lifecycle.snapshot_tree(case.cache_directory) if uses_persistent_cache else None
    if case.cache_state == CacheState.COLD and cache_before is not None and cache_before.file_count != 0:
        raise RuntimeError(f"Cold benchmark cache is not empty for {case.case_id}: {case.cache_directory}")
    if case.cache_state == CacheState.WARM and (cache_before is None or cache_before.file_count == 0):
        raise RuntimeError(f"Warm benchmark cache is empty for {case.case_id}: {case.cache_directory}")
    if uses_persistent_cache:
        case.cache_directory.mkdir(parents=True, exist_ok=True)
    case.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    timed_result = run_timed_command(case.case_id, "benchmark", command_spec)
    status = case_status_from_command_result(timed_result.result)
    measurement = OutputMeasurement(row_count=None, total_bytes=None, stage_seconds={})
    if status == RunStatus.SUCCESS:
        measurement = measure_case_outputs(case, timed_result.result.stdout)
    cache_after = native_lifecycle.snapshot_tree(case.cache_directory) if uses_persistent_cache else None
    if (
        status == RunStatus.SUCCESS
        and case.cache_state == CacheState.COLD
        and cache_after is not None
        and cache_after.file_count == 0
    ):
        raise RuntimeError(f"Cold benchmark did not populate its cache for {case.case_id}")
    if status == RunStatus.SUCCESS and case.cache_state == CacheState.WARM and cache_before != cache_after:
        raise RuntimeError(f"Warm benchmark changed its cache tree for {case.case_id}")
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
        output_row_count=measurement.row_count,
        output_total_bytes=measurement.total_bytes,
        cache_total_bytes=directory_size_bytes(case.cache_directory) if uses_persistent_cache else None,
        cache_before=cache_before,
        cache_after=cache_after,
        stage_seconds=measurement.stage_seconds,
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
        "cache_before": dataclasses.asdict(case_result.cache_before) if case_result.cache_before is not None else None,
        "cache_after": dataclasses.asdict(case_result.cache_after) if case_result.cache_after is not None else None,
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
    return {
        "chromosome_label": arguments.chromosome_label,
        "trait_type": "quantitative",
        "phenotype_column": arguments.phenotype_column,
        "tool": case_result.tool.value,
        "genotype_format": "bgen" if case_result.tool == BenchmarkTool.G else "plink",
        "cache_state": case_result.cache_state.value,
        "device": "gpu",
        "chunk_size": arguments.chunk_size,
        "tensorqtl_commit": arguments.tensorqtl_commit if case_result.tool == BenchmarkTool.TENSORQTL else None,
        "comparison_scope": "nominal_linear_workflow_runtime_not_statistical_parity",
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
    return [
        tooling_artifact_format.build_input_file_record(path=arguments.bgen_path, kind="g_bgen"),
        tooling_artifact_format.build_input_file_record(path=arguments.sample_path, kind="sample"),
        tooling_artifact_format.build_input_file_record(path=arguments.bed_path, kind="tensorqtl_bed"),
        tooling_artifact_format.build_input_file_record(path=arguments.bim_path, kind="tensorqtl_bim"),
        tooling_artifact_format.build_input_file_record(path=arguments.fam_path, kind="tensorqtl_fam"),
        tooling_artifact_format.build_input_file_record(path=arguments.phenotype_path, kind="phenotype"),
        tooling_artifact_format.build_input_file_record(path=arguments.covariate_path, kind="covariates"),
        tooling_artifact_format.build_input_file_record(
            path=arguments.prediction_list_path,
            kind="regenie_prediction_list",
        ),
    ]


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
            message=f"{arguments.chromosome_label} tensorQTL benchmark completed.",
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
        "one_sentence": f"Completed {len(successful_cases)}/{len(case_results)} tensorQTL chr22 benchmark cases.",
        "key_observations": key_observations,
        "risks": [
            "tensorQTL trans dense is nominal QTL-style linear association, not REGENIE Step 2 LOCO output.",
            "tensorQTL first/repeat cases are process-lifecycle measurements with possible filesystem cache effects; "
            "they do not claim a persistent application-cache state.",
        ],
        "next_actions": [],
    }


def build_markdown_report(arguments: BenchmarkArguments, case_results: list[CaseResult]) -> str:
    """Render a Markdown summary."""
    lines = [
        f"# {arguments.chromosome_label} tensorQTL Benchmark",
        "",
        f"- Output directory: `{arguments.output_directory}`",
        f"- tensorQTL commit: `{arguments.tensorqtl_commit}`",
        f"- `g` genotype input: `{arguments.bgen_path}` (`bgen`)",
        f"- tensorQTL genotype input: `{arguments.plink_prefix_path}` (`bed/bim/fam`)",
        "",
        "This is a workflow/runtime comparison, not strict statistical parity: `g` runs REGENIE Step 2 with LOCO",
        "predictions, while tensorQTL runs dense trans nominal linear association on generated QTL-style matrices.",
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
        title=f"{arguments.chromosome_label} tensorQTL Benchmark",
        configuration=arguments_to_json_dict(arguments),
        summary={
            "headline": f"{arguments.chromosome_label} tensorQTL benchmark finished with status {status.value}.",
            "agent_summary": build_agent_summary(case_results),
            "comparison_scope": "single_trait_quantitative_nominal_linear_workflow_runtime",
            "input_boundary": {
                "g_genotype_format": "bgen",
                "g_genotype_path": str(arguments.bgen_path),
                "tensorqtl_genotype_format": "plink",
                "tensorqtl_genotype_prefix_path": str(arguments.plink_prefix_path),
                "tensorqtl_mode": "trans_dense",
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
            "tensorqtl_source_directory": str(tensorqtl_source_directory(arguments)),
            "tensorqtl_venv_directory": str(tensorqtl_venv_directory(arguments)),
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
            "tensorQTL trans dense is nominal QTL-style linear association, not REGENIE Step 2 LOCO output.",
            "Benchmark compares workflow/runtime shape, not strict statistical parity.",
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
    """Run setup commands for tensorQTL unless this is a dry run."""
    if arguments.dry_run:
        return []
    timed_results: list[TimedCommandResult] = []
    for command_id, command_spec in tensorqtl_setup_specs(arguments):
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
    input_spec = prepare_tensorqtl_matrices(arguments=arguments)
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
        message = f"One or more {arguments.chromosome_label} tensorQTL benchmark cases failed."
        raise SystemExit(message)
    return case_results


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark_tensorqtl_chr22")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Hydra entrypoint for the tensorQTL chr22 benchmark."""
    run_benchmark(build_arguments_from_config(config), hydra_config=config)


def main() -> None:
    """Run the Hydra entrypoint."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
