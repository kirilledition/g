"""Typed TOML and command rendering for supported ``g regenie`` runs."""

from __future__ import annotations

import enum
import json
import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from pathlib import Path


class RegenieTraitKind(enum.StrEnum):
    """Supported REGENIE step 2 trait kinds."""

    QUANTITATIVE = "quantitative"
    BINARY = "binary"


class RegenieDevice(enum.StrEnum):
    """Supported execution devices."""

    CPU = "cpu"
    GPU = "gpu"


class RegenieBinaryFallback(enum.StrEnum):
    """Supported binary fallback methods."""

    SCORE_ONLY = "score_only"
    FIRTH_APPROXIMATE = "firth_approximate"


class RegenieMultiPhenotypeSampleMode(enum.StrEnum):
    """Supported multi-phenotype sample policies."""

    PER_PHENOTYPE = "per-phenotype"
    COMPLETE_CASE = "complete-case"


class RegenieTelemetry(enum.StrEnum):
    """Supported runtime telemetry modes."""

    OFF = "off"
    PROGRESS = "progress"
    PROFILE = "profile"


@dataclass(frozen=True)
class RegenieInputSpec:
    """Scientific inputs for one REGENIE step 2 run."""

    bgen_path: Path
    sample_path: Path
    phenotype_path: Path
    phenotype_columns: tuple[str, ...]
    covariate_path: Path | None
    covariate_columns: tuple[str, ...]
    prediction_list_path: Path
    output_prefix: Path


@dataclass(frozen=True)
class RegenieComputeOptions:
    """Runtime options exposed by the current TOML contract."""

    device: RegenieDevice
    bsize: int
    cpu_threads: int | None = None
    multi_phenotype_sample_mode: RegenieMultiPhenotypeSampleMode | None = None
    firth_batch_size: int | None = None
    firth_candidate_capacity: int | None = None
    jax_cache_dir: Path | None = None


@dataclass(frozen=True)
class RegenieOutputOptions:
    """Output options exposed by the current TOML contract."""

    output_run_directory: Path | None = None
    writer_threads: int | None = None
    resume: bool = False


@dataclass(frozen=True)
class RegenieDiagnosticsOptions:
    """Diagnostics options exposed by the current TOML contract."""

    telemetry: RegenieTelemetry = RegenieTelemetry.OFF


@dataclass(frozen=True)
class RegenieBinaryOptions:
    """Binary-trait options for one REGENIE step 2 run."""

    fallback_method: RegenieBinaryFallback
    p_threshold: float | None = None
    firth_se: bool | None = None


@dataclass(frozen=True)
class RegenieRunSpec:
    """Complete supported REGENIE run specification for tooling."""

    trait_kind: RegenieTraitKind
    command_prefix: tuple[str, ...]
    inputs: RegenieInputSpec
    compute: RegenieComputeOptions
    output: RegenieOutputOptions
    diagnostics: RegenieDiagnosticsOptions
    binary: RegenieBinaryOptions | None


def toml_string(value: str | Path) -> str:
    """Encode one TOML basic string."""
    return json.dumps(str(value))


def toml_string_array(values: tuple[str, ...]) -> str:
    """Encode one TOML string array."""
    return "[" + ", ".join(toml_string(value) for value in values) + "]"


def validate_regenie_run_spec(spec: RegenieRunSpec) -> None:
    """Validate invariants shared by TOML and command rendering."""
    if not spec.command_prefix:
        raise ValueError("REGENIE command_prefix must not be empty.")
    if not spec.inputs.phenotype_columns:
        raise ValueError("REGENIE run spec must include at least one phenotype column.")
    if any(not column.strip() for column in spec.inputs.phenotype_columns):
        raise ValueError("REGENIE phenotype columns must not contain empty values.")
    if any(not column.strip() for column in spec.inputs.covariate_columns):
        raise ValueError("REGENIE covariate columns must not contain empty values.")
    if spec.inputs.covariate_path is None and spec.inputs.covariate_columns:
        raise ValueError("REGENIE covariate columns require a covariate file.")
    if spec.compute.bsize <= 0:
        raise ValueError("REGENIE bsize must be positive.")
    if spec.compute.cpu_threads is not None and spec.compute.cpu_threads <= 0:
        raise ValueError("REGENIE cpu_threads must be positive when provided.")
    if spec.output.writer_threads is not None and spec.output.writer_threads <= 0:
        raise ValueError("REGENIE writer_threads must be positive when provided.")
    if spec.trait_kind == RegenieTraitKind.BINARY and spec.binary is None:
        raise ValueError("Binary REGENIE run spec requires binary options.")
    if spec.trait_kind == RegenieTraitKind.QUANTITATIVE and spec.binary is not None:
        raise ValueError("Quantitative REGENIE run spec must not include binary options.")
    if (
        spec.binary is not None
        and spec.binary.fallback_method == RegenieBinaryFallback.SCORE_ONLY
        and spec.binary.firth_se is True
    ):
        raise ValueError("Score-only binary runs cannot request Firth standard errors.")


def render_regenie_toml(spec: RegenieRunSpec) -> str:
    """Render a run spec using the current production TOML schema."""
    validate_regenie_run_spec(spec)
    lines = [
        "[input]",
        f"bgen = {toml_string(spec.inputs.bgen_path)}",
        f"sample = {toml_string(spec.inputs.sample_path)}",
        f"pheno_file = {toml_string(spec.inputs.phenotype_path)}",
        f"pheno_columns = {toml_string_array(spec.inputs.phenotype_columns)}",
    ]
    if spec.inputs.covariate_path is not None:
        lines.append(f"covar_file = {toml_string(spec.inputs.covariate_path)}")
    if spec.inputs.covariate_columns:
        lines.append(f"covar_columns = {toml_string_array(spec.inputs.covariate_columns)}")
    lines.extend(
        [
            f"pred = {toml_string(spec.inputs.prediction_list_path)}",
            "",
            "[trait]",
            f"trait_type = {toml_string(spec.trait_kind.value)}",
            f"bsize = {spec.compute.bsize}",
        ]
    )
    if spec.binary is not None:
        lines.extend(["", "[binary]", f"fallback_method = {toml_string(spec.binary.fallback_method.value)}"])
        if spec.binary.p_threshold is not None:
            lines.append(f"p_threshold = {spec.binary.p_threshold!r}")
        if spec.binary.firth_se is not None:
            lines.append(f"firth_se = {str(spec.binary.firth_se).lower()}")
    lines.extend(["", "[compute]", f"device = {toml_string(spec.compute.device.value)}"])
    if spec.compute.cpu_threads is not None:
        lines.append(f"cpu_threads = {spec.compute.cpu_threads}")
    if spec.compute.multi_phenotype_sample_mode is not None:
        lines.append(f"multi_phenotype_sample_mode = {toml_string(spec.compute.multi_phenotype_sample_mode.value)}")
    if spec.compute.firth_batch_size is not None:
        lines.append(f"firth_batch_size = {spec.compute.firth_batch_size}")
    if spec.compute.firth_candidate_capacity is not None:
        lines.append(f"firth_candidate_capacity = {spec.compute.firth_candidate_capacity}")
    if spec.compute.jax_cache_dir is not None:
        lines.append(f"jax_cache_dir = {toml_string(spec.compute.jax_cache_dir)}")
    lines.extend(["", "[output]", f"out = {toml_string(spec.inputs.output_prefix)}"])
    if spec.output.output_run_directory is not None:
        lines.append(f"output_run_directory = {toml_string(spec.output.output_run_directory)}")
    if spec.output.writer_threads is not None:
        lines.append(f"writer_threads = {spec.output.writer_threads}")
    lines.append(f"resume = {str(spec.output.resume).lower()}")
    lines.extend(["", "[diagnostics]", f"telemetry = {toml_string(spec.diagnostics.telemetry.value)}", ""])
    return "\n".join(lines)


def write_regenie_toml(spec: RegenieRunSpec, config_path: Path) -> Path:
    """Write a supported REGENIE TOML config and return its path."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(render_regenie_toml(spec), encoding="utf-8")
    return config_path


def render_g_regenie_command(spec: RegenieRunSpec, config_path: Path) -> list[str]:
    """Render the current shell-free CLI command for a written config."""
    validate_regenie_run_spec(spec)
    return [*spec.command_prefix, "--config", str(config_path)]


def render_native_cli_arguments(config_path: Path) -> list[str]:
    """Render arguments accepted by ``g._core.cli.run``."""
    return ["regenie", "--config", str(config_path)]
