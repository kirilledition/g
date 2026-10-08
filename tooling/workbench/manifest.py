"""Strict, versioned job manifests independent of the native and JAX runtimes."""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import re
import struct
import typing
import urllib.parse
from pathlib import Path

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,79}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class Device(enum.StrEnum):
    """Execution device requested explicitly by a manifest."""

    CPU = "cpu"
    GPU = "gpu"


class TraitType(enum.StrEnum):
    """Supported association models."""

    QUANTITATIVE = "quantitative"
    BINARY = "binary"


class FallbackMethod(enum.StrEnum):
    """Supported binary correction policies."""

    SCORE_ONLY = "score_only"
    FIRTH_APPROXIMATE = "firth_approximate"


@dataclasses.dataclass(frozen=True)
class InputFile:
    """A byte-exact input and optional immutable cloud generation."""

    uri: str
    sha256: str
    size_bytes: int
    generation: str | None


@dataclasses.dataclass(frozen=True)
class PredictionFile:
    """One named phenotype's genome-wide step 1 LOCO file."""

    phenotype: str
    file: InputFile


@dataclasses.dataclass(frozen=True)
class Dataset:
    """Explicit source release identity carried into all artifacts."""

    name: str
    release: str
    reference_build: str


@dataclasses.dataclass(frozen=True)
class Analysis:
    """Scientific configuration with explicit selected columns."""

    trait_type: TraitType
    phenotype_columns: tuple[str, ...]
    covariate_columns: tuple[str, ...]
    fallback_method: FallbackMethod | None
    p_threshold: float | None


@dataclasses.dataclass(frozen=True)
class Resources:
    """Declared limits used for a conservative chunk working-set estimate."""

    device: Device
    cpu_threads: int
    writer_threads: int
    memory_gib: float | None
    gpu_memory_gib: float | None
    memory_fraction: float
    max_chunk_variants: int
    firth_batch_size: int | None


@dataclasses.dataclass(frozen=True)
class JobManifest:
    """Validated version-one job description."""

    dataset: Dataset
    inputs: dict[str, InputFile]
    predictions: tuple[PredictionFile, ...]
    analysis: Analysis
    resources: Resources
    output_uri: str


def mapping(value: JsonValue, context: str) -> dict[str, JsonValue]:
    """Require a JSON object without accepting implicit coercions."""
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object.")
    return value


def fields(value: dict[str, JsonValue], required: set[str], optional: set[str], context: str) -> None:
    """Reject missing fields and unknown fields, including misspelled limits."""
    if missing := required - value.keys():
        raise ValueError(f"{context} is missing fields: {', '.join(sorted(missing))}.")
    if unknown := value.keys() - required - optional:
        raise ValueError(f"{context} has unknown fields: {', '.join(sorted(unknown))}.")


def text(value: JsonValue, context: str) -> str:
    """Require nonempty, printable text."""
    if not isinstance(value, str) or not value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError(f"{context} must be nonempty text without control characters.")
    return value


def safe_name(value: JsonValue, context: str) -> str:
    """Require a stable identifier safe for tables, logs and file names."""
    result = text(value, context)
    if not NAME_PATTERN.fullmatch(result) or result in {"FID", "IID"}:
        raise ValueError(f"{context} must begin with a letter and contain at most 80 letters, digits, '.', '_' or '-'.")
    return result


def positive_integer(value: JsonValue, context: str) -> int:
    """Require a positive JSON integer rather than bool or floating point."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{context} must be a positive integer.")
    return value


def positive_float(value: JsonValue, context: str) -> float:
    """Require a finite positive numeric value."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{context} must be a finite positive number.")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{context} exceeds the supported numeric range.") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{context} must be a finite positive number.")
    return result


def names(value: JsonValue, context: str, *, allow_empty: bool) -> tuple[str, ...]:
    """Validate an explicit ordered column selection."""
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{context} must be {'a nonempty' if not allow_empty else 'an'} array.")
    result = tuple(safe_name(item, context) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{context} contains duplicate names.")
    return result


def validate_uri(value: JsonValue, context: str) -> str:
    """Accept absolute local paths and non-wildcard GCS object paths only."""
    result = text(value, context)
    if result.startswith("gs://"):
        parsed = urllib.parse.urlsplit(result)
        if (
            parsed.scheme != "gs"
            or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{1,220}[a-z0-9]", parsed.netloc)
            or parsed.query
            or parsed.fragment
            or not parsed.path.strip("/")
            or any(character in result for character in "*?[]%#\\")
            or any(part in {".", "..", ""} for part in parsed.path[1:].split("/"))
        ):
            raise ValueError(
                f"{context} must be a literal gs://bucket/object without query, generation suffix or traversal."
            )
    elif not Path(result).is_absolute() or any(part == ".." for part in Path(result).parts):
        raise ValueError(f"{context} must be an absolute local path or literal gs://bucket/object.")
    return result


def input_file(value: JsonValue, context: str) -> InputFile:
    """Parse a pinned input specification."""
    values = mapping(value, context)
    fields(values, {"uri", "sha256", "size_bytes"}, {"generation"}, context)
    uri = validate_uri(values["uri"], f"{context}.uri")
    sha256 = text(values["sha256"], f"{context}.sha256")
    if not SHA256_PATTERN.fullmatch(sha256):
        raise ValueError(f"{context}.sha256 must contain 64 lower-case hexadecimal characters.")
    generation = None
    if "generation" in values:
        generation = text(values["generation"], f"{context}.generation")
        if (
            not uri.startswith("gs://")
            or not generation.isascii()
            or not generation.isdecimal()
            or int(generation) <= 0
        ):
            raise ValueError(f"{context}.generation must be a positive decimal string for a GCS input.")
    return InputFile(uri, sha256, positive_integer(values["size_bytes"], f"{context}.size_bytes"), generation)


def parse_analysis(value: JsonValue) -> Analysis:
    """Validate the selected trait family and correction configuration."""
    values = mapping(value, "analysis")
    fields(values, {"trait_type", "phenotype_columns", "covariate_columns"}, {"binary"}, "analysis")
    trait_type = TraitType(text(values["trait_type"], "analysis.trait_type"))
    fallback_method = None
    p_threshold = None
    if trait_type == TraitType.BINARY:
        binary = mapping(values.get("binary"), "analysis.binary")
        fields(binary, {"fallback_method", "p_threshold"}, set(), "analysis.binary")
        fallback_method = FallbackMethod(text(binary["fallback_method"], "analysis.binary.fallback_method"))
        p_threshold = positive_float(binary["p_threshold"], "analysis.binary.p_threshold")
        if p_threshold >= 1:
            raise ValueError("analysis.binary.p_threshold must lie strictly between 0 and 1.")
        p_threshold = float(struct.unpack("<f", struct.pack("<f", p_threshold))[0])
        if not 0 < p_threshold < 1:
            raise ValueError("analysis.binary.p_threshold must remain strictly between 0 and 1 after float32 rounding.")
    elif "binary" in values:
        raise ValueError("Quantitative analyses must not contain binary correction settings.")
    return Analysis(
        trait_type,
        names(values["phenotype_columns"], "analysis.phenotype_columns", allow_empty=False),
        names(values["covariate_columns"], "analysis.covariate_columns", allow_empty=True),
        fallback_method,
        p_threshold,
    )


def parse_resources(value: JsonValue) -> Resources:
    """Validate declared memory and parallelism without probing a GPU."""
    values = mapping(value, "resources")
    fields(
        values,
        {"device", "cpu_threads", "writer_threads", "memory_fraction", "max_chunk_variants"},
        {"memory_gib", "gpu_memory_gib", "firth_batch_size"},
        "resources",
    )
    device = Device(text(values["device"], "resources.device"))
    memory_gib = positive_float(values["memory_gib"], "resources.memory_gib") if "memory_gib" in values else None
    gpu_memory_gib = (
        positive_float(values["gpu_memory_gib"], "resources.gpu_memory_gib") if "gpu_memory_gib" in values else None
    )
    if (device == Device.CPU and memory_gib is None) or (device == Device.GPU and gpu_memory_gib is None):
        raise ValueError("Declare memory_gib for CPU jobs and gpu_memory_gib for GPU jobs.")
    memory_fraction = positive_float(values["memory_fraction"], "resources.memory_fraction")
    if memory_fraction > 0.8:
        raise ValueError("resources.memory_fraction must be at most 0.8 to reserve unmodelled working memory.")
    return Resources(
        device,
        positive_integer(values["cpu_threads"], "resources.cpu_threads"),
        positive_integer(values["writer_threads"], "resources.writer_threads"),
        memory_gib,
        gpu_memory_gib,
        memory_fraction,
        positive_integer(values["max_chunk_variants"], "resources.max_chunk_variants"),
        positive_integer(values["firth_batch_size"], "resources.firth_batch_size")
        if "firth_batch_size" in values
        else None,
    )


def parse_manifest(value: JsonValue) -> JobManifest:
    """Validate a version-one job, refusing unknown or ambiguous settings."""
    values = mapping(value, "manifest")
    fields(
        values,
        {"schema_version", "dataset", "inputs", "predictions", "analysis", "resources", "output_uri"},
        set(),
        "manifest",
    )
    if (
        not isinstance(values["schema_version"], int)
        or isinstance(values["schema_version"], bool)
        or values["schema_version"] != 1
    ):
        raise ValueError("Only manifest schema_version 1 is supported.")
    dataset = mapping(values["dataset"], "dataset")
    fields(dataset, {"name", "release", "reference_build"}, set(), "dataset")
    inputs = mapping(values["inputs"], "inputs")
    fields(inputs, {"bgen", "sample", "pheno_file"}, {"covar_file"}, "inputs")
    analysis = parse_analysis(values["analysis"])
    if analysis.covariate_columns and "covar_file" not in inputs:
        raise ValueError("Selected covariate columns require inputs.covar_file.")
    if "covar_file" in inputs and not analysis.covariate_columns:
        raise ValueError("A covariate file requires explicit nonempty covariate_columns.")
    prediction_values = values["predictions"]
    if not isinstance(prediction_values, list):
        raise ValueError("predictions must be an array.")
    predictions: list[PredictionFile] = []
    for index, prediction in enumerate(prediction_values):
        entry = mapping(prediction, f"predictions[{index}]")
        fields(entry, {"phenotype", "uri", "sha256", "size_bytes"}, {"generation"}, f"predictions[{index}]")
        predictions.append(
            PredictionFile(
                safe_name(entry["phenotype"], "predictions.phenotype"),
                input_file({key: item for key, item in entry.items() if key != "phenotype"}, f"predictions[{index}]"),
            )
        )
    if sorted(entry.phenotype for entry in predictions) != sorted(analysis.phenotype_columns):
        raise ValueError("predictions must contain exactly one LOCO file for each selected phenotype.")
    resources = parse_resources(values["resources"])
    if analysis.trait_type != TraitType.BINARY and resources.firth_batch_size is not None:
        raise ValueError("firth_batch_size is only valid for binary analyses.")
    return JobManifest(
        Dataset(*(text(dataset[name], f"dataset.{name}") for name in ("name", "release", "reference_build"))),
        {name: input_file(item, f"inputs.{name}") for name, item in inputs.items()},
        tuple(predictions),
        analysis,
        resources,
        validate_uri(values["output_uri"], "output_uri"),
    )


def reject_duplicate_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    """Reject duplicate JSON keys rather than silently using the last value."""
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}.")
        result[key] = value
    return result


def load_manifest(path: Path) -> JobManifest:
    """Read and validate a pinned JSON job manifest."""
    value = typing.cast(
        "JsonValue", json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    )
    return parse_manifest(value)


def iter_input_files(job: JobManifest) -> tuple[InputFile, ...]:
    """Enumerate all pinned source files in stable order."""
    return (*job.inputs.values(), *(entry.file for entry in job.predictions))
