"""Input alignment and conservative capacity planning without importing JAX."""

from __future__ import annotations

import csv
import dataclasses
import math
import re
import shutil
import struct
from pathlib import Path

from tooling.workbench import manifest, profile, storage

MISSING_TOKENS = frozenset({"", "NA", "NaN", "nan", "-9"})
GIBIBYTE = 1024**3
FIRTH_CANDIDATE_CAPACITY = 1024


@dataclasses.dataclass(frozen=True)
class CapacityPlan:
    """A deliberately conservative heuristic, never an allocation guarantee."""

    chunk_variants: int
    declared_memory_bytes: int
    working_memory_budget_bytes: int
    fixed_memory_estimate_bytes: int
    correction_memory_estimate_bytes: int
    correction_preparation_estimate_bytes: int
    correction_solver_estimate_bytes: int
    padded_correction_candidate_lanes: int
    bytes_per_variant_estimate: int
    estimated_chunk_working_set_bytes: int
    firth_batch_size: int | None
    out_of_memory_guarantee: bool = False


@dataclasses.dataclass(frozen=True)
class AlignmentReport:
    """Aggregate selected sample counts without reporting participant identifiers."""

    source_sample_count: int
    selected_sample_counts: dict[str, int]
    prediction_chromosomes: dict[str, tuple[str, ...]]
    statistical_validity_qualified: bool = False


@dataclasses.dataclass(frozen=True)
class PreflightReport:
    """Explicit validation scope including unlocalized cloud inputs."""

    schema_version: int
    local_inputs_verified: int
    remote_inputs_pending_content_verification: int
    remote_metadata_checked: int
    required_free_disk_bytes: int
    available_free_disk_bytes: int
    header: profile.BgenHeader | None
    alignment: AlignmentReport | None
    capacity: CapacityPlan | None
    all_input_content_verified: bool
    native_genotype_validation_required: bool = True
    all_of_us_qualification: bool = False


def read_sample_identifiers(path: Path, expected_sample_count: int) -> tuple[profile.SampleIdentifier, ...]:
    """Use the shared Oxford parser and validate LOCO serialization identity."""
    result = tuple(profile.iter_sample_identifiers(path))
    if len(result) != expected_sample_count:
        raise ValueError("Oxford sample count differs from the BGEN header.")
    if len({identifier.loco_key() for identifier in result}) != len(result):
        raise ValueError("Distinct sample pairs serialize to ambiguous REGENIE LOCO header keys.")
    return result


def finite_float32(value: str) -> float:
    """Match native float32 parsing and reject overflow before engine launch."""
    try:
        result = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except OverflowError as error:
        raise ValueError("Selected numeric value exceeds the finite float32 range.") from error
    if not math.isfinite(result):
        raise ValueError("Selected numeric values must be finite float32 values.")
    return result


def read_selected_table(
    path: Path, columns: tuple[str, ...], *, binary: bool
) -> dict[profile.SampleIdentifier, tuple[float | None, ...]]:
    """Read selected numeric values with the engine's missing-value conventions."""
    result: dict[profile.SampleIdentifier, tuple[float | None, ...]] = {}
    with path.open(encoding="utf-8", newline="") as source:
        records = csv.reader(source, delimiter="\t")
        header: list[str] = []
        for row in records:
            header = [item.strip() for item in row]
            if any(header):
                break
        selected = ("FID", "IID", *columns)
        if any(header.count(column) != 1 for column in selected):
            raise ValueError("Tab-separated table must contain exactly one FID, IID and each selected column.")
        indexes = [header.index(column) for column in selected]
        for row_number, row in enumerate(records, start=2):
            fields = [item.strip() for item in row]
            if not any(fields):
                continue
            if max(indexes) >= len(fields):
                raise ValueError(f"Selected input table is missing a selected field at row {row_number}.")
            identifier = profile.SampleIdentifier(fields[indexes[0]], fields[indexes[1]])
            if not identifier.family or not identifier.individual:
                raise ValueError("Selected input table has empty sample identifiers.")
            if identifier in result:
                raise ValueError("Selected input table contains duplicate family/individual pairs.")
            values: list[float | None] = []
            for index in indexes[2:]:
                value = fields[index]
                if value in MISSING_TOKENS:
                    values.append(None)
                    continue
                numeric = finite_float32(value)
                if not math.isfinite(numeric) or (binary and numeric not in {1.0, 2.0}):
                    raise ValueError("Selected values must be finite; binary phenotypes must use 1=control and 2=case.")
                values.append(numeric)
            result[identifier] = tuple(values)
    return result


def normalize_chromosome(value: str) -> str:
    """Match native LOCO normalization, including irrelevant genome-wide X rows."""
    chromosome = value.translate(str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"))
    chromosome = chromosome.removeprefix("chr")
    if chromosome == "x":
        return "23"
    if chromosome.isascii() and chromosome.isdecimal() and int(chromosome) <= 2**64 - 1:
        return str(int(chromosome))
    return chromosome


def ascii_fields(line: str) -> list[str]:
    """Match the native LOCO parser's ASCII whitespace boundaries."""
    return re.split(r"[ \t\r\n\v\f]+", line.strip(" \t\r\n\v\f"))


def validate_loco(path: Path, selected: set[profile.SampleIdentifier]) -> tuple[str, ...]:
    """Validate selected LOCO membership and finite values without retaining rows."""
    expected_keys = {identifier.loco_key() for identifier in selected}
    with path.open(encoding="utf-8-sig") as source:
        header = ascii_fields(source.readline())
        if len(header) < 2 or header[0] != "FID_IID" or len(set(header[1:])) != len(header) - 1:
            raise ValueError("LOCO prediction header must begin FID_IID and contain unique sample keys.")
        if not expected_keys.issubset(header[1:]):
            raise ValueError("LOCO prediction header is missing selected phenotype samples.")
        selected_indexes = [index for index, key in enumerate(header) if index > 0 and key in expected_keys]
        chromosomes: set[str] = set()
        for line in source:
            if not line.strip():
                continue
            fields = ascii_fields(line)
            if len(fields) != len(header):
                raise ValueError("LOCO prediction row has an inconsistent field count.")
            chromosome = normalize_chromosome(fields[0])
            if chromosome in chromosomes:
                raise ValueError("LOCO prediction file contains duplicate chromosome rows.")
            chromosomes.add(chromosome)
            if any(not math.isfinite(finite_float32(fields[index])) for index in selected_indexes):
                raise ValueError("LOCO predictions for selected samples must be finite and nonmissing.")
    if not chromosomes:
        raise ValueError("LOCO prediction file has no chromosome rows.")
    return tuple(sorted(chromosomes))


def validate_alignment(
    job: manifest.JobManifest, paths: dict[str, Path], header: profile.BgenHeader
) -> AlignmentReport:
    """Check structural alignment and aggregate sample availability, without QC claims."""
    identifiers = read_sample_identifiers(paths["sample"], header.sample_count)
    phenotypes = read_selected_table(
        paths["pheno_file"], job.analysis.phenotype_columns, binary=job.analysis.trait_type == manifest.TraitType.BINARY
    )
    covariates = (
        read_selected_table(paths["covar_file"], job.analysis.covariate_columns, binary=False)
        if "covar_file" in paths
        else None
    )
    selected_counts: dict[str, int] = {}
    prediction_chromosomes: dict[str, tuple[str, ...]] = {}
    for column_index, phenotype in enumerate(job.analysis.phenotype_columns):
        selected = {
            identifier
            for identifier in identifiers
            if identifier in phenotypes
            and phenotypes[identifier][column_index] is not None
            and (
                covariates is None
                or (identifier in covariates and all(value is not None for value in covariates[identifier]))
            )
        }
        if len(selected) <= len(job.analysis.covariate_columns) + 2:
            raise ValueError("Selected phenotype has too few aligned nonmissing samples for the covariate design.")
        if (
            job.analysis.trait_type == manifest.TraitType.BINARY
            and len({phenotypes[identifier][column_index] for identifier in selected}) != 2
        ):
            raise ValueError("Each selected binary phenotype must contain both controls and cases after alignment.")
        selected_counts[phenotype] = len(selected)
        prediction_chromosomes[phenotype] = validate_loco(paths[f"prediction:{phenotype}"], selected)
    return AlignmentReport(header.sample_count, selected_counts, prediction_chromosomes)


@dataclasses.dataclass(frozen=True)
class CorrectionEstimate:
    """Worst-case grouped candidate preparation and sequential solver workspace."""

    preparation_bytes: int
    solver_bytes: int
    padded_candidate_lanes: int

    def total_bytes(self) -> int:
        """Return the combined correction working-set estimate."""
        return self.preparation_bytes + self.solver_bytes


def estimate_correction_memory(
    sample_count: int,
    phenotype_count: int,
    covariate_count: int,
    chunk_variants: int,
    firth_batch_size: int | None,
) -> CorrectionEstimate:
    """Include materialized candidates and covariate-dependent gathered designs."""
    if firth_batch_size is None:
        return CorrectionEstimate(0, 0, 0)
    candidate_count = min(FIRTH_CANDIDATE_CAPACITY, chunk_variants) * phenotype_count
    padded_count = ((candidate_count + firth_batch_size - 1) // firth_batch_size) * firth_batch_size
    preparation_bytes = sample_count * padded_count * 4 * (covariate_count + 8)
    solver_bytes = sample_count * firth_batch_size * 4 * 8
    return CorrectionEstimate(preparation_bytes, solver_bytes, padded_count)


def plan_capacity(job: manifest.JobManifest, sample_count: int, variant_count: int) -> CapacityPlan:
    """Bound chunk width and Firth preparation while preserving the full cohort."""
    resources = job.resources
    memory_gib = resources.gpu_memory_gib if resources.device == manifest.Device.GPU else resources.memory_gib
    if memory_gib is None:
        raise ValueError("Declared device memory is required for capacity planning.")
    memory_bytes = int(memory_gib * GIBIBYTE)
    budget = int(memory_bytes * resources.memory_fraction)
    phenotype_count = len(job.analysis.phenotype_columns)
    covariate_count = len(job.analysis.covariate_columns) + 1
    fixed_memory = GIBIBYTE + sample_count * 8 * (covariate_count * (3 + phenotype_count) + phenotype_count * 6 + 16)
    bytes_per_variant = sample_count * 4 * (8 + 2 * phenotype_count)
    firth_batch_size = None
    if job.analysis.fallback_method == manifest.FallbackMethod.FIRTH_APPROXIMATE:
        firth_batch_size = resources.firth_batch_size
        if firth_batch_size is None:
            maximum_batch = min(256, resources.max_chunk_variants)
            candidate_batch = 1 << (maximum_batch.bit_length() - 1)
            while candidate_batch >= 1:
                estimate = estimate_correction_memory(
                    sample_count, phenotype_count, covariate_count, 1, candidate_batch
                )
                if fixed_memory + bytes_per_variant + estimate.total_bytes() <= budget:
                    firth_batch_size = candidate_batch
                    break
                candidate_batch //= 2
            if firth_batch_size is None:
                raise ValueError("Declared memory budget cannot accommodate one variant's Firth preparation.")
    # Candidate capacity grows with width until the native per-trait cap of 1024.
    # Search the monotone estimate rather than subtracting only a solver batch.
    lower_width = 0
    upper_width = min(resources.max_chunk_variants, variant_count, (2**31 - 1) // phenotype_count)
    while lower_width < upper_width:
        candidate_width = (lower_width + upper_width + 1) // 2
        correction = estimate_correction_memory(
            sample_count, phenotype_count, covariate_count, candidate_width, firth_batch_size
        )
        working_set = fixed_memory + correction.total_bytes() + candidate_width * bytes_per_variant
        if working_set <= budget and correction.padded_candidate_lanes <= 2**31 - 1:
            lower_width = candidate_width
        else:
            upper_width = candidate_width - 1
    if lower_width < 1:
        raise ValueError("Declared memory budget cannot accommodate one variant under the conservative estimate.")
    correction = estimate_correction_memory(
        sample_count, phenotype_count, covariate_count, lower_width, firth_batch_size
    )
    return CapacityPlan(
        chunk_variants=lower_width,
        declared_memory_bytes=memory_bytes,
        working_memory_budget_bytes=budget,
        fixed_memory_estimate_bytes=fixed_memory,
        correction_memory_estimate_bytes=correction.total_bytes(),
        correction_preparation_estimate_bytes=correction.preparation_bytes,
        correction_solver_estimate_bytes=correction.solver_bytes,
        padded_correction_candidate_lanes=correction.padded_candidate_lanes,
        bytes_per_variant_estimate=bytes_per_variant,
        estimated_chunk_working_set_bytes=fixed_memory + correction.total_bytes() + lower_width * bytes_per_variant,
        firth_batch_size=firth_batch_size,
    )


def available_disk_bytes(directory: Path) -> int:
    """Inspect the nearest existing ancestor without creating a work directory."""
    directory = directory.resolve()
    while not directory.exists():
        if directory.parent == directory:
            raise ValueError("Cannot resolve an existing work-directory ancestor.")
        directory = directory.parent
    if not directory.is_dir():
        raise ValueError("Work directory or its ancestor is not a directory.")
    return shutil.disk_usage(directory).free


def estimated_disk_bytes(job: manifest.JobManifest, variant_count: int | None) -> int:
    """Reserve localization, output publication copies and compilation headroom."""
    source_bytes = sum(source.size_bytes for source in manifest.iter_input_files(job))
    output_bytes = (variant_count or 0) * len(job.analysis.phenotype_columns) * 512
    return source_bytes + output_bytes * 3 + 2 * GIBIBYTE


def input_paths(job: manifest.JobManifest) -> dict[str, Path]:
    """Return only already-local sources without performing localization."""
    result = {name: Path(source.uri) for name, source in job.inputs.items() if not source.uri.startswith("gs://")}
    result.update(
        {
            f"prediction:{entry.phenotype}": Path(entry.file.uri)
            for entry in job.predictions
            if not entry.file.uri.startswith("gs://")
        }
    )
    return result


def run_preflight(
    job: manifest.JobManifest,
    work_directory: Path,
    *,
    dry_run: bool,
    billing_project: str | None,
) -> PreflightReport:
    """Verify local inputs and optionally remote metadata, never downloading data."""
    local_count = 0
    remote_count = 0
    remote_checked = 0
    for source in manifest.iter_input_files(job):
        if source.uri.startswith("gs://"):
            remote_count += 1
            if not dry_run:
                storage.check_remote_source(source, billing_project)
                remote_checked += 1
        else:
            storage.verify_file(Path(source.uri), source)
            local_count += 1
    paths = input_paths(job)
    header = profile.read_header(paths["bgen"]) if "bgen" in paths else None
    if header is not None and (header.layout != 2 or header.sample_count <= 0 or header.variant_count <= 0):
        raise ValueError("Workbench pilot requires a nonempty BGEN Layout 2 file.")
    required_count = len(manifest.iter_input_files(job))
    alignment = validate_alignment(job, paths, header) if header is not None and len(paths) == required_count else None
    capacity = plan_capacity(job, header.sample_count, header.variant_count) if header is not None else None
    required_disk = estimated_disk_bytes(job, header.variant_count if header is not None else None)
    available_disk = available_disk_bytes(work_directory)
    if available_disk < required_disk:
        raise ValueError("Insufficient free local disk for verified inputs, estimated outputs and reserved headroom.")
    return PreflightReport(
        1,
        local_count,
        remote_count,
        remote_checked,
        required_disk,
        available_disk,
        header,
        alignment,
        capacity,
        remote_count == 0,
    )
