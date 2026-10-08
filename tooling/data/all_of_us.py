"""Plan and prepare audited local cohorts for All of Us Workbench analyses."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import shutil
import subprocess
import typing
from dataclasses import dataclass
from pathlib import Path

from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.data import all_of_us_inputs as cohort_inputs
from tooling.workbench import profile as bgen_profile

if typing.TYPE_CHECKING:
    import omegaconf


class PreparationMode(enum.StrEnum):
    """Actions supported without implicitly launching expensive conversion."""

    PLAN = "plan"
    TABLES = "tables"
    EXECUTE = "execute"


class SourceFormat(enum.StrEnum):
    """Local genotype formats and explicit table-only preparation."""

    PGEN = "pgen"
    BED = "bed"
    BGEN = "bgen"
    TABLES = "tables"


class ReferenceOrder(enum.StrEnum):
    """Reference-allele assertions supported by PLINK's BGEN importer."""

    FIRST = "ref-first"
    LAST = "ref-last"
    UNKNOWN = "ref-unknown"


class PhenotypeTrait(enum.StrEnum):
    """Public phenotype coding contracts."""

    QUANTITATIVE = "quantitative"
    BINARY = "binary"


class PreparationStatus(enum.StrEnum):
    """Persisted state of a new preparation attempt."""

    PLANNED = "planned"
    TABLES_PREPARED = "tables_prepared"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class PreparationArguments:
    """Explicit inputs, selection policy, and bounded PLINK resources."""

    mode: PreparationMode
    source_format: SourceFormat
    input_prefix: Path | None
    input_bgen: Path | None
    input_sample: Path | None
    pvar_zstd: bool
    bgen_reference: ReferenceOrder | None
    confirm_diploid: bool
    identity_map: Path
    phenotype_file: Path
    phenotype_columns: tuple[str, ...]
    phenotype_trait: PhenotypeTrait
    covariate_file: Path | None
    covariate_columns: tuple[str, ...]
    keep_file: Path | None
    remove_file: Path | None
    release_exclusion_files: tuple[Path, ...]
    variant_extract_file: Path | None
    minimum_allele_count: float | None
    minimum_variant_call_rate: float | None
    release_label: str
    genome_build: str
    output_directory: Path
    plink_executable: str
    threads: int
    memory_megabytes: int
    hash_genotype_inputs: bool


@dataclass(frozen=True)
class FileFingerprint:
    """Content identity for small inputs and filesystem identity for large ones."""

    path: str
    byte_count: int
    modification_time_ns: int
    change_time_ns: int
    inode: int
    device: int
    sha256: str | None


@dataclass(frozen=True)
class PlinkCommand:
    """An argument vector executed directly, without a shell."""

    name: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class PreparationPlan:
    """Auditable input identities, cohort policy, and exact conversion commands."""

    schema_version: int
    release_label: str
    genome_build: str
    source_format: SourceFormat
    mode: PreparationMode
    input_files: tuple[FileFingerprint, ...]
    sample_counts: dict[str, int]
    sample_selection_files: dict[str, tuple[str, ...]]
    phenotype_trait: PhenotypeTrait
    phenotype_columns: tuple[str, ...]
    covariate_columns: tuple[str, ...]
    phenotype_missing_counts: dict[str, int]
    covariate_missing_counts: dict[str, int]
    analysis_complete_case_counts: dict[str, int]
    minimum_allele_count: float | None
    minimum_variant_call_rate: float | None
    variant_extract_file: str | None
    variant_policy: str
    source_reference: ReferenceOrder | None
    diploid_input_confirmed: bool
    missingness_policy: str
    probability_conversion: str
    outputs: dict[str, str]
    commands: tuple[PlinkCommand, ...]
    step1_requirement: str


@dataclass(frozen=True)
class PreparedCohort:
    """Validated in-memory tables and a side-effect-free preparation plan."""

    plan: PreparationPlan
    mappings: tuple[cohort_inputs.IdentityMapping, ...]
    phenotypes: cohort_inputs.AlignedTable
    covariates: cohort_inputs.AlignedTable | None


@dataclass(frozen=True)
class CohortSelection:
    """Source-ordered selected mappings and explicit exclusion accounting."""

    mappings: tuple[cohort_inputs.IdentityMapping, ...]
    counts: dict[str, int]


@dataclass(frozen=True)
class PreparationManifest:
    """Persistent attempt state and successful output fingerprints."""

    status: PreparationStatus
    plan: PreparationPlan
    output_files: tuple[FileFingerprint, ...]
    variant_count: int | None
    error: str | None


def with_extension(prefix: Path, extension: str) -> Path:
    """Append an extension while retaining dots within a genotype prefix."""
    return Path(f"{prefix}{extension}")


def fingerprint_file(path: Path, *, content: bool) -> FileFingerprint:
    """Hash an input when requested and reject mutation during fingerprinting."""
    resolved_path = path.resolve(strict=True)
    before = resolved_path.stat()
    if not resolved_path.is_file() or before.st_size == 0:
        raise ValueError(f"Input or output is not a nonempty regular file: {path}.")
    content_hash = None
    if content:
        with resolved_path.open("rb") as input_file:
            content_hash = hashlib.file_digest(input_file, "sha256").hexdigest()
    after = resolved_path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or before.st_ino != after.st_ino
        or before.st_dev != after.st_dev
    ):
        raise ValueError(f"File changed while fingerprinting: {path}.")
    return FileFingerprint(
        path=str(resolved_path),
        byte_count=after.st_size,
        modification_time_ns=after.st_mtime_ns,
        change_time_ns=after.st_ctime_ns,
        inode=after.st_ino,
        device=after.st_dev,
        sha256=content_hash,
    )


def validate_arguments(arguments: PreparationArguments) -> None:
    """Validate policy and resource settings before reading or writing data."""
    if not arguments.release_label.strip() or not arguments.genome_build.strip():
        raise ValueError("A release_label and genome_build must be recorded explicitly.")
    if arguments.threads < 1 or arguments.memory_megabytes < 640:
        raise ValueError("PLINK resources require threads >= 1 and memory_megabytes >= 640.")
    if arguments.output_directory.exists() or arguments.output_directory.is_symlink():
        raise FileExistsError(f"Refusing an existing output directory: {arguments.output_directory}.")
    if arguments.minimum_allele_count is not None and (
        not math.isfinite(arguments.minimum_allele_count) or arguments.minimum_allele_count < 0
    ):
        raise ValueError("minimum_allele_count must be finite and nonnegative.")
    if arguments.minimum_variant_call_rate is not None and (
        not math.isfinite(arguments.minimum_variant_call_rate) or not 0 <= arguments.minimum_variant_call_rate <= 1
    ):
        raise ValueError("minimum_variant_call_rate must lie within [0, 1].")
    if (arguments.covariate_file is None) != (not arguments.covariate_columns):
        raise ValueError("covariate_file and nonempty covariate_columns must be supplied together.")
    if arguments.source_format == SourceFormat.TABLES:
        if arguments.mode == PreparationMode.EXECUTE:
            raise ValueError("Table-only sources cannot execute genotype conversion.")
        if arguments.variant_extract_file is not None or arguments.minimum_allele_count is not None:
            raise ValueError("Table-only sources cannot apply variant filters.")
        if arguments.minimum_variant_call_rate is not None:
            raise ValueError("Table-only sources cannot apply a genotype call-rate filter.")
    elif not arguments.confirm_diploid:
        raise ValueError("Confirm diploid human autosomal inputs after upstream QC with confirm_diploid=true.")
    if arguments.source_format == SourceFormat.BGEN and arguments.bgen_reference is None:
        raise ValueError("BGEN inputs require an explicit bgen_reference allele-order assertion.")


def source_files(arguments: PreparationArguments) -> tuple[Path, ...]:
    """Resolve every local genotype component without remote fetching."""
    if arguments.source_format == SourceFormat.TABLES:
        return ()
    if arguments.source_format == SourceFormat.BGEN:
        if arguments.input_bgen is None or arguments.input_sample is None:
            raise ValueError("BGEN input requires input_bgen and an explicit two-ID input_sample.")
        return (arguments.input_bgen, arguments.input_sample)
    if arguments.input_prefix is None:
        raise ValueError("PGEN and BED inputs require input_prefix.")
    if arguments.source_format == SourceFormat.BED:
        return tuple(with_extension(arguments.input_prefix, extension) for extension in (".bed", ".bim", ".fam"))
    variant_extension = ".pvar.zst" if arguments.pvar_zstd else ".pvar"
    paths = tuple(
        with_extension(arguments.input_prefix, extension) for extension in (".pgen", variant_extension, ".psam")
    )
    index_path = with_extension(arguments.input_prefix, ".pgen.pgi")
    return (*paths, index_path) if index_path.exists() else paths


def source_identities(
    arguments: PreparationArguments,
    mappings: tuple[cohort_inputs.IdentityMapping, ...],
) -> tuple[cohort_inputs.SampleIdentity, ...]:
    """Read physical genotype sample order instead of assuming mapping order."""
    paths = source_files(arguments)
    if arguments.source_format == SourceFormat.TABLES:
        return tuple(mapping.source for mapping in mappings)
    if arguments.source_format == SourceFormat.BGEN:
        with paths[1].open(encoding="utf-8") as sample_file:
            if next(sample_file, "").split()[:2] != ["ID_1", "ID_2"]:
                raise ValueError("PLINK preparation requires Oxford ID_1 and ID_2 as the first two sample columns.")
        return cohort_inputs.read_oxford_samples(paths[1])
    if arguments.source_format == SourceFormat.BED:
        return cohort_inputs.read_family_samples(paths[2])
    return cohort_inputs.read_plink_samples(paths[2])


def select_mappings(
    arguments: PreparationArguments,
    mappings: tuple[cohort_inputs.IdentityMapping, ...],
    ordered_source_ids: tuple[cohort_inputs.SampleIdentity, ...],
) -> CohortSelection:
    """Apply explicit inclusion and exclusion sets while retaining source order."""
    mapping_by_source = {mapping.source: mapping for mapping in mappings}
    if set(mapping_by_source) != set(ordered_source_ids):
        raise ValueError("identity_map must cover every source sample exactly, without extra source IDs.")
    target_ids = frozenset(mapping.target for mapping in mappings)
    keep_ids = target_ids if arguments.keep_file is None else cohort_inputs.read_cohort_list(arguments.keep_file)
    if not keep_ids or not keep_ids <= target_ids:
        raise ValueError("keep_file must select nonempty, known canonical FID/IID pairs.")
    remove_ids = frozenset() if arguments.remove_file is None else cohort_inputs.read_cohort_list(arguments.remove_file)
    release_ids = frozenset(
        identity for path in arguments.release_exclusion_files for identity in cohort_inputs.read_cohort_list(path)
    )
    excluded_ids = remove_ids | release_ids
    eligible_ids = keep_ids - excluded_ids
    selected = tuple(
        mapping_by_source[source_id]
        for source_id in ordered_source_ids
        if mapping_by_source[source_id].target in eligible_ids
    )
    if not selected:
        raise ValueError("Explicit sample selection and exclusions leave no participants.")
    return CohortSelection(
        mappings=selected,
        counts={
            "source": len(ordered_source_ids),
            "requested_keep": len(keep_ids),
            "remove_matching_source": len(remove_ids & target_ids),
            "release_exclusions_matching_source": len(release_ids & target_ids),
            "exclusions_absent_from_source": len(excluded_ids - target_ids),
            "selected": len(selected),
            "excluded": len(mappings) - len(selected),
        },
    )


def output_paths(arguments: PreparationArguments) -> dict[str, str]:
    """Return stable artifact names inside one new preparation attempt."""
    directory = arguments.output_directory.resolve()
    paths = {
        "manifest": "preparation.json",
        "cohort_keep": "cohort.keep.tsv",
        "regenie_keep": "regenie.keep",
        "source_keep": "source.keep.tsv",
        "identity_update": "identity.update.tsv",
        "phenotypes": "phenotypes.tsv",
        "selected_prefix": "selected",
        "genotype_prefix": "genotypes",
        "bgen": "genotypes.bgen",
        "sample": "genotypes.sample",
    }
    if arguments.covariate_file is not None:
        paths["covariates"] = "covariates.tsv"
    return {name: str(directory / filename) for name, filename in paths.items()}


def build_plink_commands(arguments: PreparationArguments, outputs: dict[str, str]) -> tuple[PlinkCommand, ...]:
    """Build bounded source-selection and identity-preserving export commands."""
    if arguments.source_format == SourceFormat.TABLES:
        return ()
    source_arguments: list[str]
    if arguments.source_format == SourceFormat.BGEN:
        source_arguments = [
            "--bgen",
            str(arguments.input_bgen.resolve()) if arguments.input_bgen is not None else "",
            str(arguments.bgen_reference),
            "--sample",
            str(arguments.input_sample.resolve()) if arguments.input_sample is not None else "",
            "--import-dosage-certainty",
            "0",
            "--dosage-erase-threshold",
            "0",
        ]
    else:
        source_arguments = [
            "--pfile" if arguments.source_format == SourceFormat.PGEN else "--bfile",
            str(arguments.input_prefix.resolve()) if arguments.input_prefix is not None else "",
        ]
        if arguments.source_format == SourceFormat.PGEN and arguments.pvar_zstd:
            source_arguments.append("vzs")
    resources = ("--threads", str(arguments.threads), "--memory", str(arguments.memory_megabytes))
    filters = ["--keep", outputs["source_keep"], "--human", "--autosome", "--min-alleles", "2", "--max-alleles", "2"]
    if arguments.variant_extract_file is not None:
        filters.extend(("--extract", str(arguments.variant_extract_file.resolve())))
    if arguments.minimum_allele_count is not None:
        filters.extend(("--nonfounders", "--mac", str(arguments.minimum_allele_count)))
    if arguments.minimum_variant_call_rate is not None:
        filters.extend(("--geno", format(1.0 - arguments.minimum_variant_call_rate, ".17g"), "dosage"))
    return (
        PlinkCommand(
            name="select",
            arguments=(
                arguments.plink_executable,
                *source_arguments,
                *filters,
                "--make-pgen",
                *resources,
                "--out",
                outputs["selected_prefix"],
            ),
        ),
        PlinkCommand(
            name="export",
            arguments=(
                arguments.plink_executable,
                "--pfile",
                outputs["selected_prefix"],
                "--update-ids",
                outputs["identity_update"],
                "--make-pgen",
                "--export",
                "bgen-1.2",
                "ref-first",
                "bits=8",
                "id-paste=fid,iid",
                *resources,
                "--out",
                outputs["genotype_prefix"],
            ),
        ),
    )


def prepare_cohort(arguments: PreparationArguments) -> PreparedCohort:
    """Read and validate local inputs and return a plan without creating files."""
    validate_arguments(arguments)
    genotype_paths = source_files(arguments)
    metadata_paths = (
        arguments.identity_map,
        arguments.phenotype_file,
        *(() if arguments.covariate_file is None else (arguments.covariate_file,)),
        *(() if arguments.keep_file is None else (arguments.keep_file,)),
        *(() if arguments.remove_file is None else (arguments.remove_file,)),
        *arguments.release_exclusion_files,
        *(() if arguments.variant_extract_file is None else (arguments.variant_extract_file,)),
    )
    fingerprints = tuple(fingerprint_file(path, content=True) for path in metadata_paths) + tuple(
        fingerprint_file(path, content=arguments.hash_genotype_inputs or path.suffix in {".psam", ".fam", ".sample"})
        for path in genotype_paths
    )
    mappings = cohort_inputs.read_identity_mapping(arguments.identity_map)
    ordered_source_ids = source_identities(arguments, mappings)
    selection = select_mappings(arguments, mappings, ordered_source_ids)
    selected = selection.mappings
    if arguments.variant_extract_file is not None:
        cohort_inputs.validate_variant_list(arguments.variant_extract_file)
    target_ids = tuple(mapping.target for mapping in selected)
    phenotypes = cohort_inputs.align_numeric_table(
        arguments.phenotype_file,
        arguments.phenotype_columns,
        target_ids,
        binary=arguments.phenotype_trait == PhenotypeTrait.BINARY,
    )
    covariates = (
        None
        if arguments.covariate_file is None
        else cohort_inputs.align_numeric_table(
            arguments.covariate_file,
            arguments.covariate_columns,
            target_ids,
            binary=False,
        )
    )
    complete_case_counts = {
        column: sum(
            row[column_index] not in cohort_inputs.MISSING_VALUES
            and (
                covariates is None
                or not any(value in cohort_inputs.MISSING_VALUES for value in covariates.values[index])
            )
            for index, row in enumerate(phenotypes.values)
        )
        for column_index, column in enumerate(phenotypes.columns)
    }
    outputs = output_paths(arguments)
    plan = PreparationPlan(
        schema_version=1,
        release_label=arguments.release_label,
        genome_build=arguments.genome_build,
        source_format=arguments.source_format,
        mode=arguments.mode,
        input_files=fingerprints,
        sample_counts=selection.counts,
        sample_selection_files={
            "identity_map": (str(arguments.identity_map.resolve()),),
            "keep": () if arguments.keep_file is None else (str(arguments.keep_file.resolve()),),
            "remove": () if arguments.remove_file is None else (str(arguments.remove_file.resolve()),),
            "release_exclusions": tuple(str(path.resolve()) for path in arguments.release_exclusion_files),
        },
        phenotype_trait=arguments.phenotype_trait,
        phenotype_columns=arguments.phenotype_columns,
        covariate_columns=arguments.covariate_columns,
        phenotype_missing_counts=phenotypes.missing_counts,
        covariate_missing_counts={} if covariates is None else covariates.missing_counts,
        analysis_complete_case_counts=complete_case_counts,
        minimum_allele_count=arguments.minimum_allele_count,
        minimum_variant_call_rate=arguments.minimum_variant_call_rate,
        variant_extract_file=None
        if arguments.variant_extract_file is None
        else str(arguments.variant_extract_file.resolve()),
        variant_policy=(
            "Human autosomes 1-22; exactly two alleles; unique nonmissing variant IDs; "
            "no ancestry or relatedness exclusions."
        ),
        source_reference=arguments.bgen_reference if arguments.source_format == SourceFormat.BGEN else None,
        diploid_input_confirmed=arguments.confirm_diploid,
        missingness_policy=(
            "Preserve missing genotypes; normalize missing table tokens to NA; "
            "call rate counts available dosages; no fill or imputation."
        ),
        probability_conversion=(
            "PLINK retains dosage, not arbitrary genotype posteriors; BGEN export quantizes probabilities to 8 bits."
        ),
        outputs=outputs,
        commands=build_plink_commands(arguments, outputs),
        step1_requirement=(
            "Run external REGENIE Step 1 on genome-wide QC-selected markers from the same canonical cohort and tables. "
            "Do not train Step 1 on the chromosome-22 pilot alone. "
            "Supply matching phenotype-specific LOCO predictions to g Step 2."
        ),
    )
    revalidate_inputs(plan)
    return PreparedCohort(plan=plan, mappings=selected, phenotypes=phenotypes, covariates=covariates)


def revalidate_inputs(plan: PreparationPlan) -> None:
    """Reject changes to pinned local inputs before or during a preparation."""
    for expected in plan.input_files:
        if fingerprint_file(Path(expected.path), content=expected.sha256 is not None) != expected:
            raise ValueError(f"Preparation input changed: {expected.path}.")


def write_aligned_table(
    path: Path,
    table: cohort_inputs.AlignedTable,
    mappings: tuple[cohort_inputs.IdentityMapping, ...],
) -> None:
    """Persist canonical identities and selected numeric fields in cohort order."""
    cohort_inputs.write_tsv(
        path,
        ("FID", "IID", *table.columns),
        (
            (
                mapping.target.family_id,
                mapping.target.individual_id,
                *("NA" if value in cohort_inputs.MISSING_VALUES else value for value in values),
            )
            for mapping, values in zip(mappings, table.values, strict=True)
        ),
    )


def write_cohort_tables(prepared: PreparedCohort) -> None:
    """Create the keep lists, ID update map, and aligned analysis tables."""
    outputs = prepared.plan.outputs
    cohort_inputs.write_tsv(
        Path(outputs["cohort_keep"]),
        ("#FID", "IID"),
        ((mapping.target.family_id, mapping.target.individual_id) for mapping in prepared.mappings),
    )
    with Path(outputs["regenie_keep"]).open("x", encoding="utf-8") as keep_file:
        keep_file.writelines(
            f"{mapping.target.family_id}\t{mapping.target.individual_id}\n" for mapping in prepared.mappings
        )
    cohort_inputs.write_tsv(
        Path(outputs["source_keep"]),
        ("#FID", "IID"),
        ((mapping.source.family_id, mapping.source.individual_id) for mapping in prepared.mappings),
    )
    cohort_inputs.write_tsv(
        Path(outputs["identity_update"]),
        ("#OLD_FID", "OLD_IID", "NEW_FID", "NEW_IID"),
        (
            (
                mapping.source.family_id,
                mapping.source.individual_id,
                mapping.target.family_id,
                mapping.target.individual_id,
            )
            for mapping in prepared.mappings
        ),
    )
    write_aligned_table(Path(outputs["phenotypes"]), prepared.phenotypes, prepared.mappings)
    if prepared.covariates is not None:
        write_aligned_table(Path(outputs["covariates"]), prepared.covariates, prepared.mappings)


def write_manifest(path: Path, manifest: PreparationManifest) -> None:
    """Atomically update only the manifest owned by this fresh attempt."""
    temporary_path = path.with_suffix(".json.tmp")
    with temporary_path.open("x", encoding="utf-8") as output_file:
        output_file.write(json.dumps(dataclasses.asdict(manifest), indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def run_plink(command: PlinkCommand, output_directory: Path) -> None:
    """Run an exact argument vector and retain stdout, stderr, and PLINK logs."""
    with (
        (output_directory / f"{command.name}.stdout.log").open("x", encoding="utf-8") as stdout_file,
        (output_directory / f"{command.name}.stderr.log").open("x", encoding="utf-8") as stderr_file,
    ):
        subprocess.run(command.arguments, check=True, stdout=stdout_file, stderr=stderr_file)


def validate_selected_variants(prefix: Path) -> int:
    """Reject empty, nonautosomal, multiallelic or ambiguous retained variants."""
    variant_ids: set[str] = set()
    with with_extension(prefix, ".pvar").open(encoding="utf-8") as variant_file:
        header: list[str] = []
        for line in variant_file:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.split()
                continue
            fields = line.split()
            if not header or len(fields) != len(header):
                raise ValueError("Retained PVAR is malformed.")
            chromosome = fields[header.index("#CHROM")].removeprefix("chr")
            variant_id = fields[header.index("ID")]
            alternate = fields[header.index("ALT")]
            if not chromosome.isdigit() or not 1 <= int(chromosome) <= 22 or "," in alternate or alternate == ".":
                raise ValueError("Retained variants must be biallelic human autosomes.")
            if variant_id == "." or variant_id in variant_ids:
                raise ValueError("Retained variant IDs must be present and unique; normalize them upstream.")
            variant_ids.add(variant_id)
    if not variant_ids:
        raise ValueError("No variants remain after explicit filters.")
    return len(variant_ids)


def validate_export(prepared: PreparedCohort, variant_count: int) -> None:
    """Verify canonical sample order and every exported BGEN record boundary."""
    outputs = prepared.plan.outputs
    expected = tuple(mapping.target for mapping in prepared.mappings)
    if cohort_inputs.read_oxford_samples(Path(outputs["sample"])) != expected:
        raise ValueError("Exported Oxford identities or order differ from the selected canonical cohort.")
    canonical_prefix = Path(outputs["genotype_prefix"])
    for extension in (".pgen", ".pvar", ".psam"):
        fingerprint_file(with_extension(canonical_prefix, extension), content=False)
    if cohort_inputs.read_plink_samples(with_extension(canonical_prefix, ".psam")) != expected:
        raise ValueError("Exported PGEN identities or order differ from the selected canonical cohort.")
    if validate_selected_variants(canonical_prefix) != variant_count:
        raise ValueError("Exported PGEN variant count differs from the selected source.")
    header = bgen_profile.validate_record_boundaries(Path(outputs["bgen"]))
    if header.variant_count != variant_count or header.sample_count != len(expected):
        raise ValueError("Exported BGEN header does not match prepared variant/sample counts.")


def execute_preparation(arguments: PreparationArguments, prepared: PreparedCohort) -> PreparationManifest:
    """Create one audited preparation attempt, preserving failures for diagnosis."""
    if arguments.mode == PreparationMode.EXECUTE and shutil.which(arguments.plink_executable) is None:
        raise FileNotFoundError("PLINK2 is unavailable; use plan or tables mode without genotype conversion.")
    revalidate_inputs(prepared.plan)
    arguments.output_directory.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(prepared.plan.outputs["manifest"])
    manifest = PreparationManifest(
        status=PreparationStatus.RUNNING,
        plan=prepared.plan,
        output_files=(),
        variant_count=None,
        error=None,
    )
    write_manifest(manifest_path, manifest)
    try:
        write_cohort_tables(prepared)
        variant_count = None
        if arguments.mode == PreparationMode.EXECUTE:
            version_result = subprocess.run(
                [arguments.plink_executable, "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
            (arguments.output_directory / "plink-version.txt").write_text(version_result.stdout, encoding="utf-8")
            for command in prepared.plan.commands:
                run_plink(command, arguments.output_directory)
                if command.name == "select":
                    selected_prefix = Path(prepared.plan.outputs["selected_prefix"])
                    selected_ids = cohort_inputs.read_plink_samples(with_extension(selected_prefix, ".psam"))
                    if selected_ids != tuple(mapping.source for mapping in prepared.mappings):
                        raise ValueError(
                            "PLINK selected sample identities or order differ from the audited source keep list."
                        )
                    variant_count = validate_selected_variants(selected_prefix)
            if variant_count is None:
                raise RuntimeError("No source selection command completed.")
            validate_export(prepared, variant_count)
        revalidate_inputs(prepared.plan)
        output_files = tuple(
            fingerprint_file(path, content=True)
            for path in sorted(arguments.output_directory.iterdir())
            if path.is_file() and path.resolve() != manifest_path.resolve() and path.stat().st_size > 0
        )
        manifest = dataclasses.replace(
            manifest,
            status=PreparationStatus.COMPLETE
            if arguments.mode == PreparationMode.EXECUTE
            else PreparationStatus.TABLES_PREPARED,
            output_files=output_files,
            variant_count=variant_count,
        )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        write_manifest(manifest_path, dataclasses.replace(manifest, status=PreparationStatus.FAILED, error=str(error)))
        raise
    write_manifest(manifest_path, manifest)
    return manifest


def run_tool(arguments: PreparationArguments) -> None:
    """Print a dry plan or prepare a new local cohort attempt."""
    prepared = prepare_cohort(arguments)
    if arguments.mode == PreparationMode.PLAN:
        print(json.dumps(dataclasses.asdict(prepared.plan), indent=2, sort_keys=True))
        return
    manifest = execute_preparation(arguments, prepared)
    print(f"Preparation {manifest.status}: {prepared.plan.outputs['manifest']}")


def build_arguments_from_config(config: omegaconf.DictConfig) -> PreparationArguments:
    """Resolve the grouped Hydra configuration into typed preparation arguments."""
    values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    reference = values["bgen_reference"]
    return PreparationArguments(
        mode=PreparationMode(str(values["mode"])),
        source_format=SourceFormat(str(values["source_format"])),
        input_prefix=tooling_hydra_arguments.path_or_none(values["input_prefix"]),
        input_bgen=tooling_hydra_arguments.path_or_none(values["input_bgen"]),
        input_sample=tooling_hydra_arguments.path_or_none(values["input_sample"]),
        pvar_zstd=tooling_hydra_arguments.boolean_value(values["pvar_zstd"]),
        bgen_reference=None if reference is None else ReferenceOrder(str(reference)),
        confirm_diploid=tooling_hydra_arguments.boolean_value(values["confirm_diploid"]),
        identity_map=Path(str(values["identity_map"])),
        phenotype_file=Path(str(values["phenotype_file"])),
        phenotype_columns=tuple(str(column) for column in values["phenotype_columns"]),
        phenotype_trait=PhenotypeTrait(str(values["phenotype_trait"])),
        covariate_file=tooling_hydra_arguments.path_or_none(values["covariate_file"]),
        covariate_columns=tuple(str(column) for column in values["covariate_columns"]),
        keep_file=tooling_hydra_arguments.path_or_none(values["keep_file"]),
        remove_file=tooling_hydra_arguments.path_or_none(values["remove_file"]),
        release_exclusion_files=tuple(Path(str(path)) for path in values["release_exclusion_files"]),
        variant_extract_file=tooling_hydra_arguments.path_or_none(values["variant_extract_file"]),
        minimum_allele_count=tooling_hydra_arguments.float_or_none(values["minimum_allele_count"]),
        minimum_variant_call_rate=tooling_hydra_arguments.float_or_none(values["minimum_variant_call_rate"]),
        release_label=str(values["release_label"]),
        genome_build=str(values["genome_build"]),
        output_directory=Path(str(values["output_directory"])),
        plink_executable=str(values["plink_executable"]),
        threads=int(values["threads"]),
        memory_megabytes=int(values["memory_megabytes"]),
        hash_genotype_inputs=tooling_hydra_arguments.boolean_value(values["hash_genotype_inputs"]),
    )
