"""Format-agnostic genotype source orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from g import models, types
from g.io import bgen, plink, reader


@dataclass(frozen=True)
class GenotypeSourceConfig:
    """Configuration describing one genotype input source."""

    source_format: types.GenotypeSourceFormat
    source_path: Path
    sample_path: Path | None = None


resolve_bgen_sample_path = bgen.resolve_bgen_sample_path
load_bgen_sample_table = bgen.load_bgen_sample_table
build_sample_identifier_table = bgen.build_sample_identifier_table
load_aligned_sample_data = plink.load_aligned_sample_data
load_aligned_sample_data_from_individual_identifier_table = (
    plink.load_aligned_sample_data_from_individual_identifier_table
)


def build_plink_source_config(bed_prefix: Path | str) -> GenotypeSourceConfig:
    """Build a genotype source config for a PLINK dataset prefix."""
    return GenotypeSourceConfig(source_format=types.GenotypeSourceFormat.PLINK, source_path=Path(bed_prefix))


def build_bgen_source_config(bgen_path: Path | str, sample_path: Path | str | None = None) -> GenotypeSourceConfig:
    """Build a genotype source config for a BGEN file."""
    return GenotypeSourceConfig(
        source_format=types.GenotypeSourceFormat.BGEN,
        source_path=Path(bgen_path),
        sample_path=Path(sample_path) if sample_path is not None else None,
    )


def resolve_genotype_source_config(
    bfile: Path | str | None,
    bgen: Path | str | None,
    sample: Path | str | None = None,
) -> GenotypeSourceConfig:
    """Resolve the requested genotype source from public API arguments."""
    if (bfile is None) == (bgen is None):
        message = "Exactly one genotype source must be provided via bfile or bgen."
        raise ValueError(message)
    if bfile is not None:
        if sample is not None:
            message = "A BGEN sample file can only be provided together with `bgen`."
            raise ValueError(message)
        return build_plink_source_config(bfile)
    assert bgen is not None
    return build_bgen_source_config(bgen, sample_path=sample)


def validate_genotype_source_config(genotype_source_config: GenotypeSourceConfig) -> None:
    """Validate that a genotype source config uses a supported format."""
    if not isinstance(genotype_source_config.source_format, types.GenotypeSourceFormat):
        message = (
            f"Unsupported genotype source format '{genotype_source_config.source_format}'. "
            f"Expected one of {[source_format.value for source_format in types.GenotypeSourceFormat]}."
        )
        raise ValueError(message)
    if (
        genotype_source_config.source_format != types.GenotypeSourceFormat.BGEN
        and genotype_source_config.sample_path is not None
    ):
        message = "Only BGEN source configs may include an explicit sample file."
        raise ValueError(message)


def build_genotype_source_signature_paths(genotype_source_config: GenotypeSourceConfig) -> tuple[Path, ...]:
    """Return the input files that define reproducibility for one source."""
    validate_genotype_source_config(genotype_source_config)
    if genotype_source_config.source_format == types.GenotypeSourceFormat.PLINK:
        return (
            genotype_source_config.source_path.with_suffix(".bed"),
            genotype_source_config.source_path.with_suffix(".bim"),
            genotype_source_config.source_path.with_suffix(".fam"),
        )
    resolved_sample_path = resolve_bgen_sample_path(
        genotype_source_config.source_path,
        genotype_source_config.sample_path,
    )
    if resolved_sample_path is None:
        return (genotype_source_config.source_path,)
    return (genotype_source_config.source_path, resolved_sample_path)


def open_genotype_reader(genotype_source_config: GenotypeSourceConfig) -> reader.GenotypeReader:
    """Open a concrete reader for one genotype source config."""
    validate_genotype_source_config(genotype_source_config)
    if genotype_source_config.source_format == types.GenotypeSourceFormat.PLINK:
        return plink.PlinkReader(genotype_source_config.source_path)
    return bgen.BgenReader(
        genotype_source_config.source_path,
        sample_path=genotype_source_config.sample_path,
    )


def load_aligned_sample_data_from_source(
    genotype_source_config: GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    *,
    is_binary_trait: bool,
    genotype_reader: reader.GenotypeReader | None = None,
) -> models.AlignedSampleData:
    """Load aligned sample data for any supported genotype source."""
    validate_genotype_source_config(genotype_source_config)
    if genotype_source_config.source_format == types.GenotypeSourceFormat.PLINK:
        return load_aligned_sample_data(
            bed_prefix=genotype_source_config.source_path,
            phenotype_path=phenotype_path,
            phenotype_name=phenotype_name,
            covariate_path=covariate_path,
            covariate_names=covariate_names,
            is_binary_trait=is_binary_trait,
        )
    if genotype_reader is not None:
        if genotype_source_config.sample_path is not None:
            sample_table = load_bgen_sample_table(
                genotype_source_config.source_path,
                genotype_source_config.sample_path,
            )
            return load_aligned_sample_data_from_individual_identifier_table(
                sample_table=sample_table,
                phenotype_path=phenotype_path,
                phenotype_name=phenotype_name,
                covariate_path=covariate_path,
                covariate_names=covariate_names,
                is_binary_trait=is_binary_trait,
            )
        sample_identifier_source = getattr(
            genotype_reader,
            "sample_identifier_source",
            types.SampleIdentifierSource.EMBEDDED,
        )
        if sample_identifier_source == types.SampleIdentifierSource.GENERATED:
            message = "BGEN file does not contain samples and no .sample file was found."
            raise ValueError(message)
        sample_table = build_sample_identifier_table(genotype_reader.samples)
    else:
        sample_table = load_bgen_sample_table(
            genotype_source_config.source_path,
            genotype_source_config.sample_path,
        )
    return load_aligned_sample_data_from_individual_identifier_table(
        sample_table=sample_table,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=is_binary_trait,
    )
