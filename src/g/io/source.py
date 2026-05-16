"""BGEN genotype source orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from g import types
from g.io import bgen, models, reader, samples


@dataclass(frozen=True)
class GenotypeSourceConfig:
    """Configuration describing one BGEN input source."""

    source_path: Path
    sample_path: Path | None = None


resolve_bgen_sample_path = bgen.resolve_bgen_sample_path
load_bgen_sample_table = bgen.load_bgen_sample_table
build_sample_identifier_table = bgen.build_sample_identifier_table
load_aligned_sample_data_from_individual_identifier_table = (
    samples.load_aligned_sample_data_from_individual_identifier_table
)


def build_bgen_source_config(bgen_path: Path | str, sample_path: Path | str | None = None) -> GenotypeSourceConfig:
    """Build a genotype source config for a BGEN file."""
    return GenotypeSourceConfig(
        source_path=Path(bgen_path),
        sample_path=Path(sample_path) if sample_path is not None else None,
    )


def resolve_genotype_source_config(
    bgen: Path | str | None,
    sample: Path | str | None = None,
) -> GenotypeSourceConfig:
    """Resolve the requested BGEN source from public API arguments."""
    if bgen is None:
        message = "A BGEN source must be provided via bgen."
        raise ValueError(message)
    return build_bgen_source_config(bgen, sample_path=sample)


def validate_genotype_source_config(genotype_source_config: GenotypeSourceConfig) -> None:
    """Validate a BGEN source config."""
    if genotype_source_config.source_path.suffix != ".bgen":
        message = f"Expected a .bgen source path, found '{genotype_source_config.source_path}'."
        raise ValueError(message)


def build_genotype_source_signature_paths(genotype_source_config: GenotypeSourceConfig) -> tuple[Path, ...]:
    """Return the input files that define reproducibility for one source."""
    validate_genotype_source_config(genotype_source_config)
    resolved_sample_path = resolve_bgen_sample_path(
        genotype_source_config.source_path,
        genotype_source_config.sample_path,
    )
    if resolved_sample_path is None:
        return (genotype_source_config.source_path,)
    return (genotype_source_config.source_path, resolved_sample_path)


def open_genotype_reader(genotype_source_config: GenotypeSourceConfig) -> reader.GenotypeReader:
    """Open a BGEN reader for one genotype source config."""
    validate_genotype_source_config(genotype_source_config)
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
    """Load aligned sample data for a BGEN source."""
    validate_genotype_source_config(genotype_source_config)
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
