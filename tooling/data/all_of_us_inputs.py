"""Strict local identity and table readers for genomic cohort preparation."""

from __future__ import annotations

import csv
import dataclasses
import math
import re
import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from pathlib import Path

MISSING_VALUES = frozenset({"", "NA", "NaN", "nan", "-9"})
NUMBER_PATTERN = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
FLOAT32_MAXIMUM = 3.4028234663852886e38


@dataclass(frozen=True)
class SampleIdentity:
    """A full family and individual identity, without implicit matching."""

    family_id: str
    individual_id: str


@dataclass(frozen=True)
class IdentityMapping:
    """An explicit source identity and its analysis identity."""

    source: SampleIdentity
    target: SampleIdentity


@dataclass(frozen=True)
class TextTable:
    """A structurally validated tab-delimited table."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class AlignedTable:
    """Numeric analysis columns aligned to ordered cohort identities."""

    columns: tuple[str, ...]
    values: tuple[tuple[str, ...], ...]
    missing_counts: dict[str, int]


def validate_token(value: str, description: str) -> str:
    """Reject blank tokens and whitespace that external formats cannot retain."""
    if not value or any(character.isspace() for character in value) or "\x00" in value or '"' in value:
        raise ValueError(f"{description} must be nonempty and contain no whitespace, double quotes, or NUL bytes.")
    return value


def make_identity(family_id: str, individual_id: str) -> SampleIdentity:
    """Construct a validated full sample identity."""
    return SampleIdentity(
        family_id=validate_token(family_id, "FID"),
        individual_id=validate_token(individual_id, "IID"),
    )


def validate_unique_identities(identities: tuple[SampleIdentity, ...], description: str) -> None:
    """Reject duplicate pairs and ambiguous REGENIE LOCO header encodings."""
    if not identities:
        raise ValueError(f"{description} contains no sample identities.")
    if len(set(identities)) != len(identities):
        raise ValueError(f"{description} contains duplicate FID/IID pairs.")
    prediction_tokens = {f"{identity.family_id}_{identity.individual_id}" for identity in identities}
    if len(prediction_tokens) != len(identities):
        raise ValueError(f"{description} contains ambiguous FID_IID prediction tokens.")


def read_text_table(path: Path) -> TextTable:
    """Read a strict TSV, retaining explicitly empty numeric fields."""
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.reader(input_file, delimiter="\t", strict=True)
        columns = tuple(next(reader, ()))
        if not columns or len(set(columns)) != len(columns):
            raise ValueError(f"{path}: table header is empty or has duplicate columns.")
        for column in columns:
            validate_token(column, f"{path}: column name")
        rows: list[tuple[str, ...]] = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(columns):
                raise ValueError(f"{path}:{line_number}: expected {len(columns)} tab-delimited fields.")
            rows.append(tuple(row))
    return TextTable(columns=columns, rows=tuple(rows))


def require_columns(table: TextTable, columns: tuple[str, ...], path: Path) -> tuple[int, ...]:
    """Resolve required unique columns or fail with their source table path."""
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Selected columns must be nonempty and unique.")
    missing_columns = set(columns) - set(table.columns)
    if missing_columns:
        raise ValueError(f"{path}: required columns are absent: {', '.join(sorted(missing_columns))}.")
    return tuple(table.columns.index(column) for column in columns)


def read_identity_mapping(path: Path) -> tuple[IdentityMapping, ...]:
    """Read an explicit four-column source-to-analysis identity map."""
    table = read_text_table(path)
    indices = require_columns(table, ("source_FID", "source_IID", "FID", "IID"), path)
    mappings = tuple(
        IdentityMapping(
            source=make_identity(row[indices[0]], row[indices[1]]),
            target=make_identity(row[indices[2]], row[indices[3]]),
        )
        for row in table.rows
    )
    validate_unique_identities(tuple(mapping.source for mapping in mappings), "Source identity map")
    validate_unique_identities(tuple(mapping.target for mapping in mappings), "Analysis identity map")
    return mappings


def read_cohort_list(path: Path) -> frozenset[SampleIdentity]:
    """Read a headered canonical FID/IID list, including an empty exclusion list."""
    table = read_text_table(path)
    if table.columns[0] == "#FID":
        table = dataclasses.replace(table, columns=("FID", *table.columns[1:]))
    indices = require_columns(table, ("FID", "IID"), path)
    identities = tuple(make_identity(row[indices[0]], row[indices[1]]) for row in table.rows)
    if identities:
        validate_unique_identities(identities, str(path))
    return frozenset(identities)


def read_plink_samples(path: Path) -> tuple[SampleIdentity, ...]:
    """Read a PSAM, treating omitted FIDs as PLINK's explicit zero value."""
    with path.open(encoding="utf-8") as input_file:
        header = next(input_file, "").split()
        if not header or header[0] not in {"#FID", "#IID"}:
            raise ValueError(f"{path}: PSAM must start with #FID or #IID.")
        header[0] = header[0].removeprefix("#")
        if len(set(header)) != len(header) or "IID" not in header:
            raise ValueError(f"{path}: PSAM columns are ambiguous or IID is absent.")
        individual_column = header.index("IID")
        family_column = header.index("FID") if "FID" in header else None
        source_column = header.index("SID") if "SID" in header else None
        identities: list[SampleIdentity] = []
        for line_number, line in enumerate(input_file, start=2):
            fields = line.split()
            if len(fields) != len(header):
                raise ValueError(f"{path}:{line_number}: invalid PSAM row length.")
            if source_column is not None and fields[source_column] != "0":
                raise ValueError(f"{path}: nonzero SID identities require explicit upstream normalization.")
            identities.append(
                make_identity("0" if family_column is None else fields[family_column], fields[individual_column]),
            )
    result = tuple(identities)
    validate_unique_identities(result, str(path))
    return result


def read_family_samples(path: Path) -> tuple[SampleIdentity, ...]:
    """Read the ordered full identities from a standard six-column FAM."""
    identities: list[SampleIdentity] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"{path}:{line_number}: expected six FAM columns.")
            identities.append(make_identity(fields[0], fields[1]))
    result = tuple(identities)
    validate_unique_identities(result, str(path))
    return result


def read_oxford_samples(path: Path) -> tuple[SampleIdentity, ...]:
    """Read ordered Oxford IDs and require the engine's two-column contract."""
    with path.open(encoding="utf-8") as input_file:
        header = next(input_file, "").split()
        types = next(input_file, "").split()
        if len(set(header)) != len(header) or "ID_1" not in header or "ID_2" not in header:
            raise ValueError(f"{path}: Oxford sample header requires unique ID_1 and ID_2 columns.")
        family_column, individual_column = header.index("ID_1"), header.index("ID_2")
        if len(types) != len(header) or types[family_column] != "0" or types[individual_column] != "0":
            raise ValueError(f"{path}: Oxford ID columns must have type 0.")
        identities: list[SampleIdentity] = []
        for line_number, line in enumerate(input_file, start=3):
            fields = line.split()
            if len(fields) != len(header):
                raise ValueError(f"{path}:{line_number}: invalid Oxford sample row length.")
            identities.append(make_identity(fields[family_column], fields[individual_column]))
    result = tuple(identities)
    validate_unique_identities(result, str(path))
    return result


def validate_numeric_value(value: str, *, binary: bool, description: str) -> None:
    """Accept engine-compatible missing tokens and finite float32 numeric values."""
    if value in MISSING_VALUES:
        return
    if NUMBER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{description}: value is not a supported numeric or missing token.")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or abs(numeric_value) > FLOAT32_MAXIMUM:
        raise ValueError(f"{description}: value is not finite within float32 range.")
    if binary and numeric_value not in {1.0, 2.0}:
        raise ValueError(f"{description}: binary phenotypes require 1=control or 2=case.")


def align_numeric_table(
    path: Path,
    columns: tuple[str, ...],
    identities: tuple[SampleIdentity, ...],
    *,
    binary: bool,
) -> AlignedTable:
    """Validate numeric values and align every selected sample without imputation."""
    if set(columns) & {"FID", "IID"}:
        raise ValueError("Analysis columns cannot be identity columns.")
    for column in columns:
        validate_token(column, "Selected column")
        if column.startswith("-") or "," in column:
            raise ValueError("Analysis column names cannot start with '-' or contain commas.")
    table = read_text_table(path)
    identity_indices = require_columns(table, ("FID", "IID"), path)
    value_indices = require_columns(table, columns, path)
    values_by_identity: dict[SampleIdentity, tuple[str, ...]] = {}
    for line_number, row in enumerate(table.rows, start=2):
        identity = make_identity(row[identity_indices[0]], row[identity_indices[1]])
        if identity in values_by_identity:
            raise ValueError(f"{path}:{line_number}: duplicate FID/IID pair.")
        values = tuple(row[index] for index in value_indices)
        for column, value in zip(columns, values, strict=True):
            validate_numeric_value(value, binary=binary, description=f"{path}:{line_number}:{column}")
        values_by_identity[identity] = values
    if any(identity not in values_by_identity for identity in identities):
        raise ValueError(f"{path}: selected samples are absent; provide explicit NA rows instead.")
    values = tuple(values_by_identity[identity] for identity in identities)
    missing_counts = {
        column: sum(row[column_index] in MISSING_VALUES for row in values)
        for column_index, column in enumerate(columns)
    }
    return AlignedTable(columns=columns, values=values, missing_counts=missing_counts)


def write_tsv(path: Path, columns: tuple[str, ...], rows: typing.Iterable[tuple[str, ...]]) -> None:
    """Create a new TSV without replacing an existing file."""
    with path.open("x", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, quotechar=None)
        writer.writerow(columns)
        writer.writerows(rows)


def validate_variant_list(path: Path) -> None:
    """Require a nonempty, unique one-ID-per-line PLINK extraction list."""
    with path.open(encoding="utf-8") as input_file:
        variant_ids = tuple(line.rstrip("\r\n") for line in input_file)
    for variant_id in variant_ids:
        validate_token(variant_id, "Variant extraction ID")
    if not variant_ids or len(set(variant_ids)) != len(variant_ids) or "." in variant_ids:
        raise ValueError("Variant extraction list must contain nonempty unique variant IDs.")
