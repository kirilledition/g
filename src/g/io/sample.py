"""Oxford sample-file parsing helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt


def split_sample_file_line(raw_line: str) -> list[str]:
    """Split one Oxford sample-file line on arbitrary whitespace."""
    return raw_line.strip().split()


def resolve_bgen_sample_path(bgen_path: Path, sample_path: Path | None = None) -> Path | None:
    """Resolve an explicit or adjacent Oxford sample file for one BGEN file."""
    if sample_path is not None:
        return sample_path
    adjacent_sample_path = bgen_path.with_suffix(".sample")
    return adjacent_sample_path if adjacent_sample_path.exists() else None


def build_sample_identifier_table(sample_identifiers: npt.NDArray[np.str_]) -> pl.DataFrame:
    """Build the normalized identifier table used by the alignment pipeline."""
    normalized_sample_identifiers = np.asarray(sample_identifiers, dtype=np.str_)
    return pl.DataFrame(
        {
            "family_identifier": normalized_sample_identifiers,
            "individual_identifier": normalized_sample_identifiers,
        }
    ).with_row_index("sample_index")


def load_sample_identifier_table(sample_path: Path) -> pl.DataFrame:
    """Load normalized identifiers from an Oxford sample file.

    Args:
        sample_path: Path to the sample file.

    Returns:
        A normalized table containing `sample_index`, `family_identifier`, and
        `individual_identifier` columns.

    Raises:
        ValueError: The file is malformed or missing mandatory identifier columns.

    """
    sample_lines = sample_path.read_text(encoding="utf-8").splitlines()
    non_empty_lines = [line for line in sample_lines if line.strip()]
    if len(non_empty_lines) < 2:
        message = f"Sample file '{sample_path}' must contain at least two header lines."
        raise ValueError(message)

    column_names = split_sample_file_line(non_empty_lines[0])
    column_types = split_sample_file_line(non_empty_lines[1])
    if len(column_names) != len(column_types):
        message = f"Sample file '{sample_path}' header and type lines have different column counts."
        raise ValueError(message)
    if not column_names:
        message = f"Sample file '{sample_path}' does not contain any columns."
        raise ValueError(message)
    if column_types[0] != "0":
        message = f"Sample file '{sample_path}' must mark the first identifier column with type '0'."
        raise ValueError(message)
    if "ID_2" in column_names and column_types[column_names.index("ID_2")] != "0":
        message = f"Sample file '{sample_path}' must mark 'ID_2' with type '0'."
        raise ValueError(message)

    data_rows: list[list[str]] = []
    for row_index, raw_line in enumerate(non_empty_lines[2:], start=3):
        row_values = split_sample_file_line(raw_line)
        if len(row_values) != len(column_names):
            message = (
                f"Sample file '{sample_path}' line {row_index} has {len(row_values)} values, "
                f"but the header declares {len(column_names)} columns."
            )
            raise ValueError(message)
        data_rows.append(row_values)

    if not data_rows:
        empty_identifiers = np.asarray([], dtype=np.str_)
        return build_sample_identifier_table(empty_identifiers)

    column_values = {
        column_name: [row_values[column_index] for row_values in data_rows]
        for column_index, column_name in enumerate(column_names)
    }
    sample_table = pl.DataFrame(column_values)
    family_identifier_column_name = column_names[0]
    individual_identifier_column_name = "ID_2" if "ID_2" in column_names else family_identifier_column_name
    return pl.DataFrame(
        {
            "family_identifier": sample_table.get_column(family_identifier_column_name).cast(pl.String),
            "individual_identifier": sample_table.get_column(individual_identifier_column_name).cast(pl.String),
        }
    ).with_row_index("sample_index")
