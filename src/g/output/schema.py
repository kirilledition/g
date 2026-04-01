"""Explicit schema definitions for chunked association output."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_VERSION: Final[str] = "1"

LINEAR_OUTPUT_SCHEMA: Final[dict[str, pl.DataType]] = {
    "chunk_identifier": pl.Int64(),
    "variant_start_index": pl.Int64(),
    "variant_stop_index": pl.Int64(),
    "chromosome": pl.String(),
    "position": pl.Int64(),
    "variant_identifier": pl.String(),
    "allele_one": pl.String(),
    "allele_two": pl.String(),
    "allele_one_frequency": pl.Float32(),
    "observation_count": pl.Int32(),
    "beta": pl.Float32(),
    "standard_error": pl.Float32(),
    "t_statistic": pl.Float32(),
    "p_value": pl.Float32(),
    "is_valid": pl.Boolean(),
}

LOGISTIC_OUTPUT_SCHEMA: Final[dict[str, pl.DataType]] = {
    "chunk_identifier": pl.Int64(),
    "variant_start_index": pl.Int64(),
    "variant_stop_index": pl.Int64(),
    "chromosome": pl.String(),
    "position": pl.Int64(),
    "variant_identifier": pl.String(),
    "allele_one": pl.String(),
    "allele_two": pl.String(),
    "allele_one_frequency": pl.Float32(),
    "observation_count": pl.Int32(),
    "beta": pl.Float32(),
    "standard_error": pl.Float32(),
    "z_statistic": pl.Float32(),
    "p_value": pl.Float32(),
    "firth_flag": pl.String(),
    "error_code": pl.String(),
    "converged": pl.Boolean(),
    "iteration_count": pl.Int32(),
    "is_valid": pl.Boolean(),
}


def get_output_schema(association_mode: str) -> dict[str, pl.DataType]:
    """Return the fixed output schema for the requested mode."""
    if association_mode == "linear":
        return LINEAR_OUTPUT_SCHEMA
    if association_mode == "logistic":
        return LOGISTIC_OUTPUT_SCHEMA
    message = f"Unsupported association mode '{association_mode}'."
    raise ValueError(message)


def write_schema_file(schema_path: Path, association_mode: str) -> None:
    """Persist schema metadata in JSON format for run reproducibility."""
    schema = get_output_schema(association_mode)
    serialized_schema = {
        "schema_version": SCHEMA_VERSION,
        "association_mode": association_mode,
        "columns": [{"name": name, "dtype": str(dtype)} for name, dtype in schema.items()],
    }
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(serialized_schema, indent=2), encoding="utf-8")


def cast_frame_to_schema(data_frame: pl.DataFrame, association_mode: str) -> pl.DataFrame:
    """Cast an output frame to the fixed mode-specific schema."""
    schema = get_output_schema(association_mode)
    return data_frame.select(
        [pl.col(column_name).cast(column_type).alias(column_name) for column_name, column_type in schema.items()]
    )
