from __future__ import annotations

import json
from typing import TYPE_CHECKING

import polars as pl
import pytest

from g.output.schema import cast_frame_to_schema, get_output_schema, write_schema_file

if TYPE_CHECKING:
    from pathlib import Path


def test_get_output_schema_rejects_unknown_mode() -> None:
    """Ensure schema lookup fails fast for unsupported association modes."""
    with pytest.raises(ValueError, match="Unsupported association mode 'poisson'"):
        get_output_schema("poisson")


def test_get_output_schema_returns_logistic_columns() -> None:
    """Ensure schema lookup returns the logistic output schema."""
    logistic_schema = get_output_schema("logistic")

    assert "firth_flag" in logistic_schema
    assert "iteration_count" in logistic_schema


def test_write_schema_file_persists_json_metadata(tmp_path: Path) -> None:
    """Ensure schema files are written with stable metadata for reproducibility."""
    schema_path = tmp_path / "output" / "schema.arrow.json"

    write_schema_file(schema_path, "linear")

    serialized_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert serialized_schema["association_mode"] == "linear"
    assert serialized_schema["schema_version"] == "1"
    assert serialized_schema["columns"][0]["name"] == "chunk_identifier"


def test_cast_frame_to_schema_reorders_and_casts_columns() -> None:
    """Ensure output frames are cast and reordered to the fixed linear schema."""
    data_frame = pl.DataFrame(
        {
            "beta": [1.5],
            "chromosome": [1],
            "chunk_identifier": [7],
            "variant_start_index": [0],
            "variant_stop_index": [1],
            "position": [123],
            "variant_identifier": ["variant1"],
            "allele_one": ["A"],
            "allele_two": ["G"],
            "allele_one_frequency": [0.25],
            "observation_count": [100],
            "standard_error": [0.1],
            "t_statistic": [15.0],
            "p_value": [1.0e-5],
            "is_valid": [True],
        }
    )

    cast_data_frame = cast_frame_to_schema(data_frame, "linear")

    assert cast_data_frame.schema == get_output_schema("linear")
    assert cast_data_frame.columns == list(get_output_schema("linear"))
    assert cast_data_frame.get_column("chromosome").to_list() == ["1"]
