"""Check the real container's synthetic quantitative and binary output contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet

from tooling.workbench import demo, outputs


def validate_demo(directory: Path) -> None:
    """Require two committed jobs and exactly preserved missing-call counts."""
    expected_counts = {
        f"synthetic_{index:04d}": demo.SAMPLE_COUNT - (10 if index % 4 == 0 else 0)
        for index in range(demo.VARIANT_COUNT)
    }
    for phenotype in ("quantitative", "binary"):
        markers = list((directory / "published" / phenotype).glob("*/COMMITTED.json"))
        if len(markers) != 1:
            raise ValueError(f"Expected exactly one committed {phenotype} attempt.")
        if outputs.read_json_object(markers[0]).get("status") != "completed":
            raise ValueError(f"The {phenotype} attempt is not complete.")
        observed_counts: dict[str, int] = {}
        for part in sorted((markers[0].parent / "outputs").rglob("*.parquet")):
            table = pyarrow.parquet.ParquetFile(part).read(columns=["ID", "N"])
            identifiers = table.column("ID").to_pylist()
            sample_counts = table.column("N").to_pylist()
            for identifier, sample_count in zip(identifiers, sample_counts, strict=True):
                if not isinstance(identifier, str) or not isinstance(sample_count, int):
                    raise ValueError("Synthetic variant identity or sample count has the wrong type.")
                if identifier in observed_counts:
                    raise ValueError(f"Duplicate synthetic variant: {identifier}.")
                observed_counts[identifier] = sample_count
        if observed_counts != expected_counts:
            raise ValueError(f"The {phenotype} outputs changed variant identities, rows or missing-call N counts.")
    print("Both synthetic jobs committed 32 variants; missing-call N=118 and complete-call N=128 are exact.")


def main() -> None:
    """Validate one participant-free integration bundle."""
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m deploy.workbench.check_demo DEMO_DIRECTORY")
    validate_demo(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
