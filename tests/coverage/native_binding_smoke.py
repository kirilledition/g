"""Exercise every supported PyO3 binding file under LLVM instrumentation."""

from __future__ import annotations

import dataclasses
import struct
import tempfile
from pathlib import Path

import g._core


@dataclasses.dataclass(frozen=True)
class NativeInputPaths:
    """Paths for the complete tiny native lifecycle fixture."""

    bgen: Path
    sample: Path
    phenotype: Path
    prediction_list: Path
    loco: Path
    output: Path


def bgen_string(value: str) -> bytes:
    """Encode one uint16-length BGEN identifier."""
    encoded_value = value.encode("utf-8")
    return struct.pack("<H", len(encoded_value)) + encoded_value


def single_variant_bgen() -> bytes:
    """Build a valid uncompressed Layout 2 BGEN with three samples."""
    probability_block = b"".join(
        [
            struct.pack("<I", 3),
            struct.pack("<H", 2),
            bytes([2, 2]),
            bytes([2, 2, 2]),
            bytes([0, 8]),
            bytes([0, 0, 255, 0, 0, 255]),
        ],
    )
    variant_block = b"".join(
        [
            bgen_string("variant-1"),
            bgen_string("rs-1"),
            bgen_string("22"),
            struct.pack("<I", 1),
            struct.pack("<H", 2),
            struct.pack("<I", 1),
            b"A",
            struct.pack("<I", 1),
            b"G",
            struct.pack("<I", len(probability_block)),
            probability_block,
        ],
    )
    header = b"".join(
        [
            struct.pack("<I", 20),
            struct.pack("<I", 20),
            struct.pack("<I", 1),
            struct.pack("<I", 3),
            b"bgen",
            struct.pack("<I", 2 << 2),
        ],
    )
    return header + variant_block


def write_native_inputs(root_path: Path) -> NativeInputPaths:
    """Write a complete tiny CPU lifecycle fixture."""
    paths = NativeInputPaths(
        bgen=root_path / "genotypes.bgen",
        sample=root_path / "samples.sample",
        phenotype=root_path / "phenotypes.tsv",
        prediction_list=root_path / "predictions.list",
        loco=root_path / "predictions.loco",
        output=root_path / "output",
    )
    paths.bgen.write_bytes(single_variant_bgen())
    paths.sample.write_text(
        "ID_1 ID_2\n0 0\nfamily-1 individual-1\nfamily-2 individual-2\nfamily-3 individual-3\n",
        encoding="utf-8",
    )
    paths.phenotype.write_text(
        "FID\tIID\tcoverage-trait\nfamily-1\tindividual-1\t1\nfamily-2\tindividual-2\t2\nfamily-3\tindividual-3\t4\n",
        encoding="utf-8",
    )
    paths.loco.write_text(
        "FID_IID family-1_individual-1 family-2_individual-2 family-3_individual-3\n22 0 0 0\n",
        encoding="utf-8",
    )
    paths.prediction_list.write_text("coverage-trait predictions.loco\n", encoding="utf-8")
    return paths


def exercise_native_bindings() -> None:
    """Run far enough through the native CPU lifecycle to reach every binding."""
    with tempfile.TemporaryDirectory(prefix="g-binding-coverage-") as temporary_directory:
        root_path = Path(temporary_directory)
        paths = write_native_inputs(root_path)
        result = g._core.cli.run(
            [
                "regenie",
                "--bgen",
                str(paths.bgen),
                "--sample",
                str(paths.sample),
                "--phenoFile",
                str(paths.phenotype),
                "--phenoCol",
                "coverage-trait",
                "--pred",
                str(paths.prediction_list),
                "--qt",
                "--out",
                str(paths.output),
            ],
        )
        stderr_text = "".join(result.stderr_chunks)
        stdout_text = "".join(result.stdout_chunks)
        if result.exit_code != 0:
            raise AssertionError(f"Native binding lifecycle failed with exit code {result.exit_code}:\n{stderr_text}")
        if "Success. Run saved to" not in stdout_text:
            raise AssertionError(f"Native binding lifecycle produced unexpected stdout:\n{stdout_text}")
        if stderr_text:
            raise AssertionError(f"Native binding lifecycle produced unexpected stderr:\n{stderr_text}")


def main() -> None:
    """Run the binding coverage smoke contract."""
    exercise_native_bindings()
    print("Completed the tiny supported PyO3 CPU lifecycle.")


if __name__ == "__main__":
    main()
