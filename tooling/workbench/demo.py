"""Create a deterministic, participant-free Workbench integration fixture."""

from __future__ import annotations

import hashlib
import json
import struct
import typing
import zlib
from dataclasses import dataclass

from tooling.workbench import manifest

if typing.TYPE_CHECKING:
    from pathlib import Path

SAMPLE_COUNT = 128
VARIANT_COUNT = 32


@dataclass(frozen=True)
class DemoBundle:
    """Paths to independent quantitative and binary integration jobs."""

    quantitative_manifest: Path
    binary_manifest: Path


def encode_short_text(value: str) -> bytes:
    """Encode one short BGEN identifying string."""
    encoded = value.encode("ascii")
    return struct.pack("<H", len(encoded)) + encoded


def encode_variant(variant_index: int) -> bytes:
    """Encode one diploid zlib record, retaining deliberate missing calls."""
    identifier = f"synthetic_{variant_index:04d}"
    metadata = b"".join(encode_short_text(value) for value in (identifier, identifier, "22"))
    metadata += struct.pack("<IH", 100_000 + variant_index, 2)
    metadata += struct.pack("<I", 1) + b"A" + struct.pack("<I", 1) + b"G"
    ploidies = bytearray()
    probabilities = bytearray()
    for sample_index in range(SAMPLE_COUNT):
        missing = variant_index % 4 == 0 and sample_index % 13 == 0
        ploidies.append(0x82 if missing else 2)
        genotype = (sample_index * 17 + variant_index * 11 + (sample_index // 3) * (variant_index % 5 + 1)) % 3
        probabilities.extend((255 if genotype == 0 and not missing else 0, 255 if genotype == 1 and not missing else 0))
    payload = struct.pack("<IHBB", SAMPLE_COUNT, 2, 2, 2) + ploidies + bytes((0, 8)) + probabilities
    compressed = zlib.compress(payload)
    return metadata + struct.pack("<II", len(compressed) + 4, len(payload)) + compressed


def describe_file(path: Path) -> dict[str, manifest.JsonValue]:
    """Pin one small synthetic file by absolute location, length and digest."""
    content = path.read_bytes()
    return {"uri": str(path), "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def write_job(directory: Path, trait_type: manifest.TraitType) -> Path:
    """Write and validate a CPU job with explicit synthetic provenance."""
    phenotype = "quantitative" if trait_type == manifest.TraitType.QUANTITATIVE else "binary"
    prediction = {"phenotype": phenotype, **describe_file(directory / "synthetic.loco")}
    job: dict[str, manifest.JsonValue] = {
        "schema_version": 1,
        "dataset": {"name": "synthetic-integration", "release": "fixture-v1", "reference_build": "synthetic"},
        "inputs": {
            "bgen": describe_file(directory / "synthetic.bgen"),
            "sample": describe_file(directory / "synthetic.sample"),
            "pheno_file": describe_file(directory / "phenotypes.tsv"),
            "covar_file": describe_file(directory / "covariates.tsv"),
        },
        "predictions": [prediction],
        "analysis": {
            "trait_type": trait_type.value,
            "phenotype_columns": [phenotype],
            "covariate_columns": ["age"],
        },
        "resources": {
            "device": "cpu",
            "cpu_threads": 2,
            "writer_threads": 1,
            "memory_gib": 4,
            "memory_fraction": 0.5,
            "max_chunk_variants": 16,
        },
        "output_uri": str(directory / "published" / phenotype),
    }
    if trait_type == manifest.TraitType.BINARY:
        analysis = job["analysis"]
        analysis["binary"] = {"fallback_method": "firth_approximate", "p_threshold": 0.999999}
    manifest.parse_manifest(job)
    destination = directory / f"{phenotype}.job.json"
    destination.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    return destination


def create_demo_bundle(directory: Path) -> DemoBundle:
    """Create a new synthetic job bundle without downloading any data.

    Args:
        directory: New directory to create; existing destinations are refused.

    Returns:
        Paths to byte-pinned quantitative and binary job manifests.

    Raises:
        FileExistsError: The destination already exists.

    """
    directory = directory.absolute()
    directory.mkdir(parents=True, exist_ok=False)
    header = struct.pack("<IIII4sI", 20, 20, VARIANT_COUNT, SAMPLE_COUNT, b"bgen", 9)
    (directory / "synthetic.bgen").write_bytes(
        header + b"".join(encode_variant(index) for index in range(VARIANT_COUNT))
    )
    sample_rows = [f"family{index:04d} sample{index:04d}" for index in range(SAMPLE_COUNT)]
    (directory / "synthetic.sample").write_text("ID_1 ID_2\n0 0\n" + "\n".join(sample_rows) + "\n", encoding="utf-8")
    phenotype_rows = ["FID\tIID\tquantitative\tbinary"]
    covariate_rows = ["FID\tIID\tage"]
    for index in range(SAMPLE_COUNT):
        identifiers = f"family{index:04d}\tsample{index:04d}"
        age = 20 + (index * 7) % 63
        quantitative = age * 0.13 + ((index * 19) % 23) / 7 + (index % 3) * 0.4
        binary = 2 if index % 5 < 2 else 1
        phenotype_rows.append(f"{identifiers}\t{quantitative:.12g}\t{binary}")
        covariate_rows.append(f"{identifiers}\t{age}")
    (directory / "phenotypes.tsv").write_text("\n".join(phenotype_rows) + "\n", encoding="utf-8")
    (directory / "covariates.tsv").write_text("\n".join(covariate_rows) + "\n", encoding="utf-8")
    prediction_header = "FID_IID " + " ".join(f"family{index:04d}_sample{index:04d}" for index in range(SAMPLE_COUNT))
    (directory / "synthetic.loco").write_text(
        prediction_header
        + "\n"
        + "".join(f"{chromosome} " + " ".join("0" for _ in sample_rows) + "\n" for chromosome in range(1, 23)),
        encoding="utf-8",
    )
    (directory / "README.txt").write_text(
        "Synthetic integration fixture: no real participants or genomic observations.\n"
        "Zero LOCO values test the execution contract only; they are not trained Step 1 predictions.\n"
        "The binary fixture uses p_threshold=0.999999 to exercise approximate Firth correction deliberately.\n"
        "The 128 samples and 32 autosomal diploid variants include deliberate missing calls.\n"
        "This fixture does not qualify All of Us compatibility, scientific calibration or cohort-scale capacity.\n"
        "Manifests contain absolute paths; regenerate the bundle after moving to another filesystem.\n",
        encoding="utf-8",
    )
    return DemoBundle(
        quantitative_manifest=write_job(directory, manifest.TraitType.QUANTITATIVE),
        binary_manifest=write_job(directory, manifest.TraitType.BINARY),
    )
