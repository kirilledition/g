"""Verify the synthetic handoff bundle retains identity and missingness."""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import pytest

from tooling.workbench import demo, manifest


def test_demo_jobs_pin_synthetic_sources_and_refuse_overwrite(tmp_path: Path) -> None:
    bundle = demo.create_demo_bundle(tmp_path / "fixture")
    for manifest_path in (bundle.quantitative_manifest, bundle.binary_manifest):
        job = manifest.load_manifest(manifest_path)
        assert job.dataset.name == "synthetic-integration"
        assert job.resources.device == manifest.Device.CPU
        assert job.analysis.covariate_columns == ("age",)
        for source in manifest.iter_input_files(job):
            content = Path(source.uri).read_bytes()
            assert len(content) == source.size_bytes
            assert hashlib.sha256(content).hexdigest() == source.sha256
        prediction_rows = Path(job.predictions[0].file.uri).read_text().splitlines()
        assert [int(row.split()[0]) for row in prediction_rows[1:]] == list(range(1, 23))
    with pytest.raises(FileExistsError):
        demo.create_demo_bundle(tmp_path / "fixture")


def test_demo_retains_missing_calls_and_lossless_probability_encoding(tmp_path: Path) -> None:
    bundle = demo.create_demo_bundle(tmp_path / "fixture")
    job = manifest.load_manifest(bundle.quantitative_manifest)
    content = Path(job.inputs["bgen"].uri).read_bytes()
    assert struct.unpack("<IIII4sI", content[:24]) == (20, 20, 32, 128, b"bgen", 9)
    position = 24
    for _ in range(3):
        length = struct.unpack_from("<H", content, position)[0]
        position += 2 + length
    position += 6
    for _ in range(2):
        length = struct.unpack_from("<I", content, position)[0]
        position += 4 + length
    compressed_length, decoded_length = struct.unpack_from("<II", content, position)
    decoded = zlib.decompress(content[position + 8 : position + 4 + compressed_length])
    assert len(decoded) == decoded_length
    ploidies = decoded[8 : 8 + demo.SAMPLE_COUNT]
    assert sum(value == 0x82 for value in ploidies) == 10
    assert set(ploidies) == {2, 0x82}
    assert set(decoded[10 + demo.SAMPLE_COUNT :]) <= {0, 255}
