from __future__ import annotations

import copy
import json
import typing
from pathlib import Path

import pytest

from deploy.workbench import localize_wdl
from tooling.workbench import manifest


def localization_fixture(directory: Path) -> dict[str, manifest.JsonValue]:
    filenames = ["input 'quoted'.bgen", "samples.sample", "phenotypes.tsv", "trait.loco"]
    paths = []
    for filename in filenames:
        path = directory / filename
        path.write_bytes(b"fixture")
        paths.append(str(path))
    source: dict[str, manifest.JsonValue] = {
        "schema_version": 1,
        "dataset": {"name": "synthetic", "release": "test", "reference_build": "GRCh38"},
        "inputs": {
            name: {"uri": f"gs://test-bucket/{name}", "sha256": "a" * 64, "size_bytes": 7, "generation": "123"}
            for name in ("bgen", "sample", "pheno_file")
        },
        "predictions": [
            {
                "phenotype": "trait",
                "uri": "gs://test-bucket/predictions",
                "sha256": "b" * 64,
                "size_bytes": 7,
                "generation": "456",
            },
        ],
        "analysis": {"trait_type": "quantitative", "phenotype_columns": ["trait"], "covariate_columns": []},
        "resources": {
            "device": "cpu",
            "cpu_threads": 2,
            "writer_threads": 1,
            "memory_gib": 4,
            "memory_fraction": 0.5,
            "max_chunk_variants": 100,
        },
        "output_uri": "gs://test-bucket/results",
    }
    source_path = directory / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    return {
        "manifest": str(source_path),
        "inputs": {"bgen": paths[0], "sample": paths[1], "pheno_file": paths[2], "covar_file": None},
        "predictions": [paths[3]],
        "cpu_threads": 2,
        "memory_gib": 4,
    }


def run_localization(directory: Path, bindings: dict[str, manifest.JsonValue]) -> Path:
    bindings_path = directory / "bindings.json"
    bindings_path.write_text(json.dumps(bindings), encoding="utf-8")
    destination = directory / "job.local.json"
    localize_wdl.localize_job(bindings_path, destination, directory / "published")
    return destination


def test_wdl_localization_preserves_pinned_content_and_source(tmp_path: Path) -> None:
    bindings = localization_fixture(tmp_path)
    source_path = Path(str(bindings["manifest"]))
    source_bytes = source_path.read_bytes()
    localized = manifest.load_manifest(run_localization(tmp_path, bindings))
    assert localized.inputs["bgen"].uri == str(tmp_path / "input 'quoted'.bgen")
    assert localized.inputs["bgen"].sha256 == "a" * 64
    assert localized.inputs["bgen"].size_bytes == 7
    assert localized.inputs["bgen"].generation is None
    assert localized.predictions[0].file.sha256 == "b" * 64
    assert localized.predictions[0].file.generation is None
    assert localized.output_uri == str(tmp_path / "published")
    assert source_path.read_bytes() == source_bytes


@pytest.mark.parametrize("reservation", ["cpu_threads", "memory_gib"])
def test_wdl_rejects_under_reserved_resources(tmp_path: Path, reservation: str) -> None:
    bindings = localization_fixture(tmp_path)
    bindings[reservation] = 1
    with pytest.raises(ValueError, match="exceeds"):
        run_localization(tmp_path, bindings)
    assert not (tmp_path / "job.local.json").exists()


def test_wdl_rejects_gpu_without_explicit_backend_support(tmp_path: Path) -> None:
    bindings = localization_fixture(tmp_path)
    source_path = Path(str(bindings["manifest"]))
    source = typing.cast("dict[str, manifest.JsonValue]", json.loads(source_path.read_text()))
    resources = manifest.mapping(source["resources"], "resources")
    resources["device"] = "gpu"
    resources["gpu_memory_gib"] = 16
    source_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="CPU only"):
        run_localization(tmp_path, bindings)


@pytest.mark.parametrize("mismatch", ["missing_input", "extra_input", "missing_prediction"])
def test_wdl_rejects_incomplete_or_extra_localization(tmp_path: Path, mismatch: str) -> None:
    bindings = copy.deepcopy(localization_fixture(tmp_path))
    inputs = manifest.mapping(bindings["inputs"], "inputs")
    if mismatch == "missing_input":
        del inputs["sample"]
    elif mismatch == "extra_input":
        inputs["extra"] = inputs["sample"]
    else:
        bindings["predictions"] = []
    with pytest.raises(ValueError, match=r"match every|one prediction"):
        run_localization(tmp_path, bindings)


def test_wdl_refuses_manifest_overwrite(tmp_path: Path) -> None:
    bindings = localization_fixture(tmp_path)
    destination = run_localization(tmp_path, bindings)
    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        run_localization(tmp_path, bindings)
    assert destination.read_bytes() == original
