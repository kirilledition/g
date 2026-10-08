"""Portable runner contracts independent of controlled genomic resources."""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import hashlib
import json
import os
import signal
import struct
import subprocess
import sys
import time
import typing
from pathlib import Path

import pytest

from tooling.workbench import manifest, preflight, runner, storage


def source_record(path: Path) -> dict[str, manifest.JsonValue]:
    content = path.read_bytes()
    return {"uri": str(path), "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def job_payload(root: Path) -> dict[str, manifest.JsonValue]:
    root.mkdir(parents=True, exist_ok=True)
    bgen = root / "input.bgen"
    bgen.write_bytes(struct.pack("<IIII4sI", 20, 20, 2, 8, b"bgen", 8) + b"records")
    sample = root / "input.sample"
    sample.write_text("ID_1 ID_2 missing\n0 0 0\n" + "".join(f"family{index} person{index} 0\n" for index in range(8)))
    phenotype = root / "phenotype.tsv"
    phenotype.write_text(
        "FID\tIID\ttrait\n" + "".join(f"family{index}\tperson{index}\t{index / 4}\n" for index in range(8))
    )
    covariate = root / "covariates.tsv"
    covariate.write_text(
        "FID\tIID\tage\n" + "".join(f"family{index}\tperson{index}\t{20 + index}\n" for index in range(8))
    )
    prediction = root / "step1.loco"
    prediction.write_text(
        "FID_IID "
        + " ".join(f"family{index}_person{index}" for index in range(8))
        + "\n22 "
        + " ".join(["0"] * 8)
        + "\n"
    )
    return {
        "schema_version": 1,
        "dataset": {"name": "synthetic", "release": "test-v1", "reference_build": "GRCh38"},
        "inputs": {
            "bgen": source_record(bgen),
            "sample": source_record(sample),
            "pheno_file": source_record(phenotype),
            "covar_file": source_record(covariate),
        },
        "predictions": [{"phenotype": "trait", **source_record(prediction)}],
        "analysis": {"trait_type": "quantitative", "phenotype_columns": ["trait"], "covariate_columns": ["age"]},
        "resources": {
            "device": "cpu",
            "cpu_threads": 1,
            "writer_threads": 1,
            "memory_gib": 8,
            "memory_fraction": 0.5,
            "max_chunk_variants": 16384,
        },
        "output_uri": str(root / "published"),
    }


def job_arguments(
    root: Path, payload: dict[str, manifest.JsonValue], prefix: tuple[str, ...]
) -> runner.RunnerArguments:
    manifest_path = root / "job.json"
    manifest_path.write_text(json.dumps(payload))
    return runner.RunnerArguments(
        action=runner.Action.RUN,
        manifest_path=manifest_path,
        work_directory=root / "work with spaces",
        dry_run=False,
        billing_project=None,
        runner_prefix=prefix,
        profile_max_variants=8,
    )


def fake_engine(root: Path, mode: str) -> tuple[str, ...]:
    path = root / "fake_engine.py"
    path.write_text(
        """import json
import os
import signal
import pathlib
import sys
import tomllib
import pyarrow
import pyarrow.parquet

configuration = tomllib.loads(pathlib.Path(sys.argv[-1]).read_text())
output = pathlib.Path(configuration["output"]["output_run_directory"])
mode = sys.argv[1]
if mode == "exit_failure":
    output.mkdir()
    (output / "preserved-evidence.txt").write_text("failed")
    raise SystemExit(7)
if mode == "mutate_input":
    pathlib.Path(configuration["input"]["bgen"]).write_bytes(b"changed")
if mode == "no_outputs":
    raise SystemExit(0)
for phenotype in configuration["input"]["pheno_columns"]:
    run = output / (phenotype + ".run")
    parts = run / "parts"
    parts.mkdir(parents=True)
    commits = [{"chunk_identifier": 0, "variant_start_index": 0, "variant_stop_index": 2,
                "row_count": 2, "chunk_file_name": "part.parquet"}]
    table = pyarrow.table({"GENPOS": [1, 2]}).replace_schema_metadata(
        {b"g.output.chunk_commits": json.dumps(commits).encode()})
    pyarrow.parquet.write_table(table, parts / "part.parquet")
    (run / "run_manifest.json").write_text(json.dumps({"status": "completed",
        "execution_plan": {"phenotype_name": phenotype, "variant_count": 2}, "committed_chunks": commits}))
"""
    )
    return (sys.executable, str(path), mode)


def test_manifest_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    path = tmp_path / "job.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        manifest.load_manifest(path)


@pytest.mark.parametrize(
    "uri",
    ["relative.bgen", "/work/../secret", "gs://bucket/../x", "gs://bucket/a*", "gs://bucket/a#1", "gs://bucket/a#"],
)
def test_input_uri_rejects_traversal_and_wildcards(uri: str) -> None:
    with pytest.raises(ValueError):
        manifest.validate_uri(uri, "test")


@pytest.mark.parametrize("field,value", [("cpu_threads", True), ("writer_threads", 0), ("memory_fraction", 0.95)])
def test_manifest_rejects_unsafe_resources(tmp_path: Path, field: str, value: manifest.JsonValue) -> None:
    payload = job_payload(tmp_path)
    resources = manifest.mapping(payload["resources"], "resources")
    resources[field] = value
    with pytest.raises(ValueError):
        manifest.parse_manifest(payload)


def test_manifest_requires_exact_prediction_coverage(tmp_path: Path) -> None:
    payload = job_payload(tmp_path)
    payload["predictions"] = []
    with pytest.raises(ValueError, match="exactly one"):
        manifest.parse_manifest(payload)


def test_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = job_payload(tmp_path)
    payload["resoruces"] = payload["resources"]
    with pytest.raises(ValueError, match="unknown fields"):
        manifest.parse_manifest(payload)


def test_manifest_rejects_invalid_generation_for_local_input(tmp_path: Path) -> None:
    source: dict[str, manifest.JsonValue] = {"uri": "/work/x", "sha256": "0" * 64, "size_bytes": 1}
    source["generation"] = "123"
    with pytest.raises(ValueError, match="generation"):
        manifest.input_file(source, "input")


def test_preflight_validates_content_alignment_and_no_runtime_import(tmp_path: Path) -> None:
    payload = job_payload(tmp_path)
    report = preflight.run_preflight(
        manifest.parse_manifest(payload), tmp_path / "work", dry_run=False, billing_project=None
    )
    assert report.all_input_content_verified
    assert report.alignment is not None
    assert report.alignment.selected_sample_counts == {"trait": 8}
    assert report.capacity is not None and report.capacity.chunk_variants == 2
    assert not report.capacity.out_of_memory_guarantee
    assert "jax" not in sys.modules or "jax" not in runner.__dict__


def test_preflight_rejects_changed_bytes_before_engine(tmp_path: Path) -> None:
    job = manifest.parse_manifest(job_payload(tmp_path))
    Path(job.inputs["bgen"].uri).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum or byte size"):
        preflight.run_preflight(job, tmp_path, dry_run=False, billing_project=None)


def test_preflight_detects_sample_count_mismatch(tmp_path: Path) -> None:
    payload = job_payload(tmp_path)
    inputs = manifest.mapping(payload["inputs"], "inputs")
    sample = tmp_path / "input.sample"
    sample.write_text(sample.read_text().replace("family7 person7 0\n", ""))
    inputs["sample"] = source_record(sample)
    with pytest.raises(ValueError, match="sample count"):
        preflight.run_preflight(manifest.parse_manifest(payload), tmp_path, dry_run=False, billing_project=None)


def test_loco_missing_selected_sample_fails(tmp_path: Path) -> None:
    path = tmp_path / "pred.loco"
    path.write_text("FID_IID other_person\n22 0\n")
    with pytest.raises(ValueError, match="missing selected"):
        preflight.validate_loco(path, {preflight.profile.SampleIdentifier("family", "person")})


def test_capacity_reduces_width_preserving_full_sample_count(tmp_path: Path) -> None:
    job = manifest.parse_manifest(job_payload(tmp_path))
    plan = preflight.plan_capacity(job, sample_count=400_000, variant_count=100_000)
    assert 1 <= plan.chunk_variants < 16384
    assert plan.bytes_per_variant_estimate == 400_000 * 4 * 10
    assert plan.estimated_chunk_working_set_bytes <= plan.working_memory_budget_bytes


def test_capacity_rejects_insufficient_declared_memory(tmp_path: Path) -> None:
    payload = job_payload(tmp_path)
    manifest.mapping(payload["resources"], "resources")["memory_gib"] = 0.1
    with pytest.raises(ValueError, match="cannot accommodate"):
        preflight.plan_capacity(manifest.parse_manifest(payload), 400_000, 100_000)


def test_remote_dry_run_never_starts_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = job_payload(tmp_path)
    inputs = manifest.mapping(payload["inputs"], "inputs")
    for name, source in inputs.items():
        manifest.mapping(source, name)["uri"] = f"gs://test-bucket/{name}"
    predictions = typing.cast("list[manifest.JsonValue]", payload["predictions"])
    manifest.mapping(predictions[0], "prediction")["uri"] = "gs://test-bucket/prediction"

    def forbidden(*arguments: object, **keywords: object) -> typing.NoReturn:
        raise AssertionError("Dry-run must not start a subprocess.")

    monkeypatch.setattr(subprocess, "run", forbidden)
    arguments = dataclasses.replace(job_arguments(tmp_path, payload, ("g", "regenie")), dry_run=True)
    result = runner.run_attempt(arguments)
    report = json.loads((result.attempt_directory / "preflight.json").read_text())
    assert report["remote_inputs_pending_content_verification"] == 5
    assert report["remote_metadata_checked"] == 0
    assert not report["all_input_content_verified"]
    assert not (result.attempt_directory / "inputs").exists()
    assert result.publication_uri is None


def test_local_success_publishes_atomic_complete_bundle(tmp_path: Path) -> None:
    payload = job_payload(tmp_path)
    arguments = job_arguments(tmp_path, payload, fake_engine(tmp_path, "success"))
    result = runner.run_attempt(arguments)
    assert result.publication_uri is not None
    published = Path(result.publication_uri)
    marker = json.loads((published / "COMMITTED.json").read_text())
    assert marker["status"] == "completed"
    assert marker["safe_for_external_export"] is False
    assert not (published / "inputs").exists()
    assert not list(published.parent.glob("*.publishing"))
    for entry in marker["files"]:
        digest = storage.file_digest(published / entry["relative_path"])
        assert digest.sha256 == entry["sha256"]
    assert (result.attempt_directory.stat().st_mode & 0o777) == 0o700
    config = (result.attempt_directory / "engine.toml").read_text()
    assert "resume = false" in config
    assert "trusted-jax-cache" in config
    prediction_line = (result.attempt_directory / "predictions.list").read_text()
    assert prediction_line == "trait inputs/prediction-trait.loco\n"


@pytest.mark.parametrize("mode", ["exit_failure", "no_outputs", "mutate_input"])
def test_failed_process_never_publishes_and_retry_is_new(tmp_path: Path, mode: str) -> None:
    payload = job_payload(tmp_path)
    arguments = job_arguments(tmp_path, payload, fake_engine(tmp_path, mode))
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        runner.run_attempt(arguments)
    attempts = tuple(arguments.work_directory.glob("attempt-*"))
    assert len(attempts) == 1
    status = json.loads((attempts[0] / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["publication_committed"] is False
    assert not Path(str(payload["output_uri"])).exists()
    successful = dataclasses.replace(arguments, runner_prefix=fake_engine(tmp_path, "success"))
    result = runner.run_attempt(successful)
    assert result.attempt_directory != attempts[0]
    assert (attempts[0] / "engine.stderr.log").exists()


def test_cloud_localization_pins_generation_and_verifies_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = b"synthetic source"
    source = manifest.InputFile("gs://test-bucket/input", hashlib.sha256(content).hexdigest(), len(content), None)
    calls: list[list[str]] = []

    def run(arguments: list[str], **keywords: object) -> subprocess.CompletedProcess[str]:
        del keywords
        calls.append(arguments)
        if "describe" in arguments:
            return subprocess.CompletedProcess(
                arguments, 0, json.dumps({"generation": "1234", "size": str(len(content))})
            )
        Path(arguments[arguments.index("cp") + 2]).write_bytes(content)
        return subprocess.CompletedProcess(arguments, 0, "")

    monkeypatch.setattr(subprocess, "run", run)
    localized = storage.localize_file(source, tmp_path / "localized", "billing-project")
    assert localized.generation == "1234"
    assert "gs://test-bucket/input#1234" in calls[1]
    assert "--billing-project" in calls[1]
    assert "--do-not-decompress" in calls[1]
    assert localized.path.read_bytes() == content


def test_cloud_failed_hash_preserves_partial_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = manifest.InputFile("gs://test-bucket/input", "0" * 64, 7, "1234")

    def run(arguments: list[str], **keywords: object) -> subprocess.CompletedProcess[str]:
        del keywords
        if "describe" in arguments:
            return subprocess.CompletedProcess(arguments, 0, '{"generation":"1234","size":"7"}')
        Path(arguments[arguments.index("cp") + 2]).write_bytes(b"changed")
        return subprocess.CompletedProcess(arguments, 0, "")

    monkeypatch.setattr(subprocess, "run", run)
    destination = tmp_path / "localized"
    with pytest.raises(ValueError, match="checksum"):
        storage.localize_file(source, destination, None)
    assert not destination.exists()
    assert destination.with_name("localized.partial").exists()


def test_cloud_publication_commits_marker_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "COMMITTED.json").write_text("{}")
    (bundle / "output.parquet").write_bytes(b"output")
    observed: list[str] = []

    def upload(path: Path, uri: str, billing_project: str | None) -> None:
        del path, billing_project
        observed.append(uri)

    monkeypatch.setattr(storage, "publish_cloud_file", upload)
    destination = storage.publish_bundle(bundle, "gs://test-bucket/outputs", "attempt-abc", None)
    assert destination == "gs://test-bucket/outputs/attempt-abc"
    assert observed[-1].endswith("/COMMITTED.json")


def test_cloud_upload_failure_never_sends_commit_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "COMMITTED.json").write_text("{}")
    (bundle / "output.parquet").write_bytes(b"output")
    observed: list[str] = []

    def fail_upload(path: Path, uri: str, billing_project: str | None) -> typing.NoReturn:
        del path, billing_project
        observed.append(uri)
        raise OSError("network failure")

    monkeypatch.setattr(storage, "publish_cloud_file", fail_upload)
    with pytest.raises(OSError, match="network failure"):
        storage.publish_bundle(bundle, "gs://test-bucket/outputs", "attempt-abc", None)
    assert all(not uri.endswith("/COMMITTED.json") for uri in observed)


def test_publication_refuses_existing_destination_and_symlink(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "COMMITTED.json").write_text("{}")
    output = tmp_path / "output"
    (output / "attempt-existing").mkdir(parents=True)
    with pytest.raises(ValueError, match="replace"):
        storage.publish_bundle(bundle, str(output), "attempt-existing", None)
    (bundle / "unexpected").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symbolic links"):
        storage.publish_bundle(bundle, str(output), "attempt-new", None)


def test_dry_run_preflight_module_imports_no_jax() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import tooling.workbench.runner; assert 'jax' not in sys.modules"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_tsv_preserves_empty_values_and_literal_na_identifiers(tmp_path: Path) -> None:
    path = tmp_path / "phenotypes.tsv"
    path.write_text("FID\tIID\ttrait\nNA\t-9\t\nfamily\tperson\t3.5\n")
    values = preflight.read_selected_table(path, ("trait",), binary=False)
    assert values[preflight.profile.SampleIdentifier("NA", "-9")] == (None,)
    assert values[preflight.profile.SampleIdentifier("family", "person")] == (3.5,)


def test_preflight_rejects_space_delimited_phenotype_table(tmp_path: Path) -> None:
    path = tmp_path / "phenotypes.tsv"
    path.write_text("FID IID trait\nfamily person 1\n")
    with pytest.raises(ValueError, match="Tab-separated"):
        preflight.read_selected_table(path, ("trait",), binary=False)


def test_loco_accepts_genome_wide_x_rows_and_rejects_duplicate_alias(tmp_path: Path) -> None:
    path = tmp_path / "pred.loco"
    path.write_text("FID_IID family_person\nchr22 0.1\nchrX 0.2\n")
    selected = {preflight.profile.SampleIdentifier("family", "person")}
    assert preflight.validate_loco(path, selected) == ("22", "23")
    with path.open("a") as target:
        target.write("23 0.3\n")
    with pytest.raises(ValueError, match="duplicate chromosome"):
        preflight.validate_loco(path, selected)


def test_preflight_rejects_insufficient_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = manifest.parse_manifest(job_payload(tmp_path))
    monkeypatch.setattr(preflight, "available_disk_bytes", lambda directory: 1)
    with pytest.raises(ValueError, match="Insufficient free local disk"):
        preflight.run_preflight(job, tmp_path, dry_run=False, billing_project=None)


def test_runner_rejects_untrusted_work_cache_directory(tmp_path: Path) -> None:
    directory = tmp_path / "unsafe"
    directory.mkdir(mode=0o777)
    directory.chmod(0o777)
    with pytest.raises(ValueError, match="not writable"):
        runner.validate_private_directory(directory)


def test_cancellation_retains_attempt_and_never_publishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = job_payload(tmp_path)
    arguments = job_arguments(tmp_path, payload, ("g", "regenie"))

    def cancel(command: list[str], attempt: Path, environment: dict[str, str]) -> typing.NoReturn:
        del command, environment
        (attempt / "engine.stderr.log").write_text("interrupted")
        raise KeyboardInterrupt("cancelled")

    monkeypatch.setattr(runner, "execute_engine", cancel)
    with pytest.raises(KeyboardInterrupt):
        runner.run_attempt(arguments)
    attempt = next(arguments.work_directory.glob("attempt-*"))
    status = json.loads((attempt / "status.json").read_text())
    assert status["status"] == "cancelled"
    assert status["publication_committed"] is False
    assert (attempt / "inputs" / "bgen").exists()
    assert not Path(str(payload["output_uri"])).exists()


def test_publication_lost_acknowledgement_records_uncertain_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = job_payload(tmp_path)
    arguments = job_arguments(tmp_path, payload, fake_engine(tmp_path, "success"))

    def publish(bundle: Path, output_uri: str, attempt_id: str, billing_project: str | None) -> typing.NoReturn:
        del output_uri, attempt_id, billing_project
        assert (bundle / "COMMITTED.json").exists()
        raise OSError("lost acknowledgement")

    monkeypatch.setattr(storage, "publish_bundle", publish)
    with pytest.raises(OSError, match="lost acknowledgement"):
        runner.run_attempt(arguments)
    attempt = next(arguments.work_directory.glob("attempt-*"))
    status = json.loads((attempt / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["publication_committed"] is None


def test_upload_disables_composite_objects_and_handles_empty_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "empty.log"
    path.write_bytes(b"")
    checksum = base64.b64encode(hashlib.md5(b"", usedforsecurity=False).digest()).decode("ascii")
    environments: list[object] = []

    def run(arguments: list[str], **keywords: object) -> subprocess.CompletedProcess[str]:
        if "cp" in arguments:
            environments.append(keywords.get("env"))
            return subprocess.CompletedProcess(arguments, 0, "")
        return subprocess.CompletedProcess(arguments, 0, json.dumps({"size": "0", "md5_hash": checksum}))

    monkeypatch.setattr(subprocess, "run", run)
    storage.publish_cloud_file(path, "gs://test-bucket/empty.log", None)
    environment = typing.cast("dict[str, str]", environments[0])
    assert environment["CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED"] == "false"


@pytest.mark.parametrize("version", [True, 1.0, "1", 2])
def test_manifest_requires_exact_integer_schema_version(tmp_path: Path, version: manifest.JsonValue) -> None:
    payload = job_payload(tmp_path)
    payload["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version"):
        manifest.parse_manifest(payload)


def test_publication_rejects_traversal_attempt_identifier(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "COMMITTED.json").write_text("{}")
    with pytest.raises(ValueError, match="attempt_id"):
        storage.publish_bundle(bundle, str(tmp_path / "published"), "../outside", None)
    assert not (tmp_path / "outside").exists()


def binary_job(job: manifest.JobManifest, phenotype_count: int, covariate_count: int) -> manifest.JobManifest:
    return dataclasses.replace(
        job,
        analysis=dataclasses.replace(
            job.analysis,
            trait_type=manifest.TraitType.BINARY,
            phenotype_columns=tuple(f"trait{index}" for index in range(phenotype_count)),
            covariate_columns=tuple(f"covariate{index}" for index in range(covariate_count)),
            fallback_method=manifest.FallbackMethod.FIRTH_APPROXIMATE,
            p_threshold=0.05,
        ),
        resources=dataclasses.replace(
            job.resources, device=manifest.Device.GPU, gpu_memory_gib=16, memory_fraction=0.5
        ),
    )


def test_firth_capacity_models_all_trait_candidates_and_covariate_gathers(tmp_path: Path) -> None:
    job = binary_job(manifest.parse_manifest(job_payload(tmp_path)), phenotype_count=8, covariate_count=20)
    plan = preflight.plan_capacity(job, 100_000, 100_000)
    assert plan.firth_batch_size is not None
    candidate_count = min(1024, plan.chunk_variants) * 8
    padded_count = ((candidate_count + plan.firth_batch_size - 1) // plan.firth_batch_size) * plan.firth_batch_size
    assert plan.padded_correction_candidate_lanes == padded_count
    assert plan.correction_preparation_estimate_bytes == 100_000 * padded_count * 4 * (21 + 8)
    assert plan.correction_solver_estimate_bytes == 100_000 * plan.firth_batch_size * 4 * 8
    assert plan.estimated_chunk_working_set_bytes <= plan.working_memory_budget_bytes
    assert plan.correction_memory_estimate_bytes == (
        plan.correction_preparation_estimate_bytes + plan.correction_solver_estimate_bytes
    )
    assert preflight.plan_capacity(binary_job(job, 8, 1), 100_000, 100_000).chunk_variants > plan.chunk_variants


def test_firth_default_batch_adapts_to_large_cohort_without_discarding_samples(tmp_path: Path) -> None:
    job = binary_job(manifest.parse_manifest(job_payload(tmp_path)), phenotype_count=1, covariate_count=20)
    plan = preflight.plan_capacity(job, 400_000, 100_000)
    assert plan.firth_batch_size is not None and plan.firth_batch_size < 256
    assert plan.chunk_variants >= 1
    assert plan.estimated_chunk_working_set_bytes <= plan.working_memory_budget_bytes
    explicit = dataclasses.replace(job, resources=dataclasses.replace(job.resources, firth_batch_size=256))
    with pytest.raises(ValueError, match="cannot accommodate"):
        preflight.plan_capacity(explicit, 400_000, 100_000)


@pytest.mark.parametrize("value", ["1e100", "-1e100", "inf"])
def test_preflight_rejects_float32_overflow(value: str) -> None:
    with pytest.raises(ValueError, match="float32"):
        preflight.finite_float32(value)


@pytest.mark.skipif(sys.platform != "linux", reason="Workbench native deployment uses Linux process sessions.")
@pytest.mark.parametrize("ignore_interrupt,repeat_interrupt", [(False, False), (True, False), (True, True)])
def test_cancellation_stops_launcher_grandchild(
    tmp_path: Path, *, ignore_interrupt: bool, repeat_interrupt: bool
) -> None:
    grandchild = tmp_path / "grandchild.py"
    grandchild.write_text(
        """import os
import pathlib
import signal
import sys
import time

root = pathlib.Path(sys.argv[1])
ignore = sys.argv[2] == "ignore"
def interrupted(signum, frame):
    (root / "grandchild.interrupted").write_text("SIGINT")
    if not ignore:
        raise SystemExit(0)
signal.signal(signal.SIGINT, interrupted)
(root / "grandchild.pid").write_text(str(os.getpid()))
while True:
    time.sleep(0.1)
"""
    )
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        """import os
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
(root / "wrapper.pid").write_text(str(os.getpid()))
child = subprocess.Popen([sys.executable, str(root / "grandchild.py"), str(root), sys.argv[2]])
try:
    child.wait()
except KeyboardInterrupt:
    child.wait()
"""
    )
    controller = tmp_path / "controller.py"
    controller.write_text(
        """import os
import pathlib
import sys
from tooling.workbench import runner

root = pathlib.Path(sys.argv[1])
runner.CANCELLATION_GRACE_SECONDS = 1.0
try:
    runner.execute_engine([sys.executable, str(root / "wrapper.py"), str(root), sys.argv[2]], root, dict(os.environ))
except KeyboardInterrupt:
    (root / "controller.cancelled").write_text("cancelled")
"""
    )
    mode = "ignore" if ignore_interrupt else "exit"
    process = subprocess.Popen(
        [sys.executable, str(controller), str(tmp_path), mode],
        env={**os.environ, "PYTHONPATH": str(Path.cwd()) + os.pathsep + os.environ.get("PYTHONPATH", "")},
    )
    wrapper_identifier: int | None = None
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "grandchild.pid").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (tmp_path / "grandchild.pid").exists()
        grandchild_identifier = int((tmp_path / "grandchild.pid").read_text())
        wrapper_identifier = int((tmp_path / "wrapper.pid").read_text())
        process.send_signal(signal.SIGINT)
        if repeat_interrupt:
            deadline = time.monotonic() + 5
            while not (tmp_path / "grandchild.interrupted").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert (tmp_path / "grandchild.interrupted").exists()
            process.send_signal(signal.SIGINT)
        assert process.wait(timeout=10) == 0
        assert (tmp_path / "grandchild.interrupted").read_text() == "SIGINT"
        assert (tmp_path / "controller.cancelled").exists()
        state = Path(f"/proc/{grandchild_identifier}/stat")
        if state.exists():
            assert state.read_text().split()[2] == "Z"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if wrapper_identifier is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(wrapper_identifier, signal.SIGKILL)


@pytest.mark.parametrize("threshold", [0.0, 1.0, 1.01, 0.999999999, 1e-100])
def test_binary_threshold_matches_native_open_interval(tmp_path: Path, threshold: float) -> None:
    payload = job_payload(tmp_path)
    analysis = manifest.mapping(payload["analysis"], "analysis")
    analysis["trait_type"] = "binary"
    analysis["binary"] = {"fallback_method": "firth_approximate", "p_threshold": threshold}
    with pytest.raises(ValueError, match="p_threshold"):
        manifest.parse_manifest(payload)
