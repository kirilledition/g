"""Verify native resume validation precedes optional JAX runtime initialization."""

from __future__ import annotations

import enum
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import subprocess
import sys
import types
import typing
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import tests.test_native_cli_interruptions as interruption_tests


class StartupStage(enum.StrEnum):
    NONE = "none"
    IMPORT = "import"
    CONFIGURATION = "configuration"
    DEVICES = "devices"
    BACKEND = "backend"


class ResumeDamage(enum.StrEnum):
    MISSING_PART = "missing-part"
    CORRUPT_PART = "corrupt-part"


class ResumeChange(enum.StrEnum):
    PHENOTYPE = "phenotype"
    PREDICTION = "prediction"
    NUMERICAL_POLICY = "numerical-policy"
    HARDWARE_POLICY = "hardware-policy"
    PROVENANCE = "provenance"


@dataclass(frozen=True)
class StartupScenario:
    directory: Path
    stage: StartupStage = StartupStage.NONE
    failure: interruption_tests.FailureKind = interruption_tests.FailureKind.NONE
    interrupt_second_chromosome: bool = False
    resume_with_conflicting_cache: bool = False


@dataclass(frozen=True)
class StartupObservation:
    exit_code: int
    stdout: str
    stderr: str
    startup_calls: list[str]
    visited_stages: list[str]
    jax_imported: bool


@dataclass(frozen=True)
class ArtifactSignatures:
    manifest: str
    parts: dict[str, str]


class StartupProbe:
    def __init__(self, scenario: StartupScenario) -> None:
        self.scenario = scenario
        self.calls: list[StartupStage] = []

    def checkpoint(self, stage: StartupStage) -> None:
        self.calls.append(stage)
        if stage != self.scenario.stage:
            return
        if self.scenario.failure == interruption_tests.FailureKind.KEYBOARD_INTERRUPT:
            raise KeyboardInterrupt
        if self.scenario.failure == interruption_tests.FailureKind.RUNTIME_ERROR:
            raise RuntimeError(f"Startup sentinel rejected {stage}; KeyboardInterrupt is only text")


class RejectJaxImports(importlib.abc.MetaPathFinder):
    def __init__(self, probe: StartupProbe) -> None:
        self.probe = probe

    def find_spec(
        self,
        fullname: str,
        path: typing.Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname in {"jax", "jaxlib", "g.jax_backend"}:
            self.probe.checkpoint(StartupStage.IMPORT)
        return None


class ObservedJaxConfig:
    def __init__(self, probe: StartupProbe) -> None:
        self.probe = probe

    def update(self, setting_name: str, value: object) -> None:
        self.probe.checkpoint(StartupStage.CONFIGURATION)


class ObservedJaxModule(types.ModuleType):
    __version__ = interruption_tests.StubJaxModule.__version__

    def __init__(self, probe: StartupProbe) -> None:
        super().__init__("jax")
        self.probe = probe
        self.config = ObservedJaxConfig(probe)

    def devices(self) -> list[object]:
        self.probe.checkpoint(StartupStage.DEVICES)
        raise AssertionError("The CPU-only fixture must never discover JAX devices")


class ObservedBackendModule(interruption_tests.StubBackendModule):
    def __init__(self, backend: interruption_tests.StubLinearBackend, probe: StartupProbe) -> None:
        super().__init__(backend)
        self.probe = probe

    def create_linear_backend(
        self, *, minimum_variance: float, relative_variance_tolerance: float
    ) -> interruption_tests.StubLinearBackend:
        self.probe.checkpoint(StartupStage.BACKEND)
        return super().create_linear_backend(
            minimum_variance=minimum_variance,
            relative_variance_tolerance=relative_variance_tolerance,
        )


def native_arguments(directory: Path) -> list[str]:
    return [
        "regenie",
        "--config",
        str(directory / "run.toml"),
        "--qt",
        "--bsize",
        "1",
        "--bgen",
        str(directory / "input.bgen"),
        "--sample",
        str(directory / "input.sample"),
        "--phenoFile",
        str(directory / "phenotypes.tsv"),
        "--phenoCol",
        "trait",
        "--covarFile",
        str(directory / "covariates.tsv"),
        "--covarCol",
        "age",
        "--pred",
        str(directory / "predictions.list"),
        "--out",
        str(directory / "output"),
    ]


def run_child_scenario(scenario: StartupScenario) -> None:
    import g._core

    callback_failure = (
        interruption_tests.FailureKind.KEYBOARD_INTERRUPT
        if scenario.interrupt_second_chromosome
        else interruption_tests.FailureKind.NONE
    )
    backend = interruption_tests.StubLinearBackend(
        interruption_tests.CallbackScenario(
            scenario.directory,
            interruption_tests.CallbackStage.PREPARE_CHROMOSOME,
            callback_failure,
            2,
        ),
    )
    probe = StartupProbe(scenario)
    assert "jax" not in sys.modules
    if scenario.stage == StartupStage.IMPORT:
        sys.meta_path.insert(0, RejectJaxImports(probe))
    else:
        sys.modules["jax"] = ObservedJaxModule(probe)
        sys.modules["jaxlib"] = interruption_tests.StubJaxlibModule("jaxlib")
        sys.modules["g.jax_backend"] = ObservedBackendModule(backend, probe)
    result = g._core.cli.run(native_arguments(scenario.directory))
    if scenario.resume_with_conflicting_cache:
        assert result.exit_code == 0, "".join(result.stderr_chunks)
        assert StartupStage.CONFIGURATION in probe.calls
        assert StartupStage.BACKEND in probe.calls
        baseline_signatures = artifact_signatures(scenario.directory)
        enable_resume(scenario.directory)
        config_path = scenario.directory / "run.toml"
        cache_directory = json.dumps(str(scenario.directory / "different-jax-cache"))
        config_path.write_text(
            config_path.read_text(encoding="utf-8").replace(
                "cpu_threads = 1", f"cpu_threads = 1\njax_cache_dir = {cache_directory}"
            ),
            encoding="utf-8",
        )
        probe.calls.clear()
        backend.visited_stages.clear()
        result = g._core.cli.run(native_arguments(scenario.directory))
        assert artifact_signatures(scenario.directory) == baseline_signatures
    print(
        json.dumps(
            {
                "exit_code": result.exit_code,
                "stdout": "".join(result.stdout_chunks),
                "stderr": "".join(result.stderr_chunks),
                "startup_calls": probe.calls,
                "visited_stages": backend.visited_stages,
                "jax_imported": "jax" in sys.modules,
            },
        ),
    )


def observe_startup(scenario: StartupScenario) -> StartupObservation:
    environment = os.environ.copy()
    repository_directory = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (environment.get("PYTHONPATH", ""), str(repository_directory)) if value
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(scenario.directory),
            scenario.stage.value,
            scenario.failure.value,
            str(int(scenario.interrupt_second_chromosome)),
            str(int(scenario.resume_with_conflicting_cache)),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    observation = typing.cast("dict[str, object]", json.loads(completed.stdout))
    return StartupObservation(
        exit_code=typing.cast("int", observation["exit_code"]),
        stdout=typing.cast("str", observation["stdout"]),
        stderr=typing.cast("str", observation["stderr"]),
        startup_calls=typing.cast("list[str]", observation["startup_calls"]),
        visited_stages=typing.cast("list[str]", observation["visited_stages"]),
        jax_imported=typing.cast("bool", observation["jax_imported"]),
    )


def enable_resume(directory: Path) -> None:
    config_path = directory / "run.toml"
    config_path.write_text(config_path.read_text(encoding="utf-8") + "resume = true\n", encoding="utf-8")


def artifact_signatures(directory: Path) -> ArtifactSignatures:
    manifest_json = json.dumps(interruption_tests.read_manifest(directory), sort_keys=True).encode("utf-8")
    parts_directory = interruption_tests.native_run_directory(directory) / "parts"
    return ArtifactSignatures(
        manifest=hashlib.sha256(manifest_json).hexdigest(),
        parts={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in parts_directory.glob("*.parquet")},
    )


def complete_fixture(directory: Path) -> ArtifactSignatures:
    interruption_tests.write_native_inputs(directory)
    observed = observe_startup(StartupScenario(directory))
    assert observed.exit_code == 0, observed
    assert StartupStage.CONFIGURATION in observed.startup_calls
    assert StartupStage.BACKEND in observed.startup_calls
    assert interruption_tests.read_manifest(directory)["status"] == "completed"
    return artifact_signatures(directory)


@pytest.mark.parametrize("stage", [StartupStage.IMPORT, StartupStage.CONFIGURATION, StartupStage.BACKEND])
def test_completed_resume_skips_jax_startup(tmp_path: Path, stage: StartupStage) -> None:
    baseline = complete_fixture(tmp_path)
    enable_resume(tmp_path)

    observed = observe_startup(StartupScenario(tmp_path, stage, interruption_tests.FailureKind.RUNTIME_ERROR))

    assert observed.exit_code == 0, observed
    assert "Success. Run saved to" in observed.stdout
    assert observed.startup_calls == []
    assert observed.visited_stages == []
    if stage == StartupStage.IMPORT:
        assert not observed.jax_imported
    assert artifact_signatures(tmp_path) == baseline


def test_completed_resume_rejects_configured_process_cache_conflict(tmp_path: Path) -> None:
    interruption_tests.write_native_inputs(tmp_path)

    observed = observe_startup(StartupScenario(tmp_path, resume_with_conflicting_cache=True))

    assert observed.exit_code == 1, observed
    assert "JAX runtime is already configured" in observed.stderr
    assert "incompatible settings" in observed.stderr
    assert "jax-cache-directory" in observed.stderr
    assert observed.startup_calls == []
    assert observed.visited_stages == []
    assert "Success." not in observed.stdout
    assert interruption_tests.read_manifest(tmp_path)["status"] == "completed"


@pytest.mark.parametrize("damage", list(ResumeDamage))
def test_completed_resume_validates_committed_parts_before_jax(tmp_path: Path, damage: ResumeDamage) -> None:
    complete_fixture(tmp_path)
    enable_resume(tmp_path)
    part_path = next((interruption_tests.native_run_directory(tmp_path) / "parts").glob("*.parquet"))
    if damage == ResumeDamage.MISSING_PART:
        part_path.unlink()
    else:
        part_path.write_bytes(b"corrupt parquet part")
    damaged_signatures = artifact_signatures(tmp_path)

    observed = observe_startup(
        StartupScenario(tmp_path, StartupStage.IMPORT, interruption_tests.FailureKind.RUNTIME_ERROR),
    )

    assert observed.exit_code == 1, observed
    assert observed.startup_calls == []
    assert observed.visited_stages == []
    assert "Success." not in observed.stdout
    if damage == ResumeDamage.MISSING_PART:
        assert "missing chunk file" in observed.stderr
    else:
        assert "parquet" in observed.stderr.lower()
    assert artifact_signatures(tmp_path) == damaged_signatures


@pytest.mark.parametrize("change", list(ResumeChange))
def test_completed_resume_validates_inputs_and_provenance_before_jax(tmp_path: Path, change: ResumeChange) -> None:
    complete_fixture(tmp_path)
    enable_resume(tmp_path)
    if change == ResumeChange.PHENOTYPE:
        input_path = tmp_path / "phenotypes.tsv"
        input_path.write_text(input_path.read_text(encoding="utf-8").replace("i4\t6", "i4\t7"), encoding="utf-8")
    elif change == ResumeChange.PREDICTION:
        input_path = tmp_path / "predictions.loco"
        input_path.write_text(
            input_path.read_text(encoding="utf-8").replace("23 0 0 0 0", "23 0 0 0 1"), encoding="utf-8"
        )
    elif change == ResumeChange.HARDWARE_POLICY:
        config_path = tmp_path / "run.toml"
        config_path.write_text(
            config_path.read_text(encoding="utf-8").replace("cpu_threads = 1", "cpu_threads = 1\ndevice = 'gpu'"),
            encoding="utf-8",
        )
    elif change == ResumeChange.NUMERICAL_POLICY:
        config_path = tmp_path / "run.toml"
        config_path.write_text(
            config_path.read_text(encoding="utf-8").replace(
                "cpu_threads = 1", "cpu_threads = 1\nlinear_minimum_variance = 0.01"
            ),
            encoding="utf-8",
        )
    else:
        manifest = interruption_tests.read_manifest(tmp_path)
        execution_plan = typing.cast("dict[str, object]", manifest["execution_plan"])
        execution_plan["sample_set_fingerprint"] = "changed provenance"
        manifest_path = interruption_tests.native_run_directory(tmp_path) / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    changed_signatures = artifact_signatures(tmp_path)

    observed = observe_startup(
        StartupScenario(tmp_path, StartupStage.IMPORT, interruption_tests.FailureKind.RUNTIME_ERROR),
    )

    assert observed.exit_code == 1, observed
    assert "incompatible with the requested run" in observed.stderr
    assert observed.startup_calls == []
    assert observed.visited_stages == []
    assert artifact_signatures(tmp_path) == changed_signatures


def test_partial_resume_initializes_backend_and_computes_only_pending_chromosome(tmp_path: Path) -> None:
    interruption_tests.write_native_inputs(tmp_path)
    interrupted = observe_startup(StartupScenario(tmp_path, interrupt_second_chromosome=True))
    assert interrupted.exit_code == 130, interrupted
    manifest = interruption_tests.read_manifest(tmp_path)
    assert manifest["status"] == "interrupted"
    assert len(typing.cast("list[object]", manifest["committed_chunks"])) == 1
    preserved_parts = artifact_signatures(tmp_path).parts
    enable_resume(tmp_path)

    resumed = observe_startup(StartupScenario(tmp_path))

    assert resumed.exit_code == 0, resumed
    assert StartupStage.CONFIGURATION in resumed.startup_calls
    assert resumed.startup_calls.count(StartupStage.BACKEND) == 1
    assert resumed.visited_stages.count(interruption_tests.CallbackStage.PREPARE_CHROMOSOME) == 1
    assert resumed.visited_stages.count(interruption_tests.CallbackStage.COMPUTE_BATCH) == 1
    completed_manifest = interruption_tests.read_manifest(tmp_path)
    assert completed_manifest["status"] == "completed"
    assert "interrupted_signal" not in completed_manifest
    assert len(typing.cast("list[object]", completed_manifest["committed_chunks"])) == 2
    completed_parts = artifact_signatures(tmp_path).parts
    assert preserved_parts.items() <= completed_parts.items()
    part_paths = sorted((interruption_tests.native_run_directory(tmp_path) / "parts").glob("*.parquet"))
    chromosomes = [chromosome for path in part_paths for chromosome in pq.read_table(path).column("CHROM").to_pylist()]
    assert chromosomes == ["22", "23"]


@pytest.mark.parametrize("stage", [StartupStage.CONFIGURATION, StartupStage.BACKEND])
@pytest.mark.parametrize(
    "failure",
    [interruption_tests.FailureKind.KEYBOARD_INTERRUPT, interruption_tests.FailureKind.RUNTIME_ERROR],
)
def test_fresh_startup_failures_preserve_error_kind_without_creating_output(
    tmp_path: Path, stage: StartupStage, failure: interruption_tests.FailureKind
) -> None:
    interruption_tests.write_native_inputs(tmp_path)

    observed = observe_startup(StartupScenario(tmp_path, stage, failure))

    assert stage in observed.startup_calls
    assert observed.visited_stages == []
    assert not (tmp_path / "output.g").exists()
    assert "Success." not in observed.stdout
    if failure == interruption_tests.FailureKind.KEYBOARD_INTERRUPT:
        assert observed.exit_code == 130, observed
        assert "Interrupted by SIGINT" in observed.stderr
        assert "saved committed output for resume" not in observed.stderr
    else:
        assert observed.exit_code == 1, observed
        assert "Startup sentinel rejected" in observed.stderr
        assert "Interrupted by SIGINT" not in observed.stderr


@pytest.mark.parametrize("stage", [StartupStage.CONFIGURATION, StartupStage.BACKEND])
@pytest.mark.parametrize(
    "failure",
    [interruption_tests.FailureKind.KEYBOARD_INTERRUPT, interruption_tests.FailureKind.RUNTIME_ERROR],
)
def test_resume_startup_failures_preserve_commits_and_typed_interruption(
    tmp_path: Path, stage: StartupStage, failure: interruption_tests.FailureKind
) -> None:
    interruption_tests.write_native_inputs(tmp_path)
    interrupted = observe_startup(StartupScenario(tmp_path, interrupt_second_chromosome=True))
    assert interrupted.exit_code == 130, interrupted
    baseline_manifest = interruption_tests.read_manifest(tmp_path)
    baseline_commits = typing.cast("list[object]", baseline_manifest["committed_chunks"])
    assert len(baseline_commits) == 1
    baseline_parts = artifact_signatures(tmp_path).parts
    enable_resume(tmp_path)

    observed = observe_startup(StartupScenario(tmp_path, stage, failure))

    assert stage in observed.startup_calls
    assert observed.visited_stages == []
    assert "Success." not in observed.stdout
    manifest = interruption_tests.read_manifest(tmp_path)
    assert manifest["committed_chunks"] == baseline_commits
    assert artifact_signatures(tmp_path).parts == baseline_parts
    if failure == interruption_tests.FailureKind.KEYBOARD_INTERRUPT:
        assert observed.exit_code == 130, observed
        assert "Interrupted by SIGINT" in observed.stderr
        assert "saved committed output for resume" in observed.stderr
        assert manifest["status"] == "interrupted"
        assert manifest["interrupted_signal"] == "SIGINT"
    else:
        assert observed.exit_code == 1, observed
        assert "Startup sentinel rejected" in observed.stderr
        assert "Interrupted by SIGINT" not in observed.stderr
        assert manifest["status"] != "completed"
        assert "interrupted_signal" not in manifest


if __name__ == "__main__":
    run_child_scenario(
        StartupScenario(
            Path(sys.argv[1]),
            StartupStage(sys.argv[2]),
            interruption_tests.FailureKind(sys.argv[3]),
            interrupt_second_chromosome=bool(int(sys.argv[4])),
            resume_with_conflicting_cache=bool(int(sys.argv[5])),
        ),
    )
