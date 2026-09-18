"""Exercise native interruption handling without initializing JAX or compiling kernels."""

from __future__ import annotations

import enum
import json
import signal
import struct
import subprocess
import sys
import types
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import pytest


class CallbackStage(enum.StrEnum):
    """Native-to-Python callbacks covered by the interruption contract."""

    PREPARE_GROUP = "prepare_group"
    PREPARE_CHROMOSOME = "prepare_chromosome"
    TRANSFER_BATCH = "transfer_batch"
    COMPUTE_BATCH = "compute_batch"
    MATERIALIZE_BATCH = "materialize_batch"


class FailureKind(enum.StrEnum):
    """Distinguish typed interruptions, real signals, and ordinary failures."""

    NONE = "none"
    KEYBOARD_INTERRUPT = "keyboard-interrupt"
    SIGINT = "sigint"
    RUNTIME_ERROR = "runtime-error"


@dataclass(frozen=True)
class CallbackScenario:
    directory: Path
    stage: CallbackStage
    failure: FailureKind
    occurrence: int


@dataclass(frozen=True)
class StubAssociation:
    beta: npt.NDArray[np.float32]
    standard_error: npt.NDArray[np.float32]
    chi_squared: npt.NDArray[np.float32]
    log10_p_value: npt.NDArray[np.float32]
    correction_code: None


@dataclass(frozen=True)
class StubMaterializedBatch:
    association: StubAssociation
    raw_packed8_statistics: None


class StubLinearBackend:
    """Produce fixed host statistics while preserving the native callback lifecycle."""

    def __init__(self, scenario: CallbackScenario) -> None:
        self.scenario = scenario
        self.visited_stages: list[CallbackStage] = []

    def checkpoint(self, stage: CallbackStage) -> None:
        self.visited_stages.append(stage)
        if stage != self.scenario.stage or self.visited_stages.count(stage) != self.scenario.occurrence:
            return
        if self.scenario.failure == FailureKind.KEYBOARD_INTERRUPT:
            raise KeyboardInterrupt
        if self.scenario.failure == FailureKind.SIGINT:
            signal.raise_signal(signal.SIGINT)
        elif self.scenario.failure == FailureKind.RUNTIME_ERROR:
            raise RuntimeError("KeyboardInterrupt is only text in this ordinary backend failure")

    def prepare_group(
        self,
        phenotypes: npt.NDArray[np.float32],
        covariates: npt.NDArray[np.float32],
        source_sample_count: None,
        selected_sample_count: None,
        selection_start: None,
        selected_sample_indices: None,
    ) -> None:
        self.checkpoint(CallbackStage.PREPARE_GROUP)

    def prepare_chromosome(self, group: None, predictions: npt.NDArray[np.float32]) -> None:
        self.checkpoint(CallbackStage.PREPARE_CHROMOSOME)

    def transfer_batch(
        self,
        genotype_values: npt.NDArray[np.float32],
        genotype_mean: npt.NDArray[np.float32],
        imputed_dosage_square_sum: npt.NDArray[np.float32] | None,
        sparse_candidate_mask: npt.NDArray[np.bool_] | None,
    ) -> npt.NDArray[np.float32]:
        self.checkpoint(CallbackStage.TRANSFER_BATCH)
        return genotype_values

    def compute_batch(self, chromosome: None, genotype_values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        self.checkpoint(CallbackStage.COMPUTE_BATCH)
        return genotype_values

    def materialize_batch(
        self,
        result: npt.NDArray[np.float32],
        active_trait_indices: npt.NDArray[np.int32] | None,
        logical_variant_count: int,
    ) -> StubMaterializedBatch:
        self.checkpoint(CallbackStage.MATERIALIZE_BATCH)
        statistics = np.full((1, logical_variant_count), 0.25, dtype=np.float32)
        return StubMaterializedBatch(
            association=StubAssociation(
                beta=statistics,
                standard_error=statistics,
                chi_squared=statistics,
                log10_p_value=statistics,
                correction_code=None,
            ),
            raw_packed8_statistics=None,
        )


class StubJaxConfig:
    def update(self, setting_name: str, value: object) -> None:
        """Accept the runner's configuration without initializing a device."""


class StubJaxModule(types.ModuleType):
    __version__ = "0.11.0"
    config = StubJaxConfig()


class StubJaxlibModule(types.ModuleType):
    __version__ = "0.11.0"


class StubBackendModule(types.ModuleType):
    def __init__(self, backend: StubLinearBackend) -> None:
        super().__init__("g.jax_backend")
        self.backend = backend
        self.LinearJaxBackend = self.create_linear_backend

    def create_linear_backend(
        self, *, minimum_variance: float, relative_variance_tolerance: float
    ) -> StubLinearBackend:
        return self.backend


def encode_bgen_variant(chromosome: str) -> bytes:
    """Encode one uncompressed layout-2 diploid variant with four samples."""
    metadata = b"".join(
        struct.pack("<H", len(value)) + value for value in (b"variant", b"rs", chromosome.encode("ascii"))
    )
    metadata += struct.pack("<IH", 1, 2) + struct.pack("<I", 1) + b"A" + struct.pack("<I", 1) + b"G"
    probabilities = struct.pack("<IH", 4, 2) + bytes([2, 2, 2, 2, 2, 2, 0, 8, 0, 0, 255, 0, 0, 255, 128, 0])
    return metadata + struct.pack("<I", len(probabilities)) + probabilities


def write_native_inputs(directory: Path) -> Path:
    """Write two chromosomes so a later interruption can flush earlier results."""
    header = struct.pack("<IIII4sI", 20, 20, 2, 4, b"bgen", 8)
    (directory / "input.bgen").write_bytes(header + encode_bgen_variant("22") + encode_bgen_variant("23"))
    (directory / "input.sample").write_text("ID_1 ID_2\n0 0\nf1 i1\nf2 i2\nf3 i3\nf4 i4\n", encoding="utf-8")
    (directory / "phenotypes.tsv").write_text(
        "FID\tIID\ttrait\nf1\ti1\t0\nf2\ti2\t1\nf3\ti3\t3\nf4\ti4\t6\n",
        encoding="utf-8",
    )
    (directory / "covariates.tsv").write_text(
        "FID\tIID\tage\nf1\ti1\t1\nf2\ti2\t2\nf3\ti3\t3\nf4\ti4\t4\n",
        encoding="utf-8",
    )
    (directory / "predictions.loco").write_text(
        "FID_IID f1_i1 f2_i2 f3_i3 f4_i4\n22 0 0 0 0\n23 0 0 0 0\n",
        encoding="utf-8",
    )
    (directory / "predictions.list").write_text("trait predictions.loco\n", encoding="utf-8")
    config_path = directory / "run.toml"
    config_path.write_text(
        "[diagnostics]\ntelemetry = 'off'\n[compute]\ncpu_threads = 1\n[output]\nwriter_threads = 1\n",
        encoding="utf-8",
    )
    return config_path


def native_run_directory(directory: Path) -> Path:
    return directory / "output.g" / "trait_0001_trait.regenie2_linear.run"


def run_child_scenario(scenario: CallbackScenario) -> None:
    """Run the real frontend, engine, and writers against host-only callback stubs."""
    import g._core

    backend = StubLinearBackend(scenario)
    sys.modules["jax"] = StubJaxModule("jax")
    sys.modules["jaxlib"] = StubJaxlibModule("jaxlib")
    sys.modules["g.jax_backend"] = StubBackendModule(backend)
    config_path = write_native_inputs(scenario.directory)
    arguments = [
        "regenie",
        "--config",
        str(config_path),
        "--qt",
        "--bsize",
        "1",
        "--bgen",
        str(scenario.directory / "input.bgen"),
        "--sample",
        str(scenario.directory / "input.sample"),
        "--phenoFile",
        str(scenario.directory / "phenotypes.tsv"),
        "--phenoCol",
        "trait",
        "--covarFile",
        str(scenario.directory / "covariates.tsv"),
        "--covarCol",
        "age",
        "--pred",
        str(scenario.directory / "predictions.list"),
        "--out",
        str(scenario.directory / "output"),
    ]
    result = g._core.cli.run(arguments)
    print(
        json.dumps(
            {
                "exit_code": result.exit_code,
                "stderr": "".join(result.stderr_chunks),
                "visited_stages": backend.visited_stages,
            },
        ),
    )


def observe_scenario(scenario: CallbackScenario) -> dict[str, object]:
    """Isolate native process-global configuration and Python signal handling."""
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(scenario.directory),
            scenario.stage.value,
            scenario.failure.value,
            str(scenario.occurrence),
        ],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    observation = typing.cast("dict[str, object]", json.loads(completed.stdout))
    assert scenario.stage in typing.cast("list[str]", observation["visited_stages"])
    return observation


def read_manifest(directory: Path) -> dict[str, object]:
    return typing.cast(
        "dict[str, object]",
        json.loads((native_run_directory(directory) / "run_manifest.json").read_text(encoding="utf-8")),
    )


@pytest.mark.parametrize("stage", list(CallbackStage))
def test_backend_keyboard_interrupt_preserves_signal_status(tmp_path: Path, stage: CallbackStage) -> None:
    observed = observe_scenario(CallbackScenario(tmp_path, stage, FailureKind.KEYBOARD_INTERRUPT, 1))

    assert observed["exit_code"] == 130, observed
    assert "Interrupted by SIGINT" in typing.cast("str", observed["stderr"])
    manifest = read_manifest(tmp_path)
    assert manifest["status"] == "interrupted"
    assert manifest["interrupted_signal"] == "SIGINT"


def test_real_sigint_during_python_callback_preserves_signal_status(tmp_path: Path) -> None:
    observed = observe_scenario(CallbackScenario(tmp_path, CallbackStage.PREPARE_GROUP, FailureKind.SIGINT, 1))

    assert observed["exit_code"] == 130, observed
    assert read_manifest(tmp_path)["status"] == "interrupted"


def test_backend_interrupt_flushes_buffered_chromosome_output(tmp_path: Path) -> None:
    observed = observe_scenario(
        CallbackScenario(tmp_path, CallbackStage.PREPARE_CHROMOSOME, FailureKind.KEYBOARD_INTERRUPT, 2),
    )

    assert observed["exit_code"] == 130, observed
    manifest = read_manifest(tmp_path)
    assert manifest["status"] == "interrupted"
    assert manifest["interrupted_signal"] == "SIGINT"
    commits = typing.cast("list[dict[str, object]]", manifest["committed_chunks"])
    assert len(commits) == 1
    assert commits[0]["variant_start_index"] == 0
    assert commits[0]["variant_stop_index"] == 1
    part_files = list((native_run_directory(tmp_path) / "parts").glob("*.parquet"))
    assert len(part_files) == 1
    output_table = pq.read_table(part_files[0])
    assert output_table.num_rows == 1
    assert output_table.column("CHROM").to_pylist() == ["22"]


def test_runtime_error_text_does_not_become_an_interruption(tmp_path: Path) -> None:
    observed = observe_scenario(
        CallbackScenario(tmp_path, CallbackStage.PREPARE_CHROMOSOME, FailureKind.RUNTIME_ERROR, 1),
    )

    assert observed["exit_code"] == 1, observed
    assert "ordinary backend failure" in typing.cast("str", observed["stderr"])
    assert "Interrupted by SIGINT" not in typing.cast("str", observed["stderr"])
    assert read_manifest(tmp_path)["status"] != "interrupted"


def test_callback_fixture_completes_without_an_injected_failure(tmp_path: Path) -> None:
    observed = observe_scenario(CallbackScenario(tmp_path, CallbackStage.MATERIALIZE_BATCH, FailureKind.NONE, 1))

    assert observed["exit_code"] == 0, observed
    manifest = read_manifest(tmp_path)
    assert manifest["status"] == "completed"
    assert len(typing.cast("list[object]", manifest["committed_chunks"])) == 2
    part_files = list((native_run_directory(tmp_path) / "parts").glob("*.parquet"))
    assert len(part_files) == 1
    output_table = pq.read_table(part_files[0])
    assert output_table.num_rows == 2
    assert output_table.column("CHROM").to_pylist() == ["22", "23"]


if __name__ == "__main__":
    run_child_scenario(
        CallbackScenario(Path(sys.argv[1]), CallbackStage(sys.argv[2]), FailureKind(sys.argv[3]), int(sys.argv[4])),
    )
