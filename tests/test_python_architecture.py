"""Architecture tests for Python package ownership boundaries."""

from __future__ import annotations

from pathlib import Path

from tooling.debug import check_python_architecture

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "g"


def test_python_import_policy_allows_current_production_tree() -> None:
    assert check_python_architecture.collect_python_import_policy_violations(PRODUCTION_PACKAGE_ROOT) == ()


def test_python_call_policy_allows_current_production_tree() -> None:
    assert check_python_architecture.collect_python_call_policy_violations(PRODUCTION_PACKAGE_ROOT) == ()


def test_compute_import_policy_rejects_cli_output_and_config_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    compute_directory = package_root / "compute"
    compute_directory.mkdir(parents=True)
    (compute_directory / "kernel.py").write_text(
        "\n".join(
            (
                "from g import cli",
                "from g.io import output",
                "import g.interface.config",
                "from ..io import source",
                "from g.io import *",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/compute/kernel.py"), 1, "g.cli", "g.cli"),
        (Path("g/compute/kernel.py"), 2, "g.io.output", "g.io"),
        (Path("g/compute/kernel.py"), 3, "g.interface.config", "g.interface"),
        (Path("g/compute/kernel.py"), 4, "g.io.source", "g.io"),
        (Path("g/compute/kernel.py"), 5, "g.io", "g.io"),
    ]


def test_manifest_write_policy_rejects_production_python_manifest_writes(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    output_directory = package_root / "io"
    runner_directory.mkdir(parents=True)
    output_directory.mkdir(parents=True)
    (runner_directory / "metadata.py").write_text(
        "\n".join(
            (
                "from g.io import output",
                "from g import _core",
                "def extend(paths, manifest):",
                "    output.write_run_manifest(paths, manifest)",
                "    _core.write_run_manifest_json('run', '{}')",
            )
        ),
        encoding="utf-8",
    )
    (output_directory / "output.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def write_run_manifest(run_directory, manifest_json):",
                "    _core.write_run_manifest_json(run_directory, manifest_json)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/runner/metadata.py"), 4, "output.write_run_manifest", "output.write_run_manifest"),
        (Path("g/runner/metadata.py"), 5, "_core.write_run_manifest_json", "_core.write_run_manifest_json"),
    ]


def test_callback_worker_queue_policy_rejects_python_queue_and_thread_primitives(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "runtime.py").write_text(
        "\n".join(
            (
                "import queue",
                "import threading",
                "def build_worker_state():",
                "    queue.Queue()",
                "    threading.Thread(target=lambda: None)",
                "    threading.BoundedSemaphore(value=1)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/engine/callbacks/runtime.py"), 4, "queue.Queue", "queue.Queue"),
        (Path("g/engine/callbacks/runtime.py"), 5, "threading.Thread", "threading.Thread"),
        (Path("g/engine/callbacks/runtime.py"), 6, "threading.BoundedSemaphore", "threading.BoundedSemaphore"),
    ]


def test_compute_file_io_policy_rejects_direct_file_access(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    compute_directory = package_root / "compute"
    compute_directory.mkdir(parents=True)
    (compute_directory / "kernel.py").write_text(
        "\n".join(
            (
                "import numpy as np",
                "import pandas as pd",
                "from pathlib import Path",
                "def load_kernel_inputs(path):",
                "    open(path)",
                "    Path(path).read_text()",
                "    np.load(path)",
                "    pd.read_parquet(path)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/compute/kernel.py"), 5, "open", "open"),
        (Path("g/compute/kernel.py"), 6, "read_text", "read_text"),
        (Path("g/compute/kernel.py"), 7, "np.load", "np.load"),
        (Path("g/compute/kernel.py"), 8, "pd.read_parquet", "pd.read_parquet"),
    ]


def test_jax_runtime_import_policy_rejects_runner_orchestration_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    jax_runtime_directory = package_root / "jax_runtime"
    jax_runtime_directory.mkdir(parents=True)
    (jax_runtime_directory / "setup.py").write_text(
        "\n".join(
            (
                "from g.runner import runtime",
                "import g.runner.cli",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/jax_runtime/setup.py"), 1, "g.runner.runtime", "g.runner"),
        (Path("g/jax_runtime/setup.py"), 2, "g.runner.cli", "g.runner"),
    ]
