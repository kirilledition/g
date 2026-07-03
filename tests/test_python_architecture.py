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


def test_python_cli_shim_policy_allows_current_production_tree() -> None:
    assert check_python_architecture.collect_python_cli_shim_violations(PRODUCTION_PACKAGE_ROOT) == ()


def test_public_api_import_policy_rejects_backend_bypass_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    package_root.mkdir()
    (package_root / "api.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "from g import cli",
                "from g.compute import regenie2_linear",
                "from g.engine.callbacks import binary",
                "from g.engine.native_dispatch import delivery",
                "from g.engine.regenie2_pipeline import single_trait",
                "from g.engine import run_events",
                "from g import execution_plan",
                "from g.io import output",
                "from g import jax_runtime",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/api.py"), 1, "g._core", "g._core"),
        (Path("g/api.py"), 2, "g.cli", "g.cli"),
        (Path("g/api.py"), 3, "g.compute.regenie2_linear", "g.compute"),
        (Path("g/api.py"), 4, "g.engine.callbacks.binary", "g.engine.callbacks"),
        (Path("g/api.py"), 5, "g.engine.native_dispatch.delivery", "g.engine.native_dispatch"),
        (Path("g/api.py"), 6, "g.engine.regenie2_pipeline.single_trait", "g.engine.regenie2_pipeline"),
        (Path("g/api.py"), 7, "g.engine.run_events", "g.engine.run_events"),
        (Path("g/api.py"), 8, "g.execution_plan", "g.execution_plan"),
        (Path("g/api.py"), 9, "g.io.output", "g.io"),
        (Path("g/api.py"), 10, "g.jax_runtime", "g.jax_runtime"),
    ]


def test_cli_import_policy_rejects_engine_event_lifecycle_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    package_root.mkdir()
    (package_root / "cli.py").write_text(
        "\n".join(
            (
                "from g.engine import run_events, shutdown, telemetry",
                "import g.engine.run_events",
                "import g.engine.shutdown",
                "import g.engine.telemetry",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/cli.py"), 1, "g.engine.run_events", "g.engine.run_events"),
        (Path("g/cli.py"), 1, "g.engine.shutdown", "g.engine.shutdown"),
        (Path("g/cli.py"), 1, "g.engine.telemetry", "g.engine.telemetry"),
        (Path("g/cli.py"), 2, "g.engine.run_events", "g.engine.run_events"),
        (Path("g/cli.py"), 3, "g.engine.shutdown", "g.engine.shutdown"),
        (Path("g/cli.py"), 4, "g.engine.telemetry", "g.engine.telemetry"),
    ]


def test_interface_config_import_policy_rejects_host_runtime_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    interface_directory = package_root / "interface"
    interface_directory.mkdir(parents=True)
    (interface_directory / "config.py").write_text(
        "\n".join(
            (
                "from g import api",
                "from g import cli",
                "from g.compute import common",
                "from g.engine import run_events",
                "from g import execution_plan",
                "from g.io import output",
                "from g import jax_runtime",
                "from g.runner import execution",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/interface/config.py"), 1, "g.api", "g.api"),
        (Path("g/interface/config.py"), 2, "g.cli", "g.cli"),
        (Path("g/interface/config.py"), 3, "g.compute.common", "g.compute"),
        (Path("g/interface/config.py"), 4, "g.engine.run_events", "g.engine"),
        (Path("g/interface/config.py"), 5, "g.execution_plan", "g.execution_plan"),
        (Path("g/interface/config.py"), 6, "g.io.output", "g.io"),
        (Path("g/interface/config.py"), 7, "g.jax_runtime", "g.jax_runtime"),
        (Path("g/interface/config.py"), 8, "g.runner.execution", "g.runner"),
    ]


def test_compute_import_policy_rejects_host_orchestration_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    compute_directory = package_root / "compute"
    compute_directory.mkdir(parents=True)
    (compute_directory / "kernel.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "from g import api",
                "from g import cli",
                "import g.engine.native_dispatch",
                "from g import execution_plan",
                "import g.interface.config",
                "from g.io import output",
                "from g import jax_runtime",
                "from g.runner import execution",
                "from ..io import helpers",
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
        (Path("g/compute/kernel.py"), 1, "g._core", "g._core"),
        (Path("g/compute/kernel.py"), 2, "g.api", "g.api"),
        (Path("g/compute/kernel.py"), 3, "g.cli", "g.cli"),
        (Path("g/compute/kernel.py"), 4, "g.engine.native_dispatch", "g.engine"),
        (Path("g/compute/kernel.py"), 5, "g.execution_plan", "g.execution_plan"),
        (Path("g/compute/kernel.py"), 6, "g.interface.config", "g.interface"),
        (Path("g/compute/kernel.py"), 7, "g.io.output", "g.io"),
        (Path("g/compute/kernel.py"), 8, "g.jax_runtime", "g.jax_runtime"),
        (Path("g/compute/kernel.py"), 9, "g.runner.execution", "g.runner"),
        (Path("g/compute/kernel.py"), 10, "g.io.helpers", "g.io"),
        (Path("g/compute/kernel.py"), 11, "g.io", "g.io"),
    ]


def test_output_import_policy_rejects_jax_runtime_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    output_directory = package_root / "io"
    output_directory.mkdir(parents=True)
    (output_directory / "output.py").write_text(
        "\n".join(
            (
                "from g.jax_runtime import models",
                "import g.jax_runtime.setup",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/io/output.py"), 1, "g.jax_runtime.models", "g.jax_runtime"),
        (Path("g/io/output.py"), 2, "g.jax_runtime.setup", "g.jax_runtime"),
    ]


def test_output_import_policy_rejects_engine_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    output_directory = package_root / "io"
    output_directory.mkdir(parents=True)
    (output_directory / "output.py").write_text(
        "\n".join(
            (
                "from g.engine import run_events",
                "import g.engine.telemetry",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/io/output.py"), 1, "g.engine.run_events", "g.engine"),
        (Path("g/io/output.py"), 2, "g.engine.telemetry", "g.engine"),
    ]


def test_obsolete_io_source_import_policy_rejects_source_module_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    engine_directory = package_root / "engine"
    engine_directory.mkdir(parents=True)
    (engine_directory / "pipeline.py").write_text(
        "\n".join(
            (
                "from g.io import source",
                "import g.io.source",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/engine/pipeline.py"), 1, "g.io.source", "g.io.source"),
        (Path("g/engine/pipeline.py"), 2, "g.io.source", "g.io.source"),
    ]


def test_import_policy_rejects_obsolete_warm_cache_module(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "execution.py").write_text(
        "\n".join(
            (
                "from g.engine import warm_cache",
                "import g.engine.warm_cache",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/runner/execution.py"), 1, "g.engine.warm_cache", "g.engine.warm_cache"),
        (Path("g/runner/execution.py"), 2, "g.engine.warm_cache", "g.engine.warm_cache"),
    ]


def test_execution_plan_import_policy_rejects_output_adapter_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    package_root.mkdir()
    (package_root / "execution_plan.py").write_text(
        "\n".join(
            (
                "from g.io import output",
                "import g.io.output",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/execution_plan.py"), 1, "g.io.output", "g.io"),
        (Path("g/execution_plan.py"), 2, "g.io.output", "g.io"),
    ]


def test_runner_execution_import_policy_rejects_output_adapter_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "execution.py").write_text(
        "\n".join(
            (
                "from g.io import output",
                "import g.io.output",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/runner/execution.py"), 1, "g.io.output", "g.io"),
        (Path("g/runner/execution.py"), 2, "g.io.output", "g.io"),
    ]


def test_runner_metadata_import_policy_rejects_output_adapter_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "metadata.py").write_text(
        "\n".join(
            (
                "from g.io import output",
                "import g.io.output",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/runner/metadata.py"), 1, "g.io.output", "g.io"),
        (Path("g/runner/metadata.py"), 2, "g.io.output", "g.io"),
    ]


def test_runner_output_import_policy_allows_runner_output_adapter(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "outputs.py").write_text(
        "\n".join(
            (
                "from g.io import output",
                "import g.io.output",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert violations == ()


def test_runner_import_policy_rejects_run_event_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "execution.py").write_text(
        "\n".join(
            (
                "from g.engine import run_events, telemetry",
                "import g.engine.run_events",
                "import g.engine.telemetry",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/runner/execution.py"), 1, "g.engine.run_events", "g.engine.run_events"),
        (Path("g/runner/execution.py"), 1, "g.engine.telemetry", "g.engine.telemetry"),
        (Path("g/runner/execution.py"), 2, "g.engine.run_events", "g.engine.run_events"),
        (Path("g/runner/execution.py"), 3, "g.engine.telemetry", "g.engine.telemetry"),
    ]


def test_runner_import_policy_allows_runner_event_adapter(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "events.py").write_text(
        "\n".join(
            (
                "from g.engine import run_events, telemetry",
                "import g.engine.run_events",
                "import g.engine.telemetry",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert violations == ()


def test_runner_import_policy_rejects_lifecycle_timing_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "execution.py").write_text(
        "\n".join(
            (
                "from g.engine import shutdown, timing",
                "import g.engine.shutdown",
                "import g.engine.timing",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/runner/execution.py"), 1, "g.engine.shutdown", "g.engine.shutdown"),
        (Path("g/runner/execution.py"), 1, "g.engine.timing", "g.engine.timing"),
        (Path("g/runner/execution.py"), 2, "g.engine.shutdown", "g.engine.shutdown"),
        (Path("g/runner/execution.py"), 3, "g.engine.timing", "g.engine.timing"),
    ]


def test_runner_import_policy_allows_runner_lifecycle_timing_adapters(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "lifecycle.py").write_text(
        "\n".join(
            (
                "from g.engine import shutdown",
                "import g.engine.shutdown",
            )
        ),
        encoding="utf-8",
    )
    (runner_directory / "timing.py").write_text(
        "\n".join(
            (
                "from g.engine import timing",
                "import g.engine.timing",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert violations == ()


def test_native_dispatch_import_policy_rejects_event_lifecycle_timing_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    native_dispatch_directory = package_root / "engine" / "native_dispatch"
    native_dispatch_directory.mkdir(parents=True)
    (native_dispatch_directory / "delivery.py").write_text(
        "\n".join(
            (
                "from g.engine import run_events, shutdown, timing",
                "import g.engine.run_events",
                "import g.engine.shutdown",
                "import g.engine.timing",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/engine/native_dispatch/delivery.py"), 1, "g.engine.run_events", "g.engine.run_events"),
        (Path("g/engine/native_dispatch/delivery.py"), 1, "g.engine.shutdown", "g.engine.shutdown"),
        (Path("g/engine/native_dispatch/delivery.py"), 1, "g.engine.timing", "g.engine.timing"),
        (Path("g/engine/native_dispatch/delivery.py"), 2, "g.engine.run_events", "g.engine.run_events"),
        (Path("g/engine/native_dispatch/delivery.py"), 3, "g.engine.shutdown", "g.engine.shutdown"),
        (Path("g/engine/native_dispatch/delivery.py"), 4, "g.engine.timing", "g.engine.timing"),
    ]


def test_native_dispatch_import_policy_allows_event_lifecycle_timing_adapters(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    native_dispatch_directory = package_root / "engine" / "native_dispatch"
    native_dispatch_directory.mkdir(parents=True)
    (native_dispatch_directory / "events.py").write_text(
        "\n".join(
            (
                "from g.engine import run_events",
                "import g.engine.run_events",
            )
        ),
        encoding="utf-8",
    )
    (native_dispatch_directory / "lifecycle.py").write_text(
        "\n".join(
            (
                "from g.engine import shutdown",
                "import g.engine.shutdown",
            )
        ),
        encoding="utf-8",
    )
    (native_dispatch_directory / "timing.py").write_text(
        "\n".join(
            (
                "from g.engine import timing",
                "import g.engine.timing",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert violations == ()


def test_pipeline_import_policy_rejects_output_adapter_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "single_trait.py").write_text(
        "\n".join(
            (
                "from g.io import output",
                "import g.io.output",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 1, "g.io.output", "g.io"),
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 2, "g.io.output", "g.io"),
    ]


def test_pipeline_import_policy_allows_pipeline_output_adapter(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "outputs.py").write_text(
        "\n".join(
            (
                "from g.io import output",
                "import g.io.output",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert violations == ()


def test_pipeline_import_policy_rejects_run_event_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "single_trait.py").write_text(
        "\n".join(
            (
                "from g.engine import run_events, telemetry",
                "import g.engine.run_events",
                "import g.engine.telemetry",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 1, "g.engine.run_events", "g.engine.run_events"),
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 1, "g.engine.telemetry", "g.engine.telemetry"),
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 2, "g.engine.run_events", "g.engine.run_events"),
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 3, "g.engine.telemetry", "g.engine.telemetry"),
    ]


def test_pipeline_import_policy_allows_pipeline_telemetry_adapter(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "telemetry_events.py").write_text(
        "\n".join(
            (
                "from g.engine import run_events, telemetry",
                "import g.engine.run_events",
                "import g.engine.telemetry",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert violations == ()


def test_pipeline_import_policy_rejects_timing_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "single_trait.py").write_text(
        "\n".join(
            (
                "from g.engine import timing",
                "import g.engine.timing",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 1, "g.engine.timing", "g.engine.timing"),
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 2, "g.engine.timing", "g.engine.timing"),
    ]


def test_pipeline_import_policy_allows_pipeline_timing_adapter(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "timing.py").write_text(
        "\n".join(
            (
                "from g.engine import timing",
                "import g.engine.timing",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert violations == ()


def test_pipeline_import_policy_rejects_preflight_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "single_trait.py").write_text(
        "\n".join(
            (
                "from g.engine import preflight",
                "import g.engine.preflight",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 1, "g.engine.preflight", "g.engine.preflight"),
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 2, "g.engine.preflight", "g.engine.preflight"),
    ]


def test_pipeline_import_policy_allows_pipeline_preflight_adapter(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "preflight.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build_policy():",
                "    return _core.NativeOutputPreflightDiagnosticPolicy()",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert violations == ()


def test_preflight_import_policy_rejects_run_event_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "preflight.py").write_text(
        "\n".join(
            (
                "from g.engine import run_events",
                "import g.engine.run_events",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/engine/regenie2_pipeline/preflight.py"), 1, "g.engine.run_events", "g.engine.run_events"),
        (Path("g/engine/regenie2_pipeline/preflight.py"), 2, "g.engine.run_events", "g.engine.run_events"),
    ]


def test_pipeline_import_policy_rejects_bgen_engine_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "outputs.py").write_text(
        "\n".join(
            (
                "from g.engine.native_dispatch import engine",
                "import g.engine.native_dispatch.engine",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (
            Path("g/engine/regenie2_pipeline/outputs.py"),
            1,
            "g.engine.native_dispatch.engine",
            "g.engine.native_dispatch.engine",
        ),
        (
            Path("g/engine/regenie2_pipeline/outputs.py"),
            2,
            "g.engine.native_dispatch.engine",
            "g.engine.native_dispatch.engine",
        ),
    ]


def test_pipeline_import_policy_allows_pipeline_bgen_engine_adapter(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "bgen_engine.py").write_text(
        "\n".join(
            (
                "from g.engine.native_dispatch import engine",
                "import g.engine.native_dispatch.engine",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert violations == ()


def test_pipeline_import_policy_rejects_native_input_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "single_trait.py").write_text(
        "\n".join(
            (
                "from g.engine.native_dispatch import loaders, groups, models",
                "import g.engine.native_dispatch.loaders",
                "import g.engine.native_dispatch.groups",
                "import g.engine.native_dispatch.models",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (
            Path("g/engine/regenie2_pipeline/single_trait.py"),
            1,
            "g.engine.native_dispatch.loaders",
            "g.engine.native_dispatch.loaders",
        ),
        (
            Path("g/engine/regenie2_pipeline/single_trait.py"),
            1,
            "g.engine.native_dispatch.groups",
            "g.engine.native_dispatch.groups",
        ),
        (
            Path("g/engine/regenie2_pipeline/single_trait.py"),
            1,
            "g.engine.native_dispatch.models",
            "g.engine.native_dispatch.models",
        ),
        (
            Path("g/engine/regenie2_pipeline/single_trait.py"),
            2,
            "g.engine.native_dispatch.loaders",
            "g.engine.native_dispatch.loaders",
        ),
        (
            Path("g/engine/regenie2_pipeline/single_trait.py"),
            3,
            "g.engine.native_dispatch.groups",
            "g.engine.native_dispatch.groups",
        ),
        (
            Path("g/engine/regenie2_pipeline/single_trait.py"),
            4,
            "g.engine.native_dispatch.models",
            "g.engine.native_dispatch.models",
        ),
    ]


def test_pipeline_import_policy_allows_pipeline_input_adapter(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "inputs.py").write_text(
        "\n".join(
            (
                "from g.engine.native_dispatch import loaders, groups, models",
                "import g.engine.native_dispatch.loaders",
                "import g.engine.native_dispatch.groups",
                "import g.engine.native_dispatch.models",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert violations == ()


def test_pipeline_import_policy_rejects_native_delivery_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "single_trait.py").write_text(
        "\n".join(
            (
                "from g.engine.native_dispatch import delivery",
                "import g.engine.native_dispatch.delivery",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (
            Path("g/engine/regenie2_pipeline/single_trait.py"),
            1,
            "g.engine.native_dispatch.delivery",
            "g.engine.native_dispatch.delivery",
        ),
        (
            Path("g/engine/regenie2_pipeline/single_trait.py"),
            2,
            "g.engine.native_dispatch.delivery",
            "g.engine.native_dispatch.delivery",
        ),
    ]


def test_pipeline_import_policy_allows_pipeline_delivery_adapter(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "delivery.py").write_text(
        "\n".join(
            (
                "from g.engine.native_dispatch import delivery",
                "import g.engine.native_dispatch.delivery",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert violations == ()


def test_pipeline_import_policy_rejects_callback_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "single_trait.py").write_text(
        "\n".join(
            (
                "from g.engine.callbacks import binary, grouped, linear, shared",
                "import g.engine.callbacks.binary",
                "import g.engine.callbacks.grouped",
                "import g.engine.callbacks.linear",
                "import g.engine.callbacks.shared",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 1, "g.engine.callbacks.binary", "g.engine.callbacks"),
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 1, "g.engine.callbacks.grouped", "g.engine.callbacks"),
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 1, "g.engine.callbacks.linear", "g.engine.callbacks"),
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 1, "g.engine.callbacks.shared", "g.engine.callbacks"),
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 2, "g.engine.callbacks.binary", "g.engine.callbacks"),
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 3, "g.engine.callbacks.grouped", "g.engine.callbacks"),
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 4, "g.engine.callbacks.linear", "g.engine.callbacks"),
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 5, "g.engine.callbacks.shared", "g.engine.callbacks"),
    ]


def test_pipeline_import_policy_allows_pipeline_callback_adapter(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "callbacks.py").write_text(
        "\n".join(
            (
                "from g.engine.callbacks import binary, grouped, linear, shared",
                "import g.engine.callbacks.binary",
                "import g.engine.callbacks.grouped",
                "import g.engine.callbacks.linear",
                "import g.engine.callbacks.shared",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert violations == ()


def test_callback_import_policy_rejects_timing_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "runtime.py").write_text(
        "\n".join(
            (
                "from g.engine import timing",
                "import g.engine.timing",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/engine/callbacks/runtime.py"), 1, "g.engine.timing", "g.engine.timing"),
        (Path("g/engine/callbacks/runtime.py"), 2, "g.engine.timing", "g.engine.timing"),
    ]


def test_callback_import_policy_allows_callback_timing_adapter(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "timing.py").write_text(
        "\n".join(
            (
                "from g.engine import timing",
                "import g.engine.timing",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert violations == ()


def test_callback_import_policy_rejects_event_and_telemetry_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "diagnostics.py").write_text(
        "\n".join(
            (
                "from g.engine import run_events, telemetry",
                "import g.engine.run_events",
                "import g.engine.telemetry",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/engine/callbacks/diagnostics.py"), 1, "g.engine.run_events", "g.engine.run_events"),
        (Path("g/engine/callbacks/diagnostics.py"), 1, "g.engine.telemetry", "g.engine.telemetry"),
        (Path("g/engine/callbacks/diagnostics.py"), 2, "g.engine.run_events", "g.engine.run_events"),
        (Path("g/engine/callbacks/diagnostics.py"), 3, "g.engine.telemetry", "g.engine.telemetry"),
    ]


def test_callback_import_policy_allows_callback_event_adapter(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "events.py").write_text(
        "\n".join(
            (
                "from g.engine import run_events, telemetry",
                "import g.engine.run_events",
                "import g.engine.telemetry",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert violations == ()


def test_pipeline_import_policy_rejects_compute_config_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "context.py").write_text(
        "\n".join(
            (
                "from g.compute.regenie2_binary import config as binary_config",
                "import g.compute.regenie2_linear.config",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/engine/regenie2_pipeline/context.py"), 1, "g.compute.regenie2_binary.config", "g.compute"),
        (Path("g/engine/regenie2_pipeline/context.py"), 2, "g.compute.regenie2_linear.config", "g.compute"),
    ]


def test_pipeline_import_policy_allows_pipeline_compute_config_adapter(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "compute_config.py").write_text(
        "\n".join(
            (
                "from g.compute.regenie2_binary import config as binary_config",
                "import g.compute.regenie2_linear.config",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert violations == ()


def test_pipeline_import_policy_rejects_jax_runtime_policy_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "outputs.py").write_text(
        "\n".join(
            (
                "from g.jax_runtime import models",
                "import g.jax_runtime.setup",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/engine/regenie2_pipeline/outputs.py"), 1, "g.jax_runtime.models", "g.jax_runtime"),
        (Path("g/engine/regenie2_pipeline/outputs.py"), 2, "g.jax_runtime.setup", "g.jax_runtime"),
    ]


def test_pipeline_import_policy_allows_pipeline_runtime_policy_adapter(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "runtime_policy.py").write_text(
        "\n".join(
            (
                "from g.jax_runtime import models",
                "import g.jax_runtime.setup",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert violations == ()


def test_import_policy_rejects_obsolete_backend_planner_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "context.py").write_text(
        "\n".join(
            (
                "from g.engine import backend_planner",
                "import g.engine.backend_planner",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (
            Path("g/engine/regenie2_pipeline/context.py"),
            1,
            "g.engine.backend_planner",
            "g.engine.backend_planner",
        ),
        (
            Path("g/engine/regenie2_pipeline/context.py"),
            2,
            "g.engine.backend_planner",
            "g.engine.backend_planner",
        ),
    ]


def test_import_policy_rejects_obsolete_backend_planner_from_adapter(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "backend.py").write_text(
        "\n".join(
            (
                "from g.engine import backend_planner",
                "import g.engine.backend_planner",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (
            Path("g/engine/regenie2_pipeline/backend.py"),
            1,
            "g.engine.backend_planner",
            "g.engine.backend_planner",
        ),
        (
            Path("g/engine/regenie2_pipeline/backend.py"),
            2,
            "g.engine.backend_planner",
            "g.engine.backend_planner",
        ),
    ]


def test_import_policy_rejects_obsolete_trusted_validation_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    native_dispatch_directory = package_root / "engine" / "native_dispatch"
    native_dispatch_directory.mkdir(parents=True)
    (native_dispatch_directory / "engine.py").write_text(
        "\n".join(
            (
                "from g.engine import trusted_validation",
                "import g.engine.trusted_validation",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (
            Path("g/engine/native_dispatch/engine.py"),
            1,
            "g.engine.trusted_validation",
            "g.engine.trusted_validation",
        ),
        (
            Path("g/engine/native_dispatch/engine.py"),
            2,
            "g.engine.trusted_validation",
            "g.engine.trusted_validation",
        ),
    ]


def test_import_policy_rejects_obsolete_preflight_events_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "preflight.py").write_text(
        "\n".join(
            (
                "from g.engine import preflight_events",
                "import g.engine.preflight_events",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (
            Path("g/engine/regenie2_pipeline/preflight.py"),
            1,
            "g.engine.preflight_events",
            "g.engine.preflight_events",
        ),
        (
            Path("g/engine/regenie2_pipeline/preflight.py"),
            2,
            "g.engine.preflight_events",
            "g.engine.preflight_events",
        ),
    ]


def test_pipeline_call_policy_rejects_native_schedule_policy_construction(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "grouped.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build_policy():",
                "    return _core.NativeSchedulePolicy()",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/engine/regenie2_pipeline/grouped.py"),
            3,
            "_core.NativeSchedulePolicy",
            "_core.NativeSchedulePolicy",
        ),
    ]


def test_pipeline_call_policy_allows_pipeline_schedule_adapter(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "schedule.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build_policy():",
                "    return _core.NativeSchedulePolicy()",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert violations == ()


def test_event_policy_factory_rejects_direct_native_construction(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "execution.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build_policies():",
                "    _core.NativeRunEventPayloadPolicy()",
                "    _core.NativeRunEventTelemetryPolicy()",
                "    _core.NativeRunnerDiagnosticPolicy()",
                "    _core.NativeOutputPreflightDiagnosticPolicy()",
                "    _core.NativePipelineDiagnosticPolicy()",
                "    _core.NativeDispatchDiagnosticPolicy()",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/runner/execution.py"),
            3,
            "_core.NativeRunEventPayloadPolicy",
            "_core.NativeRunEventPayloadPolicy",
        ),
        (
            Path("g/runner/execution.py"),
            4,
            "_core.NativeRunEventTelemetryPolicy",
            "_core.NativeRunEventTelemetryPolicy",
        ),
        (
            Path("g/runner/execution.py"),
            5,
            "_core.NativeRunnerDiagnosticPolicy",
            "_core.NativeRunnerDiagnosticPolicy",
        ),
        (
            Path("g/runner/execution.py"),
            6,
            "_core.NativeOutputPreflightDiagnosticPolicy",
            "_core.NativeOutputPreflightDiagnosticPolicy",
        ),
        (
            Path("g/runner/execution.py"),
            7,
            "_core.NativePipelineDiagnosticPolicy",
            "_core.NativePipelineDiagnosticPolicy",
        ),
        (
            Path("g/runner/execution.py"),
            8,
            "_core.NativeDispatchDiagnosticPolicy",
            "_core.NativeDispatchDiagnosticPolicy",
        ),
    ]


def test_event_policy_factory_allows_boundary_helpers(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    preflight_directory = package_root / "engine"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    native_dispatch_directory = package_root / "engine" / "native_dispatch"
    callback_directory = package_root / "engine" / "callbacks"
    runner_directory.mkdir(parents=True)
    preflight_directory.mkdir(parents=True, exist_ok=True)
    pipeline_directory.mkdir(parents=True)
    native_dispatch_directory.mkdir(parents=True)
    callback_directory.mkdir(parents=True)
    for path, call_name in (
        (runner_directory / "events.py", "_core.NativeRunEventPayloadPolicy()"),
        (runner_directory / "events.py", "_core.NativeRunEventTelemetryPolicy()"),
        (runner_directory / "events.py", "_core.NativeRunnerDiagnosticPolicy()"),
        (pipeline_directory / "preflight.py", "_core.NativeOutputPreflightDiagnosticPolicy()"),
        (pipeline_directory / "telemetry_events.py", "_core.NativePipelineDiagnosticPolicy()"),
        (pipeline_directory / "telemetry_events.py", "_core.NativeRunEventTelemetryPolicy()"),
        (native_dispatch_directory / "events.py", "_core.NativeDispatchDiagnosticPolicy()"),
        (callback_directory / "events.py", "_core.NativePipelineDiagnosticPolicy()"),
    ):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"from g import _core\ndef build_policy():\n    return {call_name}\n")

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert violations == ()


def test_python_cli_shim_policy_rejects_public_python_process_ownership(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    package_root.mkdir()
    (package_root / "cli.py").write_text(
        "\n".join(
            (
                'NATIVE_CLI_PYTHON_BRIDGE_SENTINEL_ENVIRONMENT_VARIABLE = "WRONG_SENTINEL"',
                "def run_args(arguments):",
                "    outcome = g._core.dispatch_cli(list(arguments))",
                "    return outcome.exit_code",
                "def run_args_legacy(arguments):",
                "    outcome = g._core.run_native_cli_python_bridge(list(arguments), sys.executable, 'SENTINEL')",
                "    return outcome.exit_code",
                "def main():",
                "    raise SystemExit(run_args_legacy(sys.argv[1:]))",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_cli_shim_violations(package_root)

    assert [(violation.path, violation.line_number, violation.subject) for violation in violations] == [
        (Path("g/cli.py"), 1, "NATIVE_CLI_PYTHON_BRIDGE_SENTINEL_ENVIRONMENT_VARIABLE"),
        (Path("g/cli.py"), 2, "run_args"),
        (Path("g/cli.py"), 2, "run_args"),
        (Path("g/cli.py"), 2, "run_args"),
        (Path("g/cli.py"), 5, "run_args_legacy"),
        (Path("g/cli.py"), 5, "run_args_legacy"),
        (Path("g/cli.py"), 8, "main"),
    ]


def test_python_cli_shim_policy_rejects_missing_cli_module(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    package_root.mkdir()

    violations = check_python_architecture.collect_python_cli_shim_violations(package_root)

    assert [(violation.path, violation.line_number, violation.subject) for violation in violations] == [
        (Path("g/cli.py"), 1, "cli.py"),
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
                "    _core.write_run_manifest('run', {})",
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
                "    _core.write_run_manifest(run_directory, {})",
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
        (Path("g/runner/metadata.py"), 6, "_core.write_run_manifest", "_core.write_run_manifest"),
    ]


def test_output_lifecycle_policy_rejects_direct_native_output_calls(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "outputs.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def initialize_output(token):",
                "    _core.prepare_output_run('root', 'regenie2_linear', 'parquet', False)",
                "    _core.load_run_manifest_payload('run')",
                "    _core.initialize_output_run('run', 'chunks', None, '{}', False, 'fast', token)",
                "    _core.initialize_output_run_from_values('run', 'chunks', None, {}, False, 'fast', token)",
                "    _core.NativeOutputLifecyclePolicy()",
                "    _core.validate_strict_manifest_chunks('chunks', '{}')",
                "    _core.validate_strict_manifest_chunks_from_value('chunks', {})",
                "    _core.repair_strict_manifest_chunk_commits_from_value('chunks', {})",
                "    _core.read_manifest_committed_chunk_identifiers_from_value({})",
                "    _core.validate_run_manifest_compatibility_from_values({}, {})",
                "    _core.finalize_output_run_chunks('run', 'chunks', 'parquet', 'zstd')",
                "    _core.build_pipeline_output_preparation_batch_from_values((), (), (), (), False, 'fast')",
                "    _core.NativePipelineOutputPreparationBatch((), (), (), (), False, 'fast')",
                "    _core.NativePipelineOutputPreparationPolicy()",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/runner/outputs.py"), 3, "_core.prepare_output_run", "_core.prepare_output_run"),
        (Path("g/runner/outputs.py"), 4, "_core.load_run_manifest_payload", "_core.load_run_manifest_payload"),
        (Path("g/runner/outputs.py"), 5, "_core.initialize_output_run", "_core.initialize_output_run"),
        (
            Path("g/runner/outputs.py"),
            6,
            "_core.initialize_output_run_from_values",
            "_core.initialize_output_run_from_values",
        ),
        (
            Path("g/runner/outputs.py"),
            7,
            "_core.NativeOutputLifecyclePolicy",
            "_core.NativeOutputLifecyclePolicy",
        ),
        (
            Path("g/runner/outputs.py"),
            8,
            "_core.validate_strict_manifest_chunks",
            "_core.validate_strict_manifest_chunks",
        ),
        (
            Path("g/runner/outputs.py"),
            9,
            "_core.validate_strict_manifest_chunks_from_value",
            "_core.validate_strict_manifest_chunks_from_value",
        ),
        (
            Path("g/runner/outputs.py"),
            10,
            "_core.repair_strict_manifest_chunk_commits_from_value",
            "_core.repair_strict_manifest_chunk_commits_from_value",
        ),
        (
            Path("g/runner/outputs.py"),
            11,
            "_core.read_manifest_committed_chunk_identifiers_from_value",
            "_core.read_manifest_committed_chunk_identifiers_from_value",
        ),
        (
            Path("g/runner/outputs.py"),
            12,
            "_core.validate_run_manifest_compatibility_from_values",
            "_core.validate_run_manifest_compatibility_from_values",
        ),
        (Path("g/runner/outputs.py"), 13, "_core.finalize_output_run_chunks", "_core.finalize_output_run_chunks"),
        (
            Path("g/runner/outputs.py"),
            14,
            "_core.build_pipeline_output_preparation_batch_from_values",
            "_core.build_pipeline_output_preparation_batch_from_values",
        ),
        (
            Path("g/runner/outputs.py"),
            15,
            "_core.NativePipelineOutputPreparationBatch",
            "_core.NativePipelineOutputPreparationBatch",
        ),
        (
            Path("g/runner/outputs.py"),
            16,
            "_core.NativePipelineOutputPreparationPolicy",
            "_core.NativePipelineOutputPreparationPolicy",
        ),
    ]


def test_output_writer_lifecycle_policy_rejects_direct_native_writer_cleanup_calls(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    writer_adapter_directory = package_root / "engine" / "native_dispatch"
    runner_directory.mkdir(parents=True)
    writer_adapter_directory.mkdir(parents=True)
    (runner_directory / "cleanup.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "import g",
                "def cleanup(session):",
                "    _core.finish_output_writer_session(session)",
                "    _core.finish_output_writer_session_interrupted(session, 'SIGINT')",
                "    g._core.abort_output_writer_session(session)",
            )
        ),
        encoding="utf-8",
    )
    (writer_adapter_directory / "writers.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def cleanup(session):",
                "    _core.finish_output_writer_session(session)",
                "    _core.finish_output_writer_session_interrupted(session, 'SIGINT')",
                "    _core.abort_output_writer_session(session)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/runner/cleanup.py"),
            4,
            "_core.finish_output_writer_session",
            "_core.finish_output_writer_session",
        ),
        (
            Path("g/runner/cleanup.py"),
            5,
            "_core.finish_output_writer_session_interrupted",
            "_core.finish_output_writer_session_interrupted",
        ),
        (
            Path("g/runner/cleanup.py"),
            6,
            "g._core.abort_output_writer_session",
            "_core.abort_output_writer_session",
        ),
    ]


def test_output_chunk_write_policy_rejects_direct_native_multi_writer_calls(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    writer_adapter_directory = package_root / "engine" / "callbacks"
    runner_directory.mkdir(parents=True)
    writer_adapter_directory.mkdir(parents=True)
    (runner_directory / "outputs.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def write_chunks(session):",
                "    _core.write_regenie2_multi_native_chunk(writer_sessions=[session], active_trait_indices=[0])",
                "    _core.write_regenie2_multi_native_chunk_f64(writer_sessions=[session], active_trait_indices=[0])",
                "    _core.NativeOutputChunkWritePolicy()",
            )
        ),
        encoding="utf-8",
    )
    (writer_adapter_directory / "writers.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def write_chunks(session):",
                "    _core.write_regenie2_multi_native_chunk(writer_sessions=[session], active_trait_indices=[0])",
                "    _core.write_regenie2_multi_native_chunk_f64(writer_sessions=[session], active_trait_indices=[0])",
                "    _core.NativeOutputChunkWritePolicy()",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/runner/outputs.py"),
            3,
            "_core.write_regenie2_multi_native_chunk",
            "_core.write_regenie2_multi_native_chunk",
        ),
        (
            Path("g/runner/outputs.py"),
            4,
            "_core.write_regenie2_multi_native_chunk_f64",
            "_core.write_regenie2_multi_native_chunk_f64",
        ),
        (
            Path("g/runner/outputs.py"),
            5,
            "_core.NativeOutputChunkWritePolicy",
            "_core.NativeOutputChunkWritePolicy",
        ),
    ]


def test_output_manifest_helper_policy_rejects_direct_native_helpers(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "metadata.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build_manifest(cache):",
                "    _core.NativeManifestFileFingerprintCache()",
                "    cache.build_file_fingerprint_payload('input.bgen', True)",
                "    cache.build_prediction_loco_file_fingerprints_json('pred.list', ['phenotype'])",
                "    cache.build_prediction_loco_file_fingerprints_payload('pred.list', ['phenotype'])",
                "    cache.build_current_run_manifest_header_json_from_input_json('{}')",
                "    cache.build_current_run_manifest_header_payload_from_input({})",
                "    _core.build_manifest_json_sha256('{}')",
                "    _core.build_manifest_json_sha256_from_value({})",
                "    _core.build_prepared_run_manifest_header_json_from_current_header_json('{}')",
                "    _core.build_prepared_run_plan_json_from_current_header_json('{}')",
                "    _core.build_prepared_run_plan_json_from_current_header({})",
                "    _core.build_manifest_file_fingerprint_payload('input.bgen', True)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/runner/metadata.py"),
            3,
            "_core.NativeManifestFileFingerprintCache",
            "_core.NativeManifestFileFingerprintCache",
        ),
        (Path("g/runner/metadata.py"), 4, "cache.build_file_fingerprint_payload", "build_file_fingerprint_payload"),
        (
            Path("g/runner/metadata.py"),
            5,
            "cache.build_prediction_loco_file_fingerprints_json",
            "build_prediction_loco_file_fingerprints_json",
        ),
        (
            Path("g/runner/metadata.py"),
            6,
            "cache.build_prediction_loco_file_fingerprints_payload",
            "build_prediction_loco_file_fingerprints_payload",
        ),
        (
            Path("g/runner/metadata.py"),
            7,
            "cache.build_current_run_manifest_header_json_from_input_json",
            "build_current_run_manifest_header_json_from_input_json",
        ),
        (
            Path("g/runner/metadata.py"),
            8,
            "cache.build_current_run_manifest_header_payload_from_input",
            "build_current_run_manifest_header_payload_from_input",
        ),
        (Path("g/runner/metadata.py"), 9, "_core.build_manifest_json_sha256", "_core.build_manifest_json_sha256"),
        (
            Path("g/runner/metadata.py"),
            10,
            "_core.build_manifest_json_sha256_from_value",
            "_core.build_manifest_json_sha256_from_value",
        ),
        (
            Path("g/runner/metadata.py"),
            11,
            "_core.build_prepared_run_manifest_header_json_from_current_header_json",
            "_core.build_prepared_run_manifest_header_json_from_current_header_json",
        ),
        (
            Path("g/runner/metadata.py"),
            12,
            "_core.build_prepared_run_plan_json_from_current_header_json",
            "_core.build_prepared_run_plan_json_from_current_header_json",
        ),
        (
            Path("g/runner/metadata.py"),
            13,
            "_core.build_prepared_run_plan_json_from_current_header",
            "_core.build_prepared_run_plan_json_from_current_header",
        ),
        (
            Path("g/runner/metadata.py"),
            14,
            "_core.build_manifest_file_fingerprint_payload",
            "_core.build_manifest_file_fingerprint_payload",
        ),
    ]


def test_run_metadata_policy_rejects_direct_native_metadata_helpers(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "execution.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "import g",
                "def build_metadata():",
                "    _core.NativeRunMetadataBuilder()",
                "    _core.build_execution_run_artifacts_payload()",
                "    g._core.extend_run_manifest_metadata()",
            )
        ),
        encoding="utf-8",
    )
    (runner_directory / "metadata.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build_metadata():",
                "    _core.NativeRunMetadataBuilder()",
                "    _core.build_execution_run_artifacts_payload()",
                "    _core.extend_run_manifest_metadata()",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/runner/execution.py"),
            4,
            "_core.NativeRunMetadataBuilder",
            "_core.NativeRunMetadataBuilder",
        ),
        (
            Path("g/runner/execution.py"),
            5,
            "_core.build_execution_run_artifacts_payload",
            "_core.build_execution_run_artifacts_payload",
        ),
        (
            Path("g/runner/execution.py"),
            6,
            "g._core.extend_run_manifest_metadata",
            "_core.extend_run_manifest_metadata",
        ),
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


def test_callback_worker_queue_policy_rejects_direct_native_resource_construction(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "runtime.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build_worker_state():",
                "    _core.NativeCallbackObjectQueue(1)",
                "    _core.NativeCallbackWaitSignal()",
                "    _core.NativeCallbackWorkerThread(target=lambda: None, name='worker')",
                "    _core.NativeCallbackSchedulerState(1, 1, None, None)",
                "    _core.NativeCallbackProgressState(1)",
                "    _core.NativeBinaryCorrectionSummary()",
                "    _core.NativeDosageBufferPoolState(1)",
                "    _core.NativeResultInFlightSlotState(1)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/engine/callbacks/runtime.py"), 3, "_core.NativeCallbackObjectQueue", "NativeCallbackObjectQueue"),
        (Path("g/engine/callbacks/runtime.py"), 4, "_core.NativeCallbackWaitSignal", "NativeCallbackWaitSignal"),
        (Path("g/engine/callbacks/runtime.py"), 5, "_core.NativeCallbackWorkerThread", "NativeCallbackWorkerThread"),
        (
            Path("g/engine/callbacks/runtime.py"),
            6,
            "_core.NativeCallbackSchedulerState",
            "NativeCallbackSchedulerState",
        ),
        (
            Path("g/engine/callbacks/runtime.py"),
            7,
            "_core.NativeCallbackProgressState",
            "NativeCallbackProgressState",
        ),
        (
            Path("g/engine/callbacks/runtime.py"),
            8,
            "_core.NativeBinaryCorrectionSummary",
            "NativeBinaryCorrectionSummary",
        ),
        (
            Path("g/engine/callbacks/runtime.py"),
            9,
            "_core.NativeDosageBufferPoolState",
            "NativeDosageBufferPoolState",
        ),
        (
            Path("g/engine/callbacks/runtime.py"),
            10,
            "_core.NativeResultInFlightSlotState",
            "NativeResultInFlightSlotState",
        ),
    ]


def test_prepared_plan_policy_rejects_python_plan_reconstruction(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    output_directory = package_root / "io"
    output_directory.mkdir(parents=True)
    (output_directory / "output.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build_prepared_plan(header):",
                "    build_native_prepared_run_plan_input_mapping(header)",
                "    _core.build_prepared_run_plan_json('{}')",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/io/output.py"),
            3,
            "build_native_prepared_run_plan_input_mapping",
            "build_native_prepared_run_plan_input_mapping",
        ),
        (Path("g/io/output.py"), 4, "_core.build_prepared_run_plan_json", "_core.build_prepared_run_plan_json"),
    ]


def test_run_request_policy_routes_native_compile_through_execution_plan(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    execution_plan_path = package_root / "execution_plan.py"
    runner_directory.mkdir(parents=True)
    (runner_directory / "execution.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build(config):",
                "    _core.compile_run_request_json(config)",
                "    _core.compile_run_request_payload(config)",
            )
        ),
        encoding="utf-8",
    )
    execution_plan_path.write_text(
        "\n".join(
            (
                "from g import _core",
                "def build(config):",
                "    _core.compile_run_request_payload(config)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/runner/execution.py"), 3, "_core.compile_run_request_json", "_core.compile_run_request_json"),
        (Path("g/runner/execution.py"), 4, "_core.compile_run_request_payload", "_core.compile_run_request_payload"),
    ]


def test_diagnostic_payload_policy_rejects_direct_payload_builders(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "execution.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "import g",
                "def log_diagnostic(event):",
                "    _core.build_runner_run_started_diagnostic_payload('linear', 'qt', 1)",
                "    g._core.build_jax_runtime_setup_diagnostic_payloads({}, True)",
                "    _core.emit_diagnostic_event('info', 'runner.event', 'message')",
                "    _core.emit_diagnostic_event_fields('info', 'runner.event', 'message', {})",
            )
        ),
        encoding="utf-8",
    )
    (runner_directory / "events.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build_compatibility_payload():",
                "    _core.build_runner_run_started_diagnostic_payload('linear', 'qt', 1)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/runner/execution.py"),
            4,
            "_core.build_runner_run_started_diagnostic_payload",
            "_core.build_*_diagnostic_payload",
        ),
        (
            Path("g/runner/execution.py"),
            5,
            "g._core.build_jax_runtime_setup_diagnostic_payloads",
            "_core.build_*_diagnostic_payloads",
        ),
        (Path("g/runner/execution.py"), 6, "_core.emit_diagnostic_event", "_core.emit_diagnostic_event"),
        (
            Path("g/runner/execution.py"),
            7,
            "_core.emit_diagnostic_event_fields",
            "_core.emit_diagnostic_event_fields",
        ),
    ]


def test_telemetry_dispatch_policy_rejects_fallback_method_calls(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "telemetry.py").write_text(
        "\n".join(
            (
                "def emit(session, event, progress_event):",
                "    session.log_run_failed(event)",
                "    session.close_with_event()",
                "    session.log_jax_runtime_diagnostic_event(event, telemetry_level='trace')",
                "    session.log_callback_progress_event(progress_event)",
                "    session.log_binary_correction_summary({})",
                "    session.log_progress(processed_chunk_count=1)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/runner/telemetry.py"), 2, "session.log_run_failed", "log_run_failed"),
        (Path("g/runner/telemetry.py"), 3, "session.close_with_event", "close_with_event"),
        (
            Path("g/runner/telemetry.py"),
            4,
            "session.log_jax_runtime_diagnostic_event",
            "log_jax_runtime_diagnostic_event",
        ),
        (
            Path("g/runner/telemetry.py"),
            5,
            "session.log_callback_progress_event",
            "log_callback_progress_event",
        ),
        (
            Path("g/runner/telemetry.py"),
            6,
            "session.log_binary_correction_summary",
            "log_binary_correction_summary",
        ),
        (Path("g/runner/telemetry.py"), 7, "session.log_progress", "log_progress"),
    ]


def test_telemetry_dispatch_policy_rejects_direct_handle_and_wrapper_calls(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "telemetry.py").write_text(
        "\n".join(
            (
                "def emit(session):",
                "    session.native_session_handle.emit_current_event('run_started', 'info', {})",
                "    session.native_telemetry_session.emit_progress(1, {})",
                "    handle = session.native_session_handle",
                "    handle.emit_run_failed_event(None)",
                "    writer = session.native_telemetry_session",
                "    writer.emit_payload({})",
                "    session.log_event('run_started', level='info')",
                "    session.build_event_payload(event='run_started', level='info')",
                "    session.write_json_line({})",
                "    session.writer_counters()",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/runner/telemetry.py"),
            2,
            "session.native_session_handle.emit_current_event",
            "native_session_handle.emit_*",
        ),
        (
            Path("g/runner/telemetry.py"),
            3,
            "session.native_telemetry_session.emit_progress",
            "native_telemetry_session.emit_*",
        ),
        (Path("g/runner/telemetry.py"), 5, "handle.emit_run_failed_event", "emit_run_failed_event"),
        (Path("g/runner/telemetry.py"), 7, "writer.emit_payload", "emit_payload"),
        (Path("g/runner/telemetry.py"), 8, "session.log_event", "log_event"),
        (
            Path("g/runner/telemetry.py"),
            9,
            "session.build_event_payload",
            "build_event_payload",
        ),
        (Path("g/runner/telemetry.py"), 10, "session.write_json_line", "write_json_line"),
        (Path("g/runner/telemetry.py"), 11, "session.writer_counters", "writer_counters"),
    ]


def test_jax_cache_resolution_policy_rejects_production_python_resolver_calls(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    jax_runtime_directory = package_root / "jax_runtime"
    runner_directory.mkdir(parents=True)
    jax_runtime_directory.mkdir(parents=True)
    (runner_directory / "runtime.py").write_text(
        "\n".join(
            (
                "from g.jax_runtime import resolution as jax_runtime_resolution",
                "def configure(policy):",
                "    jax_runtime_resolution.resolve_jax_runtime_cache_directory(policy)",
            )
        ),
        encoding="utf-8",
    )
    (jax_runtime_directory / "resolution.py").write_text(
        "\n".join(
            (
                "def resolve_jax_runtime_cache_directory(policy):",
                "    return policy.cache_directory",
                "def resolve_jax_runtime_setup(policy):",
                "    return resolve_jax_runtime_cache_directory(policy)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/jax_runtime/resolution.py"),
            4,
            "resolve_jax_runtime_cache_directory",
            "resolve_jax_runtime_cache_directory",
        ),
        (
            Path("g/runner/runtime.py"),
            3,
            "jax_runtime_resolution.resolve_jax_runtime_cache_directory",
            "resolve_jax_runtime_cache_directory",
        ),
    ]


def test_jax_setup_session_policy_rejects_raw_session_construction(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    jax_runtime_directory = package_root / "jax_runtime"
    jax_runtime_directory.mkdir(parents=True)
    (jax_runtime_directory / "setup.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build_session():",
                "    payload = _core.resolve_jax_runtime_setup_payload(",
                "        requested_device='gpu',",
                "        cache_directory='',",
                "        matmul_precision=None,",
                "        persistent_cache=False,",
                "        persistent_cache_min_entry_size_bytes=0,",
                "        persistent_cache_min_compile_time_seconds=0,",
                "        xla_autotune_cache=False,",
                "        transfer_guard=False,",
                "    )",
                "    return _core.NativeJaxRuntimeSetupSession(payload, should_configure=False)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/jax_runtime/setup.py"),
            3,
            "_core.resolve_jax_runtime_setup_payload",
            "_core.resolve_jax_runtime_setup_payload",
        ),
        (
            Path("g/jax_runtime/setup.py"),
            13,
            "_core.NativeJaxRuntimeSetupSession",
            "_core.NativeJaxRuntimeSetupSession",
        ),
    ]


def test_jax_setup_side_effect_policy_rejects_direct_jax_calls(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    jax_runtime_directory = package_root / "jax_runtime"
    jax_runtime_directory.mkdir(parents=True)
    (jax_runtime_directory / "setup.py").write_text(
        "\n".join(
            (
                "import jax",
                "def configure():",
                "    jax.config.update('jax_platforms', 'cpu')",
                "    jax.devices()",
                "    native_setup_session.side_effect_plan_payload()",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/jax_runtime/setup.py"), 3, "jax.config.update", "jax.config.update"),
        (Path("g/jax_runtime/setup.py"), 4, "jax.devices", "jax.devices"),
        (
            Path("g/jax_runtime/setup.py"),
            5,
            "native_setup_session.side_effect_plan_payload",
            "side_effect_plan_payload",
        ),
    ]


def test_preflight_numeric_scan_policy_rejects_old_numpy_reductions(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    engine_directory = package_root / "engine"
    pipeline_directory = engine_directory / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "preflight.py").write_text(
        "\n".join(
            (
                "import numpy as np",
                "def validate(values):",
                "    np.isfinite(values).all()",
                "    np.unique(values)",
                "    np.count_nonzero(values == 1.0)",
            )
        ),
        encoding="utf-8",
    )
    (engine_directory / "other.py").write_text(
        "\n".join(
            (
                "import numpy as np",
                "def still_allowed_here(values):",
                "    np.unique(values)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    observed_violations = sorted(
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    )

    assert observed_violations == [
        (Path("g/engine/regenie2_pipeline/preflight.py"), 3, "np.isfinite", "np.isfinite"),
        (Path("g/engine/regenie2_pipeline/preflight.py"), 4, "np.unique", "np.unique"),
        (Path("g/engine/regenie2_pipeline/preflight.py"), 5, "np.count_nonzero", "np.count_nonzero"),
    ]


def test_preflight_required_chromosome_policy_rejects_engine_probe(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "preflight.py").write_text(
        "\n".join(
            (
                "def collect(engine):",
                "    required_chromosomes = getattr(engine, 'required_chromosomes', None)",
                "    return required_chromosomes",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/engine/regenie2_pipeline/preflight.py"), 2, "getattr", "getattr"),
    ]


def test_covariate_rank_scan_policy_rejects_matrix_rank_in_production_python(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "preflight.py").write_text(
        "\n".join(
            (
                "import numpy as np",
                "def validate_rank(values):",
                "    np.linalg.matrix_rank(values)",
                "    matrix_rank(values)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/engine/regenie2_pipeline/preflight.py"), 3, "np.linalg.matrix_rank", "np.linalg.matrix_rank"),
        (Path("g/engine/regenie2_pipeline/preflight.py"), 4, "matrix_rank", "matrix_rank"),
    ]


def test_callback_convergence_scan_policy_rejects_old_numpy_reductions(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "diagnostics.py").write_text(
        "\n".join(
            (
                "import numpy as np",
                "def validate(values):",
                "    np.ravel(values)",
                "    np.count_nonzero(~values)",
            )
        ),
        encoding="utf-8",
    )
    (callback_directory / "other.py").write_text(
        "\n".join(
            (
                "import numpy as np",
                "def still_allowed_here(values):",
                "    np.count_nonzero(values)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    observed_violations = sorted(
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    )

    assert observed_violations == [
        (Path("g/engine/callbacks/diagnostics.py"), 3, "np.ravel", "np.ravel"),
        (Path("g/engine/callbacks/diagnostics.py"), 4, "np.count_nonzero", "np.count_nonzero"),
    ]


def test_binary_diagnostics_result_contract_policy_rejects_getattr_probe(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    diagnostics_directory = package_root / "compute" / "regenie2_binary"
    diagnostics_directory.mkdir(parents=True)
    (diagnostics_directory / "diagnostics.py").write_text(
        "\n".join(
            (
                "def count(result):",
                "    return getattr(result, 'firth_iteration_count', None)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/compute/regenie2_binary/diagnostics.py"), 2, "getattr", "getattr"),
    ]


def test_binary_diagnostics_host_materialization_policy_rejects_device_get(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    diagnostics_directory = package_root / "compute" / "regenie2_binary"
    diagnostics_directory.mkdir(parents=True)
    (diagnostics_directory / "diagnostics.py").write_text(
        "\n".join(
            (
                "import jax",
                "def materialize(values):",
                "    return jax.device_get(values)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/compute/regenie2_binary/diagnostics.py"), 3, "jax.device_get", "jax.device_get"),
    ]


def test_callback_readiness_blocker_policy_rejects_optional_method_probe(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "diagnostics.py").write_text(
        "\n".join(
            (
                "def block(value):",
                "    block_until_ready_method = getattr(value, 'block_until_ready', None)",
                "    if block_until_ready_method is not None:",
                "        block_until_ready_method()",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/engine/callbacks/diagnostics.py"), 2, "getattr", "getattr"),
    ]


def test_callback_chromosome_state_policy_rejects_single_trait_readiness_probe(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "binary.py").write_text(
        "\n".join(
            (
                "def prepare_binary(state):",
                "    ready = getattr(state, 'score_residual', state)",
                "    return ready",
            )
        ),
        encoding="utf-8",
    )
    (callback_directory / "linear.py").write_text(
        "\n".join(
            (
                "def prepare_linear(state):",
                "    ready = getattr(state, 'adjusted_residual', state)",
                "    return ready",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in sorted(violations, key=lambda violation: violation.path)
    ] == [
        (Path("g/engine/callbacks/binary.py"), 2, "getattr", "getattr"),
        (Path("g/engine/callbacks/linear.py"), 2, "getattr", "getattr"),
    ]


def test_callback_transfer_contract_policy_rejects_optional_transfer_probes(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "transfers.py").write_text(
        "\n".join(
            (
                "def metadata(array, chunk_stats):",
                "    shape = getattr(array, 'shape', None)",
                "    compute_arrays = getattr(chunk_stats, 'compute_arrays', None)",
                "    return shape, compute_arrays",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/engine/callbacks/transfers.py"), 2, "getattr", "getattr"),
        (Path("g/engine/callbacks/transfers.py"), 3, "getattr", "getattr"),
    ]


def test_callback_writer_contract_policy_rejects_method_name_probe(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "writers.py").write_text(
        "\n".join(
            (
                "def write(writer_session, write_plan):",
                "    write_method = getattr(writer_session, write_plan.method_name)",
                "    write_method()",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/engine/callbacks/writers.py"), 2, "getattr", "getattr"),
    ]


def test_delivery_callback_contract_policy_rejects_optional_callback_probe(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    native_dispatch_directory = package_root / "engine" / "native_dispatch"
    native_dispatch_directory.mkdir(parents=True)
    (native_dispatch_directory / "delivery.py").write_text(
        "\n".join(
            (
                "def plan(callback):",
                "    return getattr(callback, 'native_callback_batch_size', None)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/engine/native_dispatch/delivery.py"), 2, "getattr", "getattr"),
    ]


def test_native_dispatch_callback_lifecycle_policy_rejects_optional_hook_probe(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    native_dispatch_directory = package_root / "engine" / "native_dispatch"
    native_dispatch_directory.mkdir(parents=True)
    (native_dispatch_directory / "writers.py").write_text(
        "\n".join(
            (
                "def start(callback):",
                "    start_method = getattr(callback, 'start', None)",
                "    if start_method is not None:",
                "        start_method()",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/engine/native_dispatch/writers.py"), 2, "getattr", "getattr"),
    ]


def test_grouped_callback_fanout_policy_rejects_optional_lifecycle_probe(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "grouped.py").write_text(
        "\n".join(
            (
                "def start(callback):",
                "    start_method = getattr(callback, 'start', None)",
                "    if start_method is not None:",
                "        start_method()",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/engine/callbacks/grouped.py"), 2, "getattr", "getattr"),
    ]


def test_callback_metadata_chromosome_policy_rejects_optional_metadata_probe(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "shared.py").write_text(
        "\n".join(
            (
                "def chromosome(metadata):",
                "    chromosome_label = getattr(metadata, 'chromosome_label', None)",
                "    return chromosome_label",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/engine/callbacks/shared.py"), 2, "getattr", "getattr"),
    ]


def test_timing_snapshot_serialization_policy_rejects_getattr_probe(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    timing_directory = package_root / "engine"
    timing_directory.mkdir(parents=True)
    (timing_directory / "timing.py").write_text(
        "\n".join(
            (
                "def serialize(snapshot, field):",
                "    return getattr(snapshot, field)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/engine/timing.py"), 2, "getattr", "getattr"),
    ]


def test_jax_host_materialization_policy_rejects_device_get_outside_adapters(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    engine_directory = package_root / "engine"
    engine_directory.mkdir(parents=True)
    (engine_directory / "pipeline.py").write_text(
        "\n".join(
            (
                "import jax",
                "def materialize(values):",
                "    jax.device_get(values)",
                "    device_get(values)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/engine/pipeline.py"), 3, "jax.device_get", "jax.device_get"),
        (Path("g/engine/pipeline.py"), 4, "device_get", "device_get"),
    ]


def test_telemetry_definition_policy_rejects_fallback_methods(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    telemetry_directory = package_root / "engine"
    telemetry_directory.mkdir(parents=True)
    (telemetry_directory / "telemetry.py").write_text(
        "\n".join(
            (
                "class TelemetrySession:",
                "    def log_run_failed(self, event):",
                "        pass",
                "    def close_with_event(self):",
                "        pass",
                "    def log_event(self):",
                "        pass",
                "    def log_run_started(self):",
                "        pass",
                "    def build_event_payload(self):",
                "        pass",
                "    def native_session_policy(self):",
                "        pass",
                "async def log_progress(processed_chunk_count):",
                "    pass",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_definition_policy_violations(package_root)

    observed_violations = sorted(
        (violation.path, violation.line_number, violation.function_name) for violation in violations
    )

    assert observed_violations == [
        (Path("g/engine/telemetry.py"), 2, "log_run_failed"),
        (Path("g/engine/telemetry.py"), 4, "close_with_event"),
        (Path("g/engine/telemetry.py"), 6, "log_event"),
        (Path("g/engine/telemetry.py"), 8, "log_run_started"),
        (Path("g/engine/telemetry.py"), 10, "build_event_payload"),
        (Path("g/engine/telemetry.py"), 12, "native_session_policy"),
        (Path("g/engine/telemetry.py"), 14, "log_progress"),
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


def test_jax_runtime_import_policy_rejects_host_orchestration_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    jax_runtime_directory = package_root / "jax_runtime"
    jax_runtime_directory.mkdir(parents=True)
    (jax_runtime_directory / "setup.py").write_text(
        "\n".join(
            (
                "from g import api",
                "from g import cli",
                "from g.compute import common",
                "from g.engine import run_events",
                "from g import execution_plan",
                "from g.interface import config",
                "from g.io import output",
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
        (Path("g/jax_runtime/setup.py"), 1, "g.api", "g.api"),
        (Path("g/jax_runtime/setup.py"), 2, "g.cli", "g.cli"),
        (Path("g/jax_runtime/setup.py"), 3, "g.compute.common", "g.compute"),
        (Path("g/jax_runtime/setup.py"), 4, "g.engine.run_events", "g.engine"),
        (Path("g/jax_runtime/setup.py"), 5, "g.execution_plan", "g.execution_plan"),
        (Path("g/jax_runtime/setup.py"), 6, "g.interface.config", "g.interface"),
        (Path("g/jax_runtime/setup.py"), 7, "g.io.output", "g.io"),
        (Path("g/jax_runtime/setup.py"), 8, "g.runner.runtime", "g.runner"),
        (Path("g/jax_runtime/setup.py"), 9, "g.runner.cli", "g.runner"),
    ]


def test_runner_import_policy_rejects_jax_facing_pipeline_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "execution.py").write_text(
        "\n".join(
            (
                "import g.engine.regenie2_pipeline.single_trait",
                "from g.engine.callbacks import linear",
                "from g.compute.regenie2_binary import api",
                "from ..engine.regenie2_pipeline import multi_trait",
                "import jax",
                "from jax import numpy",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (
            Path("g/runner/execution.py"),
            1,
            "g.engine.regenie2_pipeline.single_trait",
            "g.engine.regenie2_pipeline",
        ),
        (Path("g/runner/execution.py"), 2, "g.engine.callbacks.linear", "g.engine.callbacks"),
        (Path("g/runner/execution.py"), 3, "g.compute.regenie2_binary.api", "g.compute"),
        (
            Path("g/runner/execution.py"),
            4,
            "g.engine.regenie2_pipeline.multi_trait",
            "g.engine.regenie2_pipeline",
        ),
        (Path("g/runner/execution.py"), 5, "jax", "jax"),
        (Path("g/runner/execution.py"), 6, "jax.numpy", "jax"),
    ]
