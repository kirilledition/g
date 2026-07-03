"""Architecture tests for Python package ownership boundaries."""

from __future__ import annotations

from pathlib import Path

from tooling.debug import check_python_architecture

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "g"


def test_python_import_policy_allows_current_production_tree() -> None:
    assert check_python_architecture.collect_python_import_policy_violations(PRODUCTION_PACKAGE_ROOT) == ()


def test_python_forbidden_path_policy_allows_current_production_tree() -> None:
    assert check_python_architecture.collect_python_forbidden_path_policy_violations(PRODUCTION_PACKAGE_ROOT) == ()


def test_python_alias_policy_allows_current_production_tree() -> None:
    assert check_python_architecture.collect_python_alias_policy_violations(PRODUCTION_PACKAGE_ROOT) == ()


def test_forbidden_path_policy_rejects_obsolete_python_orchestration_files(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    for source_path in (
        Path("io/__init__.py"),
        Path("io/output.py"),
        Path("engine/run_events.py"),
        Path("engine/shutdown.py"),
        Path("engine/telemetry.py"),
        Path("engine/timing.py"),
        Path("engine/callbacks/events.py"),
        Path("engine/callbacks/timing.py"),
        Path("engine/native_dispatch/events.py"),
        Path("engine/native_dispatch/lifecycle.py"),
        Path("engine/native_dispatch/timing.py"),
        Path("engine/regenie2_pipeline/timing.py"),
    ):
        path = package_root / source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    violations = check_python_architecture.collect_python_forbidden_path_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.policy_name, violation.forbidden_path)
        for violation in violations
    ] == [
        (
            Path("g/io/__init__.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("io/__init__.py"),
        ),
        (
            Path("g/io/output.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("io/output.py"),
        ),
        (
            Path("g/engine/run_events.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("engine/run_events.py"),
        ),
        (
            Path("g/engine/shutdown.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("engine/shutdown.py"),
        ),
        (
            Path("g/engine/telemetry.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("engine/telemetry.py"),
        ),
        (
            Path("g/engine/timing.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("engine/timing.py"),
        ),
        (
            Path("g/engine/callbacks/events.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("engine/callbacks/events.py"),
        ),
        (
            Path("g/engine/callbacks/timing.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("engine/callbacks/timing.py"),
        ),
        (
            Path("g/engine/native_dispatch/events.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("engine/native_dispatch/events.py"),
        ),
        (
            Path("g/engine/native_dispatch/lifecycle.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("engine/native_dispatch/lifecycle.py"),
        ),
        (
            Path("g/engine/native_dispatch/timing.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("engine/native_dispatch/timing.py"),
        ),
        (
            Path("g/engine/regenie2_pipeline/timing.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("engine/regenie2_pipeline/timing.py"),
        ),
    ]


def test_forbidden_path_policy_rejects_obsolete_support_files(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    for source_path in (
        Path("runtime_paths.py"),
        Path("jax_runtime/state.py"),
        Path("io/source.py"),
        Path("engine/backend_planner.py"),
        Path("engine/preflight.py"),
        Path("engine/preflight_events.py"),
        Path("engine/trusted_validation.py"),
        Path("engine/warm_cache.py"),
    ):
        path = package_root / source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    violations = check_python_architecture.collect_python_forbidden_path_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.policy_name, violation.forbidden_path)
        for violation in violations
    ] == [
        (
            Path("g/io/source.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("io/source.py"),
        ),
        (
            Path("g/jax_runtime/state.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("jax_runtime/state.py"),
        ),
        (
            Path("g/runtime_paths.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("runtime_paths.py"),
        ),
        (
            Path("g/engine/backend_planner.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("engine/backend_planner.py"),
        ),
        (
            Path("g/engine/preflight.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("engine/preflight.py"),
        ),
        (
            Path("g/engine/preflight_events.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("engine/preflight_events.py"),
        ),
        (
            Path("g/engine/trusted_validation.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("engine/trusted_validation.py"),
        ),
        (
            Path("g/engine/warm_cache.py"),
            1,
            "obsolete_python_orchestration_module_path_isolation",
            Path("engine/warm_cache.py"),
        ),
    ]


def test_forbidden_path_violation_renderer_includes_policy_path_and_message() -> None:
    violation = check_python_architecture.PythonForbiddenPathViolation(
        path=Path("g/engine/timing.py"),
        line_number=1,
        column_offset=0,
        policy_name="obsolete_python_orchestration_module_path_isolation",
        forbidden_path=Path("engine/timing.py"),
        message="obsolete Python orchestration modules must not be reintroduced after runner ownership moved",
    )

    assert check_python_architecture.render_forbidden_path_violation(violation) == (
        "g/engine/timing.py:1:1: obsolete_python_orchestration_module_path_isolation rejects "
        "`g/engine/timing.py` via `engine/timing.py`: obsolete Python orchestration modules must not be "
        "reintroduced after runner ownership moved"
    )


def test_python_call_policy_allows_current_production_tree() -> None:
    assert check_python_architecture.collect_python_call_policy_violations(PRODUCTION_PACKAGE_ROOT) == ()


def test_call_policy_rejects_dynamic_python_fallbacks(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    package_root.mkdir()
    (package_root / "dynamic.py").write_text(
        "\n".join(
            (
                "import json",
                "def bad(value):",
                "    getattr(value, 'field')",
                "    hasattr(value, 'field')",
                "    setattr(value, 'field', 1)",
                "    json.dumps({'field': 1})",
                "    json.loads('{\"field\": 1}')",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.policy_name, violation.call_name) for violation in violations
    ] == [
        (Path("g/dynamic.py"), 3, "dynamic_python_fallback_isolation", "getattr"),
        (Path("g/dynamic.py"), 4, "dynamic_python_fallback_isolation", "hasattr"),
        (Path("g/dynamic.py"), 5, "dynamic_python_fallback_isolation", "setattr"),
        (Path("g/dynamic.py"), 6, "dynamic_python_fallback_isolation", "json.dumps"),
        (Path("g/dynamic.py"), 7, "dynamic_python_fallback_isolation", "json.loads"),
    ]


def test_python_parameter_policy_allows_current_production_tree() -> None:
    assert check_python_architecture.collect_python_parameter_policy_violations(PRODUCTION_PACKAGE_ROOT) == ()


def test_python_definition_policy_allows_current_production_tree() -> None:
    assert check_python_architecture.collect_python_definition_policy_violations(PRODUCTION_PACKAGE_ROOT) == ()


def test_python_cli_shim_policy_allows_current_production_tree() -> None:
    assert check_python_architecture.collect_python_cli_shim_violations(PRODUCTION_PACKAGE_ROOT) == ()


def test_runner_runtime_pipeline_kwargs_keep_typed_contracts() -> None:
    source = (PRODUCTION_PACKAGE_ROOT / "runner" / "runtime.py").read_text(encoding="utf-8")

    forbidden_annotations = (
        "output_run_paths: object",
        "output_run_paths_by_phenotype: tuple[object, ...]",
        "writer_settings: object",
        "alignment_config: object | None",
        "linear_numerical_config: object | None",
        "kernel_config: object",
    )

    for forbidden_annotation in forbidden_annotations:
        assert forbidden_annotation not in source

    assert "outputs.OutputRunPaths" in source
    assert "outputs.OutputWriterSettings" in source
    assert "config.GComputeConfig" in source
    assert "execution_plan.LinearNumericalConfig" in source
    assert "execution_plan.BinaryKernelConfig" in source


def test_public_package_lazy_exports_are_explicit() -> None:
    source = (PRODUCTION_PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8")

    assert "typing.Any" not in source
    assert "getattr(" not in source


def test_interface_config_uses_native_static_constructor() -> None:
    source = (PRODUCTION_PACKAGE_ROOT / "interface" / "config.py").read_text(encoding="utf-8")

    assert "setattr(RegenieConfig" not in source
    assert "return RegenieConfig.from_options(raw_options)" in source


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
    output_directory = package_root / "runner"
    output_directory.mkdir(parents=True)
    (output_directory / "outputs.py").write_text(
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
        (Path("g/runner/outputs.py"), 1, "g.jax_runtime.models", "g.jax_runtime"),
        (Path("g/runner/outputs.py"), 2, "g.jax_runtime.setup", "g.jax_runtime"),
    ]


def test_output_import_policy_rejects_engine_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    output_directory = package_root / "runner"
    output_directory.mkdir(parents=True)
    (output_directory / "outputs.py").write_text(
        "\n".join(
            (
                "from g.engine import orchestration",
                "import g.engine.unowned_adapter",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/runner/outputs.py"), 1, "g.engine.orchestration", "g.engine"),
        (Path("g/runner/outputs.py"), 2, "g.engine.unowned_adapter", "g.engine"),
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
                "from g import _core",
                "_core.NativeOutputLifecyclePolicy()",
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

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/runner/events.py"), 1, "g.engine.run_events", "g.engine.run_events"),
        (Path("g/runner/events.py"), 1, "g.engine.telemetry", "g.engine.telemetry"),
        (Path("g/runner/events.py"), 2, "g.engine.run_events", "g.engine.run_events"),
        (Path("g/runner/events.py"), 3, "g.engine.telemetry", "g.engine.telemetry"),
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
                "from g import _core",
                "_core.NativeRunEventPayloadPolicy()",
                "_core.NativeRunEventTelemetryPolicy()",
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
    (runner_directory / "lifecycle.py").write_text(
        "\n".join(
            (
                "from g.engine import shutdown",
                "import g.engine.shutdown",
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
        (Path("g/runner/execution.py"), 2, "g.engine.shutdown", "g.engine.shutdown"),
        (Path("g/runner/lifecycle.py"), 1, "g.engine.shutdown", "g.engine.shutdown"),
        (Path("g/runner/lifecycle.py"), 2, "g.engine.shutdown", "g.engine.shutdown"),
        (Path("g/runner/execution.py"), 1, "g.engine.timing", "g.engine.timing"),
        (Path("g/runner/execution.py"), 3, "g.engine.timing", "g.engine.timing"),
    ]


def test_runner_import_policy_allows_runner_lifecycle_timing_adapters(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "lifecycle.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "_core.NativeShutdownController()",
            )
        ),
        encoding="utf-8",
    )
    (runner_directory / "timing.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "_core.NativeStageTimingRecorder(False)",
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
                "from g.engine.native_dispatch import events, lifecycle, timing as dispatch_timing",
                "import g.engine.run_events",
                "import g.engine.shutdown",
                "import g.engine.timing",
                "import g.engine.native_dispatch.events",
                "import g.engine.native_dispatch.lifecycle",
                "import g.engine.native_dispatch.timing",
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
        (
            Path("g/engine/native_dispatch/delivery.py"),
            2,
            "g.engine.native_dispatch.events",
            "g.engine.native_dispatch.events",
        ),
        (
            Path("g/engine/native_dispatch/delivery.py"),
            2,
            "g.engine.native_dispatch.lifecycle",
            "g.engine.native_dispatch.lifecycle",
        ),
        (Path("g/engine/native_dispatch/delivery.py"), 3, "g.engine.run_events", "g.engine.run_events"),
        (Path("g/engine/native_dispatch/delivery.py"), 4, "g.engine.shutdown", "g.engine.shutdown"),
        (
            Path("g/engine/native_dispatch/delivery.py"),
            6,
            "g.engine.native_dispatch.events",
            "g.engine.native_dispatch.events",
        ),
        (
            Path("g/engine/native_dispatch/delivery.py"),
            7,
            "g.engine.native_dispatch.lifecycle",
            "g.engine.native_dispatch.lifecycle",
        ),
        (Path("g/engine/native_dispatch/delivery.py"), 1, "g.engine.timing", "g.engine.timing"),
        (
            Path("g/engine/native_dispatch/delivery.py"),
            2,
            "g.engine.native_dispatch.timing",
            "g.engine.native_dispatch.timing",
        ),
        (Path("g/engine/native_dispatch/delivery.py"), 5, "g.engine.timing", "g.engine.timing"),
        (
            Path("g/engine/native_dispatch/delivery.py"),
            8,
            "g.engine.native_dispatch.timing",
            "g.engine.native_dispatch.timing",
        ),
    ]


def test_native_dispatch_import_policy_allows_runner_event_lifecycle_timing_helpers(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    native_dispatch_directory = package_root / "engine" / "native_dispatch"
    native_dispatch_directory.mkdir(parents=True)
    (native_dispatch_directory / "delivery.py").write_text(
        "\n".join(
            (
                "from g.runner import events, lifecycle, timing",
                "events.native_dispatch_diagnostic_policy()",
                "GracefulShutdownRequested = lifecycle.GracefulShutdownRequested",
                "StageTimingRecorder = timing.StageTimingRecorder",
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
                "from g.runner import outputs as runner_outputs",
                "OutputRunPaths = runner_outputs.OutputRunPaths",
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
                "from g import _core",
                "_core.NativeRunEventTelemetryPolicy()",
                "_core.NativePipelineDiagnosticPolicy()",
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
                "from g.engine.regenie2_pipeline import timing as pipeline_timing",
                "import g.engine.timing",
                "import g.engine.regenie2_pipeline.timing",
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
        (
            Path("g/engine/regenie2_pipeline/single_trait.py"),
            2,
            "g.engine.regenie2_pipeline.timing",
            "g.engine.regenie2_pipeline.timing",
        ),
        (Path("g/engine/regenie2_pipeline/single_trait.py"), 3, "g.engine.timing", "g.engine.timing"),
        (
            Path("g/engine/regenie2_pipeline/single_trait.py"),
            4,
            "g.engine.regenie2_pipeline.timing",
            "g.engine.regenie2_pipeline.timing",
        ),
    ]


def test_pipeline_import_policy_allows_runner_timing_helper(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "single_trait.py").write_text(
        "\n".join(
            (
                "from g.runner import timing",
                "StageTimingRecorder = timing.StageTimingRecorder",
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
                "from g.engine.callbacks import timing as callback_timing",
                "import g.engine.timing",
                "import g.engine.callbacks.timing",
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
        (Path("g/engine/callbacks/runtime.py"), 2, "g.engine.callbacks.timing", "g.engine.callbacks.timing"),
        (Path("g/engine/callbacks/runtime.py"), 3, "g.engine.timing", "g.engine.timing"),
        (Path("g/engine/callbacks/runtime.py"), 4, "g.engine.callbacks.timing", "g.engine.callbacks.timing"),
    ]


def test_callback_import_policy_allows_runner_timing_helper(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "runtime.py").write_text(
        "\n".join(
            (
                "from g.runner import timing",
                "StageTimingRecorder = timing.StageTimingRecorder",
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
                "from g.engine.callbacks import events",
                "import g.engine.run_events",
                "import g.engine.telemetry",
                "import g.engine.callbacks.events",
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
        (Path("g/engine/callbacks/diagnostics.py"), 2, "g.engine.callbacks.events", "g.engine.callbacks.events"),
        (Path("g/engine/callbacks/diagnostics.py"), 3, "g.engine.run_events", "g.engine.run_events"),
        (Path("g/engine/callbacks/diagnostics.py"), 4, "g.engine.telemetry", "g.engine.telemetry"),
        (Path("g/engine/callbacks/diagnostics.py"), 5, "g.engine.callbacks.events", "g.engine.callbacks.events"),
    ]


def test_callback_import_policy_allows_runner_event_helper(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "diagnostics.py").write_text(
        "\n".join(
            (
                "from g.runner import events",
                "events.native_pipeline_diagnostic_policy()",
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
    runner_directory.mkdir(parents=True)
    preflight_directory.mkdir(parents=True, exist_ok=True)
    pipeline_directory.mkdir(parents=True)
    for path, call_name in (
        (runner_directory / "events.py", "_core.NativeRunEventPayloadPolicy()"),
        (runner_directory / "events.py", "_core.NativeRunEventTelemetryPolicy()"),
        (runner_directory / "events.py", "_core.NativeRunnerDiagnosticPolicy()"),
        (runner_directory / "events.py", "_core.NativePipelineDiagnosticPolicy()"),
        (runner_directory / "events.py", "_core.NativeDispatchDiagnosticPolicy()"),
        (pipeline_directory / "preflight.py", "_core.NativeOutputPreflightDiagnosticPolicy()"),
        (pipeline_directory / "telemetry_events.py", "_core.NativePipelineDiagnosticPolicy()"),
        (pipeline_directory / "telemetry_events.py", "_core.NativeRunEventTelemetryPolicy()"),
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
    runner_directory.mkdir(parents=True)
    (runner_directory / "metadata.py").write_text(
        "\n".join(
            (
                "from g.runner import outputs",
                "from g import _core",
                "def extend(paths, manifest):",
                "    outputs.write_run_manifest(paths, manifest)",
                "    _core.write_run_manifest_json('run', '{}')",
                "    _core.write_run_manifest('run', {})",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/runner/metadata.py"), 4, "outputs.write_run_manifest", "write_run_manifest"),
        (Path("g/runner/metadata.py"), 5, "_core.write_run_manifest_json", "_core.write_run_manifest_json"),
        (Path("g/runner/metadata.py"), 6, "_core.write_run_manifest", "_core.write_run_manifest"),
    ]


def test_output_lifecycle_policy_rejects_direct_native_output_calls(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "metadata.py").write_text(
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
        (Path("g/runner/metadata.py"), 3, "_core.prepare_output_run", "_core.prepare_output_run"),
        (Path("g/runner/metadata.py"), 4, "_core.load_run_manifest_payload", "_core.load_run_manifest_payload"),
        (Path("g/runner/metadata.py"), 5, "_core.initialize_output_run", "_core.initialize_output_run"),
        (
            Path("g/runner/metadata.py"),
            6,
            "_core.initialize_output_run_from_values",
            "_core.initialize_output_run_from_values",
        ),
        (
            Path("g/runner/metadata.py"),
            7,
            "_core.NativeOutputLifecyclePolicy",
            "_core.NativeOutputLifecyclePolicy",
        ),
        (
            Path("g/runner/metadata.py"),
            8,
            "_core.validate_strict_manifest_chunks",
            "_core.validate_strict_manifest_chunks",
        ),
        (
            Path("g/runner/metadata.py"),
            9,
            "_core.validate_strict_manifest_chunks_from_value",
            "_core.validate_strict_manifest_chunks_from_value",
        ),
        (
            Path("g/runner/metadata.py"),
            10,
            "_core.repair_strict_manifest_chunk_commits_from_value",
            "_core.repair_strict_manifest_chunk_commits_from_value",
        ),
        (
            Path("g/runner/metadata.py"),
            11,
            "_core.read_manifest_committed_chunk_identifiers_from_value",
            "_core.read_manifest_committed_chunk_identifiers_from_value",
        ),
        (
            Path("g/runner/metadata.py"),
            12,
            "_core.validate_run_manifest_compatibility_from_values",
            "_core.validate_run_manifest_compatibility_from_values",
        ),
        (Path("g/runner/metadata.py"), 13, "_core.finalize_output_run_chunks", "_core.finalize_output_run_chunks"),
        (
            Path("g/runner/metadata.py"),
            14,
            "_core.build_pipeline_output_preparation_batch_from_values",
            "_core.build_pipeline_output_preparation_batch_from_values",
        ),
        (
            Path("g/runner/metadata.py"),
            15,
            "_core.NativePipelineOutputPreparationBatch",
            "_core.NativePipelineOutputPreparationBatch",
        ),
        (
            Path("g/runner/metadata.py"),
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
                "    writer_session.finish()",
                "    writer_session.finish_interrupted('SIGINT')",
                "    writer_session.abort()",
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
                "    writer_session.finish()",
                "    writer_session.finish_interrupted('SIGINT')",
                "    writer_session.abort()",
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
        (
            Path("g/runner/cleanup.py"),
            7,
            "writer_session.finish",
            "writer_session.finish",
        ),
        (
            Path("g/runner/cleanup.py"),
            8,
            "writer_session.finish_interrupted",
            "writer_session.finish_interrupted",
        ),
        (
            Path("g/runner/cleanup.py"),
            9,
            "writer_session.abort",
            "writer_session.abort",
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
                "    policy.build_prepared_run_plan_json_from_current_header({})",
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
            "policy.build_prepared_run_plan_json_from_current_header",
            "build_prepared_run_plan_json_from_current_header",
        ),
        (
            Path("g/runner/metadata.py"),
            15,
            "_core.build_manifest_file_fingerprint_payload",
            "_core.build_manifest_file_fingerprint_payload",
        ),
    ]


def test_runner_output_policy_rejects_fingerprint_payload_methods(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "outputs.py").write_text(
        "\n".join(
            (
                "def build(cache):",
                "    cache.build_file_fingerprint_payload('input.bgen', True)",
                "    cache.build_prediction_loco_file_fingerprints_payload('pred.list', ['phenotype'])",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.policy_name, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/runner/outputs.py"),
            2,
            "runner_output_fingerprint_payload_isolation",
            "cache.build_file_fingerprint_payload",
            "build_file_fingerprint_payload",
        ),
        (
            Path("g/runner/outputs.py"),
            3,
            "runner_output_fingerprint_payload_isolation",
            "cache.build_prediction_loco_file_fingerprints_payload",
            "build_prediction_loco_file_fingerprints_payload",
        ),
    ]


def test_runner_timing_policy_rejects_snapshot_payload_method(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "timing.py").write_text(
        "\n".join(
            (
                "def build(recorder):",
                "    recorder.snapshot_payload()",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.policy_name, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/runner/timing.py"),
            2,
            "runner_timing_snapshot_payload_isolation",
            "recorder.snapshot_payload",
            "snapshot_payload",
        )
    ]


def test_runner_runtime_policy_rejects_runtime_payload_methods(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "runtime.py").write_text(
        "\n".join(
            (
                "def build(native_policy, runtime_state, policy):",
                "    native_policy.logging_runtime_policy_payload()",
                "    native_policy.jax_runtime_policy_payload()",
                "    runtime_state.build_logging_runtime_policy_payload()",
                "    runtime_state.build_jax_runtime_policy_payload()",
                "    runtime_state.runtime_state_payload()",
                "    logging_runtime_policy_to_native_payload(policy)",
                "    resolution.jax_runtime_policy_to_native_payload(policy)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.policy_name, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/runner/runtime.py"),
            2,
            "runner_runtime_policy_payload_isolation",
            "native_policy.logging_runtime_policy_payload",
            "logging_runtime_policy_payload",
        ),
        (
            Path("g/runner/runtime.py"),
            3,
            "runner_runtime_policy_payload_isolation",
            "native_policy.jax_runtime_policy_payload",
            "jax_runtime_policy_payload",
        ),
        (
            Path("g/runner/runtime.py"),
            4,
            "runner_runtime_policy_payload_isolation",
            "runtime_state.build_logging_runtime_policy_payload",
            "build_logging_runtime_policy_payload",
        ),
        (
            Path("g/runner/runtime.py"),
            5,
            "runner_runtime_policy_payload_isolation",
            "runtime_state.build_jax_runtime_policy_payload",
            "build_jax_runtime_policy_payload",
        ),
        (
            Path("g/runner/runtime.py"),
            6,
            "runner_runtime_policy_payload_isolation",
            "runtime_state.runtime_state_payload",
            "runtime_state_payload",
        ),
        (
            Path("g/runner/runtime.py"),
            7,
            "runner_runtime_policy_payload_isolation",
            "logging_runtime_policy_to_native_payload",
            "logging_runtime_policy_to_native_payload",
        ),
        (
            Path("g/runner/runtime.py"),
            8,
            "runner_runtime_policy_payload_isolation",
            "resolution.jax_runtime_policy_to_native_payload",
            "jax_runtime_policy_to_native_payload",
        ),
    ]


def test_jax_runtime_setup_policy_rejects_payload_methods(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    jax_runtime_directory = package_root / "jax_runtime"
    jax_runtime_directory.mkdir(parents=True)
    (jax_runtime_directory / "setup.py").write_text(
        "\n".join(
            (
                "def build(native_session, payload):",
                "    native_session.setup_payload()",
                "    native_session.complete_validation_payload('succeeded', None)",
                "    native_session.diagnostic_event_payloads()",
                "    jax_runtime_setup_report_from_native_payload(payload)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.policy_name, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/jax_runtime/setup.py"),
            2,
            "jax_runtime_setup_payload_isolation",
            "native_session.setup_payload",
            "setup_payload",
        ),
        (
            Path("g/jax_runtime/setup.py"),
            3,
            "jax_runtime_setup_payload_isolation",
            "native_session.complete_validation_payload",
            "complete_validation_payload",
        ),
        (
            Path("g/jax_runtime/setup.py"),
            4,
            "jax_runtime_setup_payload_isolation",
            "native_session.diagnostic_event_payloads",
            "diagnostic_event_payloads",
        ),
        (
            Path("g/jax_runtime/setup.py"),
            5,
            "jax_runtime_setup_payload_isolation",
            "jax_runtime_setup_report_from_native_payload",
            "jax_runtime_setup_report_from_native_payload",
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
        (
            Path("g/runner/metadata.py"),
            4,
            "_core.build_execution_run_artifacts_payload",
            "build_execution_run_artifacts_payload",
        ),
    ]


def test_runner_metadata_policy_rejects_artifact_payload_method(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "metadata.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build_metadata():",
                "    builder = _core.NativeRunMetadataBuilder()",
                "    builder.build_execution_run_artifacts_payload(",
                "        'regenie2_linear', 1, 'parquet', (), (), (), (), ()",
                "    )",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.policy_name, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/runner/metadata.py"),
            4,
            "runner_metadata_payload_isolation",
            "builder.build_execution_run_artifacts_payload",
            "build_execution_run_artifacts_payload",
        )
    ]


def test_runner_metadata_policy_rejects_direct_effective_config_writes(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "metadata.py").write_text(
        "\n".join(
            (
                "from g.interface import config",
                "import g.interface.config as interface_config",
                "def write_metadata(regenie_config, path):",
                "    config.write_toml(regenie_config, path)",
                "    interface_config.write_toml(regenie_config, path)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (Path("g/runner/metadata.py"), 4, "config.write_toml", "write_toml"),
        (Path("g/runner/metadata.py"), 5, "interface_config.write_toml", "write_toml"),
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
    output_directory = package_root / "runner"
    output_directory.mkdir(parents=True)
    (output_directory / "outputs.py").write_text(
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
            Path("g/runner/outputs.py"),
            3,
            "build_native_prepared_run_plan_input_mapping",
            "build_native_prepared_run_plan_input_mapping",
        ),
        (Path("g/runner/outputs.py"), 4, "_core.build_prepared_run_plan_json", "_core.build_prepared_run_plan_json"),
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
                "    _core.compile_run_request(config)",
            )
        ),
        encoding="utf-8",
    )
    execution_plan_path.write_text(
        "\n".join(
            (
                "from g import _core",
                "def build(config):",
                "    _core.compile_run_request(config)",
                "    _core.compile_run_request_payload(config)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.policy_name, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/execution_plan.py"),
            4,
            "native_run_request_payload_isolation",
            "_core.compile_run_request_payload",
            "_core.compile_run_request_payload",
        ),
        (
            Path("g/runner/execution.py"),
            3,
            "native_run_request_payload_isolation",
            "_core.compile_run_request_json",
            "_core.compile_run_request_json",
        ),
        (
            Path("g/runner/execution.py"),
            4,
            "native_run_request_payload_isolation",
            "_core.compile_run_request_payload",
            "_core.compile_run_request_payload",
        ),
        (
            Path("g/runner/execution.py"),
            5,
            "native_run_request_adapter_isolation",
            "_core.compile_run_request",
            "_core.compile_run_request",
        ),
    ]


def test_host_planning_policy_rejects_payload_methods(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "backend.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build():",
                "    policy = _core.NativeHostPlanningPolicy()",
                "    policy.plan_association_backend_payload('linear', 'gpu', 'packed8')",
                "    policy.build_phenotype_compute_groups_payload(('a',), 'per-phenotype')",
                "    policy.normalize_binary_correction_payload(False, False, False, 0.05, False)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.policy_name, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/engine/regenie2_pipeline/backend.py"),
            4,
            "host_planning_payload_isolation",
            "policy.plan_association_backend_payload",
            "plan_association_backend_payload",
        ),
        (
            Path("g/engine/regenie2_pipeline/backend.py"),
            5,
            "host_planning_payload_isolation",
            "policy.build_phenotype_compute_groups_payload",
            "build_phenotype_compute_groups_payload",
        ),
        (
            Path("g/engine/regenie2_pipeline/backend.py"),
            6,
            "host_planning_payload_isolation",
            "policy.normalize_binary_correction_payload",
            "normalize_binary_correction_payload",
        ),
    ]


def test_preflight_policy_rejects_payload_methods(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    pipeline_directory = package_root / "engine" / "regenie2_pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "preflight.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build():",
                "    validator = _core.NativePreflightValidator()",
                "    validator.build_preflight_report_payload(3, 2, 1, True)",
                "    validator.validate_single_trait_preflight_shape_payload(3, 2, 3, 2)",
                "    validator.validate_multi_trait_preflight_shape_payload(2, 2, 3, 2, 3, 2)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.policy_name, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/engine/regenie2_pipeline/preflight.py"),
            4,
            "preflight_payload_isolation",
            "validator.build_preflight_report_payload",
            "build_preflight_report_payload",
        ),
        (
            Path("g/engine/regenie2_pipeline/preflight.py"),
            5,
            "preflight_payload_isolation",
            "validator.validate_single_trait_preflight_shape_payload",
            "validate_single_trait_preflight_shape_payload",
        ),
        (
            Path("g/engine/regenie2_pipeline/preflight.py"),
            6,
            "preflight_payload_isolation",
            "validator.validate_multi_trait_preflight_shape_payload",
            "validate_multi_trait_preflight_shape_payload",
        ),
    ]


def test_runner_shutdown_policy_rejects_payload_methods(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "lifecycle.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build():",
                "    controller = _core.NativeShutdownController()",
                "    controller.requested_signal_payload()",
                "    controller.request_shutdown_payload(2)",
                "    controller.request_shutdown_signal_or_raise_second_signal_payload(2)",
                "    controller.handler_install_plan_payload()",
                "    controller.handler_restore_plan_payload()",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.policy_name, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/runner/lifecycle.py"),
            4,
            "runner_shutdown_payload_isolation",
            "controller.requested_signal_payload",
            "requested_signal_payload",
        ),
        (
            Path("g/runner/lifecycle.py"),
            5,
            "runner_shutdown_payload_isolation",
            "controller.request_shutdown_payload",
            "request_shutdown_payload",
        ),
        (
            Path("g/runner/lifecycle.py"),
            6,
            "runner_shutdown_payload_isolation",
            "controller.request_shutdown_signal_or_raise_second_signal_payload",
            "request_shutdown_signal_or_raise_second_signal_payload",
        ),
        (
            Path("g/runner/lifecycle.py"),
            7,
            "runner_shutdown_payload_isolation",
            "controller.handler_install_plan_payload",
            "handler_install_plan_payload",
        ),
        (
            Path("g/runner/lifecycle.py"),
            8,
            "runner_shutdown_payload_isolation",
            "controller.handler_restore_plan_payload",
            "handler_restore_plan_payload",
        ),
    ]


def test_runner_telemetry_path_policy_rejects_payload_methods(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "events.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build():",
                "    policy = _core.NativeTelemetrySessionPolicy('profile', 0)",
                "    policy.resolve_paths_payload('out.parquet', None, None, None, None, None, None)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.policy_name, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/runner/events.py"),
            4,
            "runner_telemetry_path_payload_isolation",
            "policy.resolve_paths_payload",
            "resolve_paths_payload",
        )
    ]


def test_runner_lifecycle_event_policy_rejects_payload_methods(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "events.py").write_text(
        "\n".join(
            (
                "from g import _core",
                "def build(artifacts, shutdown_request, error):",
                "    policy = _core.NativeRunEventPayloadPolicy()",
                "    policy.attach_run_metadata_payload(artifacts, 'run-1', 'regenie2_linear', 1)",
                "    policy.build_run_completed_event_payload(artifacts)",
                "    policy.build_run_interrupted_event_payload(shutdown_request)",
                "    policy.build_run_failed_event_payload(error)",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.policy_name, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/runner/events.py"),
            4,
            "runner_lifecycle_event_payload_isolation",
            "policy.attach_run_metadata_payload",
            "attach_run_metadata_payload",
        ),
        (
            Path("g/runner/events.py"),
            5,
            "runner_lifecycle_event_payload_isolation",
            "policy.build_run_completed_event_payload",
            "build_run_completed_event_payload",
        ),
        (
            Path("g/runner/events.py"),
            6,
            "runner_lifecycle_event_payload_isolation",
            "policy.build_run_interrupted_event_payload",
            "build_run_interrupted_event_payload",
        ),
        (
            Path("g/runner/events.py"),
            7,
            "runner_lifecycle_event_payload_isolation",
            "policy.build_run_failed_event_payload",
            "build_run_failed_event_payload",
        ),
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


def test_callback_summary_policy_rejects_payload_boundary_calls(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "runtime.py").write_text(
        "\n".join(
            (
                "def emit(runtime_resources, telemetry_policy, telemetry_session):",
                "    summary_payload = runtime_resources.binary_correction_summary_payload()",
                "    telemetry_policy.emit_binary_correction_summary_telemetry(",
                "        telemetry_session,",
                "        summary_payload,",
                "        'missing session',",
                "    )",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.policy_name, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [
        (
            Path("g/engine/callbacks/runtime.py"),
            2,
            "callback_summary_payload_isolation",
            "runtime_resources.binary_correction_summary_payload",
            "binary_correction_summary_payload",
        ),
        (
            Path("g/engine/callbacks/runtime.py"),
            3,
            "callback_summary_payload_isolation",
            "telemetry_policy.emit_binary_correction_summary_telemetry",
            "emit_binary_correction_summary_telemetry",
        ),
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


def test_jax_runtime_path_policy_rejects_python_expanduser(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    jax_runtime_directory = package_root / "jax_runtime"
    jax_runtime_directory.mkdir(parents=True)
    (jax_runtime_directory / "resolution.py").write_text(
        "\n".join(
            (
                "def resolve(policy):",
                "    return policy.cache_directory.expanduser()",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.call_name, violation.forbidden_call)
        for violation in violations
    ] == [(Path("g/jax_runtime/resolution.py"), 2, "policy.cache_directory.expanduser", "expanduser")]


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
    timing_directory = package_root / "runner"
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
        (Path("g/runner/timing.py"), 2, "getattr", "getattr"),
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
    runner_directory = package_root / "runner"
    runner_directory.mkdir(parents=True)
    (runner_directory / "events.py").write_text(
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
        (Path("g/runner/events.py"), 2, "log_run_failed"),
        (Path("g/runner/events.py"), 4, "close_with_event"),
        (Path("g/runner/events.py"), 6, "log_event"),
        (Path("g/runner/events.py"), 8, "log_run_started"),
        (Path("g/runner/events.py"), 10, "build_event_payload"),
        (Path("g/runner/events.py"), 12, "native_session_policy"),
        (Path("g/runner/events.py"), 14, "log_progress"),
    ]


def test_definition_policy_rejects_removed_orchestration_helpers(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    interface_directory = package_root / "interface"
    jax_runtime_directory = package_root / "jax_runtime"
    callback_directory = package_root / "engine" / "callbacks"
    native_dispatch_directory = package_root / "engine" / "native_dispatch"
    runner_directory.mkdir(parents=True)
    interface_directory.mkdir(parents=True)
    jax_runtime_directory.mkdir(parents=True)
    callback_directory.mkdir(parents=True)
    native_dispatch_directory.mkdir(parents=True)
    (runner_directory / "timing.py").write_text(
        "\n".join(
            (
                "def serialize_chunk_stage_timings():",
                "    pass",
                "def serialize_binary_chunk_diagnostics():",
                "    pass",
                "def serialize_null_logistic_diagnostics():",
                "    pass",
                "def binary_chunk_diagnostics_snapshot_to_mapping():",
                "    pass",
                "def null_logistic_diagnostics_snapshot_to_mapping():",
                "    pass",
                "def serialize_queue_backpressure():",
                "    pass",
                "def serialize_transfer_metadata():",
                "    pass",
                "def build_chunk_stage_summary():",
                "    pass",
                "def build_binary_chunk_summary():",
                "    pass",
                "def adapt_stage_timing_snapshot_payload():",
                "    pass",
                "def adapt_chunk_stage_timing_payload():",
                "    pass",
                "def adapt_queue_backpressure_payload():",
                "    pass",
                "def adapt_transfer_metadata_payload():",
                "    pass",
            )
        ),
        encoding="utf-8",
    )
    (runner_directory / "outputs.py").write_text(
        "\n".join(
            (
                "def build_execution_plan_hash():",
                "    pass",
                "def validate_manifest_compatibility():",
                "    pass",
                "def read_manifest_committed_chunk_identifiers():",
                "    pass",
                "def validate_strict_manifest_chunks():",
                "    pass",
                "def repair_strict_manifest_chunk_commits():",
                "    pass",
                "def load_run_manifest():",
                "    pass",
                "def write_run_manifest():",
                "    pass",
                "def resolve_output_run_paths():",
                "    pass",
                "def initialize_output_run():",
                "    pass",
                "def normalize_execution_plan_value():",
                "    pass",
                "def manifest_file_fingerprint_from_native_payload():",
                "    pass",
                "def prediction_loco_file_fingerprint_from_native_payload():",
                "    pass",
                "def native_mapping_payload():",
                "    pass",
            )
        ),
        encoding="utf-8",
    )
    (runner_directory / "runtime.py").write_text(
        "\n".join(
            (
                "def build_process_runtime_state():",
                "    pass",
                "def describe_logging_runtime_policy():",
                "    pass",
                "def logging_runtime_policy_from_native_payload():",
                "    pass",
                "def logging_runtime_policy_to_native_payload():",
                "    pass",
                "def jax_runtime_policy_from_native_payload():",
                "    pass",
                "def optional_path_from_native_payload():",
                "    pass",
                "def native_mapping_payload():",
                "    pass",
                "def native_int_payload():",
                "    pass",
            )
        ),
        encoding="utf-8",
    )
    (runner_directory / "lifecycle.py").write_text(
        "\n".join(
            (
                "def shutdown_signal_from_native_payload():",
                "    pass",
                "def native_int_payload():",
                "    pass",
                "def native_mapping_payload():",
                "    pass",
            )
        ),
        encoding="utf-8",
    )
    (runner_directory / "events.py").write_text(
        "\n".join(
            (
                "def telemetry_paths_from_native_payload():",
                "    pass",
                "def run_artifacts_from_native_payload():",
                "    pass",
                "def run_completed_event_from_native_payload():",
                "    pass",
                "def run_artifact_payload_from_native_payload():",
                "    pass",
                "def run_interrupted_event_from_native_payload():",
                "    pass",
                "def run_failed_event_from_native_payload():",
                "    pass",
                "def optional_path_from_native_payload():",
                "    pass",
                "def native_mapping_payload():",
                "    pass",
            )
        ),
        encoding="utf-8",
    )
    (native_dispatch_directory / "delivery.py").write_text(
        "\n".join(
            (
                "def resolve_native_callback_batch_size():",
                "    pass",
            )
        ),
        encoding="utf-8",
    )
    (native_dispatch_directory / "writers.py").write_text(
        "\n".join(
            (
                "def finish_writer_session():",
                "    pass",
                "def finish_writer_session_interrupted():",
                "    pass",
            )
        ),
        encoding="utf-8",
    )
    (callback_directory / "runtime.py").write_text(
        "\n".join(
            (
                "def classify_result_write_item():",
                "    pass",
                "def classify_dosage_work_item():",
                "    pass",
            )
        ),
        encoding="utf-8",
    )
    (package_root / "execution_plan.py").write_text(
        "\n".join(
            (
                "def normalize_binary_correction_config():",
                "    pass",
                "def build_kernel_config():",
                "    pass",
            )
        ),
        encoding="utf-8",
    )
    (interface_directory / "config.py").write_text(
        "\n".join(
            (
                "def normalize_python_options():",
                "    pass",
                "def normalize_python_option_value():",
                "    pass",
                "def flatten_unknown_option_name():",
                "    pass",
                "def split_name_list():",
                "    pass",
                "def optional_string():",
                "    pass",
                "def normalize_trait_type():",
                "    pass",
                "def flatten_toml_mapping():",
                "    pass",
                "def flatten_mapping_section():",
                "    pass",
                "def load_toml():",
                "    pass",
                "def validate_config():",
                "    pass",
            )
        ),
        encoding="utf-8",
    )
    (jax_runtime_directory / "setup.py").write_text(
        "\n".join(
            (
                "def default_nvidia_driver_probe_paths():",
                "    pass",
                "def nvidia_driver_is_visible():",
                "    pass",
                "def complete_jax_runtime_setup_validation_report():",
                "    pass",
                "def jax_gpu_validation_report_from_native_payload():",
                "    pass",
            )
        ),
        encoding="utf-8",
    )
    (jax_runtime_directory / "resolution.py").write_text(
        "\n".join(
            (
                "def jax_runtime_policy_to_native_payload():",
                "    pass",
                "def build_native_jax_runtime_policy_payload():",
                "    pass",
                "def jax_runtime_setup_report_from_native_payload():",
                "    pass",
            )
        ),
        encoding="utf-8",
    )
    (jax_runtime_directory / "diagnostics.py").write_text(
        "\n".join(
            (
                "def diagnostic_event_from_native_payload():",
                "    pass",
                "def diagnostic_field_from_native_payload():",
                "    pass",
                "def native_mapping_payload():",
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
        (Path("g/engine/callbacks/runtime.py"), 1, "classify_result_write_item"),
        (Path("g/engine/callbacks/runtime.py"), 3, "classify_dosage_work_item"),
        (Path("g/engine/native_dispatch/delivery.py"), 1, "resolve_native_callback_batch_size"),
        (Path("g/engine/native_dispatch/writers.py"), 1, "finish_writer_session"),
        (Path("g/engine/native_dispatch/writers.py"), 3, "finish_writer_session_interrupted"),
        (Path("g/execution_plan.py"), 1, "normalize_binary_correction_config"),
        (Path("g/execution_plan.py"), 3, "build_kernel_config"),
        (Path("g/interface/config.py"), 1, "normalize_python_options"),
        (Path("g/interface/config.py"), 3, "normalize_python_option_value"),
        (Path("g/interface/config.py"), 5, "flatten_unknown_option_name"),
        (Path("g/interface/config.py"), 7, "split_name_list"),
        (Path("g/interface/config.py"), 9, "optional_string"),
        (Path("g/interface/config.py"), 11, "normalize_trait_type"),
        (Path("g/interface/config.py"), 13, "flatten_toml_mapping"),
        (Path("g/interface/config.py"), 15, "flatten_mapping_section"),
        (Path("g/interface/config.py"), 17, "load_toml"),
        (Path("g/interface/config.py"), 19, "validate_config"),
        (Path("g/jax_runtime/diagnostics.py"), 1, "diagnostic_event_from_native_payload"),
        (Path("g/jax_runtime/diagnostics.py"), 3, "diagnostic_field_from_native_payload"),
        (Path("g/jax_runtime/diagnostics.py"), 5, "native_mapping_payload"),
        (Path("g/jax_runtime/resolution.py"), 1, "jax_runtime_policy_to_native_payload"),
        (Path("g/jax_runtime/resolution.py"), 3, "build_native_jax_runtime_policy_payload"),
        (Path("g/jax_runtime/resolution.py"), 5, "jax_runtime_setup_report_from_native_payload"),
        (Path("g/jax_runtime/setup.py"), 1, "default_nvidia_driver_probe_paths"),
        (Path("g/jax_runtime/setup.py"), 3, "nvidia_driver_is_visible"),
        (Path("g/jax_runtime/setup.py"), 5, "complete_jax_runtime_setup_validation_report"),
        (Path("g/jax_runtime/setup.py"), 7, "jax_gpu_validation_report_from_native_payload"),
        (Path("g/runner/events.py"), 1, "telemetry_paths_from_native_payload"),
        (Path("g/runner/events.py"), 3, "run_artifacts_from_native_payload"),
        (Path("g/runner/events.py"), 5, "run_completed_event_from_native_payload"),
        (Path("g/runner/events.py"), 7, "run_artifact_payload_from_native_payload"),
        (Path("g/runner/events.py"), 9, "run_interrupted_event_from_native_payload"),
        (Path("g/runner/events.py"), 11, "run_failed_event_from_native_payload"),
        (Path("g/runner/events.py"), 13, "optional_path_from_native_payload"),
        (Path("g/runner/events.py"), 15, "native_mapping_payload"),
        (Path("g/runner/lifecycle.py"), 1, "shutdown_signal_from_native_payload"),
        (Path("g/runner/lifecycle.py"), 3, "native_int_payload"),
        (Path("g/runner/lifecycle.py"), 5, "native_mapping_payload"),
        (Path("g/runner/outputs.py"), 1, "build_execution_plan_hash"),
        (Path("g/runner/outputs.py"), 3, "validate_manifest_compatibility"),
        (Path("g/runner/outputs.py"), 5, "read_manifest_committed_chunk_identifiers"),
        (Path("g/runner/outputs.py"), 7, "validate_strict_manifest_chunks"),
        (Path("g/runner/outputs.py"), 9, "repair_strict_manifest_chunk_commits"),
        (Path("g/runner/outputs.py"), 11, "load_run_manifest"),
        (Path("g/runner/outputs.py"), 13, "write_run_manifest"),
        (Path("g/runner/outputs.py"), 15, "resolve_output_run_paths"),
        (Path("g/runner/outputs.py"), 17, "initialize_output_run"),
        (Path("g/runner/outputs.py"), 19, "normalize_execution_plan_value"),
        (Path("g/runner/outputs.py"), 21, "manifest_file_fingerprint_from_native_payload"),
        (Path("g/runner/outputs.py"), 23, "prediction_loco_file_fingerprint_from_native_payload"),
        (Path("g/runner/outputs.py"), 25, "native_mapping_payload"),
        (Path("g/runner/runtime.py"), 1, "build_process_runtime_state"),
        (Path("g/runner/runtime.py"), 3, "describe_logging_runtime_policy"),
        (Path("g/runner/runtime.py"), 5, "logging_runtime_policy_from_native_payload"),
        (Path("g/runner/runtime.py"), 7, "logging_runtime_policy_to_native_payload"),
        (Path("g/runner/runtime.py"), 9, "jax_runtime_policy_from_native_payload"),
        (Path("g/runner/runtime.py"), 11, "optional_path_from_native_payload"),
        (Path("g/runner/runtime.py"), 13, "native_mapping_payload"),
        (Path("g/runner/runtime.py"), 15, "native_int_payload"),
        (Path("g/runner/timing.py"), 1, "serialize_chunk_stage_timings"),
        (Path("g/runner/timing.py"), 3, "serialize_binary_chunk_diagnostics"),
        (Path("g/runner/timing.py"), 5, "serialize_null_logistic_diagnostics"),
        (Path("g/runner/timing.py"), 7, "binary_chunk_diagnostics_snapshot_to_mapping"),
        (Path("g/runner/timing.py"), 9, "null_logistic_diagnostics_snapshot_to_mapping"),
        (Path("g/runner/timing.py"), 11, "serialize_queue_backpressure"),
        (Path("g/runner/timing.py"), 13, "serialize_transfer_metadata"),
        (Path("g/runner/timing.py"), 15, "build_chunk_stage_summary"),
        (Path("g/runner/timing.py"), 17, "build_binary_chunk_summary"),
        (Path("g/runner/timing.py"), 19, "adapt_stage_timing_snapshot_payload"),
        (Path("g/runner/timing.py"), 21, "adapt_chunk_stage_timing_payload"),
        (Path("g/runner/timing.py"), 23, "adapt_queue_backpressure_payload"),
        (Path("g/runner/timing.py"), 25, "adapt_transfer_metadata_payload"),
    ]


def test_alias_policy_rejects_removed_callback_helper_reexports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    callback_directory = package_root / "engine" / "callbacks"
    callback_directory.mkdir(parents=True)
    (callback_directory / "runtime.py").write_text(
        "\n".join(
            (
                "record_stage_duration_with_optional_chunk = transfers.record_stage_duration_with_optional_chunk",
                "binary_chunk_diagnostics_to_summary_counts = diagnostics.binary_chunk_diagnostics_to_summary_counts",
            )
        ),
        encoding="utf-8",
    )
    (callback_directory / "writers.py").write_text(
        "\n".join(
            (
                "cast_statistic_array_for_native_writer = transfers.cast_statistic_array_for_native_writer",
                "select_active_trait_rows_on_device = transfers.select_active_trait_rows_on_device",
            )
        ),
        encoding="utf-8",
    )
    (callback_directory / "linear.py").write_text(
        "\n".join(
            (
                "require_current_chromosome_state = runtime.require_current_chromosome_state",
                "put_compute_array_on_device = transfers.put_compute_array_on_device",
            )
        ),
        encoding="utf-8",
    )
    (callback_directory / "binary.py").write_text(
        "\n".join(
            (
                "collect_binary_chunk_diagnostics_if_needed = diagnostics.collect_binary_chunk_diagnostics_if_needed",
                "record_null_logistic_chromosome_diagnostics = diagnostics.record_null_logistic_chromosome_diagnostics",
            )
        ),
        encoding="utf-8",
    )
    (callback_directory / "grouped.py").write_text(
        "\n".join(
            (
                "MultiPhenotypeGroupFanout = shared.MultiPhenotypeGroupFanout",
                "build_projected_variant_major_dosage_chunk_stats = "
                "transfers.build_projected_variant_major_dosage_chunk_stats",
            )
        ),
        encoding="utf-8",
    )
    (callback_directory / "transfers.py").write_text(
        "\n".join(
            (
                "block_until_ready = diagnostics.block_until_ready",
                "get_metadata_chromosome = shared.get_metadata_chromosome",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_alias_policy_violations(package_root)

    observed_violations = sorted(
        (violation.path, violation.line_number, violation.alias_name) for violation in violations
    )
    assert observed_violations == [
        (Path("g/engine/callbacks/binary.py"), 1, "collect_binary_chunk_diagnostics_if_needed"),
        (Path("g/engine/callbacks/binary.py"), 2, "record_null_logistic_chromosome_diagnostics"),
        (Path("g/engine/callbacks/grouped.py"), 1, "MultiPhenotypeGroupFanout"),
        (Path("g/engine/callbacks/grouped.py"), 2, "build_projected_variant_major_dosage_chunk_stats"),
        (Path("g/engine/callbacks/linear.py"), 1, "require_current_chromosome_state"),
        (Path("g/engine/callbacks/linear.py"), 2, "put_compute_array_on_device"),
        (Path("g/engine/callbacks/runtime.py"), 1, "record_stage_duration_with_optional_chunk"),
        (Path("g/engine/callbacks/runtime.py"), 2, "binary_chunk_diagnostics_to_summary_counts"),
        (Path("g/engine/callbacks/transfers.py"), 1, "block_until_ready"),
        (Path("g/engine/callbacks/transfers.py"), 2, "get_metadata_chromosome"),
        (Path("g/engine/callbacks/writers.py"), 1, "cast_statistic_array_for_native_writer"),
        (Path("g/engine/callbacks/writers.py"), 2, "select_active_trait_rows_on_device"),
    ]


def test_parameter_policy_rejects_native_loader_injection_arguments(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    loader_directory = package_root / "engine" / "native_dispatch"
    loader_directory.mkdir(parents=True)
    (loader_directory / "loaders.py").write_text(
        "\n".join(
            (
                "def load_native_bgen_run_input(",
                "    *,",
                "    build_native_bgen_run_input_callable,",
                "    load_aligned_sample_data_callable,",
                "):",
                "    pass",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_parameter_policy_violations(package_root)

    assert [(violation.path, violation.line_number, violation.parameter_name) for violation in violations] == [
        (Path("g/engine/native_dispatch/loaders.py"), 3, "build_native_bgen_run_input_callable"),
        (Path("g/engine/native_dispatch/loaders.py"), 4, "load_aligned_sample_data_callable"),
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
