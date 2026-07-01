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
                "    _core.initialize_output_run('run', 'chunks', None, '{}', False, 'fast', token)",
                "    _core.validate_strict_manifest_chunks('chunks', '{}')",
                "    _core.finalize_output_run_chunks('run', 'chunks', 'parquet', 'zstd')",
                "    _core.NativePipelineOutputPreparationBatch((), (), (), (), False, 'fast')",
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
        (Path("g/runner/outputs.py"), 4, "_core.initialize_output_run", "_core.initialize_output_run"),
        (
            Path("g/runner/outputs.py"),
            5,
            "_core.validate_strict_manifest_chunks",
            "_core.validate_strict_manifest_chunks",
        ),
        (Path("g/runner/outputs.py"), 6, "_core.finalize_output_run_chunks", "_core.finalize_output_run_chunks"),
        (
            Path("g/runner/outputs.py"),
            7,
            "_core.NativePipelineOutputPreparationBatch",
            "_core.NativePipelineOutputPreparationBatch",
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


def test_diagnostic_payload_policy_rejects_direct_payload_builders(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    runner_directory = package_root / "runner"
    run_events_directory = package_root / "engine"
    runner_directory.mkdir(parents=True)
    run_events_directory.mkdir(parents=True)
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
    (run_events_directory / "run_events.py").write_text(
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
    engine_directory.mkdir(parents=True)
    (engine_directory / "preflight.py").write_text(
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
        (Path("g/engine/preflight.py"), 3, "np.isfinite", "np.isfinite"),
        (Path("g/engine/preflight.py"), 4, "np.unique", "np.unique"),
        (Path("g/engine/preflight.py"), 5, "np.count_nonzero", "np.count_nonzero"),
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
        (Path("g/engine/telemetry.py"), 6, "log_progress"),
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
