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
                "    _core.validate_strict_manifest_chunks('chunks', '{}')",
                "    _core.validate_strict_manifest_chunks_from_value('chunks', {})",
                "    _core.repair_strict_manifest_chunk_commits_from_value('chunks', {})",
                "    _core.read_manifest_committed_chunk_identifiers_from_value({})",
                "    _core.validate_run_manifest_compatibility_from_values({}, {})",
                "    _core.finalize_output_run_chunks('run', 'chunks', 'parquet', 'zstd')",
                "    _core.build_pipeline_output_preparation_batch_from_values((), (), (), (), False, 'fast')",
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
            "_core.validate_strict_manifest_chunks",
            "_core.validate_strict_manifest_chunks",
        ),
        (
            Path("g/runner/outputs.py"),
            8,
            "_core.validate_strict_manifest_chunks_from_value",
            "_core.validate_strict_manifest_chunks_from_value",
        ),
        (
            Path("g/runner/outputs.py"),
            9,
            "_core.repair_strict_manifest_chunk_commits_from_value",
            "_core.repair_strict_manifest_chunk_commits_from_value",
        ),
        (
            Path("g/runner/outputs.py"),
            10,
            "_core.read_manifest_committed_chunk_identifiers_from_value",
            "_core.read_manifest_committed_chunk_identifiers_from_value",
        ),
        (
            Path("g/runner/outputs.py"),
            11,
            "_core.validate_run_manifest_compatibility_from_values",
            "_core.validate_run_manifest_compatibility_from_values",
        ),
        (Path("g/runner/outputs.py"), 12, "_core.finalize_output_run_chunks", "_core.finalize_output_run_chunks"),
        (
            Path("g/runner/outputs.py"),
            13,
            "_core.build_pipeline_output_preparation_batch_from_values",
            "_core.build_pipeline_output_preparation_batch_from_values",
        ),
        (
            Path("g/runner/outputs.py"),
            14,
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
            "_core.build_execution_run_artifacts_payload",
            "_core.build_execution_run_artifacts_payload",
        ),
        (
            Path("g/runner/execution.py"),
            5,
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


def test_preflight_required_chromosome_policy_rejects_engine_probe(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    engine_directory = package_root / "engine"
    engine_directory.mkdir(parents=True)
    (engine_directory / "preflight.py").write_text(
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
        (Path("g/engine/preflight.py"), 2, "getattr", "getattr"),
    ]


def test_covariate_rank_scan_policy_rejects_matrix_rank_in_production_python(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    engine_directory = package_root / "engine"
    engine_directory.mkdir(parents=True)
    (engine_directory / "preflight.py").write_text(
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
        (Path("g/engine/preflight.py"), 3, "np.linalg.matrix_rank", "np.linalg.matrix_rank"),
        (Path("g/engine/preflight.py"), 4, "matrix_rank", "matrix_rank"),
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
