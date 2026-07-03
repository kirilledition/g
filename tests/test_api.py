from __future__ import annotations

import json
import signal
import subprocess
import sys
import textwrap
import typing
from pathlib import Path
from unittest.mock import patch

import pytest

import g
import g.cli as cli_module
import g.engine.shutdown as shutdown_module
import g.engine.telemetry as telemetry_module
from g import _core, api, execution_plan, types
from g.interface import config
from g.io import output
from g.io.output import OutputRunPaths, PreparedOutputRun
from g.jax_runtime import diagnostics as jax_runtime_diagnostics
from g.jax_runtime import models as jax_runtime_models
from g.jax_runtime import resolution as jax_runtime_resolution
from g.runner import execution as runner_execution
from g.runner import metadata as runner_metadata
from g.runner import outputs as runner_outputs
from g.runner import runtime as runner_runtime


def complete_mock_output_initialization(
    keyword_arguments: dict[str, object],
    phenotype_names: tuple[str, ...] = ("trait",),
) -> None:
    """Invoke the runner callback exposed to mocked engine pipelines."""
    output_initialized_callback = typing.cast(
        "typing.Callable[[tuple[str, ...]], None]",
        keyword_arguments["output_initialized_callback"],
    )
    output_initialized_callback(phenotype_names)


def build_minimal_options(**overrides: object) -> dict[str, object]:
    raw_options: dict[str, object] = {
        "step": 2,
        "qt": True,
        "bgen": "dataset.bgen",
        "sample": "dataset.sample",
        "phenoFile": "phenotype.tsv",
        "phenoCol": "trait",
        "covarColList": "age,sex",
        "pred": "predictions.list",
        "out": "results/output",
        "format": "parquet",
    }
    raw_options.update(overrides)
    return raw_options


def build_minimal_config() -> config.RegenieConfig:
    return config.RegenieConfig.from_options(build_minimal_options())


def test_shutdown_controller_uses_native_default_signals() -> None:
    shutdown_controller = shutdown_module.GracefulShutdownController(handled_signals=None)

    install_plan = shutdown_controller.native_controller.handler_install_plan_payload()
    handled_signals = typing.cast("tuple[dict[str, object], ...]", install_plan["handled_signals"])

    assert [signal_payload["name"] for signal_payload in handled_signals] == ["SIGINT", "SIGTERM"]


def test_shutdown_controller_first_and_second_signal_behavior() -> None:
    interrupt_controller = shutdown_module.GracefulShutdownController(handled_signals=None)
    termination_controller = shutdown_module.GracefulShutdownController(handled_signals=None)

    with pytest.raises(shutdown_module.GracefulShutdownRequested) as first_interrupt:
        interrupt_controller.handle_signal(int(signal.SIGINT), frame=None)
    with pytest.raises(KeyboardInterrupt):
        interrupt_controller.handle_signal(int(signal.SIGINT), frame=None)
    with pytest.raises(shutdown_module.GracefulShutdownRequested) as first_termination:
        termination_controller.handle_signal(int(signal.SIGTERM), frame=None)
    with pytest.raises(SystemExit) as second_termination:
        termination_controller.handle_signal(int(signal.SIGTERM), frame=None)

    assert first_interrupt.value.signal_name == "SIGINT"
    assert first_interrupt.value.exit_code == 130
    assert first_termination.value.signal_name == "SIGTERM"
    assert first_termination.value.exit_code == 143
    assert second_termination.value.code == 143


def build_compute_config(**overrides: object) -> config.GComputeConfig:
    """Build packaged compute config with test overrides."""
    return config.RegenieConfig.from_options(build_minimal_options(**overrides)).g_compute


def build_trait_config(**overrides: object) -> config.TraitConfig:
    """Build packaged trait config with test overrides."""
    return config.RegenieConfig.from_options(build_minimal_options(**overrides)).trait


def build_binary_config(**overrides: object) -> config.BinaryConfig:
    """Build packaged binary config with test overrides."""
    normalized_overrides = dict(overrides)
    if "p_threshold" in normalized_overrides:
        normalized_overrides["pThresh"] = normalized_overrides.pop("p_threshold")
    raw_options = build_minimal_options(**normalized_overrides)
    raw_options["qt"] = False
    raw_options["bt"] = True
    return config.RegenieConfig.from_options(raw_options).binary


def build_test_process_runtime_state(
    logging_policy: runner_runtime.LoggingRuntimePolicy | None,
    rayon_thread_count: int | None,
    jax_policy: jax_runtime_models.JaxRuntimePolicy | None = None,
) -> object:
    """Build a native process runtime state handle for isolated tests."""
    return runner_runtime.build_process_runtime_state(logging_policy, rayon_thread_count, jax_policy)


def build_test_runtime_compatibility_token(
    regenie_config: config.RegenieConfig,
) -> _core.NativeRuntimeCompatibilityToken:
    """Build a native compatibility token for direct execution-plan tests."""
    telemetry_paths = telemetry_module.resolve_telemetry_paths(regenie_config)
    runtime_policy = runner_runtime.build_runtime_policy(regenie_config, telemetry_paths)
    runtime_state = typing.cast("_core.NativeRuntimeState", build_test_process_runtime_state(None, None))
    return runtime_state.require_compatible_runtime_policy_handle(runtime_policy.native_policy)


def prepare_test_execution_plan_outputs(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> tuple[runner_outputs.PreparedPhenotypeRunPlan, ...]:
    """Prepare output state for direct execution-plan tests."""
    return runner_outputs.prepare_execution_plan_outputs(
        plan=plan,
        runtime_compatibility_token=runtime_compatibility_token,
    )


class FakeLoggingProcessRuntimeState:
    """Capture initialized logging policies while simulating native policy resolution."""

    def __init__(self, calls: list[dict[str, object]]) -> None:
        """Initialize the fake with a shared call sink."""
        self.calls = calls

    def build_logging_runtime_policy_payload(self, *arguments: object) -> dict[str, object]:
        """Build the resolved native logging policy payload."""
        if len(arguments) != 12:
            message = "Expected the native logging policy builder argument shape."
            raise AssertionError(message)
        log_filter = typing.cast("str", arguments[0])
        log_file = typing.cast("str | None", arguments[1])
        log_stderr = typing.cast("bool", arguments[2])
        log_queue_size = typing.cast("int", arguments[3])
        log_lossy = typing.cast("bool", arguments[4])
        include_source_location = typing.cast("bool", arguments[5])
        include_span_events = typing.cast("bool", arguments[6])
        trace_file = typing.cast("str | None", arguments[7])
        trace_filter = typing.cast("str", arguments[8])
        trace_event_cap = typing.cast("int | None", arguments[9])
        telemetry_mode = typing.cast("str", arguments[10])
        telemetry_stream_file = typing.cast("str | None", arguments[11])
        resolved_log_file = log_file
        resolved_trace_file = trace_file
        if telemetry_stream_file is not None:
            resolved_log_file = None
            resolved_trace_file = telemetry_stream_file
        return {
            "log_filter": log_filter,
            "log_file": resolved_log_file,
            "log_stderr": log_stderr,
            "log_queue_size": log_queue_size,
            "log_lossy": log_lossy,
            "include_source_location": include_source_location,
            "include_span_events": include_span_events,
            "trace_file": resolved_trace_file,
            "trace_filter": trace_filter,
            "trace_event_cap": trace_event_cap if telemetry_mode == types.TelemetryMode.TRACE.value else None,
        }

    def initialize_logging_runtime_policy(self, payload: dict[str, object]) -> bool:
        """Capture the initialized logging policy payload."""
        self.calls.append(payload)
        return True


def build_diagnostics_config(**overrides: object) -> config.GDiagnosticsConfig:
    """Build packaged diagnostics config with test overrides."""
    return config.RegenieConfig.from_options(build_minimal_options(**overrides)).g_diagnostics


def test_public_package_keeps_lazy_public_boundary_without_all() -> None:
    assert not hasattr(g, "__all__")
    assert g.regenie is api.regenie


def test_public_package_lazy_exports_cli_and_type_symbols() -> None:
    assert g.main is cli_module.main
    assert g.RunArtifacts is api.RunArtifacts
    assert g.RuntimeState is api.RuntimeState
    assert g.describe_runtime_state is api.describe_runtime_state
    assert g.OutputFormat is types.OutputFormat
    assert g.ArrayMemoryOrder is types.ArrayMemoryOrder


def test_public_package_rejects_unknown_attributes() -> None:
    with pytest.raises(AttributeError, match="module 'g' has no attribute 'missing'"):
        g.__getattr__("missing")


def test_importing_api_does_not_import_jax_heavy_modules() -> None:
    script = textwrap.dedent(
        """
        import sys

        import g.api

        forbidden_modules = (
            "jax",
            "jax.numpy",
            "g.jax_runtime.setup",
            "g.compute.regenie2_linear.api",
            "g.compute.regenie2_binary.api",
            "g.compute.common.genotype",
            "g.engine.callbacks",
            "g.engine.native_dispatch",
            "g.engine.regenie2_pipeline",
        )
        imported_modules = [module_name for module_name in forbidden_modules if module_name in sys.modules]
        if imported_modules:
            raise AssertionError(f"unexpected eager imports: {imported_modules}")
        """
    )

    completed_process = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed_process.returncode == 0, completed_process.stderr


def test_regenie_config_from_options_maps_regenie_names() -> None:
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "bt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoColList": "trait_a,trait_b",
            "covarCol": ["age", "sex"],
            "pred": "predictions.list",
            "out": "results/output",
            "firth": True,
            "approx": True,
            "pThresh": 0.01,
            "device": "gpu",
            "format": "arrow",
            "output_statistic_dtype": "float64",
        }
    )

    assert regenie_config.input.bgen == Path("dataset.bgen")
    assert regenie_config.input.pheno_columns == ("trait_a", "trait_b")
    assert regenie_config.input.covar_columns == ("age", "sex")
    assert regenie_config.trait.trait_type == types.RegenieTraitType.BINARY
    assert regenie_config.binary.p_threshold == pytest.approx(0.01)
    assert regenie_config.g_compute.device == types.Device.GPU
    assert regenie_config.g_output.format == types.OutputFormat.ARROW
    assert regenie_config.g_output.output_statistic_dtype == types.FloatingPointDtype.FLOAT64


def test_execution_plan_uses_safe_phenotype_output_directories() -> None:
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoColList": "../bad,a/b,/tmp/outside",
            "pred": "predictions.list",
            "out": "results/output",
        }
    )
    prepared_output_run = output.PreparedOutputRun(
        output.OutputRunPaths(Path("unused"), Path("unused/chunks")),
        None,
    )

    with patch(
        "g.runner.outputs.output.prepare_output_run", return_value=prepared_output_run
    ) as mock_prepare_output_run:
        runtime_compatibility_token = build_test_runtime_compatibility_token(regenie_config)
        plan = execution_plan.build_regenie_execution_plan(regenie_config)
        prepare_test_execution_plan_outputs(
            plan=plan,
            runtime_compatibility_token=runtime_compatibility_token,
        )

    assert tuple(phenotype_plan.phenotype_name for phenotype_plan in plan.phenotype_run_plans) == (
        "../bad",
        "a/b",
        "/tmp/outside",
    )
    assert tuple(phenotype_plan.output_directory_name for phenotype_plan in plan.phenotype_run_plans) == (
        "trait_0001_bad",
        "trait_0002_a_b",
        "trait_0003_tmp_outside",
    )
    output_roots = tuple(call.kwargs["output_root"] for call in mock_prepare_output_run.call_args_list)
    assert output_roots == (
        Path("results/output.g/trait_0001_bad"),
        Path("results/output.g/trait_0002_a_b"),
        Path("results/output.g/trait_0003_tmp_outside"),
    )
    assert all(
        call.kwargs["output_format"] == types.OutputFormat.PARQUET for call in mock_prepare_output_run.call_args_list
    )


def test_build_binary_kernel_config_maps_compute_options() -> None:
    kernel_config = execution_plan.build_binary_kernel_config(
        build_compute_config(
            firth_batch_size=7,
            firth_candidate_capacity=11,
            binary_null_maximum_iterations=13,
            binary_null_coefficient_tolerance=1.0e-5,
            binary_minimum_probability=1.0e-7,
            binary_minimum_variance=1.0e-9,
            binary_relative_variance_tolerance=2.0e-6,
            firth_maximum_iterations=17,
            firth_gradient_tolerance=2.0e-5,
            firth_coefficient_tolerance=3.0e-5,
            firth_likelihood_tolerance=4.0e-5,
            firth_maximum_step_size=6.0,
            firth_pseudo_maximum_iterations=19,
            firth_pseudo_inner_maximum_iterations=23,
            firth_newton_raphson_zero_start_iterations=29,
            firth_line_search_maximum_attempts=31,
            firth_step_halving_maximum_attempts=37,
            firth_initial_response_scale=4.5,
            firth_sparse_carrier_dosage_threshold=1.0e-3,
            firth_step_halving_scale=0.25,
            null_firth_maximum_iterations=41,
            null_firth_gradient_tolerance=5.0e-5,
            null_firth_maximum_step_size=7.0,
            null_firth_fallback_iteration_multiplier=43,
            null_firth_fallback_step_divisor=11.0,
            null_firth_line_search_maximum_attempts=47,
            null_firth_step_halving_scale=0.125,
            use_block_firth_math=True,
        )
    )

    assert kernel_config.firth_candidate.batch_size == 7
    assert kernel_config.firth_candidate.candidate_capacity == 11
    assert kernel_config.null_logistic.maximum_iterations == 13
    assert kernel_config.null_logistic.coefficient_tolerance == pytest.approx(1.0e-5)
    assert kernel_config.numerical.minimum_probability == pytest.approx(1.0e-7)
    assert kernel_config.numerical.minimum_variance == pytest.approx(1.0e-9)
    assert kernel_config.numerical.relative_variance_tolerance == pytest.approx(2.0e-6)
    assert kernel_config.approximate_firth.maximum_iterations == 17
    assert kernel_config.approximate_firth.gradient_tolerance == pytest.approx(2.0e-5)
    assert kernel_config.approximate_firth.coefficient_tolerance == pytest.approx(3.0e-5)
    assert kernel_config.approximate_firth.likelihood_tolerance == pytest.approx(4.0e-5)
    assert kernel_config.approximate_firth.maximum_step_size == pytest.approx(6.0)
    assert kernel_config.approximate_firth.pseudo_maximum_iterations == 19
    assert kernel_config.approximate_firth.pseudo_inner_maximum_iterations == 23
    assert kernel_config.approximate_firth.newton_raphson_zero_start_iterations == 29
    assert kernel_config.approximate_firth.line_search_maximum_attempts == 31
    assert kernel_config.approximate_firth.step_halving_maximum_attempts == 37
    assert kernel_config.approximate_firth.initial_response_scale == pytest.approx(4.5)
    assert kernel_config.approximate_firth.sparse_carrier_dosage_threshold == pytest.approx(1.0e-3)
    assert kernel_config.approximate_firth.step_halving_scale == pytest.approx(0.25)
    assert kernel_config.null_firth.maximum_iterations == 41
    assert kernel_config.null_firth.gradient_tolerance == pytest.approx(5.0e-5)
    assert kernel_config.null_firth.maximum_step_size == pytest.approx(7.0)
    assert kernel_config.null_firth.fallback_iteration_multiplier == 43
    assert kernel_config.null_firth.fallback_step_divisor == pytest.approx(11.0)
    assert kernel_config.null_firth.line_search_maximum_attempts == 47
    assert kernel_config.null_firth.step_halving_scale == pytest.approx(0.125)
    assert kernel_config.approximate_firth.use_block_math is True


def test_quantitative_kernel_config_maps_linear_numerical_options() -> None:
    regenie_config = config.RegenieConfig.from_options(
        build_minimal_options(
            linear_minimum_variance=3.0e-9,
            linear_relative_variance_tolerance=4.0e-6,
        )
    )
    kernel_config = execution_plan.build_kernel_config(regenie_config)
    linear_numerical_config = kernel_config.linear_numerical_config

    assert linear_numerical_config is not None
    assert linear_numerical_config.minimum_variance == pytest.approx(3.0e-9)
    assert linear_numerical_config.relative_variance_tolerance == pytest.approx(4.0e-6)


def test_normalize_binary_correction_config_maps_approximate_firth() -> None:
    plan = execution_plan.normalize_binary_correction_config(
        build_binary_config(firth=True, approx=True, p_threshold=0.01)
    )

    assert plan.method == types.BinaryFallbackMethod.FIRTH_APPROXIMATE
    assert plan.p_threshold == pytest.approx(0.01)
    assert plan.firth_se is False


def test_regenie_callable_dispatches_linear_pipeline() -> None:
    run_paths = OutputRunPaths(
        run_directory=Path("results/output.g/trait.regenie2_linear.run"),
        chunks_directory=Path("results/output.g/trait.regenie2_linear.run/parts"),
    )

    def complete_pipeline(**keyword_arguments: object) -> None:
        complete_mock_output_initialization(keyword_arguments)

    with (
        patch("g.interface.config.validate_config_for_run"),
        patch("g.runner.runtime.configure_runtime_before_jax_import") as mock_configure_runtime_before_jax_import,
        patch("g.runner.runtime.PROCESS_RUNTIME_STATE", build_test_process_runtime_state(None, None)),
        patch(
            "g.runner.outputs.output.prepare_output_run",
            return_value=PreparedOutputRun(output_run_paths=run_paths, existing_manifest={"committed_chunks": []}),
        ) as mock_prepare_output_run,
        patch("g.runner.runtime.run_regenie2_linear_bgen_pipeline", side_effect=complete_pipeline) as mock_pipeline,
        patch("g.runner.metadata.extend_run_manifest") as mock_extend_run_manifest,
        patch("g.interface.config.write_toml") as mock_write_toml,
        patch("g.engine.shutdown.install_graceful_shutdown_handlers") as mock_install_shutdown_handlers,
    ):
        artifacts = api.regenie(build_minimal_config())

    assert artifacts.output_run_directory == Path("results/output.g/trait.regenie2_linear.run")
    assert artifacts.final_dataset == Path("results/output.g/trait.regenie2_linear.run/parts")
    assert artifacts.final_parquet is None
    assert artifacts.effective_config == Path("results/output.g/trait.regenie2_linear.run/effective_config.toml")
    mock_configure_runtime_before_jax_import.assert_called_once()
    assert mock_configure_runtime_before_jax_import.call_args.args[0].device == types.Device.CPU
    mock_prepare_output_run.assert_called_once()
    assert mock_pipeline.call_args.kwargs["existing_manifest"] == {"committed_chunks": []}
    assert mock_pipeline.call_args.kwargs["resume"] is False
    assert mock_pipeline.call_args.kwargs["covariate_names"] == ("age", "sex")
    assert mock_pipeline.call_args.kwargs["prediction_list_path"] == Path("predictions.list")
    assert mock_pipeline.call_args.kwargs["alignment_config"].sample_key_mode == types.SampleKeyMode.IID
    writer_settings = mock_pipeline.call_args.kwargs["writer_settings"]
    assert writer_settings.chunks_per_arrow_file == 16
    assert writer_settings.arrow_compression == types.ArrowCompression.ZSTD
    assert writer_settings.parquet_compression == types.ParquetCompression.NONE
    assert writer_settings.output_statistic_dtype == types.FloatingPointDtype.FLOAT32
    assert writer_settings.finalize_parquet is False
    mock_install_shutdown_handlers.assert_not_called()
    mock_extend_run_manifest.assert_called_once()
    mock_write_toml.assert_called_once()


def test_regenie_completion_event_includes_user_visible_artifacts(tmp_path: Path) -> None:
    run_paths = OutputRunPaths(
        run_directory=tmp_path / "output.g" / "trait.regenie2_linear.run",
        chunks_directory=tmp_path / "output.g" / "trait.regenie2_linear.run" / "parts",
    )
    run_paths.chunks_directory.mkdir(parents=True)
    final_parquet = run_paths.run_directory / "final.parquet"
    event_stream_path = tmp_path / "events.jsonl"
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "sample": "dataset.sample",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": str(tmp_path / "output"),
            "format": "parquet",
            "log_file": str(event_stream_path),
        }
    )

    with (
        patch("g.runner.runtime.PROCESS_RUNTIME_STATE", build_test_process_runtime_state(None, None)),
        patch("g.runner.runtime.initialize_logging"),
        patch("g.runner.runtime.configure_runtime"),
        patch("g.runner.runtime.configure_runtime_before_jax_import"),
        patch(
            "g.runner.outputs.output.prepare_output_run",
            return_value=PreparedOutputRun(output_run_paths=run_paths, existing_manifest=None),
        ),
        patch("g.runner.runtime.run_regenie2_linear_bgen_pipeline", return_value=final_parquet),
        patch("g.runner.metadata.extend_run_manifest"),
        patch("g.interface.config.validate_config_for_run"),
        patch("g.interface.config.write_toml"),
    ):
        artifacts = api.regenie(regenie_config)

    event_payloads = [json.loads(line) for line in event_stream_path.read_text(encoding="utf-8").splitlines()]
    completed_payload = [
        event_payload for event_payload in event_payloads if event_payload["event"] == "run_completed"
    ][-1]

    assert artifacts.run_id == completed_payload["run_id"]
    assert completed_payload["association_mode"] == types.AssociationMode.REGENIE2_LINEAR.value
    assert completed_payload["phenotype_count"] == 1
    assert completed_payload["output_run_directory"] == str(run_paths.run_directory)
    assert completed_payload["final_dataset"] == str(run_paths.chunks_directory)
    assert completed_payload["final_parquet"] == str(final_parquet)
    assert completed_payload["phenotype_artifacts"] == [
        {
            "effective_config": str(run_paths.run_directory / "effective_config.toml"),
            "final_dataset": str(run_paths.chunks_directory),
            "final_parquet": str(final_parquet),
            "output_run_directory": str(run_paths.run_directory),
            "phenotype": "trait",
        }
    ]


def test_regenie_runtime_configuration_failure_writes_run_failed_telemetry(tmp_path: Path) -> None:
    event_stream_path = tmp_path / "events.jsonl"
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "sample": "dataset.sample",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": str(tmp_path / "output"),
            "log_file": str(event_stream_path),
        }
    )

    with (
        patch("g.runner.runtime.PROCESS_RUNTIME_STATE", build_test_process_runtime_state(None, None)),
        patch("g.runner.runtime.initialize_logging"),
        patch("g.runner.runtime.configure_runtime", side_effect=RuntimeError("rayon failed")),
        patch("g.interface.config.validate_config_for_run"),
        pytest.raises(RuntimeError, match="rayon failed"),
    ):
        api.regenie(regenie_config)

    event_payloads = [json.loads(line) for line in event_stream_path.read_text(encoding="utf-8").splitlines()]
    event_names = [event_payload["event"] for event_payload in event_payloads]
    failed_payload = [event_payload for event_payload in event_payloads if event_payload["event"] == "run_failed"][-1]

    assert event_names == ["run_started", "run_failed", "telemetry_session_closed"]
    assert failed_payload["level"] == "ERROR"
    assert failed_payload["failure_kind"] == "exception"
    assert failed_payload["error_type"] == "RuntimeError"
    assert failed_payload["error_message"] == "rayon failed"


def test_regenie_graceful_shutdown_event_preserves_signal_exit(tmp_path: Path) -> None:
    run_paths = OutputRunPaths(
        run_directory=tmp_path / "output.g" / "trait.regenie2_linear.run",
        chunks_directory=tmp_path / "output.g" / "trait.regenie2_linear.run" / "parts",
    )
    run_paths.chunks_directory.mkdir(parents=True)
    event_stream_path = tmp_path / "events.jsonl"
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "sample": "dataset.sample",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": str(tmp_path / "output"),
            "log_file": str(event_stream_path),
        }
    )
    shutdown_request = shutdown_module.GracefulShutdownRequested(
        shutdown_module.ShutdownSignal(number=2, name="SIGINT", exit_code=130)
    )

    with (
        patch("g.runner.runtime.PROCESS_RUNTIME_STATE", build_test_process_runtime_state(None, None)),
        patch("g.runner.runtime.initialize_logging"),
        patch("g.runner.runtime.configure_runtime"),
        patch("g.runner.runtime.configure_runtime_before_jax_import"),
        patch(
            "g.runner.outputs.output.prepare_output_run",
            return_value=PreparedOutputRun(output_run_paths=run_paths, existing_manifest=None),
        ),
        patch("g.runner.runtime.run_regenie2_linear_bgen_pipeline", side_effect=shutdown_request),
        patch("g.runner.metadata.extend_run_manifest"),
        patch("g.interface.config.validate_config_for_run"),
        patch("g.interface.config.write_toml"),
        pytest.raises(shutdown_module.GracefulShutdownRequested),
    ):
        api.regenie(regenie_config)

    event_payloads = [json.loads(line) for line in event_stream_path.read_text(encoding="utf-8").splitlines()]
    failed_payload = [event_payload for event_payload in event_payloads if event_payload["event"] == "run_failed"][-1]

    assert failed_payload["level"] == "WARN"
    assert failed_payload["failure_kind"] == "graceful_shutdown"
    assert failed_payload["signal_name"] == "SIGINT"
    assert failed_payload["signal_number"] == 2
    assert failed_payload["exit_code"] == 130
    assert failed_payload["flushed_for_resume"] is True
    assert "error_type" not in failed_payload
    assert "error_message" not in failed_payload


def test_regenie_does_not_write_run_start_metadata_before_output_initialization_failure() -> None:
    run_paths = OutputRunPaths(
        run_directory=Path("results/output.g/trait.regenie2_linear.run"),
        chunks_directory=Path("results/output.g/trait.regenie2_linear.run/chunks"),
    )
    call_order: list[str] = []

    def record_write_toml(*args: object, **kwargs: object) -> None:
        del args
        del kwargs
        call_order.append("effective_config")

    def record_extend_run_manifest(*args: object, **kwargs: object) -> None:
        del args
        del kwargs
        call_order.append("manifest")

    def fail_pipeline(**kwargs: object) -> Path:
        del kwargs
        call_order.append("pipeline")
        message = "pipeline failed"
        raise RuntimeError(message)

    with (
        patch("g.runner.runtime.PROCESS_RUNTIME_STATE", build_test_process_runtime_state(None, None)),
        patch("g.runner.runtime.initialize_logging"),
        patch("g.interface.config.validate_config_for_run"),
        patch("g.runner.runtime.configure_runtime_before_jax_import"),
        patch(
            "g.runner.outputs.output.prepare_output_run",
            return_value=PreparedOutputRun(output_run_paths=run_paths, existing_manifest=None),
        ),
        patch("g.runner.runtime.run_regenie2_linear_bgen_pipeline", side_effect=fail_pipeline),
        patch("g.runner.metadata.extend_run_manifest", side_effect=record_extend_run_manifest),
        patch("g.interface.config.write_toml", side_effect=record_write_toml),
        pytest.raises(RuntimeError, match="pipeline failed"),
    ):
        api.regenie(build_minimal_config())

    assert call_order == ["pipeline"]


def test_regenie_writes_run_start_metadata_after_output_initialization() -> None:
    run_paths = OutputRunPaths(
        run_directory=Path("results/output.g/trait.regenie2_linear.run"),
        chunks_directory=Path("results/output.g/trait.regenie2_linear.run/chunks"),
    )
    call_order: list[str] = []

    def record_write_toml(*args: object, **kwargs: object) -> None:
        del args
        del kwargs
        call_order.append("effective_config")

    def record_extend_run_manifest(*args: object, **kwargs: object) -> None:
        del args
        del kwargs
        call_order.append("manifest")

    def fail_after_output_initialization(**keyword_arguments: object) -> Path:
        call_order.append("pipeline")
        complete_mock_output_initialization(keyword_arguments)
        call_order.append("after_initialization")
        message = "pipeline failed after initialization"
        raise RuntimeError(message)

    with (
        patch("g.runner.runtime.PROCESS_RUNTIME_STATE", build_test_process_runtime_state(None, None)),
        patch("g.runner.runtime.initialize_logging"),
        patch("g.interface.config.validate_config_for_run"),
        patch("g.runner.runtime.configure_runtime_before_jax_import"),
        patch(
            "g.runner.outputs.output.prepare_output_run",
            return_value=PreparedOutputRun(output_run_paths=run_paths, existing_manifest=None),
        ),
        patch("g.runner.runtime.run_regenie2_linear_bgen_pipeline", side_effect=fail_after_output_initialization),
        patch("g.runner.metadata.extend_run_manifest", side_effect=record_extend_run_manifest),
        patch("g.interface.config.write_toml", side_effect=record_write_toml),
        pytest.raises(RuntimeError, match="pipeline failed after initialization"),
    ):
        api.regenie(build_minimal_config())

    assert call_order == ["pipeline", "effective_config", "manifest", "after_initialization"]


def test_regenie_bootstraps_jax_before_preparing_execution_plan() -> None:
    run_paths = OutputRunPaths(
        run_directory=Path("results/output.g/trait.regenie2_linear.run"),
        chunks_directory=Path("results/output.g/trait.regenie2_linear.run/chunks"),
    )
    call_order: list[str] = []

    def record_jax_bootstrap(*args: object, **kwargs: object) -> None:
        del args
        del kwargs
        call_order.append("jax")

    def record_logging_bootstrap(*args: object, **kwargs: object) -> None:
        del args
        del kwargs
        call_order.append("logging")

    def record_native_runtime_bootstrap(*args: object, **kwargs: object) -> None:
        del args
        del kwargs
        call_order.append("native")

    def record_prepare_output_run(*args: object, **kwargs: object) -> PreparedOutputRun:
        del args
        del kwargs
        call_order.append("plan")
        return PreparedOutputRun(output_run_paths=run_paths, existing_manifest=None)

    def record_pipeline(**kwargs: object) -> Path:
        del kwargs
        call_order.append("pipeline")
        return Path("results/output.g/trait.regenie2_linear.run/final.parquet")

    with (
        patch("g.runner.runtime.PROCESS_RUNTIME_STATE", build_test_process_runtime_state(None, None)),
        patch("g.runner.runtime.initialize_logging", side_effect=record_logging_bootstrap),
        patch("g.runner.runtime.configure_runtime", side_effect=record_native_runtime_bootstrap),
        patch("g.runner.runtime.configure_runtime_before_jax_import", side_effect=record_jax_bootstrap),
        patch("g.runner.outputs.output.prepare_output_run", side_effect=record_prepare_output_run),
        patch("g.runner.runtime.run_regenie2_linear_bgen_pipeline", side_effect=record_pipeline),
        patch("g.runner.metadata.extend_run_manifest"),
        patch("g.interface.config.validate_config_for_run"),
        patch("g.interface.config.write_toml"),
    ):
        api.regenie(build_minimal_config())

    assert call_order == ["logging", "native", "jax", "plan", "pipeline"]


def test_initialize_logging_passes_diagnostics_to_native_state(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    diagnostics_config = build_diagnostics_config(
        telemetry=types.TelemetryMode.TRACE,
        log_filter="g=debug",
        log_file=tmp_path / "logs" / "g.jsonl",
        log_stderr=False,
        log_queue_size=1024,
        log_lossy=False,
        include_source_location=True,
        include_span_events=True,
        trace_file=tmp_path / "logs" / "trace.jsonl",
        trace_filter="g=trace",
        trace_event_cap=2048,
    )

    with patch("g.runner.runtime.PROCESS_RUNTIME_STATE", FakeLoggingProcessRuntimeState(calls)):
        runner_runtime.initialize_logging(diagnostics_config, telemetry_paths=None)

    assert calls == [
        {
            "log_filter": "g=debug",
            "log_file": str(tmp_path / "logs" / "g.jsonl"),
            "log_stderr": False,
            "log_queue_size": 1024,
            "log_lossy": False,
            "include_source_location": True,
            "include_span_events": True,
            "trace_file": str(tmp_path / "logs" / "trace.jsonl"),
            "trace_filter": "g=trace",
            "trace_event_cap": 2048,
        }
    ]


def test_initialize_logging_uses_unified_telemetry_stream(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    stream_file = tmp_path / "logs" / "events.jsonl"
    diagnostics_config = build_diagnostics_config(log_file=stream_file)
    telemetry_paths = telemetry_module.TelemetryPaths(
        log_dir=tmp_path / "logs",
        stream_file=stream_file,
        profile_summary_json=None,
        stage_timings_json=None,
    )

    with patch("g.runner.runtime.PROCESS_RUNTIME_STATE", FakeLoggingProcessRuntimeState(calls)):
        runner_runtime.initialize_logging(diagnostics_config, telemetry_paths)

    assert calls[0]["log_file"] is None
    assert calls[0]["trace_file"] == str(stream_file)
    assert calls[0]["trace_event_cap"] is None


def test_initialize_logging_applies_trace_cap_only_in_trace_mode(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    stream_file = tmp_path / "logs" / "events.jsonl"
    diagnostics_config = build_diagnostics_config(
        telemetry=types.TelemetryMode.TRACE,
        log_file=stream_file,
        trace_event_cap=17,
    )
    telemetry_paths = telemetry_module.TelemetryPaths(
        log_dir=tmp_path / "logs",
        stream_file=stream_file,
        profile_summary_json=None,
        stage_timings_json=None,
    )

    with patch("g.runner.runtime.PROCESS_RUNTIME_STATE", FakeLoggingProcessRuntimeState(calls)):
        runner_runtime.initialize_logging(diagnostics_config, telemetry_paths)

    assert calls[0]["trace_event_cap"] == 17


def test_initialize_logging_uses_trace_file_alias_as_unified_stream(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    stream_file = tmp_path / "logs" / "events.jsonl"
    diagnostics_config = build_diagnostics_config(trace_file=stream_file)
    telemetry_paths = telemetry_module.TelemetryPaths(
        log_dir=tmp_path / "logs",
        stream_file=stream_file,
        profile_summary_json=None,
        stage_timings_json=None,
    )

    with patch("g.runner.runtime.PROCESS_RUNTIME_STATE", FakeLoggingProcessRuntimeState(calls)):
        runner_runtime.initialize_logging(diagnostics_config, telemetry_paths)

    assert calls[0]["log_file"] is None
    assert calls[0]["trace_file"] == str(stream_file)
    assert calls[0]["trace_event_cap"] is None


def test_initialize_logging_rejects_incompatible_process_global_policy(tmp_path: Path) -> None:
    configured_policy = runner_runtime.LoggingRuntimePolicy(
        log_filter="info",
        log_file=tmp_path / "logs" / "first.jsonl",
        log_stderr=True,
        log_queue_size=config.load_packaged_config().g_diagnostics.log_queue_size,
        log_lossy=True,
        include_source_location=False,
        include_span_events=False,
        trace_file=None,
        trace_filter=config.load_packaged_config().g_diagnostics.trace_filter,
        trace_event_cap=None,
    )
    diagnostics_config = build_diagnostics_config(log_file=tmp_path / "logs" / "second.jsonl")

    with (
        patch("g.runner.runtime.PROCESS_RUNTIME_STATE", build_test_process_runtime_state(configured_policy, None)),
        pytest.raises(RuntimeError, match="Logging runtime policy is process-global"),
    ):
        runner_runtime.initialize_logging(diagnostics_config, telemetry_paths=None)


def test_configure_runtime_sets_native_knobs_and_threads() -> None:
    calls: list[tuple[int, int | None]] = []

    class FakeProcessRuntimeState:
        def configure_runtime_knobs(
            self,
            bgen_decode_tile_variant_count: int,
            rayon_thread_count: int | None,
        ) -> object:
            calls.append((bgen_decode_tile_variant_count, rayon_thread_count))
            return object()

    with patch("g.runner.runtime.PROCESS_RUNTIME_STATE", FakeProcessRuntimeState()):
        runner_runtime.configure_runtime(
            build_compute_config(bgen_decode_tile_variant_count=32),
            build_trait_config(threads=4),
        )

    assert calls == [(32, 4)]


def test_configure_runtime_passes_missing_rayon_thread_policy() -> None:
    calls: list[tuple[int, int | None]] = []

    class FakeProcessRuntimeState:
        def configure_runtime_knobs(
            self,
            bgen_decode_tile_variant_count: int,
            rayon_thread_count: int | None,
        ) -> object:
            calls.append((bgen_decode_tile_variant_count, rayon_thread_count))
            return object()

    with patch("g.runner.runtime.PROCESS_RUNTIME_STATE", FakeProcessRuntimeState()):
        runner_runtime.configure_runtime(
            build_compute_config(bgen_decode_tile_variant_count=32),
            build_trait_config(),
        )

    assert calls == [(32, None)]


def test_configure_runtime_propagates_native_runtime_knob_failure() -> None:
    calls: list[tuple[int, int | None]] = []

    class FakeProcessRuntimeState:
        def configure_runtime_knobs(
            self,
            bgen_decode_tile_variant_count: int,
            rayon_thread_count: int | None,
        ) -> object:
            calls.append((bgen_decode_tile_variant_count, rayon_thread_count))
            raise RuntimeError(
                f"Unable to configure Rayon global thread pool for --threads={rayon_thread_count}; "
                "existing Rayon settings are unknown: global pool already initialized"
            )

    with (
        patch("g.runner.runtime.PROCESS_RUNTIME_STATE", FakeProcessRuntimeState()),
        pytest.raises(RuntimeError, match="Unable to configure Rayon global thread pool"),
    ):
        runner_runtime.configure_runtime(
            build_compute_config(bgen_decode_tile_variant_count=32),
            build_trait_config(threads=4),
        )

    assert calls == [(32, 4)]


def test_finalize_execution_plan_records_native_metadata_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostic_calls: list[tuple[str, int]] = []

    class FakeRunnerDiagnosticPolicy:
        def record_runner_metadata_artifacts_finalized_diagnostic_event(
            self,
            *,
            association_mode: str,
            phenotype_count: int,
        ) -> None:
            diagnostic_calls.append((association_mode, phenotype_count))

    monkeypatch.setattr(
        runner_metadata.run_events,
        "native_runner_diagnostic_policy",
        FakeRunnerDiagnosticPolicy,
    )

    output_run_paths = output.OutputRunPaths(Path("run/trait"), Path("run/trait/chunks"))
    phenotype_run_plan = runner_outputs.PreparedPhenotypeRunPlan(
        phenotype_name="trait",
        output_run_paths=output_run_paths,
        existing_manifest=None,
        effective_config_path=Path("run/trait/effective.toml"),
    )
    requested_phenotype_run_plan = execution_plan.PhenotypeRunPlan(
        phenotype_name="trait",
        output_directory_name="trait.regenie2_binary.run",
    )
    writer_settings = execution_plan.OutputWriterPlan(
        finalize_parquet=True,
        writer_thread_count=1,
        writer_queue_depth=8,
        chunks_per_arrow_file=16,
        arrow_compression=types.ArrowCompression.ZSTD,
        parquet_compression=types.ParquetCompression.ZSTD,
        output_format=types.OutputFormat.PARQUET,
        output_statistic_dtype=types.FloatingPointDtype.FLOAT32,
    )
    plan = execution_plan.RegenieExecutionPlan(
        association_mode=types.AssociationMode.REGENIE2_BINARY,
        genotype_source_config=typing.cast("typing.Any", object()),
        phenotype_path=Path("phenotypes.tsv"),
        prediction_list_path=Path("predictions.list"),
        covariate_path=None,
        covariate_names=None,
        phenotype_run_plans=(requested_phenotype_run_plan,),
        phenotype_compute_groups=(),
        binary_correction_plan=typing.cast("typing.Any", object()),
        kernel_config=typing.cast("typing.Any", object()),
        output_plan=execution_plan.OutputPlan(
            output_prefix=Path("results"),
            output_run_root=Path("run"),
            resume=False,
            resume_mode=types.ResumeMode.FAST,
            writer_settings=writer_settings,
        ),
        stage_timings_json=None,
    )

    artifacts = runner_metadata.finalize_execution_plan(
        regenie_config=typing.cast("typing.Any", object()),
        plan=plan,
        phenotype_run_plans=(phenotype_run_plan,),
        final_output_paths=(Path("run/trait/final.parquet"),),
    )

    assert diagnostic_calls == [("regenie2_binary", 1)]
    assert artifacts.final_parquet == Path("run/trait/final.parquet")


def test_effective_rayon_thread_count_prefers_configured_thread_count() -> None:
    with patch("g.runner.runtime.PROCESS_RUNTIME_STATE", build_test_process_runtime_state(None, 4)):
        assert runner_runtime.effective_rayon_thread_count(8) == 4


def test_effective_rayon_thread_count_returns_requested_thread_count_without_configuration() -> None:
    with patch("g.runner.runtime.PROCESS_RUNTIME_STATE", build_test_process_runtime_state(None, None)):
        assert runner_runtime.effective_rayon_thread_count(8) == 8


def test_runtime_bootstrap_delegates_policy_to_jax_runtime_setup_once() -> None:
    call_order: list[str] = []

    class FakeJaxSetupModule:
        def configure_before_backend_init(
            self,
            native_setup_session: _core.NativeJaxRuntimeSetupSession,
            diagnostic_sink: typing.Callable[[jax_runtime_models.JaxRuntimeDiagnosticEvent], None],
        ) -> jax_runtime_models.JaxRuntimeSetupReport:
            del diagnostic_sink
            assert isinstance(native_setup_session, _core.NativeJaxRuntimeSetupSession)
            assert native_setup_session.should_configure is True
            native_setup_session.complete_validation_payload("succeeded", None)
            setup_report = jax_runtime_resolution.jax_runtime_setup_report_from_native_payload(
                native_setup_session.setup_payload()
            )
            call_order.append(f"setup:{setup_report.requested_device.value}")
            return setup_report

    def import_module(module_name: str) -> object:
        call_order.append(f"import:{module_name}")
        if module_name == "g.jax_runtime.setup":
            return FakeJaxSetupModule()
        raise AssertionError(f"Unexpected import: {module_name}")

    process_runtime_state = build_test_process_runtime_state(None, None)
    assert not hasattr(jax_runtime_resolution, "resolve_jax_runtime_cache_directory")

    with (
        patch("g.runner.runtime.PROCESS_RUNTIME_STATE", process_runtime_state),
        patch("g.runner.runtime.importlib.import_module", side_effect=import_module),
    ):
        runner_runtime.configure_runtime_before_jax_import(
            build_compute_config(device=types.Device.GPU),
            telemetry_session=None,
        )

    assert call_order == ["import:g.jax_runtime.setup", "setup:gpu"]
    configured_jax_policy = typing.cast("_core.NativeRuntimeState", process_runtime_state).jax_runtime_policy_payload()
    assert configured_jax_policy is not None
    assert configured_jax_policy["device"] == "gpu"


def test_runtime_bootstrap_records_jax_runtime_diagnostics() -> None:
    recorded_events: list[tuple[str, str, dict[str, object]]] = []

    class RecordingNativeTelemetrySession:
        def emit_jax_runtime_diagnostic_event(
            self,
            diagnostic_event: jax_runtime_models.JaxRuntimeDiagnosticEvent,
            telemetry_level: str,
        ) -> None:
            event_fields = {field.name: field.value for field in diagnostic_event.fields}
            recorded_events.append((diagnostic_event.event_name, telemetry_level, event_fields))

    class RecordingTelemetrySession:
        def __init__(self) -> None:
            self.native_telemetry_session = RecordingNativeTelemetrySession()

        def log_event(self, event_name: str, level: str = "info", **fields: object) -> None:
            recorded_events.append((event_name, level, fields))

    class FakeJaxSetupModule:
        def configure_before_backend_init(
            self,
            native_setup_session: _core.NativeJaxRuntimeSetupSession,
            diagnostic_sink: typing.Callable[[jax_runtime_models.JaxRuntimeDiagnosticEvent], None],
        ) -> jax_runtime_models.JaxRuntimeSetupReport:
            assert isinstance(native_setup_session, _core.NativeJaxRuntimeSetupSession)
            setup_report = jax_runtime_resolution.jax_runtime_setup_report_from_native_payload(
                native_setup_session.setup_payload()
            )
            for diagnostic_event in jax_runtime_diagnostics.diagnostic_events_from_native_setup_session(
                native_setup_session
            ):
                diagnostic_sink(diagnostic_event)
            return setup_report

    def import_module(module_name: str) -> object:
        if module_name == "g.jax_runtime.setup":
            return FakeJaxSetupModule()
        raise AssertionError(f"Unexpected import: {module_name}")

    telemetry_session = typing.cast("telemetry_module.TelemetrySession", RecordingTelemetrySession())

    with (
        patch("g.runner.runtime.PROCESS_RUNTIME_STATE", build_test_process_runtime_state(None, None)),
        patch("g.runner.runtime.importlib.import_module", side_effect=import_module),
    ):
        runner_runtime.configure_runtime_before_jax_import(build_compute_config(), telemetry_session=telemetry_session)

    assert [recorded_event[0] for recorded_event in recorded_events] == [
        "jax_platform_selected",
        "jax_persistent_cache_configured",
        "jax_xla_auxiliary_cache_configured",
        "jax_transfer_guard_configured",
        "jax_gpu_validation",
    ]
    assert recorded_events[0][2]["platform"] == "cpu"
    assert recorded_events[-1][2]["status"] == "skipped"


def test_runtime_diagnostic_recording_uses_native_record_plan() -> None:
    recorded_events: list[tuple[str, str, dict[str, object]]] = []
    logged_records: list[tuple[str, str, str, dict[str, object]]] = []

    class RecordingNativeTelemetrySession:
        def emit_jax_runtime_diagnostic_event(
            self,
            diagnostic_event: jax_runtime_models.JaxRuntimeDiagnosticEvent,
            telemetry_level: str,
        ) -> None:
            event_fields = {field.name: field.value for field in diagnostic_event.fields}
            recorded_events.append((diagnostic_event.event_name, telemetry_level, event_fields))

    class RecordingTelemetrySession:
        def __init__(self) -> None:
            self.native_telemetry_session = RecordingNativeTelemetrySession()

        def log_event(self, event_name: str, level: str = "info", **fields: object) -> None:
            recorded_events.append((event_name, level, fields))

    class NativeDiagnosticRecordPlan:
        logging_level_name: str
        should_emit_telemetry: bool
        telemetry_level: str

        def __init__(
            self,
            *,
            logging_level_name: str,
            should_emit_telemetry: bool,
            telemetry_level: str,
        ) -> None:
            self.logging_level_name = logging_level_name
            self.should_emit_telemetry = should_emit_telemetry
            self.telemetry_level = telemetry_level

    def record_jax_runtime_diagnostic_event(
        event: jax_runtime_models.JaxRuntimeDiagnosticEvent,
        telemetry_session: telemetry_module.TelemetrySession | None,
    ) -> NativeDiagnosticRecordPlan:
        assert event is diagnostic_event
        assert telemetry_session is active_telemetry_session
        logged_records.append(
            (
                "error",
                event.event_name,
                event.message,
                {diagnostic_field.name: diagnostic_field.value for diagnostic_field in event.fields},
            )
        )
        native_telemetry_session = active_telemetry_session.native_telemetry_session
        assert native_telemetry_session is not None
        native_telemetry_session.emit_jax_runtime_diagnostic_event(
            event,
            telemetry_level="trace",
        )
        return NativeDiagnosticRecordPlan(
            logging_level_name="ERROR",
            should_emit_telemetry=True,
            telemetry_level="trace",
        )

    class FakeNativeJaxRuntimeDiagnosticPolicy:
        def record_jax_runtime_diagnostic_event(
            self,
            event: jax_runtime_models.JaxRuntimeDiagnosticEvent,
            telemetry_session: telemetry_module.TelemetrySession | None,
        ) -> NativeDiagnosticRecordPlan:
            return record_jax_runtime_diagnostic_event(event, telemetry_session)

    diagnostic_event = jax_runtime_models.JaxRuntimeDiagnosticEvent(
        event_name="jax_native_plan_test",
        level=jax_runtime_models.JaxRuntimeDiagnosticLevel.INFO,
        message="planned diagnostic",
        fields=(jax_runtime_models.JaxRuntimeDiagnosticField(name="field", value="value"),),
    )
    active_telemetry_session = typing.cast("telemetry_module.TelemetrySession", RecordingTelemetrySession())

    with patch(
        "g.runner.runtime.native_jax_runtime_diagnostic_policy",
        return_value=FakeNativeJaxRuntimeDiagnosticPolicy(),
    ):
        runner_runtime.record_jax_runtime_diagnostic_event(
            diagnostic_event,
            telemetry_session=active_telemetry_session,
        )

    assert logged_records == [("error", "jax_native_plan_test", "planned diagnostic", {"field": "value"})]
    assert recorded_events == [("jax_native_plan_test", "trace", {"field": "value"})]


def test_repeated_runs_allow_same_jax_runtime_and_reject_incompatible_cache(tmp_path: Path) -> None:
    run_paths = OutputRunPaths(
        run_directory=Path("results/output.g/trait.regenie2_linear.run"),
        chunks_directory=Path("results/output.g/trait.regenie2_linear.run/chunks"),
    )
    call_order: list[str] = []

    class FakeJaxSetupModule:
        def configure_before_backend_init(
            self,
            native_setup_session: _core.NativeJaxRuntimeSetupSession,
            diagnostic_sink: typing.Callable[[jax_runtime_models.JaxRuntimeDiagnosticEvent], None],
        ) -> jax_runtime_models.JaxRuntimeSetupReport:
            del diagnostic_sink
            assert isinstance(native_setup_session, _core.NativeJaxRuntimeSetupSession)
            setup_report = jax_runtime_resolution.jax_runtime_setup_report_from_native_payload(
                native_setup_session.setup_payload()
            )
            call_order.append(f"setup:{setup_report.cache_directory}")
            return setup_report

    def import_module(module_name: str) -> object:
        if module_name == "g.jax_runtime.setup":
            return FakeJaxSetupModule()
        raise AssertionError(f"Unexpected import: {module_name}")

    first_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "sample": "dataset.sample",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
            "jax_cache_dir": str(tmp_path / "jax-cache"),
        }
    )
    incompatible_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "sample": "dataset.sample",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
            "jax_cache_dir": str(tmp_path / "other-jax-cache"),
        }
    )
    with (
        patch("g.runner.runtime.PROCESS_RUNTIME_STATE", build_test_process_runtime_state(None, None)),
        patch(
            "g.runner.outputs.output.prepare_output_run",
            return_value=PreparedOutputRun(output_run_paths=run_paths, existing_manifest=None),
        ),
        patch(
            "g.runner.runtime.run_regenie2_linear_bgen_pipeline",
            return_value=Path("results/output.g/trait.regenie2_linear.run/final.parquet"),
        ),
        patch("g.runner.runtime.initialize_logging"),
        patch("g.runner.runtime.configure_runtime"),
        patch("g.runner.metadata.extend_run_manifest"),
        patch("g.interface.config.validate_config_for_run"),
        patch("g.interface.config.write_toml"),
        patch("g.runner.runtime.importlib.import_module", side_effect=import_module),
    ):
        api.regenie(first_config)
        api.regenie(first_config)

        with pytest.raises(RuntimeError, match=r"JAX runtime is already configured.*incompatible settings"):
            api.regenie(incompatible_config)

    assert call_order.count(f"setup:{tmp_path / 'jax-cache'}") == 1
    assert f"setup:{tmp_path / 'other-jax-cache'}" not in call_order


def test_describe_runtime_state_reports_process_global_state() -> None:
    regenie_config = config.RegenieConfig.from_options(
        build_minimal_options(telemetry="off", log_filter="g=debug", threads=4)
    )
    telemetry_paths = telemetry_module.resolve_telemetry_paths(regenie_config)
    runtime_policy = runner_runtime.build_runtime_policy(regenie_config, telemetry_paths)
    process_runtime_state = build_test_process_runtime_state(
        runtime_policy.logging_policy,
        runtime_policy.rayon_thread_count,
        runtime_policy.jax_policy,
    )

    assert isinstance(runtime_policy.native_policy, _core.NativeRuntimePolicy)
    with (
        patch("g.runner.runtime.PROCESS_RUNTIME_STATE", process_runtime_state),
    ):
        run_runtime = runner_runtime.build_run_runtime(runtime_policy)
        runtime_state = api.describe_runtime_state()

    assert isinstance(run_runtime.native_runtime, _core.NativeRunRuntime)
    assert isinstance(run_runtime.runtime_compatibility_token, _core.NativeRuntimeCompatibilityToken)
    assert run_runtime.logging_policy == runtime_policy.logging_policy
    assert run_runtime.rayon_thread_count == runtime_policy.rayon_thread_count
    assert run_runtime.jax_policy == runtime_policy.jax_policy
    assert runtime_state == api.RuntimeState(
        logging_policy=runtime_policy.logging_policy,
        rayon_thread_count=4,
        jax_policy=runtime_policy.jax_policy,
    )
    assert runtime_state.logging_policy is not None
    assert "log-filter=g=debug" in runner_runtime.describe_logging_runtime_policy(runtime_state.logging_policy)


def test_regenie_rejects_incompatible_logging_policy_before_output_prepare(tmp_path: Path) -> None:
    configured_config = config.RegenieConfig.from_options(
        build_minimal_options(telemetry="off", log_filter="g=info", out=str(tmp_path / "first"))
    )
    requested_config = config.RegenieConfig.from_options(
        build_minimal_options(telemetry="off", log_filter="g=debug", out=str(tmp_path / "second"))
    )
    configured_policy = runner_runtime.build_runtime_policy(
        configured_config,
        telemetry_module.resolve_telemetry_paths(configured_config),
    )

    with (
        patch(
            "g.runner.runtime.PROCESS_RUNTIME_STATE",
            build_test_process_runtime_state(configured_policy.logging_policy, None),
        ),
        patch("g.runner.outputs.output.prepare_output_run") as prepare_output_run_mock,
        patch("g.runner.runtime.initialize_logging") as initialize_logging_mock,
        patch("g.interface.config.validate_config_for_run"),
        pytest.raises(RuntimeError, match=r"Logging runtime policy is process-global.*fresh Python process"),
    ):
        api.regenie(requested_config)

    prepare_output_run_mock.assert_not_called()
    initialize_logging_mock.assert_not_called()


def test_regenie_rejects_incompatible_rayon_policy_before_output_prepare() -> None:
    requested_config = config.RegenieConfig.from_options(build_minimal_options(telemetry="off", threads=8))

    with (
        patch("g.runner.runtime.PROCESS_RUNTIME_STATE", build_test_process_runtime_state(None, 4)),
        patch("g.runner.outputs.output.prepare_output_run") as prepare_output_run_mock,
        patch("g.runner.runtime.initialize_logging") as initialize_logging_mock,
        patch("g.interface.config.validate_config_for_run"),
        pytest.raises(RuntimeError, match=r"Rayon --threads is process-global.*fresh Python process"),
    ):
        api.regenie(requested_config)

    prepare_output_run_mock.assert_not_called()
    initialize_logging_mock.assert_not_called()


def test_regenie_rejects_incompatible_jax_policy_before_output_prepare(tmp_path: Path) -> None:
    configured_config = config.RegenieConfig.from_options(
        build_minimal_options(telemetry="off", jax_cache_dir=str(tmp_path / "first-cache"))
    )
    requested_config = config.RegenieConfig.from_options(
        build_minimal_options(telemetry="off", jax_cache_dir=str(tmp_path / "second-cache"))
    )
    configured_jax_policy = jax_runtime_resolution.resolve_jax_runtime_policy(configured_config.g_compute)

    with (
        patch(
            "g.runner.runtime.PROCESS_RUNTIME_STATE",
            build_test_process_runtime_state(None, None, configured_jax_policy),
        ),
        patch("g.runner.outputs.output.prepare_output_run") as prepare_output_run_mock,
        patch("g.runner.runtime.initialize_logging") as initialize_logging_mock,
        patch("g.interface.config.validate_config_for_run"),
        pytest.raises(RuntimeError, match=r"JAX runtime is already configured.*fresh Python process"),
    ):
        api.regenie(requested_config)

    prepare_output_run_mock.assert_not_called()
    initialize_logging_mock.assert_not_called()


def test_regenie_callable_dispatches_binary_pipeline_with_option_derived_kernel_config() -> None:
    run_paths = OutputRunPaths(
        run_directory=Path("results/output.g/trait.regenie2_binary.run"),
        chunks_directory=Path("results/output.g/trait.regenie2_binary.run/chunks"),
    )
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "bt": True,
            "bgen": "dataset.bgen",
            "sample": "dataset.sample",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
            "firth": True,
            "approx": True,
            "format": "parquet",
            "firth_batch_size": 7,
            "firth_candidate_capacity": 11,
            "binary_null_maximum_iterations": 13,
            "binary_null_coefficient_tolerance": 1.0e-5,
            "null_logistic_nonconvergence_policy": "warn",
            "binary_minimum_probability": 1.0e-7,
            "binary_minimum_variance": 1.0e-9,
            "binary_relative_variance_tolerance": 2.0e-6,
            "firth_maximum_iterations": 17,
            "firth_gradient_tolerance": 2.0e-5,
            "firth_coefficient_tolerance": 3.0e-5,
            "firth_likelihood_tolerance": 4.0e-5,
            "firth_maximum_step_size": 6.0,
            "firth_pseudo_maximum_iterations": 19,
            "firth_pseudo_inner_maximum_iterations": 23,
            "firth_newton_raphson_zero_start_iterations": 29,
            "firth_line_search_maximum_attempts": 31,
            "firth_step_halving_maximum_attempts": 37,
            "firth_initial_response_scale": 4.5,
            "firth_sparse_carrier_dosage_threshold": 1.0e-3,
            "firth_step_halving_scale": 0.25,
            "null_firth_maximum_iterations": 41,
            "null_firth_gradient_tolerance": 5.0e-5,
            "null_firth_maximum_step_size": 7.0,
            "null_firth_fallback_iteration_multiplier": 43,
            "null_firth_fallback_step_divisor": 11.0,
            "null_firth_line_search_maximum_attempts": 47,
            "null_firth_step_halving_scale": 0.125,
            "use_block_firth_math": True,
        }
    )

    with (
        patch("g.runner.runtime.PROCESS_RUNTIME_STATE", build_test_process_runtime_state(None, None)),
        patch("g.runner.runtime.initialize_logging"),
        patch("g.runner.runtime.configure_runtime_before_jax_import"),
        patch(
            "g.runner.outputs.output.prepare_output_run",
            return_value=PreparedOutputRun(output_run_paths=run_paths, existing_manifest=None),
        ),
        patch("g.runner.runtime.run_regenie2_binary_bgen_pipeline") as mock_binary_pipeline,
        patch("g.runner.metadata.extend_run_manifest"),
        patch("g.interface.config.validate_config_for_run"),
        patch("g.interface.config.write_toml"),
    ):
        mock_binary_pipeline.return_value = Path("results/output.g/trait.regenie2_binary.run/final.parquet")
        api.regenie(regenie_config)

    kernel_config = mock_binary_pipeline.call_args.kwargs["kernel_config"]
    assert kernel_config.firth_candidate.batch_size == 7
    assert kernel_config.firth_candidate.candidate_capacity == 11
    assert kernel_config.null_logistic.maximum_iterations == 13
    assert kernel_config.null_logistic.coefficient_tolerance == pytest.approx(1.0e-5)
    assert kernel_config.numerical.minimum_probability == pytest.approx(1.0e-7)
    assert kernel_config.numerical.minimum_variance == pytest.approx(1.0e-9)
    assert kernel_config.numerical.relative_variance_tolerance == pytest.approx(2.0e-6)
    assert kernel_config.approximate_firth.maximum_iterations == 17
    assert kernel_config.approximate_firth.gradient_tolerance == pytest.approx(2.0e-5)
    assert kernel_config.approximate_firth.coefficient_tolerance == pytest.approx(3.0e-5)
    assert kernel_config.approximate_firth.likelihood_tolerance == pytest.approx(4.0e-5)
    assert kernel_config.approximate_firth.maximum_step_size == pytest.approx(6.0)
    assert kernel_config.approximate_firth.pseudo_maximum_iterations == 19
    assert kernel_config.approximate_firth.pseudo_inner_maximum_iterations == 23
    assert kernel_config.approximate_firth.newton_raphson_zero_start_iterations == 29
    assert kernel_config.approximate_firth.line_search_maximum_attempts == 31
    assert kernel_config.approximate_firth.step_halving_maximum_attempts == 37
    assert kernel_config.approximate_firth.initial_response_scale == pytest.approx(4.5)
    assert kernel_config.approximate_firth.sparse_carrier_dosage_threshold == pytest.approx(1.0e-3)
    assert kernel_config.approximate_firth.step_halving_scale == pytest.approx(0.25)
    assert kernel_config.null_firth.maximum_iterations == 41
    assert kernel_config.null_firth.gradient_tolerance == pytest.approx(5.0e-5)
    assert kernel_config.null_firth.maximum_step_size == pytest.approx(7.0)
    assert kernel_config.null_firth.fallback_iteration_multiplier == 43
    assert kernel_config.null_firth.fallback_step_divisor == pytest.approx(11.0)
    assert kernel_config.null_firth.line_search_maximum_attempts == 47
    assert kernel_config.null_firth.step_halving_scale == pytest.approx(0.125)
    assert kernel_config.approximate_firth.use_block_math is True
    assert (
        mock_binary_pipeline.call_args.kwargs["correction_plan"].method == types.BinaryFallbackMethod.FIRTH_APPROXIMATE
    )
    assert (
        mock_binary_pipeline.call_args.kwargs["null_logistic_nonconvergence_policy"]
        == types.NullLogisticNonconvergencePolicy.WARN
    )


def test_quantitative_kernel_config_does_not_import_binary_runtime() -> None:
    regenie_config = build_minimal_config()
    binary_runtime_module_names = (
        "g.compute.regenie2_binary.api",
        "g.compute.regenie2_binary.score",
        "g.compute.regenie2_binary.state",
        "g.compute.regenie2_binary.firth.batch",
    )
    previous_modules = {module_name: sys.modules.pop(module_name, None) for module_name in binary_runtime_module_names}

    try:
        kernel_config = execution_plan.build_kernel_config(regenie_config)
        imported_modules = [module_name for module_name in binary_runtime_module_names if module_name in sys.modules]
    finally:
        for module_name, previous_module in previous_modules.items():
            if previous_module is not None:
                sys.modules[module_name] = previous_module

    assert kernel_config.binary_kernel_config is None
    assert imported_modules == []


def test_dispatch_engine_pipeline_forwards_binary_kernel_config() -> None:
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "bt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
            "firth": True,
            "approx": True,
            "firth_batch_size": 5,
            "native_callback_batch_size": 3,
            "result_in_flight_limit": 7,
            "dosage_buffer_limit": 8,
            "null_logistic_nonconvergence_policy": "warn",
        }
    )
    run_paths = output.OutputRunPaths(Path("run"), Path("run/chunks"))
    runtime_compatibility_token = build_test_runtime_compatibility_token(regenie_config)

    with (
        patch(
            "g.runner.outputs.output.prepare_output_run",
            return_value=output.PreparedOutputRun(run_paths, None),
        ),
        patch("g.runner.runtime.run_regenie2_binary_bgen_pipeline") as mock_binary_pipeline,
    ):
        plan = execution_plan.build_regenie_execution_plan(regenie_config)
        phenotype_run_plans = prepare_test_execution_plan_outputs(
            plan=plan,
            runtime_compatibility_token=runtime_compatibility_token,
        )
        runner_execution.dispatch_one_phenotype_engine_pipeline(
            plan=plan,
            phenotype_run_plan=phenotype_run_plans[0],
            stage_timing_recorder=None,
            telemetry_session=None,
            runtime_compatibility_token=runtime_compatibility_token,
            output_initialized_callback=lambda phenotype_names: None,
        )

    assert mock_binary_pipeline.call_args.kwargs["kernel_config"] is plan.kernel_config.binary_kernel_config
    assert mock_binary_pipeline.call_args.kwargs["kernel_config"].firth_candidate.batch_size == 5
    assert mock_binary_pipeline.call_args.kwargs["native_callback_batch_size"] == 3
    assert mock_binary_pipeline.call_args.kwargs["result_in_flight_limit"] == 7
    assert mock_binary_pipeline.call_args.kwargs["dosage_buffer_limit"] == 8
    assert mock_binary_pipeline.call_args.kwargs["gpu_genotype_format"] == types.GpuGenotypeFormat.AUTO
    assert (
        mock_binary_pipeline.call_args.kwargs["null_logistic_nonconvergence_policy"]
        == types.NullLogisticNonconvergencePolicy.WARN
    )


def test_dispatch_multi_engine_pipeline_forwards_binary_kernel_config() -> None:
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "bt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoColList": "trait_a,trait_b",
            "pred": "predictions.list",
            "out": "results/output",
            "firth": True,
            "approx": True,
            "firth_batch_size": 5,
            "null_logistic_nonconvergence_policy": "warn",
        }
    )
    run_paths = (
        output.OutputRunPaths(Path("run/trait_a"), Path("run/trait_a/chunks")),
        output.OutputRunPaths(Path("run/trait_b"), Path("run/trait_b/chunks")),
    )

    with (
        patch(
            "g.runner.outputs.output.prepare_output_run",
            side_effect=(
                output.PreparedOutputRun(run_paths[0], None),
                output.PreparedOutputRun(run_paths[1], None),
            ),
        ),
        patch("g.runner.runtime.run_regenie2_multi_phenotype_binary_bgen_pipeline") as mock_binary_pipeline,
    ):
        runtime_compatibility_token = build_test_runtime_compatibility_token(regenie_config)
        plan = execution_plan.build_regenie_execution_plan(regenie_config)
        phenotype_run_plans = prepare_test_execution_plan_outputs(
            plan=plan,
            runtime_compatibility_token=runtime_compatibility_token,
        )
        runner_execution.dispatch_multi_phenotype_engine_pipeline(
            plan=plan,
            phenotype_run_plans=phenotype_run_plans,
            stage_timing_recorder=None,
            telemetry_session=None,
            runtime_compatibility_token=runtime_compatibility_token,
            output_initialized_callback=lambda phenotype_names: None,
        )

    assert mock_binary_pipeline.call_args.kwargs["kernel_config"] is plan.kernel_config.binary_kernel_config
    assert mock_binary_pipeline.call_args.kwargs["kernel_config"].firth_candidate.batch_size == 5
    assert (
        mock_binary_pipeline.call_args.kwargs["null_logistic_nonconvergence_policy"]
        == types.NullLogisticNonconvergencePolicy.WARN
    )


def test_regenie_from_options_dispatches_multiple_phenotypes() -> None:
    with patch("g.api.runner_execution.regenie") as mock_runner_regenie:
        mock_runner_regenie.return_value = api.RunArtifacts(
            output_run_directory=None,
            final_dataset=None,
            final_parquet=None,
            final_regenie=None,
            effective_config=None,
            phenotype_artifacts=(
                api.RunArtifacts(
                    output_run_directory=Path("one"),
                    final_dataset=None,
                    final_parquet=None,
                    final_regenie=None,
                    effective_config=None,
                    phenotype_artifacts=(),
                    phenotype_name="one",
                    association_mode=None,
                    phenotype_count=2,
                    run_id=None,
                ),
                api.RunArtifacts(
                    output_run_directory=Path("two"),
                    final_dataset=None,
                    final_parquet=None,
                    final_regenie=None,
                    effective_config=None,
                    phenotype_artifacts=(),
                    phenotype_name="two",
                    association_mode=None,
                    phenotype_count=2,
                    run_id=None,
                ),
            ),
            phenotype_name=None,
            association_mode=None,
            phenotype_count=2,
            run_id=None,
        )
        artifacts = api.regenie.from_options(
            {
                "step": 2,
                "qt": True,
                "bgen": "dataset.bgen",
                "phenoFile": "phenotype.tsv",
                "phenoColList": "one,two",
                "pred": "predictions.list",
                "out": "results/output",
            }
        )

    assert len(artifacts.phenotype_artifacts) == 2
    mock_runner_regenie.assert_called_once()
    assert mock_runner_regenie.call_args.args[0].input.pheno_columns == ("one", "two")


def test_default_multi_phenotype_plan_dispatches_grouped_multi_phenotype_run() -> None:
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoColList": "one,two",
            "pred": "predictions.list",
            "out": "results/output",
        }
    )
    run_paths = output.OutputRunPaths(Path("run"), Path("run/chunks"))
    runtime_compatibility_token = build_test_runtime_compatibility_token(regenie_config)

    with (
        patch(
            "g.runner.outputs.output.prepare_output_run",
            return_value=output.PreparedOutputRun(run_paths, None),
        ),
        patch("g.runner.runtime.run_regenie2_multi_phenotype_linear_bgen_pipeline") as mock_multi_pipeline,
    ):
        plan = execution_plan.build_regenie_execution_plan(regenie_config)
        phenotype_run_plans = prepare_test_execution_plan_outputs(
            plan=plan,
            runtime_compatibility_token=runtime_compatibility_token,
        )
        runner_execution.dispatch_execution_plan(
            regenie_config=regenie_config,
            plan=plan,
            phenotype_run_plans=phenotype_run_plans,
            stage_timing_recorder=None,
            telemetry_session=None,
            runtime_compatibility_token=runtime_compatibility_token,
        )

    mock_multi_pipeline.assert_called_once()
    assert mock_multi_pipeline.call_args.kwargs["sample_mode"] == types.MultiPhenotypeSampleMode.PER_PHENOTYPE
    assert mock_multi_pipeline.call_args.kwargs["gpu_genotype_format"] == types.GpuGenotypeFormat.AUTO
    assert mock_multi_pipeline.call_args.kwargs["phenotype_compute_groups"] == plan.phenotype_compute_groups
    assert tuple(group.group_mode for group in plan.phenotype_compute_groups) == (
        types.PhenotypeComputeGroupMode.PER_PHENOTYPE_COMPATIBLE,
        types.PhenotypeComputeGroupMode.PER_PHENOTYPE_COMPATIBLE,
    )
    assert tuple(group.phenotype_indices for group in plan.phenotype_compute_groups) == ((0,), (1,))
    assert tuple(group.phenotype_names for group in plan.phenotype_compute_groups) == (("one",), ("two",))


def test_multi_phenotype_plan_dispatch_forwards_packed8_genotype_format() -> None:
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoColList": "one,two",
            "pred": "predictions.list",
            "out": "results/output",
            "device": "gpu",
            "gpu_genotype_format": "packed8",
            "multi_phenotype_sample_mode": "complete-case",
        }
    )
    run_paths = (
        output.OutputRunPaths(Path("run/one"), Path("run/one/chunks")),
        output.OutputRunPaths(Path("run/two"), Path("run/two/chunks")),
    )

    with (
        patch(
            "g.runner.outputs.output.prepare_output_run",
            side_effect=(
                output.PreparedOutputRun(run_paths[0], None),
                output.PreparedOutputRun(run_paths[1], None),
            ),
        ),
        patch("g.runner.runtime.run_regenie2_multi_phenotype_linear_bgen_pipeline") as mock_multi_pipeline,
    ):
        runtime_compatibility_token = build_test_runtime_compatibility_token(regenie_config)
        plan = execution_plan.build_regenie_execution_plan(regenie_config)
        phenotype_run_plans = prepare_test_execution_plan_outputs(
            plan=plan,
            runtime_compatibility_token=runtime_compatibility_token,
        )
        runner_execution.dispatch_multi_phenotype_engine_pipeline(
            plan=plan,
            phenotype_run_plans=phenotype_run_plans,
            stage_timing_recorder=None,
            telemetry_session=None,
            runtime_compatibility_token=runtime_compatibility_token,
            output_initialized_callback=lambda phenotype_names: None,
        )

    assert mock_multi_pipeline.call_args.kwargs["gpu_genotype_format"] == types.GpuGenotypeFormat.PACKED8
    assert mock_multi_pipeline.call_args.kwargs["sample_mode"] == types.MultiPhenotypeSampleMode.COMPLETE_CASE
    assert mock_multi_pipeline.call_args.kwargs["phenotype_compute_groups"] == plan.phenotype_compute_groups
    assert len(plan.phenotype_compute_groups) == 1
    compute_group = plan.phenotype_compute_groups[0]
    assert compute_group.group_mode == types.PhenotypeComputeGroupMode.COMPLETE_CASE
    assert compute_group.phenotype_indices == (0, 1)
    assert compute_group.phenotype_names == ("one", "two")


def test_multi_run_plan_forwards_existing_manifests() -> None:
    regenie_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "phenoFile": "phenotype.tsv",
            "phenoColList": "one,two",
            "pred": "predictions.list",
            "out": "results/output",
            "multi_phenotype_sample_mode": "complete-case",
        }
    )
    run_paths = (
        output.OutputRunPaths(Path("run/one"), Path("run/one/chunks")),
        output.OutputRunPaths(Path("run/two"), Path("run/two/chunks")),
    )
    existing_manifests = ({"phenotype_name": "one"}, {"phenotype_name": "two"})

    with patch(
        "g.runner.outputs.output.prepare_output_run",
        side_effect=(
            output.PreparedOutputRun(run_paths[0], existing_manifests[0]),
            output.PreparedOutputRun(run_paths[1], existing_manifests[1]),
        ),
    ):
        runtime_compatibility_token = build_test_runtime_compatibility_token(regenie_config)
        plan = execution_plan.build_regenie_execution_plan(regenie_config)
        phenotype_run_plans = prepare_test_execution_plan_outputs(
            plan=plan,
            runtime_compatibility_token=runtime_compatibility_token,
        )

    assert tuple(phenotype_plan.output_run_paths for phenotype_plan in phenotype_run_plans) == run_paths
    assert tuple(phenotype_plan.existing_manifest for phenotype_plan in phenotype_run_plans) == existing_manifests
    with patch("g.runner.runtime.run_regenie2_multi_phenotype_linear_bgen_pipeline") as mock_pipeline:
        runner_execution.dispatch_multi_phenotype_engine_pipeline(
            plan=plan,
            phenotype_run_plans=phenotype_run_plans,
            stage_timing_recorder=None,
            telemetry_session=None,
            runtime_compatibility_token=runtime_compatibility_token,
            output_initialized_callback=lambda phenotype_names: None,
        )

    assert mock_pipeline.call_args.kwargs["existing_manifests_by_phenotype"] == existing_manifests
    assert mock_pipeline.call_args.kwargs["sample_mode"] == types.MultiPhenotypeSampleMode.COMPLETE_CASE
    assert mock_pipeline.call_args.kwargs["resume"] is False


def test_extend_run_manifest_adds_command_metadata(tmp_path: Path) -> None:
    run_paths = output.OutputRunPaths(tmp_path, tmp_path / "chunks")
    run_paths.chunks_directory.mkdir()
    output.write_run_manifest(
        run_paths,
        {
            "schema_version": output.RUN_MANIFEST_SCHEMA_VERSION,
            "association_mode": types.AssociationMode.REGENIE2_LINEAR.value,
            "bgen": {"path": "/inputs/dataset.bgen", "size": 1, "mtime_ns": 2},
            "committed_chunks": [],
        },
    )
    regenie_config = build_minimal_config()

    with patch(
        "g.runner.outputs.output.prepare_output_run",
        return_value=output.PreparedOutputRun(run_paths, None),
    ):
        runtime_compatibility_token = build_test_runtime_compatibility_token(regenie_config)
        plan = execution_plan.build_regenie_execution_plan(regenie_config)
        phenotype_run_plans = prepare_test_execution_plan_outputs(
            plan=plan,
            runtime_compatibility_token=runtime_compatibility_token,
        )

    with (
        patch("g.io.output.load_run_manifest", side_effect=AssertionError("metadata extension must stay native")),
        patch("g.io.output.write_run_manifest", side_effect=AssertionError("metadata extension must stay native")),
    ):
        runner_metadata.extend_run_manifest(plan=plan, phenotype_run_plan=phenotype_run_plans[0])

    manifest = output.load_run_manifest(run_paths)
    assert manifest is not None
    assert manifest["command"]["interface"] == "g regenie"
    assert manifest["command"]["phenotype"] == "trait"
    assert manifest["bgen"] == {"path": "/inputs/dataset.bgen", "size": 1, "mtime_ns": 2}
    assert manifest["runtime"]["parquet_compression"] == "none"
    assert manifest["runtime"]["output_statistic_dtype"] == "float32"
    assert "input_fingerprints" not in manifest
