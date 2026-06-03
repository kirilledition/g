from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

import g
import g.engine.telemetry as telemetry_module
from g import api, execution_plan, runner, types
from g.interface import config
from g.io import output
from g.io.output import OutputRunPaths, PreparedOutputRun


def build_minimal_config() -> config.RegenieConfig:
    return config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "sample": "dataset.sample",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "covarColList": "age,sex",
            "pred": "predictions.list",
            "out": "results/output",
            "g-output-format": "parquet",
        }
    )


def test_public_package_exposes_only_new_regenie_interface() -> None:
    assert "regenie" in g.__all__
    assert "RegenieConfig" not in g.__all__
    assert "InputConfig" not in g.__all__
    assert "TraitConfig" not in g.__all__
    assert "BinaryConfig" not in g.__all__
    assert "GComputeConfig" not in g.__all__
    assert "GDiagnosticsConfig" not in g.__all__
    assert "GOutputConfig" not in g.__all__
    assert "regenie2" not in g.__all__
    assert "regenie2_linear" not in g.__all__
    assert "ComputeConfig" not in g.__all__
    assert g.regenie is api.regenie


def test_importing_api_does_not_import_jax_heavy_modules() -> None:
    script = textwrap.dedent(
        """
        import sys

        import g.api

        forbidden_modules = (
            "jax",
            "jax.numpy",
            "g.jax_setup",
            "g.compute.regenie2_linear",
            "g.compute.regenie2_binary",
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
            "g-device": "gpu",
            "g-output-format": "arrow",
        }
    )

    assert regenie_config.input.bgen == Path("dataset.bgen")
    assert regenie_config.input.pheno_columns == ("trait_a", "trait_b")
    assert regenie_config.input.covar_columns == ("age", "sex")
    assert regenie_config.trait.trait_type == types.RegenieTraitType.BINARY
    assert regenie_config.binary.p_threshold == 0.01
    assert regenie_config.g_compute.device == types.Device.GPU
    assert regenie_config.g_output.format == types.OutputFormat.ARROW


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
        "g.execution_plan.output.prepare_output_run", return_value=prepared_output_run
    ) as mock_prepare_output_run:
        plan = execution_plan.build_regenie_execution_plan(regenie_config)

    assert tuple(phenotype_plan.phenotype_name for phenotype_plan in plan.phenotype_run_plans) == (
        "../bad",
        "a/b",
        "/tmp/outside",
    )
    output_roots = tuple(call.kwargs["output_root"] for call in mock_prepare_output_run.call_args_list)
    assert output_roots == (
        Path("results/output.g/trait_0001_bad"),
        Path("results/output.g/trait_0002_a_b"),
        Path("results/output.g/trait_0003_tmp_outside"),
    )


def test_build_binary_kernel_config_maps_compute_options() -> None:
    kernel_config = execution_plan.build_binary_kernel_config(
        config.GComputeConfig(
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
    assert kernel_config.null_logistic.coefficient_tolerance == 1.0e-5
    assert kernel_config.numerical.minimum_probability == 1.0e-7
    assert kernel_config.numerical.minimum_variance == 1.0e-9
    assert kernel_config.numerical.relative_variance_tolerance == 2.0e-6
    assert kernel_config.approximate_firth.maximum_iterations == 17
    assert kernel_config.approximate_firth.gradient_tolerance == 2.0e-5
    assert kernel_config.approximate_firth.coefficient_tolerance == 3.0e-5
    assert kernel_config.approximate_firth.likelihood_tolerance == 4.0e-5
    assert kernel_config.approximate_firth.maximum_step_size == 6.0
    assert kernel_config.approximate_firth.pseudo_maximum_iterations == 19
    assert kernel_config.approximate_firth.pseudo_inner_maximum_iterations == 23
    assert kernel_config.approximate_firth.newton_raphson_zero_start_iterations == 29
    assert kernel_config.approximate_firth.line_search_maximum_attempts == 31
    assert kernel_config.approximate_firth.step_halving_maximum_attempts == 37
    assert kernel_config.approximate_firth.initial_response_scale == 4.5
    assert kernel_config.approximate_firth.sparse_carrier_dosage_threshold == 1.0e-3
    assert kernel_config.approximate_firth.step_halving_scale == 0.25
    assert kernel_config.null_firth.maximum_iterations == 41
    assert kernel_config.null_firth.gradient_tolerance == 5.0e-5
    assert kernel_config.null_firth.maximum_step_size == 7.0
    assert kernel_config.null_firth.fallback_iteration_multiplier == 43
    assert kernel_config.null_firth.fallback_step_divisor == 11.0
    assert kernel_config.null_firth.line_search_maximum_attempts == 47
    assert kernel_config.null_firth.step_halving_scale == 0.125
    assert kernel_config.approximate_firth.use_block_math is True


def test_normalize_binary_correction_config_maps_approximate_firth() -> None:
    plan = execution_plan.normalize_binary_correction_config(
        config.BinaryConfig(firth=True, approx=True, p_threshold=0.01)
    )

    assert plan == types.BinaryCorrectionPlan(
        method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE,
        p_threshold=0.01,
        firth_se=False,
    )


def test_regenie_callable_dispatches_linear_pipeline() -> None:
    run_paths = OutputRunPaths(
        run_directory=Path("results/output.g/trait.regenie2_linear.run"),
        chunks_directory=Path("results/output.g/trait.regenie2_linear.run/chunks"),
    )
    with (
        patch("g.runner.configure_runtime_before_jax_import") as mock_configure_runtime_before_jax_import,
        patch(
            "g.execution_plan.output.prepare_output_run",
            return_value=PreparedOutputRun(output_run_paths=run_paths, existing_manifest={"committed_chunks": []}),
        ) as mock_prepare_output_run,
        patch("g.runner.run_regenie2_linear_bgen_pipeline") as mock_pipeline,
        patch("g.runner.extend_run_manifest") as mock_extend_run_manifest,
        patch("g.interface.config.write_toml") as mock_write_toml,
    ):
        mock_pipeline.return_value = Path("results/output.g/trait.regenie2_linear.run/final.parquet")
        artifacts = api.regenie(build_minimal_config())

    assert artifacts.output_run_directory == Path("results/output.g/trait.regenie2_linear.run")
    assert artifacts.final_parquet == Path("results/output.g/trait.regenie2_linear.run/final.parquet")
    assert artifacts.effective_config == Path("results/output.g/trait.regenie2_linear.run/effective_config.toml")
    mock_configure_runtime_before_jax_import.assert_called_once()
    assert mock_configure_runtime_before_jax_import.call_args.args[0].device == types.Device.CPU
    mock_prepare_output_run.assert_called_once()
    assert mock_pipeline.call_args.kwargs["existing_manifest"] == {"committed_chunks": []}
    assert mock_pipeline.call_args.kwargs["resume"] is False
    assert mock_pipeline.call_args.kwargs["covariate_names"] == ("age", "sex")
    assert mock_pipeline.call_args.kwargs["prediction_list_path"] == Path("predictions.list")
    assert mock_pipeline.call_args.kwargs["alignment_config"].sample_key_mode == types.SampleKeyMode.IID
    assert mock_pipeline.call_args.kwargs["chunks_per_arrow_file"] == 16
    assert mock_pipeline.call_args.kwargs["arrow_compression"] == types.ArrowCompression.ZSTD
    mock_extend_run_manifest.assert_called_once()
    mock_write_toml.assert_called_once()


def test_regenie_writes_run_start_metadata_before_pipeline_failure() -> None:
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
        patch("g.runner.configure_runtime_before_jax_import"),
        patch(
            "g.execution_plan.output.prepare_output_run",
            return_value=PreparedOutputRun(output_run_paths=run_paths, existing_manifest=None),
        ),
        patch("g.runner.run_regenie2_linear_bgen_pipeline", side_effect=fail_pipeline),
        patch("g.runner.extend_run_manifest", side_effect=record_extend_run_manifest),
        patch("g.interface.config.write_toml", side_effect=record_write_toml),
        pytest.raises(RuntimeError, match="pipeline failed"),
    ):
        api.regenie(build_minimal_config())

    assert call_order == ["effective_config", "manifest", "pipeline"]


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
        patch("g.runner.initialize_logging", side_effect=record_logging_bootstrap),
        patch("g.runner.configure_runtime", side_effect=record_native_runtime_bootstrap),
        patch("g.runner.configure_runtime_before_jax_import", side_effect=record_jax_bootstrap),
        patch("g.execution_plan.output.prepare_output_run", side_effect=record_prepare_output_run),
        patch("g.runner.run_regenie2_linear_bgen_pipeline", side_effect=record_pipeline),
        patch("g.runner.extend_run_manifest"),
        patch("g.interface.config.write_toml"),
    ):
        api.regenie(build_minimal_config())

    assert call_order == ["logging", "native", "jax", "plan", "pipeline"]


def test_initialize_logging_passes_diagnostics_to_core(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class FakeCoreModule:
        def initialize_logging(self, **kwargs: object) -> None:
            calls.append(kwargs)

    diagnostics_config = config.GDiagnosticsConfig(
        log_filter="g=debug",
        log_file=tmp_path / "logs" / "g.jsonl",
        log_stderr=False,
        log_queue_size=1024,
        log_lossy=False,
        include_source_location=True,
        include_span_events=True,
        trace_file=tmp_path / "logs" / "trace.jsonl",
        trace_filter="g=trace",
    )

    with patch("g.runner.importlib.import_module", return_value=FakeCoreModule()) as mock_import_module:
        runner.initialize_logging(diagnostics_config)

    mock_import_module.assert_called_once_with("g._core")
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
        }
    ]


def test_initialize_logging_rejects_incompatible_process_global_policy(tmp_path: Path) -> None:
    class FakeCoreModule:
        def initialize_logging(self, **kwargs: object) -> bool:
            del kwargs
            return False

    configured_policy = runner.LoggingRuntimePolicy(
        log_filter="info",
        log_file=tmp_path / "logs" / "first.jsonl",
        log_stderr=True,
        log_queue_size=config.DEFAULT_LOG_QUEUE_SIZE,
        log_lossy=True,
        include_source_location=False,
        include_span_events=False,
        trace_file=None,
        trace_filter=config.DEFAULT_TRACE_FILTER,
    )
    diagnostics_config = config.GDiagnosticsConfig(log_file=tmp_path / "logs" / "second.jsonl")

    with (
        patch("g.runner.CONFIGURED_LOGGING_RUNTIME_POLICY", configured_policy),
        patch("g.runner.importlib.import_module", return_value=FakeCoreModule()),
        pytest.raises(RuntimeError, match="Logging is process-global"),
    ):
        runner.initialize_logging(diagnostics_config)


def test_configure_runtime_sets_native_knobs_and_threads() -> None:
    calls: list[tuple[str, int]] = []

    class FakeCoreModule:
        def configure_bgen_decode_tile_variant_count(self, tile_variant_count: int) -> None:
            calls.append(("tile", tile_variant_count))

        def configure_rayon_global_thread_pool(self, thread_count: int) -> None:
            calls.append(("threads", thread_count))

    with patch("g.runner.importlib.import_module", return_value=FakeCoreModule()) as mock_import_module:
        runner.configure_runtime(
            config.GComputeConfig(bgen_decode_tile_variant_count=32),
            config.TraitConfig(threads=4),
        )

    mock_import_module.assert_called_once_with("g._core")
    assert calls == [("tile", 32), ("threads", 4)]


def test_runtime_bootstrap_sets_jax_platform_before_setup_import() -> None:
    call_order: list[str] = []

    class FakeJaxConfig:
        def update(self, setting_name: str, value: object) -> None:
            call_order.append(f"{setting_name}:{value}")

    class FakeJaxModule:
        config = FakeJaxConfig()

    class FakeJaxSetupModule:
        def configure_jax_runtime_before_backend_init(self, **kwargs: object) -> None:
            del kwargs
            call_order.append("setup")

    def import_module(module_name: str) -> object:
        call_order.append(f"import:{module_name}")
        if module_name == "jax":
            return FakeJaxModule()
        if module_name == "g.jax_setup":
            return FakeJaxSetupModule()
        raise AssertionError(f"Unexpected import: {module_name}")

    with (
        patch("g.runner.CONFIGURED_JAX_RUNTIME_POLICY", None),
        patch("g.runner.importlib.import_module", side_effect=import_module),
    ):
        runner.configure_runtime_before_jax_import(config.GComputeConfig(device=types.Device.GPU))

    assert call_order == ["import:jax", "jax_platforms:cuda", "import:g.jax_setup", "setup"]


def test_repeated_runs_allow_same_jax_runtime_and_reject_incompatible_cache(tmp_path: Path) -> None:
    run_paths = OutputRunPaths(
        run_directory=Path("results/output.g/trait.regenie2_linear.run"),
        chunks_directory=Path("results/output.g/trait.regenie2_linear.run/chunks"),
    )
    call_order: list[str] = []

    class FakeJaxConfig:
        def update(self, setting_name: str, value: object) -> None:
            call_order.append(f"jax:{setting_name}:{value}")

    class FakeJaxModule:
        config = FakeJaxConfig()

    class FakeJaxSetupModule:
        def configure_jax_runtime_before_backend_init(self, **kwargs: object) -> None:
            call_order.append(f"setup:{kwargs['cache_directory']}:{kwargs['enable_x64']}")

    class FakeTimingModule:
        def build_stage_timing_recorder(self, stage_timing_path: Path | None) -> None:
            del stage_timing_path

        def record_stage_duration(self, stage_timing_recorder: object, stage_name: str, start_time: float) -> None:
            del stage_timing_recorder, start_time
            call_order.append(f"timing:{stage_name}")

        def write_stage_timing_snapshot(self, stage_timing_recorder: object, stage_timing_path: Path | None) -> None:
            del stage_timing_recorder, stage_timing_path

    def import_module(module_name: str) -> object:
        if module_name == "jax":
            return FakeJaxModule()
        if module_name == "g.jax_setup":
            return FakeJaxSetupModule()
        if module_name == "g.engine.timing":
            return FakeTimingModule()
        if module_name == "g.engine.telemetry":
            return telemetry_module
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
            "g-jax-cache-dir": str(tmp_path / "jax-cache"),
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
            "g-jax-cache-dir": str(tmp_path / "other-jax-cache"),
        }
    )
    incompatible_x64_config = config.RegenieConfig.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "sample": "dataset.sample",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "trait",
            "pred": "predictions.list",
            "out": "results/output",
            "g-jax-cache-dir": str(tmp_path / "jax-cache"),
            "g-jax-enable-x64": False,
        }
    )

    with (
        patch("g.runner.CONFIGURED_JAX_RUNTIME_POLICY", None),
        patch(
            "g.execution_plan.output.prepare_output_run",
            return_value=PreparedOutputRun(output_run_paths=run_paths, existing_manifest=None),
        ),
        patch(
            "g.runner.run_regenie2_linear_bgen_pipeline",
            return_value=Path("results/output.g/trait.regenie2_linear.run/final.parquet"),
        ),
        patch("g.runner.initialize_logging"),
        patch("g.runner.configure_runtime"),
        patch("g.runner.extend_run_manifest"),
        patch("g.interface.config.write_toml"),
        patch("g.runner.importlib.import_module", side_effect=import_module),
    ):
        api.regenie(first_config)
        api.regenie(first_config)

        with pytest.raises(RuntimeError, match=r"JAX runtime is already configured.*incompatible settings"):
            api.regenie(incompatible_config)
        with pytest.raises(RuntimeError, match=r"jax-enable-x64=True.*jax-enable-x64=False"):
            api.regenie(incompatible_x64_config)

    assert call_order.count(f"setup:{tmp_path / 'jax-cache'}:True") == 1
    assert f"setup:{tmp_path / 'jax-cache'}:False" not in call_order
    assert f"setup:{tmp_path / 'other-jax-cache'}:True" not in call_order


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
            "g-output-format": "parquet",
            "g-firth-batch-size": 7,
            "g-firth-candidate-capacity": 11,
            "g-binary-null-maximum-iterations": 13,
            "g-binary-null-coefficient-tolerance": 1.0e-5,
            "g-null-logistic-nonconvergence": "warn",
            "g-binary-minimum-probability": 1.0e-7,
            "g-binary-minimum-variance": 1.0e-9,
            "g-binary-relative-variance-tolerance": 2.0e-6,
            "g-firth-maximum-iterations": 17,
            "g-firth-gradient-tolerance": 2.0e-5,
            "g-firth-coefficient-tolerance": 3.0e-5,
            "g-firth-likelihood-tolerance": 4.0e-5,
            "g-firth-maximum-step-size": 6.0,
            "g-firth-pseudo-maximum-iterations": 19,
            "g-firth-pseudo-inner-maximum-iterations": 23,
            "g-firth-newton-raphson-zero-start-iterations": 29,
            "g-firth-line-search-maximum-attempts": 31,
            "g-firth-step-halving-maximum-attempts": 37,
            "g-firth-initial-response-scale": 4.5,
            "g-firth-sparse-carrier-dosage-threshold": 1.0e-3,
            "g-firth-step-halving-scale": 0.25,
            "g-null-firth-maximum-iterations": 41,
            "g-null-firth-gradient-tolerance": 5.0e-5,
            "g-null-firth-maximum-step-size": 7.0,
            "g-null-firth-fallback-iteration-multiplier": 43,
            "g-null-firth-fallback-step-divisor": 11.0,
            "g-null-firth-line-search-maximum-attempts": 47,
            "g-null-firth-step-halving-scale": 0.125,
            "g-use-block-firth-math": True,
        }
    )

    with (
        patch("g.runner.configure_runtime_before_jax_import"),
        patch(
            "g.execution_plan.output.prepare_output_run",
            return_value=PreparedOutputRun(output_run_paths=run_paths, existing_manifest=None),
        ),
        patch("g.runner.run_regenie2_binary_bgen_pipeline") as mock_binary_pipeline,
        patch("g.runner.extend_run_manifest"),
        patch("g.interface.config.write_toml"),
    ):
        mock_binary_pipeline.return_value = Path("results/output.g/trait.regenie2_binary.run/final.parquet")
        api.regenie(regenie_config)

    kernel_config = mock_binary_pipeline.call_args.kwargs["kernel_config"]
    assert kernel_config.firth_candidate.batch_size == 7
    assert kernel_config.firth_candidate.candidate_capacity == 11
    assert kernel_config.null_logistic.maximum_iterations == 13
    assert kernel_config.null_logistic.coefficient_tolerance == 1.0e-5
    assert kernel_config.numerical.minimum_probability == 1.0e-7
    assert kernel_config.numerical.minimum_variance == 1.0e-9
    assert kernel_config.numerical.relative_variance_tolerance == 2.0e-6
    assert kernel_config.approximate_firth.maximum_iterations == 17
    assert kernel_config.approximate_firth.gradient_tolerance == 2.0e-5
    assert kernel_config.approximate_firth.coefficient_tolerance == 3.0e-5
    assert kernel_config.approximate_firth.likelihood_tolerance == 4.0e-5
    assert kernel_config.approximate_firth.maximum_step_size == 6.0
    assert kernel_config.approximate_firth.pseudo_maximum_iterations == 19
    assert kernel_config.approximate_firth.pseudo_inner_maximum_iterations == 23
    assert kernel_config.approximate_firth.newton_raphson_zero_start_iterations == 29
    assert kernel_config.approximate_firth.line_search_maximum_attempts == 31
    assert kernel_config.approximate_firth.step_halving_maximum_attempts == 37
    assert kernel_config.approximate_firth.initial_response_scale == 4.5
    assert kernel_config.approximate_firth.sparse_carrier_dosage_threshold == 1.0e-3
    assert kernel_config.approximate_firth.step_halving_scale == 0.25
    assert kernel_config.null_firth.maximum_iterations == 41
    assert kernel_config.null_firth.gradient_tolerance == 5.0e-5
    assert kernel_config.null_firth.maximum_step_size == 7.0
    assert kernel_config.null_firth.fallback_iteration_multiplier == 43
    assert kernel_config.null_firth.fallback_step_divisor == 11.0
    assert kernel_config.null_firth.line_search_maximum_attempts == 47
    assert kernel_config.null_firth.step_halving_scale == 0.125
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

    with patch("g.execution_plan.importlib.import_module", side_effect=AssertionError("unexpected binary import")):
        kernel_config = execution_plan.build_kernel_config(regenie_config)

    assert kernel_config.binary_kernel_config is None


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
            "g-firth-batch-size": 5,
            "g-null-logistic-nonconvergence": "warn",
        }
    )
    run_paths = output.OutputRunPaths(Path("run"), Path("run/chunks"))

    with (
        patch(
            "g.execution_plan.output.prepare_output_run",
            return_value=output.PreparedOutputRun(run_paths, None),
        ),
        patch("g.runner.run_regenie2_binary_bgen_pipeline") as mock_binary_pipeline,
    ):
        plan = execution_plan.build_regenie_execution_plan(regenie_config)
        runner.dispatch_one_phenotype_engine_pipeline(
            plan=plan,
            phenotype_run_plan=plan.phenotype_run_plans[0],
            stage_timing_recorder=None,
        )

    assert mock_binary_pipeline.call_args.kwargs["kernel_config"] is plan.kernel_config.binary_kernel_config
    assert mock_binary_pipeline.call_args.kwargs["kernel_config"].firth_candidate.batch_size == 5
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
            "g-firth-batch-size": 5,
            "g-null-logistic-nonconvergence": "warn",
        }
    )
    run_paths = (
        output.OutputRunPaths(Path("run/trait_a"), Path("run/trait_a/chunks")),
        output.OutputRunPaths(Path("run/trait_b"), Path("run/trait_b/chunks")),
    )

    with (
        patch(
            "g.execution_plan.output.prepare_output_run",
            side_effect=(
                output.PreparedOutputRun(run_paths[0], None),
                output.PreparedOutputRun(run_paths[1], None),
            ),
        ),
        patch("g.runner.run_regenie2_multi_phenotype_binary_bgen_pipeline") as mock_binary_pipeline,
    ):
        plan = execution_plan.build_regenie_execution_plan(regenie_config)
        runner.dispatch_multi_phenotype_engine_pipeline(
            plan=plan,
            stage_timing_recorder=None,
        )

    assert mock_binary_pipeline.call_args.kwargs["kernel_config"] is plan.kernel_config.binary_kernel_config
    assert mock_binary_pipeline.call_args.kwargs["kernel_config"].firth_candidate.batch_size == 5
    assert (
        mock_binary_pipeline.call_args.kwargs["null_logistic_nonconvergence_policy"]
        == types.NullLogisticNonconvergencePolicy.WARN
    )


def test_regenie_from_options_dispatches_multiple_phenotypes() -> None:
    with patch("g.api.runner.regenie") as mock_runner_regenie:
        mock_runner_regenie.return_value = api.RunArtifacts(
            phenotype_artifacts=(
                api.RunArtifacts(output_run_directory=Path("one")),
                api.RunArtifacts(output_run_directory=Path("two")),
            )
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

    with (
        patch(
            "g.execution_plan.output.prepare_output_run",
            return_value=output.PreparedOutputRun(run_paths, None),
        ),
        patch("g.runner.run_regenie2_multi_phenotype_linear_bgen_pipeline") as mock_multi_pipeline,
    ):
        plan = execution_plan.build_regenie_execution_plan(regenie_config)
        runner.dispatch_execution_plan(plan=plan, stage_timing_recorder=None)

    mock_multi_pipeline.assert_called_once()
    assert mock_multi_pipeline.call_args.kwargs["sample_mode"] == types.MultiPhenotypeSampleMode.PER_PHENOTYPE


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
            "g-multi-phenotype-sample-mode": "complete-case",
        }
    )
    run_paths = (
        output.OutputRunPaths(Path("run/one"), Path("run/one/chunks")),
        output.OutputRunPaths(Path("run/two"), Path("run/two/chunks")),
    )
    existing_manifests = ({"phenotype_name": "one"}, {"phenotype_name": "two"})

    with patch(
        "g.execution_plan.output.prepare_output_run",
        side_effect=(
            output.PreparedOutputRun(run_paths[0], existing_manifests[0]),
            output.PreparedOutputRun(run_paths[1], existing_manifests[1]),
        ),
    ):
        plan = execution_plan.build_regenie_execution_plan(regenie_config)

    assert tuple(phenotype_plan.output_run_paths for phenotype_plan in plan.phenotype_run_plans) == run_paths
    assert tuple(phenotype_plan.existing_manifest for phenotype_plan in plan.phenotype_run_plans) == existing_manifests
    with patch("g.runner.run_regenie2_multi_phenotype_linear_bgen_pipeline") as mock_pipeline:
        runner.dispatch_multi_phenotype_engine_pipeline(
            plan=plan,
            stage_timing_recorder=None,
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
        "g.execution_plan.output.prepare_output_run",
        return_value=output.PreparedOutputRun(run_paths, None),
    ):
        plan = execution_plan.build_regenie_execution_plan(regenie_config)

    runner.extend_run_manifest(plan=plan, phenotype_run_plan=plan.phenotype_run_plans[0])

    manifest = output.load_run_manifest(run_paths)
    assert manifest is not None
    assert manifest["command"]["interface"] == "g regenie"
    assert manifest["command"]["phenotype"] == "trait"
    assert manifest["bgen"] == {"path": "/inputs/dataset.bgen", "size": 1, "mtime_ns": 2}
    assert "input_fingerprints" not in manifest
