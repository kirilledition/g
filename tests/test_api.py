from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import g
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


def test_build_binary_kernel_config_maps_compute_options() -> None:
    kernel_config = execution_plan.build_binary_kernel_config(
        config.GComputeConfig(
            firth_batch_size=7,
            firth_candidate_capacity=11,
            binary_null_maximum_iterations=13,
            binary_null_coefficient_tolerance=1.0e-5,
            firth_maximum_iterations=17,
            firth_gradient_tolerance=2.0e-5,
            firth_coefficient_tolerance=3.0e-5,
            firth_likelihood_tolerance=4.0e-5,
            firth_maximum_step_size=6.0,
            use_block_firth_math=True,
        )
    )

    assert kernel_config.firth_batch_size == 7
    assert kernel_config.firth_candidate_capacity == 11
    assert kernel_config.maximum_null_iterations == 13
    assert kernel_config.use_block_firth_math is True


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
    assert mock_pipeline.call_args.kwargs["chunks_per_arrow_file"] == 4
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
        patch("g.runner.configure_runtime_before_jax_import", side_effect=record_jax_bootstrap),
        patch("g.execution_plan.output.prepare_output_run", side_effect=record_prepare_output_run),
        patch("g.runner.run_regenie2_linear_bgen_pipeline", side_effect=record_pipeline),
        patch("g.runner.extend_run_manifest"),
        patch("g.interface.config.write_toml"),
    ):
        api.regenie(build_minimal_config())

    assert call_order == ["jax", "plan", "pipeline"]


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

    with patch("g.runner.importlib.import_module", side_effect=import_module):
        runner.configure_runtime_before_jax_import(config.GComputeConfig(device=types.Device.GPU))

    assert call_order == ["import:jax", "jax_platforms:cuda", "import:g.jax_setup", "setup"]


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
            "g-firth-maximum-iterations": 17,
            "g-firth-gradient-tolerance": 2.0e-5,
            "g-firth-coefficient-tolerance": 3.0e-5,
            "g-firth-likelihood-tolerance": 4.0e-5,
            "g-firth-maximum-step-size": 6.0,
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
    assert kernel_config.firth_batch_size == 7
    assert kernel_config.firth_candidate_capacity == 11
    assert kernel_config.maximum_null_iterations == 13
    assert kernel_config.null_logistic_coefficient_tolerance == 1.0e-5
    assert kernel_config.firth_maximum_iterations == 17
    assert kernel_config.firth_gradient_tolerance == 2.0e-5
    assert kernel_config.firth_coefficient_tolerance == 3.0e-5
    assert kernel_config.firth_likelihood_tolerance == 4.0e-5
    assert kernel_config.firth_maximum_step_size == 6.0
    assert kernel_config.use_block_firth_math is True
    assert (
        mock_binary_pipeline.call_args.kwargs["correction_plan"].method == types.BinaryFallbackMethod.FIRTH_APPROXIMATE
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
    assert mock_binary_pipeline.call_args.kwargs["kernel_config"].firth_batch_size == 5


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
    assert mock_binary_pipeline.call_args.kwargs["kernel_config"].firth_batch_size == 5


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
