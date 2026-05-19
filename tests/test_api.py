from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import g
from g import api, types
from g.interface import config
from g.io import output, source
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
            "g-output-format": "both",
        }
    )

    assert regenie_config.input.bgen == Path("dataset.bgen")
    assert regenie_config.input.pheno_columns == ("trait_a", "trait_b")
    assert regenie_config.input.covar_columns == ("age", "sex")
    assert regenie_config.trait.trait_type == types.RegenieTraitType.BINARY
    assert regenie_config.binary.p_threshold == 0.01
    assert regenie_config.g_compute.device == types.Device.GPU
    assert regenie_config.g_output.format == types.OutputFormat.BOTH


def test_build_binary_kernel_config_maps_compute_options() -> None:
    kernel_config = api.build_binary_kernel_config(
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
    plan = api.normalize_binary_correction_config(config.BinaryConfig(firth=True, approx=True, p_threshold=0.01))

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
        patch("g.api.configure_jax_device") as mock_configure_jax_device,
        patch(
            "g.api.output.prepare_output_run",
            return_value=PreparedOutputRun(output_run_paths=run_paths, existing_manifest={"committed_chunks": []}),
        ) as mock_prepare_output_run,
        patch("g.api.run_regenie2_linear_bgen_pipeline") as mock_pipeline,
        patch("g.api.output.finalize_chunks_to_regenie_text") as mock_finalize_regenie,
        patch("g.api.extend_run_manifest") as mock_extend_run_manifest,
        patch("g.interface.config.write_toml") as mock_write_toml,
    ):
        mock_pipeline.return_value = Path("results/output.g/trait.regenie2_linear.run/final.parquet")
        artifacts = api.regenie(build_minimal_config())

    assert artifacts.output_run_directory == Path("results/output.g/trait.regenie2_linear.run")
    assert artifacts.final_parquet == Path("results/output.g/trait.regenie2_linear.run/final.parquet")
    assert artifacts.final_regenie is None
    mock_configure_jax_device.assert_called_once_with(types.Device.CPU)
    mock_prepare_output_run.assert_called_once()
    assert mock_pipeline.call_args.kwargs["existing_manifest"] == {"committed_chunks": []}
    assert mock_pipeline.call_args.kwargs["resume"] is False
    assert mock_pipeline.call_args.kwargs["covariate_names"] == ("age", "sex")
    assert mock_pipeline.call_args.kwargs["prediction_list_path"] == Path("predictions.list")
    assert mock_pipeline.call_args.kwargs["alignment_config"].sample_key_mode == types.SampleKeyMode.IID
    assert mock_pipeline.call_args.kwargs["chunks_per_arrow_file"] == 4
    assert mock_pipeline.call_args.kwargs["arrow_compression"] == types.ArrowCompression.ZSTD
    mock_finalize_regenie.assert_not_called()
    mock_extend_run_manifest.assert_called_once()
    mock_write_toml.assert_called_once()


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
        patch("g.api.configure_jax_device"),
        patch(
            "g.api.output.prepare_output_run",
            return_value=PreparedOutputRun(output_run_paths=run_paths, existing_manifest=None),
        ),
        patch("g.api.run_regenie2_binary_bgen_pipeline") as mock_binary_pipeline,
        patch("g.api.extend_run_manifest"),
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
        }
    )
    kernel_config = api.build_binary_kernel_config(config.GComputeConfig(firth_batch_size=5))
    engine_config = api.EngineRunConfig(
        chunk_size=32,
        device=types.Device.CPU,
        staging_depth=1,
        output_run_directory=Path("run"),
        resume=False,
        resume_mode=types.ResumeMode.FAST,
        finalize_parquet=True,
        writer_threads=1,
        writer_queue_depth=1,
        chunks_per_arrow_file=1,
        arrow_compression=types.ArrowCompression.ZSTD,
        trusted_no_missing_diploid=False,
        trusted_bgen_validation_mode=types.TrustedBgenValidationMode.CACHE_ON_MISS,
        alignment_config=regenie_config.g_compute,
        binary_kernel_config=kernel_config,
    )

    with patch("g.api.run_regenie2_binary_bgen_pipeline") as mock_binary_pipeline:
        api.dispatch_engine_pipeline(
            regenie_config=regenie_config,
            phenotype_name="trait",
            genotype_source_config=source.build_bgen_source_config(Path("dataset.bgen")),
            engine_config=engine_config,
            output_run_paths=output.OutputRunPaths(Path("run"), Path("run/chunks")),
            existing_manifest=None,
            binary_correction_plan=types.BinaryCorrectionPlan(method=types.BinaryFallbackMethod.FIRTH_APPROXIMATE),
            stage_timing_recorder=None,
        )

    assert mock_binary_pipeline.call_args.kwargs["kernel_config"] is kernel_config


def test_regenie_from_options_dispatches_multiple_phenotypes() -> None:
    with patch("g.api.run_multi_phenotype_config") as mock_run_multi_phenotype_config:
        mock_run_multi_phenotype_config.return_value = api.RunArtifacts(
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
    mock_run_multi_phenotype_config.assert_called_once()
    assert mock_run_multi_phenotype_config.call_args.args[0].input.pheno_columns == ("one", "two")


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
        "g.api.output.prepare_output_run",
        side_effect=(
            output.PreparedOutputRun(run_paths[0], existing_manifests[0]),
            output.PreparedOutputRun(run_paths[1], existing_manifests[1]),
        ),
    ):
        plan = api.build_regenie_multi_run_plan(regenie_config, Path("run"))

    assert plan.output_run_paths_by_phenotype == run_paths
    assert plan.existing_manifests_by_phenotype == existing_manifests
    with patch("g.api.run_regenie2_multi_phenotype_linear_bgen_pipeline") as mock_pipeline:
        api.dispatch_multi_engine_pipeline(
            regenie_config=regenie_config,
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

    api.extend_run_manifest(tmp_path, regenie_config, "trait", tmp_path / "effective_config.toml")

    manifest = output.load_run_manifest(run_paths)
    assert manifest is not None
    assert manifest["command"]["interface"] == "g regenie"
    assert manifest["command"]["phenotype"] == "trait"
    assert manifest["bgen"] == {"path": "/inputs/dataset.bgen", "size": 1, "mtime_ns": 2}
    assert "input_fingerprints" not in manifest
