from __future__ import annotations

import json
import typing
from pathlib import Path

from g import _core, types
from g.interface import config as interface_config
from g.runner import events as run_events
from g.runner import lifecycle as shutdown


def test_run_completed_event_uses_native_artifact_tree_builder() -> None:
    artifacts = run_events.RunArtifacts(
        output_run_directory=None,
        final_dataset=None,
        final_parquet=None,
        final_regenie=None,
        effective_config=None,
        phenotype_artifacts=(
            run_events.RunArtifacts(
                output_run_directory=Path("out/height/run"),
                final_dataset=None,
                final_parquet=Path("out/height/final.parquet"),
                final_regenie=None,
                effective_config=Path("out/height/effective_config.toml"),
                phenotype_artifacts=(),
                phenotype_name="height",
                association_mode=None,
                phenotype_count=None,
                run_id=None,
            ),
            run_events.RunArtifacts(
                output_run_directory=Path("out/weight/run"),
                final_dataset=None,
                final_parquet=Path("out/weight/final.parquet"),
                final_regenie=None,
                effective_config=Path("out/weight/effective_config.toml"),
                phenotype_artifacts=(),
                phenotype_name="weight",
                association_mode=None,
                phenotype_count=None,
                run_id=None,
            ),
        ),
        phenotype_name=None,
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        phenotype_count=None,
        run_id="run-1",
    )

    native_policy = _core.NativeRunEventPayloadPolicy()
    native_payload = native_policy.build_run_completed_event_payload(artifacts)
    native_event = native_policy.build_run_completed_event(artifacts)
    event = run_events.build_run_completed_event(artifacts)

    assert native_payload["phenotype_count"] == 2
    assert isinstance(native_event, _core.NativeRunCompletedEvent)
    assert native_event.phenotype_count == 2
    assert native_event.artifacts[1].phenotype_name == "weight"
    assert event.run_id == "run-1"
    assert event.association_mode == types.AssociationMode.REGENIE2_LINEAR
    assert event.phenotype_count == 2
    assert tuple(artifact.phenotype_name for artifact in event.artifacts) == ("height", "weight")
    assert event.artifacts[1].output_run_directory == Path("out/weight/run")
    assert event.artifacts[1].final_parquet == Path("out/weight/final.parquet")


def test_attach_run_metadata_uses_native_artifact_tree_builder() -> None:
    artifacts = run_events.RunArtifacts(
        output_run_directory=None,
        final_dataset=None,
        final_parquet=None,
        final_regenie=None,
        effective_config=None,
        phenotype_artifacts=(
            run_events.RunArtifacts(
                output_run_directory=Path("out/height/run"),
                final_dataset=None,
                final_parquet=Path("out/height/final.parquet"),
                final_regenie=None,
                effective_config=Path("out/height/effective_config.toml"),
                phenotype_artifacts=(),
                phenotype_name="height",
                association_mode=None,
                phenotype_count=None,
                run_id=None,
            ),
            run_events.RunArtifacts(
                output_run_directory=Path("out/weight/run"),
                final_dataset=None,
                final_parquet=Path("out/weight/final.parquet"),
                final_regenie=None,
                effective_config=Path("out/weight/effective_config.toml"),
                phenotype_artifacts=(),
                phenotype_name="weight",
                association_mode=None,
                phenotype_count=None,
                run_id=None,
            ),
        ),
        phenotype_name=None,
        association_mode=None,
        phenotype_count=None,
        run_id=None,
    )

    native_policy = _core.NativeRunEventPayloadPolicy()
    native_payload = native_policy.attach_run_metadata_payload(
        artifacts,
        "run-1",
        types.AssociationMode.REGENIE2_LINEAR.value,
        2,
    )
    native_artifacts = native_policy.attach_run_metadata(
        artifacts,
        "run-1",
        types.AssociationMode.REGENIE2_LINEAR.value,
        2,
    )
    attached_artifacts = run_events.attach_run_metadata(
        artifacts,
        run_id="run-1",
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        phenotype_count=2,
    )
    phenotype_payloads = typing.cast("tuple[dict[str, object], ...]", native_payload["phenotype_artifacts"])

    assert native_payload["run_id"] == "run-1"
    assert native_payload["association_mode"] == "regenie2_linear"
    assert native_payload["phenotype_count"] == 2
    assert isinstance(native_artifacts, _core.NativeRunArtifacts)
    assert native_artifacts.run_id == "run-1"
    assert native_artifacts.phenotype_artifacts[1].association_mode == "regenie2_linear"
    assert phenotype_payloads[0]["run_id"] == "run-1"
    assert phenotype_payloads[1]["association_mode"] == "regenie2_linear"
    assert attached_artifacts.run_id == "run-1"
    assert attached_artifacts.association_mode == types.AssociationMode.REGENIE2_LINEAR
    assert attached_artifacts.phenotype_count == 2
    assert attached_artifacts.phenotype_artifacts[1].run_id == "run-1"
    assert attached_artifacts.phenotype_artifacts[1].association_mode == types.AssociationMode.REGENIE2_LINEAR
    assert attached_artifacts.phenotype_artifacts[1].phenotype_count == 2


def test_execution_run_artifacts_uses_native_artifact_tree_builder() -> None:
    native_metadata_builder = _core.NativeRunMetadataBuilder()
    native_payload = native_metadata_builder.build_execution_run_artifacts_payload(
        types.AssociationMode.REGENIE2_LINEAR.value,
        2,
        "parquet",
        ("out/height/run", "out/weight/run"),
        ("out/height/run/parts", "out/weight/run/parts"),
        ("out/height/effective_config.toml", "out/weight/effective_config.toml"),
        ("height", "weight"),
        ("out/height/final.parquet", "out/weight/final.parquet"),
    )
    artifacts = run_events.run_artifacts_from_native_payload(native_payload)
    phenotype_payloads = typing.cast("tuple[dict[str, object], ...]", native_payload["phenotype_artifacts"])

    assert native_payload["output_run_directory"] is None
    assert native_payload["association_mode"] == types.AssociationMode.REGENIE2_LINEAR.value
    assert native_payload["phenotype_count"] == 2
    assert phenotype_payloads[1]["phenotype_name"] == "weight"
    assert phenotype_payloads[1]["final_dataset"] == "out/weight/run/parts"
    assert phenotype_payloads[1]["final_parquet"] == "out/weight/final.parquet"
    assert artifacts.phenotype_artifacts[0].phenotype_name == "height"
    assert artifacts.phenotype_artifacts[1].final_dataset == Path("out/weight/run/parts")
    assert artifacts.phenotype_artifacts[1].phenotype_count == 2
    assert not hasattr(_core, "build_execution_run_artifacts_payload")
    assert not hasattr(_core, "extend_run_manifest_metadata")


def test_run_start_metadata_uses_native_effective_config_and_manifest_writer(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    effective_config_path = run_directory / "effective_config.toml"
    regenie_config = interface_config.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": "dataset.bgen",
            "sample": "dataset.sample",
            "phenoFile": "phenotype.tsv",
            "phenoCol": "height",
            "pred": "predictions.list",
            "out": str(tmp_path / "output"),
        }
    )
    trusted_no_missing_diploid = False

    _core.NativeRunMetadataBuilder().write_run_start_metadata(
        regenie_config,
        str(run_directory),
        "height",
        str(effective_config_path),
        "parquet",
        "cpu",
        2,
        16,
        None,
        4,
        8,
        16,
        "zstd",
        "none",
        "float32",
        512,
        trusted_no_missing_diploid,
        "full",
    )

    manifest_payload = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    assert "default-config-hash" in effective_config_path.read_text(encoding="utf-8")
    assert manifest_payload["command"]["effective_config"] == str(effective_config_path)
    assert manifest_payload["command"]["phenotype"] == "height"
    assert manifest_payload["runtime"]["device"] == "cpu"
    assert manifest_payload["runtime"]["writer_threads"] == 4


def test_execution_run_artifacts_single_phenotype_has_no_wrapper() -> None:
    native_metadata_builder = _core.NativeRunMetadataBuilder()
    native_payload = native_metadata_builder.build_execution_run_artifacts_payload(
        types.AssociationMode.REGENIE2_LINEAR.value,
        1,
        "regenie",
        ("out/height/run",),
        ("out/height/run/parts",),
        ("out/height/effective_config.toml",),
        ("height",),
        ("out/height.regenie",),
    )
    artifacts = run_events.run_artifacts_from_native_payload(native_payload)

    assert native_payload["output_run_directory"] == "out/height/run"
    assert native_payload["phenotype_artifacts"] == ()
    assert native_payload["final_dataset"] is None
    assert native_payload["final_regenie"] == "out/height.regenie"
    assert artifacts.phenotype_artifacts == ()
    assert artifacts.final_regenie == Path("out/height.regenie")


def test_run_completed_event_preserves_missing_native_metadata() -> None:
    event = run_events.build_run_completed_event(
        run_events.RunArtifacts(
            output_run_directory=Path("out/run"),
            final_dataset=None,
            final_parquet=None,
            final_regenie=None,
            effective_config=None,
            phenotype_artifacts=(),
            phenotype_name=None,
            association_mode=None,
            phenotype_count=None,
            run_id=None,
        )
    )

    assert event.run_id is None
    assert event.association_mode is None
    assert event.phenotype_count is None
    assert event.artifacts[0].output_run_directory == Path("out/run")


def test_recorder_only_diagnostic_payload_builders_are_not_exported() -> None:
    builder_names = (
        "build_runner_run_started_diagnostic_payload",
        "build_runner_run_interrupted_diagnostic_payload",
        "build_runner_run_failed_diagnostic_payload",
        "build_runner_run_completed_diagnostic_payload",
        "build_runner_jax_runtime_configuration_started_diagnostic_payload",
        "build_runner_execution_plan_build_started_diagnostic_payload",
        "build_runner_execution_plan_prepared_diagnostic_payload",
        "build_runner_execution_plan_dispatch_started_diagnostic_payload",
        "build_runner_execution_plan_finalization_started_diagnostic_payload",
        "build_runner_multi_phenotype_dispatch_started_diagnostic_payload",
        "build_runner_single_phenotype_dispatch_started_diagnostic_payload",
        "build_runner_binary_engine_dispatch_started_diagnostic_payload",
        "build_runner_linear_engine_dispatch_started_diagnostic_payload",
        "build_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_payload",
        "build_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_payload",
        "build_native_cli_stdout_diagnostic_payload",
        "build_native_cli_stderr_diagnostic_payload",
        "build_native_cli_interrupted_line_diagnostic_payload",
        "build_native_cli_failed_line_diagnostic_payload",
        "build_native_cli_completed_line_diagnostic_payload",
        "build_native_runtime_knobs_configured_diagnostic_payload",
        "build_runner_metadata_artifacts_finalized_diagnostic_payload",
        "build_preflight_warning_diagnostic_payload",
        "build_io_output_resume_committed_chunks_diagnostic_payload",
        "build_pipeline_bgen_engine_open_started_diagnostic_payload",
        "build_pipeline_bgen_engine_opened_diagnostic_payload",
        "build_pipeline_prevalidated_bgen_engine_used_diagnostic_payload",
        "build_pipeline_output_resume_committed_chunks_diagnostic_payload",
        "build_pipeline_output_writer_sessions_create_started_diagnostic_payload",
        "build_pipeline_gpu_genotype_format_resolved_diagnostic_payload",
        "build_callback_null_logistic_nonconvergence_warning_diagnostic_payload",
        "build_pipeline_multi_phenotype_sample_summary_diagnostic_payload",
        "build_pipeline_multi_trait_started_diagnostic_payload",
        "build_pipeline_multi_trait_input_load_started_diagnostic_payload",
        "build_pipeline_multi_trait_input_aligned_diagnostic_payload",
        "build_pipeline_multi_trait_prediction_source_load_started_diagnostic_payload",
        "build_pipeline_grouped_per_phenotype_started_diagnostic_payload",
        "build_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_payload",
        "build_pipeline_grouped_union_delivery_selected_diagnostic_payload",
        "build_pipeline_multi_group_preflight_started_diagnostic_payload",
        "build_pipeline_multi_group_preflight_completed_diagnostic_payload",
        "build_pipeline_single_trait_started_diagnostic_payload",
        "build_pipeline_single_trait_input_load_started_diagnostic_payload",
        "build_pipeline_single_trait_input_aligned_diagnostic_payload",
        "build_pipeline_single_trait_prediction_source_load_started_diagnostic_payload",
        "build_pipeline_single_trait_preflight_started_diagnostic_payload",
        "build_pipeline_single_trait_preflight_completed_diagnostic_payload",
        "build_native_dispatch_bgen_engine_constructing_diagnostic_payload",
        "build_native_dispatch_trusted_bgen_validation_started_diagnostic_payload",
        "build_native_dispatch_callback_drain_started_diagnostic_payload",
        "build_native_dispatch_delivery_started_diagnostic_payload",
        "build_native_dispatch_delivery_finished_diagnostic_payload",
        "build_native_dispatch_delivery_interrupted_diagnostic_payload",
        "build_native_dispatch_delivery_failed_diagnostic_payload",
        "build_native_dispatch_pipeline_finished_diagnostic_payload",
        "build_native_dispatch_writer_session_finish_started_diagnostic_payload",
        "build_native_dispatch_writer_sessions_finish_started_diagnostic_payload",
        "build_native_dispatch_writer_session_interrupted_flush_started_diagnostic_payload",
        "build_native_dispatch_writer_sessions_interrupted_flush_started_diagnostic_payload",
    )

    for builder_name in builder_names:
        assert not hasattr(run_events, builder_name)
        assert not hasattr(_core, builder_name)


def test_run_event_telemetry_field_builders_are_not_exported() -> None:
    builder_names = (
        "build_run_completed_telemetry_fields",
        "build_run_interrupted_telemetry_fields",
        "build_run_failed_telemetry_fields",
    )
    wrapper_names = (
        "run_completed_telemetry_fields",
        "run_interrupted_telemetry_fields",
        "run_failed_telemetry_fields",
    )

    for builder_name in builder_names:
        assert not hasattr(_core, builder_name)
    for wrapper_name in wrapper_names:
        assert not hasattr(run_events, wrapper_name)


def test_run_completed_rendering_uses_native_renderer() -> None:
    event = run_events.RunCompletedEvent(
        run_id=None,
        association_mode=None,
        phenotype_count=None,
        artifacts=(
            run_events.RunArtifactPayload(
                phenotype_name=None,
                output_run_directory=Path("out/run"),
                final_dataset=Path("out/dataset"),
                final_parquet=Path("out/final.parquet"),
                final_regenie=Path("out/final.regenie"),
                effective_config=None,
            ),
        ),
    )

    assert run_events.render_run_completed_lines(event) == (
        "Success. Chunked run saved to out/run",
        "Parquet dataset saved to out/dataset",
        "Finalized Parquet saved to out/final.parquet",
        "REGENIE text output saved to out/final.regenie",
    )


def test_interrupted_run_event_uses_native_payload_builder_and_renderer() -> None:
    shutdown_request = shutdown.GracefulShutdownRequested(
        shutdown.ShutdownSignal(number=2, name="SIGINT", exit_code=130)
    )
    native_policy = _core.NativeRunEventPayloadPolicy()
    native_payload = native_policy.build_run_interrupted_event_payload(shutdown_request)
    native_event = native_policy.build_run_interrupted_event(shutdown_request)
    event = run_events.build_run_interrupted_event(shutdown_request)

    assert native_payload == {
        "signal_number": 2,
        "signal_name": "SIGINT",
        "exit_code": 130,
        "flushed_for_resume": True,
    }
    assert isinstance(native_event, _core.NativeRunInterruptedEvent)
    assert native_event.signal_name == "SIGINT"
    assert event.signal_number == 2
    assert event.signal_name == "SIGINT"
    assert event.exit_code == 130
    assert event.flushed_for_resume is True
    assert run_events.render_run_interrupted_lines(event) == (
        "Interrupted by SIGINT. Flushed queued chunks and saved committed output for --resume.",
    )


def test_failed_run_event_uses_native_payload_builder_and_renderer() -> None:
    error = RuntimeError("boom")
    native_policy = _core.NativeRunEventPayloadPolicy()
    native_payload = native_policy.build_run_failed_event_payload(error)
    native_event = native_policy.build_run_failed_event(error)
    event = run_events.build_run_failed_event(error)

    assert native_payload == {"error_type": "RuntimeError", "error_message": "boom"}
    assert isinstance(native_event, _core.NativeRunFailedEvent)
    assert native_event.error_message == "boom"
    assert event.error_type == "RuntimeError"
    assert event.error_message == "boom"
    assert run_events.render_run_failed_lines(event) == ("Error: boom",)
