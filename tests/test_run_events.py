from __future__ import annotations

import typing
from pathlib import Path

from g import _core, types
from g.engine import run_events, shutdown


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

    native_payload = _core.build_run_completed_event_payload(artifacts)
    event = run_events.build_run_completed_event(artifacts)

    assert native_payload["phenotype_count"] == 2
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

    native_payload = _core.attach_run_metadata_payload(
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
    assert phenotype_payloads[0]["run_id"] == "run-1"
    assert phenotype_payloads[1]["association_mode"] == "regenie2_linear"
    assert attached_artifacts.run_id == "run-1"
    assert attached_artifacts.association_mode == types.AssociationMode.REGENIE2_LINEAR
    assert attached_artifacts.phenotype_count == 2
    assert attached_artifacts.phenotype_artifacts[1].run_id == "run-1"
    assert attached_artifacts.phenotype_artifacts[1].association_mode == types.AssociationMode.REGENIE2_LINEAR
    assert attached_artifacts.phenotype_artifacts[1].phenotype_count == 2


def test_execution_run_artifacts_uses_native_artifact_tree_builder() -> None:
    native_payload = _core.build_execution_run_artifacts_payload(
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


def test_execution_run_artifacts_single_phenotype_has_no_wrapper() -> None:
    native_payload = _core.build_execution_run_artifacts_payload(
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


def test_run_completed_telemetry_fields_use_native_payload_builder() -> None:
    event = run_events.build_run_completed_event(
        run_events.RunArtifacts(
            output_run_directory=Path("out/run"),
            final_dataset=Path("out/dataset"),
            final_parquet=Path("out/final.parquet"),
            final_regenie=None,
            effective_config=Path("out/effective_config.toml"),
            phenotype_artifacts=(),
            phenotype_name="height",
            association_mode=types.AssociationMode.REGENIE2_LINEAR,
            phenotype_count=1,
            run_id="run-1",
        )
    )

    telemetry_fields = run_events.run_completed_telemetry_fields(event)

    assert telemetry_fields == {
        "artifact_count": 1,
        "phenotype_artifacts": (
            {
                "phenotype": "height",
                "output_run_directory": "out/run",
                "final_dataset": "out/dataset",
                "final_parquet": "out/final.parquet",
                "effective_config": "out/effective_config.toml",
            },
        ),
        "run_id": "run-1",
        "association_mode": "regenie2_linear",
        "phenotype_count": 1,
        "phenotype": "height",
        "output_run_directory": "out/run",
        "final_dataset": "out/dataset",
        "final_parquet": "out/final.parquet",
        "effective_config": "out/effective_config.toml",
    }


def test_runner_lifecycle_diagnostic_payloads_use_native_builders() -> None:
    started_payload = run_events.build_runner_run_started_diagnostic_payload(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        trait_type=types.RegenieTraitType.QUANTITATIVE,
        phenotype_count=2,
    )
    interrupted_event = run_events.RunInterruptedEvent(
        signal_number=2,
        signal_name="SIGINT",
        exit_code=130,
        flushed_for_resume=True,
    )
    failed_event = run_events.RunFailedEvent(error_type="RuntimeError", error_message="boom")
    completed_event = run_events.RunCompletedEvent(
        run_id="run-1",
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        phenotype_count=2,
        artifacts=(),
    )

    assert started_payload == {
        "level": "info",
        "event_name": "runner_regenie_run_started",
        "message": "Starting REGENIE run.",
        "fields": {
            "association_mode": "regenie2_linear",
            "trait_type": "quantitative",
            "phenotype_count": 2,
        },
    }
    assert run_events.build_runner_run_interrupted_diagnostic_payload(interrupted_event) == {
        "level": "warn",
        "event_name": "runner_regenie_run_interrupted",
        "message": "REGENIE run interrupted by SIGINT.",
        "fields": {
            "signal_number": 2,
            "signal_name": "SIGINT",
            "exit_code": 130,
            "flushed_for_resume": True,
        },
    }
    assert run_events.build_runner_run_failed_diagnostic_payload(failed_event) == {
        "level": "error",
        "event_name": "runner_regenie_run_failed",
        "message": "REGENIE run failed.",
        "fields": {
            "error_type": "RuntimeError",
            "error_message": "boom",
        },
    }
    assert run_events.build_runner_run_completed_diagnostic_payload(completed_event) == {
        "level": "info",
        "event_name": "runner_regenie_run_completed",
        "message": "Finished REGENIE run.",
        "fields": {
            "run_id": "run-1",
            "association_mode": "regenie2_linear",
            "phenotype_count": 2,
        },
    }


def test_runner_execution_plan_diagnostic_payloads_use_native_builders() -> None:
    assert run_events.build_runner_jax_runtime_configuration_started_diagnostic_payload() == {
        "level": "debug",
        "event_name": "runner_jax_runtime_configuration_started",
        "message": "Configuring JAX runtime before backend initialization.",
        "fields": {},
    }
    assert run_events.build_runner_execution_plan_build_started_diagnostic_payload() == {
        "level": "debug",
        "event_name": "runner_execution_plan_build_started",
        "message": "Building REGENIE execution plan.",
        "fields": {},
    }
    assert run_events.build_runner_execution_plan_prepared_diagnostic_payload(
        association_mode=types.AssociationMode.REGENIE2_BINARY,
        phenotype_count=3,
        chunk_size=1024,
        variant_limit=4096,
        device=types.Device.GPU,
    ) == {
        "level": "info",
        "event_name": "runner_execution_plan_prepared",
        "message": "Prepared REGENIE execution plan for 3 phenotype(s).",
        "fields": {
            "association_mode": "regenie2_binary",
            "phenotype_count": 3,
            "chunk_size": 1024,
            "variant_limit": 4096,
            "device": "gpu",
        },
    }
    assert run_events.build_runner_execution_plan_dispatch_started_diagnostic_payload(
        phenotype_count=3,
        association_mode=types.AssociationMode.REGENIE2_BINARY,
    ) == {
        "level": "debug",
        "event_name": "runner_execution_plan_dispatch_started",
        "message": "Dispatching REGENIE execution plan.",
        "fields": {"phenotype_count": 3, "association_mode": "regenie2_binary"},
    }
    assert run_events.build_runner_execution_plan_finalization_started_diagnostic_payload(
        phenotype_count=3,
        association_mode=types.AssociationMode.REGENIE2_BINARY,
    ) == {
        "level": "debug",
        "event_name": "runner_execution_plan_finalization_started",
        "message": "Finalizing REGENIE execution plan.",
        "fields": {"phenotype_count": 3, "association_mode": "regenie2_binary"},
    }


def test_runner_dispatch_diagnostic_payloads_use_native_builders() -> None:
    assert run_events.build_runner_multi_phenotype_dispatch_started_diagnostic_payload(
        phenotype_count=3,
        association_mode=types.AssociationMode.REGENIE2_BINARY,
    ) == {
        "level": "debug",
        "event_name": "runner_multi_phenotype_dispatch_started",
        "message": "Dispatching multi-phenotype native engine pipeline.",
        "fields": {"phenotype_count": 3, "association_mode": "regenie2_binary"},
    }
    assert run_events.build_runner_single_phenotype_dispatch_started_diagnostic_payload(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        phenotype="height",
    ) == {
        "level": "debug",
        "event_name": "runner_single_phenotype_dispatch_started",
        "message": "Dispatching single-phenotype native engine pipeline.",
        "fields": {"association_mode": "regenie2_linear", "phenotype": "height"},
    }
    assert run_events.build_runner_binary_engine_dispatch_started_diagnostic_payload(
        phenotype="height",
    ) == {
        "level": "debug",
        "event_name": "runner_binary_engine_dispatch_started",
        "message": "Dispatching binary native engine pipeline.",
        "fields": {"phenotype": "height"},
    }
    assert run_events.build_runner_linear_engine_dispatch_started_diagnostic_payload(
        phenotype="height",
    ) == {
        "level": "debug",
        "event_name": "runner_linear_engine_dispatch_started",
        "message": "Dispatching linear native engine pipeline.",
        "fields": {"phenotype": "height"},
    }
    assert run_events.build_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_payload(
        phenotype_count=3,
    ) == {
        "level": "debug",
        "event_name": "runner_multi_phenotype_binary_engine_dispatch_started",
        "message": "Dispatching multi-phenotype binary native engine pipeline.",
        "fields": {"phenotype_count": 3},
    }
    assert run_events.build_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_payload(
        phenotype_count=3,
    ) == {
        "level": "debug",
        "event_name": "runner_multi_phenotype_linear_engine_dispatch_started",
        "message": "Dispatching multi-phenotype linear native engine pipeline.",
        "fields": {"phenotype_count": 3},
    }


def test_native_runtime_knobs_diagnostic_payload_uses_native_builder() -> None:
    assert run_events.build_native_runtime_knobs_configured_diagnostic_payload(
        bgen_decode_tile_variant_count=32,
        threads=None,
    ) == {
        "level": "debug",
        "event_name": "native_runtime_knobs_configured",
        "message": "Configuring native runtime knobs.",
        "fields": {"bgen_decode_tile_variant_count": 32, "threads": None},
    }
    assert run_events.build_native_runtime_knobs_configured_diagnostic_payload(
        bgen_decode_tile_variant_count=32,
        threads=4,
    ) == {
        "level": "debug",
        "event_name": "native_runtime_knobs_configured",
        "message": "Configuring native runtime knobs.",
        "fields": {"bgen_decode_tile_variant_count": 32, "threads": 4},
    }


def test_runner_metadata_artifacts_finalized_diagnostic_payload_uses_native_builder() -> None:
    assert run_events.build_runner_metadata_artifacts_finalized_diagnostic_payload(
        association_mode=types.AssociationMode.REGENIE2_BINARY,
        phenotype_count=3,
    ) == {
        "level": "info",
        "event_name": "runner_metadata_artifacts_finalized",
        "message": "Finalized REGENIE run artifacts for 3 phenotype(s).",
        "fields": {"association_mode": "regenie2_binary", "phenotype_count": 3},
    }


def test_preflight_warning_diagnostic_payload_uses_native_builder() -> None:
    assert run_events.build_preflight_warning_diagnostic_payload(
        message="REGENIE step 2 is running with fewer than 10 residual degrees of freedom.",
        chromosome_count=1,
        covariate_count=2,
        preflight_scope="single_trait",
        sample_count=3,
        trusted_no_missing_diploid=True,
        warning_index=0,
    ) == {
        "level": "warning",
        "event_name": "preflight_warning",
        "message": "REGENIE step 2 is running with fewer than 10 residual degrees of freedom.",
        "fields": {
            "chromosome_count": 1,
            "covariate_count": 2,
            "preflight_scope": "single_trait",
            "sample_count": 3,
            "trusted_no_missing_diploid": True,
            "warning_index": 0,
        },
    }


def test_io_output_resume_committed_chunks_diagnostic_payload_uses_native_builder() -> None:
    assert run_events.build_io_output_resume_committed_chunks_diagnostic_payload(
        chunks_directory="out/chunks",
        committed_chunk_count=2,
        run_directory="out/run",
    ) == {
        "level": "info",
        "event_name": "io_output_resume_committed_chunks",
        "message": "Resuming run with 2 previously committed chunks.",
        "fields": {
            "chunks_directory": "out/chunks",
            "committed_chunk_count": 2,
            "run_directory": "out/run",
        },
    }


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
    native_payload = _core.build_run_interrupted_event_payload(shutdown_request)
    event = run_events.build_run_interrupted_event(shutdown_request)

    assert native_payload == {
        "signal_number": 2,
        "signal_name": "SIGINT",
        "exit_code": 130,
        "flushed_for_resume": True,
    }
    assert event.signal_number == 2
    assert event.signal_name == "SIGINT"
    assert event.exit_code == 130
    assert event.flushed_for_resume is True
    assert run_events.run_interrupted_telemetry_fields(event) == {
        "failure_kind": "graceful_shutdown",
        "signal_number": 2,
        "signal_name": "SIGINT",
        "exit_code": 130,
        "flushed_for_resume": True,
    }
    assert run_events.render_run_interrupted_lines(event) == (
        "Interrupted by SIGINT. Flushed queued chunks and saved committed output for --resume.",
    )


def test_failed_run_event_uses_native_payload_builder_and_renderer() -> None:
    error = RuntimeError("boom")
    native_payload = _core.build_run_failed_event_payload(error)
    event = run_events.build_run_failed_event(error)

    assert native_payload == {"error_type": "RuntimeError", "error_message": "boom"}
    assert event.error_type == "RuntimeError"
    assert event.error_message == "boom"
    assert run_events.run_failed_telemetry_fields(event) == {
        "failure_kind": "exception",
        "error_type": "RuntimeError",
        "error_message": "boom",
    }
    assert run_events.render_run_failed_lines(event) == ("Error: boom",)
