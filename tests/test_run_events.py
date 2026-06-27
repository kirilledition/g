from __future__ import annotations

from pathlib import Path

from g import types
from g.engine import run_events


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
    event = run_events.RunInterruptedEvent(
        signal_number=2,
        signal_name="SIGINT",
        exit_code=130,
        flushed_for_resume=True,
    )

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
    event = run_events.RunFailedEvent(error_type="RuntimeError", error_message="boom")

    assert run_events.run_failed_telemetry_fields(event) == {
        "failure_kind": "exception",
        "error_type": "RuntimeError",
        "error_message": "boom",
    }
    assert run_events.render_run_failed_lines(event) == ("Error: boom",)
