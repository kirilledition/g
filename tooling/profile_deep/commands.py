"""Command construction for application profiling runs."""

from __future__ import annotations

import json
import sys
import textwrap
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from tooling.profile_deep import models as profile_deep_models


def write_trial_config(
    *,
    candidate: profile_deep_models.Step2Candidate,
    output_prefix: Path,
    jax_cache_directory: Path | None,
    diagnostic_options: dict[str, object] | None,
) -> Path:
    """Write the supported runtime settings for one native CLI trial."""
    telemetry = str((diagnostic_options or {}).get("telemetry", "off"))
    if telemetry not in {"off", "progress", "profile"}:
        message = f"Unsupported telemetry mode for the current application: {telemetry!r}."
        raise ValueError(message)

    config_lines = [
        "[compute]",
        f"device = {json.dumps(candidate.device)}",
    ]
    if candidate.rayon_thread_count is not None:
        config_lines.append(f"cpu_threads = {candidate.rayon_thread_count}")
    if candidate.firth_batch_size is not None:
        config_lines.append(f"firth_batch_size = {candidate.firth_batch_size}")
    if jax_cache_directory is not None:
        config_lines.append(f"jax_cache_dir = {json.dumps(str(jax_cache_directory))}")
    config_lines.extend(
        [
            "",
            "[output]",
            f"writer_threads = {candidate.output_writer_thread_count}",
            "resume = false",
            "",
            "[diagnostics]",
            f"telemetry = {json.dumps(telemetry)}",
            "",
        ]
    )
    config_path = Path(f"{output_prefix}.profile.toml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(config_lines), encoding="utf-8")
    return config_path


def build_cli_arguments(
    *,
    baseline_paths: typing.Any,
    candidate: profile_deep_models.Step2Candidate,
    output_prefix: Path,
    config_path: Path,
) -> list[str]:
    """Build arguments accepted by the current Rust-owned CLI."""
    is_binary_trait = candidate.trait_type == "binary"
    phenotype_path = (
        baseline_paths.binary_phenotype_path if is_binary_trait else baseline_paths.continuous_phenotype_path
    )
    phenotype_name = "phenotype_binary" if is_binary_trait else "phenotype_continuous"
    prediction_path = (
        baseline_paths.regenie_prediction_list_path
        if is_binary_trait
        else baseline_paths.regenie_qt_prediction_list_path
    )
    arguments = [
        "regenie",
        "--config",
        str(config_path),
        "--bt" if is_binary_trait else "--qt",
        "--bsize",
        str(candidate.chunk_size),
        "--bgen",
        str(baseline_paths.bgen_path),
        "--sample",
        str(baseline_paths.sample_path),
        "--phenoFile",
        str(phenotype_path),
        "--phenoCol",
        phenotype_name,
        "--covarFile",
        str(baseline_paths.covariate_path),
        "--covarCol",
        "age",
        "--covarCol",
        "sex",
        "--pred",
        str(prediction_path),
        "--out",
        str(output_prefix),
    ]
    if is_binary_trait:
        arguments.extend(["--binary-fallback", "firth_approximate"])
    return arguments


def build_g_step2_child_command(
    *,
    baseline_paths: typing.Any,
    candidate: profile_deep_models.Step2Candidate,
    output_prefix: Path,
    cache_directory: Path | None = None,
    stage_timing_path: Path | None = None,
    trace_directory: Path | None = None,
    memory_profile_path: Path | None = None,
    diagnostic_options: dict[str, object] | None = None,
) -> list[str]:
    """Build an isolated child that profiles the current native CLI boundary."""
    jax_cache_directory = cache_directory
    config_path = write_trial_config(
        candidate=candidate,
        output_prefix=output_prefix,
        jax_cache_directory=jax_cache_directory,
        diagnostic_options=diagnostic_options,
    )
    cli_arguments = build_cli_arguments(
        baseline_paths=baseline_paths,
        candidate=candidate,
        output_prefix=output_prefix,
        config_path=config_path,
    )
    child_code = textwrap.dedent(
        """
        import json
        import shutil
        import time
        from pathlib import Path

        import g.cli
        from tooling.benchmark import native_lifecycle


        cli_arguments = json.loads({cli_arguments_payload!r})
        output_root = Path({output_root!r})
        trace_directory = {trace_directory!r}
        memory_profile_path = {memory_profile_path!r}
        requested_stage_timing_path = {stage_timing_path!r}
        jax_module = None
        if trace_directory is not None:
            import jax as jax_module

            jax_module.profiler.start_trace(trace_directory)
        try:
            start_time = time.perf_counter()
            exit_code = g.cli.run(cli_arguments)
            wall_time_seconds = time.perf_counter() - start_time
        finally:
            if trace_directory is not None:
                jax_module.profiler.stop_trace()
        if exit_code != 0:
            raise RuntimeError(f"g CLI exited with status {{exit_code}}.")
        run_directory = native_lifecycle.discover_completed_run_directory(
            expected_run_directory=None,
            output_root=output_root,
            glob_pattern={run_glob_pattern!r},
            run_label="deep profile child",
        )
        output_evidence = native_lifecycle.measure_completed_output_run(run_directory)
        output_paths = list(output_evidence.parquet_paths)
        if requested_stage_timing_path is not None:
            source_stage_timing_path = run_directory / "output_stage_timings.json"
            if source_stage_timing_path.exists():
                shutil.copyfile(source_stage_timing_path, requested_stage_timing_path)
        if memory_profile_path is not None:
            if jax_module is None:
                import jax as jax_module

            jax_module.profiler.save_device_memory_profile(memory_profile_path)
        profile_summary_path = output_root / "logs" / "profile.summary.json"
        print(json.dumps({{
            "wall_time_seconds": wall_time_seconds,
            "output_path": output_paths[0],
            "output_paths": output_paths,
            "application_output_run_directory": str(run_directory),
            "profile_summary_path": str(profile_summary_path) if profile_summary_path.exists() else None,
            "jax_cache_directory": {jax_cache_directory!r},
            "jax_persistent_cache_used": True,
        }}))
        """
    ).format(
        cli_arguments_payload=json.dumps(cli_arguments),
        output_root=str(Path(f"{output_prefix}.g")),
        trace_directory=str(trace_directory) if trace_directory is not None else None,
        memory_profile_path=str(memory_profile_path) if memory_profile_path is not None else None,
        stage_timing_path=str(stage_timing_path) if stage_timing_path is not None else None,
        jax_cache_directory=str(jax_cache_directory) if jax_cache_directory is not None else None,
        run_glob_pattern=("*.regenie2_binary.run" if candidate.trait_type == "binary" else "*.regenie2_linear.run"),
    )
    return [sys.executable, "-c", child_code]
