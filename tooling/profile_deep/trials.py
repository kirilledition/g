"""Measured, discarded-warm and diagnostic application trials."""

from __future__ import annotations

import dataclasses
import json
import statistics
import typing

from tooling.profile_deep import application as profile_deep_application
from tooling.profile_deep import baseline as profile_deep_baseline
from tooling.profile_deep import budget as profile_deep_budget
from tooling.profile_deep import candidates as profile_deep_candidates
from tooling.profile_deep import config as profile_deep_config
from tooling.profile_deep import environment as profile_deep_environment
from tooling.profile_deep import execution as profile_deep_execution
from tooling.profile_deep import jax_cache as profile_deep_jax_cache
from tooling.profile_deep import models as profile_deep_models

if typing.TYPE_CHECKING:
    from pathlib import Path


def run_g_trial(
    *,
    name: str,
    baseline_paths: typing.Any,
    candidate: profile_deep_models.Step2Candidate,
    output_directory: Path,
    log_directory: Path,
    cache_directory: Path,
    emit_stage_timings: bool,
    trace_directory: Path | None = None,
    memory_profile_path: Path | None = None,
    diagnostic_options: dict[str, object] | None = None,
) -> profile_deep_models.TrialResult:
    """Run one g trial in a fresh Python process."""
    output_prefix = output_directory / name
    stage_timing_path = output_directory / f"{name}.stage_timings.json" if emit_stage_timings else None
    resolved_cache_directory = profile_deep_jax_cache.resolve_profile_jax_cache_directory(candidate, cache_directory)
    before_cache_snapshot = profile_deep_jax_cache.collect_jax_cache_snapshot(resolved_cache_directory)
    command_arguments = profile_deep_application.build_g_step2_child_command(
        baseline_paths=baseline_paths,
        candidate=candidate,
        output_prefix=output_prefix,
        cache_directory=cache_directory,
        stage_timing_path=stage_timing_path,
        trace_directory=trace_directory,
        memory_profile_path=memory_profile_path,
        diagnostic_options=diagnostic_options,
    )
    telemetry_mode = str((diagnostic_options or {}).get("telemetry", "off"))
    environment_overrides = profile_deep_environment.build_g_trial_environment(
        enable_jax_debug_logging=(
            telemetry_mode != "off" or trace_directory is not None or memory_profile_path is not None
        )
    )
    result = profile_deep_execution.run_logged_command(
        name=name,
        implementation="g",
        trait_type=candidate.trait_type,
        device=candidate.device,
        command_arguments=command_arguments,
        environment_overrides=environment_overrides,
        log_directory=log_directory,
    )
    after_cache_snapshot = profile_deep_jax_cache.collect_jax_cache_snapshot(resolved_cache_directory)
    application_metadata = profile_deep_application.collect_g_trial_application_metadata(
        stdout_log_path=result.stdout_log_path,
        expected_output_root=output_prefix.with_name(f"{output_prefix.name}.g"),
        require_child_artifacts=result.status == "success",
    )
    jax_cache_diagnostics = profile_deep_jax_cache.build_jax_cache_diagnostics(
        cache_directory=resolved_cache_directory,
        child_reported_cache_directory=application_metadata.child_reported_cache_directory,
        persistent_cache_used=True,
        before_snapshot=before_cache_snapshot,
        after_snapshot=after_cache_snapshot,
        stderr_log_path=result.stderr_log_path,
    )
    return dataclasses.replace(
        result,
        wall_time_seconds=(
            application_metadata.wall_time_seconds
            if application_metadata.wall_time_seconds is not None
            else result.wall_time_seconds
        ),
        output_row_count=application_metadata.output_row_count,
        output_path=application_metadata.output_path,
        stage_timing_path=str(stage_timing_path)
        if stage_timing_path is not None and stage_timing_path.exists()
        else None,
        profile_summary_path=application_metadata.profile_summary_path,
        application_output_prefix=str(output_prefix),
        application_output_run_directory=application_metadata.application_output_run_directory,
        device_diagnostics=application_metadata.device_diagnostics,
        jax_cache_diagnostics=jax_cache_diagnostics,
    )


def run_regenie_trial(
    *,
    name: str,
    trait_type: str,
    regenie_executable: str,
    baseline_paths: typing.Any,
    output_directory: Path,
    log_directory: Path,
    baseline_scope: profile_deep_models.RegenieBaselineScope,
) -> profile_deep_models.TrialResult:
    """Run one original REGENIE step 2 trial."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_prefix = output_directory / name
    regenie_profile_path = output_directory / f"{name}.regenie_profile.json"
    command_arguments = profile_deep_baseline.build_regenie_step2_command(
        trait_type=trait_type,
        regenie_executable=regenie_executable,
        baseline_paths=baseline_paths,
        output_prefix=output_prefix,
        baseline_scope=baseline_scope,
    )
    result = profile_deep_execution.run_logged_command(
        name=name,
        implementation="regenie",
        trait_type=trait_type,
        device="external_cpu",
        command_arguments=command_arguments,
        environment_overrides={"REGENIE_PROFILE_JSON": str(regenie_profile_path)},
        log_directory=log_directory,
    )
    output_row_count = count_regenie_step2_rows(output_prefix)
    output_suffix = "phenotype_binary" if trait_type == "binary" else "phenotype_continuous"
    output_path = output_prefix.parent / f"{output_prefix.name}_{output_suffix}.regenie"
    return dataclasses.replace(
        result,
        output_row_count=output_row_count,
        output_path=str(output_path) if output_path.exists() else None,
        regenie_profile_path=str(regenie_profile_path) if regenie_profile_path.exists() else None,
    )


def count_regenie_step2_rows(output_prefix: Path) -> int | None:
    """Count data rows in an upstream REGENIE step 2 output file."""
    result_path = output_prefix.with_name(f"{output_prefix.name}_phenotype_continuous.regenie")
    if not result_path.exists():
        result_path = output_prefix.with_name(f"{output_prefix.name}_phenotype_binary.regenie")
    if not result_path.exists():
        return None
    with result_path.open(encoding="utf-8") as result_file:
        line_count = sum(1 for line in result_file if line.strip())
    return max(0, line_count - 1)


def aggregate_trial_results(
    *,
    name: str,
    implementation: str,
    trait_type: str,
    device: str,
    warmup_count: int,
    trial_results: list[profile_deep_models.TrialResult],
    warmup_trials: list[profile_deep_models.TrialResult] | None = None,
    diagnostic_trials: list[profile_deep_models.TrialResult] | None = None,
    candidate: profile_deep_models.Step2Candidate | None = None,
) -> profile_deep_models.AggregateResult:
    """Aggregate successful measured trial results."""
    observed_warmup_trials = [] if warmup_trials is None else warmup_trials
    observed_diagnostic_trials = [] if diagnostic_trials is None else diagnostic_trials
    jax_cold_warm_summary = profile_deep_jax_cache.build_jax_cold_warm_diagnostics(
        warmup_trials=observed_warmup_trials,
        trial_results=trial_results,
    )
    successful_trials = [
        trial_result
        for trial_result in trial_results
        if trial_result.status == "success" and trial_result.wall_time_seconds is not None
    ]
    if not successful_trials:
        return profile_deep_models.AggregateResult(
            name=name,
            implementation=implementation,
            trait_type=trait_type,
            device=device,
            status="failed",
            trial_count=len(trial_results),
            warmup_count=warmup_count,
            median_wall_time_seconds=None,
            mean_wall_time_seconds=None,
            min_wall_time_seconds=None,
            max_wall_time_seconds=None,
            standard_deviation_seconds=None,
            rows_per_second=None,
            trials=trial_results,
            warmup_trials=observed_warmup_trials,
            diagnostic_trials=observed_diagnostic_trials,
            jax_cold_warm_summary=jax_cold_warm_summary,
            candidate=candidate,
        )
    wall_times = [typing.cast("float", trial_result.wall_time_seconds) for trial_result in successful_trials]
    row_counts = [
        trial_result.output_row_count for trial_result in successful_trials if trial_result.output_row_count is not None
    ]
    median_wall_time = statistics.median(wall_times)
    rows_per_second = None
    if row_counts and median_wall_time > 0.0:
        rows_per_second = statistics.median(row_counts) / median_wall_time
    return profile_deep_models.AggregateResult(
        name=name,
        implementation=implementation,
        trait_type=trait_type,
        device=device,
        status="success" if len(successful_trials) == len(trial_results) else "partial",
        trial_count=len(trial_results),
        warmup_count=warmup_count,
        median_wall_time_seconds=median_wall_time,
        mean_wall_time_seconds=statistics.fmean(wall_times),
        min_wall_time_seconds=min(wall_times),
        max_wall_time_seconds=max(wall_times),
        standard_deviation_seconds=statistics.stdev(wall_times) if len(wall_times) > 1 else 0.0,
        rows_per_second=rows_per_second,
        trials=trial_results,
        warmup_trials=observed_warmup_trials,
        diagnostic_trials=observed_diagnostic_trials,
        jax_cold_warm_summary=jax_cold_warm_summary,
        candidate=candidate,
    )


def run_repeated_g_trials(
    *,
    name: str,
    baseline_paths: typing.Any,
    candidate: profile_deep_models.Step2Candidate,
    output_directory: Path,
    log_directory: Path,
    cache_directory: Path,
    warmup_count: int,
    trial_count: int,
    emit_stage_timings: bool,
) -> profile_deep_models.AggregateResult:
    """Warm and measure one g candidate in fresh child processes."""
    warmup_results: list[profile_deep_models.TrialResult] = []
    for warmup_index in range(warmup_count):
        warmup_results.append(
            run_g_trial(
                name=f"{name}_warmup{warmup_index:02d}",
                baseline_paths=baseline_paths,
                candidate=candidate,
                output_directory=output_directory,
                log_directory=log_directory,
                cache_directory=cache_directory,
                emit_stage_timings=False,
            )
        )
    trial_results = [
        run_g_trial(
            name=f"{name}_trial{trial_index:02d}",
            baseline_paths=baseline_paths,
            candidate=candidate,
            output_directory=output_directory,
            log_directory=log_directory,
            cache_directory=cache_directory,
            emit_stage_timings=False,
        )
        for trial_index in range(trial_count)
    ]
    diagnostic_trials = (
        [
            run_g_trial(
                name=f"{name}_stage_diagnostic",
                baseline_paths=baseline_paths,
                candidate=candidate,
                output_directory=output_directory,
                log_directory=log_directory,
                cache_directory=cache_directory,
                emit_stage_timings=True,
                diagnostic_options={"telemetry": "profile"},
            )
        ]
        if emit_stage_timings
        else []
    )
    return aggregate_trial_results(
        name=name,
        implementation="g",
        trait_type=candidate.trait_type,
        device=candidate.device,
        warmup_count=warmup_count,
        trial_results=trial_results,
        warmup_trials=warmup_results,
        diagnostic_trials=diagnostic_trials,
        candidate=candidate,
    )


def run_repeated_regenie_trials(
    *,
    name: str,
    trait_type: str,
    regenie_executable: str,
    baseline_paths: typing.Any,
    output_directory: Path,
    log_directory: Path,
    baseline_scope: profile_deep_models.RegenieBaselineScope,
    warmup_count: int,
    trial_count: int,
) -> profile_deep_models.AggregateResult:
    """Warm and measure original REGENIE step 2."""
    profile_deep_baseline.write_regenie_baseline_extract_file(baseline_scope)
    warmup_results: list[profile_deep_models.TrialResult] = []
    for warmup_index in range(warmup_count):
        warmup_results.append(
            run_regenie_trial(
                name=f"{name}_warmup{warmup_index:02d}",
                trait_type=trait_type,
                regenie_executable=regenie_executable,
                baseline_paths=baseline_paths,
                output_directory=output_directory,
                log_directory=log_directory,
                baseline_scope=baseline_scope,
            )
        )
    trial_results = [
        run_regenie_trial(
            name=f"{name}_trial{trial_index:02d}",
            trait_type=trait_type,
            regenie_executable=regenie_executable,
            baseline_paths=baseline_paths,
            output_directory=output_directory,
            log_directory=log_directory,
            baseline_scope=baseline_scope,
        )
        for trial_index in range(trial_count)
    ]
    return aggregate_trial_results(
        name=name,
        implementation="regenie",
        trait_type=trait_type,
        device="external_cpu",
        warmup_count=warmup_count,
        trial_results=trial_results,
        warmup_trials=warmup_results,
    )


def run_candidate_tuning(
    *,
    arguments: profile_deep_models.ProfileArguments,
    baseline_paths: typing.Any,
    thread_candidates: tuple[profile_deep_models.NativeThreadCandidate, ...],
    output_directory: Path,
    cache_directory: Path,
) -> profile_deep_models.CandidateTuningResults:
    """Tune g candidates for each trait/device and return winners."""
    winners: dict[str, profile_deep_models.AggregateResult] = {}
    finalist_results_by_key: dict[str, list[profile_deep_models.AggregateResult]] = {}
    emit_stage_timings = profile_deep_config.should_emit_stage_timings(arguments)
    chunk_sizes = profile_deep_budget.parse_int_list(arguments.chunk_sizes)
    writer_thread_counts = profile_deep_budget.parse_int_list(arguments.output_writer_thread_counts)
    firth_batch_sizes = profile_deep_budget.parse_int_list(arguments.firth_batch_sizes)
    for workload_key in profile_deep_budget.parse_profile_workload_keys(arguments.workload_keys):
        candidates = profile_deep_candidates.build_step2_candidates(
            trait_type=workload_key.trait_type,
            device=workload_key.device,
            thread_candidates=thread_candidates,
            chunk_sizes=chunk_sizes,
            writer_thread_counts=writer_thread_counts,
            firth_batch_sizes=firth_batch_sizes,
        )
        if arguments.smoke:
            candidates = candidates[:1]
        initial_results = [
            run_repeated_g_trials(
                name=f"tune_{profile_deep_candidates.build_candidate_slug(candidate)}",
                baseline_paths=baseline_paths,
                candidate=candidate,
                output_directory=output_directory / "tuning_runs",
                log_directory=output_directory / "logs",
                cache_directory=cache_directory,
                warmup_count=arguments.tuning_warmups,
                trial_count=arguments.tuning_trials,
                emit_stage_timings=False,
            )
            for candidate in candidates
        ]
        successful_initial_results = [
            result for result in initial_results if result.median_wall_time_seconds is not None
        ]
        finalists = sorted(
            successful_initial_results,
            key=lambda result: typing.cast("float", result.median_wall_time_seconds),
        )[: arguments.top_finalists]
        finalist_results: list[profile_deep_models.AggregateResult] = []
        for finalist in finalists:
            candidate = profile_deep_candidates.recover_candidate_from_trial(finalist.trials[0], candidates)
            finalist_results.append(
                run_repeated_g_trials(
                    name=f"finalist_{profile_deep_candidates.build_candidate_slug(candidate)}",
                    baseline_paths=baseline_paths,
                    candidate=candidate,
                    output_directory=output_directory / "finalist_runs",
                    log_directory=output_directory / "logs",
                    cache_directory=cache_directory,
                    warmup_count=arguments.finalist_warmups,
                    trial_count=arguments.finalist_trials,
                    emit_stage_timings=emit_stage_timings,
                )
            )
        if finalist_results:
            winner = sorted(
                finalist_results,
                key=lambda result: typing.cast("float", result.median_wall_time_seconds),
            )[0]
            winners[workload_key.value] = winner
            finalist_results_by_key[workload_key.value] = finalist_results
        tuning_path = output_directory / f"tuning_{workload_key.value}.json"
        tuning_path.write_text(
            json.dumps(
                {
                    "initial_results": [dataclasses.asdict(result) for result in initial_results],
                    "finalist_results": [dataclasses.asdict(result) for result in finalist_results],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return profile_deep_models.CandidateTuningResults(winners=winners, finalist_results_by_key=finalist_results_by_key)


def run_headline_trials(
    *,
    arguments: profile_deep_models.ProfileArguments,
    baseline_paths: typing.Any,
    regenie_executable: str | None,
    regenie_baseline_scope: profile_deep_models.RegenieBaselineScope,
    winners: dict[str, profile_deep_models.AggregateResult],
    output_directory: Path,
    cache_directory: Path,
) -> list[profile_deep_models.AggregateResult]:
    """Run headline original REGENIE and winning g configurations."""
    headline_results: list[profile_deep_models.AggregateResult] = []
    emit_stage_timings = profile_deep_config.should_emit_stage_timings(arguments)
    if arguments.include_regenie_baseline:
        regenie_trait_types = profile_deep_budget.selected_regenie_baseline_trait_types(arguments)
        if regenie_executable is None:
            for trait_type in regenie_trait_types:
                headline_results.append(
                    profile_deep_execution.unsupported_aggregate_result(
                        name=f"headline_regenie_{trait_type}",
                        trait_type=trait_type,
                        device="external_cpu",
                        log_directory=output_directory / "logs",
                        notes=(
                            "Original REGENIE baseline was requested, but no executable was resolved from "
                            f"{profile_deep_config.configured_regenie_executable(arguments)!r}."
                        ),
                    )
                )
        elif regenie_baseline_scope.status == profile_deep_models.RegenieBaselineScopeStatus.UNSUPPORTED:
            for trait_type in regenie_trait_types:
                headline_results.append(
                    profile_deep_execution.unsupported_aggregate_result(
                        name=f"headline_regenie_{trait_type}",
                        trait_type=trait_type,
                        device="external_cpu",
                        log_directory=output_directory / "logs",
                        notes=regenie_baseline_scope.notes,
                    )
                )
        for trait_type in regenie_trait_types:
            if (
                regenie_executable is None
                or regenie_baseline_scope.status == profile_deep_models.RegenieBaselineScopeStatus.UNSUPPORTED
            ):
                continue
            headline_results.append(
                run_repeated_regenie_trials(
                    name=f"headline_regenie_{trait_type}",
                    trait_type=trait_type,
                    regenie_executable=regenie_executable,
                    baseline_paths=baseline_paths,
                    output_directory=output_directory / "headline_runs",
                    log_directory=output_directory / "logs",
                    baseline_scope=regenie_baseline_scope,
                    warmup_count=arguments.regenie_baseline_warmups,
                    trial_count=arguments.regenie_baseline_trials,
                )
            )
    for winner_key, winner in sorted(winners.items()):
        if not winner.trials:
            continue
        candidate = profile_deep_candidates.candidate_from_aggregate_name(winner_key, winner)
        headline_results.append(
            run_repeated_g_trials(
                name=f"headline_g_{winner_key}",
                baseline_paths=baseline_paths,
                candidate=candidate,
                output_directory=output_directory / "headline_runs",
                log_directory=output_directory / "logs",
                cache_directory=cache_directory,
                warmup_count=arguments.headline_warmups,
                trial_count=arguments.headline_trials,
                emit_stage_timings=emit_stage_timings,
            )
        )
    return headline_results


def run_logging_perturbation_profiles(
    *,
    arguments: profile_deep_models.ProfileArguments,
    baseline_paths: typing.Any,
    winners: dict[str, profile_deep_models.AggregateResult],
    output_directory: Path,
    cache_directory: Path,
) -> list[dict[str, typing.Any]]:
    """Run representative winners under telemetry/logging perturbation cases."""
    if not arguments.enable_logging_perturbation:
        return []
    perturbation_directory = output_directory / "logging_perturbation"
    perturbation_directory.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, typing.Any]] = []
    emit_stage_timings = profile_deep_config.should_emit_stage_timings(arguments)
    for winner_key, winner in sorted(winners.items()):
        if not winner.trials:
            continue
        candidate = profile_deep_candidates.candidate_from_aggregate_name(winner_key, winner)
        for perturbation_case in profile_deep_budget.build_logging_perturbation_cases(
            smoke=arguments.smoke,
        ):
            diagnostic_options = dict(perturbation_case.diagnostic_options)
            trial_result = run_g_trial(
                name=f"logging_{winner_key}_{perturbation_case.name}",
                baseline_paths=baseline_paths,
                candidate=candidate,
                output_directory=perturbation_directory,
                log_directory=output_directory / "logs",
                cache_directory=cache_directory,
                emit_stage_timings=emit_stage_timings,
                diagnostic_options=diagnostic_options,
            )
            results.append(
                {
                    "winner_key": winner_key,
                    "case": {
                        "name": perturbation_case.name,
                        "diagnostic_options": diagnostic_options,
                    },
                    "trial": dataclasses.asdict(trial_result),
                }
            )
    (perturbation_directory / "logging_perturbation.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    return results
