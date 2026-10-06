"""Deep-profile campaign orchestration."""

from __future__ import annotations

import dataclasses
import json
import logging
import typing

from tooling.benchmark import benchmark as baseline_benchmark
from tooling.common import artifact_format as tooling_artifact_format
from tooling.common import logging as tooling_logging
from tooling.profile_deep import artifacts as profile_deep_artifacts
from tooling.profile_deep import baseline as profile_deep_baseline
from tooling.profile_deep import budget as profile_deep_budget
from tooling.profile_deep import candidates as profile_deep_candidates
from tooling.profile_deep import config as profile_deep_config
from tooling.profile_deep import diagnostics as profile_deep_diagnostics
from tooling.profile_deep import environment as profile_deep_environment
from tooling.profile_deep import jax_cache as profile_deep_jax_cache
from tooling.profile_deep import models as profile_deep_models
from tooling.profile_deep import planning as profile_deep_planning
from tooling.profile_deep import profilers as profile_deep_profilers
from tooling.profile_deep import reports as profile_deep_reports
from tooling.profile_deep import tools as profile_deep_tools
from tooling.profile_deep import trials as profile_deep_trials

if typing.TYPE_CHECKING:
    import omegaconf

logger = logging.getLogger(__name__)


def run_tool(arguments: profile_deep_models.ProfileArguments, hydra_config: omegaconf.DictConfig | None = None) -> None:
    """Run the landau deep profiling campaign."""
    arguments = profile_deep_config.apply_smoke_overrides(arguments)
    output_directory = profile_deep_config.build_output_directory(arguments)
    output_directory.mkdir(parents=True, exist_ok=True)
    log_directory = output_directory / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    cache_directory = output_directory / "jax_cache"
    cache_directory.mkdir(parents=True, exist_ok=True)
    tooling_logging.configure_tool_logging(output_directory / "tooling.log")

    logger.info("Starting %s deep profile campaign", arguments.chromosome_label)
    logger.info("Writing profile artifacts under %s", output_directory)
    baseline_paths = profile_deep_baseline.build_baseline_paths(arguments)
    profiler_tool_status = profile_deep_tools.build_profiler_tool_status(arguments)
    regenie_baseline_scope = profile_deep_baseline.build_regenie_baseline_scope(
        arguments=arguments,
        baseline_paths=baseline_paths,
        output_directory=output_directory,
    )
    campaign_budget = profile_deep_budget.build_campaign_budget(arguments=arguments)
    profile_deep_budget.log_campaign_budget(campaign_budget)
    if arguments.dry_run:
        profile_plan = profile_deep_planning.build_profile_plan(
            arguments=arguments,
            baseline_paths=baseline_paths,
            output_directory=output_directory,
            campaign_budget=campaign_budget,
        )
        profile_deep_planning.write_profile_plan(profile_plan, output_directory)
        profile_deep_artifacts.write_artifact_manifest(
            output_directory=output_directory,
            profiler_tool_status=profiler_tool_status,
            profile_plan=profile_plan,
        )
        profile_deep_artifacts.write_standard_profile_artifacts(
            arguments=arguments,
            output_directory=output_directory,
            profiler_tool_status=profiler_tool_status,
            status=tooling_artifact_format.ToolArtifactStatus.DRY_RUN,
            profile_plan=profile_plan,
            summary_markdown=(output_directory / "profile_plan.md").read_text(encoding="utf-8"),
            hydra_config=hydra_config,
        )
        logger.info("Wrote dry-run profile plan under %s", output_directory)
        return
    if profile_deep_budget.campaign_budget_is_over_limit(campaign_budget) and not arguments.allow_over_budget:
        profile_plan = profile_deep_planning.build_profile_plan(
            arguments=arguments,
            baseline_paths=baseline_paths,
            output_directory=output_directory,
            campaign_budget=campaign_budget,
        )
        profile_deep_planning.write_profile_plan(profile_plan, output_directory)
        profile_deep_artifacts.write_artifact_manifest(
            output_directory=output_directory,
            profiler_tool_status=profiler_tool_status,
            profile_plan=profile_plan,
        )
        profile_deep_artifacts.write_standard_profile_artifacts(
            arguments=arguments,
            output_directory=output_directory,
            profiler_tool_status=profiler_tool_status,
            status=tooling_artifact_format.ToolArtifactStatus.INVALID,
            status_reason="Campaign budget exceeds configured limits.",
            profile_plan=profile_plan,
            summary_markdown=(output_directory / "profile_plan.md").read_text(encoding="utf-8"),
            hydra_config=hydra_config,
        )
    profile_deep_budget.enforce_campaign_budget(arguments, campaign_budget)
    logger.info("Validating profile inputs")
    baseline_benchmark.validate_input_files(baseline_paths)
    prediction_list_paths = [
        baseline_paths.regenie_prediction_list_path,
        baseline_paths.regenie_qt_prediction_list_path,
    ]
    missing_prediction_list_paths = [path for path in prediction_list_paths if path is not None and not path.exists()]
    regenie_executable: str | None = None
    setup_results: list[profile_deep_models.TrialResult] = []
    if arguments.include_regenie_baseline:
        regenie_executable = profile_deep_environment.resolve_available_regenie_executable(arguments)
        if regenie_executable is not None:
            logger.info("Ensuring REGENIE step 1 prediction lists")
            setup_results = profile_deep_baseline.ensure_prediction_lists(
                baseline_paths=baseline_paths,
                regenie_executable=regenie_executable,
                log_directory=log_directory,
            )
        elif missing_prediction_list_paths:
            formatted_paths = "\n".join(str(path) for path in missing_prediction_list_paths)
            message = (
                "Step 1 prediction lists are required before g runs and REGENIE setup cannot run because "
                f"{profile_deep_config.configured_regenie_executable(arguments)!r} is unavailable:\n{formatted_paths}"
            )
            raise FileNotFoundError(message)
        else:
            logger.warning("Original REGENIE baseline requested, but executable is unavailable")
    elif missing_prediction_list_paths:
        formatted_paths = "\n".join(str(path) for path in missing_prediction_list_paths)
        message = f"Step 1 prediction lists are required when REGENIE setup is disabled:\n{formatted_paths}"
        raise FileNotFoundError(message)
    else:
        logger.info("Using existing REGENIE step 1 prediction lists")
    logger.info("Collecting preflight metadata")
    preflight_metadata = profile_deep_environment.collect_environment_metadata(baseline_paths, regenie_executable)
    (output_directory / "preflight.json").write_text(json.dumps(preflight_metadata, indent=2) + "\n", encoding="utf-8")

    logger.info("Recording native thread candidates")
    thread_candidates = profile_deep_candidates.build_native_thread_candidates(
        arguments=arguments,
        output_directory=output_directory,
    )
    logger.info("Running candidate tuning")
    tuning_results = profile_deep_trials.run_candidate_tuning(
        arguments=arguments,
        baseline_paths=baseline_paths,
        thread_candidates=thread_candidates,
        output_directory=output_directory,
        cache_directory=cache_directory,
    )
    winners = tuning_results.winners
    logger.info("Running headline trials")
    headline_results = profile_deep_trials.run_headline_trials(
        arguments=arguments,
        baseline_paths=baseline_paths,
        regenie_executable=regenie_executable,
        regenie_baseline_scope=regenie_baseline_scope,
        winners=winners,
        output_directory=output_directory,
        cache_directory=cache_directory,
    )
    deep_profile_results: dict[str, typing.Any] = {}
    if not arguments.skip_deep_profiles:
        logger.info("Running full profiler bundle")
        deep_profile_results = profile_deep_profilers.run_deep_profiles(
            arguments=arguments,
            baseline_paths=baseline_paths,
            winners=winners,
            output_directory=output_directory,
            cache_directory=cache_directory,
        )
    else:
        logger.info("Skipping full profiler bundle")
    logger.info("Running logging perturbation profiles")
    logging_perturbation_results = profile_deep_trials.run_logging_perturbation_profiles(
        arguments=arguments,
        baseline_paths=baseline_paths,
        winners=winners,
        output_directory=output_directory,
        cache_directory=cache_directory,
    )
    comparisons = profile_deep_reports.build_runtime_comparisons(headline_results)
    comparison_notes = profile_deep_reports.build_runtime_comparison_notes(headline_results)
    jax_cache_diagnostics = profile_deep_jax_cache.collect_jax_cache_diagnostics(headline_results)
    stage_totals = profile_deep_diagnostics.collect_stage_totals(headline_results)
    binary_correction_diagnostics = profile_deep_diagnostics.build_binary_correction_diagnostics(
        headline_results=headline_results,
        finalist_results_by_key=tuning_results.finalist_results_by_key,
        stage_timing_mode=arguments.stage_timing_mode,
    )
    summary_payload = {
        "stage_timing_mode": arguments.stage_timing_mode.value,
        "preflight": preflight_metadata,
        "campaign_budget": dataclasses.asdict(campaign_budget),
        "setup_results": [dataclasses.asdict(result) for result in setup_results],
        "thread_candidates": [dataclasses.asdict(candidate) for candidate in thread_candidates],
        "winners": {key: dataclasses.asdict(value) for key, value in winners.items()},
        "headline_results": [dataclasses.asdict(result) for result in headline_results],
        "regenie_baseline_scope": profile_deep_artifacts.serialize_regenie_baseline_scope(regenie_baseline_scope),
        "comparisons": comparisons,
        "runtime_comparison_notes": dataclasses.asdict(comparison_notes),
        "jax_cache_diagnostics": jax_cache_diagnostics,
        "stage_totals": stage_totals,
        "binary_correction_diagnostics": binary_correction_diagnostics,
        "deep_profiles": deep_profile_results,
        "logging_perturbation_results": logging_perturbation_results,
    }
    (output_directory / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    summary_markdown = profile_deep_reports.build_summary_markdown(
        aggregate_results=headline_results,
        comparisons=comparisons,
        comparison_notes=comparison_notes,
        regenie_baseline_scope=regenie_baseline_scope,
        stage_totals=stage_totals,
        logging_perturbation_results=logging_perturbation_results,
        binary_correction_diagnostics=binary_correction_diagnostics,
    )
    (output_directory / "summary.md").write_text(summary_markdown, encoding="utf-8")
    profile_deep_artifacts.write_artifact_manifest(
        output_directory=output_directory,
        profiler_tool_status=profiler_tool_status,
        summary_payload=summary_payload,
    )
    profile_deep_artifacts.write_standard_profile_artifacts(
        arguments=arguments,
        output_directory=output_directory,
        profiler_tool_status=profiler_tool_status,
        status=profile_deep_artifacts.profile_summary_artifact_status(summary_payload),
        summary_payload=summary_payload,
        summary_markdown=summary_markdown,
        hydra_config=hydra_config,
    )
    logger.info("Wrote deep profile artifacts under %s", output_directory)
