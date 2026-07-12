"""Deep-profile campaign budget helpers."""

from __future__ import annotations

import logging
import typing

from tooling.profile_deep import models as profile_deep_models

if typing.TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def parse_int_list(raw_values: str) -> tuple[int, ...]:
    """Parse a comma-separated list of integers."""
    parsed_values = tuple(int(value.strip()) for value in raw_values.split(",") if value.strip())
    if not parsed_values:
        message = "At least one integer is required."
        raise ValueError(message)
    return parsed_values


def parse_string_list(raw_values: str) -> tuple[str, ...]:
    """Parse a comma-separated list of strings."""
    parsed_values = tuple(value.strip() for value in raw_values.split(",") if value.strip())
    if not parsed_values:
        message = "At least one string value is required."
        raise ValueError(message)
    return parsed_values


def parse_regenie_baseline_trait_types(raw_values: str) -> tuple[str, ...]:
    """Parse and validate original REGENIE baseline trait types."""
    trait_types = parse_string_list(raw_values)
    valid_trait_types = {"quantitative", "binary"}
    invalid_trait_types = sorted(set(trait_types) - valid_trait_types)
    if invalid_trait_types:
        message = f"Unsupported REGENIE baseline trait types: {', '.join(invalid_trait_types)}"
        raise ValueError(message)
    return trait_types


def parse_profile_workload_keys(raw_values: str) -> tuple[profile_deep_models.ProfileWorkloadKey, ...]:
    """Parse and expand workload selection tokens."""
    selected_workload_keys: list[profile_deep_models.ProfileWorkloadKey] = []
    invalid_selectors: list[str] = []
    for raw_selector in parse_string_list(raw_values):
        try:
            selector = profile_deep_models.ProfileWorkloadSelector(raw_selector)
        except ValueError:
            invalid_selectors.append(raw_selector)
            continue
        if selector == profile_deep_models.ProfileWorkloadSelector.ALL:
            selected_workload_keys.extend(profile_deep_models.PROFILE_WORKLOAD_KEYS)
        elif selector == profile_deep_models.ProfileWorkloadSelector.QUANTITATIVE:
            selected_workload_keys.extend(
                workload_key
                for workload_key in profile_deep_models.PROFILE_WORKLOAD_KEYS
                if workload_key.trait_type == "quantitative"
            )
        elif selector == profile_deep_models.ProfileWorkloadSelector.BINARY:
            selected_workload_keys.extend(
                workload_key
                for workload_key in profile_deep_models.PROFILE_WORKLOAD_KEYS
                if workload_key.trait_type == "binary"
            )
        elif selector == profile_deep_models.ProfileWorkloadSelector.CPU:
            selected_workload_keys.extend(
                workload_key
                for workload_key in profile_deep_models.PROFILE_WORKLOAD_KEYS
                if workload_key.device == "cpu"
            )
        elif selector == profile_deep_models.ProfileWorkloadSelector.GPU:
            selected_workload_keys.extend(
                workload_key
                for workload_key in profile_deep_models.PROFILE_WORKLOAD_KEYS
                if workload_key.device == "gpu"
            )
        else:
            selected_workload_keys.append(profile_deep_models.ProfileWorkloadKey(selector.value))
    if invalid_selectors:
        valid_values = ", ".join(selector.value for selector in profile_deep_models.ProfileWorkloadSelector)
        message = (
            f"Unsupported deep-profile workload selectors: {', '.join(invalid_selectors)}. "
            f"Valid selectors: {valid_values}."
        )
        raise ValueError(message)
    deduplicated_workload_keys = tuple(dict.fromkeys(selected_workload_keys))
    if not deduplicated_workload_keys:
        message = "At least one deep-profile workload key is required."
        raise ValueError(message)
    return deduplicated_workload_keys


def selected_regenie_baseline_trait_types(
    arguments: profile_deep_models.ProfileArguments,
) -> tuple[str, ...]:
    """Return REGENIE baseline traits that match the selected workload traits."""
    requested_trait_types = parse_regenie_baseline_trait_types(arguments.regenie_baseline_trait_types)
    selected_trait_types = {
        workload_key.trait_type for workload_key in parse_profile_workload_keys(arguments.workload_keys)
    }
    return tuple(trait_type for trait_type in requested_trait_types if trait_type in selected_trait_types)


def build_logging_perturbation_cases(
    *,
    output_directory: Path,
    smoke: bool,
) -> tuple[profile_deep_models.LoggingPerturbationCase, ...]:
    """Build perturbations supported by the current diagnostics contract."""
    del output_directory
    cases = (
        profile_deep_models.LoggingPerturbationCase(
            name="telemetry_off",
            diagnostic_options={"telemetry": "off"},
        ),
        profile_deep_models.LoggingPerturbationCase(
            name="telemetry_progress",
            diagnostic_options={"telemetry": "progress"},
        ),
        profile_deep_models.LoggingPerturbationCase(
            name="telemetry_profile",
            diagnostic_options={"telemetry": "profile"},
        ),
    )
    if smoke:
        return cases[:2]
    return cases


def build_campaign_budget_section(
    *,
    section_name: profile_deep_models.CampaignBudgetSectionName,
    candidate_count: int,
    subprocess_run_count: int,
    major_profiler_run_count: int = 0,
    notes: str,
) -> profile_deep_models.CampaignBudgetSection:
    """Build one campaign budget section."""
    return profile_deep_models.CampaignBudgetSection(
        name=section_name.value,
        display_name=profile_deep_models.CAMPAIGN_BUDGET_SECTION_DISPLAY_NAMES[section_name],
        candidate_count=candidate_count,
        subprocess_run_count=subprocess_run_count,
        major_profiler_run_count=major_profiler_run_count,
        notes=notes,
    )


def count_step2_tuning_candidates(
    *,
    workload_key: profile_deep_models.ProfileWorkloadKey,
    thread_candidate_count: int,
    chunk_sizes: tuple[int, ...],
    writer_thread_counts: tuple[int, ...],
    firth_batch_sizes: tuple[int, ...],
    smoke: bool,
) -> int:
    """Count combinations of settings exposed by the current application."""
    candidate_count = thread_candidate_count * len(chunk_sizes) * len(writer_thread_counts)
    if workload_key.trait_type == "binary":
        candidate_count *= len(firth_batch_sizes)
    if smoke:
        return min(candidate_count, 1)
    return candidate_count


def count_enabled_deep_profiler_modes(arguments: profile_deep_models.ProfileArguments) -> int:
    """Count profiler subprocess modes run for each selected winner."""
    mode_count = 0
    if arguments.enable_jax_trace or arguments.enable_jax_memory_profile:
        mode_count += 1
    enabled_modes = (
        arguments.enable_python_cprofile,
        arguments.enable_py_spy,
        arguments.enable_scalene,
        arguments.enable_memray,
        arguments.enable_linux_perf,
        arguments.enable_nsight_systems,
        arguments.enable_nsight_compute,
    )
    return mode_count + sum(1 for enabled in enabled_modes if enabled)


def campaign_budget_is_over_limit(campaign_budget: profile_deep_models.CampaignBudget) -> bool:
    """Return whether a campaign exceeds either configured budget."""
    return campaign_budget.over_subprocess_budget or campaign_budget.over_major_profiler_budget


def build_campaign_budget(
    *,
    arguments: profile_deep_models.ProfileArguments,
    output_directory: Path,
) -> profile_deep_models.CampaignBudget:
    """Estimate campaign section counts before executing workloads."""
    workload_keys = parse_profile_workload_keys(arguments.workload_keys)
    chunk_sizes = parse_int_list(arguments.chunk_sizes)
    writer_thread_counts = parse_int_list(arguments.output_writer_thread_counts)
    firth_batch_sizes = parse_int_list(arguments.firth_batch_sizes)
    rayon_thread_counts = parse_int_list(arguments.rayon_thread_counts)
    thread_candidate_count = len(rayon_thread_counts)
    tuning_candidate_counts = [
        count_step2_tuning_candidates(
            workload_key=workload_key,
            thread_candidate_count=thread_candidate_count,
            chunk_sizes=chunk_sizes,
            writer_thread_counts=writer_thread_counts,
            firth_batch_sizes=firth_batch_sizes,
            smoke=arguments.smoke,
        )
        for workload_key in workload_keys
    ]
    tuning_candidate_count = sum(tuning_candidate_counts)
    finalist_candidate_counts = [
        min(arguments.top_finalists, tuning_candidate_count_for_workload)
        for tuning_candidate_count_for_workload in tuning_candidate_counts
    ]
    finalist_candidate_count = sum(finalist_candidate_counts)
    expected_winner_count = sum(
        1
        for finalist_count in finalist_candidate_counts
        if finalist_count > 0 and arguments.tuning_trials > 0 and arguments.finalist_trials > 0
    )
    regenie_baseline_trait_count = 0
    if arguments.include_regenie_baseline:
        regenie_baseline_trait_count = len(selected_regenie_baseline_trait_types(arguments))
    g_headline_run_count = expected_winner_count * (arguments.headline_warmups + arguments.headline_trials)
    regenie_headline_run_count = regenie_baseline_trait_count * (
        arguments.regenie_baseline_warmups + arguments.regenie_baseline_trials
    )
    deep_profiler_mode_count = 0 if arguments.skip_deep_profiles else count_enabled_deep_profiler_modes(arguments)
    deep_profiler_run_count = expected_winner_count * deep_profiler_mode_count
    logging_case_count = 0
    if arguments.enable_logging_perturbation:
        logging_case_count = len(
            build_logging_perturbation_cases(output_directory=output_directory, smoke=arguments.smoke)
        )
    logging_run_count = expected_winner_count * logging_case_count
    rust_benchmark_count = 0
    if arguments.enable_rust_criterion and not arguments.skip_deep_profiles:
        rust_benchmark_count = len(parse_string_list(arguments.rust_benchmarks))
    sections = (
        build_campaign_budget_section(
            section_name=profile_deep_models.CampaignBudgetSectionName.THREAD_CANDIDATES,
            candidate_count=thread_candidate_count,
            subprocess_run_count=0,
            notes="Static Rayon worker-count choices; Criterion owns internal reader profiling.",
        ),
        build_campaign_budget_section(
            section_name=profile_deep_models.CampaignBudgetSectionName.TUNING,
            candidate_count=tuning_candidate_count,
            subprocess_run_count=tuning_candidate_count * (arguments.tuning_warmups + arguments.tuning_trials),
            notes=(
                f"{len(workload_keys)} selected workload(s), "
                f"{thread_candidate_count} thread candidate(s), "
                f"{arguments.tuning_warmups} warmup(s), and {arguments.tuning_trials} measured trial(s)."
            ),
        ),
        build_campaign_budget_section(
            section_name=profile_deep_models.CampaignBudgetSectionName.FINALISTS,
            candidate_count=finalist_candidate_count,
            subprocess_run_count=finalist_candidate_count * (arguments.finalist_warmups + arguments.finalist_trials),
            notes=(
                f"Up to {arguments.top_finalists} finalist(s) per selected workload, "
                f"{arguments.finalist_warmups} warmup(s), and {arguments.finalist_trials} measured trial(s)."
            ),
        ),
        build_campaign_budget_section(
            section_name=profile_deep_models.CampaignBudgetSectionName.HEADLINE_TRIALS,
            candidate_count=expected_winner_count + regenie_baseline_trait_count,
            subprocess_run_count=g_headline_run_count + regenie_headline_run_count,
            notes=(
                f"{expected_winner_count} expected g winner(s) and "
                f"{regenie_baseline_trait_count} selected REGENIE baseline trait(s)."
            ),
        ),
        build_campaign_budget_section(
            section_name=profile_deep_models.CampaignBudgetSectionName.DEEP_PROFILERS,
            candidate_count=deep_profiler_run_count,
            subprocess_run_count=deep_profiler_run_count,
            major_profiler_run_count=deep_profiler_run_count,
            notes=(
                "Skipped by tool.skip_deep_profiles=true."
                if arguments.skip_deep_profiles
                else f"{deep_profiler_mode_count} profiler mode(s) per expected g winner."
            ),
        ),
        build_campaign_budget_section(
            section_name=profile_deep_models.CampaignBudgetSectionName.LOGGING_PERTURBATION,
            candidate_count=logging_run_count,
            subprocess_run_count=logging_run_count,
            notes=(
                "Disabled by tool.enable_logging_perturbation=false."
                if not arguments.enable_logging_perturbation
                else f"{logging_case_count} logging case(s) per expected g winner."
            ),
        ),
        build_campaign_budget_section(
            section_name=profile_deep_models.CampaignBudgetSectionName.RUST_CRITERION,
            candidate_count=rust_benchmark_count,
            subprocess_run_count=rust_benchmark_count,
            major_profiler_run_count=rust_benchmark_count,
            notes=(
                "Skipped because Rust Criterion is disabled or tool.skip_deep_profiles=true."
                if rust_benchmark_count == 0
                else "Each configured Criterion benchmark is one cargo bench subprocess."
            ),
        ),
    )
    total_candidate_count = sum(section.candidate_count for section in sections)
    total_subprocess_run_count = sum(section.subprocess_run_count for section in sections)
    total_major_profiler_run_count = sum(section.major_profiler_run_count for section in sections)
    over_subprocess_budget = (
        arguments.max_subprocess_runs is not None and total_subprocess_run_count > arguments.max_subprocess_runs
    )
    over_major_profiler_budget = (
        arguments.max_major_profiler_runs is not None
        and total_major_profiler_run_count > arguments.max_major_profiler_runs
    )
    guidance = (
        "Run a dry run first and inspect profile_plan.md for the section counts.",
        "Reduce tool.workload_keys, tool.top_finalists, trial counts, Firth batch sizes, writer counts, "
        "or Rayon thread counts to fit the budget.",
        "For an intentional huge campaign, pass tool.allow_over_budget=true and keep the run on an appropriate "
        "SLURM node.",
    )
    return profile_deep_models.CampaignBudget(
        workload_keys=tuple(workload_key.value for workload_key in workload_keys),
        max_subprocess_runs=arguments.max_subprocess_runs,
        max_major_profiler_runs=arguments.max_major_profiler_runs,
        total_candidate_count=total_candidate_count,
        total_subprocess_run_count=total_subprocess_run_count,
        total_major_profiler_run_count=total_major_profiler_run_count,
        over_subprocess_budget=over_subprocess_budget,
        over_major_profiler_budget=over_major_profiler_budget,
        sections=sections,
        guidance=guidance,
    )


def log_campaign_budget(campaign_budget: profile_deep_models.CampaignBudget) -> None:
    """Log section-level campaign budget estimates."""
    logger.info(
        "Estimated campaign budget: candidates=%s subprocess_runs=%s major_profiler_runs=%s",
        campaign_budget.total_candidate_count,
        campaign_budget.total_subprocess_run_count,
        campaign_budget.total_major_profiler_run_count,
    )
    for section in campaign_budget.sections:
        logger.info(
            "Budget section %s: candidates=%s subprocess_runs=%s major_profiler_runs=%s",
            section.display_name,
            section.candidate_count,
            section.subprocess_run_count,
            section.major_profiler_run_count,
        )


def enforce_campaign_budget(
    arguments: profile_deep_models.ProfileArguments,
    campaign_budget: profile_deep_models.CampaignBudget,
) -> None:
    """Fail early when a non-dry-run campaign exceeds the configured budget."""
    if arguments.allow_over_budget or not campaign_budget_is_over_limit(campaign_budget):
        return
    budget_messages = [
        "Deep profile campaign exceeds the configured budget.",
        (
            f"Estimated subprocess runs: {campaign_budget.total_subprocess_run_count} "
            f"(limit: {campaign_budget.max_subprocess_runs})."
        ),
        (
            f"Estimated major profiler runs: {campaign_budget.total_major_profiler_run_count} "
            f"(limit: {campaign_budget.max_major_profiler_runs})."
        ),
        "Section counts:",
    ]
    for section in campaign_budget.sections:
        budget_messages.append(
            f"- {section.display_name}: candidates={section.candidate_count}, "
            f"subprocess_runs={section.subprocess_run_count}, "
            f"major_profiler_runs={section.major_profiler_run_count}"
        )
    budget_messages.extend(campaign_budget.guidance)
    raise ValueError("\n".join(budget_messages))
