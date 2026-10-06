"""Dry-run plans for bounded deep-profile campaigns."""

from __future__ import annotations

import dataclasses
import json
import shlex
import typing

from tooling.profile_deep import artifacts as profile_deep_artifacts
from tooling.profile_deep import baseline as profile_deep_baseline
from tooling.profile_deep import budget as profile_deep_budget
from tooling.profile_deep import config as profile_deep_config
from tooling.profile_deep import models as profile_deep_models
from tooling.profile_deep import tools as profile_deep_tools

if typing.TYPE_CHECKING:
    from pathlib import Path

    from tooling.benchmark import benchmark as baseline_benchmark


def required_profile_input_paths(baseline_paths: baseline_benchmark.BaselinePaths) -> list[Path]:
    """Return input and prediction paths used by real profile runs."""
    return [
        baseline_paths.bed_prefix.with_suffix(".bed"),
        baseline_paths.bed_prefix.with_suffix(".bim"),
        baseline_paths.bed_prefix.with_suffix(".fam"),
        baseline_paths.bgen_path,
        baseline_paths.sample_path,
        baseline_paths.continuous_phenotype_path,
        baseline_paths.binary_phenotype_path,
        baseline_paths.covariate_path,
        baseline_paths.regenie_prediction_list_path,
        typing.cast("Path", baseline_paths.regenie_qt_prediction_list_path),
    ]


def build_profile_plan(
    *,
    arguments: profile_deep_models.ProfileArguments,
    baseline_paths: baseline_benchmark.BaselinePaths,
    output_directory: Path,
    campaign_budget: profile_deep_models.CampaignBudget,
) -> profile_deep_models.ProfilePlan:
    """Build a dry-run plan for the full profile campaign."""
    profiler_modes = {
        "regenie_baseline": arguments.include_regenie_baseline,
        "jax_trace": arguments.enable_jax_trace,
        "jax_memory_profile": arguments.enable_jax_memory_profile,
        "python_cprofile": arguments.enable_python_cprofile,
        "py_spy": arguments.enable_py_spy,
        "scalene": arguments.enable_scalene,
        "memray": arguments.enable_memray,
        "linux_perf": arguments.enable_linux_perf,
        "nsight_systems": arguments.enable_nsight_systems,
        "nsight_compute": arguments.enable_nsight_compute,
        "logging_perturbation": arguments.enable_logging_perturbation,
    }
    profiler_tools = profile_deep_tools.serialize_profiler_tool_status(
        profile_deep_tools.build_profiler_tool_status(arguments)
    )
    logging_perturbation_cases = []
    if arguments.enable_logging_perturbation:
        logging_perturbation_cases = [
            dataclasses.asdict(perturbation_case)
            for perturbation_case in profile_deep_budget.build_logging_perturbation_cases(
                smoke=arguments.smoke,
            )
        ]
    regenie_baseline_scope = None
    regenie_baseline_commands: list[dict[str, object]] = []
    if arguments.include_regenie_baseline:
        scope = profile_deep_baseline.build_regenie_baseline_scope(
            arguments=arguments,
            baseline_paths=baseline_paths,
            output_directory=output_directory,
        )
        regenie_baseline_scope = profile_deep_artifacts.serialize_regenie_baseline_scope(scope)
        if scope.status != profile_deep_models.RegenieBaselineScopeStatus.UNSUPPORTED:
            regenie_executable = profile_deep_config.configured_regenie_executable(arguments)
            for trait_type in profile_deep_budget.selected_regenie_baseline_trait_types(arguments):
                command_arguments = profile_deep_baseline.build_regenie_step2_command(
                    trait_type=trait_type,
                    regenie_executable=regenie_executable,
                    baseline_paths=baseline_paths,
                    output_prefix=output_directory / "headline_runs" / f"headline_regenie_{trait_type}_trial00",
                    baseline_scope=scope,
                )
                regenie_baseline_commands.append(
                    profile_deep_artifacts.build_command_manifest(
                        command_name=f"headline_regenie_{trait_type}_trial00",
                        status="planned",
                        command_arguments=command_arguments,
                    )
                )
    notes = [
        "Dry run only: no workloads, profilers, or setup commands were executed.",
        "Real runs generate summary.json, summary.md, preflight.json, subprocess logs, stage timings, "
        "deep_profiles artifacts, logging perturbation results, and artifact_manifest.json.",
    ]
    if arguments.skip_deep_profiles:
        notes.append("Deep profiler captures are disabled by tool.skip_deep_profiles=true.")
    if not profile_deep_config.should_emit_stage_timings(arguments):
        notes.append("Exact stage timing diagnostics are disabled by telemetry.stage_timing_mode=off.")
    if not arguments.include_regenie_baseline:
        notes.append("Original REGENIE headline trials are disabled by tool.include_regenie_baseline=false.")
    elif regenie_baseline_scope is not None:
        notes.append(str(regenie_baseline_scope["notes"]))
    return profile_deep_models.ProfilePlan(
        chromosome_label=arguments.chromosome_label,
        output_directory=str(output_directory),
        required_inputs=[str(path) for path in required_profile_input_paths(baseline_paths)],
        workload_keys=list(campaign_budget.workload_keys),
        campaign_budget=campaign_budget,
        profiler_modes=profiler_modes,
        profiler_tools=profiler_tools,
        logging_perturbation_cases=logging_perturbation_cases,
        regenie_baseline_scope=regenie_baseline_scope,
        regenie_baseline_commands=regenie_baseline_commands,
        notes=notes,
    )


def write_profile_plan(plan: profile_deep_models.ProfilePlan, output_directory: Path) -> None:
    """Persist dry-run profile plan artifacts."""
    (output_directory / "profile_plan.json").write_text(
        json.dumps(dataclasses.asdict(plan), indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Full App Profile Plan",
        "",
        f"- Chromosome: `{plan.chromosome_label}`",
        f"- Output directory: `{plan.output_directory}`",
        f"- Workloads: `{', '.join(plan.workload_keys)}`",
        "",
        "## Campaign Budget",
        "",
        f"- Total candidates/cases: `{plan.campaign_budget.total_candidate_count}`",
        f"- Estimated subprocess runs: `{plan.campaign_budget.total_subprocess_run_count}`",
        f"- Estimated major profiler runs: `{plan.campaign_budget.total_major_profiler_run_count}`",
        f"- Max subprocess runs: `{plan.campaign_budget.max_subprocess_runs}`",
        f"- Max major profiler runs: `{plan.campaign_budget.max_major_profiler_runs}`",
        f"- Over subprocess budget: `{str(plan.campaign_budget.over_subprocess_budget).lower()}`",
        f"- Over major profiler budget: `{str(plan.campaign_budget.over_major_profiler_budget).lower()}`",
        "",
        "| section | candidates/cases | subprocess runs | major profiler runs | notes |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for section in plan.campaign_budget.sections:
        lines.append(
            "| "
            f"{section.display_name} | "
            f"{section.candidate_count} | "
            f"{section.subprocess_run_count} | "
            f"{section.major_profiler_run_count} | "
            f"{section.notes} |"
        )
    lines.extend(["", "### Budget Guidance", ""])
    for guidance_item in plan.campaign_budget.guidance:
        lines.append(f"- {guidance_item}")
    lines.extend(["", "## Profiler Modes", ""])
    for mode_name, enabled in plan.profiler_modes.items():
        lines.append(f"- `{mode_name}`: `{str(enabled).lower()}`")
    lines.extend(["", "## Profiler Tool Availability", ""])
    for tool_name, tool_status in plan.profiler_tools.items():
        available = str(tool_status["available"]).lower()
        enabled = str(tool_status["enabled"]).lower()
        notes = str(tool_status["notes"])
        lines.append(f"- `{tool_name}`: enabled=`{enabled}`, available=`{available}`; {notes}")
    lines.extend(["", "## Logging Perturbation Cases", ""])
    if plan.logging_perturbation_cases:
        for perturbation_case in plan.logging_perturbation_cases:
            lines.append(f"- `{perturbation_case['name']}`: `{perturbation_case['diagnostic_options']}`")
    else:
        lines.append("- Logging perturbation profiling is disabled.")
    lines.extend(["", "## Inputs And Step 1 Prediction Lists", ""])
    for input_path in plan.required_inputs:
        lines.append(f"- `{input_path}`")
    lines.extend(["", "## REGENIE Baseline Scope", ""])
    if plan.regenie_baseline_scope is None:
        lines.append("- Original REGENIE baseline profiling is disabled.")
    else:
        for key, value in plan.regenie_baseline_scope.items():
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## REGENIE Baseline Commands", ""])
    if plan.regenie_baseline_commands:
        for command_manifest in plan.regenie_baseline_commands:
            command_arguments = typing.cast("list[str]", command_manifest["command_arguments"])
            lines.append(f"- `{command_manifest['name']}`: `{shlex.join(command_arguments)}`")
    else:
        lines.append("- No REGENIE baseline commands are planned.")
    lines.extend(["", "## Notes", ""])
    for note in plan.notes:
        lines.append(f"- {note}")
    (output_directory / "profile_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
