"""Versioned artifact bundles for deep-profile campaigns."""

from __future__ import annotations

import typing
from datetime import UTC, datetime
from pathlib import Path

from tooling.common import artifact_format as tooling_artifact_format
from tooling.common import paths as tooling_paths
from tooling.common import reports as tooling_reports
from tooling.profile_deep import config as profile_deep_config
from tooling.profile_deep import environment as profile_deep_environment
from tooling.profile_deep import models as profile_deep_models
from tooling.profile_deep import tools as profile_deep_tools

if typing.TYPE_CHECKING:
    import omegaconf

REPOSITORY_ROOT = tooling_paths.find_repository_root(Path(__file__))


ARTIFACT_MANIFEST_SCHEMA_VERSION = 2


ARTIFACT_MANIFEST_CONTRACT = tooling_reports.VersionedReportContract(
    schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
    required_fields=(
        "generated_at",
        "output_directory",
        "profiler_tools",
        "input_files",
        "regenie_baseline_scope",
        "regenie_baseline_commands",
        "artifact_paths",
        "profiler_runs",
        "skipped_profiles",
    ),
    optional_fields=(),
    schema_field_name="schema_version",
    reject_unknown_fields=True,
)


def manifest_path_value(*, output_directory: Path, path_value: typing.Any) -> str | None:
    """Return a manifest path relative to the campaign directory when possible."""
    if path_value is None:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        try:
            return str(path.relative_to(output_directory))
        except ValueError:
            return str(path)
    return str(path)


def collect_profiler_run_manifest_entries(
    *,
    output_directory: Path,
    sampling_profiles: list[dict[str, typing.Any]],
) -> list[dict[str, str | None]]:
    """Collect per-profiler artifact and application output paths for the manifest."""
    profiler_runs: list[dict[str, str | None]] = []
    for profile in sampling_profiles:
        profiler_artifact_path = manifest_path_value(
            output_directory=output_directory,
            path_value=profile.get("profiler_artifact_path"),
        )
        application_output_prefix = manifest_path_value(
            output_directory=output_directory,
            path_value=profile.get("application_output_prefix"),
        )
        application_output_run_directory = manifest_path_value(
            output_directory=output_directory,
            path_value=profile.get("application_output_run_directory"),
        )
        stage_timing_path = manifest_path_value(
            output_directory=output_directory,
            path_value=profile.get("stage_timing_path"),
        )
        profile_summary_path = manifest_path_value(
            output_directory=output_directory,
            path_value=profile.get("profile_summary_path"),
        )
        if (
            profiler_artifact_path is None
            and application_output_prefix is None
            and application_output_run_directory is None
            and stage_timing_path is None
            and profile_summary_path is None
        ):
            continue
        profiler_runs.append(
            {
                "name": str(profile.get("name", "")),
                "implementation": str(profile.get("implementation", "")),
                "status": str(profile.get("status", "")),
                "profiler_artifact_path": profiler_artifact_path,
                "application_output_prefix": application_output_prefix,
                "application_output_run_directory": application_output_run_directory,
                "stage_timing_path": stage_timing_path,
                "profile_summary_path": profile_summary_path,
            }
        )
    return profiler_runs


def serialize_regenie_baseline_scope(scope: profile_deep_models.RegenieBaselineScope) -> dict[str, object]:
    """Serialize baseline scope without embedding long extract lists."""
    return {
        "status": scope.status.value,
        "variant_limit": scope.variant_limit,
        "extract_path": str(scope.extract_path) if scope.extract_path is not None else None,
        "metadata_path": str(scope.metadata_path) if scope.metadata_path is not None else None,
        "selected_variant_count": scope.selected_variant_count,
        "notes": scope.notes,
    }


def command_input_paths(command_arguments: list[str]) -> list[str]:
    """Extract input file paths from a REGENIE command line."""
    input_flags = {"--bgen", "--sample", "--phenoFile", "--covarFile", "--pred", "--extract"}
    input_paths: list[str] = []
    argument_index = 0
    while argument_index < len(command_arguments):
        argument = command_arguments[argument_index]
        if argument in input_flags and argument_index + 1 < len(command_arguments):
            input_paths.append(command_arguments[argument_index + 1])
            argument_index += 2
            continue
        if argument == "--bed" and argument_index + 1 < len(command_arguments):
            bed_prefix = Path(command_arguments[argument_index + 1])
            input_paths.extend(str(bed_prefix.with_suffix(suffix)) for suffix in (".bed", ".bim", ".fam"))
            argument_index += 2
            continue
        argument_index += 1
    return sorted(set(input_paths))


def build_command_manifest(command_name: str, status: str, command_arguments: list[str]) -> dict[str, object]:
    """Build manifest metadata for one baseline command."""
    executable_name = command_arguments[0] if command_arguments else None
    return {
        "name": command_name,
        "status": status,
        "binary": profile_deep_environment.resolved_binary_path(executable_name),
        "command_arguments": command_arguments,
        "input_files": command_input_paths(command_arguments),
    }


def collect_summary_baseline_commands(summary_payload: dict[str, typing.Any] | None) -> list[dict[str, object]]:
    """Collect actual baseline commands from a run summary payload."""
    if summary_payload is None:
        return []
    command_manifests: list[dict[str, object]] = []
    result_groups = [
        typing.cast("list[dict[str, typing.Any]]", summary_payload.get("setup_results", [])),
        typing.cast("list[dict[str, typing.Any]]", summary_payload.get("headline_results", [])),
    ]
    for result_group in result_groups:
        for result_payload in result_group:
            trial_payloads = typing.cast("list[dict[str, typing.Any]]", result_payload.get("trials", []))
            if not trial_payloads:
                trial_payloads = [result_payload]
            for trial_payload in trial_payloads:
                if trial_payload.get("implementation") != "regenie":
                    continue
                command_manifests.append(
                    build_command_manifest(
                        command_name=str(trial_payload.get("name", "")),
                        status=str(trial_payload.get("status", "")),
                        command_arguments=[str(value) for value in trial_payload.get("command_arguments", [])],
                    )
                )
    return command_manifests


def collect_manifest_input_files(
    *,
    profile_plan: profile_deep_models.ProfilePlan | None,
    summary_payload: dict[str, typing.Any] | None,
) -> list[dict[str, object]]:
    """Collect profile input files for the artifact manifest."""
    if summary_payload is not None:
        preflight_payload = typing.cast("dict[str, typing.Any]", summary_payload.get("preflight", {}))
        file_sizes = typing.cast("dict[str, int]", preflight_payload.get("input_file_sizes", {}))
        if file_sizes:
            return [
                {"path": path, "size_bytes": size_bytes}
                for path, size_bytes in sorted(file_sizes.items(), key=lambda item: item[0])
            ]
    if profile_plan is None:
        return []
    return [{"path": input_path, "size_bytes": None} for input_path in profile_plan.required_inputs]


def collect_artifact_manifest(
    *,
    output_directory: Path,
    profiler_tool_status: dict[str, profile_deep_models.ProfilerToolStatus],
    summary_payload: dict[str, typing.Any] | None = None,
    profile_plan: profile_deep_models.ProfilePlan | None = None,
) -> dict[str, typing.Any]:
    """Build a structured artifact manifest for one profile campaign."""
    artifact_paths = sorted(
        str(path.relative_to(output_directory))
        for path in output_directory.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    skipped_profiles: list[dict[str, typing.Any]] = []
    if summary_payload is not None:
        deep_profile_results = typing.cast("dict[str, typing.Any]", summary_payload.get("deep_profiles", {}))
        sampling_profiles = typing.cast(
            "list[dict[str, typing.Any]]", deep_profile_results.get("sampling_profiles", [])
        )
        skipped_profiles = [profile for profile in sampling_profiles if profile.get("status") == "skipped"]
    else:
        sampling_profiles = []
    baseline_commands = collect_summary_baseline_commands(summary_payload)
    regenie_baseline_scope = None
    if summary_payload is not None:
        regenie_baseline_scope = summary_payload.get("regenie_baseline_scope")
    if profile_plan is not None:
        baseline_commands = profile_plan.regenie_baseline_commands
        regenie_baseline_scope = profile_plan.regenie_baseline_scope
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "output_directory": str(output_directory),
        "profiler_tools": profile_deep_tools.serialize_profiler_tool_status(profiler_tool_status),
        "input_files": collect_manifest_input_files(profile_plan=profile_plan, summary_payload=summary_payload),
        "regenie_baseline_scope": regenie_baseline_scope,
        "regenie_baseline_commands": baseline_commands,
        "artifact_paths": artifact_paths,
        "profiler_runs": collect_profiler_run_manifest_entries(
            output_directory=output_directory,
            sampling_profiles=sampling_profiles,
        ),
        "skipped_profiles": skipped_profiles,
    }


def write_artifact_manifest(
    *,
    output_directory: Path,
    profiler_tool_status: dict[str, profile_deep_models.ProfilerToolStatus],
    summary_payload: dict[str, typing.Any] | None = None,
    profile_plan: profile_deep_models.ProfilePlan | None = None,
) -> Path:
    """Write the legacy profile artifact manifest."""
    manifest = collect_artifact_manifest(
        output_directory=output_directory,
        profiler_tool_status=profiler_tool_status,
        summary_payload=summary_payload,
        profile_plan=profile_plan,
    )
    manifest_path = output_directory / "legacy_artifact_manifest.v2.json"
    tooling_reports.write_versioned_json_report(
        manifest_path,
        manifest,
        ARTIFACT_MANIFEST_CONTRACT,
        sort_keys=True,
    )
    return manifest_path


def profile_status_to_artifact_status(status: str) -> tooling_artifact_format.ToolArtifactStatus:
    """Convert a profile-local status string to the shared artifact status enum."""
    if status == "success":
        return tooling_artifact_format.ToolArtifactStatus.SUCCESS
    if status == "partial":
        return tooling_artifact_format.ToolArtifactStatus.PARTIAL
    if status == "failed":
        return tooling_artifact_format.ToolArtifactStatus.FAILED
    if status == "skipped":
        return tooling_artifact_format.ToolArtifactStatus.SKIPPED
    if status == "unsupported":
        return tooling_artifact_format.ToolArtifactStatus.UNSUPPORTED
    if status == "dry_run" or status == "planned":
        return tooling_artifact_format.ToolArtifactStatus.DRY_RUN
    return tooling_artifact_format.ToolArtifactStatus.INVALID


def profile_command_identifier(raw_name: str, index: int) -> str:
    """Build a filesystem-safe command identifier."""
    normalized = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in raw_name)
    if normalized:
        return normalized
    return f"profile_command_{index:04d}"


def collect_summary_trial_payloads(summary_payload: dict[str, typing.Any] | None) -> list[dict[str, typing.Any]]:
    """Collect trial-like payloads from a deep-profile summary."""
    if summary_payload is None:
        return []
    trial_payloads: list[dict[str, typing.Any]] = []
    for payload in typing.cast("list[dict[str, typing.Any]]", summary_payload.get("setup_results", [])):
        trial_payloads.append(payload)
    for aggregate_payload in typing.cast("list[dict[str, typing.Any]]", summary_payload.get("headline_results", [])):
        for field_name in ("warmup_trials", "trials", "diagnostic_trials"):
            trial_payloads.extend(typing.cast("list[dict[str, typing.Any]]", aggregate_payload.get(field_name, [])))
    deep_profile_results = typing.cast("dict[str, typing.Any]", summary_payload.get("deep_profiles", {}))
    for profile_payload in typing.cast(
        "list[dict[str, typing.Any]]", deep_profile_results.get("sampling_profiles", [])
    ):
        if isinstance(profile_payload.get("command_arguments"), list):
            trial_payloads.append(profile_payload)
    for perturbation_payload in typing.cast(
        "list[dict[str, typing.Any]]",
        summary_payload.get("logging_perturbation_results", []),
    ):
        trial_payload = perturbation_payload.get("trial")
        if isinstance(trial_payload, dict):
            trial_payloads.append(typing.cast("dict[str, typing.Any]", trial_payload))
    return trial_payloads


def build_profile_command_records(
    *,
    output_directory: Path,
    run_id: str,
    summary_payload: dict[str, typing.Any] | None = None,
    profile_plan: profile_deep_models.ProfilePlan | None = None,
) -> list[tooling_artifact_format.CommandRecord]:
    """Build command ledger records for deep-profile subprocesses."""
    command_records: list[tooling_artifact_format.CommandRecord] = []
    for command_index, trial_payload in enumerate(collect_summary_trial_payloads(summary_payload), start=1):
        command_arguments = [str(value) for value in trial_payload.get("command_arguments", [])]
        if not command_arguments:
            continue
        raw_name = str(trial_payload.get("name", f"profile_command_{command_index:04d}"))
        status = profile_status_to_artifact_status(str(trial_payload.get("status", "invalid")))
        command_records.append(
            tooling_artifact_format.build_command_record(
                command_id=profile_command_identifier(raw_name, command_index),
                tool_name="profile_regenie2_deep",
                run_id=run_id,
                phase=str(trial_payload.get("trait_type", "profile")),
                args=command_arguments,
                output_directory=output_directory,
                cwd=REPOSITORY_ROOT,
                environment_overrides=typing.cast(
                    "dict[str, str]",
                    trial_payload.get("environment_overrides", {}),
                ),
                stdout_log=Path(str(trial_payload["stdout_log_path"]))
                if trial_payload.get("stdout_log_path") is not None
                else None,
                stderr_log=Path(str(trial_payload["stderr_log_path"]))
                if trial_payload.get("stderr_log_path") is not None
                else None,
                status=status,
                return_code=None,
                wall_time_seconds=(
                    float(trial_payload["wall_time_seconds"])
                    if isinstance(trial_payload.get("wall_time_seconds"), (int, float))
                    else None
                ),
            )
        )
    if profile_plan is not None:
        next_index = len(command_records) + 1
        for command_manifest in profile_plan.regenie_baseline_commands:
            raw_name = str(command_manifest["name"])
            command_arguments = typing.cast("list[object]", command_manifest["command_arguments"])
            command_records.append(
                tooling_artifact_format.build_command_record(
                    command_id=profile_command_identifier(raw_name, next_index),
                    tool_name="profile_regenie2_deep",
                    run_id=run_id,
                    phase="regenie_baseline",
                    args=[str(value) for value in command_arguments],
                    output_directory=output_directory,
                    cwd=REPOSITORY_ROOT,
                    status=tooling_artifact_format.ToolArtifactStatus.DRY_RUN,
                )
            )
            next_index += 1
    return command_records


def build_profile_input_file_records(
    *,
    profile_plan: profile_deep_models.ProfilePlan | None,
    summary_payload: dict[str, typing.Any] | None,
) -> list[tooling_artifact_format.InputFileRecord]:
    """Build input-file records for profile artifacts."""
    input_records: list[tooling_artifact_format.InputFileRecord] = []
    for input_payload in collect_manifest_input_files(profile_plan=profile_plan, summary_payload=summary_payload):
        path_value = input_payload.get("path")
        if path_value is None:
            continue
        input_records.append(
            tooling_artifact_format.build_input_file_record(
                path=Path(str(path_value)),
                kind="profile_input",
            )
        )
    return input_records


def profile_metric_dimensions(aggregate_payload: dict[str, typing.Any]) -> dict[str, object]:
    """Build metric dimensions for a profile aggregate."""
    return {
        "implementation": str(aggregate_payload.get("implementation", "")),
        "trait_type": str(aggregate_payload.get("trait_type", "")),
        "device": str(aggregate_payload.get("device", "")),
        "status": str(aggregate_payload.get("status", "")),
    }


def optional_float(raw_value: typing.Any) -> float | None:
    """Return a float for numeric values."""
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    return None


def append_profile_summary_metrics(
    *,
    metric_records: list[tooling_artifact_format.MetricRecord],
    run_id: str,
    summary_payload: dict[str, typing.Any],
) -> None:
    """Append normalized metrics from a profile summary payload."""
    metric_specs = (
        ("median_wall_time_seconds", "wall_time_seconds", tooling_artifact_format.MetricAggregation.MEDIAN.value),
        ("mean_wall_time_seconds", "wall_time_seconds", tooling_artifact_format.MetricAggregation.MEAN.value),
        ("min_wall_time_seconds", "wall_time_seconds", tooling_artifact_format.MetricAggregation.MINIMUM.value),
        ("max_wall_time_seconds", "wall_time_seconds", tooling_artifact_format.MetricAggregation.MAXIMUM.value),
        (
            "standard_deviation_seconds",
            "wall_time_seconds",
            tooling_artifact_format.MetricAggregation.STANDARD_DEVIATION.value,
        ),
        (
            "rows_per_second",
            "throughput_rows_per_second",
            tooling_artifact_format.MetricAggregation.MEDIAN.value,
        ),
    )
    for aggregate_index, aggregate_payload in enumerate(
        typing.cast("list[dict[str, typing.Any]]", summary_payload.get("headline_results", []))
    ):
        case_id = str(aggregate_payload.get("name", f"headline_{aggregate_index}"))
        for source_field, metric_name, aggregation in metric_specs:
            unit = (
                tooling_artifact_format.MetricUnit.ROW.value
                if metric_name == "throughput_rows_per_second"
                else tooling_artifact_format.MetricUnit.SECONDS.value
            )
            metric_records.append(
                tooling_artifact_format.build_metric_record(
                    run_id=run_id,
                    case_id=case_id,
                    metric_name=metric_name,
                    value=optional_float(aggregate_payload.get(source_field)),
                    unit=unit,
                    aggregation=aggregation,
                    higher_is_better=metric_name == "throughput_rows_per_second",
                    dimensions=profile_metric_dimensions(aggregate_payload),
                    phase="headline_trials",
                    source=tooling_artifact_format.MetricSource(
                        artifact_path="summary.json",
                        json_pointer=f"/headline_results/{aggregate_index}/{source_field}",
                    ),
                )
            )
    stage_totals = typing.cast("dict[str, typing.Any]", summary_payload.get("stage_totals", {}))
    for stage_name, seconds in sorted(stage_totals.items()):
        metric_records.append(
            tooling_artifact_format.build_metric_record(
                run_id=run_id,
                case_id=None,
                metric_name=f"stage.{stage_name}.seconds",
                value=optional_float(seconds),
                unit=tooling_artifact_format.MetricUnit.SECONDS.value,
                aggregation=tooling_artifact_format.MetricAggregation.EXACT.value,
                higher_is_better=False,
                dimensions={},
                phase="stage_totals",
                source=tooling_artifact_format.MetricSource(
                    artifact_path="summary.json",
                    json_pointer=f"/stage_totals/{stage_name}",
                ),
            )
        )


def append_profile_plan_metrics(
    *,
    metric_records: list[tooling_artifact_format.MetricRecord],
    run_id: str,
    profile_plan: profile_deep_models.ProfilePlan,
) -> None:
    """Append normalized metrics from a dry-run profile plan."""
    budget_metrics = {
        "candidate_count": profile_plan.campaign_budget.total_candidate_count,
        "subprocess_run_count": profile_plan.campaign_budget.total_subprocess_run_count,
        "major_profiler_run_count": profile_plan.campaign_budget.total_major_profiler_run_count,
    }
    for metric_name, metric_value in budget_metrics.items():
        metric_records.append(
            tooling_artifact_format.build_metric_record(
                run_id=run_id,
                case_id="campaign_budget",
                metric_name=metric_name,
                value=metric_value,
                unit=tooling_artifact_format.MetricUnit.COUNT.value,
                aggregation=tooling_artifact_format.MetricAggregation.EXACT.value,
                higher_is_better=None,
                dimensions={"chromosome_label": profile_plan.chromosome_label},
                phase="planning",
                source=tooling_artifact_format.MetricSource(
                    artifact_path="profile_plan.json",
                    json_pointer=f"/campaign_budget/{metric_name}",
                ),
            )
        )


def build_profile_metrics(
    *,
    run_id: str,
    summary_payload: dict[str, typing.Any] | None = None,
    profile_plan: profile_deep_models.ProfilePlan | None = None,
) -> list[tooling_artifact_format.MetricRecord]:
    """Build normalized profile metrics."""
    metric_records: list[tooling_artifact_format.MetricRecord] = []
    if summary_payload is not None:
        append_profile_summary_metrics(
            metric_records=metric_records,
            run_id=run_id,
            summary_payload=summary_payload,
        )
    if profile_plan is not None:
        append_profile_plan_metrics(
            metric_records=metric_records,
            run_id=run_id,
            profile_plan=profile_plan,
        )
    return metric_records


def build_profile_failure_records(
    summary_payload: dict[str, typing.Any] | None,
) -> list[tooling_artifact_format.FailureRecord]:
    """Build structured failure records for profile trials."""
    failure_records: list[tooling_artifact_format.FailureRecord] = []
    for failure_index, trial_payload in enumerate(
        (
            payload
            for payload in collect_summary_trial_payloads(summary_payload)
            if str(payload.get("status", "")) == "failed"
        ),
        start=1,
    ):
        failure_records.append(
            tooling_artifact_format.FailureRecord(
                failure_id=f"F{failure_index:03d}",
                phase=str(trial_payload.get("trait_type", "profile")),
                status=tooling_artifact_format.ToolArtifactStatus.FAILED,
                message=f"Profile trial {trial_payload.get('name', failure_index)} failed.",
                exception_type=None,
                stderr_excerpt=None,
                stdout_log=str(trial_payload.get("stdout_log_path"))
                if trial_payload.get("stdout_log_path") is not None
                else None,
                stderr_log=str(trial_payload.get("stderr_log_path"))
                if trial_payload.get("stderr_log_path") is not None
                else None,
                command_id=profile_command_identifier(str(trial_payload.get("name", "")), failure_index),
            )
        )
    return failure_records


def build_profile_cases(
    summary_payload: dict[str, typing.Any] | None, profile_plan: profile_deep_models.ProfilePlan | None
) -> list[dict[str, object]]:
    """Build report case records for profile artifacts."""
    if summary_payload is not None:
        cases: list[dict[str, object]] = []
        for aggregate_payload in typing.cast(
            "list[dict[str, typing.Any]]", summary_payload.get("headline_results", [])
        ):
            case_payload = dict(aggregate_payload)
            case_payload.pop("trials", None)
            case_payload.pop("warmup_trials", None)
            cases.append(typing.cast("dict[str, object]", case_payload))
        return cases
    if profile_plan is None:
        return []
    return [
        {
            "case_id": section.name,
            "display_name": section.display_name,
            "candidate_count": section.candidate_count,
            "subprocess_run_count": section.subprocess_run_count,
            "major_profiler_run_count": section.major_profiler_run_count,
        }
        for section in profile_plan.campaign_budget.sections
    ]


def build_profile_agent_summary(
    *,
    status: tooling_artifact_format.ToolArtifactStatus,
    summary_payload: dict[str, typing.Any] | None,
    profile_plan: profile_deep_models.ProfilePlan | None,
) -> dict[str, object]:
    """Build a concise agent-oriented profile summary."""
    if summary_payload is not None:
        headline_count = len(typing.cast("list[dict[str, typing.Any]]", summary_payload.get("headline_results", [])))
        failures = build_profile_failure_records(summary_payload)
        return {
            "one_sentence": f"Deep profile completed with {headline_count} headline aggregate results.",
            "key_observations": [
                f"Status: {status.value}.",
                f"Headline aggregate count: {headline_count}.",
                f"Structured failure count: {len(failures)}.",
            ],
            "risks": [failure.message for failure in failures[:5]],
            "next_actions": [],
        }
    if profile_plan is not None:
        return {
            "one_sentence": "Deep profile plan was written without executing workloads.",
            "key_observations": [
                f"Status: {status.value}.",
                f"Estimated subprocess runs: {profile_plan.campaign_budget.total_subprocess_run_count}.",
                f"Estimated major profiler runs: {profile_plan.campaign_budget.total_major_profiler_run_count}.",
            ],
            "risks": list(profile_plan.campaign_budget.guidance),
            "next_actions": [],
        }
    return {
        "one_sentence": f"Deep profile finished with status {status.value}.",
        "key_observations": [f"Status: {status.value}."],
        "risks": [],
        "next_actions": [],
    }


def profile_summary_artifact_status(
    summary_payload: dict[str, typing.Any],
) -> tooling_artifact_format.ToolArtifactStatus:
    """Determine the overall standard status for a completed profile summary."""
    headline_results = typing.cast("list[dict[str, typing.Any]]", summary_payload.get("headline_results", []))
    if not headline_results:
        return tooling_artifact_format.ToolArtifactStatus.FAILED
    aggregate_statuses = {str(result.get("status", "")) for result in headline_results}
    if "success" not in aggregate_statuses and "partial" not in aggregate_statuses:
        return tooling_artifact_format.ToolArtifactStatus.FAILED
    trial_statuses = {str(payload.get("status", "")) for payload in collect_summary_trial_payloads(summary_payload)}
    if aggregate_statuses != {"success"} or trial_statuses.intersection({"failed", "partial"}):
        return tooling_artifact_format.ToolArtifactStatus.PARTIAL
    return tooling_artifact_format.ToolArtifactStatus.SUCCESS


def write_standard_profile_artifacts(
    *,
    arguments: profile_deep_models.ProfileArguments,
    output_directory: Path,
    profiler_tool_status: dict[str, profile_deep_models.ProfilerToolStatus],
    status: tooling_artifact_format.ToolArtifactStatus,
    status_reason: str | None = None,
    summary_payload: dict[str, typing.Any] | None = None,
    profile_plan: profile_deep_models.ProfilePlan | None = None,
    summary_markdown: str | None = None,
    hydra_config: omegaconf.DictConfig | None = None,
) -> None:
    """Write Tooling Artifact Format v1 artifacts for the deep profiler."""
    producer = tooling_artifact_format.build_producer(
        tool_name="profile_regenie2_deep",
        repository_root=REPOSITORY_ROOT,
    )
    run = tooling_artifact_format.build_run_identity(
        tool_name="profile_regenie2_deep",
        output_directory=output_directory,
        status=status,
        status_reason=status_reason,
    )
    context_snapshot = tooling_artifact_format.build_context_snapshot(
        output_directory=output_directory,
        repository_root=REPOSITORY_ROOT,
    )
    report = tooling_artifact_format.build_report_envelope(
        producer=producer,
        run=run,
        context=context_snapshot,
        title=f"{arguments.chromosome_label} Deep REGENIE Step 2 Profile",
        configuration=profile_deep_config.profile_configuration_payload(arguments),
        summary={
            "headline": f"Deep profile finished with status {status.value}.",
            "agent_summary": build_profile_agent_summary(
                status=status,
                summary_payload=summary_payload,
                profile_plan=profile_plan,
            ),
            "legacy_summary_path": "summary.json" if summary_payload is not None else None,
            "profile_plan_path": "profile_plan.json" if profile_plan is not None else None,
        },
        cases=build_profile_cases(summary_payload, profile_plan),
        trials=typing.cast("list[dict[str, object]]", collect_summary_trial_payloads(summary_payload)),
        metrics=build_profile_metrics(
            run_id=run.run_id,
            summary_payload=summary_payload,
            profile_plan=profile_plan,
        ),
        diagnostics={
            "profiler_tools": profile_deep_tools.serialize_profiler_tool_status(profiler_tool_status),
            "legacy_artifact_manifest": "legacy_artifact_manifest.v2.json",
        },
        failures=build_profile_failure_records(summary_payload),
    )
    events = [
        tooling_artifact_format.build_tool_event(
            tool_name="profile_regenie2_deep",
            run_id=run.run_id,
            phase="profile",
            event="profile_artifacts_written",
            message=f"Deep profile artifacts written with status {status.value}.",
            fields={
                "chromosome_label": arguments.chromosome_label,
                "dry_run": arguments.dry_run,
                "status_reason": status_reason,
            },
        )
    ]
    tooling_artifact_format.write_standard_artifact_bundle(
        output_directory=output_directory,
        report=report,
        events=events,
        commands=build_profile_command_records(
            output_directory=output_directory,
            run_id=run.run_id,
            summary_payload=summary_payload,
            profile_plan=profile_plan,
        ),
        input_files=build_profile_input_file_records(
            profile_plan=profile_plan,
            summary_payload=summary_payload,
        ),
        summary_markdown=summary_markdown,
        hydra_config=hydra_config,
        tool_payload=profile_deep_config.profile_configuration_payload(arguments),
        notes=["legacy_artifact_manifest.v2.json preserves the pre-v1 deep-profile manifest shape."],
    )
