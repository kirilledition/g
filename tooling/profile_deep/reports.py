"""Runtime comparisons and Markdown summaries for profile campaigns."""

from __future__ import annotations

import typing

from tooling.profile_deep import diagnostics as profile_deep_diagnostics
from tooling.profile_deep import jax_cache as profile_deep_jax_cache
from tooling.profile_deep import models as profile_deep_models


def build_runtime_comparisons(
    aggregate_results: list[profile_deep_models.AggregateResult],
) -> dict[str, dict[str, float]]:
    """Build speedup/slowdown comparisons against original REGENIE."""
    by_name = {result.name: result for result in aggregate_results}
    comparisons: dict[str, dict[str, float]] = {}
    for trait_type in ("quantitative", "binary"):
        baseline = by_name.get(f"headline_regenie_{trait_type}")
        if baseline is None or baseline.median_wall_time_seconds is None:
            continue
        for result in aggregate_results:
            if (
                result.implementation != "g"
                or result.trait_type != trait_type
                or result.median_wall_time_seconds is None
            ):
                continue
            comparison_name = f"{result.name}_vs_regenie_{trait_type}"
            comparisons[comparison_name] = {
                "speedup_ratio": baseline.median_wall_time_seconds / result.median_wall_time_seconds,
                "absolute_delta_seconds": result.median_wall_time_seconds - baseline.median_wall_time_seconds,
            }
    return comparisons


def build_runtime_comparison_notes(
    aggregate_results: list[profile_deep_models.AggregateResult],
) -> profile_deep_models.RuntimeComparisonNotes:
    """Build explicit non-success direct comparison notes."""
    unsupported: list[str] = []
    failed: list[str] = []
    regenie_results = {result.trait_type: result for result in aggregate_results if result.implementation == "regenie"}
    for g_result in aggregate_results:
        if g_result.implementation != "g":
            continue
        comparison_name = f"{g_result.name}_vs_regenie_{g_result.trait_type}"
        regenie_result = regenie_results.get(g_result.trait_type)
        if regenie_result is None:
            unsupported.append(f"{comparison_name}: no original REGENIE baseline was scheduled for this trait.")
            continue
        if regenie_result.status == "unsupported":
            notes = next((trial.notes for trial in regenie_result.trials if trial.notes), None)
            suffix = f" {notes}" if notes is not None else ""
            unsupported.append(f"{comparison_name}: original REGENIE baseline is unsupported.{suffix}")
            continue
        if regenie_result.median_wall_time_seconds is None:
            notes = next((trial.notes for trial in regenie_result.trials if trial.notes), None)
            suffix = f" {notes}" if notes is not None else ""
            failed.append(f"{comparison_name}: original REGENIE baseline did not produce a measured runtime.{suffix}")
            continue
        if g_result.median_wall_time_seconds is None:
            failed.append(f"{comparison_name}: g result did not produce a measured runtime.")
    return profile_deep_models.RuntimeComparisonNotes(unsupported=unsupported, failed=failed)


def build_summary_markdown(
    *,
    aggregate_results: list[profile_deep_models.AggregateResult],
    comparisons: dict[str, dict[str, float]],
    stage_totals: dict[str, float],
    comparison_notes: profile_deep_models.RuntimeComparisonNotes | None = None,
    regenie_baseline_scope: profile_deep_models.RegenieBaselineScope | None = None,
    logging_perturbation_results: list[dict[str, typing.Any]] | None = None,
    binary_correction_diagnostics: dict[str, typing.Any] | None = None,
) -> str:
    """Build the human-readable campaign summary."""
    lines = ["# Landau Deep REGENIE Step 2 Profile", ""]
    lines.append("## Headline Runtimes")
    lines.append("")
    lines.append("| name | trait | device | median s | mean s | min s | max s | std s | rows/s |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in aggregate_results:
        lines.append(
            "| "
            f"{result.name} | {result.trait_type} | {result.device} | "
            f"{format_optional_float(result.median_wall_time_seconds)} | "
            f"{format_optional_float(result.mean_wall_time_seconds)} | "
            f"{format_optional_float(result.min_wall_time_seconds)} | "
            f"{format_optional_float(result.max_wall_time_seconds)} | "
            f"{format_optional_float(result.standard_deviation_seconds)} | "
            f"{format_optional_float(result.rows_per_second)} |"
        )
    lines.extend(["", "## JAX Compile And Cache Diagnostics", ""])
    jax_cache_diagnostics = profile_deep_jax_cache.collect_jax_cache_diagnostics(aggregate_results)
    if jax_cache_diagnostics:
        lines.append(
            "_Cold is the first successful g subprocess for an aggregate; warm summarizes later successful "
            "subprocesses sharing the same persistent cache directory._"
        )
        lines.append("")
        lines.append(
            "| name | persistent cache | cache dir | cold s | warm median s | cold/warm | "
            "cache files Δ cold/warm | cache bytes Δ cold/warm | compiles cold/warm | "
            "persistent hits cold/warm | persistent misses cold/warm | trace misses cold/warm |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for result_name, diagnostics in jax_cache_diagnostics.items():
            cache_directory = diagnostics.get("cache_directory")
            lines.append(
                "| "
                f"{result_name} | "
                f"{str(diagnostics['persistent_cache_used']).lower()} | "
                f"`{cache_directory}` | "
                f"{format_optional_float(typing.cast('float | None', diagnostics['cold_wall_time_seconds']))} | "
                f"{format_optional_float(typing.cast('float | None', diagnostics['warm_median_wall_time_seconds']))} | "
                f"{format_optional_float(typing.cast('float | None', diagnostics['cold_to_warm_speedup_ratio']))} | "
                f"{format_optional_integer(typing.cast('int | None', diagnostics['cold_cache_file_count_delta']))}/"
                f"{format_optional_integer(typing.cast('int | None', diagnostics['warm_cache_file_count_delta']))} | "
                f"{format_optional_integer(typing.cast('int | None', diagnostics['cold_cache_size_bytes_delta']))}/"
                f"{format_optional_integer(typing.cast('int | None', diagnostics['warm_cache_size_bytes_delta']))} | "
                f"{diagnostics['cold_compilation_event_count']}/{diagnostics['warm_compilation_event_count']} | "
                f"{diagnostics['cold_cache_hit_count']}/{diagnostics['warm_cache_hit_count']} | "
                f"{diagnostics['cold_cache_miss_count']}/{diagnostics['warm_cache_miss_count']} | "
                f"{diagnostics['cold_tracing_cache_miss_count']}/{diagnostics['warm_tracing_cache_miss_count']} |"
            )
    else:
        lines.append("- No JAX cache diagnostics were available.")
    lines.extend(["", "## Runtime Comparisons", ""])
    lines.append("### Successful")
    lines.append("")
    if comparisons:
        for comparison_name, comparison in comparisons.items():
            lines.append(
                f"- {comparison_name}: speedup={comparison['speedup_ratio']:.4f}x, "
                f"delta={comparison['absolute_delta_seconds']:.4f}s"
            )
    else:
        lines.append("- No successful direct comparisons were available.")
    comparison_details = comparison_notes or profile_deep_models.RuntimeComparisonNotes(unsupported=[], failed=[])
    lines.extend(["", "### Unsupported", ""])
    if comparison_details.unsupported:
        for note in comparison_details.unsupported:
            lines.append(f"- {note}")
    else:
        lines.append("- No unsupported direct comparisons were recorded.")
    lines.extend(["", "### Failed", ""])
    if comparison_details.failed:
        for note in comparison_details.failed:
            lines.append(f"- {note}")
    else:
        lines.append("- No failed direct comparisons were recorded.")
    lines.extend(["", "## REGENIE Baseline Scope", ""])
    if regenie_baseline_scope is None:
        lines.append("- Original REGENIE baseline scope was not requested.")
    else:
        lines.append(f"- Status: `{regenie_baseline_scope.status.value}`")
        lines.append(f"- Variant limit: `{regenie_baseline_scope.variant_limit}`")
        if regenie_baseline_scope.extract_path is not None:
            lines.append(f"- Extract list: `{regenie_baseline_scope.extract_path}`")
        if regenie_baseline_scope.metadata_path is not None:
            lines.append(f"- Variant metadata: `{regenie_baseline_scope.metadata_path}`")
        if regenie_baseline_scope.selected_variant_count is not None:
            lines.append(f"- Selected variants: `{regenie_baseline_scope.selected_variant_count}`")
        lines.append(f"- Notes: {regenie_baseline_scope.notes}")
    lines.extend(["", "## Stage Timings", ""])
    lines.append(
        "_Raw profiler stages are reported without mapping g's inclusive native execution timer onto "
        "REGENIE's finer-grained decode or compute stages._"
    )
    if stage_totals:
        for stage_name, seconds in sorted(stage_totals.items()):
            lines.append(f"- `{stage_name}`: `{seconds:.6f}` seconds")
    else:
        lines.append("- No stage timing JSON files were available.")
    append_binary_correction_diagnostics_markdown(lines, binary_correction_diagnostics or {})
    lines.extend(["", "## Logging And Telemetry Perturbation", ""])
    logging_rows = build_logging_perturbation_rows(logging_perturbation_results or [])
    if logging_rows:
        lines.append("| winner | case | wall s | delta vs off s | ratio vs off | status |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for row in logging_rows:
            lines.append(
                "| "
                f"{row['winner_key']} | {row['case_name']} | "
                f"{format_optional_float(typing.cast('float | None', row['wall_time_seconds']))} | "
                f"{format_optional_float(typing.cast('float | None', row['delta_vs_off_seconds']))} | "
                f"{format_optional_float(typing.cast('float | None', row['ratio_vs_off']))} | "
                f"{row['status']} |"
            )
    else:
        lines.append("- No logging perturbation results were available.")
    lines.extend(["", "## Ranked Bottlenecks", ""])
    if stage_totals:
        for stage_name, seconds in sorted(stage_totals.items(), key=lambda item: item[1], reverse=True)[:20]:
            lines.append(f"- {stage_name}: {seconds:.6f}s")
    else:
        lines.append("- No stage timing JSON files were available.")
    lines.extend(["", "## Next Optimization Targets", ""])
    if stage_totals:
        for stage_name, seconds in sorted(stage_totals.items(), key=lambda item: item[1], reverse=True)[:5]:
            lines.append(f"- Reduce `{stage_name}` first; it is one of the largest measured wall-time shares.")
    else:
        lines.append("- Re-run with successful g diagnostic trials to rank measured stage shares.")
    return "\n".join(lines) + "\n"


def diagnostic_mapping(raw_value: typing.Any) -> dict[str, typing.Any]:
    """Return a diagnostic mapping or an empty mapping."""
    if isinstance(raw_value, dict):
        return typing.cast("dict[str, typing.Any]", raw_value)
    return {}


def binary_diagnostic_markdown_rows(
    binary_correction_diagnostics: dict[str, typing.Any],
    group_name: str,
) -> list[dict[str, typing.Any]]:
    """Flatten headline or finalist diagnostic payloads into Markdown rows."""
    rows: list[dict[str, typing.Any]] = []
    raw_group = binary_correction_diagnostics.get(group_name)
    if group_name == "headline":
        group_payload = diagnostic_mapping(raw_group)
        for aggregate_name, raw_diagnostics in sorted(group_payload.items()):
            diagnostics = diagnostic_mapping(raw_diagnostics)
            if diagnostics:
                row = dict(diagnostics)
                row["display_name"] = aggregate_name
                rows.append(row)
        return rows
    nested_payload = diagnostic_mapping(raw_group)
    for winner_key, raw_aggregate_payload in sorted(nested_payload.items()):
        aggregate_payload = diagnostic_mapping(raw_aggregate_payload)
        for aggregate_name, raw_diagnostics in sorted(aggregate_payload.items()):
            diagnostics = diagnostic_mapping(raw_diagnostics)
            if diagnostics:
                row = dict(diagnostics)
                row["display_name"] = f"{winner_key}/{aggregate_name}"
                rows.append(row)
    return rows


def format_diagnostic_integer(raw_value: typing.Any) -> str:
    """Format an optional diagnostic integer."""
    numeric_value = profile_deep_diagnostics.optional_numeric_value(raw_value)
    if numeric_value is None:
        return ""
    return str(int(numeric_value))


def format_diagnostic_ratio(raw_value: typing.Any) -> str:
    """Format an optional diagnostic ratio as a percentage."""
    numeric_value = profile_deep_diagnostics.optional_numeric_value(raw_value)
    if numeric_value is None:
        return ""
    return f"{numeric_value * 100.0:.2f}%"


def format_binary_diagnostic_status(diagnostics: dict[str, typing.Any]) -> str:
    """Format binary diagnostic availability for Markdown."""
    if diagnostics.get("available") is True:
        return "available"
    reason = str(diagnostics.get("reason", "unavailable"))
    if reason == profile_deep_diagnostics.BINARY_DIAGNOSTIC_UNAVAILABLE_EXACT_TIMING_DISABLED:
        return "unavailable: stage timing mode off"
    return f"unavailable: {reason}"


def format_binary_failure_counts(failure_counts: dict[str, typing.Any]) -> str:
    """Format compact failure-code counts."""
    if not failure_counts:
        return ""
    formatted_counts = []
    for field_name in ("numerical", "max_iterations", "invalid_statistic", "step_halving"):
        numeric_value = profile_deep_diagnostics.optional_numeric_value(failure_counts.get(field_name))
        if numeric_value is not None and numeric_value > 0.0:
            formatted_counts.append(f"{field_name}={int(numeric_value)}")
    if formatted_counts:
        return ", ".join(formatted_counts)
    none_count = profile_deep_diagnostics.optional_numeric_value(failure_counts.get("none"))
    if none_count is None:
        return ""
    return f"none={int(none_count)}"


def format_binary_branch_mix(branch_counts: dict[str, typing.Any]) -> str:
    """Format compact Firth correction branch counts."""
    if not branch_counts:
        return ""
    return (
        "pseudo="
        f"{format_diagnostic_integer(branch_counts.get('pseudo_firth'))}, "
        "zero="
        f"{format_diagnostic_integer(branch_counts.get('newton_raphson_zero_start'))}, "
        "warm="
        f"{format_diagnostic_integer(branch_counts.get('newton_raphson_warm_start'))}"
    )


def format_binary_firth_iterations(iteration_counts: dict[str, typing.Any]) -> str:
    """Format compact Firth iteration summary."""
    if not iteration_counts:
        return ""
    minimum = profile_deep_diagnostics.optional_numeric_value(iteration_counts.get("minimum"))
    median_mean = profile_deep_diagnostics.optional_numeric_value(iteration_counts.get("median_per_chunk_mean"))
    maximum = profile_deep_diagnostics.optional_numeric_value(iteration_counts.get("maximum"))
    if minimum is None or median_mean is None or maximum is None:
        return ""
    return f"{minimum:.0f}/{median_mean:.1f}/{maximum:.0f}"


def append_binary_diagnostic_table(
    lines: list[str],
    *,
    title: str,
    rows: list[dict[str, typing.Any]],
) -> None:
    """Append one compact binary diagnostic Markdown table."""
    lines.extend(["", f"### {title}", ""])
    if not rows:
        lines.append("- No binary correction diagnostics were available.")
        return
    lines.append(
        "| run | device | status | trials | chunks | score cand | Firth cand | corrected/failed | failures | "
        "iters min/mean/max | branch mix | sparse/dense | Firth density |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |")
    for diagnostics in rows:
        candidate_counts = diagnostic_mapping(diagnostics.get("candidate_counts"))
        outcome_counts = diagnostic_mapping(diagnostics.get("correction_outcome_counts"))
        failure_counts = diagnostic_mapping(diagnostics.get("failure_code_counts"))
        iteration_counts = diagnostic_mapping(diagnostics.get("firth_iteration_counts"))
        branch_counts = diagnostic_mapping(diagnostics.get("correction_branch_counts"))
        input_counts = diagnostic_mapping(diagnostics.get("correction_input_counts"))
        fallback_density = diagnostic_mapping(diagnostics.get("fallback_density"))
        lines.append(
            "| "
            f"{diagnostics.get('display_name', diagnostics.get('aggregate_name', ''))} | "
            f"{diagnostics.get('device', '')} | "
            f"{format_binary_diagnostic_status(diagnostics)} | "
            f"{format_diagnostic_integer(diagnostics.get('available_trial_count'))} | "
            f"{format_diagnostic_integer(diagnostics.get('chunk_count'))} | "
            f"{format_diagnostic_integer(candidate_counts.get('score_test'))} | "
            f"{format_diagnostic_integer(candidate_counts.get('firth'))} | "
            f"{format_diagnostic_integer(outcome_counts.get('corrected'))}/"
            f"{format_diagnostic_integer(outcome_counts.get('failed'))} | "
            f"{format_binary_failure_counts(failure_counts)} | "
            f"{format_binary_firth_iterations(iteration_counts)} | "
            f"{format_binary_branch_mix(branch_counts)} | "
            f"{format_diagnostic_integer(input_counts.get('sparse'))}/"
            f"{format_diagnostic_integer(input_counts.get('dense'))} | "
            f"{format_diagnostic_ratio(fallback_density.get('firth_candidates_per_output_row'))} |"
        )


def append_binary_correction_diagnostics_markdown(
    lines: list[str],
    binary_correction_diagnostics: dict[str, typing.Any],
) -> None:
    """Append compact binary correction diagnostics to the summary report."""
    lines.extend(["", "## Binary Correction Diagnostics", ""])
    if not binary_correction_diagnostics:
        lines.append("- No binary correction diagnostics were computed.")
        return
    stage_timing_mode = str(binary_correction_diagnostics.get("stage_timing_mode", "unknown"))
    lines.append(
        "_Exact stage timing JSON is required for correction diagnostics; bounded per-chunk outliers remain in "
        "`summary.json` and raw stage timing artifacts._"
    )
    if stage_timing_mode == profile_deep_models.ProfileStageTimingMode.OFF.value:
        lines.append("- Diagnostics are unavailable because `telemetry.stage_timing_mode=off`.")
    append_binary_diagnostic_table(
        lines,
        title="Headline Winners",
        rows=binary_diagnostic_markdown_rows(binary_correction_diagnostics, "headline"),
    )
    append_binary_diagnostic_table(
        lines,
        title="Finalists",
        rows=binary_diagnostic_markdown_rows(binary_correction_diagnostics, "finalists"),
    )


def build_logging_perturbation_rows(
    logging_perturbation_results: list[dict[str, typing.Any]],
) -> list[dict[str, float | str | None]]:
    """Build comparable telemetry/logging perturbation rows."""
    baseline_times: dict[str, float] = {}
    for result in logging_perturbation_results:
        winner_key = str(result["winner_key"])
        case_payload = typing.cast("dict[str, typing.Any]", result["case"])
        trial_payload = typing.cast("dict[str, typing.Any]", result["trial"])
        wall_time = trial_payload.get("wall_time_seconds")
        if case_payload.get("name") == "telemetry_off" and isinstance(wall_time, (int, float)):
            baseline_times[winner_key] = float(wall_time)
    rows: list[dict[str, float | str | None]] = []
    for result in logging_perturbation_results:
        winner_key = str(result["winner_key"])
        case_payload = typing.cast("dict[str, typing.Any]", result["case"])
        trial_payload = typing.cast("dict[str, typing.Any]", result["trial"])
        wall_time_value = trial_payload.get("wall_time_seconds")
        wall_time = float(wall_time_value) if isinstance(wall_time_value, (int, float)) else None
        baseline_time = baseline_times.get(winner_key)
        delta_vs_off = None
        ratio_vs_off = None
        if wall_time is not None and baseline_time is not None and baseline_time > 0.0:
            delta_vs_off = wall_time - baseline_time
            ratio_vs_off = wall_time / baseline_time
        rows.append(
            {
                "winner_key": winner_key,
                "case_name": str(case_payload.get("name", "")),
                "wall_time_seconds": wall_time,
                "delta_vs_off_seconds": delta_vs_off,
                "ratio_vs_off": ratio_vs_off,
                "status": str(trial_payload.get("status", "")),
            }
        )
    return rows


def format_optional_float(value: float | None) -> str:
    """Format optional floats for markdown tables."""
    if value is None:
        return ""
    return f"{value:.6f}"


def format_optional_integer(value: int | None) -> str:
    """Format optional integers for markdown tables."""
    if value is None:
        return ""
    return str(value)
