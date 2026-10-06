"""Native-stage and binary-correction diagnostics for profile campaigns."""

from __future__ import annotations

import json
import statistics
import typing
from pathlib import Path

from tooling.benchmark import native_lifecycle
from tooling.profile_deep import models as profile_deep_models

BINARY_DIAGNOSTIC_UNAVAILABLE_EXACT_TIMING_DISABLED = "exact_stage_timings_disabled"


BINARY_DIAGNOSTIC_UNAVAILABLE_STAGE_TIMING_FILE_MISSING = "stage_timing_file_missing"


BINARY_DIAGNOSTIC_UNAVAILABLE_STAGE_TIMING_FILE_INVALID = "stage_timing_file_invalid"


BINARY_DIAGNOSTIC_UNAVAILABLE_BINARY_DIAGNOSTICS_MISSING = "binary_chunk_diagnostics_missing"


BINARY_DIAGNOSTIC_UNAVAILABLE_BINARY_DIAGNOSTICS_INVALID = "binary_chunk_diagnostics_invalid"


BINARY_DIAGNOSTIC_COUNT_FIELDS = (
    "score_test_candidate_count",
    "firth_candidate_count",
    "firth_converged_count",
    "firth_failed_count",
    "firth_numerical_failure_count",
    "firth_max_iteration_failure_count",
    "firth_invalid_statistic_failure_count",
    "firth_step_halving_failure_count",
    "pseudo_firth_attempt_count",
    "pseudo_firth_success_count",
    "nr_zero_start_attempt_count",
    "nr_zero_start_success_count",
    "nr_warm_start_attempt_count",
    "nr_warm_start_success_count",
    "sparse_correction_count",
    "dense_correction_count",
)


BINARY_CHUNK_OUTLIER_LIMIT = 5


def read_json_file(path: str | None) -> dict[str, typing.Any] | None:
    """Read a JSON file when the path exists."""
    if path is None:
        return None
    json_path = Path(path)
    if not json_path.exists():
        return None
    return typing.cast("dict[str, typing.Any]", json.loads(json_path.read_text(encoding="utf-8")))


def collect_trial_stage_totals(trial: profile_deep_models.TrialResult) -> dict[str, float]:
    """Collect raw stage totals for one g or REGENIE trial."""
    stage_totals: dict[str, float] = {}
    for profile_path in (trial.stage_timing_path, trial.profile_summary_path, trial.regenie_profile_path):
        stage_payload = read_json_file(profile_path)
        if stage_payload is None:
            continue
        stage_totals.update(
            {
                stage_name: float(seconds)
                for stage_name, seconds in stage_payload.get("stage_totals_seconds", {}).items()
            }
        )
    if trial.implementation == "g" and (trial.stage_timing_path is not None or trial.profile_summary_path is not None):
        missing_stage_names = [
            stage_name for stage_name in native_lifecycle.NATIVE_PROFILE_STAGE_NAMES if stage_name not in stage_totals
        ]
        if missing_stage_names:
            raise RuntimeError(f"g trial {trial.name} is missing current native profile stages: {missing_stage_names}")
    if trial.implementation == "g" and trial.application_output_run_directory is not None:
        output_stage_path = Path(trial.application_output_run_directory) / "output_stage_timings.json"
        output_stage_payload = read_json_file(str(output_stage_path))
        if output_stage_payload is not None:
            stage_totals.update(
                {
                    stage_name: float(seconds)
                    for stage_name, seconds in output_stage_payload.get("stage_totals_seconds", {}).items()
                    if str(stage_name).startswith("rust_output_")
                }
            )
    return stage_totals


def collect_stage_totals(aggregate_results: list[profile_deep_models.AggregateResult]) -> dict[str, float]:
    """Collect representative stage totals from g and REGENIE trials."""
    stage_totals: dict[str, float] = {}
    for aggregate_result in aggregate_results:
        for trial in (*aggregate_result.trials, *aggregate_result.diagnostic_trials):
            for stage_name, seconds in collect_trial_stage_totals(trial).items():
                key = f"{aggregate_result.name}:{stage_name}"
                stage_totals[key] = seconds
    return stage_totals


def numeric_diagnostic_value(raw_value: typing.Any) -> float:
    """Convert a diagnostic JSON value into a numeric value."""
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        return 0.0
    return float(raw_value)


def optional_numeric_value(raw_value: typing.Any) -> float | None:
    """Convert a JSON value into a float when it is numeric."""
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        return None
    return float(raw_value)


def sum_binary_diagnostic_count(binary_chunk_diagnostics: list[dict[str, typing.Any]], field_name: str) -> int:
    """Sum one integer diagnostic field across binary chunks."""
    return int(
        sum(numeric_diagnostic_value(diagnostics.get(field_name, 0)) for diagnostics in binary_chunk_diagnostics)
    )


def mean_binary_diagnostic_value(binary_chunk_diagnostics: list[dict[str, typing.Any]], field_name: str) -> float:
    """Average one diagnostic field across chunks."""
    if not binary_chunk_diagnostics:
        return 0.0
    total = sum(numeric_diagnostic_value(diagnostics.get(field_name, 0)) for diagnostics in binary_chunk_diagnostics)
    return total / len(binary_chunk_diagnostics)


def active_firth_iteration_values(
    binary_chunk_diagnostics: list[dict[str, typing.Any]],
    field_name: str,
) -> list[float]:
    """Return a Firth iteration field for chunks with attempted Firth correction."""
    return [
        numeric_diagnostic_value(diagnostics.get(field_name, 0))
        for diagnostics in binary_chunk_diagnostics
        if numeric_diagnostic_value(diagnostics.get("firth_candidate_count", 0)) > 0.0
    ]


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """Divide only when the denominator is positive."""
    if denominator <= 0.0:
        return None
    return numerator / denominator


def load_binary_diagnostic_trial_payload(
    *,
    stage_timing_mode: profile_deep_models.ProfileStageTimingMode,
    trial: profile_deep_models.TrialResult,
) -> profile_deep_models.BinaryDiagnosticTrialPayload:
    """Load one trial's stage timing payload with an explicit unavailable reason."""
    if trial.stage_timing_path is None:
        reason = (
            BINARY_DIAGNOSTIC_UNAVAILABLE_EXACT_TIMING_DISABLED
            if stage_timing_mode == profile_deep_models.ProfileStageTimingMode.OFF
            else BINARY_DIAGNOSTIC_UNAVAILABLE_STAGE_TIMING_FILE_MISSING
        )
        return profile_deep_models.BinaryDiagnosticTrialPayload(
            trial_name=trial.name,
            stage_timing_path=None,
            unavailable_reason=reason,
            payload=None,
        )
    stage_timing_path = Path(trial.stage_timing_path)
    if not stage_timing_path.exists():
        return profile_deep_models.BinaryDiagnosticTrialPayload(
            trial_name=trial.name,
            stage_timing_path=trial.stage_timing_path,
            unavailable_reason=BINARY_DIAGNOSTIC_UNAVAILABLE_STAGE_TIMING_FILE_MISSING,
            payload=None,
        )
    try:
        raw_payload = json.loads(stage_timing_path.read_text(encoding="utf-8"))
    except OSError:
        return profile_deep_models.BinaryDiagnosticTrialPayload(
            trial_name=trial.name,
            stage_timing_path=trial.stage_timing_path,
            unavailable_reason=BINARY_DIAGNOSTIC_UNAVAILABLE_STAGE_TIMING_FILE_INVALID,
            payload=None,
        )
    except json.JSONDecodeError:
        return profile_deep_models.BinaryDiagnosticTrialPayload(
            trial_name=trial.name,
            stage_timing_path=trial.stage_timing_path,
            unavailable_reason=BINARY_DIAGNOSTIC_UNAVAILABLE_STAGE_TIMING_FILE_INVALID,
            payload=None,
        )
    if not isinstance(raw_payload, dict):
        return profile_deep_models.BinaryDiagnosticTrialPayload(
            trial_name=trial.name,
            stage_timing_path=trial.stage_timing_path,
            unavailable_reason=BINARY_DIAGNOSTIC_UNAVAILABLE_STAGE_TIMING_FILE_INVALID,
            payload=None,
        )
    return profile_deep_models.BinaryDiagnosticTrialPayload(
        trial_name=trial.name,
        stage_timing_path=trial.stage_timing_path,
        unavailable_reason=None,
        payload=typing.cast("dict[str, typing.Any]", raw_payload),
    )


def extract_binary_chunk_diagnostics(
    loaded_payload: profile_deep_models.BinaryDiagnosticTrialPayload,
) -> list[dict[str, typing.Any]] | None:
    """Extract valid binary chunk diagnostic mappings from one loaded payload."""
    if loaded_payload.payload is None:
        return None
    raw_binary_chunk_diagnostics = loaded_payload.payload.get("binary_chunk_diagnostics")
    if raw_binary_chunk_diagnostics is None or not isinstance(raw_binary_chunk_diagnostics, list):
        return None
    binary_chunk_diagnostics: list[dict[str, typing.Any]] = []
    for raw_chunk_diagnostics in raw_binary_chunk_diagnostics:
        if not isinstance(raw_chunk_diagnostics, dict):
            return None
        binary_chunk_diagnostics.append(typing.cast("dict[str, typing.Any]", raw_chunk_diagnostics))
    return binary_chunk_diagnostics


def summarize_stage_mapping(
    stage_timing_payloads: list[dict[str, typing.Any]],
    field_name: str,
) -> dict[str, float]:
    """Sum numeric values from a stage timing mapping field across trials."""
    summary: dict[str, float] = {}
    for stage_timing_payload in stage_timing_payloads:
        raw_mapping = stage_timing_payload.get(field_name)
        if not isinstance(raw_mapping, dict):
            continue
        for raw_key, raw_value in raw_mapping.items():
            numeric_value = optional_numeric_value(raw_value)
            if numeric_value is None:
                continue
            key = str(raw_key)
            summary[key] = summary.get(key, 0.0) + numeric_value
    return summary


def summarize_null_logistic_diagnostics(stage_timing_payloads: list[dict[str, typing.Any]]) -> dict[str, typing.Any]:
    """Aggregate null logistic diagnostics across available binary trials."""
    diagnostics: list[dict[str, typing.Any]] = []
    for stage_timing_payload in stage_timing_payloads:
        raw_diagnostics = stage_timing_payload.get("null_logistic_diagnostics")
        if not isinstance(raw_diagnostics, list):
            continue
        for raw_diagnostic in raw_diagnostics:
            if isinstance(raw_diagnostic, dict):
                diagnostics.append(typing.cast("dict[str, typing.Any]", raw_diagnostic))
    iteration_counts = [
        numeric_diagnostic_value(diagnostic.get("iteration_count", diagnostic.get("null_logistic_iteration_count", 0)))
        for diagnostic in diagnostics
    ]
    firth_iteration_counts = [
        numeric_diagnostic_value(diagnostic.get("firth_iteration_count", 0)) for diagnostic in diagnostics
    ]
    correction_method_counts: dict[str, int] = {}
    convergence_reason_counts: dict[str, int] = {}
    converged_count = 0
    for diagnostic in diagnostics:
        if numeric_diagnostic_value(diagnostic.get("converged", 0)) > 0.0:
            converged_count += 1
        correction_method = diagnostic.get("correction_method")
        if correction_method is not None:
            correction_method_key = str(correction_method)
            correction_method_counts[correction_method_key] = correction_method_counts.get(correction_method_key, 0) + 1
        convergence_reason_code = diagnostic.get("firth_convergence_reason_code")
        if convergence_reason_code is not None:
            convergence_reason_key = str(convergence_reason_code)
            convergence_reason_counts[convergence_reason_key] = (
                convergence_reason_counts.get(convergence_reason_key, 0) + 1
            )
    return {
        "chromosome_count": len(diagnostics),
        "converged_count": converged_count,
        "failed_count": max(len(diagnostics) - converged_count, 0),
        "iteration_counts": summarize_numeric_values(iteration_counts),
        "firth_iteration_counts": summarize_numeric_values(firth_iteration_counts),
        "correction_method_counts": correction_method_counts,
        "firth_convergence_reason_code_counts": convergence_reason_counts,
    }


def summarize_numeric_values(values: list[float]) -> dict[str, float | int | None]:
    """Summarize a numeric vector for JSON output."""
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "mean": None,
            "median": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def summarize_queue_backpressure(stage_timing_payloads: list[dict[str, typing.Any]]) -> list[dict[str, typing.Any]]:
    """Aggregate queue/backpressure observations across available trials."""
    summaries: dict[str, dict[str, typing.Any]] = {}
    for stage_timing_payload in stage_timing_payloads:
        raw_queue_backpressure = stage_timing_payload.get("queue_backpressure")
        if not isinstance(raw_queue_backpressure, list):
            continue
        for raw_queue_snapshot in raw_queue_backpressure:
            if not isinstance(raw_queue_snapshot, dict):
                continue
            queue_name = str(raw_queue_snapshot.get("queue_name", ""))
            operation_name = str(raw_queue_snapshot.get("operation_name", ""))
            summary_key = f"{queue_name}:{operation_name}"
            summary = summaries.setdefault(
                summary_key,
                {
                    "queue_name": queue_name,
                    "operation_name": operation_name,
                    "observation_count": 0,
                    "max_depth": 0,
                    "max_capacity": 0,
                    "total_elapsed_seconds": 0.0,
                    "total_blocked_seconds": 0.0,
                },
            )
            summary["observation_count"] = int(summary["observation_count"]) + int(
                numeric_diagnostic_value(raw_queue_snapshot.get("observation_count", 0))
            )
            summary["max_depth"] = max(
                int(summary["max_depth"]),
                int(numeric_diagnostic_value(raw_queue_snapshot.get("max_depth", 0))),
            )
            summary["max_capacity"] = max(
                int(summary["max_capacity"]),
                int(numeric_diagnostic_value(raw_queue_snapshot.get("max_capacity", 0))),
            )
            summary["total_elapsed_seconds"] = float(summary["total_elapsed_seconds"]) + numeric_diagnostic_value(
                raw_queue_snapshot.get("total_elapsed_seconds", 0.0)
            )
            summary["total_blocked_seconds"] = float(summary["total_blocked_seconds"]) + numeric_diagnostic_value(
                raw_queue_snapshot.get("total_blocked_seconds", 0.0)
            )
    rows = list(summaries.values())
    for row in rows:
        row["blocked_fraction"] = safe_ratio(
            float(row["total_blocked_seconds"]),
            float(row["total_elapsed_seconds"]),
        )
    return sorted(rows, key=lambda row: float(row["total_blocked_seconds"]), reverse=True)


def collect_chunk_identities(stage_timing_payload: dict[str, typing.Any]) -> list[dict[str, typing.Any]]:
    """Collect first-seen chunk identities from exact chunk stage timings."""
    raw_chunk_stage_timings = stage_timing_payload.get("chunk_stage_timings")
    if not isinstance(raw_chunk_stage_timings, list):
        return []
    identities: list[dict[str, typing.Any]] = []
    seen_chunk_identifiers: set[str] = set()
    for raw_chunk_stage_timing in raw_chunk_stage_timings:
        if not isinstance(raw_chunk_stage_timing, dict):
            continue
        chunk_identifier = str(raw_chunk_stage_timing.get("chunk_identifier", len(identities)))
        if chunk_identifier in seen_chunk_identifiers:
            continue
        seen_chunk_identifiers.add(chunk_identifier)
        identities.append(
            {
                "chunk_identifier": raw_chunk_stage_timing.get("chunk_identifier"),
                "chromosome": raw_chunk_stage_timing.get("chromosome"),
                "variant_start_index": raw_chunk_stage_timing.get("variant_start_index"),
                "variant_stop_index": raw_chunk_stage_timing.get("variant_stop_index"),
                "variant_count": raw_chunk_stage_timing.get("variant_count"),
            }
        )
    return identities


def build_binary_chunk_outliers(
    available_trials: list[dict[str, typing.Any]],
) -> list[dict[str, typing.Any]]:
    """Build a compact top-N list of per-chunk binary correction outliers."""
    outliers: list[dict[str, typing.Any]] = []
    for available_trial in available_trials:
        trial = typing.cast("profile_deep_models.BinaryDiagnosticTrialPayload", available_trial["trial"])
        stage_timing_payload = typing.cast("dict[str, typing.Any]", available_trial["payload"])
        binary_chunk_diagnostics = typing.cast(
            "list[dict[str, typing.Any]]",
            available_trial["binary_chunk_diagnostics"],
        )
        chunk_identities = collect_chunk_identities(stage_timing_payload)
        for chunk_index, diagnostics in enumerate(binary_chunk_diagnostics):
            firth_candidate_count = sum_binary_diagnostic_count([diagnostics], "firth_candidate_count")
            firth_failed_count = sum_binary_diagnostic_count([diagnostics], "firth_failed_count")
            score_test_candidate_count = sum_binary_diagnostic_count([diagnostics], "score_test_candidate_count")
            if firth_candidate_count == 0 and score_test_candidate_count == 0:
                continue
            outlier = {
                "trial_name": trial.trial_name,
                "chunk_index": chunk_index,
                "rank_fields": {
                    "firth_candidate_count": firth_candidate_count,
                    "firth_failed_count": firth_failed_count,
                    "firth_iteration_max": numeric_diagnostic_value(diagnostics.get("firth_iteration_max", 0)),
                    "score_test_candidate_count": score_test_candidate_count,
                },
                "diagnostics": {
                    "score_test_candidate_count": score_test_candidate_count,
                    "firth_candidate_count": firth_candidate_count,
                    "firth_converged_count": sum_binary_diagnostic_count([diagnostics], "firth_converged_count"),
                    "firth_failed_count": firth_failed_count,
                    "firth_iteration_min": numeric_diagnostic_value(diagnostics.get("firth_iteration_min", 0)),
                    "firth_iteration_median": numeric_diagnostic_value(diagnostics.get("firth_iteration_median", 0)),
                    "firth_iteration_max": numeric_diagnostic_value(diagnostics.get("firth_iteration_max", 0)),
                    "sparse_correction_count": sum_binary_diagnostic_count([diagnostics], "sparse_correction_count"),
                    "dense_correction_count": sum_binary_diagnostic_count([diagnostics], "dense_correction_count"),
                },
                "chunk_identity": chunk_identities[chunk_index] if chunk_index < len(chunk_identities) else None,
            }
            outliers.append(outlier)
    return sorted(
        outliers,
        key=lambda outlier: (
            int(typing.cast("dict[str, typing.Any]", outlier["rank_fields"])["firth_candidate_count"]),
            int(typing.cast("dict[str, typing.Any]", outlier["rank_fields"])["firth_failed_count"]),
            float(typing.cast("dict[str, typing.Any]", outlier["rank_fields"])["firth_iteration_max"]),
            int(typing.cast("dict[str, typing.Any]", outlier["rank_fields"])["score_test_candidate_count"]),
        ),
        reverse=True,
    )[:BINARY_CHUNK_OUTLIER_LIMIT]


def unavailable_binary_correction_diagnostics(
    *,
    aggregate_result: profile_deep_models.AggregateResult,
    stage_timing_mode: profile_deep_models.ProfileStageTimingMode,
    reason: str,
    unavailable_trials: list[dict[str, str | None]],
) -> dict[str, typing.Any]:
    """Build an explicit unavailable binary correction diagnostic payload."""
    return {
        "available": False,
        "reason": reason,
        "aggregate_name": aggregate_result.name,
        "trait_type": aggregate_result.trait_type,
        "device": aggregate_result.device,
        "status": aggregate_result.status,
        "stage_timing_mode": stage_timing_mode.value,
        "trial_count": aggregate_result.trial_count,
        "available_trial_count": 0,
        "unavailable_trials": unavailable_trials,
        "chunk_count": None,
        "candidate_counts": {
            "score_test": None,
            "firth": None,
        },
        "correction_outcome_counts": {
            "corrected": None,
            "failed": None,
            "score_test_or_uncorrected": None,
        },
        "failure_code_counts": {
            "none": None,
            "numerical": None,
            "max_iterations": None,
            "invalid_statistic": None,
            "step_halving": None,
        },
        "firth_iteration_counts": {
            "active_chunk_count": None,
            "minimum": None,
            "median_per_chunk_mean": None,
            "maximum": None,
        },
        "correction_branch_counts": {
            "pseudo_firth": None,
            "newton_raphson_zero_start": None,
            "newton_raphson_warm_start": None,
        },
        "correction_attempt_counts": {
            "pseudo_firth": None,
            "newton_raphson_zero_start": None,
            "newton_raphson_warm_start": None,
        },
        "correction_input_counts": {
            "sparse": None,
            "dense": None,
        },
        "fallback_density": {
            "firth_candidates_per_output_row": None,
            "firth_candidates_per_score_test_candidate": None,
        },
        "stage_counts": None,
        "stage_totals_seconds": None,
        "null_logistic": None,
        "queue_backpressure": None,
        "chunk_outliers": [],
    }


def build_binary_correction_diagnostics_for_aggregate(
    *,
    aggregate_result: profile_deep_models.AggregateResult,
    stage_timing_mode: profile_deep_models.ProfileStageTimingMode,
) -> dict[str, typing.Any]:
    """Build aggregate binary correction diagnostics for one g binary result."""
    diagnostic_source_trials = aggregate_result.diagnostic_trials or aggregate_result.trials
    loaded_payloads = [
        load_binary_diagnostic_trial_payload(stage_timing_mode=stage_timing_mode, trial=trial)
        for trial in diagnostic_source_trials
        if trial.status == "success"
    ]
    unavailable_trials: list[dict[str, str | None]] = []
    available_trials: list[dict[str, typing.Any]] = []
    for loaded_payload in loaded_payloads:
        if loaded_payload.unavailable_reason is not None:
            unavailable_trials.append(
                {
                    "trial_name": loaded_payload.trial_name,
                    "stage_timing_path": loaded_payload.stage_timing_path,
                    "reason": loaded_payload.unavailable_reason,
                }
            )
            continue
        binary_chunk_diagnostics = extract_binary_chunk_diagnostics(loaded_payload)
        if binary_chunk_diagnostics is None:
            reason = BINARY_DIAGNOSTIC_UNAVAILABLE_BINARY_DIAGNOSTICS_MISSING
            if (
                loaded_payload.payload is not None
                and "binary_chunk_diagnostics" in loaded_payload.payload
                and not isinstance(loaded_payload.payload["binary_chunk_diagnostics"], list)
            ):
                reason = BINARY_DIAGNOSTIC_UNAVAILABLE_BINARY_DIAGNOSTICS_INVALID
            unavailable_trials.append(
                {
                    "trial_name": loaded_payload.trial_name,
                    "stage_timing_path": loaded_payload.stage_timing_path,
                    "reason": reason,
                }
            )
            continue
        available_trials.append(
            {
                "trial": loaded_payload,
                "payload": typing.cast("dict[str, typing.Any]", loaded_payload.payload),
                "binary_chunk_diagnostics": binary_chunk_diagnostics,
            }
        )
    if not available_trials:
        reason = BINARY_DIAGNOSTIC_UNAVAILABLE_STAGE_TIMING_FILE_MISSING
        if unavailable_trials:
            reason = str(unavailable_trials[0]["reason"])
        return unavailable_binary_correction_diagnostics(
            aggregate_result=aggregate_result,
            stage_timing_mode=stage_timing_mode,
            reason=reason,
            unavailable_trials=unavailable_trials,
        )
    all_binary_chunk_diagnostics: list[dict[str, typing.Any]] = []
    for available_trial in available_trials:
        all_binary_chunk_diagnostics.extend(
            typing.cast("list[dict[str, typing.Any]]", available_trial["binary_chunk_diagnostics"])
        )
    diagnostic_counts = {
        field_name: sum_binary_diagnostic_count(all_binary_chunk_diagnostics, field_name)
        for field_name in BINARY_DIAGNOSTIC_COUNT_FIELDS
    }
    non_none_failure_count = (
        diagnostic_counts["firth_numerical_failure_count"]
        + diagnostic_counts["firth_max_iteration_failure_count"]
        + diagnostic_counts["firth_invalid_statistic_failure_count"]
        + diagnostic_counts["firth_step_halving_failure_count"]
    )
    stage_timing_payloads = [
        typing.cast("dict[str, typing.Any]", available_trial["payload"]) for available_trial in available_trials
    ]
    output_row_count_by_trial = {
        trial.name: trial.output_row_count
        for trial in diagnostic_source_trials
        if trial.status == "success" and trial.output_row_count is not None
    }
    available_output_row_count = sum(
        output_row_count_by_trial.get(
            typing.cast("profile_deep_models.BinaryDiagnosticTrialPayload", available_trial["trial"]).trial_name,
            0,
        )
        or 0
        for available_trial in available_trials
    )
    minimum_iteration_values = active_firth_iteration_values(all_binary_chunk_diagnostics, "firth_iteration_min")
    maximum_iteration_values = active_firth_iteration_values(all_binary_chunk_diagnostics, "firth_iteration_max")
    score_test_candidate_count = diagnostic_counts["score_test_candidate_count"]
    firth_candidate_count = diagnostic_counts["firth_candidate_count"]
    firth_converged_count = diagnostic_counts["firth_converged_count"]
    firth_failed_count = diagnostic_counts["firth_failed_count"]
    return {
        "available": True,
        "reason": None,
        "aggregate_name": aggregate_result.name,
        "trait_type": aggregate_result.trait_type,
        "device": aggregate_result.device,
        "status": aggregate_result.status,
        "stage_timing_mode": stage_timing_mode.value,
        "trial_count": aggregate_result.trial_count,
        "available_trial_count": len(available_trials),
        "unavailable_trials": unavailable_trials,
        "chunk_count": len(all_binary_chunk_diagnostics),
        "candidate_counts": {
            "score_test": score_test_candidate_count,
            "firth": firth_candidate_count,
            "score_test_per_available_trial_mean": score_test_candidate_count / len(available_trials),
            "firth_per_available_trial_mean": firth_candidate_count / len(available_trials),
        },
        "correction_outcome_counts": {
            "corrected": firth_converged_count,
            "failed": firth_failed_count,
            "score_test_or_uncorrected": max(
                score_test_candidate_count - firth_converged_count - firth_failed_count, 0
            ),
        },
        "failure_code_counts": {
            "none": max(firth_candidate_count - non_none_failure_count, 0),
            "numerical": diagnostic_counts["firth_numerical_failure_count"],
            "max_iterations": diagnostic_counts["firth_max_iteration_failure_count"],
            "invalid_statistic": diagnostic_counts["firth_invalid_statistic_failure_count"],
            "step_halving": diagnostic_counts["firth_step_halving_failure_count"],
        },
        "firth_iteration_counts": {
            "active_chunk_count": len(minimum_iteration_values),
            "minimum": min(minimum_iteration_values) if minimum_iteration_values else 0,
            "median_per_chunk_mean": mean_binary_diagnostic_value(
                all_binary_chunk_diagnostics,
                "firth_iteration_median",
            ),
            "maximum": max(maximum_iteration_values) if maximum_iteration_values else 0,
        },
        "correction_branch_counts": {
            "pseudo_firth": diagnostic_counts["pseudo_firth_success_count"],
            "newton_raphson_zero_start": diagnostic_counts["nr_zero_start_success_count"],
            "newton_raphson_warm_start": diagnostic_counts["nr_warm_start_success_count"],
        },
        "correction_attempt_counts": {
            "pseudo_firth": diagnostic_counts["pseudo_firth_attempt_count"],
            "newton_raphson_zero_start": diagnostic_counts["nr_zero_start_attempt_count"],
            "newton_raphson_warm_start": diagnostic_counts["nr_warm_start_attempt_count"],
        },
        "correction_input_counts": {
            "sparse": diagnostic_counts["sparse_correction_count"],
            "dense": diagnostic_counts["dense_correction_count"],
        },
        "fallback_density": {
            "firth_candidates_per_output_row": safe_ratio(
                float(firth_candidate_count), float(available_output_row_count)
            ),
            "firth_candidates_per_score_test_candidate": safe_ratio(
                float(firth_candidate_count),
                float(score_test_candidate_count),
            ),
        },
        "stage_counts": summarize_stage_mapping(stage_timing_payloads, "stage_counts"),
        "stage_totals_seconds": summarize_stage_mapping(stage_timing_payloads, "stage_totals_seconds"),
        "null_logistic": summarize_null_logistic_diagnostics(stage_timing_payloads),
        "queue_backpressure": summarize_queue_backpressure(stage_timing_payloads),
        "chunk_outliers": build_binary_chunk_outliers(available_trials),
    }


def build_binary_correction_diagnostics(
    *,
    headline_results: list[profile_deep_models.AggregateResult],
    finalist_results_by_key: dict[str, list[profile_deep_models.AggregateResult]],
    stage_timing_mode: profile_deep_models.ProfileStageTimingMode,
) -> dict[str, typing.Any]:
    """Build binary correction diagnostics for headline and finalist g runs."""
    headline_diagnostics = {
        aggregate_result.name: build_binary_correction_diagnostics_for_aggregate(
            aggregate_result=aggregate_result,
            stage_timing_mode=stage_timing_mode,
        )
        for aggregate_result in headline_results
        if aggregate_result.implementation == "g" and aggregate_result.trait_type == "binary"
    }
    finalist_diagnostics: dict[str, dict[str, typing.Any]] = {}
    for winner_key, finalist_results in sorted(finalist_results_by_key.items()):
        if not winner_key.startswith("binary_"):
            continue
        finalist_diagnostics[winner_key] = {
            aggregate_result.name: build_binary_correction_diagnostics_for_aggregate(
                aggregate_result=aggregate_result,
                stage_timing_mode=stage_timing_mode,
            )
            for aggregate_result in finalist_results
            if aggregate_result.implementation == "g" and aggregate_result.trait_type == "binary"
        }
    return {
        "stage_timing_mode": stage_timing_mode.value,
        "headline": headline_diagnostics,
        "finalists": finalist_diagnostics,
    }
