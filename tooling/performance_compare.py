"""Compare repository benchmark JSON summaries."""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import typing

if typing.TYPE_CHECKING:
    from pathlib import Path


class PerformanceComparisonError(ValueError):
    """Raised when benchmark summaries cannot be compared."""


class MetricCategory(enum.StrEnum):
    """Supported benchmark metric categories."""

    SPEED = "speed"
    MEMORY = "memory"
    NUMERICAL = "numerical"


@dataclasses.dataclass(frozen=True)
class MetricRecord:
    """One numeric benchmark metric extracted from a summary.

    Attributes:
        name: Stable metric name.
        category: Metric category used for reporting.
        value: Numeric metric value.
        unit: Optional metric unit.

    """

    name: str
    category: MetricCategory
    value: float
    unit: str | None = None


@dataclasses.dataclass(frozen=True)
class MetricComparison:
    """Comparison of one metric between two summaries.

    Attributes:
        name: Stable metric name.
        category: Metric category.
        baseline_value: Value from the baseline summary.
        new_value: Value from the new summary.
        delta: New value minus baseline value.
        ratio: New value divided by baseline value when the baseline is nonzero.
        unit: Optional metric unit.

    """

    name: str
    category: MetricCategory
    baseline_value: float
    new_value: float
    delta: float
    ratio: float | None
    unit: str | None = None


@dataclasses.dataclass(frozen=True)
class ComparisonReport:
    """Complete comparison between two benchmark summaries.

    Attributes:
        baseline_path: Baseline summary path.
        new_path: New summary path.
        comparisons: Common metric comparisons.

    """

    baseline_path: Path
    new_path: Path
    comparisons: list[MetricComparison]


MetricMap = dict[str, MetricRecord]

SPEED_FIELD_NAMES = frozenset(
    {
        "wall_time_seconds",
        "mean_seconds",
        "median_seconds",
    }
)
MEMORY_FIELD_NAMES = frozenset(
    {
        "chunk_bytes",
        "final_parquet_bytes",
        "output_total_bytes",
    }
)
NUMERICAL_FIELD_NAMES = frozenset(
    {
        "checksum",
        "output_row_count",
        "output_file_count",
        "committed_chunk_count",
        "chunk_file_count",
        "info_non_null_count",
    }
)
CATEGORY_SORT_ORDER = {
    MetricCategory.SPEED: 0,
    MetricCategory.MEMORY: 1,
    MetricCategory.NUMERICAL: 2,
}


def load_json_summary(path: Path) -> dict[str, typing.Any]:
    """Load a benchmark JSON summary object.

    Args:
        path: JSON summary path.

    Returns:
        Top-level JSON object.

    Raises:
        PerformanceComparisonError: If the file is unreadable, invalid JSON, or
            not a JSON object.

    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as error:
        message = f"Could not read benchmark summary `{path}`: {error}"
        raise PerformanceComparisonError(message) from error
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as error:
        message = (
            f"Benchmark summary `{path}` is not valid JSON: {error.msg} at line {error.lineno} column {error.colno}."
        )
        raise PerformanceComparisonError(message) from error
    if not isinstance(payload, dict):
        message = f"Benchmark summary `{path}` must be a JSON object."
        raise PerformanceComparisonError(message)
    return typing.cast("dict[str, typing.Any]", payload)


def infer_metric_category(metric_name: str) -> MetricCategory:
    """Infer the metric category from a metric name.

    Args:
        metric_name: Metric name.

    Returns:
        Inferred category.

    """
    normalized_name = metric_name.lower()
    if any(token in normalized_name for token in ("second", "duration", "elapsed", "wall_time", "stage.")):
        return MetricCategory.SPEED
    if any(token in normalized_name for token in ("byte", "memory", "rss", "heap", "allocated")):
        return MetricCategory.MEMORY
    return MetricCategory.NUMERICAL


def default_unit(metric_name: str, category: MetricCategory) -> str | None:
    """Return a default display unit for a metric.

    Args:
        metric_name: Metric name.
        category: Metric category.

    Returns:
        Unit string when one is known.

    """
    normalized_name = metric_name.lower()
    if category == MetricCategory.SPEED:
        return "seconds"
    if category == MetricCategory.MEMORY and "byte" in normalized_name:
        return "bytes"
    return None


def parse_metric_category(raw_category: typing.Any, metric_name: str) -> MetricCategory:
    """Parse a user-provided metric category.

    Args:
        raw_category: Raw category value.
        metric_name: Metric name used for error context.

    Returns:
        Parsed category.

    Raises:
        PerformanceComparisonError: If the category is invalid.

    """
    if raw_category is None:
        return infer_metric_category(metric_name)
    if not isinstance(raw_category, str):
        message = f"Metric `{metric_name}` category must be a string."
        raise PerformanceComparisonError(message)
    try:
        return MetricCategory(raw_category)
    except ValueError as error:
        accepted_values = ", ".join(category.value for category in MetricCategory)
        message = f"Metric `{metric_name}` category must be one of: {accepted_values}."
        raise PerformanceComparisonError(message) from error


def coerce_numeric_metric(raw_value: typing.Any, metric_name: str) -> float | None:
    """Convert a raw JSON metric value into a finite float.

    Args:
        raw_value: Raw JSON value.
        metric_name: Metric name used for error context.

    Returns:
        Numeric value, or None when the metric is explicitly unavailable.

    Raises:
        PerformanceComparisonError: If a present metric is not finite numeric.

    """
    if raw_value is None:
        return None
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        message = f"Metric `{metric_name}` must be numeric or null."
        raise PerformanceComparisonError(message)
    value = float(raw_value)
    if not math.isfinite(value):
        message = f"Metric `{metric_name}` must be finite."
        raise PerformanceComparisonError(message)
    return value


def add_metric(
    metrics: MetricMap,
    metric_name: str,
    raw_value: typing.Any,
    *,
    category: MetricCategory | None = None,
    unit: str | None = None,
) -> None:
    """Add one metric when its value is present.

    Args:
        metrics: Metric map to update.
        metric_name: Stable metric name.
        raw_value: Raw JSON value.
        category: Optional category override.
        unit: Optional unit override.

    Raises:
        PerformanceComparisonError: If the metric value is malformed.

    """
    value = coerce_numeric_metric(raw_value, metric_name)
    if value is None:
        return
    resolved_category = category or infer_metric_category(metric_name)
    metrics[metric_name] = MetricRecord(
        name=metric_name,
        category=resolved_category,
        value=value,
        unit=unit or default_unit(metric_name, resolved_category),
    )


def mapping_or_error(raw_value: typing.Any, context: str) -> dict[str, typing.Any]:
    """Return a JSON object or raise a formatted comparison error.

    Args:
        raw_value: Raw JSON value.
        context: Human-readable context.

    Returns:
        JSON object.

    Raises:
        PerformanceComparisonError: If the value is not an object.

    """
    if not isinstance(raw_value, dict):
        message = f"{context} must be a JSON object."
        raise PerformanceComparisonError(message)
    return typing.cast("dict[str, typing.Any]", raw_value)


def list_or_error(raw_value: typing.Any, context: str) -> list[typing.Any]:
    """Return a JSON list or raise a formatted comparison error.

    Args:
        raw_value: Raw JSON value.
        context: Human-readable context.

    Returns:
        JSON list.

    Raises:
        PerformanceComparisonError: If the value is not a list.

    """
    if not isinstance(raw_value, list):
        message = f"{context} must be a JSON list."
        raise PerformanceComparisonError(message)
    return typing.cast("list[typing.Any]", raw_value)


def extract_explicit_metrics(payload: dict[str, typing.Any], metrics: MetricMap) -> None:
    """Extract metrics from the generic `metrics` summary schema.

    Args:
        payload: Benchmark summary payload.
        metrics: Metric map to update.

    Raises:
        PerformanceComparisonError: If the metrics object is malformed.

    """
    raw_metrics = payload.get("metrics")
    if raw_metrics is None:
        return
    metric_payloads = mapping_or_error(raw_metrics, "`metrics`")
    for raw_metric_name, raw_metric_value in metric_payloads.items():
        metric_name = str(raw_metric_name)
        if isinstance(raw_metric_value, dict):
            metric_payload = typing.cast("dict[str, typing.Any]", raw_metric_value)
            if "value" not in metric_payload:
                message = f"Metric `{metric_name}` object must contain `value`."
                raise PerformanceComparisonError(message)
            category = parse_metric_category(metric_payload.get("category"), metric_name)
            raw_unit = metric_payload.get("unit")
            unit = str(raw_unit) if raw_unit is not None else None
            add_metric(metrics, metric_name, metric_payload["value"], category=category, unit=unit)
        else:
            add_metric(metrics, metric_name, raw_metric_value)


def extract_headline_metrics(payload: dict[str, typing.Any], metrics: MetricMap) -> None:
    """Extract binary-hot headline timing metrics.

    Args:
        payload: Benchmark summary payload.
        metrics: Metric map to update.

    """
    raw_headline = payload.get("headline")
    if raw_headline is not None:
        headline = mapping_or_error(raw_headline, "`headline`")
        for raw_metric_name, raw_metric_value in headline.items():
            metric_name = f"headline.{raw_metric_name}"
            add_metric(metrics, metric_name, raw_metric_value, category=infer_metric_category(metric_name))

    raw_headline_by_case = payload.get("headline_by_case")
    if raw_headline_by_case is None:
        return
    headline_by_case = mapping_or_error(raw_headline_by_case, "`headline_by_case`")
    for raw_case_name, raw_case_payload in headline_by_case.items():
        case_payload = mapping_or_error(raw_case_payload, f"`headline_by_case.{raw_case_name}`")
        for raw_metric_name, raw_metric_value in case_payload.items():
            metric_name = f"headline_by_case.{raw_case_name}.{raw_metric_name}"
            add_metric(metrics, metric_name, raw_metric_value, category=infer_metric_category(metric_name))


def benchmark_case_name(raw_case_payload: typing.Any, fallback_name: str) -> str:
    """Return a stable benchmark case name.

    Args:
        raw_case_payload: Raw benchmark case object.
        fallback_name: Name used when the case object has no name.

    Returns:
        Stable name.

    """
    if isinstance(raw_case_payload, dict):
        raw_case_name = raw_case_payload.get("name")
        if isinstance(raw_case_name, str) and raw_case_name:
            return raw_case_name
    return fallback_name


def extract_binary_hot_result_metrics(payload: dict[str, typing.Any], metrics: MetricMap) -> None:
    """Extract metrics from binary-hot `results` entries.

    Args:
        payload: Benchmark summary payload.
        metrics: Metric map to update.

    """
    raw_results = payload.get("results")
    if raw_results is None:
        return
    result_payloads = list_or_error(raw_results, "`results`")
    for result_index, raw_result_payload in enumerate(result_payloads):
        result_payload = mapping_or_error(raw_result_payload, f"`results[{result_index}]`")
        if "wall_time_seconds" not in result_payload and "output_metrics" not in result_payload:
            continue
        case_name = benchmark_case_name(result_payload.get("benchmark_case"), f"case{result_index}")
        raw_result_name = result_payload.get("name", result_payload.get("mode", f"result{result_index}"))
        result_name = str(raw_result_name)
        metric_prefix = f"results.{case_name}.{result_name}"
        add_metric(
            metrics,
            f"{metric_prefix}.wall_time_seconds",
            result_payload.get("wall_time_seconds"),
            category=MetricCategory.SPEED,
        )
        raw_output_metrics = result_payload.get("output_metrics")
        if raw_output_metrics is None:
            continue
        output_metrics = mapping_or_error(raw_output_metrics, f"`{metric_prefix}.output_metrics`")
        for field_name in MEMORY_FIELD_NAMES:
            if field_name in output_metrics:
                add_metric(
                    metrics,
                    f"{metric_prefix}.output_metrics.{field_name}",
                    output_metrics[field_name],
                    category=MetricCategory.MEMORY,
                )
        for field_name in NUMERICAL_FIELD_NAMES:
            if field_name in output_metrics:
                add_metric(
                    metrics,
                    f"{metric_prefix}.output_metrics.{field_name}",
                    output_metrics[field_name],
                    category=MetricCategory.NUMERICAL,
                )


def matrix_stage_metric_name(run_name: str, stage_name: str) -> str:
    """Build a stable stage timing metric name.

    Args:
        run_name: Matrix run name.
        stage_name: Stage name from the stage-timing summary.

    Returns:
        Metric name.

    """
    return f"runs.{run_name}.stage.{stage_name}_seconds"


def extract_matrix_run_metrics(payload: dict[str, typing.Any], metrics: MetricMap) -> None:
    """Extract metrics from matrix manifest `runs` entries.

    Args:
        payload: Matrix manifest payload.
        metrics: Metric map to update.

    """
    raw_runs = payload.get("runs")
    if raw_runs is None:
        return
    run_payloads = list_or_error(raw_runs, "`runs`")
    for run_index, raw_run_payload in enumerate(run_payloads):
        run_payload = mapping_or_error(raw_run_payload, f"`runs[{run_index}]`")
        raw_run_name = run_payload.get("name", f"run{run_index}")
        run_name = str(raw_run_name)
        if "wall_time_seconds" in run_payload:
            add_metric(
                metrics,
                f"runs.{run_name}.wall_time_seconds",
                run_payload["wall_time_seconds"],
                category=MetricCategory.SPEED,
            )
        for field_name in MEMORY_FIELD_NAMES:
            if field_name in run_payload:
                add_metric(
                    metrics,
                    f"runs.{run_name}.{field_name}",
                    run_payload[field_name],
                    category=MetricCategory.MEMORY,
                )
        for field_name in NUMERICAL_FIELD_NAMES:
            if field_name in run_payload:
                add_metric(
                    metrics,
                    f"runs.{run_name}.{field_name}",
                    run_payload[field_name],
                    category=MetricCategory.NUMERICAL,
                )
        raw_stage_seconds = run_payload.get("stage_seconds")
        if raw_stage_seconds is None:
            continue
        stage_seconds = mapping_or_error(raw_stage_seconds, f"`runs[{run_index}].stage_seconds`")
        for raw_stage_name, raw_stage_value in stage_seconds.items():
            add_metric(
                metrics,
                matrix_stage_metric_name(run_name, str(raw_stage_name)),
                raw_stage_value,
                category=MetricCategory.SPEED,
            )


def bgen_case_identifier(case_payload: dict[str, typing.Any], case_index: int) -> str:
    """Build a stable BGEN reader case identifier.

    Args:
        case_payload: BGEN reader case payload.
        case_index: Case index used as fallback.

    Returns:
        Case identifier.

    """
    parts = [
        f"case{case_index}",
        f"chunk{case_payload.get('chunk_size', 'unknown')}",
        f"selection{case_payload.get('sample_selection_mode', 'unknown')}",
        f"trusted{str(case_payload.get('trusted_no_missing_diploid', 'unknown')).lower()}",
    ]
    if case_payload.get("decode_tile_variant_count") is not None:
        parts.append(f"tile{case_payload['decode_tile_variant_count']}")
    if case_payload.get("rayon_thread_count") is not None:
        parts.append(f"rayon{case_payload['rayon_thread_count']}")
    return ".".join(str(part) for part in parts)


def extract_bgen_reader_metrics(payload: dict[str, typing.Any], metrics: MetricMap) -> None:
    """Extract metrics from BGEN reader benchmark `cases` entries.

    Args:
        payload: BGEN reader benchmark payload.
        metrics: Metric map to update.

    """
    raw_cases = payload.get("cases")
    if raw_cases is None:
        return
    case_payloads = list_or_error(raw_cases, "`cases`")
    for case_index, raw_case_payload in enumerate(case_payloads):
        case_payload = mapping_or_error(raw_case_payload, f"`cases[{case_index}]`")
        raw_path_results = case_payload.get("path_results")
        if raw_path_results is None:
            continue
        path_results = list_or_error(raw_path_results, f"`cases[{case_index}].path_results`")
        case_identifier = bgen_case_identifier(case_payload, case_index)
        for path_index, raw_path_result in enumerate(path_results):
            path_result = mapping_or_error(raw_path_result, f"`cases[{case_index}].path_results[{path_index}]`")
            raw_path_mode = path_result.get("path_mode", f"path{path_index}")
            path_mode = str(raw_path_mode)
            for field_name in SPEED_FIELD_NAMES:
                if field_name in path_result:
                    add_metric(
                        metrics,
                        f"cases.{case_identifier}.{path_mode}.{field_name}",
                        path_result[field_name],
                        category=MetricCategory.SPEED,
                    )
            if "checksum" in path_result:
                add_metric(
                    metrics,
                    f"cases.{case_identifier}.{path_mode}.checksum",
                    path_result["checksum"],
                    category=MetricCategory.NUMERICAL,
                )


def generic_metric_path(parent_path: str, key: str) -> str:
    """Append one field to a generic metric path.

    Args:
        parent_path: Existing path.
        key: Field name.

    Returns:
        Combined metric path.

    """
    if parent_path:
        return f"{parent_path}.{key}"
    return key


def extract_generic_numeric_metrics(raw_value: typing.Any, metrics: MetricMap, path: str = "") -> None:
    """Extract numeric metrics from otherwise unknown JSON summaries.

    Args:
        raw_value: Raw JSON value.
        metrics: Metric map to update.
        path: Current JSON path.

    """
    if isinstance(raw_value, dict):
        for raw_key, nested_value in raw_value.items():
            extract_generic_numeric_metrics(nested_value, metrics, generic_metric_path(path, str(raw_key)))
        return
    if isinstance(raw_value, list):
        for item_index, nested_value in enumerate(raw_value):
            if isinstance(nested_value, dict):
                extract_generic_numeric_metrics(nested_value, metrics, generic_metric_path(path, f"item{item_index}"))
        return
    if path and raw_value is not None and not isinstance(raw_value, str | bool):
        add_metric(metrics, path, raw_value)


def extract_metrics(payload: dict[str, typing.Any]) -> MetricMap:
    """Extract comparable metrics from a benchmark JSON summary.

    Args:
        payload: Benchmark summary payload.

    Returns:
        Extracted metrics by metric name.

    Raises:
        PerformanceComparisonError: If no comparable metrics are present or a
            recognized metric is malformed.

    """
    metrics: MetricMap = {}
    extract_explicit_metrics(payload, metrics)
    extract_headline_metrics(payload, metrics)
    extract_binary_hot_result_metrics(payload, metrics)
    extract_matrix_run_metrics(payload, metrics)
    extract_bgen_reader_metrics(payload, metrics)
    if not metrics:
        extract_generic_numeric_metrics(payload, metrics)
    if not metrics:
        message = "No comparable numeric metrics were found in the benchmark summary."
        raise PerformanceComparisonError(message)
    return metrics


def compare_metric_maps(baseline_metrics: MetricMap, new_metrics: MetricMap) -> list[MetricComparison]:
    """Compare common metrics from two metric maps.

    Args:
        baseline_metrics: Baseline metrics.
        new_metrics: New metrics.

    Returns:
        Common metric comparisons.

    Raises:
        PerformanceComparisonError: If the summaries have no metrics in common.

    """
    common_metric_names = sorted(set(baseline_metrics).intersection(new_metrics), key=metric_sort_key)
    if not common_metric_names:
        message = "The benchmark summaries have no metric names in common."
        raise PerformanceComparisonError(message)
    comparisons: list[MetricComparison] = []
    for metric_name in common_metric_names:
        baseline_metric = baseline_metrics[metric_name]
        new_metric = new_metrics[metric_name]
        delta = new_metric.value - baseline_metric.value
        ratio = new_metric.value / baseline_metric.value if baseline_metric.value != 0.0 else None
        comparisons.append(
            MetricComparison(
                name=metric_name,
                category=baseline_metric.category,
                baseline_value=baseline_metric.value,
                new_value=new_metric.value,
                delta=delta,
                ratio=ratio,
                unit=baseline_metric.unit or new_metric.unit,
            )
        )
    return comparisons


def compare_summary_paths(baseline_path: Path, new_path: Path) -> ComparisonReport:
    """Compare two benchmark summary paths.

    Args:
        baseline_path: Baseline JSON summary.
        new_path: New JSON summary.

    Returns:
        Comparison report.

    Raises:
        PerformanceComparisonError: If either summary is malformed or not
            comparable.

    """
    baseline_metrics = extract_metrics(load_json_summary(baseline_path))
    new_metrics = extract_metrics(load_json_summary(new_path))
    return ComparisonReport(
        baseline_path=baseline_path,
        new_path=new_path,
        comparisons=compare_metric_maps(baseline_metrics, new_metrics),
    )


def metric_sort_key(metric_name: str) -> tuple[int, str]:
    """Build a stable metric sort key.

    Args:
        metric_name: Metric name.

    Returns:
        Sort key.

    """
    return (CATEGORY_SORT_ORDER[infer_metric_category(metric_name)], metric_name)


def format_float(value: float | None) -> str:
    """Format a float for the comparison table.

    Args:
        value: Value to format.

    Returns:
        Display text.

    """
    if value is None:
        return "-"
    return f"{value:.6g}"


def describe_change(comparison: MetricComparison) -> str:
    """Describe a metric change in category-specific terms.

    Args:
        comparison: Metric comparison.

    Returns:
        Human-readable change summary.

    """
    if comparison.ratio is None:
        return "-"
    if math.isclose(comparison.ratio, 1.0):
        return "same"
    if comparison.category == MetricCategory.SPEED:
        if comparison.ratio < 1.0:
            return f"{format_float(1.0 / comparison.ratio)}x faster"
        return f"{format_float(comparison.ratio)}x slower"
    if comparison.category == MetricCategory.MEMORY:
        if comparison.ratio < 1.0:
            return f"{format_float(1.0 / comparison.ratio)}x lower"
        return f"{format_float(comparison.ratio)}x higher"
    if comparison.delta > 0.0:
        return "up"
    if comparison.delta < 0.0:
        return "down"
    return "same"


def render_comparison_report(report: ComparisonReport) -> str:
    """Render a concise Markdown comparison table.

    Args:
        report: Comparison report.

    Returns:
        Markdown report text.

    """
    lines = [
        f"Compared {len(report.comparisons)} common metrics.",
        f"Baseline: `{report.baseline_path}`",
        f"New: `{report.new_path}`",
        "",
        "| Category | Metric | Baseline | New | Delta | Ratio | Change |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for comparison in report.comparisons:
        unit_suffix = f" ({comparison.unit})" if comparison.unit is not None else ""
        lines.append(
            "| "
            f"{comparison.category.value} | "
            f"`{comparison.name}`{unit_suffix} | "
            f"{format_float(comparison.baseline_value)} | "
            f"{format_float(comparison.new_value)} | "
            f"{format_float(comparison.delta)} | "
            f"{format_float(comparison.ratio)} | "
            f"{describe_change(comparison)} |"
        )
    return "\n".join(lines) + "\n"
