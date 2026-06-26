#!/usr/bin/env python3
"""CLI wrapper for benchmark JSON summary comparisons."""

from __future__ import annotations

import sys
import typing
from dataclasses import dataclass
from pathlib import Path

import hydra

import tooling.performance_compare as performance_compare
from tooling.common import artifact_format as tooling_artifact_format
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat

if typing.TYPE_CHECKING:
    import omegaconf


@dataclass(frozen=True)
class PerformanceCompareArguments:
    """Resolved parameters for benchmark summary comparison.

    Attributes:
        baseline_json: Baseline benchmark JSON summary.
        new_json: New benchmark JSON summary.
        output_dir: Optional artifact output directory.

    """

    baseline_json: Path
    new_json: Path
    output_dir: Path | None


def comparison_judgement(
    comparison: performance_compare.MetricComparison,
) -> tooling_artifact_format.ComparisonJudgement:
    """Classify one performance comparison row."""
    if comparison.ratio is None:
        return tooling_artifact_format.ComparisonJudgement.INCONCLUSIVE
    if comparison.category in {performance_compare.MetricCategory.SPEED, performance_compare.MetricCategory.MEMORY}:
        percent_change = (comparison.ratio - 1.0) * 100.0
        if percent_change <= -2.0:
            return tooling_artifact_format.ComparisonJudgement.IMPROVEMENT
        if percent_change >= 2.0:
            return tooling_artifact_format.ComparisonJudgement.REGRESSION
        return tooling_artifact_format.ComparisonJudgement.NEUTRAL
    if comparison.delta == 0.0:
        return tooling_artifact_format.ComparisonJudgement.NEUTRAL
    return tooling_artifact_format.ComparisonJudgement.REGRESSION


def comparison_rows(report: performance_compare.ComparisonReport) -> list[dict[str, object]]:
    """Build standard comparison rows."""
    rows: list[dict[str, object]] = []
    for comparison in report.comparisons:
        percent_change = None if comparison.ratio is None else (comparison.ratio - 1.0) * 100.0
        higher_is_better = None
        if comparison.category in {performance_compare.MetricCategory.SPEED, performance_compare.MetricCategory.MEMORY}:
            higher_is_better = False
        rows.append(
            {
                "metric_name": comparison.name,
                "case_id": None,
                "dimensions": {"category": comparison.category.value},
                "baseline_value": comparison.baseline_value,
                "current_value": comparison.new_value,
                "unit": comparison.unit,
                "delta": comparison.delta,
                "ratio": comparison.ratio,
                "percent_change": percent_change,
                "higher_is_better": higher_is_better,
                "judgement": comparison_judgement(comparison).value,
            }
        )
    return rows


def write_standard_comparison_artifacts(
    *,
    arguments: PerformanceCompareArguments,
    report: performance_compare.ComparisonReport,
    markdown_report: str,
    hydra_config: omegaconf.DictConfig | None = None,
) -> None:
    """Write Tooling Artifact Format v1 comparison outputs."""
    if arguments.output_dir is None:
        return
    producer = tooling_artifact_format.build_producer(
        tool_name="performance_compare",
        repository_root=Path.cwd(),
    )
    run = tooling_artifact_format.build_run_identity(
        tool_name="performance_compare",
        output_directory=arguments.output_dir,
        status=tooling_artifact_format.ToolArtifactStatus.SUCCESS,
    )
    context_snapshot = tooling_artifact_format.build_context_snapshot(
        output_directory=arguments.output_dir,
        repository_root=Path.cwd(),
    )
    rows = comparison_rows(report)
    regression_count = sum(
        1 for row in rows if row["judgement"] == tooling_artifact_format.ComparisonJudgement.REGRESSION.value
    )
    improvement_count = sum(
        1 for row in rows if row["judgement"] == tooling_artifact_format.ComparisonJudgement.IMPROVEMENT.value
    )
    neutral_count = sum(
        1 for row in rows if row["judgement"] == tooling_artifact_format.ComparisonJudgement.NEUTRAL.value
    )
    comparison_report = tooling_artifact_format.ComparisonReport(
        schema_name="g.tooling.comparison",
        schema_version=tooling_artifact_format.SCHEMA_VERSION,
        producer=producer,
        run=run,
        baseline={
            "label": "baseline",
            "report_path": str(arguments.baseline_json),
            "git_head": None,
        },
        current={
            "label": "current",
            "report_path": str(arguments.new_json),
            "git_head": producer.git_head,
        },
        thresholds=[
            {
                "metric_name": "wall_time_seconds",
                "max_regression_percent": 2.0,
                "scope": {},
            }
        ],
        comparisons=rows,
        summary={
            "status": run.status.value,
            "regression_count": regression_count,
            "improvement_count": improvement_count,
            "neutral_count": neutral_count,
        },
    )
    envelope = tooling_artifact_format.build_report_envelope(
        producer=producer,
        run=run,
        context=context_snapshot,
        title="Performance Comparison",
        configuration={
            "baseline_json": str(arguments.baseline_json),
            "new_json": str(arguments.new_json),
            "output_dir": str(arguments.output_dir),
        },
        summary={
            "headline": f"Compared {len(report.comparisons)} common metrics.",
            "regression_count": regression_count,
            "improvement_count": improvement_count,
            "neutral_count": neutral_count,
        },
        comparisons=rows,
    )
    tooling_artifact_format.write_standard_artifact_bundle(
        output_directory=arguments.output_dir,
        report=envelope,
        events=[
            tooling_artifact_format.build_tool_event(
                tool_name="performance_compare",
                run_id=run.run_id,
                phase="comparison",
                event="comparison_completed",
                message="Performance comparison completed.",
                fields={
                    "metric_count": len(report.comparisons),
                    "regression_count": regression_count,
                },
            )
        ],
        input_files=[
            tooling_artifact_format.build_input_file_record(path=arguments.baseline_json, kind="baseline_report"),
            tooling_artifact_format.build_input_file_record(path=arguments.new_json, kind="current_report"),
        ],
        summary_markdown=markdown_report,
        comparisons=comparison_report,
        hydra_config=hydra_config,
        tool_payload={
            "baseline_json": str(arguments.baseline_json),
            "new_json": str(arguments.new_json),
            "output_dir": str(arguments.output_dir),
        },
    )


def run_tool(arguments: PerformanceCompareArguments, hydra_config: omegaconf.DictConfig | None = None) -> None:
    """Run the benchmark summary comparison CLI."""
    try:
        report = performance_compare.compare_summary_paths(arguments.baseline_json, arguments.new_json)
    except performance_compare.PerformanceComparisonError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    markdown_report = performance_compare.render_comparison_report(report)
    write_standard_comparison_artifacts(
        arguments=arguments,
        report=report,
        markdown_report=markdown_report,
        hydra_config=hydra_config,
    )
    print(markdown_report, end="")


def required_path(tool_values: dict[str, typing.Any], key: str) -> Path:
    """Return a required path from a Hydra tool config."""
    path = tooling_hydra_arguments.path_or_none(tool_values[key])
    if path is None:
        message = f"tool.{key} is required."
        raise ValueError(message)
    return path


def build_arguments_from_config(config: omegaconf.DictConfig) -> PerformanceCompareArguments:
    """Resolve benchmark comparison parameters from Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return PerformanceCompareArguments(
        baseline_json=required_path(tool_values, "baseline_json"),
        new_json=required_path(tool_values, "new_json"),
        output_dir=tooling_hydra_arguments.path_or_none(tool_values.get("output_dir")),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="performance_compare")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run benchmark summary comparison from Hydra configuration."""
    run_tool(build_arguments_from_config(config), hydra_config=config)


def main() -> None:
    """Run benchmark summary comparison from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
