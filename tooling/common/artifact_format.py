"""Shared Tooling Artifact Format v1 models and builders."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import os
import platform
import shlex
import subprocess
import time
import typing
import uuid
from dataclasses import dataclass
from pathlib import Path

from tooling.common import reports as tooling_reports

if typing.TYPE_CHECKING:
    import collections.abc


SCHEMA_VERSION = 1
REPOSITORY_NAME = "kirilledition/g"


class ToolArtifactStatus(enum.StrEnum):
    """Standard status values for tooling artifacts."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"
    DRY_RUN = "dry_run"
    INTERRUPTED = "interrupted"
    TIMED_OUT = "timed_out"
    INVALID = "invalid"


class PathType(enum.StrEnum):
    """Path origin used in artifact records."""

    ARTIFACT_RELATIVE = "artifact_relative"
    ABSOLUTE_INPUT = "absolute_input"


class FingerprintPolicy(enum.StrEnum):
    """Input-file fingerprint policy."""

    SHA256 = "sha256"
    METADATA_ONLY = "metadata_only"
    MISSING = "missing"


class MetricUnit(enum.StrEnum):
    """Stable metric unit tokens."""

    SECONDS = "s"
    MILLISECONDS = "ms"
    MICROSECONDS = "us"
    BYTES = "bytes"
    COUNT = "count"
    ROW = "row"
    VARIANT = "variant"
    SAMPLE = "sample"
    PHENOTYPE = "phenotype"
    RATIO = "ratio"
    PERCENT = "percent"


class MetricAggregation(enum.StrEnum):
    """Stable metric aggregation tokens."""

    EXACT = "exact"
    MEAN = "mean"
    MEDIAN = "median"
    MINIMUM = "min"
    MAXIMUM = "max"
    STANDARD_DEVIATION = "stdev"


class ComparisonJudgement(enum.StrEnum):
    """Stable comparison judgement tokens."""

    IMPROVEMENT = "improvement"
    NEUTRAL = "neutral"
    REGRESSION = "regression"
    INCONCLUSIVE = "inconclusive"
    NOT_COMPARABLE = "not_comparable"


@dataclass(frozen=True)
class ToolProducer:
    """Producer metadata for one tooling artifact.

    Attributes:
        tool_name: Stable tool identifier.
        tool_version: Tool-specific artifact producer version.
        repository: Repository identifier.
        git_head: Current Git commit hash, when available.
        dirty: Whether the repository has uncommitted changes.
        dirty_diff_sha256: SHA-256 of tracked dirty diff material, when dirty.

    """

    tool_name: str
    tool_version: int
    repository: str
    git_head: str | None
    dirty: bool
    dirty_diff_sha256: str | None


@dataclass(frozen=True)
class ToolRunIdentity:
    """Run identity metadata for one artifact directory.

    Attributes:
        run_id: Stable run identifier.
        created_at: UTC creation timestamp.
        status: Standard run status.
        status_reason: Optional human-readable status reason.
        output_directory: Output directory path.

    """

    run_id: str
    created_at: str
    status: ToolArtifactStatus
    status_reason: str | None
    output_directory: str


@dataclass(frozen=True)
class ToolContextSnapshot:
    """Resolved context included in durable tooling reports."""

    repository_root: str
    output_directory: str
    cwd: str
    hostname: str
    slurm_job_id: str | None


@dataclass(frozen=True)
class ArtifactRecord:
    """One generated artifact record."""

    path: str
    path_type: PathType
    kind: str
    media_type: str
    size_bytes: int | None
    sha256: str | None
    description: str


@dataclass(frozen=True)
class InputFileRecord:
    """One external input file record."""

    path: str
    path_type: PathType
    kind: str
    size_bytes: int | None
    sha256: str | None
    fingerprint_policy: FingerprintPolicy


@dataclass(frozen=True)
class MetricSource:
    """Pointer to the source value used for a normalized metric."""

    artifact_path: str
    json_pointer: str | None


@dataclass(frozen=True)
class MetricRecord:
    """One normalized long-form metric record."""

    schema_name: str
    schema_version: int
    run_id: str
    case_id: str | None
    metric_name: str
    value: float | int | None
    unit: str
    aggregation: str
    higher_is_better: bool | None
    dimensions: dict[str, object]
    trial_id: str | None
    phase: str | None
    source: MetricSource | None


@dataclass(frozen=True)
class ToolEventRecord:
    """One tooling JSONL event record."""

    schema_name: str
    schema_version: int
    timestamp: str
    level: str
    tool_name: str
    run_id: str
    phase: str
    event: str
    message: str
    fields: dict[str, object]


@dataclass(frozen=True)
class CommandRecord:
    """One child command ledger entry."""

    schema_name: str
    schema_version: int
    command_id: str
    tool_name: str
    phase: str
    args: list[str]
    display: str
    cwd: str | None
    environment_overrides: dict[str, str]
    redacted_environment_keys: list[str]
    stdout_log: str | None
    stderr_log: str | None
    status: ToolArtifactStatus
    return_code: int | None
    started_at: str | None
    finished_at: str | None
    wall_time_seconds: float | None


@dataclass(frozen=True)
class FailureRecord:
    """One structured failure record."""

    failure_id: str
    phase: str
    status: ToolArtifactStatus
    message: str
    exception_type: str | None
    stderr_excerpt: str | None
    stdout_log: str | None
    stderr_log: str | None
    command_id: str | None


@dataclass(frozen=True)
class FindingRecord:
    """One human or agent finding."""

    finding_id: str
    severity: str
    category: str
    title: str
    evidence: list[str]
    interpretation: str


@dataclass(frozen=True)
class RecommendedAction:
    """One recommended follow-up action."""

    action_id: str
    priority: str
    description: str


@dataclass(frozen=True)
class AgentSummary:
    """Compact agent-oriented report summary."""

    one_sentence: str
    key_observations: list[str]
    risks: list[str]
    next_actions: list[str]


@dataclass(frozen=True)
class ReportEnvelope:
    """Machine-readable report envelope for Tooling Artifact Format v1."""

    schema_name: str
    schema_version: int
    producer: ToolProducer
    run: ToolRunIdentity
    context: ToolContextSnapshot
    configuration: dict[str, object]
    summary: dict[str, object]
    cases: list[dict[str, object]]
    trials: list[dict[str, object]]
    metrics: list[MetricRecord]
    comparisons: list[dict[str, object]]
    diagnostics: dict[str, object]
    failures: list[FailureRecord]
    findings: list[FindingRecord]
    recommended_actions: list[RecommendedAction]


@dataclass(frozen=True)
class ArtifactManifest:
    """Artifact manifest envelope for Tooling Artifact Format v1."""

    schema_name: str
    schema_version: int
    producer: ToolProducer
    run: ToolRunIdentity
    context: ToolContextSnapshot
    primary_artifacts: dict[str, str | None]
    artifacts: list[ArtifactRecord]
    input_files: list[InputFileRecord]
    tooling_config: dict[str, str | None]
    notes: list[str]


@dataclass(frozen=True)
class ComparisonReport:
    """First-class comparison report envelope."""

    schema_name: str
    schema_version: int
    producer: ToolProducer
    run: ToolRunIdentity
    baseline: dict[str, object]
    current: dict[str, object]
    thresholds: list[dict[str, object]]
    comparisons: list[dict[str, object]]
    summary: dict[str, object]


@dataclass(frozen=True)
class ArtifactBundle:
    """In-memory artifact bundle payload before writing files."""

    report: ReportEnvelope
    manifest: ArtifactManifest
    events: list[ToolEventRecord]
    metrics: list[MetricRecord]
    commands: list[CommandRecord]
    summary_markdown: str
    comparisons: ComparisonReport | None


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_run_id(prefix: str) -> str:
    """Build a unique run identifier."""
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = uuid.uuid4().hex[:8]
    normalized_prefix = prefix.replace("_", "-")
    return f"{normalized_prefix}-{timestamp}-{suffix}"


def calculate_sha256(path: Path) -> str:
    """Calculate a file SHA-256 digest."""
    sha256_hash = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def git_output(repository_root: Path, command_arguments: collections.abc.Sequence[str]) -> str | None:
    """Run a Git command and return stripped stdout when successful."""
    completed_process = subprocess.run(
        ["git", *command_arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed_process.returncode != 0:
        return None
    return completed_process.stdout.strip()


def build_dirty_diff_sha256(repository_root: Path) -> str | None:
    """Hash tracked dirty diff material when present."""
    diff_parts: list[bytes] = []
    for diff_arguments in (("diff", "--binary"), ("diff", "--cached", "--binary")):
        completed_process = subprocess.run(
            ["git", *diff_arguments],
            cwd=repository_root,
            check=False,
            capture_output=True,
        )
        if completed_process.returncode == 0:
            diff_parts.append(completed_process.stdout)
    status_output = git_output(repository_root, ("status", "--short"))
    if status_output:
        diff_parts.append(status_output.encode("utf-8"))
    if not any(diff_parts):
        return None
    return hashlib.sha256(b"\n".join(diff_parts)).hexdigest()


def build_producer(
    *,
    tool_name: str,
    repository_root: Path,
    tool_version: int = 1,
) -> ToolProducer:
    """Build producer metadata from a repository checkout."""
    status_output = git_output(repository_root, ("status", "--short"))
    dirty = bool(status_output)
    return ToolProducer(
        tool_name=tool_name,
        tool_version=tool_version,
        repository=REPOSITORY_NAME,
        git_head=git_output(repository_root, ("rev-parse", "HEAD")),
        dirty=dirty,
        dirty_diff_sha256=build_dirty_diff_sha256(repository_root) if dirty else None,
    )


def build_run_identity(
    *,
    tool_name: str,
    output_directory: Path,
    status: ToolArtifactStatus,
    status_reason: str | None = None,
    run_id: str | None = None,
) -> ToolRunIdentity:
    """Build run identity metadata."""
    return ToolRunIdentity(
        run_id=run_id or build_run_id(tool_name),
        created_at=utc_now(),
        status=status,
        status_reason=status_reason,
        output_directory=str(output_directory),
    )


def build_context_snapshot(
    *,
    output_directory: Path,
    repository_root: Path,
) -> ToolContextSnapshot:
    """Build a context snapshot from the local environment."""
    return ToolContextSnapshot(
        repository_root=str(repository_root),
        output_directory=str(output_directory),
        cwd=str(Path.cwd().resolve()),
        hostname=platform.node(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )


def artifact_relative_path(output_directory: Path, path: Path) -> str:
    """Return a portable path relative to the artifact output directory."""
    try:
        return str(path.resolve().relative_to(output_directory.resolve()))
    except ValueError:
        return str(path)


def build_artifact_record(
    *,
    output_directory: Path,
    path: Path,
    kind: str,
    media_type: str,
    description: str,
    fingerprint: bool = True,
) -> ArtifactRecord:
    """Build one generated artifact record."""
    size_bytes = path.stat().st_size if path.exists() and path.is_file() else None
    return ArtifactRecord(
        path=artifact_relative_path(output_directory, path),
        path_type=PathType.ARTIFACT_RELATIVE,
        kind=kind,
        media_type=media_type,
        size_bytes=size_bytes,
        sha256=calculate_sha256(path) if fingerprint and path.exists() and path.is_file() else None,
        description=description,
    )


def build_input_file_record(
    *,
    path: Path,
    kind: str,
    fingerprint_policy: FingerprintPolicy = FingerprintPolicy.METADATA_ONLY,
) -> InputFileRecord:
    """Build one external input-file record."""
    exists = path.exists() and path.is_file()
    sha256 = calculate_sha256(path) if exists and fingerprint_policy == FingerprintPolicy.SHA256 else None
    policy = fingerprint_policy if exists else FingerprintPolicy.MISSING
    return InputFileRecord(
        path=str(path),
        path_type=PathType.ABSOLUTE_INPUT if path.is_absolute() else PathType.ARTIFACT_RELATIVE,
        kind=kind,
        size_bytes=path.stat().st_size if exists else None,
        sha256=sha256,
        fingerprint_policy=policy,
    )


def build_metric_record(
    *,
    run_id: str,
    metric_name: str,
    value: float | int | None,
    unit: str,
    aggregation: str,
    case_id: str | None = None,
    trial_id: str | None = None,
    phase: str | None = None,
    higher_is_better: bool | None = None,
    dimensions: dict[str, object] | None = None,
    source: MetricSource | None = None,
) -> MetricRecord:
    """Build one normalized metric record."""
    return MetricRecord(
        schema_name="g.tooling.metric",
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        case_id=case_id,
        metric_name=metric_name,
        value=value,
        unit=unit,
        aggregation=aggregation,
        higher_is_better=higher_is_better,
        dimensions=dimensions or {},
        trial_id=trial_id,
        phase=phase,
        source=source,
    )


def build_tool_event(
    *,
    tool_name: str,
    run_id: str,
    phase: str,
    event: str,
    message: str,
    level: str = "info",
    fields: dict[str, object] | None = None,
) -> ToolEventRecord:
    """Build one tooling event record."""
    return ToolEventRecord(
        schema_name="g.tooling.event",
        schema_version=SCHEMA_VERSION,
        timestamp=utc_now(),
        level=level,
        tool_name=tool_name,
        run_id=run_id,
        phase=phase,
        event=event,
        message=message,
        fields=fields or {},
    )


def build_command_record(
    *,
    command_id: str,
    tool_name: str,
    run_id: str,
    phase: str,
    args: collections.abc.Sequence[str],
    output_directory: Path,
    cwd: Path | None = None,
    environment_overrides: dict[str, str] | None = None,
    redacted_environment_keys: collections.abc.Sequence[str] = (),
    stdout_log: Path | None = None,
    stderr_log: Path | None = None,
    status: ToolArtifactStatus = ToolArtifactStatus.SUCCESS,
    return_code: int | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    wall_time_seconds: float | None = None,
) -> CommandRecord:
    """Build one command ledger record."""
    normalized_args = [str(argument) for argument in args]
    ledger_args = materialize_inline_python_command(
        command_id=command_id,
        args=normalized_args,
        output_directory=output_directory,
    )
    return CommandRecord(
        schema_name="g.tooling.command",
        schema_version=SCHEMA_VERSION,
        command_id=command_id,
        tool_name=tool_name,
        phase=phase,
        args=ledger_args,
        display=shlex.join(ledger_args),
        cwd=str(cwd) if cwd is not None else None,
        environment_overrides=dict(environment_overrides or {}),
        redacted_environment_keys=[str(key) for key in redacted_environment_keys],
        stdout_log=artifact_relative_path(output_directory, stdout_log) if stdout_log is not None else None,
        stderr_log=artifact_relative_path(output_directory, stderr_log) if stderr_log is not None else None,
        status=status,
        return_code=return_code,
        started_at=started_at,
        finished_at=finished_at,
        wall_time_seconds=wall_time_seconds,
    )


def materialize_inline_python_command(
    *,
    command_id: str,
    args: list[str],
    output_directory: Path,
) -> list[str]:
    """Write inline Python command payloads to scripts and return ledger args."""
    if len(args) < 3 or args[1] != "-c":
        return args
    script_directory = output_directory / "commands" / "scripts"
    script_directory.mkdir(parents=True, exist_ok=True)
    script_path = script_directory / f"{command_id}.py"
    script_path.write_text(args[2], encoding="utf-8")
    return [args[0], str(script_path), *args[3:]]


def build_default_agent_summary(
    *,
    title: str,
    status: ToolArtifactStatus,
    primary_metric_name: str | None = None,
) -> AgentSummary:
    """Build a compact default agent summary."""
    metric_text = f" Primary metric: {primary_metric_name}." if primary_metric_name else ""
    return AgentSummary(
        one_sentence=f"{title} finished with status {status.value}.{metric_text}",
        key_observations=[f"Status: {status.value}."],
        risks=[],
        next_actions=[],
    )


def build_report_envelope(
    *,
    producer: ToolProducer,
    run: ToolRunIdentity,
    context: ToolContextSnapshot,
    title: str,
    configuration: dict[str, object],
    summary: dict[str, object] | None = None,
    cases: list[dict[str, object]] | None = None,
    trials: list[dict[str, object]] | None = None,
    metrics: list[MetricRecord] | None = None,
    comparisons: list[dict[str, object]] | None = None,
    diagnostics: dict[str, object] | None = None,
    failures: list[FailureRecord] | None = None,
    findings: list[FindingRecord] | None = None,
    recommended_actions: list[RecommendedAction] | None = None,
) -> ReportEnvelope:
    """Build a standard report envelope."""
    metric_records = metrics or []
    summary_payload = dict(summary or {})
    summary_payload.setdefault("title", title)
    summary_payload.setdefault("status", run.status.value)
    summary_payload.setdefault("headline", f"{title} finished with status {run.status.value}.")
    summary_payload.setdefault(
        "agent_summary",
        dataclasses.asdict(
            build_default_agent_summary(
                title=title,
                status=run.status,
                primary_metric_name=metric_records[0].metric_name if metric_records else None,
            )
        ),
    )
    return ReportEnvelope(
        schema_name="g.tooling.report",
        schema_version=SCHEMA_VERSION,
        producer=producer,
        run=run,
        context=context,
        configuration=configuration,
        summary=summary_payload,
        cases=cases or [],
        trials=trials or [],
        metrics=metric_records,
        comparisons=comparisons or [],
        diagnostics=diagnostics or {},
        failures=failures or [],
        findings=findings or [],
        recommended_actions=recommended_actions or [],
    )


def build_artifact_manifest(
    *,
    producer: ToolProducer,
    run: ToolRunIdentity,
    context: ToolContextSnapshot,
    output_directory: Path,
    input_files: list[InputFileRecord] | None = None,
    notes: list[str] | None = None,
) -> ArtifactManifest:
    """Build a standard artifact manifest from files present under output directory."""
    primary_artifacts = {
        "report_json": "report.json" if (output_directory / "report.json").exists() else None,
        "summary_markdown": "summary.md" if (output_directory / "summary.md").exists() else None,
        "events_jsonl": "events.jsonl" if (output_directory / "events.jsonl").exists() else None,
        "metrics_jsonl": "metrics.jsonl" if (output_directory / "metrics.jsonl").exists() else None,
    }
    artifacts = [
        build_artifact_record(
            output_directory=output_directory,
            path=artifact_path,
            kind=guess_artifact_kind(artifact_path),
            media_type=guess_media_type(artifact_path),
            description=f"Generated {artifact_path.name}",
            fingerprint=artifact_path.stat().st_size <= 64 * 1024 * 1024,
        )
        for artifact_path in sorted(path for path in output_directory.rglob("*") if path.is_file())
        if artifact_path.name != "artifact_manifest.json"
    ]
    tooling_config = {
        "resolved_hydra": "config/resolved_hydra.yaml"
        if (output_directory / "config" / "resolved_hydra.yaml").exists()
        else None,
        "resolved_tool": "config/resolved_tool.json"
        if (output_directory / "config" / "resolved_tool.json").exists()
        else None,
    }
    return ArtifactManifest(
        schema_name="g.tooling.artifact_manifest",
        schema_version=SCHEMA_VERSION,
        producer=producer,
        run=run,
        context=context,
        primary_artifacts=primary_artifacts,
        artifacts=artifacts,
        input_files=input_files or [],
        tooling_config=tooling_config,
        notes=notes or [],
    )


def build_summary_markdown(report: ReportEnvelope) -> str:
    """Build a standard Markdown summary for a report envelope."""
    summary = report.summary
    lines = [
        f"# {summary.get('title', report.producer.tool_name)} Report",
        "",
        "## Status",
        "",
        f"- Status: `{report.run.status.value}`",
        f"- Run ID: `{report.run.run_id}`",
        f"- Commit: `{report.producer.git_head}`",
        f"- Machine: `{report.context.hostname}`",
        f"- Output: `{report.run.output_directory}`",
        "",
        "## Executive Summary",
        "",
        str(summary.get("headline", "")),
        "",
        "## Headline Metrics",
        "",
        "| metric | value | unit | aggregation | case |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for metric in report.metrics[:20]:
        lines.append(
            "| "
            f"{metric.metric_name} | "
            f"{metric.value if metric.value is not None else 'null'} | "
            f"{metric.unit} | "
            f"{metric.aggregation} | "
            f"{metric.case_id or ''} |"
        )
    if not report.metrics:
        lines.append("| none | null |  |  |  |")
    lines.extend(["", "## Failures And Skipped Work", ""])
    if report.failures:
        lines.extend(["| failure | phase | status | message | log |", "| --- | --- | --- | --- | --- |"])
        for failure in report.failures:
            log_path = failure.stderr_log or failure.stdout_log or ""
            lines.append(
                f"| {failure.failure_id} | {failure.phase} | {failure.status.value} | {failure.message} | {log_path} |"
            )
    else:
        lines.append("No structured failures recorded.")
    lines.extend(["", "## Recommended Actions", ""])
    if report.recommended_actions:
        for recommended_action in report.recommended_actions:
            lines.append(f"1. {recommended_action.description}")
    else:
        lines.append("No recommended actions recorded.")
    lines.extend(
        [
            "",
            "## Artifact Map",
            "",
            "- `report.json`: machine-readable report",
            "- `events.jsonl`: structured event stream",
            "- `metrics.jsonl`: normalized metrics",
            "- `commands/commands.jsonl`: child command ledger",
            "- `artifact_manifest.json`: artifact index and provenance",
            "",
        ]
    )
    return "\n".join(lines)


def write_standard_artifact_bundle(
    *,
    output_directory: Path,
    report: ReportEnvelope,
    events: list[ToolEventRecord] | None = None,
    commands: list[CommandRecord] | None = None,
    input_files: list[InputFileRecord] | None = None,
    summary_markdown: str | None = None,
    comparisons: ComparisonReport | None = None,
    hydra_config: typing.Any | None = None,
    tool_payload: typing.Any | None = None,
    legacy_json_aliases: collections.abc.Sequence[Path] = (),
    legacy_markdown_aliases: collections.abc.Sequence[Path] = (),
    notes: list[str] | None = None,
) -> ArtifactManifest:
    """Write a complete Tooling Artifact Format v1 bundle.

    Args:
        output_directory: Artifact directory.
        report: Standard report envelope.
        events: Tooling events.
        commands: Command ledger records.
        input_files: External input file records.
        summary_markdown: Optional Markdown body.
        comparisons: Optional comparison report.
        hydra_config: Optional resolved Hydra config.
        tool_payload: Optional resolved tool payload.
        legacy_json_aliases: Compatibility JSON paths.
        legacy_markdown_aliases: Compatibility Markdown paths.
        notes: Manifest notes.

    Returns:
        Written artifact manifest.

    """
    output_directory.mkdir(parents=True, exist_ok=True)
    tooling_reports.write_config_snapshots(
        output_directory,
        hydra_config=hydra_config,
        tool_payload=tool_payload or report.configuration,
    )
    report_path = output_directory / "report.json"
    tooling_reports.write_report_envelope(report_path, report, schema_name="g.tooling.report", sort_keys=True)
    metrics = report.metrics
    tooling_reports.write_jsonl(output_directory / "metrics.jsonl", metrics, sort_keys=True)
    tooling_reports.write_jsonl(output_directory / "events.jsonl", events or [], sort_keys=True)
    tooling_reports.write_jsonl(output_directory / "commands" / "commands.jsonl", commands or [], sort_keys=True)
    markdown_text = summary_markdown if summary_markdown is not None else build_summary_markdown(report)
    summary_path = output_directory / "summary.md"
    tooling_reports.write_markdown_report(summary_path, markdown_text)
    for alias_path in legacy_json_aliases:
        tooling_reports.copy_json_alias(report_path, alias_path)
    for alias_path in legacy_markdown_aliases:
        if summary_path.resolve() != alias_path.resolve():
            alias_path.parent.mkdir(parents=True, exist_ok=True)
            alias_path.write_text(markdown_text, encoding="utf-8")
    if comparisons is not None:
        tooling_reports.write_report_envelope(
            output_directory / "comparisons.json",
            comparisons,
            schema_name="g.tooling.comparison",
            sort_keys=True,
        )
    manifest = build_artifact_manifest(
        producer=report.producer,
        run=report.run,
        context=report.context,
        output_directory=output_directory,
        input_files=input_files or [],
        notes=notes or [],
    )
    tooling_reports.write_report_envelope(
        output_directory / "artifact_manifest.json",
        manifest,
        schema_name="g.tooling.artifact_manifest",
        sort_keys=True,
    )
    return manifest


def guess_artifact_kind(path: Path) -> str:
    """Guess artifact kind from a path."""
    if path.name.endswith(".stdout.log"):
        return "stdout_log"
    if path.name.endswith(".stderr.log"):
        return "stderr_log"
    if path.suffix == ".jsonl":
        return "jsonl"
    if path.suffix == ".json":
        return "json"
    if path.suffix in {".md", ".txt"}:
        return "text"
    if path.suffix == ".toml":
        return "toml"
    if path.suffix in {".yaml", ".yml"}:
        return "yaml"
    return "artifact"


def guess_media_type(path: Path) -> str:
    """Guess media type from a path."""
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".jsonl":
        return "application/jsonl"
    if path.suffix == ".md":
        return "text/markdown"
    if path.suffix == ".txt" or path.suffix == ".log":
        return "text/plain"
    if path.suffix == ".toml":
        return "application/toml"
    if path.suffix in {".yaml", ".yml"}:
        return "application/yaml"
    return "application/octet-stream"


def validate_schema_payload(payload: dict[str, typing.Any], schema_name: str) -> None:
    """Validate common schema envelope fields."""
    if payload.get("schema_name") != schema_name:
        message = f"Expected schema_name={schema_name!r}, got {payload.get('schema_name')!r}."
        raise tooling_reports.ReportSchemaError(message)
    if payload.get("schema_version") != SCHEMA_VERSION:
        message = f"Expected schema_version={SCHEMA_VERSION}, got {payload.get('schema_version')!r}."
        raise tooling_reports.ReportSchemaError(message)
