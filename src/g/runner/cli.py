"""Runner-owned CLI backend lifecycle."""

from __future__ import annotations

import sys
import typing

import g._core

if typing.TYPE_CHECKING:
    from g.runner import events


def run(arguments: typing.Sequence[str]) -> int:
    """Run CLI arguments through the native CLI driver."""
    result = g._core.cli.run_with_python_backend(
        list(arguments),
        run_validated_config_with_python_backend,
    )
    for output_text in result.stdout_chunks:
        print(output_text, end="")
    for output_text in result.stderr_chunks:
        print(output_text, end="", file=sys.stderr)
    return result.exit_code


def run_validated_config_with_python_backend(
    regenie_config: g._core.RegenieConfig,
    cli_context: g._core.NativeCliRunContext,
) -> g._core.NativeRunArtifacts:
    """Run a validated native CLI config through the Python execution backend."""
    from g.runner import execution

    artifacts = execution.regenie(
        regenie_config,
        run_telemetry_session=typing.cast("events.TelemetrySession", cli_context.telemetry_session_view()),
        close_telemetry_session_on_exit=False,
        initialize_logging_on_entry=False,
    )
    return cli_context.native_artifacts_from_python_artifacts(artifacts)


def main() -> None:
    """Run the GWAS CLI."""
    raise SystemExit(run(sys.argv[1:]))
