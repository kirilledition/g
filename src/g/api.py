"""Public Python API for GWAS execution."""

from __future__ import annotations

import typing

from g import interface
from g.runner import events
from g.runner import execution as runner_execution

RunArtifacts = events.RunArtifacts


class RegenieApi:
    """Callable public REGENIE-compatible API."""

    def __call__(self, regenie_config: interface.RegenieConfig) -> RunArtifacts:
        """Run from a normalized config without installing CLI signal handlers."""
        return runner_execution.regenie(
            regenie_config,
            run_telemetry_session=None,
            close_telemetry_session_on_exit=True,
            initialize_logging_on_entry=True,
        )

    def from_options(self, raw_options: typing.Mapping[str, typing.Any]) -> RunArtifacts:
        """Build a config from Python options and run without installing CLI signal handlers."""
        return runner_execution.regenie(
            interface.RegenieConfig.from_options(raw_options),
            run_telemetry_session=None,
            close_telemetry_session_on_exit=True,
            initialize_logging_on_entry=True,
        )


regenie = RegenieApi()
