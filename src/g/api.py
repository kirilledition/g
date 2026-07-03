"""Public Python API for GWAS execution."""

from __future__ import annotations

import typing

from g.interface import config
from g.runner import events
from g.runner import execution as runner_execution
from g.runner import runtime as runner_runtime

RunArtifacts = events.RunArtifacts
RuntimeState = runner_runtime.RuntimeState


class RegenieApi:
    """Callable public REGENIE-compatible API."""

    def __call__(self, regenie_config: config.RegenieConfig) -> RunArtifacts:
        """Run from a normalized config without installing CLI signal handlers."""
        return runner_execution.regenie(
            regenie_config,
            run_telemetry_session=None,
            close_telemetry_session_on_exit=True,
            initialize_logging_on_entry=True,
        )

    def from_options(self, raw_options: typing.Mapping[str, object]) -> RunArtifacts:
        """Build a config from Python options and run without installing CLI signal handlers."""
        return runner_execution.regenie(
            config.RegenieConfig.from_options(raw_options),
            run_telemetry_session=None,
            close_telemetry_session_on_exit=True,
            initialize_logging_on_entry=True,
        )


def describe_runtime_state() -> RuntimeState:
    """Return process-global runtime settings already configured in this process."""
    return runner_runtime.describe_runtime_state()


regenie = RegenieApi()
