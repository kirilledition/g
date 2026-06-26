"""Shared execution context for development tooling."""

from __future__ import annotations

import os
import typing
from dataclasses import dataclass
from pathlib import Path

import tooling.configuration as tooling_configuration
from tooling.common import paths as tooling_paths

if typing.TYPE_CHECKING:
    import collections.abc

    import omegaconf


@dataclass(frozen=True)
class ToolContext:
    """Resolved context shared by tooling commands.

    Attributes:
        repository_root: Repository root containing project metadata.
        data_directory: Resolved data directory after environment overrides.
        output_directory: Resolved default output directory.
        machine: Resolved machine configuration.
        telemetry: Resolved telemetry configuration.
        raw_config: Composed Hydra config.
        current_working_directory: Current working directory when the context was built.
        hydra_chdir: Whether Hydra is configured to change the process directory.

    """

    repository_root: Path
    data_directory: Path
    output_directory: Path
    machine: tooling_configuration.MachineConfig
    telemetry: tooling_configuration.TelemetryConfig
    raw_config: omegaconf.DictConfig
    current_working_directory: Path
    hydra_chdir: bool


def build_tool_context(
    config: omegaconf.DictConfig,
    *,
    start_path: Path | None = None,
    environment: collections.abc.Mapping[str, str] | None = None,
    output_directory: Path | None = None,
) -> ToolContext:
    """Build the standard resolved context for a tooling entrypoint.

    Args:
        config: Composed Hydra config.
        start_path: Optional path inside the repository.
        environment: Optional environment mapping.
        output_directory: Optional explicit output directory.

    Returns:
        Resolved tool context.

    """
    environment_values = environment if environment is not None else os.environ
    repository_root = tooling_paths.find_repository_root(start_path)
    typed_config = tooling_configuration.instantiate_config(config)
    data_directory = tooling_paths.resolve_repo_relative_path(
        Path(
            environment_values.get(
                tooling_paths.DATA_DIRECTORY_ENVIRONMENT_VARIABLE, typed_config.dataset.data_directory
            )
        ),
        repository_root,
    )
    telemetry_output_directory = tooling_paths.resolve_repo_relative_path(
        Path(typed_config.telemetry.output_parent),
        repository_root,
    )
    resolved_output_directory = (
        tooling_paths.resolve_repo_relative_path(output_directory, repository_root)
        if output_directory is not None
        else telemetry_output_directory
    )
    hydra_chdir = False
    if "hydra" in config and "job" in config.hydra and "chdir" in config.hydra.job:
        hydra_chdir = bool(config.hydra.job.chdir)
    return ToolContext(
        repository_root=repository_root,
        data_directory=data_directory,
        output_directory=resolved_output_directory,
        machine=typed_config.machine,
        telemetry=typed_config.telemetry,
        raw_config=config,
        current_working_directory=Path.cwd().resolve(),
        hydra_chdir=hydra_chdir,
    )


def context_report(context: ToolContext) -> dict[str, object]:
    """Build a JSON-ready context payload for durable reports.

    Args:
        context: Resolved tool context.

    Returns:
        Context metadata payload.

    """
    return {
        "repository_root": str(context.repository_root),
        "data_directory": str(context.data_directory),
        "output_directory": str(context.output_directory),
        "cwd": str(context.current_working_directory),
        "hydra_chdir": context.hydra_chdir,
        "machine": {
            "name": context.machine.name,
            "device": context.machine.device,
            "slurm_node": context.machine.slurm_node,
            "cpus_per_task": context.machine.cpus_per_task,
            "memory": context.machine.memory,
        },
    }
