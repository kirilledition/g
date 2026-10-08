"""Validate and execute versioned jobs inside controlled Workbench workspaces."""

from __future__ import annotations

import dataclasses
import json
import signal
import typing
from pathlib import Path

import hydra

from tooling.common import hydra_arguments, hydra_compat
from tooling.workbench import runner

if typing.TYPE_CHECKING:
    import omegaconf


def arguments_from_config(config: omegaconf.DictConfig) -> runner.RunnerArguments:
    """Resolve the small Hydra execution boundary into typed settings."""
    values = hydra_arguments.tool_config_to_dictionary(config)
    prefix = values["runner_prefix"]
    if not isinstance(prefix, list) or any(not isinstance(value, str) for value in prefix):
        raise ValueError("tool.runner_prefix must be a list of command arguments.")
    return runner.RunnerArguments(
        action=runner.Action(str(values["action"])),
        manifest_path=Path(str(values["manifest"])).resolve(),
        work_directory=Path(str(values["work_directory"])).resolve(),
        dry_run=hydra_arguments.boolean_value(values["dry_run"]),
        billing_project=str(values["billing_project"]) if values["billing_project"] is not None else None,
        runner_prefix=tuple(typing.cast("list[str]", prefix)),
        profile_max_variants=int(values["profile_max_variants"]),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="workbench")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run one action and report only the resulting evidence/publication paths."""
    arguments = arguments_from_config(config)
    previous_handler = signal.signal(signal.SIGTERM, runner.interrupt_on_termination)
    try:
        result = runner.run_attempt(arguments)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    print(json.dumps(dataclasses.asdict(result), default=str, sort_keys=True))


def main() -> None:
    """Invoke the maintained Hydra entrypoint."""
    hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
