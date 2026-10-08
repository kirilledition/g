"""Hydra entrypoint for participant-free Workbench integration fixtures."""

from __future__ import annotations

import typing
from pathlib import Path

import hydra

from tooling.common import hydra_arguments, hydra_compat
from tooling.workbench import demo

if typing.TYPE_CHECKING:
    import omegaconf


@hydra.main(version_base=None, config_path="../configs", config_name="workbench_demo")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Create a new synthetic bundle at the explicitly configured path."""
    values = hydra_arguments.tool_config_to_dictionary(config)
    if values["output_directory"] is None:
        raise ValueError("Set tool.output_directory to a new local directory.")
    bundle = demo.create_demo_bundle(Path(str(values["output_directory"])))
    print(f"Quantitative manifest: {bundle.quantitative_manifest}")
    print(f"Binary manifest: {bundle.binary_manifest}")


def main() -> None:
    """Run the synthetic fixture command."""
    hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
