"""Grouped Hydra CLI for debug and parity tooling."""

from __future__ import annotations

import enum
import typing

import hydra

from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.debug import (
    binary_firth,
    binary_regenie_parity,
    check_internal_defaults,
    check_internal_init_exports,
    check_pyo3_stub,
    check_rust_architecture,
    linear_regenie_parity,
)

if typing.TYPE_CHECKING:
    import omegaconf


class DebugToolName(enum.StrEnum):
    """Available grouped debug tools."""

    BINARY_FIRTH = "binary_firth"
    BINARY_REGENIE_PARITY = "binary_regenie_parity"
    LINEAR_REGENIE_PARITY = "linear_regenie_parity"
    CHECK_INTERNAL_DEFAULTS = "check_internal_defaults"
    CHECK_INTERNAL_INIT_EXPORTS = "check_internal_init_exports"
    CHECK_PYO3_STUB = "check_pyo3_stub"
    CHECK_RUST_ARCHITECTURE = "check_rust_architecture"


@hydra.main(version_base=None, config_path="../configs", config_name="debug")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Dispatch a debug tool from Hydra configuration."""
    tool_name = DebugToolName(str(config.tool.name))
    if tool_name == DebugToolName.BINARY_FIRTH:
        binary_firth.run_tool(binary_firth.build_arguments_from_config(config))
        return
    if tool_name == DebugToolName.BINARY_REGENIE_PARITY:
        binary_regenie_parity.run_tool(binary_regenie_parity.build_arguments_from_config(config))
        return
    if tool_name == DebugToolName.LINEAR_REGENIE_PARITY:
        linear_regenie_parity.run_tool(linear_regenie_parity.build_arguments_from_config(config))
        return
    if tool_name == DebugToolName.CHECK_PYO3_STUB:
        exit_code = check_pyo3_stub.run_tool()
        if exit_code:
            raise SystemExit(exit_code)
        return
    if tool_name == DebugToolName.CHECK_RUST_ARCHITECTURE:
        exit_code = check_rust_architecture.run_tool(check_rust_architecture.REPOSITORY_ROOT)
        if exit_code:
            raise SystemExit(exit_code)
        return
    if tool_name == DebugToolName.CHECK_INTERNAL_DEFAULTS:
        exit_code = check_internal_defaults.run_tool(check_internal_defaults.PRODUCTION_SOURCE_ROOT)
        if exit_code:
            raise SystemExit(exit_code)
        return
    if tool_name == DebugToolName.CHECK_INTERNAL_INIT_EXPORTS:
        exit_code = check_internal_init_exports.run_tool(check_internal_init_exports.PRODUCTION_PACKAGE_ROOT)
        if exit_code:
            raise SystemExit(exit_code)
        return
    typing.assert_never(tool_name)


def main() -> None:
    """Run the grouped debug CLI from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
