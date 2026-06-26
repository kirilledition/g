"""Grouped Hydra CLI for debug and parity tooling."""

from __future__ import annotations

import enum
import typing

import hydra

from tooling.cli import schema_check
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import registry as tooling_registry
from tooling.debug import (
    binary_firth,
    binary_regenie_parity,
    check_internal_defaults,
    check_internal_init_exports,
    check_pyo3_stub,
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
    SCHEMA_CHECK = "schema_check"


def build_no_arguments(config: omegaconf.DictConfig) -> None:
    """Build an empty argument payload for fixed guardrail tools."""
    del config
    return


def run_check_pyo3_stub(arguments: None) -> None:
    """Run the PyO3 stub guardrail."""
    del arguments
    exit_code = check_pyo3_stub.run_tool()
    if exit_code:
        raise SystemExit(exit_code)


def run_check_internal_defaults(arguments: None) -> None:
    """Run the internal-defaults guardrail."""
    del arguments
    exit_code = check_internal_defaults.run_tool(check_internal_defaults.PRODUCTION_SOURCE_ROOT)
    if exit_code:
        raise SystemExit(exit_code)


def run_check_internal_init_exports(arguments: None) -> None:
    """Run the package-initializer guardrail."""
    del arguments
    exit_code = check_internal_init_exports.run_tool(check_internal_init_exports.PRODUCTION_PACKAGE_ROOT)
    if exit_code:
        raise SystemExit(exit_code)


TOOLS: dict[str, tooling_registry.ToolSpec[typing.Any]] = {
    DebugToolName.BINARY_FIRTH.value: tooling_registry.ToolSpec(
        name=DebugToolName.BINARY_FIRTH.value,
        config_name="debug_binary_firth",
        build_arguments=binary_firth.build_arguments_from_config,
        run=binary_firth.run_tool,
    ),
    DebugToolName.BINARY_REGENIE_PARITY.value: tooling_registry.ToolSpec(
        name=DebugToolName.BINARY_REGENIE_PARITY.value,
        config_name="debug_binary_regenie_parity",
        build_arguments=binary_regenie_parity.build_arguments_from_config,
        run=binary_regenie_parity.run_tool,
    ),
    DebugToolName.LINEAR_REGENIE_PARITY.value: tooling_registry.ToolSpec(
        name=DebugToolName.LINEAR_REGENIE_PARITY.value,
        config_name="debug_linear_regenie_parity",
        build_arguments=linear_regenie_parity.build_arguments_from_config,
        run=linear_regenie_parity.run_tool,
    ),
    DebugToolName.CHECK_PYO3_STUB.value: tooling_registry.ToolSpec(
        name=DebugToolName.CHECK_PYO3_STUB.value,
        config_name="debug_check_pyo3_stub",
        build_arguments=build_no_arguments,
        run=run_check_pyo3_stub,
    ),
    DebugToolName.CHECK_INTERNAL_DEFAULTS.value: tooling_registry.ToolSpec(
        name=DebugToolName.CHECK_INTERNAL_DEFAULTS.value,
        config_name="debug_check_internal_defaults",
        build_arguments=build_no_arguments,
        run=run_check_internal_defaults,
    ),
    DebugToolName.CHECK_INTERNAL_INIT_EXPORTS.value: tooling_registry.ToolSpec(
        name=DebugToolName.CHECK_INTERNAL_INIT_EXPORTS.value,
        config_name="debug_check_internal_init_exports",
        build_arguments=build_no_arguments,
        run=run_check_internal_init_exports,
    ),
    DebugToolName.SCHEMA_CHECK.value: tooling_registry.ToolSpec(
        name=DebugToolName.SCHEMA_CHECK.value,
        config_name="debug_schema_check",
        build_arguments=schema_check.build_arguments_from_config,
        run=schema_check.run_tool,
    ),
}


@hydra.main(version_base=None, config_path="../configs", config_name="debug")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Dispatch a debug tool from Hydra configuration."""
    tooling_registry.dispatch_tool(config, TOOLS)


def main() -> None:
    """Run the grouped debug CLI from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
