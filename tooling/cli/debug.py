"""Grouped Hydra CLI for debug and parity tooling."""

from __future__ import annotations

import enum
import typing

import hydra

from tooling.cli import schema_check
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import registry as tooling_registry
from tooling.debug import (
    check_cuda_native,
    check_internal_defaults,
    check_internal_init_exports,
    check_justfile,
    check_native_cli_frontend,
    check_pyo3_stub,
    check_python_architecture,
    check_rust_architecture,
)

if typing.TYPE_CHECKING:
    import omegaconf


class DebugToolName(enum.StrEnum):
    """Available grouped debug tools."""

    CHECK_CUDA_NATIVE = "check_cuda_native"
    CHECK_INTERNAL_DEFAULTS = "check_internal_defaults"
    CHECK_INTERNAL_INIT_EXPORTS = "check_internal_init_exports"
    CHECK_JUSTFILE = "check_justfile"
    CHECK_NATIVE_CLI_FRONTEND = "check_native_cli_frontend"
    CHECK_PYTHON_ARCHITECTURE = "check_python_architecture"
    CHECK_PYO3_STUB = "check_pyo3_stub"
    CHECK_RUST_ARCHITECTURE = "check_rust_architecture"
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


def run_check_cuda_native(arguments: check_cuda_native.CudaNativeCheckArguments) -> None:
    """Run CUDA native static analysis."""
    exit_code = check_cuda_native.run_tool(arguments)
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


def run_check_justfile(arguments: check_justfile.JustfileCheckArguments) -> None:
    """Run the Justfile command-surface guardrail."""
    exit_code = check_justfile.run_tool(arguments)
    if exit_code:
        raise SystemExit(exit_code)


def run_check_native_cli_frontend(arguments: check_native_cli_frontend.NativeCliFrontendCheckArguments) -> None:
    """Run the native CLI frontend parity and startup guardrail."""
    exit_code = check_native_cli_frontend.run_tool(arguments)
    if exit_code:
        raise SystemExit(exit_code)


def run_check_rust_architecture(arguments: None) -> None:
    """Run the Rust workspace architecture guardrail."""
    del arguments
    exit_code = check_rust_architecture.run_tool(check_rust_architecture.REPOSITORY_ROOT)
    if exit_code:
        raise SystemExit(exit_code)


def run_check_python_architecture(arguments: None) -> None:
    """Run the Python package architecture guardrail."""
    del arguments
    exit_code = check_python_architecture.run_tool(check_python_architecture.PRODUCTION_PACKAGE_ROOT)
    if exit_code:
        raise SystemExit(exit_code)


TOOLS: dict[str, tooling_registry.ToolSpec[typing.Any]] = {
    DebugToolName.CHECK_CUDA_NATIVE.value: tooling_registry.ToolSpec(
        build_arguments=check_cuda_native.build_arguments_from_config,
        run=run_check_cuda_native,
    ),
    DebugToolName.CHECK_PYO3_STUB.value: tooling_registry.ToolSpec(
        build_arguments=build_no_arguments,
        run=run_check_pyo3_stub,
    ),
    DebugToolName.CHECK_INTERNAL_DEFAULTS.value: tooling_registry.ToolSpec(
        build_arguments=build_no_arguments,
        run=run_check_internal_defaults,
    ),
    DebugToolName.CHECK_INTERNAL_INIT_EXPORTS.value: tooling_registry.ToolSpec(
        build_arguments=build_no_arguments,
        run=run_check_internal_init_exports,
    ),
    DebugToolName.CHECK_JUSTFILE.value: tooling_registry.ToolSpec(
        build_arguments=check_justfile.build_arguments_from_config,
        run=run_check_justfile,
    ),
    DebugToolName.CHECK_NATIVE_CLI_FRONTEND.value: tooling_registry.ToolSpec(
        build_arguments=check_native_cli_frontend.build_arguments_from_config,
        run=run_check_native_cli_frontend,
    ),
    DebugToolName.CHECK_RUST_ARCHITECTURE.value: tooling_registry.ToolSpec(
        build_arguments=build_no_arguments,
        run=run_check_rust_architecture,
    ),
    DebugToolName.CHECK_PYTHON_ARCHITECTURE.value: tooling_registry.ToolSpec(
        build_arguments=build_no_arguments,
        run=run_check_python_architecture,
    ),
    DebugToolName.SCHEMA_CHECK.value: tooling_registry.ToolSpec(
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
