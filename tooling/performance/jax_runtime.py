#!/usr/bin/env python3
"""Probe JAX runtime initialization safely in subprocesses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import typing
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra

from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat

if typing.TYPE_CHECKING:
    import omegaconf

NVIDIA_DRIVER_LIBRARY_DIRECTORY = Path("/run/opengl-driver/lib")


@dataclass(frozen=True)
class ProbeResult:
    """Structured result for one JAX runtime probe."""

    probe_name: str
    success: bool
    return_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class JaxRuntimeArguments:
    """Resolved parameters for the JAX runtime probe.

    Attributes:
        include_default: Whether to probe the default environment.
        include_gpu_driver_path: Whether to probe with NVIDIA driver libraries on `LD_LIBRARY_PATH`.
        include_cpu_forced: Whether to probe with `JAX_PLATFORMS=cpu`.

    """

    include_default: bool
    include_gpu_driver_path: bool
    include_cpu_forced: bool


def run_probe(probe_name: str, environment_overrides: dict[str, str]) -> ProbeResult:
    """Run one JAX initialization probe in a subprocess."""
    environment = os.environ.copy()
    environment.update(environment_overrides)
    completed_process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, jax; "
                "print(json.dumps({"
                "'default_backend': jax.default_backend(), "
                "'devices': [str(device) for device in jax.devices()]"
                "}))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return ProbeResult(
        probe_name=probe_name,
        success=completed_process.returncode == 0,
        return_code=completed_process.returncode,
        stdout=completed_process.stdout,
        stderr=completed_process.stderr,
    )


def build_gpu_library_environment() -> dict[str, str]:
    """Build environment overrides that expose the NVIDIA driver libraries."""
    if not NVIDIA_DRIVER_LIBRARY_DIRECTORY.exists():
        return {}
    existing_library_path = os.environ.get("LD_LIBRARY_PATH")
    gpu_library_path = str(NVIDIA_DRIVER_LIBRARY_DIRECTORY)
    combined_library_path = (
        gpu_library_path if not existing_library_path else f"{gpu_library_path}:{existing_library_path}"
    )
    return {"LD_LIBRARY_PATH": combined_library_path}


def run_tool(arguments: JaxRuntimeArguments) -> None:
    """Run default and CPU-forced JAX probes and print a JSON report."""
    gpu_library_environment = build_gpu_library_environment()
    probe_results: list[ProbeResult] = []
    if arguments.include_default:
        probe_results.append(run_probe(probe_name="default", environment_overrides={}))
    if arguments.include_gpu_driver_path:
        probe_results.append(
            run_probe(probe_name="gpu_driver_path", environment_overrides=gpu_library_environment),
        )
    if arguments.include_cpu_forced:
        probe_results.append(run_probe(probe_name="cpu_forced", environment_overrides={"JAX_PLATFORMS": "cpu"}))
    print(json.dumps([asdict(probe_result) for probe_result in probe_results], indent=2))


def build_arguments_from_config(config: omegaconf.DictConfig) -> JaxRuntimeArguments:
    """Resolve JAX runtime probe parameters from Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return JaxRuntimeArguments(
        include_default=tooling_hydra_arguments.boolean_value(tool_values["include_default"]),
        include_gpu_driver_path=tooling_hydra_arguments.boolean_value(tool_values["include_gpu_driver_path"]),
        include_cpu_forced=tooling_hydra_arguments.boolean_value(tool_values["include_cpu_forced"]),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="performance_jax_runtime")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run JAX runtime probes from Hydra configuration."""
    run_tool(build_arguments_from_config(config))


def main() -> None:
    """Run JAX runtime probes from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
