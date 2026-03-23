#!/usr/bin/env python3
"""Probe JAX runtime initialization safely in subprocesses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

NVIDIA_DRIVER_LIBRARY_DIRECTORY = Path("/run/opengl-driver/lib")


@dataclass(frozen=True)
class ProbeResult:
    """Structured result for one JAX runtime probe."""

    probe_name: str
    success: bool
    return_code: int
    stdout: str
    stderr: str


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


def main() -> None:
    """Run default and CPU-forced JAX probes and print a JSON report."""
    gpu_library_environment = build_gpu_library_environment()
    probe_results = [
        run_probe(probe_name="default", environment_overrides={}),
        run_probe(probe_name="gpu_driver_path", environment_overrides=gpu_library_environment),
        run_probe(probe_name="cpu_forced", environment_overrides={"JAX_PLATFORMS": "cpu"}),
    ]
    print(json.dumps([asdict(probe_result) for probe_result in probe_results], indent=2))


if __name__ == "__main__":
    main()
