"""Child-process environments and reproducibility metadata."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import typing
from datetime import UTC, datetime
from pathlib import Path

from tooling.benchmark import benchmark as baseline_benchmark
from tooling.profile_deep import config as profile_deep_config
from tooling.profile_deep import models as profile_deep_models

DEFAULT_VARIANT_COUNT = 418_943


JAX_DEBUG_LOG_MODULES = "jax._src.compiler,jax._src.lru_cache"


def executable_is_available(executable_name: str) -> bool:
    """Return whether a command or explicit executable path is available."""
    executable_path = Path(executable_name)
    if executable_path.is_absolute() or executable_path.parent != Path():
        return executable_path.exists() and os.access(executable_path, os.X_OK)
    return shutil.which(executable_name) is not None


def resolve_available_regenie_executable(arguments: profile_deep_models.ProfileArguments) -> str | None:
    """Resolve REGENIE for optional baseline runs without failing the campaign."""
    executable_name = profile_deep_config.configured_regenie_executable(arguments)
    if executable_is_available(executable_name):
        return executable_name
    return None


def resolved_binary_path(executable_name: str | None) -> str | None:
    """Resolve a command name to an absolute executable path when possible."""
    if executable_name is None:
        return None
    executable_path = Path(executable_name)
    if executable_path.is_absolute() or executable_path.parent != Path():
        if executable_path.exists():
            return str(executable_path.resolve())
        return executable_name
    resolved_path = shutil.which(executable_name)
    if resolved_path is not None:
        return resolved_path
    return executable_name


def python_module_is_available(module_name: str) -> bool:
    """Return whether a module is importable in the active Python environment."""
    return importlib.util.find_spec(module_name) is not None


def command_output(
    command_arguments: list[str],
    environment_overrides: dict[str, str] | None = None,
) -> dict[str, typing.Any]:
    """Run a metadata command and return captured output."""
    environment = dict(os.environ)
    if environment_overrides is not None:
        environment.update(environment_overrides)
    try:
        completed_process = subprocess.run(
            command_arguments,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    except FileNotFoundError as error:
        return {
            "command": command_arguments,
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
        }
    return {
        "command": command_arguments,
        "returncode": completed_process.returncode,
        "stdout": completed_process.stdout,
        "stderr": completed_process.stderr,
    }


def dirty_diff_sha256() -> str:
    """Hash the current dirty diff without writing it into the report."""
    completed_process = subprocess.run(["git", "diff"], check=False, capture_output=True)
    return hashlib.sha256(completed_process.stdout).hexdigest()


def collect_environment_metadata(
    baseline_paths: typing.Any,
    regenie_executable: str | None = None,
) -> dict[str, typing.Any]:
    """Collect reproducibility metadata for a profiling campaign."""
    input_paths = [
        baseline_paths.bgen_path,
        baseline_paths.sample_path,
        baseline_paths.continuous_phenotype_path,
        baseline_paths.binary_phenotype_path,
        baseline_paths.covariate_path,
        baseline_paths.regenie_prediction_list_path,
        baseline_paths.regenie_qt_prediction_list_path,
    ]
    file_sizes = {
        str(input_path): input_path.stat().st_size
        for input_path in input_paths
        if input_path is not None and input_path.exists()
    }
    relevant_environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("G_", "GWAS_ENGINE_", "JAX_", "XLA_", "CUDA_", "RAYON_", "SLURM_"))
    }
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "dirty_diff_sha256": dirty_diff_sha256(),
        "lscpu": command_output(["lscpu"]),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "python": command_output([sys.executable, "--version"]),
        "jax": command_output([sys.executable, "-c", "import jax; print(jax.__version__); print(jax.devices())"]),
        "rustc": command_output(["rustc", "--version"]),
        "cargo": command_output(["cargo", "--version"]),
        "regenie": command_output([regenie_executable or "regenie", "--version"]),
        "hardware": dataclasses.asdict(baseline_benchmark.collect_hardware_summary()),
        "environment": relevant_environment,
        "input_file_sizes": file_sizes,
        "expected_full_variant_count": DEFAULT_VARIANT_COUNT,
    }


def build_g_trial_environment(
    *,
    enable_jax_debug_logging: bool,
) -> dict[str, str]:
    """Build child process environment overrides for one g trial."""
    environment: dict[str, str] = {}
    if enable_jax_debug_logging:
        environment.update(
            {
                "JAX_DEBUG_LOG_MODULES": JAX_DEBUG_LOG_MODULES,
                "JAX_LOGGING_LEVEL": "DEBUG",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": ".50",
            }
        )
    return environment


def executable_name(executable_path: str | None) -> str:
    """Return the command basename for an optional executable path."""
    if executable_path is None:
        return ""
    return Path(executable_path).name
