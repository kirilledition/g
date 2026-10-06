"""Profiler availability checks and serialization."""

from __future__ import annotations

import dataclasses
import shutil
import sys

from tooling.profile_deep import environment as profile_deep_environment
from tooling.profile_deep import models as profile_deep_models


def build_uv_injected_profiler_status(
    *,
    tool_name: str,
    executable_name: str,
    module_name: str,
    enabled: bool,
) -> profile_deep_models.ProfilerToolStatus:
    """Build availability for Python profilers that must see project dependencies."""
    if profile_deep_environment.python_module_is_available(module_name):
        return profile_deep_models.ProfilerToolStatus(
            tool_name=tool_name,
            enabled=enabled,
            available=True,
            executable_path=sys.executable,
            notes=f"{module_name} is importable in the project Python environment.",
        )
    uv_executable_path = shutil.which("uv")
    if uv_executable_path is not None:
        return profile_deep_models.ProfilerToolStatus(
            tool_name=tool_name,
            enabled=enabled,
            available=True,
            executable_path=uv_executable_path,
            notes=(
                f"{executable_name} will run through uv --no-sync --with {module_name} "
                "to preserve the project Python environment."
            ),
        )
    return profile_deep_models.ProfilerToolStatus(
        tool_name=tool_name,
        enabled=enabled,
        available=False,
        executable_path=None,
        notes=f"{module_name} is not importable in the project Python environment and uv is not on PATH.",
    )


def build_profiler_tool_status(
    arguments: profile_deep_models.ProfileArguments,
) -> dict[str, profile_deep_models.ProfilerToolStatus]:
    """Build profiler tool availability records for the current host."""
    optional_executable_tools = {
        "py_spy": ("py-spy", arguments.enable_py_spy),
        "linux_perf": ("perf", arguments.enable_linux_perf),
        "nsight_systems": ("nsys", arguments.enable_nsight_systems),
        "nsight_compute": ("ncu", arguments.enable_nsight_compute),
    }
    tool_status = {
        "python_cprofile": profile_deep_models.ProfilerToolStatus(
            tool_name="python_cprofile",
            enabled=arguments.enable_python_cprofile,
            available=True,
            executable_path=sys.executable,
            notes="Python cProfile is part of the standard library.",
        ),
        "jax_trace": profile_deep_models.ProfilerToolStatus(
            tool_name="jax_trace",
            enabled=arguments.enable_jax_trace,
            available=True,
            executable_path=None,
            notes="JAX profiler trace capture is provided by the installed JAX package.",
        ),
        "jax_memory_profile": profile_deep_models.ProfilerToolStatus(
            tool_name="jax_memory_profile",
            enabled=arguments.enable_jax_memory_profile,
            available=True,
            executable_path=None,
            notes="JAX device memory capture is provided by the installed JAX package.",
        ),
        "scalene": build_uv_injected_profiler_status(
            tool_name="scalene",
            executable_name="scalene",
            module_name="scalene",
            enabled=arguments.enable_scalene,
        ),
        "memray": build_uv_injected_profiler_status(
            tool_name="memray",
            executable_name="memray",
            module_name="memray",
            enabled=arguments.enable_memray,
        ),
    }
    for tool_name, (executable_name, enabled) in optional_executable_tools.items():
        executable_path = shutil.which(executable_name)
        available = executable_path is not None
        notes = f"{executable_name} is available on PATH." if available else f"{executable_name} is not on PATH."
        tool_status[tool_name] = profile_deep_models.ProfilerToolStatus(
            tool_name=tool_name,
            enabled=enabled,
            available=available,
            executable_path=executable_path,
            notes=notes,
        )
    return tool_status


def serialize_profiler_tool_status(
    tool_status: dict[str, profile_deep_models.ProfilerToolStatus],
) -> dict[str, dict[str, object]]:
    """Serialize profiler tool availability records."""
    return {tool_name: dataclasses.asdict(status) for tool_name, status in sorted(tool_status.items())}
