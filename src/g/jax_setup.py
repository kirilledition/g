"""JAX runtime configuration for the GWAS engine."""

from __future__ import annotations

import getpass
import typing
from pathlib import Path

import jax
import numpy as np

from g import types

DEFAULT_NODE_LOCAL_CACHE_ROOT = Path("/tmp")
DEFAULT_CACHE_DIRECTORY_NAME = "g-jax-cache"
XLA_AUTOTUNE_CACHE_OPTION = "xla_gpu_per_fusion_autotune_cache_dir"
DISABLE_XLA_CACHE_OPTION = "none"


def default_node_local_jax_compilation_cache_directory() -> Path:
    """Build the default node-local JAX compilation cache directory."""
    user_name = getpass.getuser() or "unknown"
    return DEFAULT_NODE_LOCAL_CACHE_ROOT / user_name / DEFAULT_CACHE_DIRECTORY_NAME


def path_is_beegfs(path: Path) -> bool:
    """Return whether a path is on the BeeGFS mount used by this project."""
    expanded_path = path.expanduser()
    return str(expanded_path).startswith("/mnt/beegfs/")


def path_is_node_local(path: Path) -> bool:
    """Return whether a cache path is safe for node-local XLA auxiliary caches."""
    expanded_path = path.expanduser()
    return str(expanded_path).startswith("/tmp/") or str(expanded_path) == "/tmp"


FLOAT_DTYPE = np.float32
JAX_ENABLE_X64 = False
DEFAULT_MATMUL_PRECISION = "float32"
ENABLE_PERSISTENT_COMPILATION_CACHE = True
PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES = -1
PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS = 0
CUDA_PLATFORM_NAME = "cuda"
GPU_DEVICE_PLATFORM_NAME = "gpu"
NVIDIA_CONTROL_DEVICE_PATH = Path("/dev/nvidiactl")
NVIDIA_UVM_DEVICE_PATH = Path("/dev/nvidia-uvm")
NVIDIA_DRIVER_DIRECTORY_PATH = Path("/proc/driver/nvidia")


def resolve_jax_compilation_cache_directory(cache_directory: Path | None = None) -> Path:
    """Resolve the persistent JAX compilation cache directory."""
    if cache_directory is not None:
        return cache_directory.expanduser()
    return default_node_local_jax_compilation_cache_directory()


def resolve_xla_cache_option(cache_directory: Path, *, enable_xla_autotune_cache: bool = False) -> str:
    """Resolve whether XLA auxiliary persistent caches should be enabled."""
    if enable_xla_autotune_cache and path_is_node_local(cache_directory) and not path_is_beegfs(cache_directory):
        return XLA_AUTOTUNE_CACHE_OPTION
    return DISABLE_XLA_CACHE_OPTION


def transfer_guard_diagnostics_enabled(*, enable_transfer_guard: bool = False) -> bool:
    """Return whether transfer guard diagnostics should disallow implicit transfers."""
    return enable_transfer_guard


def nvidia_driver_is_visible() -> bool:
    """Return whether the process can see a Linux NVIDIA driver/device mount."""
    return (
        NVIDIA_CONTROL_DEVICE_PATH.exists() or NVIDIA_UVM_DEVICE_PATH.exists() or NVIDIA_DRIVER_DIRECTORY_PATH.exists()
    )


def configure_jax_platform(device: types.Device) -> None:
    """Configure the JAX platform without initializing a backend."""
    if device == types.Device.GPU:
        jax.config.update("jax_platforms", CUDA_PLATFORM_NAME)
    else:
        jax.config.update("jax_platforms", "cpu")


def configure_jax_runtime(
    *,
    cache_directory: Path | None = None,
    matmul_precision: types.JaxMatmulPrecision | None = None,
    persistent_cache: bool = ENABLE_PERSISTENT_COMPILATION_CACHE,
    persistent_cache_min_entry_size_bytes: int = PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
    persistent_cache_min_compile_time_seconds: int = PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS,
    xla_autotune_cache: bool = False,
    transfer_guard: bool = False,
) -> None:
    """Configure JAX runtime knobs before engine modules are imported."""
    jax.config.update("jax_enable_x64", JAX_ENABLE_X64)
    precision_value = DEFAULT_MATMUL_PRECISION if matmul_precision is None else matmul_precision.value
    jax.config.update("jax_default_matmul_precision", precision_value)
    if persistent_cache:
        resolved_cache_directory = resolve_jax_compilation_cache_directory(cache_directory)
        resolved_cache_directory.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(resolved_cache_directory))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", persistent_cache_min_entry_size_bytes)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", persistent_cache_min_compile_time_seconds)
        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            resolve_xla_cache_option(
                resolved_cache_directory,
                enable_xla_autotune_cache=xla_autotune_cache,
            ),
        )
    if transfer_guard_diagnostics_enabled(enable_transfer_guard=transfer_guard):
        jax.config.update("jax_transfer_guard", "disallow")


def configure_jax_runtime_before_backend_init(
    *,
    device: types.Device,
    cache_directory: Path | None = None,
    matmul_precision: types.JaxMatmulPrecision | None = None,
    persistent_cache: bool = ENABLE_PERSISTENT_COMPILATION_CACHE,
    persistent_cache_min_entry_size_bytes: int = PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
    persistent_cache_min_compile_time_seconds: int = PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS,
    xla_autotune_cache: bool = False,
    transfer_guard: bool = False,
) -> None:
    """Configure JAX platform and runtime knobs before backend initialization."""
    configure_jax_platform(device)
    configure_jax_runtime(
        cache_directory=cache_directory,
        matmul_precision=matmul_precision,
        persistent_cache=persistent_cache,
        persistent_cache_min_entry_size_bytes=persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds=persistent_cache_min_compile_time_seconds,
        xla_autotune_cache=xla_autotune_cache,
        transfer_guard=transfer_guard,
    )
    if device == types.Device.GPU:
        require_gpu_device()


def configure_jax_device(device: types.Device) -> None:
    """Configure the JAX execution device.

    Args:
        device: Device enum specifying CPU or GPU execution.

    """
    configure_jax_platform(device)
    if device == types.Device.GPU:
        require_gpu_device()


def require_gpu_device() -> None:
    """Raise when JAX cannot initialize a GPU backend."""
    if not nvidia_driver_is_visible():
        message = (
            "JAX GPU execution was requested, but this process cannot see the NVIDIA driver or device files. "
            "Observed no /dev/nvidiactl, no /dev/nvidia-uvm, and no /proc/driver/nvidia. "
            "Run on a GPU allocation/node or expose the NVIDIA devices to this container/session."
        )
        raise RuntimeError(message)
    try:
        devices = jax.devices()
    except Exception as error:
        message = (
            "JAX GPU execution was requested, but no CUDA-enabled JAX backend could be initialized. "
            "The JAX CUDA plugin failed while initializing the backend. Confirm that the process is running on a "
            "GPU node, the NVIDIA driver is loaded, CUDA device files are visible, and the installed JAX CUDA plugin "
            "matches the node driver/runtime. Install the GPU dependency group when needed, for example: "
            "`uv sync --python 3.14 --group dev --group gpu`."
        )
        raise RuntimeError(message) from error
    gpu_devices = [
        device
        for device in devices
        if getattr(typing.cast("typing.Any", device), "platform", None) == GPU_DEVICE_PLATFORM_NAME
    ]
    if not gpu_devices:
        observed_devices = ", ".join(str(device) for device in devices) or "none"
        message = (
            "JAX GPU execution was requested, but JAX did not report any GPU devices. "
            f"Observed devices: {observed_devices}."
        )
        raise RuntimeError(message)
