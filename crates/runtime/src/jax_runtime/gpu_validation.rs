use std::path::Path;

use super::{
    JAX_GPU_DEVICE_PLATFORM_NAME, JAX_RUNTIME_GPU_VALIDATION_FAILED, JAX_RUNTIME_GPU_VALIDATION_SUCCEEDED,
    JaxDeviceObservation, JaxGpuValidationPlan, NVIDIA_CONTROL_DEVICE_PATH, NVIDIA_DRIVER_DIRECTORY_PATH,
    NVIDIA_UVM_DEVICE_PATH, NvidiaDriverProbePathsPayload,
};

#[must_use]
pub fn nvidia_driver_files_are_visible(
    control_device_path: &Path,
    uvm_device_path: &Path,
    driver_directory_path: &Path,
) -> bool {
    control_device_path.exists() || uvm_device_path.exists() || driver_directory_path.exists()
}

#[must_use]
pub fn default_nvidia_driver_probe_paths() -> NvidiaDriverProbePathsPayload {
    NvidiaDriverProbePathsPayload {
        control_device_path: NVIDIA_CONTROL_DEVICE_PATH.to_string(),
        uvm_device_path: NVIDIA_UVM_DEVICE_PATH.to_string(),
        driver_directory_path: NVIDIA_DRIVER_DIRECTORY_PATH.to_string(),
    }
}

#[must_use]
pub fn plan_jax_gpu_validation(
    nvidia_driver_visible: bool,
    backend_initialization_failed: bool,
    devices: &[JaxDeviceObservation],
) -> JaxGpuValidationPlan {
    if !nvidia_driver_visible {
        return JaxGpuValidationPlan {
            status: JAX_RUNTIME_GPU_VALIDATION_FAILED.to_string(),
            message: "JAX GPU execution was requested, but this process cannot see the NVIDIA driver or device files. \
                      Observed no /dev/nvidiactl, no /dev/nvidia-uvm, and no /proc/driver/nvidia. \
                      Run on a GPU allocation/node or expose the NVIDIA devices to this container/session."
                .to_string(),
            should_raise: true,
        };
    }
    if backend_initialization_failed {
        return JaxGpuValidationPlan {
            status: JAX_RUNTIME_GPU_VALIDATION_FAILED.to_string(),
            message: "JAX GPU execution was requested, but no CUDA-enabled JAX backend could be initialized. \
                      The JAX CUDA plugin failed while initializing the backend. Confirm that the process is running \
                      on a GPU node, the NVIDIA driver is loaded, CUDA device files are visible, and the installed \
                      JAX CUDA plugin matches the node driver/runtime. Install the GPU dependency group when needed, \
                      for example: `uv sync --python 3.14 --group dev --group gpu`."
                .to_string(),
            should_raise: true,
        };
    }
    if devices.iter().any(|device| device.platform == JAX_GPU_DEVICE_PLATFORM_NAME) {
        return JaxGpuValidationPlan {
            status: JAX_RUNTIME_GPU_VALIDATION_SUCCEEDED.to_string(),
            message: "JAX reported at least one GPU device.".to_string(),
            should_raise: false,
        };
    }
    let observed_devices = if devices.is_empty() {
        "none".to_string()
    } else {
        devices.iter().map(|device| device.description.as_str()).collect::<Vec<_>>().join(", ")
    };
    JaxGpuValidationPlan {
        status: JAX_RUNTIME_GPU_VALIDATION_FAILED.to_string(),
        message: format!(
            "JAX GPU execution was requested, but JAX did not report any GPU devices. Observed devices: {observed_devices}."
        ),
        should_raise: true,
    }
}
