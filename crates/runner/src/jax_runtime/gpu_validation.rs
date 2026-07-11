use std::borrow::Cow;
use std::path::Path;

use super::{
    JAX_GPU_DEVICE_PLATFORM_NAME, JaxDevice, JaxGpuValidationPlan, JaxGpuValidationStatus, NVIDIA_CONTROL_DEVICE_PATH,
    NVIDIA_DRIVER_DIRECTORY_PATH, NVIDIA_UVM_DEVICE_PATH,
};

#[must_use]
pub(crate) fn nvidia_driver_files_are_visible() -> bool {
    Path::new(NVIDIA_CONTROL_DEVICE_PATH).exists()
        || Path::new(NVIDIA_UVM_DEVICE_PATH).exists()
        || Path::new(NVIDIA_DRIVER_DIRECTORY_PATH).exists()
}

#[must_use]
pub(crate) fn plan_jax_gpu_validation(
    nvidia_driver_visible: bool,
    backend_initialization_failed: bool,
    devices: &[JaxDevice],
) -> JaxGpuValidationPlan {
    if !nvidia_driver_visible {
        return JaxGpuValidationPlan {
            status: JaxGpuValidationStatus::Failed,
            message: "JAX GPU execution was requested, but this process cannot see the NVIDIA driver or device files. \
                      Observed no /dev/nvidiactl, no /dev/nvidia-uvm, and no /proc/driver/nvidia. \
                      Run on a GPU allocation/node or expose the NVIDIA devices to this container/session."
                .into(),
        };
    }
    if backend_initialization_failed {
        return JaxGpuValidationPlan {
            status: JaxGpuValidationStatus::Failed,
            message: "JAX GPU execution was requested, but no CUDA-enabled JAX backend could be initialized. \
                      The JAX CUDA plugin failed while initializing the backend. Confirm that the process is running \
                      on a GPU node, the NVIDIA driver is loaded, CUDA device files are visible, and the installed \
                      JAX CUDA plugin matches the node driver/runtime. Install the GPU dependency group when needed, \
                      for example: `uv sync --python 3.14 --group dev --group gpu`."
                .into(),
        };
    }
    if devices.iter().any(|device| device.platform == JAX_GPU_DEVICE_PLATFORM_NAME) {
        return JaxGpuValidationPlan {
            status: JaxGpuValidationStatus::Succeeded,
            message: Cow::Borrowed("JAX reported at least one GPU device."),
        };
    }
    let mut message =
        String::from("JAX GPU execution was requested, but JAX did not report any GPU devices. Observed devices: ");
    if devices.is_empty() {
        message.push_str("none");
    } else {
        for (device_index, device) in devices.iter().enumerate() {
            if device_index != 0 {
                message.push_str(", ");
            }
            message.push_str(&device.description);
        }
    }
    message.push('.');
    JaxGpuValidationPlan { status: JaxGpuValidationStatus::Failed, message: Cow::Owned(message) }
}
