//! Deterministic JAX runtime setup policy and diagnostics.

const DEVICE_GPU: &str = "gpu";
const JAX_CUDA_PLATFORM_NAME: &str = "cuda";
const JAX_CPU_PLATFORM_NAME: &str = "cpu";
const JAX_GPU_DEVICE_PLATFORM_NAME: &str = "gpu";
const JAX_MATMUL_PRECISION_FLOAT32: &str = "float32";
const JAX_RUNTIME_DIAGNOSTIC_LEVEL_ERROR: &str = "error";
const JAX_RUNTIME_DIAGNOSTIC_LEVEL_INFO: &str = "info";
const JAX_RUNTIME_GPU_VALIDATION_FAILED: &str = "failed";
const JAX_RUNTIME_GPU_VALIDATION_SUCCEEDED: &str = "succeeded";
const XLA_AUXILIARY_CACHE_DISABLED: &str = "none";
const XLA_AUXILIARY_CACHE_PER_FUSION_AUTOTUNE: &str = "xla_gpu_per_fusion_autotune_cache_dir";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxRuntimeSetupPayload {
    pub requested_device: String,
    pub platform_name: String,
    pub cache_directory: String,
    pub matmul_precision: String,
    pub persistent_cache_enabled: bool,
    pub persistent_cache_min_entry_size_bytes: i64,
    pub persistent_cache_min_compile_time_seconds: i64,
    pub xla_auxiliary_cache_mode: String,
    pub xla_auxiliary_cache_reason: String,
    pub transfer_guard_enabled: bool,
    pub gpu_validation_status: String,
    pub gpu_validation_message: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxDeviceObservation {
    pub platform: String,
    pub description: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxGpuValidationPlan {
    pub status: String,
    pub message: String,
    pub should_raise: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JaxRuntimeDiagnosticValue {
    Boolean(bool),
    Integer(i64),
    Text(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxRuntimeDiagnosticFieldPayload {
    pub name: String,
    pub value: JaxRuntimeDiagnosticValue,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JaxRuntimeDiagnosticEventPayload {
    pub event_name: String,
    pub level: String,
    pub message: String,
    pub fields: Vec<JaxRuntimeDiagnosticFieldPayload>,
}

#[allow(clippy::fn_params_excessive_bools)]
#[must_use]
pub fn resolve_jax_runtime_setup(
    requested_device: &str,
    cache_directory: &str,
    matmul_precision: Option<&str>,
    persistent_cache: bool,
    persistent_cache_min_entry_size_bytes: i64,
    persistent_cache_min_compile_time_seconds: i64,
    xla_autotune_cache: bool,
    transfer_guard: bool,
) -> JaxRuntimeSetupPayload {
    let (gpu_validation_status, gpu_validation_message) = if requested_device == DEVICE_GPU {
        ("pending".to_string(), None)
    } else {
        ("skipped".to_string(), Some("CPU runtime requested; GPU validation skipped.".to_string()))
    };
    let platform_name = if requested_device == DEVICE_GPU { JAX_CUDA_PLATFORM_NAME } else { JAX_CPU_PLATFORM_NAME };
    let matmul_precision = matmul_precision.unwrap_or(JAX_MATMUL_PRECISION_FLOAT32).to_string();
    let (xla_auxiliary_cache_mode, xla_auxiliary_cache_reason) = if persistent_cache && xla_autotune_cache {
        (XLA_AUXILIARY_CACHE_PER_FUSION_AUTOTUNE, "XLA auxiliary cache was requested")
    } else if persistent_cache {
        (XLA_AUXILIARY_CACHE_DISABLED, "XLA auxiliary cache was not requested")
    } else {
        (XLA_AUXILIARY_CACHE_DISABLED, "persistent compilation cache is disabled")
    };
    JaxRuntimeSetupPayload {
        requested_device: requested_device.to_string(),
        platform_name: platform_name.to_string(),
        cache_directory: cache_directory.to_string(),
        matmul_precision,
        persistent_cache_enabled: persistent_cache,
        persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds,
        xla_auxiliary_cache_mode: xla_auxiliary_cache_mode.to_string(),
        xla_auxiliary_cache_reason: xla_auxiliary_cache_reason.to_string(),
        transfer_guard_enabled: transfer_guard,
        gpu_validation_status,
        gpu_validation_message,
    }
}

#[must_use]
pub fn build_jax_runtime_setup_diagnostic_events(
    setup: &JaxRuntimeSetupPayload,
) -> Vec<JaxRuntimeDiagnosticEventPayload> {
    let gpu_validation_level = if setup.gpu_validation_status == "failed" {
        JAX_RUNTIME_DIAGNOSTIC_LEVEL_ERROR
    } else {
        JAX_RUNTIME_DIAGNOSTIC_LEVEL_INFO
    };
    let xla_auxiliary_cache_enabled = setup.xla_auxiliary_cache_mode != XLA_AUXILIARY_CACHE_DISABLED;
    vec![
        JaxRuntimeDiagnosticEventPayload {
            event_name: "jax_platform_selected".to_string(),
            level: JAX_RUNTIME_DIAGNOSTIC_LEVEL_INFO.to_string(),
            message: format!("Selected JAX platform {}.", setup.platform_name),
            fields: vec![
                text_field("requested_device", setup.requested_device.clone()),
                text_field("platform", setup.platform_name.clone()),
            ],
        },
        JaxRuntimeDiagnosticEventPayload {
            event_name: "jax_persistent_cache_configured".to_string(),
            level: JAX_RUNTIME_DIAGNOSTIC_LEVEL_INFO.to_string(),
            message: if setup.persistent_cache_enabled {
                "JAX persistent compilation cache enabled.".to_string()
            } else {
                "JAX persistent compilation cache disabled.".to_string()
            },
            fields: vec![
                boolean_field("enabled", setup.persistent_cache_enabled),
                text_field("cache_directory", setup.cache_directory.clone()),
                integer_field("min_entry_size_bytes", setup.persistent_cache_min_entry_size_bytes),
                integer_field("min_compile_time_seconds", setup.persistent_cache_min_compile_time_seconds),
            ],
        },
        JaxRuntimeDiagnosticEventPayload {
            event_name: "jax_xla_auxiliary_cache_configured".to_string(),
            level: JAX_RUNTIME_DIAGNOSTIC_LEVEL_INFO.to_string(),
            message: if xla_auxiliary_cache_enabled {
                "XLA auxiliary persistent cache enabled.".to_string()
            } else {
                "XLA auxiliary persistent cache disabled.".to_string()
            },
            fields: vec![
                boolean_field("enabled", xla_auxiliary_cache_enabled),
                text_field("mode", setup.xla_auxiliary_cache_mode.clone()),
                text_field("reason", setup.xla_auxiliary_cache_reason.clone()),
            ],
        },
        JaxRuntimeDiagnosticEventPayload {
            event_name: "jax_transfer_guard_configured".to_string(),
            level: JAX_RUNTIME_DIAGNOSTIC_LEVEL_INFO.to_string(),
            message: if setup.transfer_guard_enabled {
                "JAX transfer guard diagnostics enabled.".to_string()
            } else {
                "JAX transfer guard diagnostics disabled.".to_string()
            },
            fields: vec![boolean_field("enabled", setup.transfer_guard_enabled)],
        },
        JaxRuntimeDiagnosticEventPayload {
            event_name: "jax_gpu_validation".to_string(),
            level: gpu_validation_level.to_string(),
            message: format!("JAX GPU validation {}.", setup.gpu_validation_status),
            fields: gpu_validation_fields(setup),
        },
    ]
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

fn gpu_validation_fields(setup: &JaxRuntimeSetupPayload) -> Vec<JaxRuntimeDiagnosticFieldPayload> {
    let mut fields = vec![text_field("status", setup.gpu_validation_status.clone())];
    if let Some(message) = setup.gpu_validation_message.clone() {
        fields.push(text_field("message", message));
    }
    fields
}

fn boolean_field(name: &str, value: bool) -> JaxRuntimeDiagnosticFieldPayload {
    JaxRuntimeDiagnosticFieldPayload { name: name.to_string(), value: JaxRuntimeDiagnosticValue::Boolean(value) }
}

fn integer_field(name: &str, value: i64) -> JaxRuntimeDiagnosticFieldPayload {
    JaxRuntimeDiagnosticFieldPayload { name: name.to_string(), value: JaxRuntimeDiagnosticValue::Integer(value) }
}

fn text_field(name: &str, value: String) -> JaxRuntimeDiagnosticFieldPayload {
    JaxRuntimeDiagnosticFieldPayload { name: name.to_string(), value: JaxRuntimeDiagnosticValue::Text(value) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_jax_runtime_setup_payload() {
        let setup = resolve_jax_runtime_setup("gpu", "cache", None, true, 1024, 5, true, true);
        assert_eq!(setup.platform_name, JAX_CUDA_PLATFORM_NAME);
        assert_eq!(setup.matmul_precision, JAX_MATMUL_PRECISION_FLOAT32);
        assert_eq!(setup.xla_auxiliary_cache_mode, XLA_AUXILIARY_CACHE_PER_FUSION_AUTOTUNE);
        assert!(setup.transfer_guard_enabled);

        let cpu_setup = resolve_jax_runtime_setup("cpu", "cache", Some("highest"), false, 0, 0, true, false);
        assert_eq!(cpu_setup.platform_name, JAX_CPU_PLATFORM_NAME);
        assert_eq!(cpu_setup.gpu_validation_status, "skipped");
        assert_eq!(cpu_setup.matmul_precision, "highest");
    }

    #[test]
    fn builds_jax_runtime_setup_diagnostic_events() {
        let setup = resolve_jax_runtime_setup("cpu", "cache", None, false, 0, 0, true, true);

        let events = build_jax_runtime_setup_diagnostic_events(&setup);

        assert_eq!(events.iter().map(|event| event.event_name.as_str()).collect::<Vec<_>>(), {
            vec![
                "jax_platform_selected",
                "jax_persistent_cache_configured",
                "jax_xla_auxiliary_cache_configured",
                "jax_transfer_guard_configured",
                "jax_gpu_validation",
            ]
        });
        assert_eq!(events[0].fields[1].value, JaxRuntimeDiagnosticValue::Text("cpu".to_string()));
        assert_eq!(events[1].fields[0].value, JaxRuntimeDiagnosticValue::Boolean(false));
        assert_eq!(events[2].fields[0].value, JaxRuntimeDiagnosticValue::Boolean(false));
        assert_eq!(events[3].fields[0].value, JaxRuntimeDiagnosticValue::Boolean(true));
        assert_eq!(events[4].fields[0].value, JaxRuntimeDiagnosticValue::Text("skipped".to_string()));
    }

    #[test]
    fn marks_failed_jax_gpu_validation_diagnostic_as_error() {
        let mut setup = resolve_jax_runtime_setup("gpu", "cache", None, true, 0, 0, false, false);
        setup.gpu_validation_status = "failed".to_string();
        setup.gpu_validation_message = Some("no gpu".to_string());

        let events = build_jax_runtime_setup_diagnostic_events(&setup);

        assert_eq!(events[4].level, JAX_RUNTIME_DIAGNOSTIC_LEVEL_ERROR);
        assert_eq!(events[4].fields[1].value, JaxRuntimeDiagnosticValue::Text("no gpu".to_string()));
    }

    #[test]
    fn plans_jax_gpu_validation_outcomes() {
        let missing_driver_plan = plan_jax_gpu_validation(false, false, &[]);
        assert_eq!(missing_driver_plan.status, JAX_RUNTIME_GPU_VALIDATION_FAILED);
        assert!(missing_driver_plan.should_raise);
        assert!(missing_driver_plan.message.contains("cannot see the NVIDIA driver"));

        let backend_failure_plan = plan_jax_gpu_validation(true, true, &[]);
        assert_eq!(backend_failure_plan.status, JAX_RUNTIME_GPU_VALIDATION_FAILED);
        assert!(backend_failure_plan.should_raise);
        assert!(backend_failure_plan.message.contains("no CUDA-enabled JAX backend"));

        let cpu_only_plan = plan_jax_gpu_validation(
            true,
            false,
            &[JaxDeviceObservation { platform: "cpu".to_string(), description: "CpuDevice(id=0)".to_string() }],
        );
        assert_eq!(cpu_only_plan.status, JAX_RUNTIME_GPU_VALIDATION_FAILED);
        assert!(cpu_only_plan.should_raise);
        assert!(cpu_only_plan.message.contains("Observed devices: CpuDevice(id=0)."));

        let gpu_plan = plan_jax_gpu_validation(
            true,
            false,
            &[JaxDeviceObservation {
                platform: JAX_GPU_DEVICE_PLATFORM_NAME.to_string(),
                description: "GpuDevice(id=0)".to_string(),
            }],
        );
        assert_eq!(gpu_plan.status, JAX_RUNTIME_GPU_VALIDATION_SUCCEEDED);
        assert!(!gpu_plan.should_raise);
        assert_eq!(gpu_plan.message, "JAX reported at least one GPU device.");
    }
}
