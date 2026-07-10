use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};

use super::{
    JAX_RUNTIME_DIAGNOSTIC_LEVEL_ERROR, JAX_RUNTIME_DIAGNOSTIC_LEVEL_INFO, JaxRuntimeDiagnosticEventPayload,
    JaxRuntimeDiagnosticFieldPayload, JaxRuntimeDiagnosticFields, JaxRuntimeDiagnosticRecordPlan,
    JaxRuntimeDiagnosticValue, JaxRuntimeSetupPayload, PYTHON_LOGGING_LEVEL_ERROR, PYTHON_LOGGING_LEVEL_INFO,
    XLA_AUXILIARY_CACHE_DISABLED,
};

impl Serialize for JaxRuntimeDiagnosticValue {
    fn serialize<SerializerType>(&self, serializer: SerializerType) -> Result<SerializerType::Ok, SerializerType::Error>
    where
        SerializerType: Serializer,
    {
        match self {
            Self::Boolean(value) => serializer.serialize_bool(*value),
            Self::Integer(value) => serializer.serialize_i64(*value),
            Self::Text(value) => serializer.serialize_str(value),
        }
    }
}

impl Serialize for JaxRuntimeDiagnosticFields<'_> {
    fn serialize<SerializerType>(&self, serializer: SerializerType) -> Result<SerializerType::Ok, SerializerType::Error>
    where
        SerializerType: Serializer,
    {
        let mut fields = serializer.serialize_map(Some(self.fields.len()))?;
        for field in self.fields {
            fields.serialize_entry(&field.name, &field.value)?;
        }
        fields.end()
    }
}

/// Serialize JAX runtime diagnostic fields for native diagnostic emission.
///
/// This keeps JAX diagnostic field JSON shape in `g-runtime`; PyO3 callers only
/// pass the serialized fields through to the logging boundary.
///
/// # Errors
///
/// Returns a serialization error if the diagnostic field payload cannot be
/// encoded as JSON.
pub fn serialize_jax_runtime_diagnostic_fields_json(
    fields: &[JaxRuntimeDiagnosticFieldPayload],
) -> Result<String, serde_json::Error> {
    serde_json::to_string(&JaxRuntimeDiagnosticFields::new(fields))
}

#[must_use]
pub fn plan_jax_runtime_diagnostic_record(
    diagnostic_level: &str,
    has_telemetry_session: bool,
) -> JaxRuntimeDiagnosticRecordPlan {
    let logging_level_name = if diagnostic_level == JAX_RUNTIME_DIAGNOSTIC_LEVEL_ERROR {
        PYTHON_LOGGING_LEVEL_ERROR
    } else {
        PYTHON_LOGGING_LEVEL_INFO
    };
    JaxRuntimeDiagnosticRecordPlan {
        logging_level_name: logging_level_name.to_string(),
        should_emit_telemetry: has_telemetry_session,
        telemetry_level: diagnostic_level.to_string(),
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
