use super::{
    JAX_CONFIG_COMPILATION_CACHE_DIR, JAX_CONFIG_DEFAULT_MATMUL_PRECISION, JAX_CONFIG_ENABLE_X64,
    JAX_CONFIG_PERSISTENT_CACHE_ENABLE_XLA_CACHES, JAX_CONFIG_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS,
    JAX_CONFIG_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES, JAX_CONFIG_PLATFORMS, JAX_CONFIG_TRANSFER_GUARD,
    JAX_TRANSFER_GUARD_DISALLOW, JaxRuntimeConfigUpdatePayload, JaxRuntimeConfigValue, JaxRuntimeSetupPayload,
};

#[must_use]
pub fn plan_jax_runtime_config_updates(setup: &JaxRuntimeSetupPayload) -> Vec<JaxRuntimeConfigUpdatePayload> {
    let mut updates = vec![
        text_config_update(JAX_CONFIG_PLATFORMS, setup.platform_name.clone()),
        boolean_config_update(JAX_CONFIG_ENABLE_X64, true),
        text_config_update(JAX_CONFIG_DEFAULT_MATMUL_PRECISION, setup.matmul_precision.clone()),
    ];
    if setup.persistent_cache_enabled {
        updates.extend([
            text_config_update(JAX_CONFIG_COMPILATION_CACHE_DIR, setup.cache_directory.clone()),
            integer_config_update(
                JAX_CONFIG_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
                setup.persistent_cache_min_entry_size_bytes,
            ),
            integer_config_update(
                JAX_CONFIG_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS,
                setup.persistent_cache_min_compile_time_seconds,
            ),
            text_config_update(JAX_CONFIG_PERSISTENT_CACHE_ENABLE_XLA_CACHES, setup.xla_auxiliary_cache_mode.clone()),
        ]);
    }
    if setup.transfer_guard_enabled {
        updates.push(text_config_update(JAX_CONFIG_TRANSFER_GUARD, JAX_TRANSFER_GUARD_DISALLOW.to_string()));
    }
    updates
}

fn boolean_config_update(name: &str, value: bool) -> JaxRuntimeConfigUpdatePayload {
    JaxRuntimeConfigUpdatePayload { setting_name: name.to_string(), value: JaxRuntimeConfigValue::Boolean(value) }
}

fn integer_config_update(name: &str, value: i64) -> JaxRuntimeConfigUpdatePayload {
    JaxRuntimeConfigUpdatePayload { setting_name: name.to_string(), value: JaxRuntimeConfigValue::Integer(value) }
}

fn text_config_update(name: &str, value: String) -> JaxRuntimeConfigUpdatePayload {
    JaxRuntimeConfigUpdatePayload { setting_name: name.to_string(), value: JaxRuntimeConfigValue::Text(value) }
}
