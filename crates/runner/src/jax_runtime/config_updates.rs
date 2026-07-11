use std::borrow::Cow;

use super::{
    JAX_CONFIG_COMPILATION_CACHE_DIR, JAX_CONFIG_DEFAULT_MATMUL_PRECISION, JAX_CONFIG_ENABLE_X64,
    JAX_CONFIG_PERSISTENT_CACHE_ENABLE_XLA_CACHES, JAX_CONFIG_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS,
    JAX_CONFIG_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES, JAX_CONFIG_PLATFORMS, JAX_CONFIG_TRANSFER_GUARD,
    JAX_TRANSFER_GUARD_DISALLOW, JaxRuntimeConfigUpdate, JaxRuntimeConfigValue, JaxRuntimeSetupSession,
};

#[must_use]
pub(crate) fn plan_jax_runtime_config_updates<'policy>(
    setup_session: &JaxRuntimeSetupSession<'policy>,
) -> Vec<JaxRuntimeConfigUpdate<'policy>> {
    let policy = setup_session.policy;
    let update_count =
        3 + usize::from(policy.persistent_cache.is_some()) * 4 + usize::from(policy.transfer_guard_enabled);
    let mut updates = Vec::with_capacity(update_count);
    updates.extend([
        JaxRuntimeConfigUpdate {
            setting_name: JAX_CONFIG_PLATFORMS,
            value: JaxRuntimeConfigValue::Text(Cow::Borrowed(policy.platform_name())),
        },
        JaxRuntimeConfigUpdate { setting_name: JAX_CONFIG_ENABLE_X64, value: JaxRuntimeConfigValue::Boolean(true) },
        JaxRuntimeConfigUpdate {
            setting_name: JAX_CONFIG_DEFAULT_MATMUL_PRECISION,
            value: JaxRuntimeConfigValue::Text(Cow::Borrowed(policy.matmul_precision.as_str())),
        },
    ]);
    if let Some(cache_policy) = policy.persistent_cache.as_ref() {
        updates.extend([
            JaxRuntimeConfigUpdate {
                setting_name: JAX_CONFIG_COMPILATION_CACHE_DIR,
                value: JaxRuntimeConfigValue::Text(cache_policy.directory.path().to_string_lossy()),
            },
            JaxRuntimeConfigUpdate {
                setting_name: JAX_CONFIG_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
                value: JaxRuntimeConfigValue::Integer(cache_policy.min_entry_size_bytes),
            },
            JaxRuntimeConfigUpdate {
                setting_name: JAX_CONFIG_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS,
                value: JaxRuntimeConfigValue::Integer(cache_policy.min_compile_time_seconds),
            },
            JaxRuntimeConfigUpdate {
                setting_name: JAX_CONFIG_PERSISTENT_CACHE_ENABLE_XLA_CACHES,
                value: JaxRuntimeConfigValue::Text(Cow::Borrowed(policy.xla_auxiliary_cache_mode())),
            },
        ]);
    }
    if policy.transfer_guard_enabled {
        updates.push(JaxRuntimeConfigUpdate {
            setting_name: JAX_CONFIG_TRANSFER_GUARD,
            value: JaxRuntimeConfigValue::Text(Cow::Borrowed(JAX_TRANSFER_GUARD_DISALLOW)),
        });
    }
    updates
}
