use std::borrow::Cow;

use super::{
    JAX_CONFIG_COMPILATION_CACHE_DIR, JAX_CONFIG_DEFAULT_MATMUL_PRECISION, JAX_CONFIG_ENABLE_X64,
    JAX_CONFIG_PERSISTENT_CACHE_ENABLE_XLA_CACHES, JAX_CONFIG_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS,
    JAX_CONFIG_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES, JAX_CONFIG_PLATFORMS, JAX_MATMUL_PRECISION,
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS, JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES, JaxRuntimeConfigUpdate,
    JaxRuntimeConfigValue, JaxRuntimeSetupSession, XLA_AUXILIARY_CACHE_DISABLED,
};

#[must_use]
pub(crate) fn plan_jax_runtime_config_updates<'policy>(
    setup_session: &JaxRuntimeSetupSession<'policy>,
) -> Vec<JaxRuntimeConfigUpdate<'policy>> {
    let policy = setup_session.policy;
    let mut updates = Vec::with_capacity(7);
    updates.extend([
        JaxRuntimeConfigUpdate {
            setting_name: JAX_CONFIG_PLATFORMS,
            value: JaxRuntimeConfigValue::Text(Cow::Borrowed(policy.platform_name())),
        },
        JaxRuntimeConfigUpdate { setting_name: JAX_CONFIG_ENABLE_X64, value: JaxRuntimeConfigValue::Boolean(true) },
        JaxRuntimeConfigUpdate {
            setting_name: JAX_CONFIG_DEFAULT_MATMUL_PRECISION,
            value: JaxRuntimeConfigValue::Text(Cow::Borrowed(JAX_MATMUL_PRECISION)),
        },
        JaxRuntimeConfigUpdate {
            setting_name: JAX_CONFIG_COMPILATION_CACHE_DIR,
            value: JaxRuntimeConfigValue::Text(policy.cache_directory.path().to_string_lossy()),
        },
        JaxRuntimeConfigUpdate {
            setting_name: JAX_CONFIG_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
            value: JaxRuntimeConfigValue::Integer(JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES),
        },
        JaxRuntimeConfigUpdate {
            setting_name: JAX_CONFIG_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS,
            value: JaxRuntimeConfigValue::Integer(JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS),
        },
        JaxRuntimeConfigUpdate {
            setting_name: JAX_CONFIG_PERSISTENT_CACHE_ENABLE_XLA_CACHES,
            value: JaxRuntimeConfigValue::Text(Cow::Borrowed(XLA_AUXILIARY_CACHE_DISABLED)),
        },
    ]);
    updates
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::plan_jax_runtime_config_updates;
    use crate::jax_runtime::{
        JAX_CONFIG_COMPILATION_CACHE_DIR, JAX_CONFIG_DEFAULT_MATMUL_PRECISION, JAX_CONFIG_ENABLE_X64,
        JAX_CONFIG_PERSISTENT_CACHE_ENABLE_XLA_CACHES, JAX_CONFIG_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS,
        JAX_CONFIG_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES, JAX_CONFIG_PLATFORMS, JaxCacheDirectory,
        JaxRuntimeConfigUpdate, JaxRuntimeConfigValue, JaxRuntimePolicy, JaxRuntimeSetupSession,
    };

    #[test]
    fn config_updates_are_complete_ordered_and_typed() {
        let policy = JaxRuntimePolicy {
            device: g_plan::Device::Gpu,
            cache_directory: JaxCacheDirectory::Explicit(PathBuf::from("/cache/jax")),
        };
        let setup_session = JaxRuntimeSetupSession::new(true, &policy);
        assert_eq!(
            plan_jax_runtime_config_updates(&setup_session),
            [
                JaxRuntimeConfigUpdate {
                    setting_name: JAX_CONFIG_PLATFORMS,
                    value: JaxRuntimeConfigValue::Text("cuda".into()),
                },
                JaxRuntimeConfigUpdate {
                    setting_name: JAX_CONFIG_ENABLE_X64,
                    value: JaxRuntimeConfigValue::Boolean(true),
                },
                JaxRuntimeConfigUpdate {
                    setting_name: JAX_CONFIG_DEFAULT_MATMUL_PRECISION,
                    value: JaxRuntimeConfigValue::Text("float32".into()),
                },
                JaxRuntimeConfigUpdate {
                    setting_name: JAX_CONFIG_COMPILATION_CACHE_DIR,
                    value: JaxRuntimeConfigValue::Text("/cache/jax".into()),
                },
                JaxRuntimeConfigUpdate {
                    setting_name: JAX_CONFIG_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
                    value: JaxRuntimeConfigValue::Integer(-1),
                },
                JaxRuntimeConfigUpdate {
                    setting_name: JAX_CONFIG_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS,
                    value: JaxRuntimeConfigValue::Integer(0),
                },
                JaxRuntimeConfigUpdate {
                    setting_name: JAX_CONFIG_PERSISTENT_CACHE_ENABLE_XLA_CACHES,
                    value: JaxRuntimeConfigValue::Text("none".into()),
                },
            ]
        );
    }
}
