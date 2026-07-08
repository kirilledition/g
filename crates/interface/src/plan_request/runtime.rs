//! Runtime section compilation.

use g_plan as plan;

use crate::resolved::RegenieConfigData;

use super::conversion;

#[must_use]
pub(super) fn build_runtime_plan(config: &RegenieConfigData) -> plan::RuntimePlan {
    plan::RuntimePlan {
        jax_cache_directory: config.g_compute.jax_cache_dir.clone(),
        jax_matmul_precision: config.g_compute.jax_matmul_precision.map(conversion::plan_jax_matmul_precision),
        persistent_cache_enabled: config.g_compute.jax_persistent_cache,
        persistent_cache_min_entry_size_bytes: config.g_compute.jax_persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds: config.g_compute.jax_persistent_cache_min_compile_time_seconds,
        xla_autotune_cache_enabled: config.g_compute.jax_xla_autotune_cache,
        transfer_guard_enabled: config.g_compute.jax_transfer_guard,
    }
}
