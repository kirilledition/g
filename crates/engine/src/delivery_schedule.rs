//! BGEN delivery and GPU genotype-format scheduling policy.

mod cleanup;
mod genotype_format;
mod invocation;
mod types;

pub use cleanup::plan_bgen_delivery_cleanup;
pub use genotype_format::{
    plan_auto_gpu_genotype_format_after_trusted_validation, plan_gpu_genotype_format_auto_to_dosage,
    plan_single_trait_binary_gpu_genotype_format_resolution, resolve_effective_trusted_no_missing_diploid,
    resolve_manifest_gpu_genotype_format,
};
pub use invocation::{
    plan_bgen_delivery_invocation, resolve_bgen_delivery_method, resolve_delivery_callback_batch_size,
    resolve_grouped_union_callback_batch_size,
};
pub use types::{
    BgenDeliveryCleanupAction, BgenDeliveryCleanupOutcome, BgenDeliveryCleanupPlan, BgenDeliveryInvocationPlan,
    BgenDeliveryMethod, GpuGenotypeFormatResolutionPlan,
};
