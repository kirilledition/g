use crate::runtime_policy::LoggingRuntimePolicyPayload;

use super::JaxRuntimePolicyPayload;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProcessRuntimeState {
    pub logging_policy: Option<LoggingRuntimePolicyPayload>,
    pub rayon_thread_count: Option<i64>,
    pub jax_policy: Option<JaxRuntimePolicyPayload>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RuntimeStateSnapshotPayload {
    pub logging_policy: Option<LoggingRuntimePolicyPayload>,
    pub rayon_thread_count: Option<i64>,
    pub jax_policy: Option<JaxRuntimePolicyPayload>,
}

impl ProcessRuntimeState {
    #[must_use]
    pub fn snapshot(&self) -> RuntimeStateSnapshotPayload {
        RuntimeStateSnapshotPayload {
            logging_policy: self.logging_policy.clone(),
            rayon_thread_count: self.rayon_thread_count,
            jax_policy: self.jax_policy.clone(),
        }
    }
}
