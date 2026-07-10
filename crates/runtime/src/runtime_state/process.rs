use crate::runtime_policy::LoggingRuntimePolicyPayload;

use super::JaxRuntimePolicyPayload;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProcessRuntimeState {
    pub(super) logging_policy: Option<LoggingRuntimePolicyPayload>,
    pub(super) rayon_thread_count: Option<i64>,
    pub(super) jax_policy: Option<JaxRuntimePolicyPayload>,
}
