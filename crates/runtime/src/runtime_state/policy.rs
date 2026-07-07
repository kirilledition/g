use crate::runtime_policy::LoggingRuntimePolicyPayload;

use super::JaxRuntimePolicyPayload;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimePolicyPayload {
    pub logging_policy: LoggingRuntimePolicyPayload,
    pub rayon_thread_count: Option<i64>,
    pub jax_policy: JaxRuntimePolicyPayload,
}
