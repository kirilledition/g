use super::{RuntimeCompatibilityToken, RuntimePolicyPayload};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunRuntime {
    pub runtime_policy: RuntimePolicyPayload,
    pub compatibility_token: RuntimeCompatibilityToken,
}
