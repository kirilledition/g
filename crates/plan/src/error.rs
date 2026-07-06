//! Public plan error boundary.

use std::error::Error;
use std::fmt;

use crate::host_policy::HostPolicyError;
use crate::prepared::PreparedPlanError;

#[derive(Debug)]
pub enum PlanError {
    HostPolicy(HostPolicyError),
    Prepared(PreparedPlanError),
}

impl From<HostPolicyError> for PlanError {
    fn from(error: HostPolicyError) -> Self {
        Self::HostPolicy(error)
    }
}

impl From<PreparedPlanError> for PlanError {
    fn from(error: PreparedPlanError) -> Self {
        Self::Prepared(error)
    }
}

impl fmt::Display for PlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostPolicy(error) => error.fmt(formatter),
            Self::Prepared(error) => error.fmt(formatter),
        }
    }
}

impl Error for PlanError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::HostPolicy(error) => Some(error),
            Self::Prepared(error) => Some(error),
        }
    }
}

pub type PlanResult<T> = Result<T, PlanError>;
