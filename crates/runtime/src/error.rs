//! Public runtime error boundary.

#![allow(clippy::module_name_repetitions)]

use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeCompatibilityError {
    message: String,
}

impl RuntimeCompatibilityError {
    #[must_use]
    pub fn new(message: String) -> Self {
        Self { message }
    }
}

impl fmt::Display for RuntimeCompatibilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for RuntimeCompatibilityError {}

#[derive(Debug)]
pub enum RuntimeError {
    Compatibility(RuntimeCompatibilityError),
}

impl From<RuntimeCompatibilityError> for RuntimeError {
    fn from(error: RuntimeCompatibilityError) -> Self {
        Self::Compatibility(error)
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compatibility(error) => error.fmt(formatter),
        }
    }
}

impl Error for RuntimeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Compatibility(error) => Some(error),
        }
    }
}

pub type RuntimeResult<T> = Result<T, RuntimeError>;
