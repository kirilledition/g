//! Public runtime error boundary.

use std::error::Error;
use std::fmt;

#[derive(Debug, Eq, PartialEq)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_error_preserves_its_message() {
        let error = RuntimeCompatibilityError::new("incompatible runtime".to_owned());
        assert_eq!(error.to_string(), "incompatible runtime");
        assert!(error.source().is_none());
    }
}
