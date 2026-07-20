use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShutdownError {
    message: String,
}

impl ShutdownError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl fmt::Display for ShutdownError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ShutdownError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shutdown_error_preserves_message_and_supports_equality() {
        let error = ShutdownError::new("shutdown unavailable");
        assert_eq!(error, ShutdownError::new("shutdown unavailable"));
        assert_eq!(error.to_string(), "shutdown unavailable");
        assert!(std::error::Error::source(&error).is_none());
    }
}
