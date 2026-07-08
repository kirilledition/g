use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShutdownError {
    message: String,
}

impl ShutdownError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }

    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ShutdownError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ShutdownError {}
