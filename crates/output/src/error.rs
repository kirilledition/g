//! Public output error boundary.

#[derive(Debug, thiserror::Error)]
pub enum OutputError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Runtime(String),
}

impl OutputError {
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn runtime(error: impl ToString) -> Self {
        Self::Runtime(error.to_string())
    }
}

pub type OutputResult<T> = Result<T, OutputError>;
