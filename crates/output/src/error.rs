//! Public output error boundary.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum OutputError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("Run manifest '{}' is missing during a lifecycle update.", manifest_path.display())]
    MissingRunManifest { manifest_path: PathBuf },
    #[error("Another process published a conflicting immutable output lineage record at '{}'.", record_path.display())]
    ConcurrentLineageUpdate { record_path: PathBuf },
    #[error("{0}")]
    Runtime(String),
}

impl OutputError {
    // `Result::map_err` transfers each concrete source error into this adapter;
    // accepting it by value matches that ownership boundary directly.
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn runtime(error: impl ToString) -> Self {
        Self::Runtime(error.to_string())
    }
}

pub(crate) type OutputResult<T> = Result<T, OutputError>;
