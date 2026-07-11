//! Rayon process-global runtime setup.

use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub(crate) enum RayonRuntimeError {
    InvalidThreadCount,
    GlobalThreadPool { source: rayon::ThreadPoolBuildError },
}

impl fmt::Display for RayonRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidThreadCount => formatter.write_str("Rayon thread count must be positive."),
            Self::GlobalThreadPool { source } => source.fmt(formatter),
        }
    }
}

impl Error for RayonRuntimeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidThreadCount => None,
            Self::GlobalThreadPool { source } => Some(source),
        }
    }
}

/// Configure Rayon global thread pool size.
///
/// # Errors
///
/// Returns an error when the requested thread count is zero or Rayon rejects
/// global thread-pool initialization for this process.
pub(crate) fn configure_global_rayon_thread_pool(thread_count: usize) -> Result<(), RayonRuntimeError> {
    if thread_count == 0 {
        return Err(RayonRuntimeError::InvalidThreadCount);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build_global()
        .map_err(|source| RayonRuntimeError::GlobalThreadPool { source })
}
