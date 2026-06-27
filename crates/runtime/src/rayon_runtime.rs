//! Rayon process-global runtime setup.

use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum RayonRuntimeError {
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
pub fn configure_global_rayon_thread_pool(thread_count: usize) -> Result<(), RayonRuntimeError> {
    if thread_count == 0 {
        return Err(RayonRuntimeError::InvalidThreadCount);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build_global()
        .map_err(|source| RayonRuntimeError::GlobalThreadPool { source })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_thread_count_without_initializing_global_pool() {
        let error = configure_global_rayon_thread_pool(0).unwrap_err();

        assert!(matches!(error, RayonRuntimeError::InvalidThreadCount));
        assert_eq!(error.to_string(), "Rayon thread count must be positive.");
        assert!(error.source().is_none());
    }
}
