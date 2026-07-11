use std::error::Error;
use std::fmt;

use crate::error::RuntimeCompatibilityError;
use crate::rayon_runtime;

use super::ProcessRuntimeState;

#[derive(Debug)]
pub struct RayonThreadPoolConfigurationError {
    kind: RayonThreadPoolConfigurationErrorKind,
}

#[derive(Debug)]
enum RayonThreadPoolConfigurationErrorKind {
    RuntimeCompatibility(RuntimeCompatibilityError),
    RuntimeConfiguration { thread_count: i64, source: rayon_runtime::RayonRuntimeError },
}

impl fmt::Display for RayonThreadPoolConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            RayonThreadPoolConfigurationErrorKind::RuntimeCompatibility(error) => error.fmt(formatter),
            RayonThreadPoolConfigurationErrorKind::RuntimeConfiguration { thread_count, source } => {
                if matches!(source, rayon_runtime::RayonRuntimeError::InvalidThreadCount) {
                    source.fmt(formatter)
                } else {
                    write!(
                        formatter,
                        "Unable to configure Rayon global thread pool for configured CPU threads={thread_count}; \
                         existing Rayon settings are unknown: {source}"
                    )
                }
            }
        }
    }
}

impl Error for RayonThreadPoolConfigurationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            RayonThreadPoolConfigurationErrorKind::RuntimeCompatibility(error) => Some(error),
            RayonThreadPoolConfigurationErrorKind::RuntimeConfiguration { source, .. } => Some(source),
        }
    }
}

impl ProcessRuntimeState {
    /// Require Rayon compatibility with previously configured process state.
    ///
    /// # Errors
    ///
    /// Returns an error when a previous run configured a different Rayon global
    /// thread count.
    pub fn require_compatible_rayon_thread_count(
        &self,
        requested_thread_count: Option<i64>,
    ) -> Result<(), RuntimeCompatibilityError> {
        let Some(requested_thread_count) = requested_thread_count else {
            return Ok(());
        };
        let Some(configured_thread_count) = self.rayon_thread_count else {
            return Ok(());
        };
        if configured_thread_count == requested_thread_count {
            return Ok(());
        }
        Err(RuntimeCompatibilityError::new(format!(
            "Rayon --threads is process-global for this Python process. \
             Configured thread count: {configured_thread_count}. Requested thread count: {requested_thread_count}. \
             Start a fresh Python process for incompatible Rayon settings."
        )))
    }

    /// Configure the process-global Rayon thread pool and record the result.
    ///
    /// # Errors
    ///
    /// Returns an error when the request is incompatible with previous state or
    /// Rayon rejects global thread-pool initialization for this process.
    pub fn configure_rayon_thread_pool(
        &mut self,
        requested_thread_count: i64,
    ) -> Result<(), RayonThreadPoolConfigurationError> {
        self.require_compatible_rayon_thread_count(Some(requested_thread_count)).map_err(|error| {
            RayonThreadPoolConfigurationError {
                kind: RayonThreadPoolConfigurationErrorKind::RuntimeCompatibility(error),
            }
        })?;
        if self.rayon_thread_count == Some(requested_thread_count) {
            return Ok(());
        }
        let thread_count = requested_thread_count;
        let runtime_thread_count = usize::try_from(thread_count).map_err(|_| RayonThreadPoolConfigurationError {
            kind: RayonThreadPoolConfigurationErrorKind::RuntimeConfiguration {
                thread_count,
                source: rayon_runtime::RayonRuntimeError::InvalidThreadCount,
            },
        })?;
        rayon_runtime::configure_global_rayon_thread_pool(runtime_thread_count).map_err(|source| {
            RayonThreadPoolConfigurationError {
                kind: RayonThreadPoolConfigurationErrorKind::RuntimeConfiguration { thread_count, source },
            }
        })?;
        self.rayon_thread_count = Some(thread_count);
        Ok(())
    }
}
