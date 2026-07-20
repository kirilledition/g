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
    pub(crate) fn require_compatible_rayon_thread_count(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::execute_isolated_test_body;

    #[test]
    fn rayon_configuration_validates_records_and_enforces_process_compatibility() {
        if !execute_isolated_test_body(
            "runtime_state::rayon_pool::tests::rayon_configuration_validates_records_and_enforces_process_compatibility",
            "G_RUNTIME_RAYON_TEST_CHILD",
        ) {
            return;
        }
        let mut state = ProcessRuntimeState::default();

        let zero_error = state.configure_rayon_thread_pool(0).expect_err("zero threads should fail");
        assert_eq!(zero_error.to_string(), "Rayon thread count must be positive.");
        assert!(zero_error.source().is_some());
        assert_eq!(state.rayon_thread_count, None);

        let negative_error = state.configure_rayon_thread_pool(-1).expect_err("negative threads should fail");
        assert_eq!(negative_error.to_string(), "Rayon thread count must be positive.");
        assert_eq!(state.rayon_thread_count, None);

        state.configure_rayon_thread_pool(2).expect("valid thread count should configure Rayon");
        assert_eq!(state.rayon_thread_count, Some(2));
        assert_eq!(rayon::current_num_threads(), 2);
        state.configure_rayon_thread_pool(2).expect("repeated compatible configuration should be a no-op");

        let compatibility_error = state.configure_rayon_thread_pool(3).expect_err("changed thread count should fail");
        assert!(compatibility_error.to_string().contains("Configured thread count: 2. Requested thread count: 3"));
        assert!(compatibility_error.source().is_some());
    }
}
