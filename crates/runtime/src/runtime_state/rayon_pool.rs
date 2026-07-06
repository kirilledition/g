use std::error::Error;
use std::fmt;

use crate::error::RuntimeCompatibilityError;
use crate::rayon_runtime;

use super::ProcessRuntimeState;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RayonThreadPoolConfigurationPlan {
    pub should_configure: bool,
    pub thread_count: Option<i64>,
}

#[derive(Debug)]
pub enum RayonThreadPoolConfigurationError {
    RuntimeCompatibility(RuntimeCompatibilityError),
    RuntimeConfiguration { thread_count: i64, source: rayon_runtime::RayonRuntimeError },
}

impl fmt::Display for RayonThreadPoolConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeCompatibility(error) => error.fmt(formatter),
            Self::RuntimeConfiguration { thread_count, source } => {
                if matches!(source, rayon_runtime::RayonRuntimeError::InvalidThreadCount) {
                    source.fmt(formatter)
                } else {
                    formatter.write_str(&rayon_runtime::format_global_rayon_thread_pool_configuration_error(
                        *thread_count,
                        &source.to_string(),
                    ))
                }
            }
        }
    }
}

impl Error for RayonThreadPoolConfigurationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RuntimeCompatibility(error) => Some(error),
            Self::RuntimeConfiguration { source, .. } => Some(source),
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

    pub fn record_rayon_thread_count(&mut self, thread_count: i64) {
        self.rayon_thread_count = Some(thread_count);
    }

    /// Plan process-global Rayon thread-pool configuration for one request.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested thread count conflicts with a
    /// previously configured Rayon global thread count.
    pub fn plan_rayon_thread_pool_configuration(
        &self,
        requested_thread_count: i64,
    ) -> Result<RayonThreadPoolConfigurationPlan, RuntimeCompatibilityError> {
        self.require_compatible_rayon_thread_count(Some(requested_thread_count))?;
        if self.rayon_thread_count == Some(requested_thread_count) {
            return Ok(RayonThreadPoolConfigurationPlan { should_configure: false, thread_count: None });
        }
        Ok(RayonThreadPoolConfigurationPlan { should_configure: true, thread_count: Some(requested_thread_count) })
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
    ) -> Result<RayonThreadPoolConfigurationPlan, RayonThreadPoolConfigurationError> {
        let plan = self
            .plan_rayon_thread_pool_configuration(requested_thread_count)
            .map_err(RayonThreadPoolConfigurationError::RuntimeCompatibility)?;
        let Some(thread_count) = plan.thread_count else {
            return Ok(plan);
        };
        let runtime_thread_count =
            usize::try_from(thread_count).map_err(|_| RayonThreadPoolConfigurationError::RuntimeConfiguration {
                thread_count,
                source: rayon_runtime::RayonRuntimeError::InvalidThreadCount,
            })?;
        rayon_runtime::configure_global_rayon_thread_pool(runtime_thread_count)
            .map_err(|source| RayonThreadPoolConfigurationError::RuntimeConfiguration { thread_count, source })?;
        self.record_rayon_thread_count(thread_count);
        Ok(plan)
    }

    #[must_use]
    pub fn effective_rayon_thread_count(&self, requested_thread_count: Option<i64>) -> Option<i64> {
        self.rayon_thread_count.or(requested_thread_count)
    }
}
