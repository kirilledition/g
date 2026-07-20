use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use signal_hook::consts::signal::SIGTERM;

use super::error::ShutdownError;

static SIGTERM_FLAG: OnceLock<Result<Arc<AtomicBool>, ShutdownError>> = OnceLock::new();
static SIGTERM_SCOPE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Active CLI scope for graceful first-SIGTERM handling.
pub(crate) struct SigtermShutdownScope {
    requested: Arc<AtomicBool>,
}

impl Drop for SigtermShutdownScope {
    fn drop(&mut self) {
        self.requested.store(true, Ordering::SeqCst);
        SIGTERM_SCOPE_ACTIVE.store(false, Ordering::Release);
    }
}

/// Install process SIGTERM actions and arm graceful handling for one CLI run.
///
/// The first SIGTERM sets the request flag. A second SIGTERM executes the
/// signal's default action immediately.
///
/// # Errors
///
/// Returns an error if signal handlers cannot be installed or another CLI run
/// already owns the process signal scope.
pub(crate) fn begin_sigterm_shutdown_scope() -> Result<SigtermShutdownScope, ShutdownError> {
    if SIGTERM_SCOPE_ACTIVE.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return Err(ShutdownError::new("A SIGTERM shutdown scope is already active."));
    }

    let requested_result = SIGTERM_FLAG.get_or_init(|| {
        let requested = Arc::new(AtomicBool::new(true));
        signal_hook::flag::register_conditional_default(SIGTERM, Arc::clone(&requested))
            .map_err(|error| ShutdownError::new(format!("Could not install the SIGTERM default action: {error}")))?;
        signal_hook::flag::register(SIGTERM, Arc::clone(&requested))
            .map_err(|error| ShutdownError::new(format!("Could not install the SIGTERM request action: {error}")))?;
        Ok(requested)
    });
    let requested = match requested_result {
        Ok(requested) => Arc::clone(requested),
        Err(error) => {
            SIGTERM_SCOPE_ACTIVE.store(false, Ordering::Release);
            return Err(error.clone());
        }
    };
    requested.store(false, Ordering::SeqCst);
    Ok(SigtermShutdownScope { requested })
}

/// Return whether SIGTERM requested shutdown for the active CLI run.
#[must_use]
pub fn sigterm_shutdown_requested() -> bool {
    SIGTERM_SCOPE_ACTIVE.load(Ordering::Acquire)
        && SIGTERM_FLAG
            .get()
            .and_then(|result| result.as_ref().ok())
            .is_some_and(|requested| requested.load(Ordering::SeqCst))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::execute_isolated_test_body;

    #[test]
    fn sigterm_scope_is_exclusive_resets_requests_and_disarms_on_drop() {
        if !execute_isolated_test_body(
            "shutdown::process::tests::sigterm_scope_is_exclusive_resets_requests_and_disarms_on_drop",
            "G_RUNTIME_SIGTERM_TEST_CHILD",
        ) {
            return;
        }
        assert!(!sigterm_shutdown_requested());

        let scope = begin_sigterm_shutdown_scope().expect("first shutdown scope should open");
        assert!(!sigterm_shutdown_requested());
        let duplicate_error = begin_sigterm_shutdown_scope().err().expect("concurrent shutdown scope should fail");
        assert_eq!(duplicate_error.to_string(), "A SIGTERM shutdown scope is already active.");

        let requested = SIGTERM_FLAG
            .get()
            .expect("signal flag should be initialized")
            .as_ref()
            .expect("signal handlers should install");
        requested.store(true, Ordering::SeqCst);
        assert!(sigterm_shutdown_requested());

        drop(scope);
        assert!(!sigterm_shutdown_requested());

        let next_scope = begin_sigterm_shutdown_scope().expect("shutdown scope should reopen after drop");
        assert!(!sigterm_shutdown_requested());
        drop(next_scope);
    }
}
