//! Deterministic graceful-shutdown signal metadata helpers.

use signal_hook::consts::signal;

const SIGSTKFLT_NUMBER: i32 = 16;
const SIGPWR_NUMBER: i32 = 30;
const SIGRTMIN_NUMBER: i32 = 34;
const SIGRTMAX_NUMBER: i32 = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShutdownSignalPayload {
    pub number: i32,
    pub name: String,
    pub exit_code: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShutdownRequestAction {
    Graceful,
    Force,
}

impl ShutdownRequestAction {
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Graceful => "graceful",
            Self::Force => "force",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShutdownRequestDecisionPayload {
    pub action: ShutdownRequestAction,
    pub signal: ShutdownSignalPayload,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SecondSignalExceptionPlan {
    pub raise_keyboard_interrupt: bool,
    pub exit_code: i32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShutdownControllerState {
    pub requested_signal: Option<ShutdownSignalPayload>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShutdownController {
    state: ShutdownControllerState,
    handled_signals: Vec<ShutdownSignalPayload>,
    handlers_installed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShutdownHandlerInstallPlan {
    pub handled_signals: Vec<ShutdownSignalPayload>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShutdownHandlerRestorePlan {
    pub should_restore: bool,
    pub handled_signals: Vec<ShutdownSignalPayload>,
}

impl ShutdownControllerState {
    /// Record a handled shutdown signal and return the resulting action.
    ///
    /// # Errors
    ///
    /// Returns an error when `signal_number` is not a supported Linux signal.
    pub fn request_shutdown(&mut self, signal_number: i32) -> Result<ShutdownRequestDecisionPayload, String> {
        let signal = build_shutdown_signal(signal_number)?;
        let action = if self.requested_signal.is_none() {
            self.requested_signal = Some(signal.clone());
            ShutdownRequestAction::Graceful
        } else {
            ShutdownRequestAction::Force
        };
        Ok(ShutdownRequestDecisionPayload { action, signal })
    }

    pub fn reset(&mut self) {
        self.requested_signal = None;
    }
}

impl ShutdownController {
    /// Build a shutdown controller for the configured signal numbers.
    ///
    /// # Errors
    ///
    /// Returns an error when any signal number is unsupported.
    pub fn new(handled_signal_numbers: &[i32]) -> Result<Self, String> {
        let handled_signals =
            handled_signal_numbers.iter().copied().map(build_shutdown_signal).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { state: ShutdownControllerState::default(), handled_signals, handlers_installed: false })
    }

    #[must_use]
    pub const fn handlers_installed(&self) -> bool {
        self.handlers_installed
    }

    #[must_use]
    pub const fn requested_signal(&self) -> Option<&ShutdownSignalPayload> {
        self.state.requested_signal.as_ref()
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    #[must_use]
    pub fn begin_handler_install(&mut self) -> ShutdownHandlerInstallPlan {
        self.reset();
        self.handlers_installed = false;
        ShutdownHandlerInstallPlan { handled_signals: self.handled_signals.clone() }
    }

    pub const fn mark_handlers_installed(&mut self) {
        self.handlers_installed = true;
    }

    #[must_use]
    pub fn plan_handler_restore(&self) -> ShutdownHandlerRestorePlan {
        ShutdownHandlerRestorePlan {
            should_restore: self.handlers_installed,
            handled_signals: if self.handlers_installed { self.handled_signals.clone() } else { Vec::new() },
        }
    }

    pub const fn mark_handlers_restored(&mut self) {
        self.handlers_installed = false;
    }

    /// Record a handled shutdown signal and return the resulting action.
    ///
    /// # Errors
    ///
    /// Returns an error when `signal_number` is not a supported Linux signal.
    pub fn request_shutdown(&mut self, signal_number: i32) -> Result<ShutdownRequestDecisionPayload, String> {
        self.state.request_shutdown(signal_number)
    }
}

/// Build deterministic shutdown metadata for a Unix signal number.
///
/// # Errors
///
/// Returns an error when `signal_number` is not one of the supported Linux
/// signal constants.
pub fn build_shutdown_signal(signal_number: i32) -> Result<ShutdownSignalPayload, String> {
    let signal_name =
        linux_signal_name(signal_number).ok_or_else(|| format!("{signal_number} is not a valid Signals"))?;
    Ok(ShutdownSignalPayload { number: signal_number, name: signal_name.to_string(), exit_code: 128 + signal_number })
}

/// Plan the Python exception adapter for a repeated shutdown signal.
///
/// # Errors
///
/// Returns an error when `signal_number` is not one of the supported Linux
/// signal constants.
pub fn plan_second_signal_exception(signal_number: i32) -> Result<SecondSignalExceptionPlan, String> {
    let signal = build_shutdown_signal(signal_number)?;
    Ok(SecondSignalExceptionPlan {
        raise_keyboard_interrupt: signal_number == signal::SIGINT,
        exit_code: signal.exit_code,
    })
}

fn linux_signal_name(signal_number: i32) -> Option<&'static str> {
    match signal_number {
        signal::SIGHUP => Some("SIGHUP"),
        signal::SIGINT => Some("SIGINT"),
        signal::SIGQUIT => Some("SIGQUIT"),
        signal::SIGILL => Some("SIGILL"),
        signal::SIGTRAP => Some("SIGTRAP"),
        signal::SIGABRT => Some("SIGABRT"),
        signal::SIGBUS => Some("SIGBUS"),
        signal::SIGFPE => Some("SIGFPE"),
        signal::SIGKILL => Some("SIGKILL"),
        signal::SIGUSR1 => Some("SIGUSR1"),
        signal::SIGSEGV => Some("SIGSEGV"),
        signal::SIGUSR2 => Some("SIGUSR2"),
        signal::SIGPIPE => Some("SIGPIPE"),
        signal::SIGALRM => Some("SIGALRM"),
        signal::SIGTERM => Some("SIGTERM"),
        SIGSTKFLT_NUMBER => Some("SIGSTKFLT"),
        signal::SIGCHLD => Some("SIGCHLD"),
        signal::SIGCONT => Some("SIGCONT"),
        signal::SIGSTOP => Some("SIGSTOP"),
        signal::SIGTSTP => Some("SIGTSTP"),
        signal::SIGTTIN => Some("SIGTTIN"),
        signal::SIGTTOU => Some("SIGTTOU"),
        signal::SIGURG => Some("SIGURG"),
        signal::SIGXCPU => Some("SIGXCPU"),
        signal::SIGXFSZ => Some("SIGXFSZ"),
        signal::SIGVTALRM => Some("SIGVTALRM"),
        signal::SIGPROF => Some("SIGPROF"),
        signal::SIGWINCH => Some("SIGWINCH"),
        signal::SIGIO => Some("SIGIO"),
        SIGPWR_NUMBER => Some("SIGPWR"),
        signal::SIGSYS => Some("SIGSYS"),
        SIGRTMIN_NUMBER => Some("SIGRTMIN"),
        SIGRTMAX_NUMBER => Some("SIGRTMAX"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_shutdown_signal_metadata() {
        let payload = build_shutdown_signal(signal::SIGTERM).unwrap();

        assert_eq!(payload.number, signal::SIGTERM);
        assert_eq!(payload.name, "SIGTERM");
        assert_eq!(payload.exit_code, 128 + signal::SIGTERM);
        assert_eq!(build_shutdown_signal(SIGSTKFLT_NUMBER).unwrap().name, "SIGSTKFLT");
        assert_eq!(build_shutdown_signal(SIGPWR_NUMBER).unwrap().name, "SIGPWR");
        assert_eq!(build_shutdown_signal(SIGRTMIN_NUMBER).unwrap().name, "SIGRTMIN");
        assert_eq!(build_shutdown_signal(SIGRTMAX_NUMBER).unwrap().name, "SIGRTMAX");
        assert!(build_shutdown_signal(0).is_err());
    }

    #[test]
    fn shutdown_controller_tracks_first_and_repeated_signal() {
        let mut controller = ShutdownControllerState::default();

        let first_decision = controller.request_shutdown(signal::SIGINT).unwrap();
        let second_decision = controller.request_shutdown(signal::SIGTERM).unwrap();

        assert_eq!(first_decision.action, ShutdownRequestAction::Graceful);
        assert_eq!(first_decision.signal.name, "SIGINT");
        assert_eq!(second_decision.action, ShutdownRequestAction::Force);
        assert_eq!(second_decision.signal.name, "SIGTERM");
        assert_eq!(controller.requested_signal.as_ref().unwrap().name, "SIGINT");
        controller.reset();
        assert_eq!(controller.requested_signal, None);
    }

    #[test]
    fn shutdown_controller_owns_handler_lifecycle_state() {
        let mut controller = ShutdownController::new(&[signal::SIGINT, signal::SIGTERM]).unwrap();

        let install_plan = controller.begin_handler_install();
        assert_eq!(install_plan.handled_signals[0].name, "SIGINT");
        assert!(!controller.handlers_installed());
        controller.mark_handlers_installed();
        assert!(controller.handlers_installed());

        let restore_plan = controller.plan_handler_restore();
        assert!(restore_plan.should_restore);
        assert_eq!(restore_plan.handled_signals.len(), 2);
        controller.mark_handlers_restored();
        assert!(!controller.plan_handler_restore().should_restore);

        let first_decision = controller.request_shutdown(signal::SIGINT).unwrap();
        let second_decision = controller.request_shutdown(signal::SIGTERM).unwrap();
        assert_eq!(first_decision.action, ShutdownRequestAction::Graceful);
        assert_eq!(second_decision.action, ShutdownRequestAction::Force);
        assert_eq!(controller.requested_signal().unwrap().name, "SIGINT");
        controller.reset();
        assert_eq!(controller.requested_signal(), None);
    }

    #[test]
    fn plans_second_signal_exception_adapter() {
        assert_eq!(
            plan_second_signal_exception(signal::SIGINT).unwrap(),
            SecondSignalExceptionPlan { raise_keyboard_interrupt: true, exit_code: 128 + signal::SIGINT },
        );
        assert_eq!(
            plan_second_signal_exception(signal::SIGTERM).unwrap(),
            SecondSignalExceptionPlan { raise_keyboard_interrupt: false, exit_code: 128 + signal::SIGTERM },
        );
        assert!(plan_second_signal_exception(0).is_err());
    }
}
