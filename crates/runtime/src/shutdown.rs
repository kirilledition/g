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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShutdownControllerState {
    pub requested_signal: Option<ShutdownSignalPayload>,
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
}
