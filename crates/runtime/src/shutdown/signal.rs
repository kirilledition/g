use signal_hook::consts::signal;

use super::error::ShutdownError;

const SIGSTKFLT_NUMBER: i32 = 16;
const SIGPWR_NUMBER: i32 = 30;
const SIGRTMIN_NUMBER: i32 = 34;
const SIGRTMAX_NUMBER: i32 = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShutdownSignalPayload {
    pub name: String,
    pub exit_code: i32,
}

/// Build deterministic shutdown metadata for a Unix signal number.
///
/// # Errors
///
/// Returns an error when `signal_number` is not one of the supported Linux
/// signal constants.
pub fn build_shutdown_signal(signal_number: i32) -> Result<ShutdownSignalPayload, ShutdownError> {
    let signal_name = linux_signal_name(signal_number)
        .ok_or_else(|| ShutdownError::new(format!("{signal_number} is not a valid Signals")))?;
    Ok(ShutdownSignalPayload { name: signal_name.to_string(), exit_code: 128 + signal_number })
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
