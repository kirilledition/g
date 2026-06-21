//! Deterministic graceful-shutdown signal metadata helpers.

use signal_hook::consts::signal;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ShutdownSignalPayload {
    pub(crate) number: i32,
    pub(crate) name: String,
    pub(crate) exit_code: i32,
}

pub(crate) fn build_shutdown_signal(signal_number: i32) -> Result<ShutdownSignalPayload, String> {
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
        signal::SIGSYS => Some("SIGSYS"),
        _ => None,
    }
}
