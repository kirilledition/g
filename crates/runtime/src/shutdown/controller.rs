use std::collections::BTreeMap;

use super::error::ShutdownError;
use super::signal::{ShutdownSignalPayload, build_shutdown_signal};

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ShutdownController {
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShutdownHandlerSession<Handler> {
    controller: ShutdownController,
    previous_handlers: BTreeMap<i32, Handler>,
}

impl ShutdownControllerState {
    /// Record a handled shutdown signal and return the resulting action.
    ///
    /// # Errors
    ///
    /// Returns an error when `signal_number` is not a supported Linux signal.
    pub fn request_shutdown(&mut self, signal_number: i32) -> Result<ShutdownRequestDecisionPayload, ShutdownError> {
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
    pub fn new(handled_signal_numbers: &[i32]) -> Result<Self, ShutdownError> {
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

    pub fn finish_handler_session(&mut self) {
        self.mark_handlers_restored();
        self.reset();
    }

    /// Record a handled shutdown signal and return the resulting action.
    ///
    /// # Errors
    ///
    /// Returns an error when `signal_number` is not a supported Linux signal.
    pub fn request_shutdown(&mut self, signal_number: i32) -> Result<ShutdownRequestDecisionPayload, ShutdownError> {
        self.state.request_shutdown(signal_number)
    }
}

impl<Handler> ShutdownHandlerSession<Handler> {
    /// Build a handler session for the configured signal numbers.
    ///
    /// # Errors
    ///
    /// Returns an error when any signal number is unsupported.
    pub fn new(handled_signal_numbers: &[i32]) -> Result<Self, ShutdownError> {
        Ok(Self { controller: ShutdownController::new(handled_signal_numbers)?, previous_handlers: BTreeMap::new() })
    }

    #[must_use]
    pub const fn handlers_installed(&self) -> bool {
        self.controller.handlers_installed()
    }

    #[must_use]
    pub const fn requested_signal(&self) -> Option<&ShutdownSignalPayload> {
        self.controller.requested_signal()
    }

    pub fn reset(&mut self) {
        self.controller.reset();
    }

    #[must_use]
    pub fn begin_handler_install(&mut self) -> ShutdownHandlerInstallPlan {
        self.previous_handlers.clear();
        self.controller.begin_handler_install()
    }

    pub fn record_previous_handler(&mut self, signal_number: i32, previous_handler: Handler) {
        self.previous_handlers.insert(signal_number, previous_handler);
    }

    #[must_use]
    pub fn previous_handler(&self, signal_number: i32) -> Option<&Handler> {
        self.previous_handlers.get(&signal_number)
    }

    pub fn mark_handlers_installed(&mut self) {
        self.controller.mark_handlers_installed();
    }

    #[must_use]
    pub fn plan_handler_restore(&self) -> ShutdownHandlerRestorePlan {
        self.controller.plan_handler_restore()
    }

    pub fn mark_handlers_restored(&mut self) {
        self.controller.mark_handlers_restored();
        self.previous_handlers.clear();
    }

    pub fn finish_handler_session(&mut self) {
        self.controller.finish_handler_session();
        self.previous_handlers.clear();
    }

    /// Record a handled shutdown signal and return the resulting action.
    ///
    /// # Errors
    ///
    /// Returns an error when `signal_number` is not a supported Linux signal.
    pub fn request_shutdown(&mut self, signal_number: i32) -> Result<ShutdownRequestDecisionPayload, ShutdownError> {
        self.controller.request_shutdown(signal_number)
    }
}
