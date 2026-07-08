//! Deterministic graceful-shutdown signal metadata helpers.

mod controller;
mod error;
mod signal;

pub use controller::{
    ShutdownControllerState, ShutdownHandlerInstallPlan, ShutdownHandlerRestorePlan, ShutdownHandlerSession,
    ShutdownRequestAction, ShutdownRequestDecisionPayload,
};
pub use error::ShutdownError;
pub use signal::{
    SecondSignalExceptionPlan, ShutdownSignalPayload, build_shutdown_signal, default_shutdown_signal_numbers,
    plan_second_signal_exception,
};

#[cfg(test)]
use controller::ShutdownController;
#[cfg(test)]
use signal::test_constants::{SIGPWR_NUMBER, SIGRTMAX_NUMBER, SIGRTMIN_NUMBER, SIGSTKFLT_NUMBER};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_shutdown_signal_metadata() {
        let payload = build_shutdown_signal(signal::SIGTERM).unwrap();

        assert_eq!(default_shutdown_signal_numbers(), vec![signal::SIGINT, signal::SIGTERM]);
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
        controller.finish_handler_session();
        assert_eq!(controller.requested_signal(), None);
        assert!(!controller.handlers_installed());
    }

    #[test]
    fn shutdown_handler_session_owns_previous_handler_state() {
        let mut session = ShutdownHandlerSession::new(&[signal::SIGINT, signal::SIGTERM]).unwrap();

        let install_plan = session.begin_handler_install();
        assert_eq!(install_plan.handled_signals.len(), 2);
        session.record_previous_handler(signal::SIGINT, "previous-sigint".to_string());
        session.record_previous_handler(signal::SIGTERM, "previous-sigterm".to_string());
        session.mark_handlers_installed();

        assert!(session.handlers_installed());
        assert_eq!(session.previous_handler(signal::SIGINT).map(String::as_str), Some("previous-sigint"));
        assert_eq!(session.previous_handler(signal::SIGTERM).map(String::as_str), Some("previous-sigterm"));
        assert_eq!(session.request_shutdown(signal::SIGINT).unwrap().action, ShutdownRequestAction::Graceful);
        assert!(session.requested_signal().is_some());

        let restore_plan = session.plan_handler_restore();
        assert!(restore_plan.should_restore);
        session.mark_handlers_restored();
        assert!(!session.handlers_installed());
        assert_eq!(session.previous_handler(signal::SIGINT), None);
        assert!(session.requested_signal().is_some());

        session.finish_handler_session();
        assert_eq!(session.requested_signal(), None);
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
