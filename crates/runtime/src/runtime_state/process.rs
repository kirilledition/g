use crate::runtime_policy::LoggingSubscriberPolicy;

#[derive(Debug, Default, Eq, PartialEq)]
pub struct ProcessRuntimeState {
    pub(super) logging_subscriber_policy: Option<LoggingSubscriberPolicy<'static>>,
    pub(super) rayon_thread_count: Option<i64>,
}
