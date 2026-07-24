use std::error::Error;
use std::fmt;

/// Failure to serialize or safely observe one structured runtime diagnostic event.
#[derive(Debug)]
pub struct DiagnosticEventError {
    message: String,
    source: Option<serde_json::Error>,
}

impl fmt::Display for DiagnosticEventError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(formatter)
    }
}

impl Error for DiagnosticEventError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_ref().map(|source| source as &(dyn Error + 'static))
    }
}

impl From<serde_json::Error> for DiagnosticEventError {
    fn from(source: serde_json::Error) -> Self {
        Self { message: source.to_string(), source: Some(source) }
    }
}

/// Emit a structured diagnostic through the process tracing subscriber.
///
/// # Errors
///
/// Returns an error when the diagnostic fields cannot be serialized or the
/// tracing subscriber panics while observing the event.
pub fn emit_diagnostic_event<Fields>(
    level: &str,
    event_name: &str,
    message: &str,
    fields: &Fields,
) -> Result<(), DiagnosticEventError>
where
    Fields: serde::Serialize,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let enabled = match level {
            "error" => tracing::enabled!(target: "g.native.diagnostic", tracing::Level::ERROR),
            "warn" | "warning" => tracing::enabled!(target: "g.native.diagnostic", tracing::Level::WARN),
            "info" => tracing::enabled!(target: "g.native.diagnostic", tracing::Level::INFO),
            "debug" => tracing::enabled!(target: "g.native.diagnostic", tracing::Level::DEBUG),
            "trace" => tracing::enabled!(target: "g.native.diagnostic", tracing::Level::TRACE),
            _ => tracing::enabled!(target: "g.native.diagnostic", tracing::Level::WARN),
        };
        if !enabled {
            return Ok(());
        }
        let fields_json = serde_json::to_string(fields).map_err(DiagnosticEventError::from)?;
        match level {
            "error" => {
                tracing::error!(target: "g.native.diagnostic", g_event = event_name, g_fields = %fields_json, "{message}");
            }
            "warn" | "warning" => {
                tracing::warn!(target: "g.native.diagnostic", g_event = event_name, g_fields = %fields_json, "{message}");
            }
            "info" => {
                tracing::info!(target: "g.native.diagnostic", g_event = event_name, g_fields = %fields_json, "{message}");
            }
            "debug" => {
                tracing::debug!(target: "g.native.diagnostic", g_event = event_name, g_fields = %fields_json, "{message}");
            }
            "trace" => {
                tracing::trace!(target: "g.native.diagnostic", g_event = event_name, g_fields = %fields_json, "{message}");
            }
            _ => {
                tracing::warn!(target: "g.native.diagnostic", g_event = event_name, g_fields = %fields_json, requested_level = level, "{message}");
            }
        }
        Ok(())
    })) {
        Ok(result) => result,
        Err(payload) => Err(DiagnosticEventError {
            message: format!(
                "Tracing subscriber panicked while observing a diagnostic: {}.",
                panic_message(payload.as_ref())
            ),
            source: None,
        }),
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload.downcast_ref::<&str>().map_or_else(
        || payload.downcast_ref::<String>().cloned().unwrap_or_else(|| "unknown panic payload".to_string()),
        |message| (*message).to_string(),
    )
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use serde::Serializer;
    use serde::ser::Error as _;
    use tracing_subscriber::prelude::*;

    use super::*;

    struct SerializationFailure;

    struct PanickingEventLayer;

    impl<Subscriber> tracing_subscriber::Layer<Subscriber> for PanickingEventLayer
    where
        Subscriber: tracing::Subscriber,
    {
        fn on_event(&self, _event: &tracing::Event<'_>, _context: tracing_subscriber::layer::Context<'_, Subscriber>) {
            panic!("intentional diagnostic subscriber panic");
        }
    }

    impl serde::Serialize for SerializationFailure {
        fn serialize<SerializerType>(
            &self,
            _serializer: SerializerType,
        ) -> Result<SerializerType::Ok, SerializerType::Error>
        where
            SerializerType: Serializer,
        {
            Err(SerializerType::Error::custom("intentional diagnostic serialization failure"))
        }
    }

    #[test]
    fn disabled_diagnostics_do_not_serialize_fields() {
        tracing::subscriber::with_default(tracing::subscriber::NoSubscriber::default(), || {
            emit_diagnostic_event("info", "ignored", "ignored", &SerializationFailure)
                .expect("disabled diagnostic should not serialize fields");
        });
    }

    #[test]
    fn enabled_diagnostics_propagate_serialization_errors_for_every_level_route() {
        let subscriber = tracing_subscriber::registry().with(tracing_subscriber::EnvFilter::new("trace"));
        tracing::subscriber::with_default(subscriber, || {
            let error = emit_diagnostic_event("info", "invalid", "invalid", &SerializationFailure)
                .expect_err("enabled diagnostic should serialize fields");
            assert!(error.to_string().contains("intentional diagnostic serialization failure"));
            assert!(
                error
                    .source()
                    .is_some_and(|source| source.to_string().contains("intentional diagnostic serialization failure"))
            );

            for level in ["error", "warn", "warning", "info", "debug", "trace", "notice"] {
                emit_diagnostic_event(level, "test_event", "diagnostic", &serde_json::json!({"value": 1}))
                    .expect("valid diagnostic fields should serialize");
            }
        });
    }

    #[test]
    fn panicking_subscriber_becomes_an_observer_error() {
        let subscriber = tracing_subscriber::registry().with(PanickingEventLayer);
        tracing::subscriber::with_default(subscriber, || {
            let error = emit_diagnostic_event(
                "error",
                "panicking_subscriber",
                "subscriber panic must be contained",
                &serde_json::json!({"value": 1}),
            )
            .expect_err("subscriber panic must not escape diagnostic observation");
            assert!(error.to_string().contains("intentional diagnostic subscriber panic"));
            assert!(error.source().is_none());
        });
    }
}
