use std::error::Error;
use std::fmt;

/// Failure to serialize one structured runtime diagnostic event.
#[derive(Debug)]
pub struct DiagnosticEventError {
    source: serde_json::Error,
}

impl fmt::Display for DiagnosticEventError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.source.fmt(formatter)
    }
}

impl Error for DiagnosticEventError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

impl From<serde_json::Error> for DiagnosticEventError {
    fn from(source: serde_json::Error) -> Self {
        Self { source }
    }
}

/// Emit a structured diagnostic through the process tracing subscriber.
///
/// # Errors
///
/// Returns an error when the diagnostic fields cannot be serialized.
pub fn emit_diagnostic_event<Fields>(
    level: &str,
    event_name: &str,
    message: &str,
    fields: &Fields,
) -> Result<(), DiagnosticEventError>
where
    Fields: serde::Serialize,
{
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
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use serde::Serializer;
    use serde::ser::Error as _;
    use tracing_subscriber::prelude::*;

    use super::*;

    struct SerializationFailure;

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
}
