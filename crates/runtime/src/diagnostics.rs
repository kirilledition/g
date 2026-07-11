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
) -> Result<(), serde_json::Error>
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
    let fields_json = serde_json::to_string(fields)?;
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
