//! Runner-owned terminal output assembly.

/// Native CLI output that the Python bootstrap forwards verbatim.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct CliRunResult {
    pub exit_code: i32,
    pub stdout_chunks: Vec<String>,
    pub stderr_chunks: Vec<String>,
}

impl CliRunResult {
    pub(crate) fn from_frontend_output(exit_code: i32, stdout: &str, stderr: &str) -> Self {
        Self { exit_code, stdout_chunks: non_empty_text_chunk(stdout), stderr_chunks: non_empty_text_chunk(stderr) }
    }

    pub(crate) fn from_lines(exit_code: i32, stdout_lines: Vec<String>, stderr_lines: Vec<String>) -> Self {
        Self { exit_code, stdout_chunks: lines_to_chunks(stdout_lines), stderr_chunks: lines_to_chunks(stderr_lines) }
    }

    pub(crate) fn append(&mut self, result: Self) {
        self.exit_code = result.exit_code;
        self.stdout_chunks.extend(result.stdout_chunks);
        self.stderr_chunks.extend(result.stderr_chunks);
    }
}

pub(crate) fn render_completed_lines(artifacts: &[g_engine::PhenotypeRunArtifact]) -> Vec<String> {
    let mut lines = Vec::with_capacity(artifacts.len().saturating_mul(2).max(1));
    for artifact in artifacts {
        lines.push(format!("Success. Run saved to {}", artifact.output_run_directory));
        lines.push(format!("Parquet dataset saved to {}", artifact.parquet_dataset_directory));
    }
    if lines.is_empty() {
        lines.push("Success. Run completed.".to_string());
    }
    lines
}

pub(crate) fn render_interrupted_lines(signal_name: &str, flushed_for_resume: bool) -> Vec<String> {
    if flushed_for_resume {
        return vec![format!(
            "Interrupted by {signal_name}. Flushed queued chunks and saved committed output for resume."
        )];
    }
    vec![format!("Interrupted by {signal_name}.")]
}

pub(crate) fn render_failed_lines(error_type: &str, error_message: &str) -> Vec<String> {
    if error_message.is_empty() {
        return vec![format!("Error: {error_type}")];
    }
    vec![format!("Error: {error_message}")]
}

fn non_empty_text_chunk(text: &str) -> Vec<String> {
    if text.is_empty() { Vec::new() } else { vec![text.to_string()] }
}

fn lines_to_chunks(lines: Vec<String>) -> Vec<String> {
    lines
        .into_iter()
        .map(|mut line| {
            line.push('\n');
            line
        })
        .collect()
}
