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
    artifacts
        .iter()
        .map(|artifact| format!("Parquet dataset saved to {}", artifact.parquet_dataset_directory))
        .collect()
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

#[cfg(test)]
mod tests {
    use g_engine::PhenotypeRunArtifact;

    use super::{
        CliRunResult, non_empty_text_chunk, render_completed_lines, render_failed_lines, render_interrupted_lines,
    };

    #[test]
    fn frontend_output_preserves_only_nonempty_streams() {
        assert_eq!(
            CliRunResult::from_frontend_output(0, "help\n", ""),
            CliRunResult { exit_code: 0, stdout_chunks: vec!["help\n".to_string()], stderr_chunks: Vec::new() }
        );
        assert!(non_empty_text_chunk("").is_empty());
    }

    #[test]
    fn line_output_appends_newlines_and_terminal_status() {
        let mut output = CliRunResult::from_lines(0, vec!["first".to_string()], Vec::new());
        output.append(CliRunResult::from_lines(7, Vec::new(), vec!["second".to_string()]));
        assert_eq!(output.exit_code, 7);
        assert_eq!(output.stdout_chunks, ["first\n"]);
        assert_eq!(output.stderr_chunks, ["second\n"]);
    }

    #[test]
    fn completion_lines_describe_each_parquet_dataset_once() {
        assert!(render_completed_lines(&[]).is_empty());
        let artifacts = [
            PhenotypeRunArtifact {
                output_run_directory: "run-a".to_string(),
                parquet_dataset_directory: "run-a/parquet".to_string(),
            },
            PhenotypeRunArtifact {
                output_run_directory: "run-b".to_string(),
                parquet_dataset_directory: "run-b/parquet".to_string(),
            },
        ];
        assert_eq!(
            render_completed_lines(&artifacts),
            ["Parquet dataset saved to run-a/parquet", "Parquet dataset saved to run-b/parquet",]
        );
    }

    #[test]
    fn interruption_and_failure_lines_preserve_resume_meaning() {
        assert_eq!(render_interrupted_lines("SIGINT", false), ["Interrupted by SIGINT."]);
        assert_eq!(
            render_interrupted_lines("SIGTERM", true),
            ["Interrupted by SIGTERM. Flushed queued chunks and saved committed output for resume."]
        );
        assert_eq!(render_failed_lines("ConfigurationError", ""), ["Error: ConfigurationError"]);
        assert_eq!(render_failed_lines("ConfigurationError", "invalid input"), ["Error: invalid input"]);
    }
}
