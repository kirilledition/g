//! Native CLI runtime lifecycle state.

pub const CLI_RUNTIME_FAILURE_EXIT_CODE: i32 = 1;
pub const NATIVE_CLI_OUTPUT_LOG_LIMIT: i64 = 4096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliTerminalResult {
    pub exit_code: i32,
    pub stdout_lines: Vec<String>,
    pub stderr_lines: Vec<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CliOutputBuffer {
    pub stdout_chunks: Vec<String>,
    pub stderr_chunks: Vec<String>,
}

impl CliOutputBuffer {
    #[must_use]
    pub fn from_frontend_output(stdout_text: &str, stderr_text: &str) -> Self {
        let mut buffer = Self::default();
        buffer.push_stdout_text(stdout_text);
        buffer.push_stderr_text(stderr_text);
        buffer
    }

    pub fn push_stdout_text(&mut self, text: &str) {
        push_text_chunk(&mut self.stdout_chunks, text);
    }

    pub fn push_stderr_text(&mut self, text: &str) {
        push_text_chunk(&mut self.stderr_chunks, text);
    }

    #[must_use]
    pub fn append_terminal_result(&mut self, terminal_result: CliTerminalResult) -> i32 {
        self.stdout_chunks.extend(lines_to_chunks(terminal_result.stdout_lines));
        self.stderr_chunks.extend(lines_to_chunks(terminal_result.stderr_lines));
        terminal_result.exit_code
    }
}

fn push_text_chunk(chunks: &mut Vec<String>, text: &str) {
    if !text.is_empty() {
        chunks.push(text.to_string());
    }
}

fn lines_to_chunks(lines: Vec<String>) -> impl Iterator<Item = String> {
    lines.into_iter().map(|line| format!("{line}\n"))
}
