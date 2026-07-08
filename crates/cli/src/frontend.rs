//! Native CLI frontend implementation.

const NATIVE_EXECUTION_UNAVAILABLE_EXIT_CODE: i32 = 1;
const NATIVE_EXECUTION_UNAVAILABLE_MESSAGE: &str = concat!(
    "Error: native CLI execution is not available yet; ",
    "use the Python console entry point for full REGENIE runs.\n",
);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeCliOutcome {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub validated_run_config: bool,
}

impl NativeCliOutcome {
    #[must_use]
    pub fn new(exit_code: i32, stdout: String, stderr: String, validated_run_config: bool) -> Self {
        Self { exit_code, stdout, stderr, validated_run_config }
    }
}

#[must_use]
pub fn dispatch_native_cli(arguments: &[String]) -> NativeCliOutcome {
    let g_interface::CliOutcomeData { exit_code, stdout, mut stderr, config } = g_interface::dispatch_cli(arguments);
    if config.is_some() {
        stderr.push_str(NATIVE_EXECUTION_UNAVAILABLE_MESSAGE);
        return NativeCliOutcome::new(NATIVE_EXECUTION_UNAVAILABLE_EXIT_CODE, stdout, stderr, true);
    }
    NativeCliOutcome::new(exit_code, stdout, stderr, false)
}
