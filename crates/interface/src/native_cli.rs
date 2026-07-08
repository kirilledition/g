//! Native CLI frontend implementation.

use std::process;

use crate::cli::CliOutcomeData;

pub const NATIVE_EXECUTION_UNAVAILABLE_EXIT_CODE: i32 = 1;
pub const NATIVE_EXECUTION_UNAVAILABLE_MESSAGE: &str = concat!(
    "Error: native CLI execution is not available yet; ",
    "use the Python console entry point for full REGENIE runs.\n",
);
pub const NATIVE_PYTHON_BRIDGE_ENVIRONMENT_VARIABLE: &str = "G_NATIVE_CLI_PYTHON_BRIDGE";
pub const NATIVE_PYTHON_BRIDGE_SENTINEL_ENVIRONMENT_VARIABLE: &str = "G_NATIVE_CLI_PYTHON_BRIDGE_SENTINEL";

const NATIVE_PYTHON_BRIDGE_SCRIPT: &str = "import g.cli, sys; raise SystemExit(g.cli.run_args(sys.argv[1:]))";

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
    let CliOutcomeData { exit_code, stdout, mut stderr, config } = crate::dispatch_cli(arguments);
    if config.is_some() {
        if let Some(python_bridge) = std::env::var_os(NATIVE_PYTHON_BRIDGE_ENVIRONMENT_VARIABLE) {
            return dispatch_python_bridge(arguments, python_bridge);
        }
        stderr.push_str(NATIVE_EXECUTION_UNAVAILABLE_MESSAGE);
        return NativeCliOutcome::new(NATIVE_EXECUTION_UNAVAILABLE_EXIT_CODE, stdout, stderr, true);
    }
    NativeCliOutcome::new(exit_code, stdout, stderr, false)
}

fn dispatch_python_bridge(arguments: &[String], python_bridge: std::ffi::OsString) -> NativeCliOutcome {
    match process::Command::new(python_bridge)
        .arg("-c")
        .arg(NATIVE_PYTHON_BRIDGE_SCRIPT)
        .args(arguments)
        .env(NATIVE_PYTHON_BRIDGE_SENTINEL_ENVIRONMENT_VARIABLE, "1")
        .output()
    {
        Ok(output) => NativeCliOutcome::new(
            output.status.code().unwrap_or(NATIVE_EXECUTION_UNAVAILABLE_EXIT_CODE),
            String::from_utf8_lossy(&output.stdout).into_owned(),
            String::from_utf8_lossy(&output.stderr).into_owned(),
            true,
        ),
        Err(error) => NativeCliOutcome::new(
            NATIVE_EXECUTION_UNAVAILABLE_EXIT_CODE,
            String::new(),
            format!("Error: failed to run native CLI Python bridge: {error}\n"),
            true,
        ),
    }
}
