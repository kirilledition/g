//! Native `g` command entrypoint.

use std::io::{self, Write};

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let exit_code = run(&arguments);
    std::process::exit(exit_code);
}

fn run(arguments: &[String]) -> i32 {
    let outcome = match std::env::var_os(g_cli::NATIVE_PYTHON_BRIDGE_ENVIRONMENT_VARIABLE) {
        Some(python_executable_path) => {
            let execution_adapter = g_cli::PythonBridgeExecutionAdapter::new_with_environment_overrides(
                std::path::PathBuf::from(python_executable_path),
                vec![(g_cli::PYTHON_BRIDGE_SENTINEL_ENVIRONMENT_VARIABLE.to_string(), "1".to_string())],
            );
            g_cli::dispatch_native_cli_with_adapter(arguments, &execution_adapter)
        }
        None => g_cli::dispatch_native_cli(arguments),
    };
    if let Err(error) = write_outcome(&outcome) {
        eprintln!("Error: failed to write native CLI output: {error}");
        return g_cli::NATIVE_EXECUTION_UNAVAILABLE_EXIT_CODE;
    }
    outcome.exit_code
}

fn write_outcome(outcome: &g_cli::NativeCliOutcome) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    stdout.write_all(outcome.stdout.as_bytes())?;
    stderr.write_all(outcome.stderr.as_bytes())?;
    Ok(())
}
