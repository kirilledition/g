//! Native CLI frontend for the GWAS engine.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub const NATIVE_EXECUTION_UNAVAILABLE_EXIT_CODE: i32 = 1;
pub const NATIVE_EXECUTION_UNAVAILABLE_MESSAGE: &str = concat!(
    "Error: native CLI execution is not available yet; ",
    "use the Python console entry point for full REGENIE runs.\n",
);
pub const NATIVE_EXECUTION_PANIC_EXIT_CODE: i32 = 1;
pub const NATIVE_SIGNAL_HANDLER_FAILURE_EXIT_CODE: i32 = 1;
const NATIVE_EXECUTION_PANIC_PREFIX: &str = "Error: native CLI execution adapter panicked";
const NATIVE_SIGNAL_HANDLER_FAILURE_PREFIX: &str = "Error: failed to install native CLI signal handlers";

#[derive(Clone, Debug)]
pub struct NativeExecutionContext {
    shutdown_requested: Arc<AtomicBool>,
}

impl NativeExecutionContext {
    #[must_use]
    pub fn shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::Relaxed)
    }
}

impl Default for NativeExecutionContext {
    fn default() -> Self {
        Self { shutdown_requested: Arc::new(AtomicBool::new(false)) }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeExecutionOutcome {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl NativeExecutionOutcome {
    #[must_use]
    pub fn new(exit_code: i32, stdout: String, stderr: String) -> Self {
        Self { exit_code, stdout, stderr }
    }
}

pub trait NativeExecutionAdapter {
    fn execute(
        &self,
        config: &g_interface::RegenieConfigData,
        execution_context: &NativeExecutionContext,
    ) -> NativeExecutionOutcome;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnsupportedNativeExecutionAdapter;

impl NativeExecutionAdapter for UnsupportedNativeExecutionAdapter {
    fn execute(
        &self,
        _config: &g_interface::RegenieConfigData,
        _execution_context: &NativeExecutionContext,
    ) -> NativeExecutionOutcome {
        NativeExecutionOutcome::new(
            NATIVE_EXECUTION_UNAVAILABLE_EXIT_CODE,
            String::new(),
            NATIVE_EXECUTION_UNAVAILABLE_MESSAGE.to_string(),
        )
    }
}

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
    dispatch_native_cli_with_adapter(arguments, &UnsupportedNativeExecutionAdapter)
}

#[must_use]
pub fn dispatch_native_cli_with_adapter(
    arguments: &[String],
    execution_adapter: &dyn NativeExecutionAdapter,
) -> NativeCliOutcome {
    let g_interface::CliOutcomeData { exit_code, mut stdout, mut stderr, config } =
        g_interface::dispatch_cli(arguments);
    if let Some(run_config) = config {
        let execution_context = NativeExecutionContext::default();
        let execution_outcome = match NativeSignalGuard::install_default(&execution_context) {
            Ok(_signal_guard) => execute_with_panic_boundary(execution_adapter, &run_config, &execution_context),
            Err(error) => NativeExecutionOutcome::new(
                NATIVE_SIGNAL_HANDLER_FAILURE_EXIT_CODE,
                String::new(),
                format!("{NATIVE_SIGNAL_HANDLER_FAILURE_PREFIX}: {error}.\n"),
            ),
        };
        stdout.push_str(&execution_outcome.stdout);
        stderr.push_str(&execution_outcome.stderr);
        return NativeCliOutcome::new(execution_outcome.exit_code, stdout, stderr, true);
    }
    NativeCliOutcome::new(exit_code, stdout, stderr, false)
}

fn execute_with_panic_boundary(
    execution_adapter: &dyn NativeExecutionAdapter,
    run_config: &g_interface::RegenieConfigData,
    execution_context: &NativeExecutionContext,
) -> NativeExecutionOutcome {
    let execution_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        execution_adapter.execute(run_config, execution_context)
    }));
    match execution_result {
        Ok(execution_outcome) => execution_outcome,
        Err(panic_payload) => NativeExecutionOutcome::new(
            NATIVE_EXECUTION_PANIC_EXIT_CODE,
            String::new(),
            format!("{NATIVE_EXECUTION_PANIC_PREFIX}: {}.\n", panic_payload_message(panic_payload.as_ref())),
        ),
    }
}

#[derive(Debug)]
struct NativeSignalGuard {
    signal_ids: Vec<signal_hook::SigId>,
}

impl NativeSignalGuard {
    fn install_default(execution_context: &NativeExecutionContext) -> Result<Self, std::io::Error> {
        Self::install_for_signal_numbers(
            &[signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM],
            execution_context,
            true,
        )
    }

    fn install_for_signal_numbers(
        signal_numbers: &[i32],
        execution_context: &NativeExecutionContext,
        enable_repeated_signal_default: bool,
    ) -> Result<Self, std::io::Error> {
        let mut guard = Self { signal_ids: Vec::new() };
        for signal_number in signal_numbers {
            if enable_repeated_signal_default {
                let signal_id = signal_hook::flag::register_conditional_default(
                    *signal_number,
                    Arc::clone(&execution_context.shutdown_requested),
                )?;
                guard.signal_ids.push(signal_id);
            }
            let signal_id =
                signal_hook::flag::register(*signal_number, Arc::clone(&execution_context.shutdown_requested))?;
            guard.signal_ids.push(signal_id);
        }
        Ok(guard)
    }
}

impl Drop for NativeSignalGuard {
    fn drop(&mut self) {
        for signal_id in self.signal_ids.drain(..) {
            let _was_registered = signal_hook::low_level::unregister(signal_id);
        }
    }
}

fn panic_payload_message(panic_payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = panic_payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = panic_payload.downcast_ref::<String>() {
        return message.clone();
    }
    "non-string panic payload".to_string()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Mutex;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    static SIGNAL_TEST_MUTEX: Mutex<()> = Mutex::new(());

    struct SuccessfulAdapter;

    impl NativeExecutionAdapter for SuccessfulAdapter {
        fn execute(
            &self,
            config: &g_interface::RegenieConfigData,
            execution_context: &NativeExecutionContext,
        ) -> NativeExecutionOutcome {
            assert_eq!(config.input.pheno_columns, vec!["trait".to_string()]);
            assert!(!execution_context.shutdown_requested());
            NativeExecutionOutcome::new(0, "native run succeeded\n".to_string(), String::new())
        }
    }

    struct FailingAdapter;

    impl NativeExecutionAdapter for FailingAdapter {
        fn execute(
            &self,
            config: &g_interface::RegenieConfigData,
            execution_context: &NativeExecutionContext,
        ) -> NativeExecutionOutcome {
            assert_eq!(config.input.pheno_columns, vec!["trait".to_string()]);
            assert!(!execution_context.shutdown_requested());
            NativeExecutionOutcome::new(73, String::new(), "native backend failed\n".to_string())
        }
    }

    struct PanicAdapter;

    impl NativeExecutionAdapter for PanicAdapter {
        fn execute(
            &self,
            _config: &g_interface::RegenieConfigData,
            _execution_context: &NativeExecutionContext,
        ) -> NativeExecutionOutcome {
            panic!("native execution adapter should not be called for invalid config")
        }
    }

    struct PanickingExecutionAdapter;

    impl NativeExecutionAdapter for PanickingExecutionAdapter {
        fn execute(
            &self,
            config: &g_interface::RegenieConfigData,
            execution_context: &NativeExecutionContext,
        ) -> NativeExecutionOutcome {
            assert_eq!(config.input.pheno_columns, vec!["trait".to_string()]);
            assert!(!execution_context.shutdown_requested());
            panic!("backend adapter failed after validation")
        }
    }

    fn string_arguments(arguments: &[&str]) -> Vec<String> {
        arguments.iter().map(|argument| (*argument).to_string()).collect()
    }

    fn unique_fixture_directory() -> PathBuf {
        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after UNIX epoch").as_nanos();
        std::env::temp_dir().join(format!("g-cli-native-{}-{timestamp}", std::process::id()))
    }

    fn valid_regenie_arguments(fixture_directory: &std::path::Path) -> Vec<String> {
        vec![
            "regenie".to_string(),
            "--step".to_string(),
            "2".to_string(),
            "--qt".to_string(),
            "--bgen".to_string(),
            fixture_directory.join("dataset.bgen").to_str().expect("fixture path should be UTF-8").to_string(),
            "--phenoFile".to_string(),
            fixture_directory.join("phenotype.tsv").to_str().expect("fixture path should be UTF-8").to_string(),
            "--phenoCol".to_string(),
            "trait".to_string(),
            "--pred".to_string(),
            fixture_directory.join("predictions.list").to_str().expect("fixture path should be UTF-8").to_string(),
            "--out".to_string(),
            fixture_directory
                .join("results")
                .join("output")
                .to_str()
                .expect("fixture path should be UTF-8")
                .to_string(),
        ]
    }

    fn wait_for_shutdown_request(execution_context: &NativeExecutionContext) {
        for _ in 0..1000 {
            if execution_context.shutdown_requested() {
                return;
            }
            std::thread::yield_now();
        }
        panic!("native signal handler did not record shutdown request");
    }

    #[test]
    fn native_cli_frontend_matches_interface_root_help() {
        let arguments = string_arguments(&["--help"]);
        let native_outcome = dispatch_native_cli(&arguments);
        let interface_outcome = g_interface::dispatch_cli(&arguments);

        assert_eq!(native_outcome.exit_code, interface_outcome.exit_code);
        assert_eq!(native_outcome.stdout, interface_outcome.stdout);
        assert_eq!(native_outcome.stderr, interface_outcome.stderr);
        assert!(!native_outcome.validated_run_config);
    }

    #[test]
    fn native_cli_frontend_matches_interface_regenie_help() {
        let arguments = string_arguments(&["regenie", "--help"]);
        let native_outcome = dispatch_native_cli(&arguments);
        let interface_outcome = g_interface::dispatch_cli(&arguments);

        assert_eq!(native_outcome.exit_code, interface_outcome.exit_code);
        assert_eq!(native_outcome.stdout, interface_outcome.stdout);
        assert_eq!(native_outcome.stderr, interface_outcome.stderr);
        assert!(!native_outcome.validated_run_config);
    }

    #[test]
    fn native_cli_frontend_matches_interface_parse_errors() {
        let arguments = string_arguments(&["unknown"]);
        let native_outcome = dispatch_native_cli(&arguments);
        let interface_outcome = g_interface::dispatch_cli(&arguments);

        assert_eq!(native_outcome.exit_code, interface_outcome.exit_code);
        assert_eq!(native_outcome.stdout, interface_outcome.stdout);
        assert_eq!(native_outcome.stderr, interface_outcome.stderr);
        assert!(!native_outcome.validated_run_config);
    }

    #[test]
    fn native_cli_frontend_validates_config_before_refusing_execution() {
        let fixture_directory = unique_fixture_directory();
        fs::create_dir_all(&fixture_directory).expect("fixture directory should be created");
        fs::write(fixture_directory.join("dataset.bgen"), b"").expect("BGEN fixture should be written");
        fs::write(fixture_directory.join("phenotype.tsv"), "FID IID trait\n")
            .expect("phenotype fixture should be written");
        fs::write(fixture_directory.join("predictions.list"), "").expect("prediction fixture should be written");

        let arguments = valid_regenie_arguments(&fixture_directory);
        let native_outcome = dispatch_native_cli(&arguments);

        assert_eq!(native_outcome.exit_code, NATIVE_EXECUTION_UNAVAILABLE_EXIT_CODE);
        assert_eq!(native_outcome.stdout, "");
        assert_eq!(native_outcome.stderr, NATIVE_EXECUTION_UNAVAILABLE_MESSAGE);
        assert!(native_outcome.validated_run_config);

        fs::remove_dir_all(&fixture_directory).expect("fixture directory should be removed");
    }

    #[test]
    fn native_cli_frontend_routes_validated_config_to_execution_adapter() {
        let fixture_directory = unique_fixture_directory();
        fs::create_dir_all(&fixture_directory).expect("fixture directory should be created");
        fs::write(fixture_directory.join("dataset.bgen"), b"").expect("BGEN fixture should be written");
        fs::write(fixture_directory.join("phenotype.tsv"), "FID IID trait\n")
            .expect("phenotype fixture should be written");
        fs::write(fixture_directory.join("predictions.list"), "").expect("prediction fixture should be written");

        let arguments = valid_regenie_arguments(&fixture_directory);
        let native_outcome = dispatch_native_cli_with_adapter(&arguments, &SuccessfulAdapter);

        assert_eq!(native_outcome.exit_code, 0);
        assert_eq!(native_outcome.stdout, "native run succeeded\n");
        assert_eq!(native_outcome.stderr, "");
        assert!(native_outcome.validated_run_config);

        fs::remove_dir_all(&fixture_directory).expect("fixture directory should be removed");
    }

    #[test]
    fn native_cli_frontend_preserves_execution_adapter_failures() {
        let fixture_directory = unique_fixture_directory();
        fs::create_dir_all(&fixture_directory).expect("fixture directory should be created");
        fs::write(fixture_directory.join("dataset.bgen"), b"").expect("BGEN fixture should be written");
        fs::write(fixture_directory.join("phenotype.tsv"), "FID IID trait\n")
            .expect("phenotype fixture should be written");
        fs::write(fixture_directory.join("predictions.list"), "").expect("prediction fixture should be written");

        let arguments = valid_regenie_arguments(&fixture_directory);
        let native_outcome = dispatch_native_cli_with_adapter(&arguments, &FailingAdapter);

        assert_eq!(native_outcome.exit_code, 73);
        assert_eq!(native_outcome.stdout, "");
        assert_eq!(native_outcome.stderr, "native backend failed\n");
        assert!(native_outcome.validated_run_config);

        fs::remove_dir_all(&fixture_directory).expect("fixture directory should be removed");
    }

    #[test]
    fn native_cli_frontend_keeps_validation_errors_before_execution_boundary() {
        let arguments = string_arguments(&[
            "regenie",
            "--step",
            "2",
            "--qt",
            "--bgen",
            "missing.bgen",
            "--phenoFile",
            "missing.tsv",
            "--phenoCol",
            "trait",
            "--pred",
            "missing.list",
            "--out",
            "results/output",
        ]);

        let native_outcome = dispatch_native_cli(&arguments);

        assert_eq!(native_outcome.exit_code, 1);
        assert_eq!(native_outcome.stdout, "");
        assert!(native_outcome.stderr.contains("--bgen path does not exist"));
        assert!(!native_outcome.validated_run_config);
    }

    #[test]
    fn native_cli_frontend_does_not_call_execution_adapter_for_validation_errors() {
        let arguments = string_arguments(&[
            "regenie",
            "--step",
            "2",
            "--qt",
            "--bgen",
            "missing.bgen",
            "--phenoFile",
            "missing.tsv",
            "--phenoCol",
            "trait",
            "--pred",
            "missing.list",
            "--out",
            "results/output",
        ]);

        let native_outcome = dispatch_native_cli_with_adapter(&arguments, &PanicAdapter);

        assert_eq!(native_outcome.exit_code, 1);
        assert_eq!(native_outcome.stdout, "");
        assert!(native_outcome.stderr.contains("--bgen path does not exist"));
        assert!(!native_outcome.validated_run_config);
    }

    #[test]
    fn native_cli_frontend_reports_execution_adapter_panics_as_runtime_failures() {
        let fixture_directory = unique_fixture_directory();
        fs::create_dir_all(&fixture_directory).expect("fixture directory should be created");
        fs::write(fixture_directory.join("dataset.bgen"), b"").expect("BGEN fixture should be written");
        fs::write(fixture_directory.join("phenotype.tsv"), "FID IID trait\n")
            .expect("phenotype fixture should be written");
        fs::write(fixture_directory.join("predictions.list"), "").expect("prediction fixture should be written");

        let arguments = valid_regenie_arguments(&fixture_directory);
        let native_outcome = dispatch_native_cli_with_adapter(&arguments, &PanickingExecutionAdapter);

        assert_eq!(native_outcome.exit_code, NATIVE_EXECUTION_PANIC_EXIT_CODE);
        assert_eq!(native_outcome.stdout, "");
        assert_eq!(
            native_outcome.stderr,
            "Error: native CLI execution adapter panicked: backend adapter failed after validation.\n",
        );
        assert!(native_outcome.validated_run_config);

        fs::remove_dir_all(&fixture_directory).expect("fixture directory should be removed");
    }

    #[test]
    fn native_execution_context_records_registered_signal() {
        let _signal_test_guard = SIGNAL_TEST_MUTEX.lock().expect("signal test mutex should not be poisoned");
        let execution_context = NativeExecutionContext::default();
        let _signal_guard =
            NativeSignalGuard::install_for_signal_numbers(&[signal_hook::consts::SIGUSR1], &execution_context, false)
                .expect("SIGUSR1 handler should be installed");

        signal_hook::low_level::raise(signal_hook::consts::SIGUSR1).expect("SIGUSR1 should be raised");

        wait_for_shutdown_request(&execution_context);
    }
}
