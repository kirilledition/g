//! Native CLI frontend for the GWAS engine.

pub const NATIVE_EXECUTION_UNAVAILABLE_EXIT_CODE: i32 = 1;
pub const NATIVE_EXECUTION_UNAVAILABLE_MESSAGE: &str = concat!(
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
    let validated_run_config = config.is_some();
    if validated_run_config {
        stderr.push_str(NATIVE_EXECUTION_UNAVAILABLE_MESSAGE);
        return NativeCliOutcome::new(NATIVE_EXECUTION_UNAVAILABLE_EXIT_CODE, stdout, stderr, validated_run_config);
    }
    NativeCliOutcome::new(exit_code, stdout, stderr, validated_run_config)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

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
}
