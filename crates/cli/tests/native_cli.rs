use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn string_arguments(arguments: &[&str]) -> Vec<String> {
    arguments.iter().map(|argument| (*argument).to_string()).collect()
}

fn run_native_binary(arguments: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_g")).args(arguments).output().expect("native g binary should execute")
}

fn unique_fixture_directory() -> PathBuf {
    let timestamp =
        SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after UNIX epoch").as_nanos();
    std::env::temp_dir().join(format!("g-cli-native-bin-{}-{timestamp}", std::process::id()))
}

fn write_valid_fixture(fixture_directory: &Path) {
    fs::create_dir_all(fixture_directory).expect("fixture directory should be created");
    fs::write(fixture_directory.join("dataset.bgen"), b"").expect("BGEN fixture should be written");
    fs::write(fixture_directory.join("phenotype.tsv"), "FID IID trait\n").expect("phenotype fixture should be written");
    fs::write(fixture_directory.join("predictions.list"), "").expect("prediction fixture should be written");
}

fn valid_regenie_arguments(fixture_directory: &Path) -> Vec<String> {
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
        fixture_directory.join("results").join("output").to_str().expect("fixture path should be UTF-8").to_string(),
    ]
}

fn write_regenie_toml(fixture_directory: &Path, bgen_path: &Path) -> PathBuf {
    let config_path = fixture_directory.join("config.toml");
    let phenotype_path = fixture_directory.join("phenotype.tsv");
    let prediction_path = fixture_directory.join("predictions.list");
    let output_path = fixture_directory.join("results").join("output");
    let bgen_value = toml_path_value(bgen_path);
    let phenotype_value = toml_path_value(&phenotype_path);
    let prediction_value = toml_path_value(&prediction_path);
    let output_value = toml_path_value(&output_path);
    fs::write(
        &config_path,
        format!(
            "\
[input]
bgen = {bgen_value}
phenoFile = {phenotype_value}
pred = {prediction_value}

[trait]
step = 2
qt = true

[output]
out = {output_value}
format = \"parquet\"
"
        ),
    )
    .expect("TOML fixture should be written");
    config_path
}

fn toml_path_value(path: &Path) -> String {
    let path_text = path.to_str().expect("fixture path should be UTF-8");
    format!("\"{}\"", path_text.escape_default())
}

#[test]
fn native_binary_help_matches_interface_frontend() {
    let output = run_native_binary(&["--help"]);
    let expected = g_interface::dispatch_cli(&string_arguments(&["--help"]));

    assert_eq!(output.status.code(), Some(expected.exit_code));
    assert_eq!(String::from_utf8(output.stdout).expect("stdout should be UTF-8"), expected.stdout);
    assert_eq!(String::from_utf8(output.stderr).expect("stderr should be UTF-8"), expected.stderr);
}

#[test]
fn native_binary_regenie_help_matches_interface_frontend() {
    let output = run_native_binary(&["regenie", "--help"]);
    let expected = g_interface::dispatch_cli(&string_arguments(&["regenie", "--help"]));

    assert_eq!(output.status.code(), Some(expected.exit_code));
    assert_eq!(String::from_utf8(output.stdout).expect("stdout should be UTF-8"), expected.stdout);
    assert_eq!(String::from_utf8(output.stderr).expect("stderr should be UTF-8"), expected.stderr);
}

#[test]
fn native_binary_validates_config_before_refusing_execution() {
    let fixture_directory = unique_fixture_directory();
    write_valid_fixture(&fixture_directory);
    let arguments = valid_regenie_arguments(&fixture_directory);

    let output =
        Command::new(env!("CARGO_BIN_EXE_g")).args(arguments).output().expect("native g binary should execute");

    assert_eq!(output.status.code(), Some(g_cli::NATIVE_EXECUTION_UNAVAILABLE_EXIT_CODE));
    assert_eq!(String::from_utf8(output.stdout).expect("stdout should be UTF-8"), "");
    assert_eq!(
        String::from_utf8(output.stderr).expect("stderr should be UTF-8"),
        g_cli::NATIVE_EXECUTION_UNAVAILABLE_MESSAGE,
    );

    fs::remove_dir_all(&fixture_directory).expect("fixture directory should be removed");
}

#[test]
fn native_binary_validates_toml_and_cli_overrides_before_refusing_execution() {
    let fixture_directory = unique_fixture_directory();
    write_valid_fixture(&fixture_directory);
    let missing_bgen_path = fixture_directory.join("missing.bgen");
    let config_path = write_regenie_toml(&fixture_directory, &missing_bgen_path);

    let output = Command::new(env!("CARGO_BIN_EXE_g"))
        .args([
            "regenie",
            "--config",
            config_path.to_str().expect("config path should be UTF-8"),
            "--bgen",
            fixture_directory.join("dataset.bgen").to_str().expect("BGEN path should be UTF-8"),
            "--phenoCol",
            "trait",
            "--format",
            "arrow",
        ])
        .output()
        .expect("native g binary should execute");

    assert_eq!(output.status.code(), Some(g_cli::NATIVE_EXECUTION_UNAVAILABLE_EXIT_CODE));
    assert_eq!(String::from_utf8(output.stdout).expect("stdout should be UTF-8"), "");
    assert_eq!(
        String::from_utf8(output.stderr).expect("stderr should be UTF-8"),
        g_cli::NATIVE_EXECUTION_UNAVAILABLE_MESSAGE,
    );

    fs::remove_dir_all(&fixture_directory).expect("fixture directory should be removed");
}

#[test]
fn native_binary_reports_toml_validation_errors_before_execution_boundary() {
    let fixture_directory = unique_fixture_directory();
    write_valid_fixture(&fixture_directory);
    let missing_bgen_path = fixture_directory.join("missing.bgen");
    let config_path = write_regenie_toml(&fixture_directory, &missing_bgen_path);

    let output = Command::new(env!("CARGO_BIN_EXE_g"))
        .args([
            "regenie",
            "--config",
            config_path.to_str().expect("config path should be UTF-8"),
            "--phenoCol",
            "trait",
        ])
        .output()
        .expect("native g binary should execute");
    let stderr = String::from_utf8(output.stderr).expect("stderr should be UTF-8");

    assert_eq!(output.status.code(), Some(1));
    assert_eq!(String::from_utf8(output.stdout).expect("stdout should be UTF-8"), "");
    assert!(stderr.contains("--bgen path does not exist"));
    assert!(!stderr.contains(g_cli::NATIVE_EXECUTION_UNAVAILABLE_MESSAGE));

    fs::remove_dir_all(&fixture_directory).expect("fixture directory should be removed");
}
