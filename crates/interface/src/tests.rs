use std::path::{Path, PathBuf};

use crate::test_support::TemporaryDirectory;
use crate::{CliDispatch, CompiledCliRun, dispatch_cli};

const TEST_BGEN_CONTENT_SHA256: &str = "abababababababababababababababababababababababababababababababab";

struct CliFixture {
    directory: TemporaryDirectory,
    bgen_path: PathBuf,
    alternate_bgen_path: PathBuf,
    sample_path: PathBuf,
    phenotype_path: PathBuf,
    prediction_list_path: PathBuf,
}

impl CliFixture {
    fn new(test_name: &str) -> Self {
        let directory = TemporaryDirectory::new(test_name);
        let bgen_path = directory.write("genotypes.bgen", "");
        let alternate_bgen_path = directory.write("alternate-genotypes.bgen", "");
        let sample_path = directory.write("samples.sample", "");
        let phenotype_path = directory.write("phenotypes.tsv", "");
        let prediction_list_path = directory.write("pred.list", "");
        Self { directory, bgen_path, alternate_bgen_path, sample_path, phenotype_path, prediction_list_path }
    }

    fn valid_cli_arguments(&self, phenotype_names: &[&str], output_name: &str) -> Vec<String> {
        let mut arguments = vec![
            "regenie".to_string(),
            "--bgen".to_string(),
            path_text(&self.bgen_path).to_string(),
            "--sample".to_string(),
            path_text(&self.sample_path).to_string(),
            "--phenoFile".to_string(),
            path_text(&self.phenotype_path).to_string(),
            "--pred".to_string(),
            path_text(&self.prediction_list_path).to_string(),
            "--qt".to_string(),
            "--out".to_string(),
            path_text(&self.directory.path().join(output_name)).to_string(),
        ];
        for phenotype_name in phenotype_names {
            arguments.push("--phenoCol".to_string());
            arguments.push((*phenotype_name).to_string());
        }
        arguments
    }

    fn write_config(
        &self,
        file_name: &str,
        phenotype_names: &[&str],
        output_run_directory: &Path,
        input_extra_toml: &str,
        extra_toml: &str,
    ) -> PathBuf {
        let phenotype_list =
            phenotype_names.iter().map(|phenotype_name| format!("\"{phenotype_name}\"")).collect::<Vec<_>>().join(", ");
        self.directory.write(
            file_name,
            &format!(
                "[input]\nbgen = \"{}\"\n{input_extra_toml}sample = \"{}\"\npheno_file = \"{}\"\npheno_columns = [{phenotype_list}]\npred = \"{}\"\n\n[output]\nout = \"{}\"\noutput_run_directory = \"{}\"\n\n{extra_toml}",
                self.bgen_path.display(),
                self.sample_path.display(),
                self.phenotype_path.display(),
                self.prediction_list_path.display(),
                self.directory.path().join("prefix").display(),
                output_run_directory.display(),
            ),
        )
    }
}

fn assert_option_schema_version_zero(effective_config_toml: &str) {
    let effective_config =
        toml::from_str::<toml::Table>(effective_config_toml).expect("effective config should be valid TOML");
    let metadata = effective_config
        .get("metadata")
        .and_then(toml::Value::as_table)
        .expect("effective config should contain metadata");
    assert_eq!(
        metadata.get("option-schema-version").and_then(toml::Value::as_integer),
        Some(0),
        "option schema version should remain the integer zero",
    );
}

fn path_text(path: &Path) -> &str {
    path.to_str().expect("temporary paths should be UTF-8")
}

fn one_compiled_run(arguments: &[String]) -> CompiledCliRun {
    match dispatch_cli(arguments) {
        CliDispatch::Runs(mut compiled_runs) if compiled_runs.len() == 1 => {
            compiled_runs.pop().expect("single-run dispatch should contain one compiled run")
        }
        dispatch => panic!("expected one compiled run, observed {dispatch:?}"),
    }
}

fn exit_dispatch(arguments: &[&str]) -> (i32, String, String) {
    let owned_arguments = arguments.iter().map(ToString::to_string).collect::<Vec<_>>();
    match dispatch_cli(&owned_arguments) {
        CliDispatch::Exit { exit_code, stdout, stderr } => (exit_code, stdout, stderr),
        dispatch @ CliDispatch::Runs(_) => panic!("expected exit dispatch, observed {dispatch:?}"),
    }
}

#[test]
fn cli_help_version_and_errors_use_current_dispatch_contract() {
    let (empty_code, empty_stdout, empty_stderr) = exit_dispatch(&[]);
    assert_eq!(empty_code, 2);
    assert!(empty_stdout.contains("Usage: g <COMMAND> [OPTIONS]"));
    assert!(empty_stderr.is_empty());

    let (help_code, help_stdout, help_stderr) = exit_dispatch(&["--help"]);
    assert_eq!(help_code, 0);
    assert!(help_stdout.contains("regenie"));
    assert!(help_stdout.contains("batch"));
    assert!(help_stderr.is_empty());
    let (regenie_help_code, regenie_help, _) = exit_dispatch(&["regenie", "--help"]);
    assert_eq!(regenie_help_code, 0);
    assert!(regenie_help.contains("--phenoCol"));
    assert!(!regenie_help.contains("--bgen-content-sha256"));

    let (digest_code, digest_stdout, digest_stderr) =
        exit_dispatch(&["regenie", "--bgen-content-sha256", TEST_BGEN_CONTENT_SHA256]);
    assert_eq!(digest_code, 1);
    assert!(digest_stdout.is_empty());
    assert!(digest_stderr.contains("unexpected argument '--bgen-content-sha256'"));

    let (version_code, version_stdout, version_stderr) = exit_dispatch(&["--version"]);
    assert_eq!(version_code, 2);
    assert!(version_stdout.is_empty());
    assert_eq!(version_stderr, "No such command: --version\n");
    let (unknown_code, unknown_stdout, unknown_stderr) = exit_dispatch(&["unknown"]);
    assert_eq!(unknown_code, 2);
    assert!(unknown_stdout.is_empty());
    assert_eq!(unknown_stderr, "No such command: unknown\n");
    let (parse_code, parse_stdout, parse_stderr) = exit_dispatch(&["regenie", "--unsupported"]);
    assert_eq!(parse_code, 1);
    assert!(parse_stdout.is_empty());
    assert!(parse_stderr.contains("unexpected argument '--unsupported'"));
}

#[test]
fn defaults_toml_and_cli_layers_follow_precedence() {
    let fixture = CliFixture::new("overlay");
    let toml_output_root = fixture.directory.path().join("toml-output-root");
    let config_path = fixture.write_config(
        "overlay.toml",
        &["toml-trait"],
        &toml_output_root,
        "",
        "writer_threads = 8\nresume = true\n\n[trait]\ntrait_type = \"binary\"\nbsize = 2048\n\n[compute]\ndevice = \"gpu\"\nmulti_phenotype_sample_mode = \"complete-case\"\nfirth_batch_size = 64\nfirth_candidate_capacity = 128\n\n[diagnostics]\ntelemetry = \"off\"\n",
    );
    let cli_output_prefix = fixture.directory.path().join("cli-output");
    let arguments = vec![
        "regenie".to_string(),
        "--config".to_string(),
        path_text(&config_path).to_string(),
        "--qt".to_string(),
        "--bsize".to_string(),
        "4096".to_string(),
        "--phenoCol".to_string(),
        "cli-a".to_string(),
        "--phenoCol".to_string(),
        "cli-b".to_string(),
        "--out".to_string(),
        path_text(&cli_output_prefix).to_string(),
    ];
    let compiled_run = one_compiled_run(&arguments);
    assert_eq!(compiled_run.run_plan.association_mode, g_plan::AssociationMode::Regenie2Linear);
    assert_eq!(compiled_run.run_plan.chunk_size, 4096);
    assert_eq!(compiled_run.run_plan.compute.device, g_plan::Device::Gpu);
    assert_eq!(
        compiled_run.run_plan.compute.multi_phenotype_sample_mode,
        g_plan::MultiPhenotypeSampleMode::CompleteCase
    );
    assert_eq!(compiled_run.run_plan.compute.kernels.firth.batch_size, 64);
    assert_eq!(compiled_run.run_plan.compute.kernels.firth.candidate_capacity, 128);
    assert_eq!(compiled_run.run_plan.output.writer_thread_count, 8);
    assert!(compiled_run.run_plan.output.resume);
    assert_eq!(compiled_run.run_plan.output.output_run_root, toml_output_root.display().to_string());
    assert!(compiled_run.effective_config_toml.contains(path_text(&cli_output_prefix)));
    assert_eq!(
        compiled_run.run_plan.phenotype_runs.iter().map(|run| run.phenotype_name.as_str()).collect::<Vec<_>>(),
        ["cli-a", "cli-b"]
    );
    assert!(compiled_run.effective_config_toml.contains("option-schema-version = 0"));
}

#[test]
fn current_effective_toml_round_trips_with_schema_version_zero() {
    let fixture = CliFixture::new("effective-round-trip");
    let initial_arguments = fixture.valid_cli_arguments(&["trait-a"], "initial-output");
    let initial_run = one_compiled_run(&initial_arguments);
    assert!(initial_run.effective_config_toml.contains("option-schema-version = 0"));
    assert_option_schema_version_zero(&initial_run.effective_config_toml);
    let effective_config_path = fixture.directory.write("effective.toml", &initial_run.effective_config_toml);
    let replay_arguments =
        vec!["regenie".to_string(), "--config".to_string(), path_text(&effective_config_path).to_string()];
    let replay_run = one_compiled_run(&replay_arguments);
    assert_eq!(replay_run.run_plan, initial_run.run_plan);

    let stale_metadata =
        initial_run.effective_config_toml.replace("option-schema-version = 0", "option-schema-version = 6");
    let stale_config_path = fixture.directory.write("stale-effective.toml", &stale_metadata);
    let (exit_code, _, stderr) = exit_dispatch(&["regenie", "--config", path_text(&stale_config_path)]);
    assert_eq!(exit_code, 1);
    assert!(stderr.contains("can only be used with --bt"));
}

#[test]
fn toml_bgen_content_digest_round_trips_and_survives_locator_override() {
    let fixture = CliFixture::new("bgen-content-digest");
    let output_run_directory = fixture.directory.path().join("digest-output");
    let digest_toml = format!("bgen_content_sha256 = \"{TEST_BGEN_CONTENT_SHA256}\"\n");
    let config_path = fixture.write_config("digest.toml", &["trait-a"], &output_run_directory, &digest_toml, "");
    let config_arguments = vec!["regenie".to_string(), "--config".to_string(), path_text(&config_path).to_string()];

    let initial_run = one_compiled_run(&config_arguments);
    assert_eq!(initial_run.run_plan.input.bgen_path, path_text(&fixture.bgen_path));
    assert_eq!(
        initial_run.run_plan.input.bgen_content_sha256.expect("TOML digest should reach the run plan").to_string(),
        TEST_BGEN_CONTENT_SHA256,
    );
    assert!(initial_run.effective_config_toml.contains(&digest_toml));
    assert_option_schema_version_zero(&initial_run.effective_config_toml);

    let effective_config_path = fixture.directory.write("digest-effective.toml", &initial_run.effective_config_toml);
    let replay_arguments =
        vec!["regenie".to_string(), "--config".to_string(), path_text(&effective_config_path).to_string()];
    let replay_run = one_compiled_run(&replay_arguments);
    assert_eq!(replay_run.run_plan, initial_run.run_plan);

    let override_arguments = vec![
        "regenie".to_string(),
        "--config".to_string(),
        path_text(&config_path).to_string(),
        "--bgen".to_string(),
        path_text(&fixture.alternate_bgen_path).to_string(),
    ];
    let override_run = one_compiled_run(&override_arguments);
    assert_eq!(override_run.run_plan.input.bgen_path, path_text(&fixture.alternate_bgen_path));
    assert_eq!(
        override_run
            .run_plan
            .input
            .bgen_content_sha256
            .expect("locator override should preserve the TOML digest")
            .to_string(),
        TEST_BGEN_CONTENT_SHA256,
    );
    assert!(override_run.effective_config_toml.contains(path_text(&fixture.alternate_bgen_path)));
    assert!(override_run.effective_config_toml.contains(&digest_toml));
}

#[test]
fn toml_rejects_boolean_bgen_content_digest() {
    let fixture = CliFixture::new("boolean-bgen-content-digest");
    let config_path = fixture.write_config(
        "boolean-digest.toml",
        &["trait-a"],
        &fixture.directory.path().join("boolean-digest-output"),
        "bgen_content_sha256 = true\n",
        "",
    );

    let (exit_code, stdout, stderr) = exit_dispatch(&["regenie", "--config", path_text(&config_path)]);
    assert_eq!(exit_code, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("Invalid TOML config"));
    assert!(stderr.contains("invalid type: boolean"));
}

#[test]
fn configuration_validation_rejects_conflicts_duplicates_and_missing_paths() {
    let fixture = CliFixture::new("validation");
    let duplicate_arguments = fixture.valid_cli_arguments(&["trait-a", "trait-a"], "duplicate-output");
    let (duplicate_code, _, duplicate_stderr) = match dispatch_cli(&duplicate_arguments) {
        CliDispatch::Exit { exit_code, stdout, stderr } => (exit_code, stdout, stderr),
        dispatch @ CliDispatch::Runs(_) => panic!("expected duplicate-name error, observed {dispatch:?}"),
    };
    assert_eq!(duplicate_code, 1);
    assert!(duplicate_stderr.contains("Duplicate phenotype names"));

    let mut binary_only_arguments = fixture.valid_cli_arguments(&["trait-a"], "binary-only-output");
    binary_only_arguments.push("--pThresh".to_string());
    binary_only_arguments.push("0.01".to_string());
    let (_, _, binary_only_stderr) = match dispatch_cli(&binary_only_arguments) {
        CliDispatch::Exit { exit_code, stdout, stderr } => (exit_code, stdout, stderr),
        dispatch @ CliDispatch::Runs(_) => {
            panic!("expected quantitative binary-option error, observed {dispatch:?}")
        }
    };
    assert!(binary_only_stderr.contains("can only be used with --bt"));

    let missing_path_arguments = vec![
        "regenie".to_string(),
        "--bgen".to_string(),
        path_text(&fixture.directory.path().join("missing.bgen")).to_string(),
        "--sample".to_string(),
        path_text(&fixture.sample_path).to_string(),
        "--phenoFile".to_string(),
        path_text(&fixture.phenotype_path).to_string(),
        "--phenoCol".to_string(),
        "trait-a".to_string(),
        "--pred".to_string(),
        path_text(&fixture.prediction_list_path).to_string(),
        "--out".to_string(),
        path_text(&fixture.directory.path().join("missing-output")).to_string(),
    ];
    match dispatch_cli(&missing_path_arguments) {
        CliDispatch::Exit { exit_code: 1, stderr, .. } => assert!(stderr.contains("--bgen path does not exist")),
        dispatch => panic!("expected missing-path error, observed {dispatch:?}"),
    }
}

#[test]
fn run_plan_and_batch_dispatch_preserve_geometry_and_disjoint_outputs() {
    let fixture = CliFixture::new("plan-batch");
    let mut arguments = fixture.valid_cli_arguments(&["alpha", "Beta / gamma"], "plan-output");
    arguments.push("--bsize".to_string());
    arguments.push("8192".to_string());
    let compiled_run = one_compiled_run(&arguments);
    assert_eq!(compiled_run.run_plan.chunk_size, 8192);
    assert_eq!(compiled_run.run_plan.output.writer_thread_count, 4);
    assert_eq!(compiled_run.run_plan.compute.kernels.firth.batch_size, 512);
    assert_eq!(compiled_run.run_plan.compute.kernels.firth.candidate_capacity, 1024);
    assert_eq!(compiled_run.run_plan.phenotype_runs[0].output_directory_name, "trait_0001_alpha");
    assert_eq!(compiled_run.run_plan.phenotype_runs[1].output_directory_name, "trait_0002_Beta_gamma");

    let first_config =
        fixture.write_config("first.toml", &["alpha"], &fixture.directory.path().join("batch-first"), "", "");
    let second_config =
        fixture.write_config("second.toml", &["beta"], &fixture.directory.path().join("batch-second"), "", "");
    let batch_arguments = vec![
        "batch".to_string(),
        "--config".to_string(),
        path_text(&first_config).to_string(),
        "--config".to_string(),
        path_text(&second_config).to_string(),
    ];
    match dispatch_cli(&batch_arguments) {
        CliDispatch::Runs(compiled_runs) => {
            assert_eq!(compiled_runs.len(), 2);
            assert_eq!(compiled_runs[0].run_plan.phenotype_runs[0].phenotype_name, "alpha");
            assert_eq!(compiled_runs[1].run_plan.phenotype_runs[0].phenotype_name, "beta");
        }
        dispatch @ CliDispatch::Exit { .. } => panic!("expected batch runs, observed {dispatch:?}"),
    }

    let nested_config =
        fixture.write_config("nested.toml", &["beta"], &fixture.directory.path().join("batch-first/nested"), "", "");
    let nested_arguments = vec![
        "batch".to_string(),
        "--config".to_string(),
        path_text(&first_config).to_string(),
        "--config".to_string(),
        path_text(&nested_config).to_string(),
    ];
    match dispatch_cli(&nested_arguments) {
        CliDispatch::Exit { exit_code: 1, stderr, .. } => {
            assert!(stderr.contains("equal or nested output run roots"));
        }
        dispatch => panic!("expected nested-output error, observed {dispatch:?}"),
    }
}
