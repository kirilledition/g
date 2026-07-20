use std::hint;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use criterion::{Criterion, criterion_group, criterion_main};

struct BenchmarkFixture {
    root_path: PathBuf,
    config_path: PathBuf,
}

impl BenchmarkFixture {
    fn new() -> Self {
        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after UNIX epoch").as_nanos();
        let root_path =
            std::env::temp_dir().join(format!("g-interface-frontend-bench-{}-{timestamp}", std::process::id()));
        std::fs::create_dir_all(&root_path).expect("benchmark fixture directory should be created");
        std::fs::write(root_path.join("dataset.bgen"), b"").expect("BGEN fixture should be written");
        std::fs::write(root_path.join("phenotype.tsv"), "FID IID trait\n")
            .expect("phenotype fixture should be written");
        std::fs::write(root_path.join("predictions.list"), "").expect("prediction fixture should be written");
        let config_path = write_regenie_toml(&root_path);
        Self { root_path, config_path }
    }
}

impl Drop for BenchmarkFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root_path);
    }
}

fn string_arguments(arguments: &[&str]) -> Vec<String> {
    arguments.iter().map(|argument| (*argument).to_string()).collect()
}

fn valid_regenie_arguments(fixture_root_path: &Path) -> Vec<String> {
    vec![
        "regenie".to_string(),
        "--step".to_string(),
        "2".to_string(),
        "--qt".to_string(),
        "--bgen".to_string(),
        fixture_root_path.join("dataset.bgen").to_str().expect("fixture path should be UTF-8").to_string(),
        "--phenoFile".to_string(),
        fixture_root_path.join("phenotype.tsv").to_str().expect("fixture path should be UTF-8").to_string(),
        "--phenoCol".to_string(),
        "trait".to_string(),
        "--pred".to_string(),
        fixture_root_path.join("predictions.list").to_str().expect("fixture path should be UTF-8").to_string(),
        "--out".to_string(),
        fixture_root_path.join("results").join("output").to_str().expect("fixture path should be UTF-8").to_string(),
    ]
}

fn valid_regenie_toml_arguments(config_path: &Path) -> Vec<String> {
    vec![
        "regenie".to_string(),
        "--config".to_string(),
        config_path.to_str().expect("fixture path should be UTF-8").to_string(),
    ]
}

fn write_regenie_toml(fixture_root_path: &Path) -> PathBuf {
    let config_path = fixture_root_path.join("config.toml");
    let bgen_value = toml_path_value(&fixture_root_path.join("dataset.bgen"));
    let phenotype_value = toml_path_value(&fixture_root_path.join("phenotype.tsv"));
    let prediction_value = toml_path_value(&fixture_root_path.join("predictions.list"));
    let output_value = toml_path_value(&fixture_root_path.join("results").join("output"));
    std::fs::write(
        &config_path,
        format!(
            "\
[input]
bgen = {bgen_value}
phenoFile = {phenotype_value}
phenoCol = \"trait\"
pred = {prediction_value}

[trait]
step = 2
qt = true

[output]
out = {output_value}
"
        ),
    )
    .expect("benchmark TOML fixture should be written");
    config_path
}

fn toml_path_value(path: &Path) -> String {
    let path_text = path.to_str().expect("fixture path should be UTF-8");
    format!("\"{}\"", path_text.escape_default())
}

fn benchmark_frontend_dispatch(criterion: &mut Criterion) {
    let fixture = BenchmarkFixture::new();
    let root_help_arguments = string_arguments(&["--help"]);
    let regenie_help_arguments = string_arguments(&["regenie", "--help"]);
    let parse_error_arguments = string_arguments(&["regenie", "--bad-option"]);
    let valid_config_arguments = valid_regenie_arguments(&fixture.root_path);
    let valid_toml_config_arguments = valid_regenie_toml_arguments(&fixture.config_path);

    let mut group = criterion.benchmark_group("cli_frontend_dispatch");
    group.bench_function("root_help", |bencher| {
        bencher.iter(|| hint::black_box(g_interface::dispatch_cli(hint::black_box(&root_help_arguments))));
    });
    group.bench_function("regenie_help", |bencher| {
        bencher.iter(|| hint::black_box(g_interface::dispatch_cli(hint::black_box(&regenie_help_arguments))));
    });
    group.bench_function("parse_error", |bencher| {
        bencher.iter(|| hint::black_box(g_interface::dispatch_cli(hint::black_box(&parse_error_arguments))));
    });
    group.bench_function("valid_config_refusal", |bencher| {
        bencher.iter(|| hint::black_box(g_interface::dispatch_cli(hint::black_box(&valid_config_arguments))));
    });
    group.bench_function("valid_toml_config_refusal", |bencher| {
        bencher.iter(|| {
            hint::black_box(g_interface::dispatch_cli(hint::black_box(&valid_toml_config_arguments)));
        });
    });
    group.finish();
}

criterion_group!(benches, benchmark_frontend_dispatch);
criterion_main!(benches);
