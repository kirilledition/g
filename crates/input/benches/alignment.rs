use std::fs::{self, File};
use std::hint;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const DEFAULT_SAMPLE_COUNT: usize = 100_000;

struct InputBenchmarkFixture {
    root_path: PathBuf,
    sample_path: PathBuf,
    phenotype_path_text: String,
    covariate_path_text: String,
    prediction_paths: Vec<g_input::PredictionLocoPath>,
    phenotype_names: Vec<String>,
    covariate_names: Vec<String>,
}

impl InputBenchmarkFixture {
    fn new(sample_count: usize) -> Self {
        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after UNIX epoch").as_nanos();
        let root_path =
            std::env::temp_dir().join(format!("g-input-alignment-bench-{}-{timestamp}", std::process::id()));
        fs::create_dir_all(&root_path).expect("benchmark fixture directory should be created");

        let sample_path = root_path.join("cohort.sample");
        let phenotype_path = root_path.join("phenotypes.tsv");
        let covariate_path = root_path.join("covariates.tsv");
        let prediction_path = root_path.join("trait.loco");
        write_sample_file(&sample_path, sample_count);
        write_phenotype_file(&phenotype_path, sample_count);
        write_covariate_file(&covariate_path, sample_count);
        write_prediction_file(&prediction_path, sample_count);

        Self {
            root_path,
            sample_path,
            phenotype_path_text: phenotype_path.to_string_lossy().into_owned(),
            covariate_path_text: covariate_path.to_string_lossy().into_owned(),
            prediction_paths: vec![g_input::PredictionLocoPath {
                phenotype_name: Arc::from("trait"),
                loco_file_path: prediction_path,
            }],
            phenotype_names: vec!["trait".to_string()],
            covariate_names: vec!["age".to_string(), "sex".to_string()],
        }
    }
}

impl Drop for InputBenchmarkFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root_path);
    }
}

fn buffered_file(path: &Path) -> BufWriter<File> {
    BufWriter::new(File::create(path).expect("benchmark fixture file should be created"))
}

fn write_sample_file(path: &Path, sample_count: usize) {
    let mut writer = buffered_file(path);
    writer.write_all(b"ID_1 ID_2\n0 0\n").expect("sample header should be written");
    for sample_index in 0..sample_count {
        writeln!(writer, "F{sample_index} I{sample_index}").expect("sample row should be written");
    }
}

fn write_phenotype_file(path: &Path, sample_count: usize) {
    let mut writer = buffered_file(path);
    writer.write_all(b"FID\tIID\ttrait\n").expect("phenotype header should be written");
    for sample_index in 0..sample_count {
        let phenotype_value = 1 + sample_index % 2;
        writeln!(writer, "F{sample_index}\tI{sample_index}\t{phenotype_value}")
            .expect("phenotype row should be written");
    }
}

fn write_covariate_file(path: &Path, sample_count: usize) {
    let mut writer = buffered_file(path);
    writer.write_all(b"FID\tIID\tage\tsex\n").expect("covariate header should be written");
    for sample_index in 0..sample_count {
        let age = 40 + sample_index % 40;
        let sex = sample_index % 2;
        writeln!(writer, "F{sample_index}\tI{sample_index}\t{age}\t{sex}").expect("covariate row should be written");
    }
}

fn write_prediction_file(path: &Path, sample_count: usize) {
    let mut writer = buffered_file(path);
    writer.write_all(b"FID_IID").expect("prediction header marker should be written");
    for sample_index in 0..sample_count {
        write!(writer, " F{sample_index}_I{sample_index}").expect("prediction sample key should be written");
    }
    writer.write_all(b"\n").expect("prediction header terminator should be written");
    for chromosome in 1..=22 {
        write!(writer, "{chromosome}").expect("prediction chromosome should be written");
        for sample_index in 0..sample_count {
            let prediction_hundredths = sample_index % 101;
            let prediction_whole = prediction_hundredths / 100;
            let prediction_fraction = prediction_hundredths % 100;
            write!(writer, " {prediction_whole}.{prediction_fraction:02}").expect("prediction value should be written");
        }
        writer.write_all(b"\n").expect("prediction row terminator should be written");
    }
}

fn configured_sample_count() -> usize {
    std::env::var("G_INPUT_BENCH_SAMPLE_COUNT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_SAMPLE_COUNT)
}

fn benchmark_input_alignment(criterion: &mut Criterion) {
    let sample_count = configured_sample_count();
    let fixture = InputBenchmarkFixture::new(sample_count);
    let sample_identifiers = g_input::load_sample_identifier_data_from_sample_file(&fixture.sample_path, sample_count)
        .expect("benchmark sample fixture should load");

    let mut group = criterion.benchmark_group("input_alignment");
    group.sample_size(10);
    group.throughput(Throughput::Elements(u64::try_from(sample_count).expect("benchmark sample count should fit u64")));
    group.bench_with_input(BenchmarkId::new("sample_identifiers", sample_count), &sample_count, |bencher, count| {
        bencher.iter(|| {
            hint::black_box(
                g_input::load_sample_identifier_data_from_sample_file(&fixture.sample_path, *count)
                    .expect("benchmark sample fixture should load"),
            );
        });
    });
    group.bench_with_input(BenchmarkId::new("phenotype_covariate_loco", sample_count), &sample_count, |bencher, _| {
        bencher.iter(|| {
            let request = g_input::PhenotypeGroupLoadRequest {
                sample_identifiers: &sample_identifiers,
                phenotype_path: &fixture.phenotype_path_text,
                prediction_loco_paths: &fixture.prediction_paths,
                phenotype_names: &fixture.phenotype_names,
                covariate_path: Some(&fixture.covariate_path_text),
                covariate_names: Some(&fixture.covariate_names),
                is_binary_trait: true,
                sample_mode: g_plan::MultiPhenotypeSampleMode::PerPhenotype,
            };
            let mut groups =
                g_input::load_aligned_phenotype_groups(&request).expect("benchmark aligned group should load");
            let group = groups.first_mut().expect("benchmark should produce one aligned group");
            group.plan_prediction_uses(&[Arc::from("22")]).expect("benchmark chromosome should be planned");
            hint::black_box(
                group.take_chromosome_prediction_matrix("22").expect("benchmark chromosome prediction should load"),
            );
        });
    });
    group.finish();
}

criterion_group!(benches, benchmark_input_alignment);
criterion_main!(benches);
