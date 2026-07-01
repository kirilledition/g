use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{ArrayRef, Float32Array};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use g_output::{
    CurrentRunManifestHeaderInput, NativeChunkHandle, NativeChunkStats, OutputFileFormat, OutputResumeMode,
    OutputWriterSession, VariantMetadataColumns, build_current_run_manifest_header_json, initialize_output_run,
    prepare_output_run,
};

const BENCHMARK_ROW_COUNT: usize = 1024;
const BENCHMARK_CHUNK_COUNT: usize = 8;

struct BenchmarkRun {
    root_path: PathBuf,
    run_dir: PathBuf,
    chunks_dir: PathBuf,
}

impl Drop for BenchmarkRun {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root_path);
    }
}

fn unique_benchmark_root(benchmark_name: &str) -> PathBuf {
    let unique_suffix =
        SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after Unix epoch").as_nanos();
    std::env::temp_dir().join(format!("g-output-writer-bench-{}-{unique_suffix}", benchmark_name.replace('/', "_")))
}

fn prepare_unmanifested_run(benchmark_name: &str, output_format: &str) -> BenchmarkRun {
    let root_path = unique_benchmark_root(benchmark_name);
    let output_root = root_path.join("output");
    let output_file_format = OutputFileFormat::parse(output_format).expect("benchmark output format should parse");
    let prepared_output_run =
        prepare_output_run(&output_root, "regenie2_linear", output_file_format, false).expect("run should prepare");
    BenchmarkRun {
        root_path,
        run_dir: prepared_output_run.output_run_paths.run_directory,
        chunks_dir: prepared_output_run.output_run_paths.chunks_directory,
    }
}

fn prepare_manifested_run(
    benchmark_name: &str,
    output_format: &str,
    finalize_parquet: bool,
    chunks_per_arrow_file: usize,
) -> BenchmarkRun {
    let root_path = unique_benchmark_root(benchmark_name);
    std::fs::create_dir_all(&root_path).expect("benchmark root should be created");
    let bgen_path = write_benchmark_file(&root_path, "input.bgen", b"benchmark bgen");
    let phenotype_path = write_benchmark_file(&root_path, "phenotypes.tsv", b"sample\ttrait\ns1\t1\n");
    let prediction_list_path = write_benchmark_file(&root_path, "predictions.list", b"22 predictions.loco\n");
    let output_root = root_path.join("output");
    let output_file_format = OutputFileFormat::parse(output_format).expect("benchmark output format should parse");
    let prepared_output_run =
        prepare_output_run(&output_root, "regenie2_linear", output_file_format, false).expect("run should prepare");
    let variant_count = i64::try_from(BENCHMARK_ROW_COUNT * BENCHMARK_CHUNK_COUNT)
        .expect("benchmark variant count should fit into int64");
    let chunk_size = i64::try_from(BENCHMARK_ROW_COUNT).expect("benchmark row count should fit into int64");
    let current_header_json = build_current_run_manifest_header_json(CurrentRunManifestHeaderInput {
        association_mode: "regenie2_linear".to_string(),
        association_backend_kind: "jax_dosage".to_string(),
        bgen_path,
        sample_path: None,
        phenotype_path,
        phenotype_name: "trait".to_string(),
        covariate_path: None,
        covariate_names: Vec::new(),
        prediction_list_path,
        prediction_loco_files_json: "[]".to_string(),
        sample_count: 100,
        variant_count,
        chunk_size,
        variant_limit: None,
        binary_correction_plan_method: "none".to_string(),
        binary_correction_plan_p_threshold: 0.0,
        binary_correction_plan_firth_se: false,
        trusted_no_missing_diploid: false,
        sample_key_mode: "fid_iid".to_string(),
        binary_kernel_config_json: None,
        bgen_decode_tile_variant_count: 64,
        trusted_bgen_validation_mode: "cache_on_miss".to_string(),
        jax_device: "cpu".to_string(),
        jax_enable_x64: false,
        jax_matmul_precision: None,
        requested_gpu_genotype_format: "dosage".to_string(),
        gpu_genotype_format: "dosage".to_string(),
        score_dtype: "float32".to_string(),
        firth_dtype: "float32".to_string(),
        multi_phenotype_sample_mode: "complete_case".to_string(),
        phenotype_compute_group_id: None,
        sample_set_fingerprint: None,
        covariate_design_fingerprint: None,
        prediction_alignment_fingerprint: None,
        output_format: output_format.to_string(),
        finalize_parquet,
        writer_thread_count: 1,
        writer_queue_depth: 16,
        chunks_per_arrow_file: i64::try_from(chunks_per_arrow_file)
            .expect("benchmark chunks per file should fit into int64"),
        arrow_compression: "none".to_string(),
        parquet_compression: "zstd".to_string(),
        output_statistic_dtype: "float32".to_string(),
    })
    .expect("benchmark manifest header should build");
    initialize_output_run(
        &prepared_output_run.output_run_paths.run_directory,
        &prepared_output_run.output_run_paths.chunks_directory,
        None,
        &current_header_json,
        false,
        OutputResumeMode::Fast,
    )
    .expect("benchmark run should initialize");
    BenchmarkRun {
        root_path,
        run_dir: prepared_output_run.output_run_paths.run_directory,
        chunks_dir: prepared_output_run.output_run_paths.chunks_directory,
    }
}

fn write_benchmark_file(root_path: &Path, file_name: &str, bytes: &[u8]) -> PathBuf {
    let file_path = root_path.join(file_name);
    std::fs::write(&file_path, bytes).expect("benchmark file should be written");
    file_path
}

fn build_writer_session(
    benchmark_run: &BenchmarkRun,
    output_format: &str,
    chunks_per_arrow_file: usize,
    arrow_compression: &str,
    parquet_compression: &str,
    finalize_parquet: bool,
) -> OutputWriterSession {
    OutputWriterSession::new(
        benchmark_run.run_dir.display().to_string(),
        benchmark_run.chunks_dir.display().to_string(),
        "regenie2_linear".to_string(),
        1,
        16,
        output_format,
        "float32",
        finalize_parquet,
        chunks_per_arrow_file,
        arrow_compression.to_string(),
        parquet_compression.to_string(),
        false,
    )
    .expect("benchmark writer session should start")
}

fn build_chunk_handle(chunk_index: usize) -> NativeChunkHandle {
    let row_count_i64 = i64::try_from(BENCHMARK_ROW_COUNT).expect("benchmark row count should fit into int64");
    let chunk_index_i64 = i64::try_from(chunk_index).expect("benchmark chunk index should fit into int64");
    let chunk_identifier =
        chunk_index_i64.checked_mul(row_count_i64).expect("benchmark chunk identifier should fit into int64");
    NativeChunkHandle::new(
        Arc::new(VariantMetadataColumns {
            chromosome: vec!["22".to_string(); BENCHMARK_ROW_COUNT],
            variant_identifier: (0..BENCHMARK_ROW_COUNT)
                .map(|row_index| format!("variant_{chunk_index}_{row_index}"))
                .collect(),
            position: (0..BENCHMARK_ROW_COUNT)
                .map(|row_index| {
                    chunk_identifier
                        .checked_add(i64::try_from(row_index).expect("benchmark row index should fit into int64"))
                        .expect("benchmark position should fit into int64")
                })
                .collect(),
            allele_one: vec!["A".to_string(); BENCHMARK_ROW_COUNT],
            allele_two: vec!["G".to_string(); BENCHMARK_ROW_COUNT],
        }),
        Arc::new(NativeChunkStats {
            allele_one_frequency: vec![0.5; BENCHMARK_ROW_COUNT],
            observation_count: vec![100; BENCHMARK_ROW_COUNT],
            has_missing_values: false,
            dosage_sum: vec![100.0; BENCHMARK_ROW_COUNT].into(),
            dosage_square_sum: vec![120.0; BENCHMARK_ROW_COUNT],
            imputed_dosage_square_sum: vec![120.0; BENCHMARK_ROW_COUNT],
            dosage_variance_numerator: vec![20.0; BENCHMARK_ROW_COUNT],
            info_score: vec![Some(0.98); BENCHMARK_ROW_COUNT],
            allele_count: vec![100.0; BENCHMARK_ROW_COUNT].into(),
            minor_allele_count: vec![100.0; BENCHMARK_ROW_COUNT],
            zero_count: vec![25; BENCHMARK_ROW_COUNT],
            nonzero_count: vec![75; BENCHMARK_ROW_COUNT],
            homozygous_reference_count: vec![25; BENCHMARK_ROW_COUNT],
            heterozygous_count: vec![50; BENCHMARK_ROW_COUNT],
            homozygous_alternate_count: vec![25; BENCHMARK_ROW_COUNT],
            is_sparse_candidate: vec![false; BENCHMARK_ROW_COUNT],
            is_rare_sparse_firth_candidate: vec![false; BENCHMARK_ROW_COUNT],
        }),
        chunk_identifier,
    )
}

fn build_result_array(value: f32) -> ArrayRef {
    Arc::new(Float32Array::from(vec![value; BENCHMARK_ROW_COUNT]))
}

fn enqueue_benchmark_chunks(writer_session: &OutputWriterSession) {
    for chunk_index in 0..BENCHMARK_CHUNK_COUNT {
        writer_session
            .write_regenie2_native_chunk_handle_arrays(
                build_chunk_handle(chunk_index),
                build_result_array(0.1),
                build_result_array(0.01),
                build_result_array(10.0),
                build_result_array(5.0),
                None,
            )
            .expect("benchmark chunk should enqueue");
    }
}

fn bench_enqueue_throughput(criterion: &mut Criterion) {
    criterion.bench_function("writer_enqueue_arrow_no_finish", |bencher| {
        bencher.iter_batched(
            || prepare_unmanifested_run("enqueue_arrow", "arrow"),
            |benchmark_run| {
                let writer_session = build_writer_session(&benchmark_run, "arrow", 1, "none", "zstd", false);
                enqueue_benchmark_chunks(&writer_session);
                writer_session.abort().expect("benchmark session should abort after enqueue");
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_grouped_arrow(criterion: &mut Criterion) {
    criterion.bench_function("writer_grouped_arrow_chunks", |bencher| {
        bencher.iter_batched(
            || prepare_manifested_run("grouped_arrow", "arrow", false, BENCHMARK_CHUNK_COUNT),
            |benchmark_run| {
                let writer_session =
                    build_writer_session(&benchmark_run, "arrow", BENCHMARK_CHUNK_COUNT, "none", "zstd", false);
                enqueue_benchmark_chunks(&writer_session);
                writer_session.finish().expect("benchmark session should finish");
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_arrow_compression(criterion: &mut Criterion) {
    criterion.bench_function("writer_arrow_zstd_chunks", |bencher| {
        bencher.iter_batched(
            || prepare_manifested_run("arrow_zstd", "arrow", false, BENCHMARK_CHUNK_COUNT),
            |benchmark_run| {
                let writer_session =
                    build_writer_session(&benchmark_run, "arrow", BENCHMARK_CHUNK_COUNT, "zstd", "zstd", false);
                enqueue_benchmark_chunks(&writer_session);
                writer_session.finish().expect("benchmark session should finish");
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_parquet_finalization(criterion: &mut Criterion) {
    criterion.bench_function("writer_parquet_finalization", |bencher| {
        bencher.iter_batched(
            || prepare_manifested_run("parquet_finalization", "parquet", true, BENCHMARK_CHUNK_COUNT),
            |benchmark_run| {
                let writer_session =
                    build_writer_session(&benchmark_run, "parquet", BENCHMARK_CHUNK_COUNT, "none", "zstd", true);
                enqueue_benchmark_chunks(&writer_session);
                writer_session.finish().expect("benchmark session should finish");
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_manifest_frequency(criterion: &mut Criterion) {
    criterion.bench_function("writer_manifest_many_chunk_commits", |bencher| {
        bencher.iter_batched(
            || prepare_manifested_run("manifest_frequency", "arrow", false, 1),
            |benchmark_run| {
                let writer_session = build_writer_session(&benchmark_run, "arrow", 1, "none", "zstd", false);
                enqueue_benchmark_chunks(&writer_session);
                writer_session.finish().expect("benchmark session should finish");
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_enqueue_throughput,
    bench_grouped_arrow,
    bench_arrow_compression,
    bench_parquet_finalization,
    bench_manifest_frequency
);
criterion_main!(benches);
