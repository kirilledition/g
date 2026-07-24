use std::ops::Range;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use g_genotype_contracts::{
    BgenSourceIdentity, ChunkOutputStatistics, NullableFloat32Column, VariantMetadataColumns, VariantMetadataStore,
};
use g_output::{
    Active, CurrentRunManifestHeaderInput, NativeChunkHandle, NativeVariantMetadataHandle, OutputDeliveryToken,
    OutputManager, Regenie2StatisticBatch, write_regenie2_multi_trait_chunk_f32,
};

const BENCHMARK_CHUNK_ROW_COUNT: usize = 16_384;
const BENCHMARK_CHUNK_COUNT: usize = 26;
const BENCHMARK_TAIL_ROW_COUNT: usize = 9_343;
const BENCHMARK_TOTAL_ROW_COUNT: usize =
    BENCHMARK_CHUNK_ROW_COUNT * (BENCHMARK_CHUNK_COUNT - 1) + BENCHMARK_TAIL_ROW_COUNT;
const BENCHMARK_FIRTH_SUCCESS_COUNT: usize = 17_938;
const BENCHMARK_PHENOTYPE_NAME: &str = "binary_trait";
const BENCHMARK_WRITER_THREAD_COUNTS: [u32; 3] = [1, 4, 8];
const BENCHMARK_PACED_CHUNK_INTERVAL: Duration = Duration::from_millis(12);

#[derive(Clone, Copy)]
enum CorrectionPattern {
    ScoreOnly,
    FirthSuccesses,
}

struct BenchmarkRoot {
    root_path: PathBuf,
}

impl Drop for BenchmarkRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root_path);
    }
}

struct BenchmarkChunk {
    chunk_handle: NativeChunkHandle,
    statistic_batch: Regenie2StatisticBatch,
}

struct PreparedBenchmarkRun {
    output_manager: OutputManager<Active>,
    delivery_token: OutputDeliveryToken,
    chunks: Vec<BenchmarkChunk>,
    benchmark_root: BenchmarkRoot,
}

struct CompletedBenchmarkRun {
    completed_outputs: Vec<g_output::CompletedOutputRun>,
    benchmark_root: BenchmarkRoot,
}

struct BenchmarkDurabilityStageMetrics {
    file_sync: f64,
    file_hash: f64,
    file_publish: f64,
    directory_sync: f64,
    receipt_publish: f64,
    writer_total: f64,
}

struct BenchmarkOutputMetrics {
    parquet_file_bytes: u64,
    durability: BenchmarkDurabilityStageMetrics,
}

struct SubmittedBenchmarkRun {
    output_manager: OutputManager<Active>,
    delivery_token: OutputDeliveryToken,
    benchmark_root: BenchmarkRoot,
}

fn unique_benchmark_root(benchmark_name: &str) -> BenchmarkRoot {
    let unique_suffix =
        SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after Unix epoch").as_nanos();
    BenchmarkRoot {
        root_path: std::env::temp_dir()
            .join(format!("g-output-writer-bench-{}-{unique_suffix}", benchmark_name.replace('/', "_"))),
    }
}

fn write_benchmark_file(root_path: &Path, file_name: &str, bytes: &[u8]) -> PathBuf {
    let file_path = root_path.join(file_name);
    std::fs::write(&file_path, bytes).expect("benchmark input file should be written");
    file_path
}

fn benchmark_run_plan(
    output_root: &Path,
    bgen_path: &Path,
    sample_path: &Path,
    phenotype_path: &Path,
    prediction_list_path: &Path,
    writer_thread_count: u32,
) -> g_plan::RunPlan {
    g_plan::RunPlan {
        association_mode: g_plan::AssociationMode::Regenie2Binary,
        chunk_size: u32::try_from(BENCHMARK_CHUNK_ROW_COUNT).expect("benchmark row count should fit uint32"),
        input: g_plan::InputPlan {
            bgen_path: bgen_path.display().to_string(),
            sample_path: sample_path.display().to_string(),
            phenotype_path: phenotype_path.display().to_string(),
            prediction_list_path: prediction_list_path.display().to_string(),
            covariate_path: None,
            covariate_names: Vec::new(),
        },
        compute: g_plan::ComputePlan {
            device: g_plan::Device::Gpu,
            cpu_thread_count: None,
            jax_cache_directory: None,
            multi_phenotype_sample_mode: g_plan::MultiPhenotypeSampleMode::CompleteCase,
            kernels: benchmark_kernel_plan(),
        },
        correction: g_plan::CorrectionPlan {
            method: g_plan::BinaryFallbackMethod::FirthApproximate,
            p_threshold: g_plan::Probability::try_from(0.05).expect("benchmark probability should be valid"),
            firth_se: false,
        },
        output: g_plan::OutputPlan {
            output_run_root: output_root.display().to_string(),
            resume: false,
            recover_attempt: None,
            fenced_owner_claim_id: None,
            writer_thread_count,
        },
        telemetry: g_plan::TelemetryMode::Off,
        phenotype_runs: vec![g_plan::PhenotypeRunPlan {
            phenotype_name: BENCHMARK_PHENOTYPE_NAME.to_string(),
            output_directory_name: "phenotype_0001_binary_trait".to_string(),
        }],
    }
}

fn benchmark_kernel_plan() -> g_plan::KernelPlan {
    g_plan::KernelPlan {
        linear: g_plan::LinearKernelPlan {
            minimum_variance: g_plan::PositiveF32::try_from(1.0e-8).expect("benchmark variance should be valid"),
            relative_variance_tolerance: g_plan::PositiveF32::try_from(1.0e-6)
                .expect("benchmark variance tolerance should be valid"),
        },
        binary_null: g_plan::BinaryNullKernelPlan {
            maximum_iterations: 50,
            coefficient_tolerance: g_plan::PositiveF32::try_from(1.0e-6)
                .expect("benchmark coefficient tolerance should be valid"),
            nonconvergence_policy: g_plan::NullLogisticNonconvergencePolicy::Fail,
            minimum_probability: g_plan::ProbabilityFloor::try_from(1.0e-6)
                .expect("benchmark probability floor should be valid"),
            minimum_variance: g_plan::PositiveF32::try_from(1.0e-8).expect("benchmark variance should be valid"),
            relative_variance_tolerance: g_plan::PositiveF32::try_from(1.0e-6)
                .expect("benchmark variance tolerance should be valid"),
        },
        firth: g_plan::FirthKernelPlan {
            batch_size: 512,
            candidate_capacity: 1_024,
            maximum_iterations: 250,
            gradient_tolerance: g_plan::PositiveF64::try_from(2.5e-4)
                .expect("benchmark gradient tolerance should be valid"),
            maximum_step_size: g_plan::PositiveF64::try_from(5.0).expect("benchmark maximum step should be valid"),
            pseudo_maximum_iterations: 50,
            pseudo_inner_maximum_iterations: 25,
            line_search_maximum_attempts: 25,
            sparse_carrier_dosage_threshold: g_plan::DosageThreshold::try_from(1.0e-4)
                .expect("benchmark dosage threshold should be valid"),
        },
        null_firth: g_plan::NullFirthKernelPlan {
            maximum_iterations: 1_000,
            gradient_tolerance: g_plan::PositiveF64::try_from(50.0e-6)
                .expect("benchmark gradient tolerance should be valid"),
            maximum_step_size: g_plan::PositiveF64::try_from(25.0).expect("benchmark maximum step should be valid"),
            fallback_iteration_multiplier: 5,
            fallback_step_divisor: g_plan::PositiveF64::try_from(5.0).expect("benchmark step divisor should be valid"),
            line_search_maximum_attempts: 25,
            step_halving_scale: g_plan::StepScale::try_from(0.5).expect("benchmark step scale should be valid"),
        },
    }
}

fn benchmark_bgen_source_identity(bgen_path: &Path) -> Arc<BgenSourceIdentity> {
    let canonical_path = bgen_path.canonicalize().expect("benchmark BGEN path should canonicalize");
    let metadata = canonical_path.metadata().expect("benchmark BGEN metadata should be available");
    Arc::new(BgenSourceIdentity {
        configured_path: bgen_path.to_path_buf(),
        canonical_path: Some(canonical_path),
        device_identifier: metadata.dev(),
        inode_identifier: metadata.ino(),
        change_time_nanoseconds: timestamp_nanoseconds(metadata.ctime(), metadata.ctime_nsec()),
        modification_time_nanoseconds: timestamp_nanoseconds(metadata.mtime(), metadata.mtime_nsec()),
        file_size: metadata.len(),
    })
}

fn timestamp_nanoseconds(seconds: i64, subsecond_nanoseconds: i64) -> i64 {
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|nanoseconds| nanoseconds.checked_add(subsecond_nanoseconds))
        .expect("benchmark timestamp should fit int64")
}

fn benchmark_manifest_header(bgen_path: &Path, variant_count: usize) -> CurrentRunManifestHeaderInput {
    CurrentRunManifestHeaderInput {
        phenotype_name: BENCHMARK_PHENOTYPE_NAME.to_string(),
        bgen_source_identity: benchmark_bgen_source_identity(bgen_path),
        covariate_names: Arc::from(Vec::<String>::new()),
        prediction_loco_files: Arc::from(Vec::new()),
        sample_count: 487_409,
        variant_count,
        resolved_gpu_genotype_format: g_plan::GpuGenotypeFormat::Packed8,
        sample_mode: g_plan::MultiPhenotypeSampleMode::CompleteCase,
        phenotype_compute_group_id: Arc::from("benchmark_group"),
        sample_set_fingerprint: Arc::from("benchmark_samples"),
        covariate_design_fingerprint: Arc::from("benchmark_covariates"),
        phenotype_design_fingerprint: Arc::from("benchmark_phenotype"),
        prediction_alignment_fingerprint: Arc::from("benchmark_predictions"),
    }
}

fn benchmark_metadata_store(variant_count: usize) -> Arc<VariantMetadataStore> {
    let text_dictionary: Box<[Arc<str>]> = ["22", "A", "C", "G", "T"].map(Arc::<str>::from).into();
    let chromosome_codes = vec![0_u32; variant_count].into_boxed_slice();
    let allele_one_codes = (0..variant_count)
        .map(|variant_index| 1_u32 + u32::try_from(variant_index % 4).expect("allele code should fit uint32"))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let allele_two_codes = (0..variant_count)
        .map(|variant_index| 1_u32 + u32::try_from((variant_index + 1) % 4).expect("allele code should fit uint32"))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let position = (0..variant_count)
        .map(|variant_index| {
            1_000_000_i64
                .checked_add(i64::try_from(variant_index).expect("benchmark position should fit int64"))
                .expect("benchmark position should not overflow")
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let mut variant_identifier_text = String::with_capacity(variant_count.saturating_mul(16));
    let mut variant_identifier_offsets = Vec::with_capacity(variant_count.saturating_add(1));
    variant_identifier_offsets.push(0_u32);
    for variant_index in 0..variant_count {
        use std::fmt::Write as _;

        let identifier_number = benchmark_mixed_value(variant_index, 0x69d2_5f85_2536_5e21) % 900_000_000;
        write!(&mut variant_identifier_text, "rs{:09}", identifier_number + 100_000_000)
            .expect("benchmark identifier should write to string");
        variant_identifier_offsets
            .push(u32::try_from(variant_identifier_text.len()).expect("benchmark identifier text should fit uint32"));
    }
    Arc::new(
        VariantMetadataStore::from_parts(
            text_dictionary,
            chromosome_codes,
            variant_identifier_text.into_boxed_str(),
            variant_identifier_offsets.into_boxed_slice(),
            position,
            allele_one_codes,
            allele_two_codes,
        )
        .expect("benchmark metadata store should satisfy its invariants"),
    )
}

fn benchmark_chunk(
    metadata_store: &Arc<VariantMetadataStore>,
    chunk_index: usize,
    correction_pattern: CorrectionPattern,
) -> BenchmarkChunk {
    let chunk_range = benchmark_chunk_range(chunk_index);
    let variant_start_index = chunk_range.start;
    let row_count = chunk_range.len();
    let metadata = VariantMetadataColumns::new(Arc::clone(metadata_store), chunk_range)
        .expect("benchmark metadata range should be valid");
    let metadata_handle = NativeVariantMetadataHandle::try_new(&metadata)
        .expect("benchmark native metadata handle should be constructed");
    let statistics = benchmark_chunk_statistics(variant_start_index, row_count);
    let chunk_identifier = i64::try_from(variant_start_index).expect("benchmark chunk identifier should fit int64");
    let chunk_handle = NativeChunkHandle::try_new(metadata_handle, statistics, chunk_identifier)
        .expect("benchmark native chunk should be constructed");
    BenchmarkChunk {
        chunk_handle,
        statistic_batch: benchmark_statistic_batch(variant_start_index, row_count, correction_pattern),
    }
}

fn benchmark_chunk_range(chunk_index: usize) -> Range<usize> {
    let variant_start_index =
        chunk_index.checked_mul(BENCHMARK_CHUNK_ROW_COUNT).expect("benchmark chunk start should not overflow");
    let variant_stop_index = variant_start_index
        .checked_add(BENCHMARK_CHUNK_ROW_COUNT)
        .expect("benchmark chunk stop should not overflow")
        .min(BENCHMARK_TOTAL_ROW_COUNT);
    variant_start_index..variant_stop_index
}

fn benchmark_chunk_statistics(variant_start_index: usize, row_count: usize) -> ChunkOutputStatistics {
    let mut allele_one_frequency = Vec::with_capacity(row_count);
    let mut observation_count = Vec::with_capacity(row_count);
    let mut info_score = NullableFloat32Column {
        values: Vec::with_capacity(row_count),
        validity_bytes: Vec::with_capacity(row_count.div_ceil(8)),
    };
    for row_index in 0..row_count {
        let variant_index = variant_start_index + row_index;
        allele_one_frequency.push(0.001 + 0.499 * benchmark_unit_interval_value(variant_index, 0x9dca_8e31_79f5_8a1b));
        observation_count.push(487_409 - i32::try_from(variant_index % 31).expect("missing count should fit int32"));
        info_score.push(0.8 + 0.2 * benchmark_unit_interval_value(variant_index, 0xf3a6_7c41_8e29_b5d0), true);
    }
    ChunkOutputStatistics { allele_one_frequency, observation_count, info_score }
}

fn benchmark_statistic_batch(
    variant_start_index: usize,
    row_count: usize,
    correction_pattern: CorrectionPattern,
) -> Regenie2StatisticBatch {
    let mut beta = Vec::with_capacity(row_count);
    let mut standard_error = Vec::with_capacity(row_count);
    let mut chi_squared = Vec::with_capacity(row_count);
    let mut log10_p_value = Vec::with_capacity(row_count);
    let mut correction_code = match correction_pattern {
        CorrectionPattern::ScoreOnly => None,
        CorrectionPattern::FirthSuccesses => Some(Vec::with_capacity(row_count)),
    };
    for row_index in 0..row_count {
        let variant_index = variant_start_index + row_index;
        let beta_value = (benchmark_unit_interval_value(variant_index, 0x42b7_1ce9_a85f_630d) - 0.5) * 0.1;
        let standard_error_value = 0.005 + 0.045 * benchmark_unit_interval_value(variant_index, 0x7e14_d3a9_5b62_c8f0);
        let chi_squared_value = (beta_value / standard_error_value).powi(2);
        beta.push(beta_value);
        standard_error.push(standard_error_value);
        chi_squared.push(chi_squared_value);
        log10_p_value.push(chi_squared_value * 0.217_147_25);
        if let Some(correction_values) = correction_code.as_mut() {
            correction_values.push(benchmark_correction_code(variant_index));
        }
    }
    Regenie2StatisticBatch {
        trait_count: 1,
        variant_count: row_count,
        beta,
        standard_error,
        chi_squared,
        log10_p_value,
        correction_code,
    }
}

fn benchmark_correction_code(variant_index: usize) -> u8 {
    let success_count_before = variant_index
        .checked_mul(BENCHMARK_FIRTH_SUCCESS_COUNT)
        .expect("benchmark Firth-success distribution should not overflow")
        / BENCHMARK_TOTAL_ROW_COUNT;
    let success_count_through = variant_index
        .checked_add(1)
        .and_then(|variant_count| variant_count.checked_mul(BENCHMARK_FIRTH_SUCCESS_COUNT))
        .expect("benchmark Firth-success distribution should not overflow")
        / BENCHMARK_TOTAL_ROW_COUNT;
    if success_count_through > success_count_before { 2 } else { 0 }
}

fn benchmark_mixed_value(variant_index: usize, salt: u64) -> u64 {
    let mut value = u64::try_from(variant_index).expect("benchmark variant index should fit uint64") ^ salt;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn benchmark_unit_interval_value(variant_index: usize, salt: u64) -> f32 {
    let mixed_value = benchmark_mixed_value(variant_index, salt);
    let fraction_bits = mixed_value & 0x00ff_ffff;
    let high_fraction_bits = u16::try_from(fraction_bits >> 8).expect("high benchmark fraction bits should fit uint16");
    let low_fraction_bits =
        u8::try_from(fraction_bits & u64::from(u8::MAX)).expect("low benchmark fraction bits should fit uint8");
    f32::from(high_fraction_bits) * (1.0 / 65_536.0) + f32::from(low_fraction_bits) * (1.0 / 16_777_216.0)
}

fn prepare_benchmark_run(
    benchmark_name: &str,
    correction_pattern: CorrectionPattern,
    writer_thread_count: u32,
    collect_stage_timings: bool,
) -> PreparedBenchmarkRun {
    let benchmark_root = unique_benchmark_root(benchmark_name);
    std::fs::create_dir_all(&benchmark_root.root_path).expect("benchmark root should be created");
    let bgen_path = write_benchmark_file(&benchmark_root.root_path, "input.bgen", b"benchmark bgen");
    let sample_path = write_benchmark_file(&benchmark_root.root_path, "input.sample", b"ID_1 ID_2\n0 0\n");
    let phenotype_path =
        write_benchmark_file(&benchmark_root.root_path, "phenotypes.tsv", b"FID\tIID\tbinary_trait\ns1\ts1\t1\n");
    let prediction_list_path =
        write_benchmark_file(&benchmark_root.root_path, "predictions.list", b"22 predictions.loco\n");
    let output_root = benchmark_root.root_path.join("output");
    let run_plan = Arc::new(benchmark_run_plan(
        &output_root,
        &bgen_path,
        &sample_path,
        &phenotype_path,
        &prediction_list_path,
        writer_thread_count,
    ));
    let output_manager = OutputManager::open(run_plan, "# benchmark configuration\n".to_string())
        .expect("benchmark output manager should open");
    let planned_chunk_ranges = (0..BENCHMARK_CHUNK_COUNT).map(benchmark_chunk_range).collect::<Vec<Range<usize>>>();
    let output_manager = output_manager
        .initialize(
            vec![benchmark_manifest_header(&bgen_path, BENCHMARK_TOTAL_ROW_COUNT)],
            &planned_chunk_ranges,
            collect_stage_timings,
        )
        .expect("benchmark output manager should initialize");
    let delivery_token = output_manager
        .delivery_token_for_phenotypes(&[BENCHMARK_PHENOTYPE_NAME.to_string()])
        .expect("benchmark output delivery token should be available");
    let metadata_store = benchmark_metadata_store(BENCHMARK_TOTAL_ROW_COUNT);
    let chunks = (0..BENCHMARK_CHUNK_COUNT)
        .map(|chunk_index| benchmark_chunk(&metadata_store, chunk_index, correction_pattern))
        .collect();
    PreparedBenchmarkRun { output_manager, delivery_token, chunks, benchmark_root }
}

fn submit_benchmark_run(prepared_run: PreparedBenchmarkRun, chunk_interval: Option<Duration>) -> SubmittedBenchmarkRun {
    let PreparedBenchmarkRun { output_manager, delivery_token, chunks, benchmark_root } = prepared_run;
    let chunk_count = chunks.len();
    for (chunk_index, benchmark_chunk) in chunks.into_iter().enumerate() {
        write_regenie2_multi_trait_chunk_f32(
            &delivery_token,
            None,
            &benchmark_chunk.chunk_handle,
            benchmark_chunk.statistic_batch,
        )
        .expect("benchmark chunk should enqueue");
        if chunk_index + 1 < chunk_count
            && let Some(interval) = chunk_interval
        {
            std::thread::sleep(interval);
        }
    }
    SubmittedBenchmarkRun { output_manager, delivery_token, benchmark_root }
}

fn finish_benchmark_run(submitted_run: SubmittedBenchmarkRun) -> CompletedBenchmarkRun {
    let SubmittedBenchmarkRun { output_manager, delivery_token, benchmark_root } = submitted_run;
    drop(delivery_token);
    let completed_outputs = output_manager
        .close_completed()
        .expect("benchmark output should have exact coverage")
        .finish()
        .expect("benchmark output manager should finish");
    CompletedBenchmarkRun { completed_outputs, benchmark_root }
}

fn measure_benchmark_output_metrics(
    benchmark_name: &str,
    correction_pattern: CorrectionPattern,
) -> BenchmarkOutputMetrics {
    let completed_run = finish_benchmark_run(submit_benchmark_run(
        prepare_benchmark_run(benchmark_name, correction_pattern, 8, true),
        None,
    ));
    let parquet_file_bytes = completed_run
        .completed_outputs
        .iter()
        .flat_map(|completed_output| {
            std::fs::read_dir(&completed_output.parts_directory)
                .expect("benchmark parts directory should be readable")
                .map(|entry| entry.expect("benchmark part entry should be readable"))
        })
        .filter(|entry| entry.path().extension().is_some_and(|extension| extension == "parquet"))
        .map(|entry| entry.metadata().expect("benchmark part metadata should be readable").len())
        .sum();
    let run_directory =
        &completed_run.completed_outputs.first().expect("benchmark should complete one output").run_directory;
    let timing_text = std::fs::read_to_string(run_directory.join("output_stage_timings.json"))
        .expect("benchmark stage timing should be readable");
    let timing_value: serde_json::Value =
        serde_json::from_str(&timing_text).expect("benchmark stage timing should be valid JSON");
    BenchmarkOutputMetrics {
        parquet_file_bytes,
        durability: BenchmarkDurabilityStageMetrics {
            file_sync: benchmark_stage_seconds(&timing_value, "rust_output_writer_parquet_file_sync"),
            file_hash: benchmark_stage_seconds(&timing_value, "rust_output_writer_parquet_file_hash"),
            file_publish: benchmark_stage_seconds(&timing_value, "rust_output_writer_parquet_file_publish"),
            directory_sync: benchmark_stage_seconds(&timing_value, "rust_output_writer_parquet_directory_sync"),
            receipt_publish: benchmark_stage_seconds(&timing_value, "rust_output_writer_receipt_publish"),
            writer_total: benchmark_stage_seconds(&timing_value, "rust_output_writer_total"),
        },
    }
}

fn benchmark_stage_seconds(timing_value: &serde_json::Value, stage_name: &str) -> f64 {
    timing_value["stage_totals_seconds"][stage_name]
        .as_f64()
        .unwrap_or_else(|| panic!("benchmark stage '{stage_name}' should be a floating-point number"))
}

fn bench_binary_parquet_writer(criterion: &mut Criterion) {
    let mut benchmark_group = criterion.benchmark_group("binary_parquet_writer");
    let score_only_metrics = measure_benchmark_output_metrics("score_only_size", CorrectionPattern::ScoreOnly);
    let firth_success_metrics =
        measure_benchmark_output_metrics("firth_success_size", CorrectionPattern::FirthSuccesses);
    let observed_firth_success_count =
        (0..BENCHMARK_TOTAL_ROW_COUNT).filter(|variant_index| benchmark_correction_code(*variant_index) == 2).count();
    assert_eq!(observed_firth_success_count, BENCHMARK_FIRTH_SUCCESS_COUNT);
    eprintln!(
        "binary_parquet_writer rows={BENCHMARK_TOTAL_ROW_COUNT} tail_rows={BENCHMARK_TAIL_ROW_COUNT} firth_successes={observed_firth_success_count} firth_failures=0 score_only_bytes={} firth_success_bytes={} score_only_file_sync_seconds={} score_only_file_hash_seconds={} score_only_file_publish_seconds={} score_only_directory_sync_seconds={} score_only_receipt_publish_seconds={} score_only_writer_total_seconds={} firth_file_sync_seconds={} firth_file_hash_seconds={} firth_file_publish_seconds={} firth_directory_sync_seconds={} firth_receipt_publish_seconds={} firth_writer_total_seconds={}",
        score_only_metrics.parquet_file_bytes,
        firth_success_metrics.parquet_file_bytes,
        score_only_metrics.durability.file_sync,
        score_only_metrics.durability.file_hash,
        score_only_metrics.durability.file_publish,
        score_only_metrics.durability.directory_sync,
        score_only_metrics.durability.receipt_publish,
        score_only_metrics.durability.writer_total,
        firth_success_metrics.durability.file_sync,
        firth_success_metrics.durability.file_hash,
        firth_success_metrics.durability.file_publish,
        firth_success_metrics.durability.directory_sync,
        firth_success_metrics.durability.receipt_publish,
        firth_success_metrics.durability.writer_total,
    );
    benchmark_group.throughput(Throughput::Elements(
        u64::try_from(BENCHMARK_TOTAL_ROW_COUNT).expect("benchmark row count should fit uint64"),
    ));
    for writer_thread_count in BENCHMARK_WRITER_THREAD_COUNTS {
        benchmark_group.bench_function(format!("score_only_chr22/writers_{writer_thread_count}"), |bencher| {
            bencher.iter_batched(
                || prepare_benchmark_run("score_only", CorrectionPattern::ScoreOnly, writer_thread_count, false),
                |prepared_run| {
                    let completed_run = finish_benchmark_run(submit_benchmark_run(prepared_run, None));
                    std::hint::black_box(&completed_run.completed_outputs);
                    std::hint::black_box(&completed_run.benchmark_root);
                    completed_run
                },
                BatchSize::PerIteration,
            );
        });
        benchmark_group.bench_function(format!("firth_success_chr22/writers_{writer_thread_count}"), |bencher| {
            bencher.iter_batched(
                || {
                    prepare_benchmark_run(
                        "firth_success",
                        CorrectionPattern::FirthSuccesses,
                        writer_thread_count,
                        false,
                    )
                },
                |prepared_run| {
                    let completed_run = finish_benchmark_run(submit_benchmark_run(prepared_run, None));
                    std::hint::black_box(&completed_run.completed_outputs);
                    std::hint::black_box(&completed_run.benchmark_root);
                    completed_run
                },
                BatchSize::PerIteration,
            );
        });
        benchmark_group.bench_function(
            format!("firth_success_chr22_paced_finish/writers_{writer_thread_count}"),
            |bencher| {
                bencher.iter_batched(
                    || {
                        submit_benchmark_run(
                            prepare_benchmark_run(
                                "firth_success_paced",
                                CorrectionPattern::FirthSuccesses,
                                writer_thread_count,
                                false,
                            ),
                            Some(BENCHMARK_PACED_CHUNK_INTERVAL),
                        )
                    },
                    |submitted_run| {
                        let completed_run = finish_benchmark_run(submitted_run);
                        std::hint::black_box(&completed_run.completed_outputs);
                        std::hint::black_box(&completed_run.benchmark_root);
                        completed_run
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }
    benchmark_group.finish();
}

fn criterion_configuration() -> Criterion {
    Criterion::default().warm_up_time(Duration::from_secs(3)).measurement_time(Duration::from_secs(12)).sample_size(30)
}

criterion_group! {
    name = benches;
    config = criterion_configuration();
    targets = bench_binary_parquet_writer
}
criterion_main!(benches);
