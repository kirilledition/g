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
    CurrentRunManifestHeaderInput, NativeChunkHandle, NativeVariantMetadataHandle, OutputManager, OutputWriterSession,
    Regenie2StatisticBatch, write_regenie2_multi_trait_chunk_f32,
};

const BENCHMARK_CHUNK_ROW_COUNT: usize = 8_192;
const BENCHMARK_CHUNK_COUNT: usize = 32;
const BENCHMARK_PHENOTYPE_NAME: &str = "binary_trait";
const BENCHMARK_WRITER_THREAD_COUNT: u32 = 4;

#[derive(Clone, Copy)]
enum CorrectionPattern {
    ScoreOnly,
    FirthMixed,
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
    output_manager: OutputManager,
    writer_sessions: Vec<Arc<OutputWriterSession>>,
    chunks: Vec<BenchmarkChunk>,
    benchmark_root: BenchmarkRoot,
}

struct CompletedBenchmarkRun {
    completed_outputs: Vec<g_output::CompletedOutputRun>,
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
            writer_thread_count: BENCHMARK_WRITER_THREAD_COUNT,
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
            candidate_capacity: 16_384,
            maximum_iterations: 250,
            gradient_tolerance: g_plan::PositiveF64::try_from(2.5e-4)
                .expect("benchmark gradient tolerance should be valid"),
            coefficient_tolerance: g_plan::PositiveF64::try_from(2.5e-4)
                .expect("benchmark coefficient tolerance should be valid"),
            likelihood_tolerance: g_plan::PositiveF64::try_from(2.5e-4)
                .expect("benchmark likelihood tolerance should be valid"),
            maximum_step_size: g_plan::PositiveF64::try_from(5.0).expect("benchmark maximum step should be valid"),
            pseudo_maximum_iterations: 50,
            pseudo_inner_maximum_iterations: 25,
            line_search_maximum_attempts: 25,
            step_halving_maximum_attempts: 12,
            initial_response_scale: g_plan::PositiveF64::try_from(4.863_891_244_002_886)
                .expect("benchmark response scale should be valid"),
            sparse_carrier_dosage_threshold: g_plan::DosageThreshold::try_from(1.0e-4)
                .expect("benchmark dosage threshold should be valid"),
            step_halving_scale: g_plan::StepScale::try_from(0.5).expect("benchmark step scale should be valid"),
            use_block_math: false,
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

        write!(&mut variant_identifier_text, "variant_{variant_index}")
            .expect("benchmark identifier should write to string");
        variant_identifier_offsets
            .push(u32::try_from(variant_identifier_text.len()).expect("benchmark identifier text should fit uint32"));
    }
    Arc::new(VariantMetadataStore::from_parts(
        text_dictionary,
        chromosome_codes,
        variant_identifier_text.into_boxed_str(),
        variant_identifier_offsets.into_boxed_slice(),
        position,
        allele_one_codes,
        allele_two_codes,
    ))
}

fn benchmark_chunk(
    metadata_store: &Arc<VariantMetadataStore>,
    chunk_index: usize,
    correction_pattern: CorrectionPattern,
) -> BenchmarkChunk {
    let variant_start_index =
        chunk_index.checked_mul(BENCHMARK_CHUNK_ROW_COUNT).expect("benchmark chunk start should not overflow");
    let variant_stop_index =
        variant_start_index.checked_add(BENCHMARK_CHUNK_ROW_COUNT).expect("benchmark chunk stop should not overflow");
    let metadata = VariantMetadataColumns::new(Arc::clone(metadata_store), variant_start_index..variant_stop_index);
    let metadata_handle = NativeVariantMetadataHandle::try_new(&metadata)
        .expect("benchmark native metadata handle should be constructed");
    let statistics = benchmark_chunk_statistics(variant_start_index);
    let chunk_identifier = i64::try_from(variant_start_index).expect("benchmark chunk identifier should fit int64");
    let chunk_handle = NativeChunkHandle::try_new(metadata_handle, statistics, chunk_identifier)
        .expect("benchmark native chunk should be constructed");
    BenchmarkChunk { chunk_handle, statistic_batch: benchmark_statistic_batch(variant_start_index, correction_pattern) }
}

fn benchmark_chunk_statistics(variant_start_index: usize) -> ChunkOutputStatistics {
    let mut allele_one_frequency = Vec::with_capacity(BENCHMARK_CHUNK_ROW_COUNT);
    let mut observation_count = Vec::with_capacity(BENCHMARK_CHUNK_ROW_COUNT);
    let mut info_score = NullableFloat32Column {
        values: Vec::with_capacity(BENCHMARK_CHUNK_ROW_COUNT),
        validity_bytes: Vec::with_capacity(BENCHMARK_CHUNK_ROW_COUNT.div_ceil(8)),
    };
    for row_index in 0..BENCHMARK_CHUNK_ROW_COUNT {
        let variant_index = variant_start_index + row_index;
        allele_one_frequency.push(
            0.01 + f32::from(
                u16::try_from(variant_index % 4_900).expect("benchmark frequency offset should fit uint16"),
            ) / 10_000.0,
        );
        observation_count.push(487_409 - i32::try_from(variant_index % 31).expect("missing count should fit int32"));
        info_score.push(
            0.8 + f32::from(u16::try_from(variant_index % 2_000).expect("benchmark INFO offset should fit uint16"))
                / 10_000.0,
            true,
        );
    }
    ChunkOutputStatistics { allele_one_frequency, observation_count, info_score }
}

fn benchmark_statistic_batch(
    variant_start_index: usize,
    correction_pattern: CorrectionPattern,
) -> Regenie2StatisticBatch {
    let mut beta = Vec::with_capacity(BENCHMARK_CHUNK_ROW_COUNT);
    let mut standard_error = Vec::with_capacity(BENCHMARK_CHUNK_ROW_COUNT);
    let mut chi_squared = Vec::with_capacity(BENCHMARK_CHUNK_ROW_COUNT);
    let mut log10_p_value = Vec::with_capacity(BENCHMARK_CHUNK_ROW_COUNT);
    let mut correction_code = match correction_pattern {
        CorrectionPattern::ScoreOnly => None,
        CorrectionPattern::FirthMixed => Some(Vec::with_capacity(BENCHMARK_CHUNK_ROW_COUNT)),
    };
    for row_index in 0..BENCHMARK_CHUNK_ROW_COUNT {
        let variant_index = variant_start_index + row_index;
        let beta_value =
            (f32::from(u16::try_from(variant_index % 401).expect("benchmark beta offset should fit uint16")) - 200.0)
                / 10_000.0;
        let standard_error_value = 0.01
            + f32::from(u16::try_from(variant_index % 101).expect("benchmark standard-error offset should fit uint16"))
                / 100_000.0;
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
        variant_count: BENCHMARK_CHUNK_ROW_COUNT,
        beta,
        standard_error,
        chi_squared,
        log10_p_value,
        correction_code,
    }
}

fn benchmark_correction_code(variant_index: usize) -> u8 {
    if variant_index.is_multiple_of(8_191) {
        3
    } else if variant_index.is_multiple_of(257) {
        2
    } else {
        u8::from(variant_index.is_multiple_of(4_093))
    }
}

fn prepare_benchmark_run(benchmark_name: &str, correction_pattern: CorrectionPattern) -> PreparedBenchmarkRun {
    let benchmark_root = unique_benchmark_root(benchmark_name);
    std::fs::create_dir_all(&benchmark_root.root_path).expect("benchmark root should be created");
    let bgen_path = write_benchmark_file(&benchmark_root.root_path, "input.bgen", b"benchmark bgen");
    let sample_path = write_benchmark_file(&benchmark_root.root_path, "input.sample", b"ID_1 ID_2\n0 0\n");
    let phenotype_path =
        write_benchmark_file(&benchmark_root.root_path, "phenotypes.tsv", b"FID\tIID\tbinary_trait\ns1\ts1\t1\n");
    let prediction_list_path =
        write_benchmark_file(&benchmark_root.root_path, "predictions.list", b"22 predictions.loco\n");
    let output_root = benchmark_root.root_path.join("output");
    let variant_count = BENCHMARK_CHUNK_ROW_COUNT
        .checked_mul(BENCHMARK_CHUNK_COUNT)
        .expect("benchmark variant count should not overflow");
    let run_plan =
        Arc::new(benchmark_run_plan(&output_root, &bgen_path, &sample_path, &phenotype_path, &prediction_list_path));
    let mut output_manager = OutputManager::open(run_plan, "# benchmark configuration\n".to_string())
        .expect("benchmark output manager should open");
    let planned_chunk_ranges = (0..BENCHMARK_CHUNK_COUNT)
        .map(|chunk_index| {
            let variant_start_index = chunk_index * BENCHMARK_CHUNK_ROW_COUNT;
            variant_start_index..variant_start_index + BENCHMARK_CHUNK_ROW_COUNT
        })
        .collect::<Vec<Range<usize>>>();
    output_manager
        .initialize(vec![benchmark_manifest_header(&bgen_path, variant_count)], &planned_chunk_ranges, false)
        .expect("benchmark output manager should initialize");
    let writer_sessions = output_manager
        .delivery_state_for_phenotypes(&[BENCHMARK_PHENOTYPE_NAME.to_string()])
        .expect("benchmark output delivery state should be available")
        .writer_sessions;
    let metadata_store = benchmark_metadata_store(variant_count);
    let chunks = (0..BENCHMARK_CHUNK_COUNT)
        .map(|chunk_index| benchmark_chunk(&metadata_store, chunk_index, correction_pattern))
        .collect();
    PreparedBenchmarkRun { output_manager, writer_sessions, chunks, benchmark_root }
}

fn write_and_finish_benchmark_run(prepared_run: PreparedBenchmarkRun) -> CompletedBenchmarkRun {
    let PreparedBenchmarkRun { output_manager, writer_sessions, chunks, benchmark_root } = prepared_run;
    for benchmark_chunk in chunks {
        write_regenie2_multi_trait_chunk_f32(
            &writer_sessions,
            None,
            &benchmark_chunk.chunk_handle,
            benchmark_chunk.statistic_batch,
        )
        .expect("benchmark chunk should enqueue");
    }
    drop(writer_sessions);
    let completed_outputs = output_manager.finish().expect("benchmark output manager should finish");
    CompletedBenchmarkRun { completed_outputs, benchmark_root }
}

fn bench_binary_parquet_writer(criterion: &mut Criterion) {
    let mut benchmark_group = criterion.benchmark_group("binary_parquet_writer");
    let total_row_count = BENCHMARK_CHUNK_ROW_COUNT * BENCHMARK_CHUNK_COUNT;
    benchmark_group.throughput(Throughput::Elements(
        u64::try_from(total_row_count).expect("benchmark row count should fit uint64"),
    ));
    benchmark_group.bench_function("score_only_32x8192", |bencher| {
        bencher.iter_batched(
            || prepare_benchmark_run("score_only", CorrectionPattern::ScoreOnly),
            |prepared_run| {
                let completed_run = write_and_finish_benchmark_run(prepared_run);
                std::hint::black_box(&completed_run.completed_outputs);
                std::hint::black_box(&completed_run.benchmark_root);
                completed_run
            },
            BatchSize::PerIteration,
        );
    });
    benchmark_group.bench_function("firth_mixed_32x8192", |bencher| {
        bencher.iter_batched(
            || prepare_benchmark_run("firth_mixed", CorrectionPattern::FirthMixed),
            |prepared_run| {
                let completed_run = write_and_finish_benchmark_run(prepared_run);
                std::hint::black_box(&completed_run.completed_outputs);
                std::hint::black_box(&completed_run.benchmark_root);
                completed_run
            },
            BatchSize::PerIteration,
        );
    });
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
