use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{Array, Float32Array, Int32Array, Int64Array, StringArray};
use g_genotype_contracts::{
    BgenSourceIdentity, ChunkOutputStatistics, NullableFloat32Column, VariantMetadataColumns, VariantMetadataStore,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;

use crate::{
    CurrentRunManifestHeaderInput, NativeChunkHandle, NativeVariantMetadataHandle, OutputManager, OutputWriterSession,
    Regenie2StatisticBatch, write_regenie2_multi_trait_chunk_f32,
};

const PRIMARY_PHENOTYPE: &str = "trait_alpha";

struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn new(label: &str) -> Self {
        static DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);
        let sequence = DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).expect("test time is after Unix epoch").as_nanos();
        let path = std::env::temp_dir().join(format!("g-output-{label}-{}-{timestamp}-{sequence}", std::process::id()));
        std::fs::create_dir_all(&path).expect("test directory is created");
        Self { path }
    }

    fn write(&self, name: &str, contents: &[u8]) -> PathBuf {
        let path = self.path.join(name);
        std::fs::write(&path, contents).expect("test fixture is written");
        path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

struct TestInputs {
    bgen: PathBuf,
    sample: PathBuf,
    phenotype: PathBuf,
    prediction_list: PathBuf,
}

struct TestChunk {
    handle: NativeChunkHandle,
    statistics: Regenie2StatisticBatch,
}

fn test_inputs(directory: &TestDirectory, phenotype_names: &[&str]) -> TestInputs {
    let phenotype_header = phenotype_names.join("\t");
    TestInputs {
        bgen: directory.write("input.bgen", b"test bgen identity"),
        sample: directory.write("input.sample", b"ID_1 ID_2\n0 0\nfamily sample\n"),
        phenotype: directory
            .write("phenotypes.tsv", format!("FID\tIID\t{phenotype_header}\nfamily\tsample\t1\n").as_bytes()),
        prediction_list: directory.write("predictions.list", b"trait_alpha predictions.loco\n"),
    }
}

fn kernel_plan() -> g_plan::KernelPlan {
    g_plan::KernelPlan {
        linear: g_plan::LinearKernelPlan {
            minimum_variance: g_plan::PositiveF32::try_from(1.0e-8).expect("valid minimum variance"),
            relative_variance_tolerance: g_plan::PositiveF32::try_from(1.0e-6).expect("valid relative tolerance"),
        },
        binary_null: g_plan::BinaryNullKernelPlan {
            maximum_iterations: 50,
            coefficient_tolerance: g_plan::PositiveF32::try_from(1.0e-6).expect("valid coefficient tolerance"),
            nonconvergence_policy: g_plan::NullLogisticNonconvergencePolicy::Fail,
            minimum_probability: g_plan::ProbabilityFloor::try_from(1.0e-6).expect("valid probability floor"),
            minimum_variance: g_plan::PositiveF32::try_from(1.0e-8).expect("valid minimum variance"),
            relative_variance_tolerance: g_plan::PositiveF32::try_from(1.0e-6).expect("valid relative tolerance"),
        },
        firth: g_plan::FirthKernelPlan {
            batch_size: 2,
            candidate_capacity: 4,
            maximum_iterations: 25,
            gradient_tolerance: g_plan::PositiveF64::try_from(1.0e-5).expect("valid gradient tolerance"),
            maximum_step_size: g_plan::PositiveF64::try_from(5.0).expect("valid maximum step"),
            pseudo_maximum_iterations: 10,
            pseudo_inner_maximum_iterations: 5,
            line_search_maximum_attempts: 5,
            sparse_carrier_dosage_threshold: g_plan::DosageThreshold::try_from(0.1).expect("valid dosage threshold"),
        },
        null_firth: g_plan::NullFirthKernelPlan {
            maximum_iterations: 50,
            gradient_tolerance: g_plan::PositiveF64::try_from(1.0e-5).expect("valid gradient tolerance"),
            maximum_step_size: g_plan::PositiveF64::try_from(10.0).expect("valid maximum step"),
            fallback_iteration_multiplier: 2,
            fallback_step_divisor: g_plan::PositiveF64::try_from(2.0).expect("valid step divisor"),
            line_search_maximum_attempts: 5,
            step_halving_scale: g_plan::StepScale::try_from(0.5).expect("valid step scale"),
        },
    }
}

fn run_plan(
    directory: &TestDirectory,
    inputs: &TestInputs,
    phenotype_names: &[&str],
    resume: bool,
    writer_thread_count: u32,
) -> Arc<g_plan::RunPlan> {
    Arc::new(g_plan::RunPlan {
        association_mode: g_plan::AssociationMode::Regenie2Binary,
        chunk_size: 3,
        input: g_plan::InputPlan {
            bgen_path: inputs.bgen.display().to_string(),
            sample_path: inputs.sample.display().to_string(),
            phenotype_path: inputs.phenotype.display().to_string(),
            prediction_list_path: inputs.prediction_list.display().to_string(),
            covariate_path: None,
            covariate_names: Vec::new(),
        },
        compute: g_plan::ComputePlan {
            device: g_plan::Device::Gpu,
            cpu_thread_count: None,
            jax_cache_directory: None,
            multi_phenotype_sample_mode: g_plan::MultiPhenotypeSampleMode::CompleteCase,
            kernels: kernel_plan(),
        },
        correction: g_plan::CorrectionPlan {
            method: g_plan::BinaryFallbackMethod::FirthApproximate,
            p_threshold: g_plan::Probability::try_from(0.05).expect("valid correction threshold"),
            firth_se: false,
        },
        output: g_plan::OutputPlan {
            output_run_root: directory.path.join("results").display().to_string(),
            resume,
            writer_thread_count,
        },
        telemetry: g_plan::TelemetryMode::Off,
        phenotype_runs: phenotype_names
            .iter()
            .enumerate()
            .map(|(index, phenotype_name)| g_plan::PhenotypeRunPlan {
                phenotype_name: (*phenotype_name).to_string(),
                output_directory_name: format!("phenotype_{index:04}_{phenotype_name}"),
            })
            .collect(),
    })
}

fn timestamp_nanoseconds(seconds: i64, nanoseconds: i64) -> i64 {
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(nanoseconds))
        .expect("test timestamp fits int64")
}

fn bgen_identity(path: &Path) -> Arc<BgenSourceIdentity> {
    let canonical_path = path.canonicalize().expect("test BGEN canonicalizes");
    let metadata = canonical_path.metadata().expect("test BGEN metadata exists");
    Arc::new(BgenSourceIdentity {
        configured_path: path.to_path_buf(),
        canonical_path: Some(canonical_path),
        device_identifier: metadata.dev(),
        inode_identifier: metadata.ino(),
        change_time_nanoseconds: timestamp_nanoseconds(metadata.ctime(), metadata.ctime_nsec()),
        modification_time_nanoseconds: timestamp_nanoseconds(metadata.mtime(), metadata.mtime_nsec()),
        file_size: metadata.len(),
    })
}

fn header(phenotype_name: &str, inputs: &TestInputs, variant_count: usize) -> CurrentRunManifestHeaderInput {
    CurrentRunManifestHeaderInput {
        phenotype_name: phenotype_name.to_string(),
        bgen_source_identity: bgen_identity(&inputs.bgen),
        covariate_names: Arc::from(Vec::<String>::new()),
        prediction_loco_files: Arc::from(Vec::new()),
        sample_count: 12,
        variant_count,
        resolved_gpu_genotype_format: g_plan::GpuGenotypeFormat::Packed8,
        sample_mode: g_plan::MultiPhenotypeSampleMode::CompleteCase,
        phenotype_compute_group_id: Arc::from("group-id"),
        sample_set_fingerprint: Arc::from("sample-fingerprint"),
        covariate_design_fingerprint: Arc::from("covariate-fingerprint"),
        phenotype_design_fingerprint: Arc::from(format!("phenotype-{phenotype_name}")),
        prediction_alignment_fingerprint: Arc::from("prediction-fingerprint"),
    }
}

fn metadata_store(variant_count: usize) -> Arc<VariantMetadataStore> {
    use std::fmt::Write as _;

    let dictionary: Box<[Arc<str>]> = ["22", "A", "C"].map(Arc::<str>::from).into();
    let mut identifier_text = String::new();
    let mut identifier_offsets = vec![0_u32];
    for variant_index in 0..variant_count {
        write!(&mut identifier_text, "rs{variant_index}").expect("test identifier writes");
        identifier_offsets.push(u32::try_from(identifier_text.len()).expect("test identifiers fit uint32"));
    }
    Arc::new(
        VariantMetadataStore::from_parts(
            dictionary,
            vec![0_u32; variant_count].into_boxed_slice(),
            identifier_text.into_boxed_str(),
            identifier_offsets.into_boxed_slice(),
            (0..variant_count)
                .map(|index| 1_000_i64 + i64::try_from(index).expect("test index fits int64"))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            vec![1_u32; variant_count].into_boxed_slice(),
            vec![2_u32; variant_count].into_boxed_slice(),
        )
        .expect("test metadata store should satisfy its invariants"),
    )
}

fn test_chunk(store: &Arc<VariantMetadataStore>, chunk_range: Range<usize>, trait_count: usize) -> TestChunk {
    let row_count = chunk_range.len();
    let chunk_identifier = i64::try_from(chunk_range.start).expect("test chunk identifier fits int64");
    let metadata = VariantMetadataColumns::new(Arc::clone(store), chunk_range.clone())
        .expect("test metadata range should be valid");
    let metadata_handle = NativeVariantMetadataHandle::try_new(&metadata).expect("test metadata is valid");
    let mut info_score = NullableFloat32Column {
        values: Vec::with_capacity(row_count),
        validity_bytes: Vec::with_capacity(row_count.div_ceil(8)),
    };
    for row_index in 0..row_count {
        info_score.push(0.8 + small_index_as_f32(row_index) * 0.01, row_index != 1);
    }
    let handle = NativeChunkHandle::try_new(
        metadata_handle,
        ChunkOutputStatistics {
            allele_one_frequency: (0..row_count).map(|index| 0.1 + small_index_as_f32(index) * 0.05).collect(),
            observation_count: (0..row_count)
                .map(|index| 12 - i32::try_from(index).expect("test index fits int32"))
                .collect(),
            info_score,
        },
        chunk_identifier,
    )
    .expect("test chunk is valid");
    let value_count = row_count.checked_mul(trait_count).expect("test statistic size fits");
    let beta = (0..value_count).map(|index| small_index_as_f32(index) + 0.25).collect::<Vec<_>>();
    let standard_error = (0..value_count).map(|index| small_index_as_f32(index) + 0.5).collect::<Vec<_>>();
    let chi_squared = (0..value_count).map(|index| small_index_as_f32(index) + 0.75).collect::<Vec<_>>();
    let log10_p_value = (0..value_count).map(|index| small_index_as_f32(index) + 1.0).collect::<Vec<_>>();
    let correction_code = Some((0..value_count).map(|index| u8::try_from(index % 4).expect("code fits")).collect());
    TestChunk {
        handle,
        statistics: Regenie2StatisticBatch {
            trait_count,
            variant_count: row_count,
            beta,
            standard_error,
            chi_squared,
            log10_p_value,
            correction_code,
        },
    }
}

fn small_index_as_f32(index: usize) -> f32 {
    f32::from(u16::try_from(index).expect("small test index fits uint16"))
}

fn single_chunk_plan(chunk_range: Range<usize>) -> Vec<Range<usize>> {
    std::iter::once(chunk_range).collect()
}

fn initialize_manager(
    run_plan: Arc<g_plan::RunPlan>,
    inputs: &TestInputs,
    phenotype_names: &[&str],
    planned_chunk_ranges: &[Range<usize>],
) -> OutputManager {
    let variant_count = planned_chunk_ranges.last().map_or(0, |range| range.end);
    let headers = phenotype_names.iter().map(|phenotype_name| header(phenotype_name, inputs, variant_count)).collect();
    let mut manager = OutputManager::open(run_plan, "# test configuration\n".to_string()).expect("manager opens");
    manager.initialize(headers, planned_chunk_ranges, false).expect("manager initializes");
    manager
}

fn only_parquet_part(parts_directory: &Path) -> PathBuf {
    let mut part_paths = std::fs::read_dir(parts_directory)
        .expect("parts directory exists")
        .map(|entry| entry.expect("part entry is readable").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "parquet"))
        .collect::<Vec<_>>();
    part_paths.sort();
    assert_eq!(part_paths.len(), 1);
    part_paths.remove(0)
}

fn read_manifest(run_directory: &Path) -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(run_directory.join("run_manifest.json")).expect("manifest is readable"),
    )
    .expect("manifest is valid JSON")
}

fn read_float_column(parts_directory: &Path, column_name: &str) -> Vec<f32> {
    let input_file = File::open(only_parquet_part(parts_directory)).expect("part opens");
    let batches = ParquetRecordBatchReaderBuilder::try_new(input_file)
        .expect("part metadata reads")
        .build()
        .expect("part reader builds")
        .collect::<Result<Vec<_>, _>>()
        .expect("part reads");
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column_by_name(column_name)
                .expect("column exists")
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("column is Float32")
                .values()
                .to_vec()
        })
        .collect()
}

#[test]
fn writer_persists_nullable_info_labels_and_pre_release_contract_versions() {
    let directory = TestDirectory::new("nullable-info");
    let phenotype_names = [PRIMARY_PHENOTYPE];
    let inputs = test_inputs(&directory, &phenotype_names);
    let plan = run_plan(&directory, &inputs, &phenotype_names, false, 1);
    let manager = initialize_manager(Arc::clone(&plan), &inputs, &phenotype_names, &single_chunk_plan(0..3));
    let sessions = manager
        .delivery_state_for_phenotypes(&[PRIMARY_PHENOTYPE.to_string()])
        .expect("delivery state exists")
        .writer_sessions;
    let chunk = test_chunk(&metadata_store(3), 0..3, 1);
    write_regenie2_multi_trait_chunk_f32(&sessions, None, &chunk.handle, chunk.statistics).expect("chunk is accepted");
    drop(sessions);
    let completed = manager.finish().expect("manager finishes");
    assert_eq!(completed.len(), 1);

    let manifest = read_manifest(&completed[0].run_directory);
    assert_eq!(manifest["schema_version"], 0);
    assert_eq!(manifest["output_schema_version"], 0);
    assert_eq!(manifest["status"], "completed");
    assert_eq!(manifest["committed_chunks"].as_array().expect("commits are a list").len(), 1);

    let part_path = only_parquet_part(&completed[0].parts_directory);
    assert_eq!(part_path.file_name().and_then(|name| name.to_str()), Some("part_000000000.parquet"));
    assert!(!part_path.with_extension("parquet.tmp").exists());
    let input_file = File::open(&part_path).expect("part opens");
    let builder = ParquetRecordBatchReaderBuilder::try_new(input_file).expect("part metadata reads");
    assert!(builder.schema().field_with_name("INFO").expect("INFO field exists").is_nullable());
    let footer_metadata = builder.metadata().file_metadata().key_value_metadata().expect("footer metadata exists");
    assert!(footer_metadata.iter().any(|entry| entry.key == crate::schema::CHUNK_COMMITS_METADATA_KEY));
    let batches = builder.build().expect("part reader builds").collect::<Result<Vec<_>, _>>().expect("part reads");
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 3);
    let positions = batch.column_by_name("GENPOS").expect("GENPOS exists");
    assert_eq!(
        positions.as_any().downcast_ref::<Int64Array>().expect("GENPOS is Int64").values(),
        &[1_000, 1_001, 1_002]
    );
    let observation_counts = batch.column_by_name("N").expect("N exists");
    assert_eq!(observation_counts.as_any().downcast_ref::<Int32Array>().expect("N is Int32").values(), &[12, 11, 10]);
    let info_scores = batch
        .column_by_name("INFO")
        .expect("INFO exists")
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("INFO is Float32");
    assert!(!info_scores.is_null(0));
    assert!(info_scores.is_null(1));
    assert!(!info_scores.is_null(2));
    assert!((info_scores.value(0) - 0.8).abs() < f32::EPSILON);
    let methods = batch
        .column_by_name("CORRECTION_METHOD")
        .expect("method exists")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("method reads as Utf8");
    let statuses = batch
        .column_by_name("CORRECTION_STATUS")
        .expect("status exists")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("status reads as Utf8");
    assert_eq!((0..3).map(|index| methods.value(index)).collect::<Vec<_>>(), ["score", "score", "firth_approximate"]);
    assert_eq!((0..3).map(|index| statuses.value(index)).collect::<Vec<_>>(), ["success", "failed", "success"]);
}

#[test]
fn enabled_stage_timing_records_writer_and_finish_metrics() {
    let directory = TestDirectory::new("stage-timing");
    let phenotype_names = [PRIMARY_PHENOTYPE];
    let inputs = test_inputs(&directory, &phenotype_names);
    let plan = run_plan(&directory, &inputs, &phenotype_names, false, 1);
    let planned_ranges = single_chunk_plan(0..2);
    let headers = vec![header(PRIMARY_PHENOTYPE, &inputs, 2)];
    let mut manager = OutputManager::open(plan, "# timing\n".to_string()).expect("manager opens");
    manager.initialize(headers, &planned_ranges, true).expect("manager initializes with timing");
    let sessions = manager
        .delivery_state_for_phenotypes(&[PRIMARY_PHENOTYPE.to_string()])
        .expect("delivery state exists")
        .writer_sessions;
    let chunk = test_chunk(&metadata_store(2), 0..2, 1);
    write_regenie2_multi_trait_chunk_f32(&sessions, None, &chunk.handle, chunk.statistics).expect("chunk is accepted");
    drop(sessions);
    let completed = manager.finish().expect("manager finishes");

    let timing_path = completed[0].run_directory.join("output_stage_timings.json");
    let timing: Value = serde_json::from_str(&std::fs::read_to_string(timing_path).expect("timing snapshot reads"))
        .expect("timing snapshot is valid JSON");
    assert_eq!(timing["stage_counts"]["rust_output_enqueue"], 1);
    assert_eq!(timing["stage_counts"]["rust_output_coordinator_flush"], 1);
    assert_eq!(timing["stage_counts"]["rust_output_writer_total"], 1);
    assert_eq!(timing["stage_counts"]["rust_output_manifest_commit"], 1);
    assert_eq!(timing["stage_counts"]["rust_output_finish_total"], 1);
    assert_eq!(timing["output_metrics"]["writer_chunk_file_count"], 1);
    assert_eq!(timing["output_metrics"]["writer_chunk_count"], 1);
    assert_eq!(timing["output_metrics"]["writer_row_count"], 2);
    assert!(timing["output_metrics"]["writer_arrow_array_memory_bytes"].as_u64().expect("memory is uint64") > 0);
    assert!(timing["output_metrics"]["writer_parquet_file_bytes"].as_u64().expect("bytes are uint64") > 0);
}

#[test]
fn writer_groups_eight_chunks_and_flushes_tail_with_sorted_commits() {
    let directory = TestDirectory::new("grouped-tail");
    let phenotype_names = [PRIMARY_PHENOTYPE];
    let inputs = test_inputs(&directory, &phenotype_names);
    let planned_chunk_ranges = (0..9).map(|index| index..index + 1).collect::<Vec<_>>();
    let plan = run_plan(&directory, &inputs, &phenotype_names, false, 2);
    let manager = initialize_manager(plan, &inputs, &phenotype_names, &planned_chunk_ranges);
    let sessions = manager
        .delivery_state_for_phenotypes(&[PRIMARY_PHENOTYPE.to_string()])
        .expect("delivery state exists")
        .writer_sessions;
    let store = metadata_store(9);
    for chunk_range in planned_chunk_ranges.clone() {
        let chunk = test_chunk(&store, chunk_range, 1);
        write_regenie2_multi_trait_chunk_f32(&sessions, None, &chunk.handle, chunk.statistics)
            .expect("chunk is accepted");
    }
    drop(sessions);
    let completed = manager.finish().expect("manager finishes");
    let mut part_names = std::fs::read_dir(&completed[0].parts_directory)
        .expect("parts directory exists")
        .map(|entry| entry.expect("part entry is readable").file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".parquet"))
        .collect::<Vec<_>>();
    part_names.sort();
    assert_eq!(part_names, ["part_000000000_000000007.parquet", "part_000000008.parquet"]);
    let manifest = read_manifest(&completed[0].run_directory);
    let identifiers = manifest["committed_chunks"]
        .as_array()
        .expect("commits are a list")
        .iter()
        .map(|commit| commit["chunk_identifier"].as_i64().expect("identifier is int64"))
        .collect::<Vec<_>>();
    assert_eq!(identifiers, (0_i64..9).collect::<Vec<_>>());
}

#[test]
fn reordered_partial_traits_preserve_trait_major_values_across_resume() {
    let directory = TestDirectory::new("partial-traits");
    let phenotype_names = [PRIMARY_PHENOTYPE, "trait_beta", "trait_gamma"];
    let inputs = test_inputs(&directory, &phenotype_names);
    let planned_ranges = single_chunk_plan(0..2);
    let initial_plan = run_plan(&directory, &inputs, &phenotype_names, false, 2);
    let manager = initialize_manager(initial_plan, &inputs, &phenotype_names, &planned_ranges);
    let sessions = manager
        .delivery_state_for_phenotypes(&phenotype_names.map(str::to_string))
        .expect("delivery state exists")
        .writer_sessions;
    let chunk = test_chunk(&metadata_store(2), 0..2, 2);
    let statistics = Regenie2StatisticBatch {
        trait_count: 2,
        variant_count: 2,
        beta: vec![10.0, 11.0, 20.0, 21.0],
        standard_error: chunk.statistics.standard_error,
        chi_squared: chunk.statistics.chi_squared,
        log10_p_value: chunk.statistics.log10_p_value,
        correction_code: chunk.statistics.correction_code,
    };
    write_regenie2_multi_trait_chunk_f32(&sessions, Some(&[2, 0]), &chunk.handle, statistics)
        .expect("reordered subset writes");
    drop(sessions);
    manager.finish().expect("initial manager finishes");

    let resume_plan = run_plan(&directory, &inputs, &phenotype_names, true, 2);
    let resume_manager = initialize_manager(resume_plan, &inputs, &phenotype_names, &planned_ranges);
    let delivery = resume_manager
        .delivery_state_for_phenotypes(&phenotype_names.map(str::to_string))
        .expect("resume delivery state exists");
    assert_eq!(delivery.committed_chunk_identifier_sets[0].iter().copied().collect::<Vec<_>>(), [0]);
    assert!(delivery.committed_chunk_identifier_sets[1].is_empty());
    assert_eq!(delivery.committed_chunk_identifier_sets[2].iter().copied().collect::<Vec<_>>(), [0]);
    let resumed_sessions = delivery.writer_sessions;
    let resumed_chunk = test_chunk(&metadata_store(2), 0..2, 1);
    let resumed_statistics = Regenie2StatisticBatch {
        trait_count: 1,
        variant_count: 2,
        beta: vec![30.0, 31.0],
        standard_error: resumed_chunk.statistics.standard_error,
        chi_squared: resumed_chunk.statistics.chi_squared,
        log10_p_value: resumed_chunk.statistics.log10_p_value,
        correction_code: resumed_chunk.statistics.correction_code,
    };
    write_regenie2_multi_trait_chunk_f32(&resumed_sessions, Some(&[1]), &resumed_chunk.handle, resumed_statistics)
        .expect("remaining trait writes after resume");
    drop(resumed_sessions);
    resume_manager.finish().expect("resume manager finishes");

    let output_root = directory.path.join("results");
    for (directory_name, expected_beta) in [
        ("phenotype_0000_trait_alpha.regenie2_binary.run", vec![20.0, 21.0]),
        ("phenotype_0001_trait_beta.regenie2_binary.run", vec![30.0, 31.0]),
        ("phenotype_0002_trait_gamma.regenie2_binary.run", vec![10.0, 11.0]),
    ] {
        let observed_beta = read_float_column(&output_root.join(directory_name).join("parts"), "BETA");
        assert_eq!(observed_beta.len(), expected_beta.len());
        for (observed, expected) in observed_beta.iter().zip(expected_beta) {
            assert!((*observed - expected).abs() < f32::EPSILON);
        }
    }
}

#[test]
fn interrupted_finish_flushes_pending_chunk_and_records_signal() {
    let directory = TestDirectory::new("interrupted");
    let phenotype_names = [PRIMARY_PHENOTYPE];
    let inputs = test_inputs(&directory, &phenotype_names);
    let plan = run_plan(&directory, &inputs, &phenotype_names, false, 1);
    let manager = initialize_manager(plan, &inputs, &phenotype_names, &single_chunk_plan(0..1));
    let sessions = manager
        .delivery_state_for_phenotypes(&[PRIMARY_PHENOTYPE.to_string()])
        .expect("delivery state exists")
        .writer_sessions;
    let chunk = test_chunk(&metadata_store(1), 0..1, 1);
    write_regenie2_multi_trait_chunk_f32(&sessions, None, &chunk.handle, chunk.statistics).expect("chunk is accepted");
    let run_directory = directory.path.join("results").join("phenotype_0000_trait_alpha.regenie2_binary.run");
    drop(sessions);
    manager.finish_interrupted("SIGTERM").expect("interrupted finish succeeds");
    let manifest = read_manifest(&run_directory);
    assert_eq!(manifest["status"], "interrupted");
    assert_eq!(manifest["interrupted_signal"], "SIGTERM");
    assert!(run_directory.join("parts/part_000000000.parquet").exists());
}

#[test]
fn abort_discards_pending_chunks_and_closes_retained_session() {
    let directory = TestDirectory::new("abort");
    let phenotype_names = [PRIMARY_PHENOTYPE];
    let inputs = test_inputs(&directory, &phenotype_names);
    let plan = run_plan(&directory, &inputs, &phenotype_names, false, 1);
    let manager = initialize_manager(plan, &inputs, &phenotype_names, &single_chunk_plan(0..1));
    let sessions = manager
        .delivery_state_for_phenotypes(&[PRIMARY_PHENOTYPE.to_string()])
        .expect("delivery state exists")
        .writer_sessions;
    let retained_session: Arc<OutputWriterSession> = Arc::clone(&sessions[0]);
    let chunk = test_chunk(&metadata_store(1), 0..1, 1);
    write_regenie2_multi_trait_chunk_f32(&sessions, None, &chunk.handle, chunk.statistics).expect("chunk is accepted");
    drop(sessions);
    manager.abort().expect("abort succeeds");

    let second_chunk = test_chunk(&metadata_store(1), 0..1, 1);
    let error =
        write_regenie2_multi_trait_chunk_f32(&[retained_session], None, &second_chunk.handle, second_chunk.statistics)
            .expect_err("closed session rejects writes");
    assert!(error.to_string().contains("already closed"));
    let run_directory = directory.path.join("results").join("phenotype_0000_trait_alpha.regenie2_binary.run");
    assert_eq!(read_manifest(&run_directory)["status"], "running");
    assert_eq!(std::fs::read_dir(run_directory.join("parts")).expect("parts exists").count(), 0);
}

#[test]
fn abort_leaves_completed_worker_part_for_strict_resume_repair() {
    let directory = TestDirectory::new("abort-enqueued");
    let phenotype_names = [PRIMARY_PHENOTYPE];
    let inputs = test_inputs(&directory, &phenotype_names);
    let planned_chunk_ranges = (0..8).map(|index| index..index + 1).collect::<Vec<_>>();
    let plan = run_plan(&directory, &inputs, &phenotype_names, false, 1);
    let manager = initialize_manager(plan, &inputs, &phenotype_names, &planned_chunk_ranges);
    let sessions = manager
        .delivery_state_for_phenotypes(&[PRIMARY_PHENOTYPE.to_string()])
        .expect("delivery state exists")
        .writer_sessions;
    let store = metadata_store(8);
    for chunk_range in planned_chunk_ranges.clone() {
        let chunk = test_chunk(&store, chunk_range, 1);
        write_regenie2_multi_trait_chunk_f32(&sessions, None, &chunk.handle, chunk.statistics)
            .expect("chunk is accepted");
    }
    drop(sessions);
    manager.abort().expect("abort waits for enqueued worker");

    let run_directory = directory.path.join("results").join("phenotype_0000_trait_alpha.regenie2_binary.run");
    assert!(run_directory.join("parts/part_000000000_000000007.parquet").is_file());
    assert!(read_manifest(&run_directory)["committed_chunks"].as_array().expect("commits are a list").is_empty());

    let resume_plan = run_plan(&directory, &inputs, &phenotype_names, true, 1);
    let resume_manager = initialize_manager(resume_plan, &inputs, &phenotype_names, &planned_chunk_ranges);
    let delivery = resume_manager
        .delivery_state_for_phenotypes(&[PRIMARY_PHENOTYPE.to_string()])
        .expect("resume delivery state exists");
    assert_eq!(
        delivery.committed_chunk_identifier_sets[0].iter().copied().collect::<Vec<_>>(),
        (0_usize..8).collect::<Vec<_>>()
    );
    drop(delivery);
    resume_manager.finish().expect("resume manager finishes");
    assert_eq!(read_manifest(&run_directory)["status"], "completed");
}

#[test]
fn strict_resume_repairs_orphan_commit_and_rejects_each_nonzero_contract_version() {
    let directory = TestDirectory::new("resume");
    let phenotype_names = [PRIMARY_PHENOTYPE];
    let inputs = test_inputs(&directory, &phenotype_names);
    let initial_plan = run_plan(&directory, &inputs, &phenotype_names, false, 1);
    let manager = initialize_manager(initial_plan, &inputs, &phenotype_names, &single_chunk_plan(0..1));
    let sessions = manager
        .delivery_state_for_phenotypes(&[PRIMARY_PHENOTYPE.to_string()])
        .expect("delivery state exists")
        .writer_sessions;
    let chunk = test_chunk(&metadata_store(1), 0..1, 1);
    write_regenie2_multi_trait_chunk_f32(&sessions, None, &chunk.handle, chunk.statistics).expect("chunk is accepted");
    drop(sessions);
    let completed = manager.finish().expect("manager finishes");
    let run_directory = &completed[0].run_directory;
    let manifest_path = run_directory.join("run_manifest.json");
    let mut orphan_manifest = read_manifest(run_directory);
    orphan_manifest["committed_chunks"] = Value::Array(Vec::new());
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&orphan_manifest).expect("manifest serializes"))
        .expect("orphan manifest is written");

    let resume_plan = run_plan(&directory, &inputs, &phenotype_names, true, 1);
    let mut resume_manager =
        OutputManager::open(Arc::clone(&resume_plan), "# resumed\n".to_string()).expect("resume manager opens");
    resume_manager
        .initialize(vec![header(PRIMARY_PHENOTYPE, &inputs, 1)], &single_chunk_plan(0..1), false)
        .expect("orphan part repairs manifest commits");
    let delivery = resume_manager
        .delivery_state_for_phenotypes(&[PRIMARY_PHENOTYPE.to_string()])
        .expect("resume delivery state exists");
    assert_eq!(delivery.committed_chunk_identifier_sets[0].iter().copied().collect::<Vec<_>>(), [0]);
    drop(delivery);
    resume_manager.finish().expect("resumed manager finishes");

    let compatible_manifest = read_manifest(run_directory);
    for field_name in ["schema_version", "output_schema_version"] {
        let mut incompatible_manifest = compatible_manifest.clone();
        incompatible_manifest[field_name] = Value::from(1);
        let incompatible_bytes = serde_json::to_vec_pretty(&incompatible_manifest).expect("manifest serializes");
        std::fs::write(&manifest_path, &incompatible_bytes).expect("incompatible manifest is written");
        let incompatible_plan = run_plan(&directory, &inputs, &phenotype_names, true, 1);
        let mut incompatible_manager = OutputManager::open(incompatible_plan, "# incompatible\n".to_string())
            .expect("incompatible manager opens before validation");
        let error = incompatible_manager
            .initialize(vec![header(PRIMARY_PHENOTYPE, &inputs, 1)], &single_chunk_plan(0..1), false)
            .expect_err("nonzero pre-release contract version must fail");
        assert!(error.to_string().contains(field_name), "unexpected error: {error}");
        assert_eq!(std::fs::read(&manifest_path).expect("manifest remains readable"), incompatible_bytes);
    }
}

#[test]
fn manager_validates_initialization_and_trait_routing() {
    let directory = TestDirectory::new("manager-validation");
    let phenotype_names = [PRIMARY_PHENOTYPE, "trait_beta"];
    let inputs = test_inputs(&directory, &phenotype_names);
    let plan = run_plan(&directory, &inputs, &phenotype_names, false, 2);
    let mut manager = OutputManager::open(plan, "# test\n".to_string()).expect("manager opens");
    let error = manager
        .delivery_state_for_phenotypes(&[PRIMARY_PHENOTYPE.to_string()])
        .err()
        .expect("delivery before initialization must fail");
    assert!(error.to_string().contains("must be initialized before delivery"));
    assert_eq!(
        manager.existing_manifest_gpu_genotype_format(PRIMARY_PHENOTYPE).expect("fresh run manifest lookup succeeds"),
        None
    );
    let error =
        manager.existing_manifest_gpu_genotype_format("unknown").expect_err("unknown manifest phenotype must fail");
    assert!(error.to_string().contains("Unknown planned phenotype"));
    let error = manager
        .initialize(vec![header(PRIMARY_PHENOTYPE, &inputs, 1)], &single_chunk_plan(0..1), false)
        .expect_err("missing phenotype header must fail");
    assert!(error.to_string().contains("initialization count"));
    let error = manager
        .initialize(
            vec![header(PRIMARY_PHENOTYPE, &inputs, 1), header(PRIMARY_PHENOTYPE, &inputs, 1)],
            &single_chunk_plan(0..1),
            false,
        )
        .expect_err("duplicate phenotype header must fail");
    assert!(error.to_string().contains("Duplicate output initialization"));
    manager
        .initialize(
            vec![header(PRIMARY_PHENOTYPE, &inputs, 1), header("trait_beta", &inputs, 1)],
            &single_chunk_plan(0..1),
            false,
        )
        .expect("complete initialization succeeds");
    let error = manager
        .initialize(
            vec![header(PRIMARY_PHENOTYPE, &inputs, 1), header("trait_beta", &inputs, 1)],
            &single_chunk_plan(0..1),
            false,
        )
        .expect_err("second initialization must fail");
    assert!(error.to_string().contains("already initialized"));
    let error =
        manager.delivery_state_for_phenotypes(&["unknown".to_string()]).err().expect("unknown phenotype must fail");
    assert!(error.to_string().contains("Unknown planned phenotype"));
    let delivery = manager
        .delivery_state_for_phenotypes(&["trait_beta".to_string(), PRIMARY_PHENOTYPE.to_string()])
        .expect("reordered trait delivery succeeds");
    assert_eq!(delivery.writer_sessions.len(), 2);
    drop(delivery);
    manager.abort().expect("manager aborts");

    let unfinished_plan = run_plan(&directory, &inputs, &phenotype_names, false, 2);
    let unfinished_root = directory.path.join("unfinished-results");
    let mut unfinished_plan = Arc::try_unwrap(unfinished_plan).expect("test plan has one owner");
    unfinished_plan.output.output_run_root = unfinished_root.display().to_string();
    let unfinished_manager =
        OutputManager::open(Arc::new(unfinished_plan), "# unfinished\n".to_string()).expect("manager opens");
    let error = unfinished_manager.finish().err().expect("finish before initialization must fail");
    assert!(error.to_string().contains("must be initialized before completion"));

    let duplicate_plan = run_plan(&directory, &inputs, &phenotype_names, false, 2);
    let mut duplicate_plan = Arc::try_unwrap(duplicate_plan).expect("test plan has one owner");
    duplicate_plan.output.output_run_root = directory.path.join("duplicate-results").display().to_string();
    duplicate_plan.phenotype_runs[1].phenotype_name = PRIMARY_PHENOTYPE.to_string();
    let error = OutputManager::open(Arc::new(duplicate_plan), "# duplicate\n".to_string())
        .err()
        .expect("duplicate output phenotype must fail");
    assert!(error.to_string().contains("Duplicate phenotype output name"));
}
