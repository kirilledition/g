use std::collections::BTreeMap;
use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::Float32Array;
use g_genotype_contracts::{
    BgenContentEvidence, BgenContentFingerprint, BgenContentSha256, BgenSourceIdentity, ChunkOutputStatistics,
    NullableFloat32Column, VariantMetadataColumns, VariantMetadataStore,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{
    AssociationImplementationCompatibility, CurrentRunManifestHeaderInput, FirthComponentsCompatibility,
    NativeChunkHandle, NativeVariantMetadataHandle, OutputManager, RawCudaFirthArtifactCompatibility,
    RawCudaFirthCapabilityRequirementsCompatibility, Regenie2StatisticBatch, write_regenie2_multi_trait_chunk_f32,
};

const PHENOTYPE_NAME: &str = "trait_alpha";
const OUTPUT_DIRECTORY_NAME: &str = "trait_0001_trait_alpha";
const SECOND_PHENOTYPE_NAME: &str = "trait_beta";
const SECOND_OUTPUT_DIRECTORY_NAME: &str = "trait_0002_trait_beta";
const TRANSACTION_HELPER_MODE_ENVIRONMENT: &str = "G_OUTPUT_TRANSACTION_HELPER_MODE";
const TRANSACTION_HELPER_READY_ENVIRONMENT: &str = "G_OUTPUT_TRANSACTION_HELPER_READY";
const TRANSACTION_HELPER_ROOT_ENVIRONMENT: &str = "G_OUTPUT_TRANSACTION_HELPER_ROOT";
const TRANSACTION_HELPER_TEST_NAME: &str = "tests::output_transaction_subprocess_helper";

fn single_chunk_ranges(stop: usize) -> Vec<Range<usize>> {
    std::iter::once(0..stop).collect()
}

fn single_chunk_ranges_from(start: usize, stop: usize) -> Vec<Range<usize>> {
    std::iter::once(start..stop).collect()
}

struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn new(label: &str) -> Self {
        static DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);
        let sequence = DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).expect("test time is after Unix epoch").as_nanos();
        let path = std::env::temp_dir()
            .join(format!("g-output-transaction-{label}-{}-{timestamp}-{sequence}", std::process::id()));
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

fn test_inputs(directory: &TestDirectory) -> TestInputs {
    TestInputs {
        bgen: directory.write("input.bgen", b"test bgen identity"),
        sample: directory.write("input.sample", b"ID_1 ID_2\n0 0\nfamily sample\n"),
        phenotype: directory.write("phenotypes.tsv", b"FID\tIID\ttrait_alpha\nfamily\tsample\t1\n"),
        prediction_list: directory.write("predictions.list", b"trait_alpha predictions.loco\n"),
    }
}

fn existing_test_inputs(directory: &TestDirectory) -> TestInputs {
    TestInputs {
        bgen: directory.path.join("input.bgen"),
        sample: directory.path.join("input.sample"),
        phenotype: directory.path.join("phenotypes.tsv"),
        prediction_list: directory.path.join("predictions.list"),
    }
}

fn run_plan(
    directory: &TestDirectory,
    inputs: &TestInputs,
    resume: bool,
    recover_attempt: Option<String>,
    telemetry: g_plan::TelemetryMode,
) -> Arc<g_plan::RunPlan> {
    Arc::new(g_plan::RunPlan {
        association_mode: g_plan::AssociationMode::Regenie2Binary,
        chunk_size: 2,
        input: g_plan::InputPlan {
            bgen_path: inputs.bgen.display().to_string(),
            bgen_content_sha256: None,
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
            recover_attempt,
            fenced_owner_claim_id: None,
            writer_thread_count: 2,
        },
        telemetry,
        phenotype_runs: vec![g_plan::PhenotypeRunPlan {
            phenotype_name: PHENOTYPE_NAME.to_string(),
            output_directory_name: OUTPUT_DIRECTORY_NAME.to_string(),
        }],
    })
}

fn with_association_and_correction(
    mut run_plan: Arc<g_plan::RunPlan>,
    association_mode: g_plan::AssociationMode,
    correction_method: g_plan::BinaryFallbackMethod,
) -> Arc<g_plan::RunPlan> {
    let owned_run_plan = Arc::get_mut(&mut run_plan).expect("test run plan has one owner");
    owned_run_plan.association_mode = association_mode;
    owned_run_plan.correction.method = correction_method;
    run_plan
}

fn two_phenotype_run_plan(
    directory: &TestDirectory,
    inputs: &TestInputs,
    resume: bool,
    recover_attempt: Option<String>,
    telemetry: g_plan::TelemetryMode,
) -> Arc<g_plan::RunPlan> {
    let mut plan = run_plan(directory, inputs, resume, recover_attempt, telemetry);
    Arc::get_mut(&mut plan).expect("test run plan has one owner").phenotype_runs.push(g_plan::PhenotypeRunPlan {
        phenotype_name: SECOND_PHENOTYPE_NAME.to_string(),
        output_directory_name: SECOND_OUTPUT_DIRECTORY_NAME.to_string(),
    });
    plan
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

fn header(inputs: &TestInputs, variant_count: usize) -> CurrentRunManifestHeaderInput {
    header_for_phenotype(inputs, variant_count, PHENOTYPE_NAME)
}

fn header_for_phenotype(
    inputs: &TestInputs,
    variant_count: usize,
    phenotype_name: &str,
) -> CurrentRunManifestHeaderInput {
    header_for_phenotype_with_evidence(
        variant_count,
        phenotype_name,
        owned_snapshot_evidence(&inputs.bgen),
        g_plan::GpuGenotypeFormat::Packed8,
    )
}

fn header_for_phenotype_with_evidence(
    variant_count: usize,
    phenotype_name: &str,
    bgen_content_evidence: Arc<BgenContentEvidence>,
    resolved_gpu_genotype_format: g_plan::GpuGenotypeFormat,
) -> CurrentRunManifestHeaderInput {
    CurrentRunManifestHeaderInput {
        phenotype_name: phenotype_name.to_string(),
        bgen_content_evidence,
        covariate_names: Arc::from(Vec::<String>::new()),
        prediction_loco_files: Arc::from(Vec::new()),
        sample_count: 12,
        variant_count,
        resolved_gpu_genotype_format,
        sample_mode: g_plan::MultiPhenotypeSampleMode::CompleteCase,
        phenotype_compute_group_id: Arc::from("group-id"),
        sample_set_fingerprint: Arc::from("sample-fingerprint"),
        covariate_design_fingerprint: Arc::from("covariate-fingerprint"),
        phenotype_design_fingerprint: Arc::from("phenotype-fingerprint"),
        prediction_alignment_fingerprint: Arc::from("prediction-fingerprint"),
    }
}

fn owned_snapshot_evidence(path: &Path) -> Arc<BgenContentEvidence> {
    Arc::new(BgenContentEvidence::OwnedSnapshot(bgen_content_fingerprint(path)))
}

fn bgen_content_fingerprint(path: &Path) -> BgenContentFingerprint {
    let bytes = std::fs::read(path).expect("test BGEN reads");
    let content_sha256 = BgenContentSha256::from_bytes(Sha256::digest(&bytes).into());
    BgenContentFingerprint {
        content_sha256,
        byte_count: u64::try_from(bytes.len()).expect("test BGEN byte count fits uint64"),
    }
}

fn expected_resume_agreement(
    inputs: &TestInputs,
    gpu_genotype_format: g_plan::GpuGenotypeFormat,
) -> crate::ExistingOutputResumeAgreement {
    crate::ExistingOutputResumeAgreement {
        bgen_content_fingerprint: bgen_content_fingerprint(&inputs.bgen),
        gpu_genotype_format,
        association_implementation: test_association_implementation(),
    }
}

fn test_association_implementation() -> AssociationImplementationCompatibility {
    AssociationImplementationCompatibility::new(
        "test-jax".to_string(),
        "test-jaxlib".to_string(),
        Some(FirthComponentsCompatibility::jax()),
    )
    .expect("test association implementation compatibility is valid")
}

fn test_non_firth_association_implementation() -> AssociationImplementationCompatibility {
    AssociationImplementationCompatibility::new("test-jax".to_string(), "test-jaxlib".to_string(), None)
        .expect("test non-Firth association implementation compatibility is valid")
}

fn test_raw_cuda_association_implementation_with_digests(
    handler_sha256_character: char,
    ptx_sha256_character: char,
) -> AssociationImplementationCompatibility {
    let capability_requirements = RawCudaFirthCapabilityRequirementsCompatibility::new(12_020, 7, 0)
        .expect("test raw-CUDA capability requirements are valid");
    let raw_cuda_artifact = RawCudaFirthArtifactCompatibility::new(
        "g.firth.components.v1".to_string(),
        1,
        std::iter::repeat_n(handler_sha256_character, 64).collect(),
        std::iter::repeat_n(ptx_sha256_character, 64).collect(),
        "8.2".to_string(),
        "sm_70".to_string(),
        capability_requirements,
    )
    .expect("test raw-CUDA artifact compatibility is valid");
    AssociationImplementationCompatibility::new(
        "test-jax".to_string(),
        "test-jaxlib".to_string(),
        Some(FirthComponentsCompatibility::raw_cuda(raw_cuda_artifact)),
    )
    .expect("test raw-CUDA association implementation is valid")
}

#[test]
fn raw_cuda_handler_digest_participates_in_exact_compatibility() {
    let first = test_raw_cuda_association_implementation_with_digests('a', 'c');
    let changed_handler = test_raw_cuda_association_implementation_with_digests('b', 'c');

    assert_ne!(first, changed_handler);
}

fn positioned_unattested_evidence(path: &Path) -> Arc<BgenContentEvidence> {
    Arc::new(BgenContentEvidence::PositionedUnattested(bgen_source_identity(path)))
}

fn bgen_source_identity(path: &Path) -> BgenSourceIdentity {
    let canonical_path = path.canonicalize().expect("test BGEN canonicalizes");
    let metadata = canonical_path.metadata().expect("test BGEN metadata exists");
    BgenSourceIdentity {
        configured_path: path.to_path_buf(),
        canonical_path: Some(canonical_path),
        device_identifier: metadata.dev(),
        inode_identifier: metadata.ino(),
        change_time_nanoseconds: timestamp_nanoseconds(metadata.ctime(), metadata.ctime_nsec()),
        modification_time_nanoseconds: timestamp_nanoseconds(metadata.mtime(), metadata.mtime_nsec()),
        file_size: metadata.len(),
    }
}

fn two_phenotype_headers(inputs: &TestInputs, variant_count: usize) -> Vec<CurrentRunManifestHeaderInput> {
    vec![header(inputs, variant_count), header_for_phenotype(inputs, variant_count, SECOND_PHENOTYPE_NAME)]
}

fn single_phenotype_headers(inputs: &TestInputs, chunk_ranges: &[Range<usize>]) -> Vec<CurrentRunManifestHeaderInput> {
    let variant_count = chunk_ranges.last().map_or(0, |range| range.end);
    vec![header(inputs, variant_count)]
}

fn planned_headers(
    run_plan: &g_plan::RunPlan,
    inputs: &TestInputs,
    chunk_ranges: &[Range<usize>],
) -> Vec<CurrentRunManifestHeaderInput> {
    let variant_count = chunk_ranges.last().map_or(0, |range| range.end);
    run_plan
        .phenotype_runs
        .iter()
        .map(|phenotype_run| header_for_phenotype(inputs, variant_count, &phenotype_run.phenotype_name))
        .collect()
}

fn timestamp_nanoseconds(seconds: i64, nanoseconds: i64) -> i64 {
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(nanoseconds))
        .expect("test timestamp fits int64")
}

fn initialize_manager(
    run_plan: Arc<g_plan::RunPlan>,
    inputs: &TestInputs,
    chunk_ranges: &[Range<usize>],
) -> crate::OutputManager<crate::Active> {
    let variant_count = chunk_ranges.last().map_or(0, |range| range.end);
    OutputManager::open(run_plan, "# test configuration\n".to_string())
        .expect("manager opens")
        .initialize(vec![header(inputs, variant_count)], chunk_ranges, true, test_association_implementation())
        .expect("manager initializes")
}

fn assert_no_post_session_cleanup(completion: crate::OutputCompletion) {
    let crate::OutputCompletion { completed_outputs: _, post_session_cleanup } = completion;
    assert!(post_session_cleanup.is_none(), "ordinary output lifecycle must complete cleanup internally");
}

struct CompletedNoopCleanupFixture {
    _directory: TestDirectory,
    output_root: PathBuf,
    lineage_paths: crate::persistence::lineage::OutputLineagePaths,
    claim_id: String,
    staging_attempt_id: crate::persistence::identifier::AttemptIdentifier,
    cleanup: crate::OutputPostSessionCleanup,
}

fn completed_noop_cleanup_fixture(label: &str) -> CompletedNoopCleanupFixture {
    let directory = TestDirectory::new(label);
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let initial_manager = initialize_manager(initial_plan, &inputs, &chunk_ranges);
    let delivery_token =
        initial_manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    write_chunk(&delivery_token, &metadata_store(2), 0..2);
    drop(delivery_token);
    assert_no_post_session_cleanup(
        initial_manager
            .close_completed()
            .expect("initial exact coverage closes")
            .finish()
            .expect("initial output completes"),
    );

    let output_root = directory.path.join("results");
    let lineage_paths = crate::persistence::lineage::OutputLineagePaths::new(&output_root);
    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Profile);
    let claimed_manager = OutputManager::open(resume_plan, "# completed no-op cleanup fixture\n".to_string())
        .expect("completed resume plans")
        .claim(single_phenotype_headers(&inputs, &chunk_ranges), &chunk_ranges, true)
        .expect("completed resume claims");
    let claim_id = owner_claim_identifier(&output_root);
    let staging_attempt_id =
        lineage_paths.owner_staging_attempt(&claim_id).expect("owner staging reads").expect("owner staging exists");
    let completion = claimed_manager
        .activate_with_deferred_completed_noop_cleanup(test_association_implementation())
        .expect("completed resume activates")
        .close_completed()
        .expect("completed resume reverifies")
        .finish()
        .expect("completed resume returns cleanup");
    let cleanup = completion.post_session_cleanup.expect("completed resume defers cleanup");
    CompletedNoopCleanupFixture {
        _directory: directory,
        output_root,
        lineage_paths,
        claim_id,
        staging_attempt_id,
        cleanup,
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
                .map(|index| 1_000_i64 + i64::try_from(index).expect("index fits"))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            vec![1_u32; variant_count].into_boxed_slice(),
            vec![2_u32; variant_count].into_boxed_slice(),
        )
        .expect("test metadata store is valid"),
    )
}

fn write_chunk(token: &crate::OutputDeliveryToken, store: &Arc<VariantMetadataStore>, chunk_range: Range<usize>) {
    let row_count = chunk_range.len();
    let metadata =
        VariantMetadataColumns::new(Arc::clone(store), chunk_range.clone()).expect("test metadata range is valid");
    let metadata_handle = NativeVariantMetadataHandle::try_new(&metadata).expect("metadata handle is valid");
    let chunk_handle = NativeChunkHandle::try_new(
        metadata_handle,
        ChunkOutputStatistics {
            allele_one_frequency: vec![0.25; row_count],
            observation_count: vec![12; row_count],
            info_score: NullableFloat32Column {
                values: vec![0.9; row_count],
                validity_bytes: vec![u8::MAX; row_count.div_ceil(8)],
            },
        },
        i64::try_from(chunk_range.start).expect("chunk index fits"),
    )
    .expect("chunk handle is valid");
    write_regenie2_multi_trait_chunk_f32(
        token,
        None,
        &chunk_handle,
        Regenie2StatisticBatch {
            trait_count: 1,
            variant_count: row_count,
            beta: vec![0.5; row_count],
            standard_error: vec![0.25; row_count],
            chi_squared: vec![4.0; row_count],
            log10_p_value: vec![2.0; row_count],
            correction_code: Some(vec![0; row_count]),
        },
    )
    .expect("chunk is accepted");
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).expect("JSON file reads")).expect("JSON file is valid")
}

fn rehash_manifest_execution_plan(manifest: &mut Value) {
    manifest["execution_plan_hash"] =
        Value::String(crate::manifest::build_manifest_value_sha256(&manifest["execution_plan"]).expect("plan hashes"));
}

#[derive(Clone, Copy)]
enum ManifestAuthorityTamper {
    RunSet,
    Attempt,
    Phenotype,
    OutputDirectory,
    ChunkPlan,
    ExecutionPlan,
}

#[derive(Clone, Copy)]
enum BgenAgreementMismatch {
    ContentSha256,
    ByteCount,
    GpuGenotypeFormat,
}

impl BgenAgreementMismatch {
    fn label(self) -> &'static str {
        match self {
            Self::ContentSha256 => "content SHA-256",
            Self::ByteCount => "byte count",
            Self::GpuGenotypeFormat => "GPU genotype format",
        }
    }

    fn expected_error_fragment(self) -> &'static str {
        match self {
            Self::ContentSha256 => "BGEN content SHA-256",
            Self::ByteCount => "BGEN content byte count",
            Self::GpuGenotypeFormat => "GPU genotype format",
        }
    }

    fn apply_to_header(self, header: &mut CurrentRunManifestHeaderInput) {
        match self {
            Self::ContentSha256 => {
                let BgenContentEvidence::OwnedSnapshot(fingerprint) = header.bgen_content_evidence.as_ref() else {
                    panic!("test header has owned BGEN content evidence");
                };
                header.bgen_content_evidence = Arc::new(BgenContentEvidence::OwnedSnapshot(BgenContentFingerprint {
                    content_sha256: BgenContentSha256::from_bytes([0x55; 32]),
                    byte_count: fingerprint.byte_count,
                }));
            }
            Self::ByteCount => {
                let BgenContentEvidence::OwnedSnapshot(fingerprint) = header.bgen_content_evidence.as_ref() else {
                    panic!("test header has owned BGEN content evidence");
                };
                header.bgen_content_evidence = Arc::new(BgenContentEvidence::OwnedSnapshot(BgenContentFingerprint {
                    content_sha256: fingerprint.content_sha256,
                    byte_count: fingerprint.byte_count + 1,
                }));
            }
            Self::GpuGenotypeFormat => {
                header.resolved_gpu_genotype_format = g_plan::GpuGenotypeFormat::Dosage;
            }
        }
    }

    fn apply_to_manifest(self, manifest: &mut Value) {
        match self {
            Self::ContentSha256 => {
                manifest["execution_plan"]["bgen"]["content_sha256"] = Value::String("f".repeat(64));
            }
            Self::ByteCount => {
                let byte_count =
                    manifest["execution_plan"]["bgen"]["byte_count"].as_u64().expect("BGEN byte count is uint64");
                manifest["execution_plan"]["bgen"]["byte_count"] = Value::from(byte_count + 1);
            }
            Self::GpuGenotypeFormat => {
                manifest["execution_plan"]["association_backend"]["kind"] = Value::String("jax_dosage".to_string());
                manifest["execution_plan"]["association_backend"]["genotype_format"] =
                    Value::String("dosage".to_string());
            }
        }
        rehash_manifest_execution_plan(manifest);
    }
}

impl ManifestAuthorityTamper {
    fn label(self) -> &'static str {
        match self {
            Self::RunSet => "run set",
            Self::Attempt => "attempt",
            Self::Phenotype => "phenotype",
            Self::OutputDirectory => "output directory",
            Self::ChunkPlan => "chunk plan",
            Self::ExecutionPlan => "execution plan",
        }
    }

    fn expected_error(self) -> &'static str {
        match self {
            Self::RunSet => "run set does not match its immutable lineage binding",
            Self::Attempt => "attempt does not match its immutable lineage attempt",
            Self::Phenotype | Self::OutputDirectory => "phenotype does not match its immutable lineage binding",
            Self::ChunkPlan => "immutable lineage chunk plan",
            Self::ExecutionPlan => "immutable lineage execution plan",
        }
    }

    fn apply(self, manifest: &mut Value) {
        match self {
            Self::RunSet => manifest["run_set_id"] = Value::String("run-set-tampered".to_string()),
            Self::Attempt => {
                manifest["attempt_id"] =
                    Value::String(crate::persistence::identifier::AttemptIdentifier::generate().as_str().to_string());
            }
            Self::Phenotype => {
                let phenotype_name = Value::String("trait_unbound".to_string());
                manifest["phenotype_name"] = phenotype_name.clone();
                manifest["command"]["phenotype"] = phenotype_name.clone();
                manifest["execution_plan"]["phenotype_name"] = phenotype_name;
                rehash_manifest_execution_plan(manifest);
            }
            Self::OutputDirectory => {
                manifest["output_directory_name"] = Value::String(SECOND_OUTPUT_DIRECTORY_NAME.to_string());
            }
            Self::ChunkPlan => manifest["chunk_plan_hash"] = Value::String("f".repeat(64)),
            Self::ExecutionPlan => {
                manifest["execution_plan"]["association_backend"]["kind"] = Value::String("jax_dosage".to_string());
                manifest["execution_plan"]["association_backend"]["genotype_format"] =
                    Value::String("dosage".to_string());
                rehash_manifest_execution_plan(manifest);
            }
        }
    }
}

fn attempt_identifier(output_root: &Path, parent_attempt: Option<&str>) -> String {
    let path = match parent_attempt {
        None => output_root.join(".g-output/genesis.json"),
        Some(parent_attempt) => {
            let terminal_successor = output_root.join(".g-output/successors").join(format!("{parent_attempt}.json"));
            if terminal_successor.exists() {
                terminal_successor
            } else {
                output_root.join(".g-output/outcomes").join(format!("{parent_attempt}.json"))
            }
        }
    };
    let record = read_json(&path);
    record
        .get("attempt_id")
        .or_else(|| record.pointer("/record/attempt_id"))
        .and_then(Value::as_str)
        .expect("attempt identifier exists")
        .to_string()
}

fn run_crashing_transaction_helper(directory: &TestDirectory, mode: &str, failpoint: &str) -> std::process::Output {
    std::process::Command::new(std::env::current_exe().expect("current test executable resolves"))
        .args(["--exact", TRANSACTION_HELPER_TEST_NAME, "--nocapture"])
        .env(TRANSACTION_HELPER_MODE_ENVIRONMENT, mode)
        .env(TRANSACTION_HELPER_ROOT_ENVIRONMENT, &directory.path)
        .env("G_OUTPUT_TEST_CRASH_POINT", failpoint)
        .output()
        .expect("transaction crash helper runs")
}

fn assert_expected_crash(output: &std::process::Output) {
    assert_eq!(
        output.status.code(),
        Some(86),
        "transaction helper did not stop at its failpoint\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_surviving_owner_claim(error: crate::OutputError, output_root: &Path) {
    match error {
        crate::OutputError::SurvivingOutputOwnerClaim { claim_path, claim_id, host_name, process_id } => {
            assert_eq!(claim_path, output_root.join(".g-output/session.claim.json"));
            assert!(claim_id.starts_with("owner-"));
            assert!(!host_name.is_empty());
            assert_ne!(process_id, 0);
        }
        unexpected_error => panic!("expected a surviving owner claim, got: {unexpected_error}"),
    }
}

fn assert_owner_authority_released(output_root: &Path) {
    assert!(output_root.join(".g-output/session.claim.json").is_file(), "the immutable root claim remains");
    crate::persistence::lineage::OutputLineagePaths::new(output_root)
        .reject_surviving_owner_claim()
        .expect("owner authority resolves to a released leaf");
}

fn claim_error(
    run_plan: Arc<g_plan::RunPlan>,
    inputs: &TestInputs,
    chunk_ranges: &[Range<usize>],
    context: &str,
) -> crate::OutputError {
    let current_header_inputs = planned_headers(&run_plan, inputs, chunk_ranges);
    OutputManager::open(run_plan, "# test configuration\n".to_string())
        .expect("manager planning is read-only")
        .claim(current_header_inputs, chunk_ranges, false)
        .err()
        .unwrap_or_else(|| panic!("{context}"))
}

fn owner_claim_identifier(output_root: &Path) -> String {
    crate::persistence::lineage::OutputLineagePaths::new(output_root)
        .current_owner_claim_identifier_for_test()
        .expect("owner authority resolves")
        .expect("owner authority is active")
}

fn authorize_fenced_owner_claim(
    mut run_plan: Arc<g_plan::RunPlan>,
    fenced_owner_claim_id: String,
) -> Arc<g_plan::RunPlan> {
    Arc::get_mut(&mut run_plan).expect("test run plan has one owner").output.fenced_owner_claim_id =
        Some(fenced_owner_claim_id);
    run_plan
}

fn owner_claim_candidate_temporary_paths(output_root: &Path) -> Vec<PathBuf> {
    let control_directory = output_root.join(".g-output");
    let mut paths = std::fs::read_dir(control_directory)
        .expect("control directory reads")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(".session.claim.json.") && name.strip_suffix(".tmp").is_some())
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

fn regular_file_snapshot(root: &Path) -> BTreeMap<PathBuf, (u64, i64, Vec<u8>)> {
    fn visit(root: &Path, directory: &Path, snapshot: &mut BTreeMap<PathBuf, (u64, i64, Vec<u8>)>) {
        for entry in std::fs::read_dir(directory).expect("snapshot directory reads") {
            let path = entry.expect("snapshot entry reads").path();
            if path.is_dir() {
                visit(root, &path, snapshot);
            } else {
                let metadata = path.metadata().expect("snapshot metadata reads");
                snapshot.insert(
                    path.strip_prefix(root).expect("snapshot path is relative").to_path_buf(),
                    (
                        metadata.len(),
                        timestamp_nanoseconds(metadata.mtime(), metadata.mtime_nsec()),
                        std::fs::read(&path).expect("snapshot file reads"),
                    ),
                );
            }
        }
    }

    let mut snapshot = BTreeMap::new();
    visit(root, root, &mut snapshot);
    snapshot
}

fn path_relative_to(base: &Path, target: &Path) -> PathBuf {
    let base_components = base.components().collect::<Vec<_>>();
    let target_components = target.components().collect::<Vec<_>>();
    let common_count = base_components
        .iter()
        .zip(&target_components)
        .take_while(|(base_component, target_component)| base_component == target_component)
        .count();
    let mut relative = PathBuf::new();
    for _ in &base_components[common_count..] {
        relative.push("..");
    }
    for component in &target_components[common_count..] {
        relative.push(component.as_os_str());
    }
    relative
}

#[test]
fn completed_attempt_uses_exact_layout_footer_receipt_and_terminal_ordering() {
    let directory = TestDirectory::new("completed-layout");
    let inputs = test_inputs(&directory);
    let chunk_ranges = vec![0..2, 2..4];
    let run_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Profile);
    let manager = initialize_manager(run_plan, &inputs, &chunk_ranges);
    let token = manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    let store = metadata_store(4);
    for chunk_range in chunk_ranges {
        write_chunk(&token, &store, chunk_range);
    }
    drop(token);

    let completed =
        manager.close_completed().expect("exact coverage closes").finish().expect("completed terminal publishes");
    assert!(completed.post_session_cleanup.is_none());
    assert_eq!(completed.completed_outputs.len(), 1);
    let run_directory = &completed.completed_outputs[0].run_directory;
    assert_eq!(run_directory.file_name().and_then(|name| name.to_str()), Some(OUTPUT_DIRECTORY_NAME));
    let output_root = directory.path.join("results");
    assert_owner_authority_released(&output_root);
    let attempt = attempt_identifier(&output_root, None);
    assert_eq!(run_directory.parent().and_then(Path::file_name).and_then(|name| name.to_str()), Some(attempt.as_str()));
    let manifest = read_json(&run_directory.join("run_manifest.json"));
    assert_eq!(manifest["status"], "completed");
    assert_eq!(manifest["execution_plan"]["resume_policy"], "lineage_receipts_exact_coverage");
    assert_eq!(manifest["committed_chunks"].as_array().map(Vec::len), Some(2));
    assert_eq!(manifest["committed_parts"].as_array().map(Vec::len), Some(1));
    assert!(run_directory.join("effective_config.toml").is_file());
    assert!(run_directory.join("output_stage_timings.json").is_file());

    let receipt_path = std::fs::read_dir(run_directory.join("commits"))
        .expect("receipt directory reads")
        .next()
        .expect("one receipt exists")
        .expect("receipt entry reads")
        .path();
    let receipt = read_json(&receipt_path);
    assert_eq!(receipt["footer"]["run_set_id"], manifest["run_set_id"]);
    assert_eq!(receipt["footer"]["attempt_id"], manifest["attempt_id"]);
    assert_eq!(receipt["footer"]["phenotype_name"], PHENOTYPE_NAME);
    assert!(receipt["part_sha256"].as_str().is_some_and(|digest| digest.len() == 64));
    let part_path =
        run_directory.join("parts").join(receipt["footer"]["part_file_name"].as_str().expect("part name exists"));
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(part_path).expect("part opens"))
        .expect("Parquet footer reads");
    let embedded_footer = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .and_then(|entries| entries.iter().find(|entry| entry.key == crate::schema::PART_BINDING_METADATA_KEY))
        .and_then(|entry| entry.value.as_deref())
        .expect("embedded part footer exists");
    assert_eq!(serde_json::from_str::<Value>(embedded_footer).expect("footer JSON parses"), receipt["footer"]);
    let batches = builder.build().expect("part reader builds").collect::<Result<Vec<_>, _>>().expect("part reads");
    assert_eq!(
        batches[0]
            .column_by_name("BETA")
            .expect("BETA exists")
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("BETA is Float32")
            .len(),
        4
    );
}

#[test]
fn identical_bgen_content_at_different_locators_has_one_execution_plan_identity() {
    let directory = TestDirectory::new("bgen-content-plan-identity");
    let inputs = test_inputs(&directory);
    let relocated_bgen =
        directory.write("relocated-input.bgen", &std::fs::read(&inputs.bgen).expect("original BGEN reads"));
    let relocated_inputs = TestInputs {
        bgen: relocated_bgen,
        sample: inputs.sample.clone(),
        phenotype: inputs.phenotype.clone(),
        prediction_list: inputs.prediction_list.clone(),
    };
    let original_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let relocated_plan = run_plan(&directory, &relocated_inputs, false, None, g_plan::TelemetryMode::Off);
    let original_header_input = header(&inputs, 2);
    let relocated_header_input = header(&relocated_inputs, 2);
    let mut original_cache = crate::manifest::ManifestFileFingerprintCache::default();
    let mut relocated_cache = crate::manifest::ManifestFileFingerprintCache::default();
    let original_header = crate::manifest::build_current_run_manifest_header_value_with_cache(
        &original_plan,
        &original_header_input,
        &mut original_cache,
    )
    .expect("original manifest header builds");
    let relocated_header = crate::manifest::build_current_run_manifest_header_value_with_cache(
        &relocated_plan,
        &relocated_header_input,
        &mut relocated_cache,
    )
    .expect("relocated manifest header builds");

    assert_ne!(original_plan.input.bgen_path, relocated_plan.input.bgen_path);
    assert_eq!(original_header["execution_plan_hash"], relocated_header["execution_plan_hash"]);
    assert_eq!(original_header["execution_plan"]["bgen"], relocated_header["execution_plan"]["bgen"]);
    let bgen =
        original_header["execution_plan"]["bgen"].as_object().expect("BGEN execution-plan authority is an object");
    assert_eq!(bgen.len(), 2);
    assert_eq!(
        bgen.get("content_sha256"),
        Some(&Value::String(bgen_content_fingerprint(&inputs.bgen).content_sha256.to_string()))
    );
    assert_eq!(bgen.get("byte_count"), Some(&Value::from(inputs.bgen.metadata().expect("BGEN metadata reads").len())));
}

#[test]
fn generated_manifest_uses_typed_sparse_pseudo_budget_policy_for_each_association_mode() {
    const POLICY_POINTER: &str = "/execution_plan/binary_correction_plan/approximate_firth_sparse_pseudo_budget_policy";

    let directory = TestDirectory::new("sparse-pseudo-budget-policy");
    let inputs = test_inputs(&directory);
    let header_input = header(&inputs, 2);
    let plans_and_expected_policies = [
        (
            run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off),
            Value::String("half_total_uncapped_by_dense_cap".to_string()),
        ),
        (
            with_association_and_correction(
                run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off),
                g_plan::AssociationMode::Regenie2Binary,
                g_plan::BinaryFallbackMethod::ScoreOnly,
            ),
            Value::Null,
        ),
        (
            with_association_and_correction(
                run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off),
                g_plan::AssociationMode::Regenie2Linear,
                g_plan::BinaryFallbackMethod::ScoreOnly,
            ),
            Value::Null,
        ),
    ];

    for (run_plan, expected_policy) in plans_and_expected_policies {
        let mut fingerprint_cache = crate::manifest::ManifestFileFingerprintCache::default();
        let manifest = crate::manifest::build_current_run_manifest_header_value_with_cache(
            &run_plan,
            &header_input,
            &mut fingerprint_cache,
        )
        .expect("typed schema-zero manifest header builds");
        assert_eq!(manifest["schema_version"], 0);
        assert_eq!(manifest["output_schema_version"], 0);
        assert_eq!(manifest.pointer(POLICY_POINTER), Some(&expected_policy));
        assert_eq!(
            manifest["execution_plan_hash"],
            Value::String(
                crate::manifest::build_manifest_value_sha256(&manifest["execution_plan"])
                    .expect("canonical execution plan hashes")
            )
        );
        let bgen = manifest["execution_plan"]["bgen"].as_object().expect("BGEN execution-plan authority is an object");
        assert_eq!(bgen.len(), 2);
        assert!(bgen.contains_key("content_sha256"));
        assert!(bgen.contains_key("byte_count"));
    }
}

fn assert_non_firth_association_implementation_contract(association_mode: g_plan::AssociationMode, label: &str) {
    let accepted_directory = TestDirectory::new(&format!("{label}-null-firth-accepted"));
    let accepted_inputs = test_inputs(&accepted_directory);
    let chunk_ranges = single_chunk_ranges(2);
    let accepted_plan = with_association_and_correction(
        run_plan(&accepted_directory, &accepted_inputs, false, None, g_plan::TelemetryMode::Off),
        association_mode,
        g_plan::BinaryFallbackMethod::ScoreOnly,
    );
    OutputManager::open(accepted_plan, "# non-Firth compatibility acceptance\n".to_string())
        .expect("non-Firth manager plans")
        .initialize(
            vec![header(&accepted_inputs, 2)],
            &chunk_ranges,
            false,
            test_non_firth_association_implementation(),
        )
        .expect("null Firth compatibility matches the non-Firth plan")
        .finish_interrupted("SIGTERM")
        .expect("non-Firth output publishes an interrupted terminal");
    let accepted_output_root = accepted_directory.path.join("results");
    let accepted_attempt_id = attempt_identifier(&accepted_output_root, None);
    let accepted_manifest_path = accepted_output_root
        .join("attempts")
        .join(accepted_attempt_id)
        .join(OUTPUT_DIRECTORY_NAME)
        .join("run_manifest.json");
    assert_eq!(
        read_json(&accepted_manifest_path)["runtime"]["association_implementation"]["firth_components"],
        Value::Null
    );
    assert_owner_authority_released(&accepted_output_root);

    let rejected_directory = TestDirectory::new(&format!("{label}-non-null-firth-rejected"));
    let rejected_inputs = test_inputs(&rejected_directory);
    let rejected_plan = with_association_and_correction(
        run_plan(&rejected_directory, &rejected_inputs, false, None, g_plan::TelemetryMode::Off),
        association_mode,
        g_plan::BinaryFallbackMethod::ScoreOnly,
    );
    let rejected_output_root = rejected_directory.path.join("results");
    let rejection_result = OutputManager::open(rejected_plan, "# non-Firth compatibility rejection\n".to_string())
        .expect("non-Firth rejection manager plans")
        .initialize(vec![header(&rejected_inputs, 2)], &chunk_ranges, false, test_association_implementation());
    let Err(rejection) = rejection_result else {
        panic!("non-null Firth compatibility must not match a non-Firth plan");
    };
    assert!(rejection.to_string().contains("does not match the planned association and correction mode"));
    assert_owner_authority_released(&rejected_output_root);
    assert!(
        !rejected_output_root.join(".g-output/genesis.json").exists(),
        "rejected activation must not publish lineage authority"
    );
}

#[test]
fn non_firth_plans_require_null_firth_implementation_state() {
    assert_non_firth_association_implementation_contract(g_plan::AssociationMode::Regenie2Binary, "score-only");
    assert_non_firth_association_implementation_contract(g_plan::AssociationMode::Regenie2Linear, "linear");
}

#[test]
fn positioned_unattested_bgen_can_create_fresh_output_but_cannot_authorize_resume() {
    let directory = TestDirectory::new("unattested-bgen-fresh-only");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let unattested_header = || {
        header_for_phenotype_with_evidence(
            2,
            PHENOTYPE_NAME,
            positioned_unattested_evidence(&inputs.bgen),
            g_plan::GpuGenotypeFormat::Packed8,
        )
    };
    OutputManager::open(
        run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off),
        "# unattested fresh output\n".to_string(),
    )
    .expect("fresh unattested output plans")
    .initialize(vec![unattested_header()], &chunk_ranges, false, test_association_implementation())
    .expect("fresh unattested output initializes")
    .finish_interrupted("SIGTERM")
    .expect("fresh unattested output publishes a terminal");

    let output_root = directory.path.join("results");
    let attempt_id = attempt_identifier(&output_root, None);
    let manifest_path =
        output_root.join("attempts").join(attempt_id).join(OUTPUT_DIRECTORY_NAME).join("run_manifest.json");
    let manifest = read_json(&manifest_path);
    let bgen =
        manifest["execution_plan"]["bgen"].as_object().expect("unattested BGEN execution-plan authority is an object");
    assert_eq!(bgen.len(), 2);
    assert_eq!(bgen.get("content_sha256"), Some(&Value::Null));
    assert_eq!(bgen.get("byte_count"), Some(&Value::from(inputs.bgen.metadata().expect("BGEN metadata reads").len())));

    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off);
    let agreement_error = OutputManager::open(Arc::clone(&resume_plan), "# unattested resume\n".to_string())
        .expect("unattested resume plans")
        .existing_output_resume_agreement()
        .expect_err("unattested manifest cannot provide resume agreement");
    assert!(matches!(
        agreement_error,
        crate::OutputError::ExistingOutputUnattestedBgenContent { manifest_path: error_path }
            if error_path == manifest_path
    ));

    let Err(claim_error) = OutputManager::open(resume_plan, "# unattested claim\n".to_string())
        .expect("unattested claim plans")
        .claim(vec![unattested_header()], &chunk_ranges, false)
    else {
        panic!("unattested existing output cannot be claimed for resume");
    };
    assert!(matches!(
        claim_error,
        crate::OutputError::ExistingOutputUnattestedBgenContent { manifest_path: error_path }
            if error_path == manifest_path
    ));
    assert_owner_authority_released(&output_root);
}

#[test]
fn current_multi_phenotype_bgen_agreement_is_validated_before_owner_acquisition() {
    for mismatch in [
        BgenAgreementMismatch::ContentSha256,
        BgenAgreementMismatch::ByteCount,
        BgenAgreementMismatch::GpuGenotypeFormat,
    ] {
        let directory = TestDirectory::new(&format!("current-bgen-agreement-{}", mismatch.label()));
        let inputs = test_inputs(&directory);
        let run_plan = two_phenotype_run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
        let mut headers = two_phenotype_headers(&inputs, 2);
        mismatch.apply_to_header(&mut headers[1]);
        let Err(error) = OutputManager::open(run_plan, "# disagreeing headers\n".to_string())
            .expect("disagreeing output plans")
            .claim(headers, &single_chunk_ranges(2), false)
        else {
            panic!("{} disagreement must fail before ownership", mismatch.label());
        };
        assert!(
            error.to_string().contains(mismatch.expected_error_fragment()),
            "{} mismatch returned unexpected error: {error}",
            mismatch.label()
        );
        assert!(!directory.path.join("results").exists(), "{} mismatch acquired output authority", mismatch.label());
    }
}

#[test]
fn current_bgen_must_match_existing_authority_before_owner_acquisition() {
    let directory = TestDirectory::new("current-existing-bgen-agreement");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    initialize_manager(run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off), &inputs, &chunk_ranges)
        .finish_interrupted("SIGTERM")
        .expect("initial output publishes terminal authority");
    let output_root = directory.path.join("results");
    let before_mismatch = regular_file_snapshot(&output_root);
    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off);

    for mismatch in [
        BgenAgreementMismatch::ContentSha256,
        BgenAgreementMismatch::ByteCount,
        BgenAgreementMismatch::GpuGenotypeFormat,
    ] {
        let mut current_header = header(&inputs, 2);
        mismatch.apply_to_header(&mut current_header);
        let Err(error) = OutputManager::open(
            Arc::clone(&resume_plan),
            format!("# current-existing {} mismatch\n", mismatch.label()),
        )
        .expect("resume output plans")
        .claim(vec![current_header], &chunk_ranges, false) else {
            panic!("{} mismatch must fail before ownership", mismatch.label());
        };
        assert!(
            error.to_string().contains(mismatch.expected_error_fragment()),
            "{} mismatch returned unexpected error: {error}",
            mismatch.label()
        );
        assert_eq!(
            regular_file_snapshot(&output_root),
            before_mismatch,
            "{} mismatch mutated output authority",
            mismatch.label()
        );
        assert_owner_authority_released(&output_root);
    }
}

#[test]
fn existing_multi_phenotype_manifests_require_exact_bgen_agreement() {
    for mismatch in [
        BgenAgreementMismatch::ContentSha256,
        BgenAgreementMismatch::ByteCount,
        BgenAgreementMismatch::GpuGenotypeFormat,
    ] {
        let directory = TestDirectory::new(&format!("existing-bgen-agreement-{}", mismatch.label()));
        let inputs = test_inputs(&directory);
        let chunk_ranges = single_chunk_ranges(2);
        let manager = OutputManager::open(
            two_phenotype_run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off),
            "# active two-phenotype output\n".to_string(),
        )
        .expect("two-phenotype output plans")
        .initialize(two_phenotype_headers(&inputs, 2), &chunk_ranges, false, test_association_implementation())
        .expect("two-phenotype output initializes");
        let output_root = directory.path.join("results");
        let attempt_id = attempt_identifier(&output_root, None);
        let second_manifest_path =
            output_root.join("attempts").join(&attempt_id).join(SECOND_OUTPUT_DIRECTORY_NAME).join("run_manifest.json");
        let genesis_path = output_root.join(".g-output/genesis.json");
        let original_manifest_bytes = std::fs::read(&second_manifest_path).expect("second manifest reads");
        let original_genesis_bytes = std::fs::read(&genesis_path).expect("genesis reads");
        let mut changed_manifest: Value =
            serde_json::from_slice(&original_manifest_bytes).expect("second manifest parses");
        mismatch.apply_to_manifest(&mut changed_manifest);
        let changed_execution_plan_hash = changed_manifest["execution_plan_hash"]
            .as_str()
            .expect("changed execution plan hash is a string")
            .to_string();
        let mut changed_genesis: Value = serde_json::from_slice(&original_genesis_bytes).expect("genesis parses");
        changed_genesis["phenotypes"][1]["execution_plan_sha256"] = Value::String(changed_execution_plan_hash);
        std::fs::write(
            &second_manifest_path,
            serde_json::to_vec(&changed_manifest).expect("changed manifest serializes"),
        )
        .expect("changed manifest writes");
        std::fs::write(&genesis_path, serde_json::to_vec(&changed_genesis).expect("changed genesis serializes"))
            .expect("changed genesis writes");

        let error = OutputManager::open(
            two_phenotype_run_plan(&directory, &inputs, true, Some(attempt_id), g_plan::TelemetryMode::Off),
            "# disagreeing existing manifests\n".to_string(),
        )
        .expect("existing output plans")
        .existing_output_resume_agreement()
        .expect_err("existing manifest disagreement is rejected");
        assert!(
            error.to_string().contains(mismatch.expected_error_fragment()),
            "{} mismatch returned unexpected error: {error}",
            mismatch.label()
        );

        std::fs::write(&second_manifest_path, original_manifest_bytes).expect("second manifest restores");
        std::fs::write(&genesis_path, original_genesis_bytes).expect("genesis restores");
        manager.abort("existing agreement test cleanup").expect("active output terminates");
        assert_owner_authority_released(&output_root);
    }
}

#[test]
fn existing_multi_phenotype_manifests_require_exact_association_implementation_agreement() {
    let directory = TestDirectory::new("existing-association-implementation-agreement");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let manager = OutputManager::open(
        two_phenotype_run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off),
        "# active two-phenotype output\n".to_string(),
    )
    .expect("two-phenotype output plans")
    .initialize(two_phenotype_headers(&inputs, 2), &chunk_ranges, false, test_association_implementation())
    .expect("two-phenotype output initializes");
    let output_root = directory.path.join("results");
    let attempt_id = attempt_identifier(&output_root, None);
    let first_manifest_path =
        output_root.join("attempts").join(&attempt_id).join(OUTPUT_DIRECTORY_NAME).join("run_manifest.json");
    let second_manifest_path =
        output_root.join("attempts").join(&attempt_id).join(SECOND_OUTPUT_DIRECTORY_NAME).join("run_manifest.json");
    let original_second_manifest_bytes = std::fs::read(&second_manifest_path).expect("second manifest reads");
    let mut changed_second_manifest: Value =
        serde_json::from_slice(&original_second_manifest_bytes).expect("second manifest parses");
    changed_second_manifest["runtime"]["association_implementation"]["jax_version"] =
        Value::String("different-jax".to_string());
    std::fs::write(
        &second_manifest_path,
        serde_json::to_vec_pretty(&changed_second_manifest).expect("changed manifest serializes"),
    )
    .expect("changed manifest writes");

    let error = OutputManager::open(
        two_phenotype_run_plan(&directory, &inputs, true, Some(attempt_id), g_plan::TelemetryMode::Off),
        "# disagreeing association implementations\n".to_string(),
    )
    .expect("existing output plans")
    .existing_output_resume_agreement()
    .expect_err("association implementation disagreement is rejected");
    assert!(matches!(
        error,
        crate::OutputError::ExistingOutputAssociationImplementationDisagreement {
            first_manifest_path: observed_first,
            conflicting_manifest_path: observed_second,
        } if observed_first == first_manifest_path && observed_second == second_manifest_path
    ));

    std::fs::write(&second_manifest_path, original_second_manifest_bytes).expect("second manifest restores");
    manager.abort("association agreement test cleanup").expect("active output terminates");
    assert_owner_authority_released(&output_root);
}

#[test]
fn activation_rejects_handler_only_mismatch_under_claim_before_publication() {
    let directory = TestDirectory::new("activation-handler-identity-mismatch");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    OutputManager::open(
        run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off),
        "# initial raw-CUDA implementation\n".to_string(),
    )
    .expect("initial output plans")
    .initialize(
        vec![header(&inputs, 2)],
        &chunk_ranges,
        true,
        test_raw_cuda_association_implementation_with_digests('a', 'c'),
    )
    .expect("initial raw-CUDA output initializes")
    .finish_interrupted("SIGTERM")
    .expect("initial output publishes interrupted terminal");
    let output_root = directory.path.join("results");
    let parent_attempt_id = attempt_identifier(&output_root, None);
    let lineage_paths = crate::persistence::lineage::OutputLineagePaths::new(&output_root);
    let parent_attempt =
        crate::persistence::identifier::AttemptIdentifier::parse(&parent_attempt_id).expect("parent attempt parses");
    let successor_path = lineage_paths.normal_successor_path(&parent_attempt);
    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off);
    let mut planned_manager =
        OutputManager::open(resume_plan, "# mismatched implementation resume\n".to_string()).expect("resume plans");
    let hint_control =
        planned_manager.install_manifest_hint_pause_for_test().expect("activation agreement pause installs");
    let claimed_manager = planned_manager
        .claim(vec![header(&inputs, 2)], &chunk_ranges, true)
        .expect("matching execution plan claims output");
    let claim_id = owner_claim_identifier(&output_root);
    let staging_attempt_id =
        lineage_paths.owner_staging_attempt(&claim_id).expect("owner staging reads").expect("owner staging exists");
    let staged_run_directory = lineage_paths.attempt_directory(&staging_attempt_id).join(OUTPUT_DIRECTORY_NAME);

    let activation_error = std::thread::scope(|scope| {
        let activation_thread = scope.spawn(move || {
            claimed_manager.activate_with_deferred_completed_noop_cleanup(
                test_raw_cuda_association_implementation_with_digests('b', 'c'),
            )
        });
        hint_control.wait_until_reached().expect("activation pauses after its fresh manifest reads");
        assert!(!successor_path.exists(), "mismatch check precedes successor authority");
        assert!(!staged_run_directory.exists(), "mismatch check precedes manifests and writers");
        hint_control.resume();
        match activation_thread.join().expect("activation thread does not panic") {
            Err(error) => error,
            Ok(_) => panic!("mismatched implementation must reject activation"),
        }
    });
    assert_eq!(hint_control.reach_count(), 1);
    assert_eq!(hint_control.final_inspect_count(), 1);
    let crate::OutputActivationFailureParts { source, rollback } = activation_error.into_parts();
    assert!(matches!(source, crate::OutputError::CurrentAssociationImplementationMismatch));
    assert!(!successor_path.exists(), "mismatch cannot publish successor authority");
    assert!(!staged_run_directory.exists(), "mismatch cannot publish writer state");
    let mut rollback = rollback.expect("unpublished mismatch retains rollback authority");
    rollback.abort_before_activation().expect("mismatched claim rolls back after diagnostics close");
    assert!(!lineage_paths.attempt_directory(&staging_attempt_id).exists());
    assert_owner_authority_released(&output_root);
}

#[test]
fn backend_projection_failure_can_reject_activation_without_publication() {
    let directory = TestDirectory::new("backend-projection-rejects-activation");
    let inputs = test_inputs(&directory);
    let output_root = directory.path.join("results");
    let claimed_manager = OutputManager::open(
        run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off),
        "# rejected\n".to_string(),
    )
    .expect("fresh output plans")
    .claim(vec![header(&inputs, 2)], &single_chunk_ranges(2), true)
    .expect("fresh output claims");
    let diagnostics_directory = claimed_manager.diagnostics_directory().expect("claim diagnostics exist").to_path_buf();
    let activation_error = claimed_manager.reject_activation(crate::OutputError::InvalidInput(
        "backend omitted required implementation state".to_string(),
    ));
    let crate::OutputActivationFailureParts { source, rollback } = activation_error.into_parts();
    assert!(source.to_string().contains("backend omitted required implementation state"));
    assert!(diagnostics_directory.is_dir());
    assert!(!output_root.join(".g-output/genesis.json").exists());
    assert!(
        std::fs::read_dir(output_root.join("attempts")).expect("attempt staging root reads").all(|entry| !entry
            .expect("attempt entry reads")
            .path()
            .join(OUTPUT_DIRECTORY_NAME)
            .exists()),
        "rejected activation must not create phenotype manifests or writers"
    );
    let mut rollback = rollback.expect("projection rejection retains rollback authority");
    rollback.abort_before_activation().expect("projection rejection rolls back after diagnostics close");
    assert!(!diagnostics_directory.exists());
    assert_owner_authority_released(&output_root);
}

#[test]
fn identical_raw_cuda_association_implementation_allows_resume_activation() {
    let directory = TestDirectory::new("identical-raw-cuda-association-implementation-resume");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let association_implementation = test_raw_cuda_association_implementation_with_digests('a', 'c');
    OutputManager::open(
        run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off),
        "# initial raw-CUDA implementation\n".to_string(),
    )
    .expect("initial output plans")
    .initialize(vec![header(&inputs, 2)], &chunk_ranges, true, association_implementation.clone())
    .expect("initial raw-CUDA output initializes")
    .finish_interrupted("SIGTERM")
    .expect("initial output publishes interrupted terminal");
    let output_root = directory.path.join("results");
    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off);
    let resumed_manager = OutputManager::open(resume_plan, "# identical implementation resume\n".to_string())
        .expect("resume plans")
        .claim(vec![header(&inputs, 2)], &chunk_ranges, false)
        .expect("matching execution plan claims output")
        .activate(association_implementation)
        .expect("identical implementation activates a successor");
    resumed_manager.abort("identical implementation test cleanup").expect("resumed output terminates");
    assert_owner_authority_released(&output_root);
}

#[test]
fn aggregate_resume_agreement_preserves_terminal_missing_manifest_rule() {
    let directory = TestDirectory::new("terminal-missing-manifest-agreement");
    let inputs = test_inputs(&directory);
    OutputManager::open(
        two_phenotype_run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off),
        "# terminal with two manifests\n".to_string(),
    )
    .expect("two-phenotype output plans")
    .initialize(two_phenotype_headers(&inputs, 2), &single_chunk_ranges(2), false, test_association_implementation())
    .expect("two-phenotype output initializes")
    .finish_interrupted("SIGTERM")
    .expect("two-phenotype terminal publishes");

    let output_root = directory.path.join("results");
    let attempt_id = attempt_identifier(&output_root, None);
    let missing_manifest_path =
        output_root.join("attempts").join(attempt_id).join(SECOND_OUTPUT_DIRECTORY_NAME).join("run_manifest.json");
    std::fs::remove_file(&missing_manifest_path).expect("terminal manifest is removed for the test");
    let error = OutputManager::open(
        two_phenotype_run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off),
        "# terminal missing one manifest\n".to_string(),
    )
    .expect("resume output plans")
    .existing_output_resume_agreement()
    .expect_err("terminal missing manifest is rejected");
    assert!(error.to_string().contains("Terminal output attempt is missing manifest"));
    assert!(error.to_string().contains(&missing_manifest_path.display().to_string()));
    assert_owner_authority_released(&output_root);
}

#[test]
fn activation_rollback_refuses_authority_that_appears_after_compatibility_rejection() {
    let directory = TestDirectory::new("compatibility-rejection-authority-race");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let output_root = directory.path.join("results");
    let claimed_manager = OutputManager::open(
        run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off),
        "# compatibility rejection authority race\n".to_string(),
    )
    .expect("manager plans")
    .claim(single_phenotype_headers(&inputs, &chunk_ranges), &chunk_ranges, false)
    .expect("manager claims");
    let diagnostics_directory = claimed_manager.diagnostics_directory().expect("claim diagnostics exist").to_path_buf();
    let activation_error = claimed_manager
        .activate_with_deferred_completed_noop_cleanup(test_non_firth_association_implementation())
        .err()
        .expect("planned approximate Firth rejects null Firth compatibility");
    let crate::OutputActivationFailureParts { source, rollback } = activation_error.into_parts();
    assert!(source.to_string().contains("does not match the planned association and correction mode"));
    let mut rollback = rollback.expect("compatibility rejection returns unpublished rollback authority");

    let genesis_path = output_root.join(".g-output/genesis.json");
    std::fs::write(&genesis_path, b"{}\n").expect("competing immutable authority appears");
    let rollback_error =
        rollback.abort_before_activation().expect_err("rollback must fail closed after immutable authority appears");
    assert!(rollback_error.to_string().contains("immutable authority appeared"));
    assert!(diagnostics_directory.is_dir());
    assert!(output_root.join(".g-output/session.claim.json").is_file());
}

#[test]
fn aggregate_resume_agreement_skips_missing_nonterminal_manifest() {
    let directory = TestDirectory::new("nonterminal-missing-manifest-agreement");
    let inputs = test_inputs(&directory);
    let manager = OutputManager::open(
        two_phenotype_run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off),
        "# active output with two manifests\n".to_string(),
    )
    .expect("two-phenotype output plans")
    .initialize(two_phenotype_headers(&inputs, 2), &single_chunk_ranges(2), false, test_association_implementation())
    .expect("two-phenotype output initializes");
    let output_root = directory.path.join("results");
    let attempt_id = attempt_identifier(&output_root, None);
    let second_manifest_path =
        output_root.join("attempts").join(&attempt_id).join(SECOND_OUTPUT_DIRECTORY_NAME).join("run_manifest.json");
    let second_manifest_bytes = std::fs::read(&second_manifest_path).expect("second manifest reads");
    std::fs::remove_file(&second_manifest_path).expect("nonterminal manifest is removed for the test");
    let agreement = OutputManager::open(
        two_phenotype_run_plan(&directory, &inputs, true, Some(attempt_id), g_plan::TelemetryMode::Off),
        "# active output missing one manifest\n".to_string(),
    )
    .expect("resume output plans")
    .existing_output_resume_agreement()
    .expect("missing nonterminal manifest is skipped");
    assert_eq!(agreement, Some(expected_resume_agreement(&inputs, g_plan::GpuGenotypeFormat::Packed8)));

    std::fs::write(&second_manifest_path, second_manifest_bytes).expect("second manifest restores");
    manager.abort("nonterminal missing manifest test cleanup").expect("active output terminates");
    assert_owner_authority_released(&output_root);
}

#[test]
fn existing_resume_agreement_requires_exact_immutable_lineage_bindings() {
    let directory = TestDirectory::new("gpu-format-authority-binding");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let manager = initialize_manager(initial_plan, &inputs, &chunk_ranges);
    let token = manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    write_chunk(&token, &metadata_store(2), 0..2);
    drop(token);
    assert_no_post_session_cleanup(
        manager.close_completed().expect("exact coverage closes").finish().expect("completed terminal publishes"),
    );

    let output_root = directory.path.join("results");
    let attempt_id = attempt_identifier(&output_root, None);
    let manifest_path =
        output_root.join("attempts").join(&attempt_id).join(OUTPUT_DIRECTORY_NAME).join("run_manifest.json");
    let original_manifest_bytes = std::fs::read(&manifest_path).expect("completed manifest reads");
    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off);
    let planned_manager =
        OutputManager::open(resume_plan, "# authority-bound GPU hint\n".to_string()).expect("resume plans");
    assert_eq!(
        planned_manager.existing_output_resume_agreement().expect("authority-bound agreement reads"),
        Some(expected_resume_agreement(&inputs, g_plan::GpuGenotypeFormat::Packed8))
    );

    for tamper in [
        ManifestAuthorityTamper::RunSet,
        ManifestAuthorityTamper::Attempt,
        ManifestAuthorityTamper::Phenotype,
        ManifestAuthorityTamper::OutputDirectory,
        ManifestAuthorityTamper::ChunkPlan,
        ManifestAuthorityTamper::ExecutionPlan,
    ] {
        let mut changed_manifest =
            serde_json::from_slice(&original_manifest_bytes).expect("original manifest JSON parses");
        tamper.apply(&mut changed_manifest);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&changed_manifest).expect("changed manifest serializes"),
        )
        .expect("changed manifest writes");
        let error = match planned_manager.existing_output_resume_agreement() {
            Ok(agreement) => panic!("{} tamper must be rejected, observed {agreement:?}", tamper.label()),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains(tamper.expected_error()),
            "{} tamper returned unexpected error: {error}",
            tamper.label()
        );
    }

    let mut changed_terminal_bytes = original_manifest_bytes.clone();
    changed_terminal_bytes.push(b'\n');
    std::fs::write(&manifest_path, changed_terminal_bytes).expect("raw terminal manifest tamper writes");
    let changed_terminal_error = planned_manager
        .existing_output_resume_agreement()
        .expect_err("semantically identical bytes outside the immutable terminal binding are rejected");
    assert!(changed_terminal_error.to_string().contains("immutable terminal manifest"));
}

#[test]
fn finalized_noncompleted_terminals_provide_authority_bound_gpu_format_hints() {
    for (status, interruption_signal) in [("interrupted", Some("SIGTERM")), ("failed", None)] {
        let directory = TestDirectory::new(&format!("gpu-format-finalized-{status}"));
        let inputs = test_inputs(&directory);
        let manager = initialize_manager(
            run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off),
            &inputs,
            &single_chunk_ranges(2),
        );
        match interruption_signal {
            Some(signal_name) => {
                manager.finish_interrupted(signal_name).expect("interrupted terminal publishes");
            }
            None => {
                manager.abort("injected failure").expect("failed terminal publishes");
            }
        }

        let hint_manager = OutputManager::open(
            run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off),
            format!("# finalized {status} GPU hint\n"),
        )
        .expect("terminal resume plans");
        assert_eq!(
            hint_manager.existing_output_resume_agreement().expect("terminal agreement validates"),
            Some(expected_resume_agreement(&inputs, g_plan::GpuGenotypeFormat::Packed8)),
            "{status} terminal"
        );
    }
}

#[test]
fn swapped_two_phenotype_execution_plans_fail_before_gpu_format_hint() {
    let directory = TestDirectory::new("gpu-format-swapped-phenotype-plans");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let manager = OutputManager::open(
        two_phenotype_run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off),
        "# two phenotype terminal\n".to_string(),
    )
    .expect("two phenotype manager plans")
    .initialize(two_phenotype_headers(&inputs, 2), &chunk_ranges, false, test_association_implementation())
    .expect("two phenotype manager initializes");
    manager.finish_interrupted("SIGTERM").expect("two phenotype terminal publishes");

    let output_root = directory.path.join("results");
    let attempt_id = attempt_identifier(&output_root, None);
    let attempt_directory = output_root.join("attempts").join(attempt_id);
    let first_manifest_path = attempt_directory.join(OUTPUT_DIRECTORY_NAME).join("run_manifest.json");
    let second_manifest_path = attempt_directory.join(SECOND_OUTPUT_DIRECTORY_NAME).join("run_manifest.json");
    let mut first_manifest = read_json(&first_manifest_path);
    let mut second_manifest = read_json(&second_manifest_path);
    std::mem::swap(&mut first_manifest["execution_plan"], &mut second_manifest["execution_plan"]);
    std::mem::swap(&mut first_manifest["execution_plan_hash"], &mut second_manifest["execution_plan_hash"]);
    std::fs::write(&first_manifest_path, serde_json::to_vec(&first_manifest).expect("first manifest serializes"))
        .expect("first manifest writes");
    std::fs::write(&second_manifest_path, serde_json::to_vec(&second_manifest).expect("second manifest serializes"))
        .expect("second manifest writes");

    let hint_manager = OutputManager::open(
        two_phenotype_run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off),
        "# swapped phenotype GPU hint\n".to_string(),
    )
    .expect("swapped phenotype resume plans");
    let error = hint_manager
        .existing_output_resume_agreement()
        .expect_err("swapped execution plan is rejected before providing an agreement");
    assert!(error.to_string().contains("execution plan phenotype does not match its manifest phenotype"));
}

#[test]
fn interrupted_attempt_resumes_into_successor_with_verified_hardlink_reuse() {
    let directory = TestDirectory::new("terminal-resume");
    let inputs = test_inputs(&directory);
    let chunk_ranges = vec![0..2, 2..4];
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let manager = initialize_manager(initial_plan, &inputs, &chunk_ranges);
    let token = manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    let store = metadata_store(4);
    write_chunk(&token, &store, 0..2);
    drop(token);
    manager.finish_interrupted("SIGTERM").expect("interrupted terminal publishes");

    let output_root = directory.path.join("results");
    assert_owner_authority_released(&output_root);
    let first_attempt = attempt_identifier(&output_root, None);
    let first_parts = output_root.join("attempts").join(&first_attempt).join(OUTPUT_DIRECTORY_NAME).join("parts");
    let first_part = std::fs::read_dir(&first_parts)
        .expect("first parts read")
        .find_map(|entry| {
            let path = entry.expect("part entry reads").path();
            path.extension().is_some_and(|extension| extension == "parquet").then_some(path)
        })
        .expect("first part exists");

    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off);
    let resume_manager = initialize_manager(resume_plan, &inputs, &chunk_ranges);
    let token =
        resume_manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("resume token builds");
    assert_eq!(token.committed_chunk_identifier_sets()[0].as_ref(), &std::collections::BTreeSet::from([0]));
    write_chunk(&token, &store, 2..4);
    drop(token);
    let completed = resume_manager
        .close_completed()
        .expect("resumed exact coverage closes")
        .finish()
        .expect("resumed completion publishes");
    assert!(completed.post_session_cleanup.is_none());

    let second_attempt = attempt_identifier(&output_root, Some(&first_attempt));
    let reused_part =
        completed.completed_outputs[0].parts_directory.join(first_part.file_name().expect("part file name exists"));
    assert_eq!(
        first_part.metadata().expect("source metadata reads").ino(),
        reused_part.metadata().expect("reused metadata reads").ino()
    );
    assert_ne!(first_attempt, second_attempt);
    assert_eq!(
        std::fs::read_dir(&completed.completed_outputs[0].parts_directory)
            .expect("completed parts read")
            .filter_map(Result::ok)
            .filter(|entry| { entry.path().extension().is_some_and(|extension| extension == "parquet") })
            .count(),
        2
    );
}

#[test]
fn completed_resume_reverifies_payload_without_mutating_data() {
    let directory = TestDirectory::new("completed-noop");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let manager = initialize_manager(initial_plan, &inputs, &chunk_ranges);
    let token = manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    write_chunk(&token, &metadata_store(2), 0..2);
    drop(token);
    assert_no_post_session_cleanup(
        manager
            .close_completed()
            .expect("initial exact coverage closes")
            .finish()
            .expect("initial completion publishes"),
    );

    let output_root = directory.path.join("results");
    let before = regular_file_snapshot(&output_root);
    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off);
    let resume_manager = initialize_manager(resume_plan, &inputs, &chunk_ranges);
    let token =
        resume_manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("completed token builds");
    assert!(token.is_read_only());
    drop(token);
    assert_no_post_session_cleanup(
        resume_manager
            .close_completed()
            .expect("completed attempt reverifies")
            .finish()
            .expect("completed no-op finishes"),
    );
    let after = regular_file_snapshot(&output_root);
    for (path, expected) in &before {
        assert_eq!(after.get(path), Some(expected), "completed payload changed at '{}'", path.display());
    }
    let new_paths = after.keys().filter(|path| !before.contains_key(*path)).collect::<Vec<_>>();
    assert!(
        !new_paths.is_empty() && new_paths.iter().all(|path| path.starts_with(".g-output/owner-transitions")),
        "completed resume may append only owner-authority transitions, got {new_paths:?}"
    );
    assert_owner_authority_released(&output_root);
}

#[test]
fn planning_does_not_create_the_output_root() {
    let directory = TestDirectory::new("read-only-plan");
    let inputs = test_inputs(&directory);
    let plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let output_root = PathBuf::from(&plan.output.output_run_root);
    let manager = OutputManager::open(plan, "# test configuration\n".to_string()).expect("manager plans");
    assert!(!output_root.exists());
    drop(manager);
    assert!(!output_root.exists());
}

#[test]
fn pre_release_output_contract_versions_remain_zero() {
    assert_eq!(crate::manifest::RUN_MANIFEST_SCHEMA_VERSION, 0);
    assert_eq!(crate::manifest::OUTPUT_SCHEMA_VERSION, 0);
    assert_eq!(crate::persistence::attempt::ATTEMPT_MANIFEST_SCHEMA_VERSION, 0);
    assert_eq!(crate::persistence::lineage::LINEAGE_SCHEMA_VERSION, 0);
    assert_eq!(crate::persistence::receipt::PART_RECORD_SCHEMA_VERSION, 0);
}

#[test]
fn planning_rejects_cli_line_separators_without_mutation() {
    for separator in ['\n', '\u{2028}', '\u{2029}'] {
        let directory = TestDirectory::new("line-separator");
        let inputs = test_inputs(&directory);
        let mut root_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
        Arc::get_mut(&mut root_plan).expect("test plan has one owner").output.output_run_root =
            format!("{}{}suffix", directory.path.join("results").display(), separator);
        assert!(OutputManager::open(root_plan, "# test configuration\n".to_string()).is_err());
        assert!(!directory.path.join("results").exists());

        let mut phenotype_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
        Arc::get_mut(&mut phenotype_plan).expect("test plan has one owner").phenotype_runs[0].output_directory_name =
            format!("trait{separator}split");
        assert!(OutputManager::open(phenotype_plan, "# test configuration\n".to_string()).is_err());
        assert!(!directory.path.join("results").exists());
    }
}

#[test]
fn relative_output_root_returns_absolute_completed_paths() {
    let directory = TestDirectory::new("relative-root");
    let inputs = test_inputs(&directory);
    let mut plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let expected_output_root = directory.path.join("results");
    Arc::get_mut(&mut plan).expect("test plan has one owner").output.output_run_root =
        path_relative_to(&std::env::current_dir().expect("current directory resolves"), &expected_output_root)
            .display()
            .to_string();
    let manager = initialize_manager(plan, &inputs, &single_chunk_ranges(2));
    let token = manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    write_chunk(&token, &metadata_store(2), 0..2);
    drop(token);
    let completed = manager.close_completed().expect("relative output closes").finish().expect("output completes");
    assert!(completed.post_session_cleanup.is_none());
    assert_eq!(completed.completed_outputs.len(), 1);
    assert!(completed.completed_outputs[0].run_directory.is_absolute());
    assert!(completed.completed_outputs[0].parts_directory.is_absolute());
    assert!(completed.completed_outputs[0].run_directory.starts_with(&expected_output_root));
    assert_eq!(
        completed.completed_outputs[0].parts_directory,
        completed.completed_outputs[0].run_directory.join("parts")
    );
}

#[test]
fn controlled_initialization_failures_never_leave_an_owner_claim() {
    for failure_stage in ["writer_settings", "chunk_plan", "headers"] {
        let directory = TestDirectory::new(failure_stage);
        let inputs = test_inputs(&directory);
        let mut plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
        if failure_stage == "writer_settings" {
            Arc::get_mut(&mut plan).expect("test plan has one owner").output.writer_thread_count = 0;
        }
        let output_root = directory.path.join("results");
        let manager =
            OutputManager::open(plan, "# test configuration\n".to_string()).expect("failure case manager plans");
        let initialization_result = match failure_stage {
            "writer_settings" => manager.initialize(
                vec![header(&inputs, 2)],
                &single_chunk_ranges(2),
                false,
                test_association_implementation(),
            ),
            "chunk_plan" => manager.initialize(
                vec![header(&inputs, 2)],
                &single_chunk_ranges_from(1, 2),
                false,
                test_association_implementation(),
            ),
            "headers" => {
                manager.initialize(Vec::new(), &single_chunk_ranges(2), false, test_association_implementation())
            }
            _ => unreachable!("failure stages are exhaustive"),
        };
        assert!(initialization_result.is_err());
        assert!(!output_root.exists(), "pre-claim validation must remain read-only");
    }

    for failure_point in ["after_owner_staging_intent", "after_claim_diagnostics_creation"] {
        let directory = TestDirectory::new(failure_point);
        let inputs = test_inputs(&directory);
        let plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
        let output_root = directory.path.join("results");
        let failure_guard = crate::manager::install_initialization_failure_for_test(failure_point);
        let error = OutputManager::open(plan, "# test configuration\n".to_string())
            .expect("failure case manager plans")
            .initialize(vec![header(&inputs, 2)], &single_chunk_ranges(2), false, test_association_implementation())
            .err()
            .expect("configured staging failure fires");
        drop(failure_guard);
        assert!(error.to_string().contains(failure_point));
        assert_owner_authority_released(&output_root);
        assert!(!output_root.join(".g-output/genesis.json").exists());
        assert!(
            !output_root.join("attempts").exists()
                || std::fs::read_dir(output_root.join("attempts")).expect("attempt directory reads").next().is_none()
        );
        assert!(
            std::fs::read_dir(output_root.join(".g-output/owner-staging"))
                .expect("owner staging directory reads")
                .next()
                .is_none()
        );
    }
}

#[test]
fn claim_staging_failure_retains_owner_when_guarded_authority_appears_before_cleanup() {
    let directory = TestDirectory::new("claim-staging-authority-race");
    let inputs = test_inputs(&directory);
    let plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let output_root = directory.path.join("results");
    let failure_guard = crate::manager::install_initialization_failure_for_test("after_claim_diagnostics_creation");
    let authority_race_guard =
        crate::manager::install_initialization_cleanup_failure_for_test("before_failed_claim_staging_cleanup");
    let error = OutputManager::open(plan, "# claim staging authority race\n".to_string())
        .expect("race manager plans")
        .initialize(vec![header(&inputs, 2)], &single_chunk_ranges(2), false, test_association_implementation())
        .err()
        .expect("claim staging initialization failure fires");
    drop(authority_race_guard);
    drop(failure_guard);

    let error_text = error.to_string();
    assert!(error_text.contains("after_claim_diagnostics_creation"));
    assert!(error_text.contains("immutable authority appeared"));
    assert!(error_text.contains("remains authoritative"));
    let claim_id = owner_claim_identifier(&output_root);
    let lineage_paths = crate::persistence::lineage::OutputLineagePaths::new(&output_root);
    let staging_attempt_id =
        lineage_paths.owner_staging_attempt(&claim_id).expect("staging intent reads").expect("staging intent remains");
    assert!(lineage_paths.genesis_path.is_file());
    assert!(lineage_paths.attempt_directory(&staging_attempt_id).join("diagnostics").join(&claim_id).is_dir());
    let surviving_owner_error =
        lineage_paths.reject_surviving_owner_claim().expect_err("cleanup refusal must retain exact owner authority");
    assert_surviving_owner_claim(surviving_owner_error, &output_root);
}

#[test]
fn postclaim_initialization_failures_publish_failed_terminals_and_release() {
    for failure_point in ["after_owner_claim", "after_attempt_claim", "after_attempt_preparation", "after_writer_start"]
    {
        let directory = TestDirectory::new(failure_point);
        let inputs = test_inputs(&directory);
        let plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
        let output_root = directory.path.join("results");
        let failure_guard = crate::manager::install_initialization_failure_for_test(failure_point);
        let manager =
            OutputManager::open(plan, "# test configuration\n".to_string()).expect("failure case manager plans");
        let error = manager
            .initialize(vec![header(&inputs, 2)], &single_chunk_ranges(2), false, test_association_implementation())
            .err()
            .expect("configured initialization stage fails");
        drop(failure_guard);
        assert!(error.to_string().contains("Injected output initialization failure"));
        assert_owner_authority_released(&output_root);
        if failure_point == "after_owner_claim" {
            assert!(!output_root.join(".g-output/genesis.json").exists());
        } else {
            let attempt = attempt_identifier(&output_root, None);
            assert_eq!(
                read_json(
                    &output_root.join("attempts").join(&attempt).join(OUTPUT_DIRECTORY_NAME).join("run_manifest.json")
                )["status"],
                "failed"
            );
            assert!(output_root.join(".g-output/terminal-finalizations").join(format!("{attempt}.json")).is_file());
        }
    }
}

#[test]
fn initialization_abort_cleanup_failure_is_recorded_without_retaining_the_claim() {
    let directory = TestDirectory::new("initialization-abort-diagnostic");
    let inputs = test_inputs(&directory);
    let plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let output_root = directory.path.join("results");
    let failure_guard = crate::manager::install_initialization_failure_for_test("after_writer_start");
    let cleanup_failure_guard = crate::manager::install_initialization_cleanup_failure_for_test("after_writer_abort");
    let manager = OutputManager::open(plan, "# test configuration\n".to_string()).expect("failure case manager plans");
    let error = manager
        .initialize(vec![header(&inputs, 2)], &single_chunk_ranges(2), false, test_association_implementation())
        .err()
        .expect("configured initialization and cleanup stages fail");
    drop(cleanup_failure_guard);
    drop(failure_guard);
    assert!(error.to_string().contains("after_writer_start"));
    assert!(!error.to_string().contains("claim was retained"));
    assert_owner_authority_released(&output_root);
    let attempt = attempt_identifier(&output_root, None);
    let manifest =
        read_json(&output_root.join("attempts").join(attempt).join(OUTPUT_DIRECTORY_NAME).join("run_manifest.json"));
    assert_eq!(manifest["status"], "failed");
    assert!(manifest["failure_reason"].as_str().is_some_and(|reason| reason.contains("writer cleanup also reported")));
}

#[test]
fn initialization_release_conflict_reports_the_surviving_owner() {
    let directory = TestDirectory::new("owner-claim-release-conflict");
    let inputs = test_inputs(&directory);
    let plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let output_root = directory.path.join("results");
    let failure_guard = crate::manager::install_initialization_failure_for_test("after_owner_claim_release_conflict");
    let manager = OutputManager::open(plan, "# test configuration\n".to_string()).expect("failure case manager plans");
    let error = manager
        .initialize(vec![header(&inputs, 2)], &single_chunk_ranges(2), false, test_association_implementation())
        .err()
        .expect("claim-release conflict fails initialization");
    drop(failure_guard);
    let error_text = error.to_string();
    assert!(error_text.contains("Injected output initialization failure"));
    assert!(error_text.contains("survives from process"));
    assert!(error_text.contains("fence the recorded owner"));
    assert!(output_root.join(".g-output/session.claim.json").is_file());
}

#[test]
fn failed_multi_phenotype_preactivation_allows_a_fresh_retry() {
    let directory = TestDirectory::new("multi-phenotype-preactivation-retry");
    let inputs = test_inputs(&directory);
    let mut plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    Arc::get_mut(&mut plan).expect("test plan has one owner").phenotype_runs.push(g_plan::PhenotypeRunPlan {
        phenotype_name: SECOND_PHENOTYPE_NAME.to_string(),
        output_directory_name: SECOND_OUTPUT_DIRECTORY_NAME.to_string(),
    });
    let output_root = directory.path.join("results");
    let initialization_error = OutputManager::open(Arc::clone(&plan), "# incomplete initialization\n".to_string())
        .expect("multi-phenotype manager plans")
        .initialize(vec![header(&inputs, 2)], &single_chunk_ranges(2), false, test_association_implementation())
        .err()
        .expect("missing second phenotype header rejects claim");
    assert!(initialization_error.to_string().contains("do not cover planned phenotypes exactly"));
    assert!(!output_root.exists(), "invalid headers fail before output owner acquisition");

    let retry_manager = OutputManager::open(plan, "# complete initialization\n".to_string())
        .expect("fresh multi-phenotype retry plans")
        .initialize(
            vec![header(&inputs, 2), header_for_phenotype(&inputs, 2, SECOND_PHENOTYPE_NAME)],
            &single_chunk_ranges(2),
            false,
            test_association_implementation(),
        )
        .expect("fresh multi-phenotype retry activates");
    let attempt = attempt_identifier(&output_root, None);
    for output_directory_name in [OUTPUT_DIRECTORY_NAME, SECOND_OUTPUT_DIRECTORY_NAME] {
        let manifest = read_json(
            &output_root.join("attempts").join(&attempt).join(output_directory_name).join("run_manifest.json"),
        );
        assert_eq!(manifest["status"], "running");
    }
    retry_manager.abort("multi-phenotype fresh retry test").expect("fresh retry terminates");
    assert_owner_authority_released(&output_root);
}

#[test]
fn preactivation_failure_retains_ownership_until_explicit_rollback() {
    let directory = TestDirectory::new("preactivation-deferred-rollback");
    let inputs = test_inputs(&directory);
    let mut plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Profile);
    Arc::get_mut(&mut plan).expect("test plan has one owner").phenotype_runs.push(g_plan::PhenotypeRunPlan {
        phenotype_name: SECOND_PHENOTYPE_NAME.to_string(),
        output_directory_name: SECOND_OUTPUT_DIRECTORY_NAME.to_string(),
    });
    let manager = OutputManager::open(Arc::clone(&plan), "# deferred rollback\n".to_string())
        .expect("manager plans")
        .claim(two_phenotype_headers(&inputs, 2), &single_chunk_ranges(2), true)
        .expect("manager claims");
    let diagnostics_directory = manager.diagnostics_directory().expect("claim diagnostics exist").to_path_buf();
    let failure_guard = crate::manager::install_initialization_failure_for_test("after_owner_claim");
    let activation_error = manager
        .activate_with_deferred_completed_noop_cleanup(test_association_implementation())
        .err()
        .expect("prepublication activation failure is injected");
    drop(failure_guard);
    let crate::OutputActivationFailureParts { source, rollback } = activation_error.into_parts();
    assert!(source.to_string().contains("after_owner_claim"));
    assert!(diagnostics_directory.is_dir());

    let output_root = directory.path.join("results");
    let contender_error = claim_error(
        Arc::clone(&plan),
        &inputs,
        &single_chunk_ranges(2),
        "preactivation owner must survive until diagnostics close",
    );
    assert_surviving_owner_claim(contender_error, &output_root);

    rollback
        .expect("unpublished activation returns rollback authority")
        .abort_before_activation()
        .expect("post-session rollback cleans staging and releases");
    assert!(!diagnostics_directory.exists());
    assert_owner_authority_released(&output_root);

    OutputManager::open(plan, "# retry after rollback\n".to_string())
        .expect("retry manager plans")
        .claim(two_phenotype_headers(&inputs, 2), &single_chunk_ranges(2), false)
        .expect("retry claims after rollback")
        .abort_before_activation()
        .expect("retry staging cleans and releases");
}

#[test]
fn preactivation_rollback_retries_after_transient_staging_cleanup_failure() {
    let directory = TestDirectory::new("preactivation-rollback-cleanup-retry");
    let inputs = test_inputs(&directory);
    let mut plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Profile);
    Arc::get_mut(&mut plan).expect("test plan has one owner").phenotype_runs.push(g_plan::PhenotypeRunPlan {
        phenotype_name: SECOND_PHENOTYPE_NAME.to_string(),
        output_directory_name: SECOND_OUTPUT_DIRECTORY_NAME.to_string(),
    });
    let output_root = directory.path.join("results");
    let lineage_paths = crate::persistence::lineage::OutputLineagePaths::new(&output_root);
    let manager = OutputManager::open(Arc::clone(&plan), "# retryable deferred rollback\n".to_string())
        .expect("manager plans")
        .claim(two_phenotype_headers(&inputs, 2), &single_chunk_ranges(2), true)
        .expect("manager claims");
    let claim_id = owner_claim_identifier(&output_root);
    let staging_attempt_id =
        lineage_paths.owner_staging_attempt(&claim_id).expect("staging intent reads").expect("staging intent exists");
    let activation_failure_guard = crate::manager::install_initialization_failure_for_test("after_owner_claim");
    let Err(activation_error) =
        manager.activate_with_deferred_completed_noop_cleanup(test_association_implementation())
    else {
        panic!("prepublication activation failure must be injected");
    };
    drop(activation_failure_guard);
    let crate::OutputActivationFailureParts { source, rollback } = activation_error.into_parts();
    assert!(source.to_string().contains("after_owner_claim"));
    let mut rollback = rollback.expect("unpublished activation returns rollback authority");

    let failure_guard = crate::manager::install_initialization_cleanup_failure_for_test("after_claim_staging_removal");
    let cleanup_error =
        rollback.abort_before_activation().expect_err("transient staging cleanup failure retains rollback authority");
    drop(failure_guard);
    assert!(cleanup_error.to_string().contains("after_claim_staging_removal"));
    assert!(cleanup_error.to_string().contains(&claim_id));
    assert!(!lineage_paths.attempt_directory(&staging_attempt_id).exists());
    assert_eq!(
        lineage_paths.owner_staging_attempt(&claim_id).expect("staging intent remains readable"),
        Some(staging_attempt_id)
    );
    assert_surviving_owner_claim(
        claim_error(plan, &inputs, &single_chunk_ranges(2), "failed rollback retains exact authority"),
        &output_root,
    );

    rollback.abort_before_activation().expect("same rollback authority retries cleanup and release");
    rollback.abort_before_activation().expect("completed rollback is idempotent");
    assert_eq!(lineage_paths.owner_staging_attempt(&claim_id).expect("staging intent absence reads"), None);
    assert_owner_authority_released(&output_root);
}

#[test]
fn dropped_deferred_rollback_fails_closed_until_exactly_fenced() {
    let directory = TestDirectory::new("dropped-deferred-rollback");
    let inputs = test_inputs(&directory);
    let mut plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Profile);
    Arc::get_mut(&mut plan).expect("test plan has one owner").phenotype_runs.push(g_plan::PhenotypeRunPlan {
        phenotype_name: SECOND_PHENOTYPE_NAME.to_string(),
        output_directory_name: SECOND_OUTPUT_DIRECTORY_NAME.to_string(),
    });
    let manager = OutputManager::open(Arc::clone(&plan), "# dropped deferred rollback\n".to_string())
        .expect("manager plans")
        .claim(two_phenotype_headers(&inputs, 2), &single_chunk_ranges(2), true)
        .expect("manager claims");
    let diagnostics_directory = manager.diagnostics_directory().expect("claim diagnostics exist").to_path_buf();
    let failure_guard = crate::manager::install_initialization_failure_for_test("after_owner_claim");
    let activation_error = manager
        .activate_with_deferred_completed_noop_cleanup(test_association_implementation())
        .err()
        .expect("prepublication activation failure is injected");
    drop(failure_guard);
    let crate::OutputActivationFailureParts { source, rollback } = activation_error.into_parts();
    assert!(source.to_string().contains("after_owner_claim"));
    drop(rollback);

    let output_root = directory.path.join("results");
    assert!(diagnostics_directory.is_dir(), "dropped rollback retains claim-scoped diagnostics");
    let active_claim_id = owner_claim_identifier(&output_root);
    let contender_error = claim_error(
        Arc::clone(&plan),
        &inputs,
        &single_chunk_ranges(2),
        "dropped rollback must leave a surviving claim",
    );
    assert_surviving_owner_claim(contender_error, &output_root);

    let fenced_plan = authorize_fenced_owner_claim(plan, active_claim_id);
    let fenced_manager = OutputManager::open(fenced_plan, "# fenced dropped rollback\n".to_string())
        .expect("fenced manager plans")
        .claim(two_phenotype_headers(&inputs, 2), &single_chunk_ranges(2), false)
        .expect("exact fence takes over the dropped rollback");
    assert!(!diagnostics_directory.exists());
    fenced_manager.abort_before_activation().expect("fenced manager staging cleans and releases");
    assert_owner_authority_released(&output_root);
}

#[test]
fn postpublication_activation_failure_terminalizes_without_rollback_authority() {
    let directory = TestDirectory::new("postpublication-activation-failure");
    let inputs = test_inputs(&directory);
    let plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let output_root = directory.path.join("results");
    let manager = OutputManager::open(plan, "# published activation failure\n".to_string())
        .expect("manager plans")
        .claim(vec![header(&inputs, 2)], &single_chunk_ranges(2), false)
        .expect("manager claims");
    let failure_guard = crate::manager::install_initialization_failure_for_test("after_attempt_claim");
    let activation_error = manager
        .activate_with_deferred_completed_noop_cleanup(test_association_implementation())
        .err()
        .expect("postpublication failure is injected");
    drop(failure_guard);
    let crate::OutputActivationFailureParts { source, rollback } = activation_error.into_parts();
    assert!(source.to_string().contains("after_attempt_claim"));
    assert!(rollback.is_none(), "published attempt failure must not expose preactivation rollback");
    assert_owner_authority_released(&output_root);
    let attempt = attempt_identifier(&output_root, None);
    let manifest =
        read_json(&output_root.join("attempts").join(attempt).join(OUTPUT_DIRECTORY_NAME).join("run_manifest.json"));
    assert_eq!(manifest["status"], "failed");
}

#[test]
fn prelink_genesis_failure_returns_rollback_and_rechecks_before_deletion() {
    for authority_appears_before_rollback in [false, true] {
        let directory = TestDirectory::new(if authority_appears_before_rollback {
            "genesis-prelink-rollback-race"
        } else {
            "genesis-prelink-rollback"
        });
        let inputs = test_inputs(&directory);
        let plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Profile);
        let output_root = directory.path.join("results");
        let genesis_path = output_root.join(".g-output/genesis.json");
        let manager = OutputManager::open(plan, "# prelink failure\n".to_string())
            .expect("manager plans")
            .claim(vec![header(&inputs, 2)], &single_chunk_ranges(2), true)
            .expect("manager claims");
        let diagnostics_directory = manager.diagnostics_directory().expect("claim diagnostics exist").to_path_buf();
        let failure_guard =
            crate::persistence::io::install_immutable_publication_file_sync_failure_for_test(genesis_path.clone());
        let activation_error = manager
            .activate_with_deferred_completed_noop_cleanup(test_association_implementation())
            .err()
            .expect("prelink synchronization failure rejects activation");
        drop(failure_guard);
        let crate::OutputActivationFailureParts { source, rollback } = activation_error.into_parts();
        assert!(source.to_string().contains("file synchronization failure"));
        let mut rollback = rollback.expect("definitely absent authority returns rollback");
        assert!(!genesis_path.exists());

        if authority_appears_before_rollback {
            std::fs::write(&genesis_path, b"{}\n").expect("racing authority target appears");
            let rollback_error = rollback
                .abort_before_activation()
                .expect_err("rollback refuses to remove staging after authority appears");
            assert!(rollback_error.to_string().contains("immutable authority appeared"));
            assert!(diagnostics_directory.is_dir());
            assert!(output_root.join(".g-output/session.claim.json").is_file());
        } else {
            rollback.abort_before_activation().expect("absent authority rollback cleans and releases");
            assert!(!diagnostics_directory.exists());
            assert_owner_authority_released(&output_root);
        }
    }
}

#[test]
fn visible_genesis_after_durability_error_never_exposes_rollback() {
    let directory = TestDirectory::new("genesis-visible-durability-error");
    let inputs = test_inputs(&directory);
    let plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Profile);
    let output_root = directory.path.join("results");
    let lineage_paths = crate::persistence::lineage::OutputLineagePaths::new(&output_root);
    let manager = OutputManager::open(Arc::clone(&plan), "# visible genesis failure\n".to_string())
        .expect("manager plans")
        .claim(vec![header(&inputs, 2)], &single_chunk_ranges(2), true)
        .expect("manager claims");
    let claim_id = owner_claim_identifier(&output_root);
    let staged_attempt_id =
        lineage_paths.owner_staging_attempt(&claim_id).expect("owner staging reads").expect("owner staging exists");
    let diagnostics_directory = manager.diagnostics_directory().expect("claim diagnostics exist").to_path_buf();
    let failure_guard = crate::persistence::io::install_immutable_publication_directory_sync_failure_for_test(
        lineage_paths.genesis_path.clone(),
        3,
    );
    let activation_error = manager
        .activate_with_deferred_completed_noop_cleanup(test_association_implementation())
        .err()
        .expect("postlink durability failure rejects activation");
    drop(failure_guard);
    let crate::OutputActivationFailureParts { source, rollback } = activation_error.into_parts();
    assert!(source.to_string().contains("directory durability"));
    assert!(rollback.is_none(), "visible genesis must never expose destructive rollback");
    assert!(lineage_paths.genesis_path.is_file());
    assert!(diagnostics_directory.is_dir());
    assert_eq!(
        lineage_paths.owner_staging_attempt(&claim_id).expect("owner staging remains readable"),
        Some(staged_attempt_id.clone())
    );
    let blocked_plan =
        run_plan(&directory, &inputs, true, Some(staged_attempt_id.as_str().to_string()), g_plan::TelemetryMode::Off);
    assert_surviving_owner_claim(
        claim_error(blocked_plan, &inputs, &single_chunk_ranges(2), "visible genesis retains its exact owner"),
        &output_root,
    );

    let recovery_plan = authorize_fenced_owner_claim(
        run_plan(&directory, &inputs, true, Some(staged_attempt_id.as_str().to_string()), g_plan::TelemetryMode::Off),
        claim_id,
    );
    let recovery_manager = OutputManager::open(recovery_plan, "# fenced visible genesis\n".to_string())
        .expect("fenced recovery plans")
        .claim(vec![header(&inputs, 2)], &single_chunk_ranges(2), false)
        .expect("fenced recovery claims");
    assert!(diagnostics_directory.is_dir(), "referenced genesis diagnostics survive fencing");
    recovery_manager.abort_before_activation().expect("recovery staging cleans and releases");
    assert_owner_authority_released(&output_root);
}

#[test]
fn visible_successor_after_durability_error_never_loses_staging() {
    let directory = TestDirectory::new("successor-visible-durability-error");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    initialize_manager(initial_plan, &inputs, &chunk_ranges)
        .finish_interrupted("SIGTERM")
        .expect("predecessor interrupts");

    let output_root = directory.path.join("results");
    let lineage_paths = crate::persistence::lineage::OutputLineagePaths::new(&output_root);
    let parent_attempt_id = attempt_identifier(&output_root, None);
    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Profile);
    let manager = OutputManager::open(Arc::clone(&resume_plan), "# visible successor failure\n".to_string())
        .expect("resume plans")
        .claim(single_phenotype_headers(&inputs, &chunk_ranges), &chunk_ranges, true)
        .expect("resume claims");
    let claim_id = owner_claim_identifier(&output_root);
    let staged_attempt_id =
        lineage_paths.owner_staging_attempt(&claim_id).expect("owner staging reads").expect("owner staging exists");
    let diagnostics_directory = manager.diagnostics_directory().expect("claim diagnostics exist").to_path_buf();
    let successor_path = lineage_paths.normal_successor_path(
        &crate::persistence::identifier::AttemptIdentifier::parse(&parent_attempt_id)
            .expect("parent attempt identifier parses"),
    );
    let failure_guard = crate::persistence::io::install_immutable_publication_directory_sync_failure_for_test(
        successor_path.clone(),
        3,
    );
    let activation_error = manager
        .activate(test_association_implementation())
        .err()
        .expect("postlink successor durability failure rejects activation");
    drop(failure_guard);
    assert!(activation_error.to_string().contains("directory durability"));
    assert!(successor_path.is_file());
    assert!(diagnostics_directory.is_dir());
    assert_eq!(
        attempt_identifier(&output_root, Some(&parent_attempt_id)),
        staged_attempt_id.as_str(),
        "visible successor references the reserved attempt"
    );

    let recovery_plan = authorize_fenced_owner_claim(
        run_plan(&directory, &inputs, true, Some(staged_attempt_id.as_str().to_string()), g_plan::TelemetryMode::Off),
        claim_id,
    );
    let recovery_manager = OutputManager::open(recovery_plan, "# fenced visible successor\n".to_string())
        .expect("fenced successor recovery plans")
        .claim(single_phenotype_headers(&inputs, &chunk_ranges), &chunk_ranges, false)
        .expect("fenced successor recovery claims");
    assert!(diagnostics_directory.is_dir(), "referenced successor diagnostics survive fencing");
    recovery_manager.abort_before_activation().expect("recovery staging cleans and releases");
    assert_owner_authority_released(&output_root);
}

#[test]
fn conflicting_genesis_target_fails_closed_without_rollback() {
    let directory = TestDirectory::new("genesis-conflicting-target");
    let inputs = test_inputs(&directory);
    let plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Profile);
    let output_root = directory.path.join("results");
    let manager = OutputManager::open(plan, "# conflicting genesis\n".to_string())
        .expect("manager plans")
        .claim(vec![header(&inputs, 2)], &single_chunk_ranges(2), true)
        .expect("manager claims");
    let diagnostics_directory = manager.diagnostics_directory().expect("claim diagnostics exist").to_path_buf();
    let genesis_path = output_root.join(".g-output/genesis.json");
    std::fs::write(&genesis_path, b"{}\n").expect("conflicting target publishes");
    let activation_error = manager
        .activate_with_deferred_completed_noop_cleanup(test_association_implementation())
        .err()
        .expect("conflicting genesis rejects activation");
    let crate::OutputActivationFailureParts { rollback, .. } = activation_error.into_parts();
    assert!(rollback.is_none(), "present conflicting authority must fail closed");
    assert!(diagnostics_directory.is_dir());
    assert!(output_root.join(".g-output/session.claim.json").is_file());
}

#[test]
fn completed_claim_has_no_cleanup_authority_before_read_only_finalization() {
    let directory = TestDirectory::new("completed-noop-finalization-gate");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let initial_manager = initialize_manager(initial_plan, &inputs, &chunk_ranges);
    let token =
        initial_manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    write_chunk(&token, &metadata_store(2), 0..2);
    drop(token);
    assert_no_post_session_cleanup(
        initial_manager
            .close_completed()
            .expect("initial exact coverage closes")
            .finish()
            .expect("initial output completes"),
    );

    let output_root = directory.path.join("results");
    let completed_attempt_id = attempt_identifier(&output_root, None);
    let before = regular_file_snapshot(&output_root);
    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Profile);
    let claimed_manager = OutputManager::open(Arc::clone(&resume_plan), "# completed no-op\n".to_string())
        .expect("resume manager plans")
        .claim(single_phenotype_headers(&inputs, &chunk_ranges), &chunk_ranges, true)
        .expect("completed output is claimed");
    let lineage_paths = crate::persistence::lineage::OutputLineagePaths::new(&output_root);
    let active_claim_id = owner_claim_identifier(&output_root);
    let staging_attempt_id = lineage_paths
        .owner_staging_attempt(&active_claim_id)
        .expect("completed staging reads")
        .expect("completed staging exists");
    assert_ne!(staging_attempt_id.as_str(), completed_attempt_id);
    let diagnostics_directory =
        claimed_manager.diagnostics_directory().expect("completed diagnostics path exists").to_path_buf();
    std::fs::write(diagnostics_directory.join("events.jsonl"), b"{\"schema_version\":0}\n")
        .expect("test telemetry is written");

    let contender_error =
        claim_error(Arc::clone(&resume_plan), &inputs, &chunk_ranges, "unfinalized completed claim remains exclusive");
    assert_surviving_owner_claim(contender_error, &output_root);

    let completed_manager = claimed_manager
        .activate_with_deferred_completed_noop_cleanup(test_association_implementation())
        .expect("completed output activates read-only");
    let token =
        completed_manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("read-only token builds");
    assert!(token.is_read_only());
    drop(token);
    let completion = completed_manager
        .close_completed()
        .expect("completed output reverifies")
        .finish()
        .expect("completed output returns cleanup authority");
    assert_eq!(owner_claim_identifier(&output_root), active_claim_id);
    assert!(diagnostics_directory.is_dir());
    let mut cleanup = completion.post_session_cleanup.expect("deferred completed no-op returns cleanup");
    cleanup.cleanup().expect("post-session cleanup releases the current claim");
    cleanup.cleanup().expect("completed cleanup is idempotent after exact release");

    assert!(!lineage_paths.attempt_directory(&staging_attempt_id).exists());
    assert_eq!(lineage_paths.owner_staging_attempt(&active_claim_id).expect("staging absence reads"), None);
    assert_owner_authority_released(&output_root);
    OutputManager::open(resume_plan, "# reacquired after cleanup\n".to_string())
        .expect("post-cleanup manager plans")
        .claim(single_phenotype_headers(&inputs, &chunk_ranges), &chunk_ranges, false)
        .expect("ordinary reacquisition succeeds after current-owner cleanup")
        .abort_before_activation()
        .expect("reacquired staging cleans and releases");
    assert_owner_authority_released(&output_root);
    let after = regular_file_snapshot(&output_root);
    let new_paths = after.keys().filter(|path| !before.contains_key(*path)).collect::<Vec<_>>();
    assert!(
        !new_paths.is_empty() && new_paths.iter().all(|path| path.starts_with(".g-output/owner-transitions")),
        "completed resume may append only owner-authority transitions, got {new_paths:?}"
    );
}

#[test]
fn completed_noop_cleanup_retries_the_same_authority_after_each_durable_boundary() {
    for failure_point in [
        "after_post_session_staging_removal",
        "after_post_session_intent_retirement",
        "after_post_session_owner_release",
    ] {
        let CompletedNoopCleanupFixture {
            _directory,
            output_root,
            lineage_paths,
            claim_id,
            staging_attempt_id,
            mut cleanup,
        } = completed_noop_cleanup_fixture(failure_point);
        let failure_guard = crate::manager::install_terminal_cleanup_failure_for_test(failure_point);
        let cleanup_error = cleanup.cleanup().expect_err("configured cleanup boundary fails");
        drop(failure_guard);
        assert!(cleanup_error.to_string().contains(failure_point));
        assert!(!lineage_paths.attempt_directory(&staging_attempt_id).exists());
        if failure_point == "after_post_session_staging_removal" {
            assert_eq!(
                lineage_paths.owner_staging_attempt(&claim_id).expect("owner staging reads"),
                Some(staging_attempt_id.clone())
            );
            assert!(output_root.join(".g-output/session.claim.json").is_file());
        } else {
            assert_eq!(lineage_paths.owner_staging_attempt(&claim_id).expect("owner staging reads"), None);
        }
        if failure_point == "after_post_session_owner_release" {
            assert_owner_authority_released(&output_root);
        } else {
            assert!(output_root.join(".g-output/session.claim.json").is_file());
        }

        cleanup.cleanup().expect("the same cleanup authority retries successfully");
        cleanup.cleanup().expect("completed cleanup remains idempotent");
        assert_eq!(lineage_paths.owner_staging_attempt(&claim_id).expect("owner staging reads"), None);
        assert_owner_authority_released(&output_root);
    }
}

#[test]
fn completed_noop_cleanup_reconfirms_visible_release_durability_on_retry() {
    let CompletedNoopCleanupFixture { _directory, output_root, mut cleanup, .. } =
        completed_noop_cleanup_fixture("completed-noop-release-durability-retry");
    crate::persistence::io::fail_owner_publication_syncs_for_test(5);

    let first_error = cleanup.cleanup().expect_err("release publication exhausts its internal durability retries");
    assert!(first_error.to_string().contains("directory synchronization failure"));
    let second_error =
        cleanup.cleanup().expect_err("the same token must consume and report the remaining release sync failure");
    assert!(second_error.to_string().contains("directory synchronization failure"));
    cleanup.cleanup().expect("the same token finally reconfirms the visible release");
    cleanup.cleanup().expect("durably released cleanup is idempotent");
    assert_owner_authority_released(&output_root);
}

#[test]
fn completed_noop_cleanup_durably_confirms_a_visible_competing_takeover() {
    let CompletedNoopCleanupFixture {
        _directory: directory,
        output_root,
        lineage_paths,
        claim_id,
        staging_attempt_id,
        mut cleanup,
    } = completed_noop_cleanup_fixture("completed-noop-visible-takeover-durability");
    let inputs = test_inputs(&directory);
    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off);
    let transition_path = output_root.join(".g-output/owner-transitions").join(format!("{claim_id}.json"));
    let failure_guard =
        crate::persistence::io::install_immutable_publication_directory_sync_failure_for_test(transition_path, 4);
    let successor_plan = authorize_fenced_owner_claim(resume_plan, claim_id.clone());
    let takeover_result = OutputManager::open(successor_plan, "# visible competing takeover\n".to_string())
        .expect("successor plans")
        .claim(vec![header(&inputs, 2)], &single_chunk_ranges(2), false);
    let Err(takeover_error) = takeover_result else {
        panic!("four synchronization failures must leave takeover durability unresolved");
    };
    let visible_takeover_id = match takeover_error {
        crate::OutputError::PublishedOutputOwnerClaimDurability { claim_id, .. } => claim_id,
        unexpected => panic!("expected visible takeover durability authority, got: {unexpected}"),
    };

    let cleanup_error =
        cleanup.cleanup().expect_err("predecessor cleanup must report the remaining competing-transition sync failure");
    assert!(cleanup_error.to_string().contains("directory synchronization failure"));
    assert!(!lineage_paths.attempt_directory(&staging_attempt_id).exists());
    assert_eq!(lineage_paths.owner_staging_attempt(&claim_id).expect("retired intent reads"), None);
    drop(failure_guard);

    cleanup.cleanup().expect("same cleanup token durably reconfirms the competing takeover");
    cleanup.cleanup().expect("superseded cleanup remains idempotent");
    assert_eq!(
        lineage_paths.current_owner_claim_identifier_for_test().expect("current owner resolves"),
        Some(visible_takeover_id.clone())
    );
    let mut final_claim = lineage_paths
        .take_over_fenced_owner_claim(&visible_takeover_id)
        .expect("visible takeover supports exact final recovery");
    final_claim.release().expect("final recovery releases");
    assert_owner_authority_released(&output_root);
}

#[test]
fn completed_noop_cleanup_continues_when_its_staging_intent_is_already_absent() {
    let CompletedNoopCleanupFixture {
        _directory,
        output_root,
        lineage_paths,
        claim_id,
        staging_attempt_id,
        mut cleanup,
    } = completed_noop_cleanup_fixture("completed-noop-missing-intent");
    lineage_paths
        .retire_owner_staging_intent(&claim_id, &staging_attempt_id)
        .expect("test retires owner staging intent first");
    assert!(lineage_paths.attempt_directory(&staging_attempt_id).is_dir());
    assert!(output_root.join(".g-output/session.claim.json").is_file());

    cleanup.cleanup().expect("cleanup continues from an already-retired intent");
    cleanup.cleanup().expect("cleanup is idempotent after missing-intent recovery");
    assert!(!lineage_paths.attempt_directory(&staging_attempt_id).exists());
    assert_owner_authority_released(&output_root);
}

#[test]
fn completed_noop_cleanup_is_a_noop_after_fenced_successor_release() {
    let directory = TestDirectory::new("completed-noop-cleanup-after-successor-release");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let initial_manager = initialize_manager(initial_plan, &inputs, &chunk_ranges);
    let delivery_token =
        initial_manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    write_chunk(&delivery_token, &metadata_store(2), 0..2);
    drop(delivery_token);
    assert_no_post_session_cleanup(
        initial_manager
            .close_completed()
            .expect("initial exact coverage closes")
            .finish()
            .expect("initial output completes"),
    );

    let output_root = directory.path.join("results");
    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Profile);
    let completion = OutputManager::open(Arc::clone(&resume_plan), "# predecessor cleanup\n".to_string())
        .expect("completed resume plans")
        .claim(single_phenotype_headers(&inputs, &chunk_ranges), &chunk_ranges, true)
        .expect("completed resume claims")
        .activate_with_deferred_completed_noop_cleanup(test_association_implementation())
        .expect("completed resume activates")
        .close_completed()
        .expect("completed resume reverifies")
        .finish()
        .expect("completed resume returns cleanup");
    let mut cleanup = completion.post_session_cleanup.expect("predecessor cleanup exists");
    let predecessor_claim_id = owner_claim_identifier(&output_root);

    let successor = OutputManager::open(
        authorize_fenced_owner_claim(resume_plan, predecessor_claim_id),
        "# fenced successor\n".to_string(),
    )
    .expect("fenced successor plans")
    .claim(single_phenotype_headers(&inputs, &chunk_ranges), &chunk_ranges, false)
    .expect("fenced successor claims");
    successor.abort_before_activation().expect("fenced successor releases");
    assert_owner_authority_released(&output_root);

    cleanup.cleanup().expect("superseded predecessor cleanup is harmless after successor release");
    cleanup.cleanup().expect("superseded cleanup remains idempotent");
    assert_owner_authority_released(&output_root);
}

#[test]
fn every_completed_noop_terminal_failure_returns_cleanup_authority() {
    let directory = TestDirectory::new("completed-noop-terminal-failures");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let initial_manager = initialize_manager(initial_plan, &inputs, &chunk_ranges);
    let delivery_token =
        initial_manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    write_chunk(&delivery_token, &metadata_store(2), 0..2);
    drop(delivery_token);
    assert_no_post_session_cleanup(
        initial_manager
            .close_completed()
            .expect("initial exact coverage closes")
            .finish()
            .expect("initial output completes"),
    );
    let output_root = directory.path.join("results");

    for (operation, expected_error) in [
        ("close", "close_completed_noop"),
        ("interrupt-empty", "signal name"),
        ("interrupt", "cannot be interrupted"),
        ("abort-empty", "failure reason"),
        ("abort", "cannot be aborted"),
    ] {
        let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Profile);
        let manager = OutputManager::open(resume_plan, format!("# completed no-op {operation}\n"))
            .expect("completed resume plans")
            .claim(single_phenotype_headers(&inputs, &chunk_ranges), &chunk_ranges, true)
            .expect("completed resume claims")
            .activate_with_deferred_completed_noop_cleanup(test_association_implementation())
            .expect("completed resume activates");
        let lifecycle_guard =
            (operation == "close").then(|| crate::manager::install_lifecycle_failure_for_test("close_completed_noop"));
        let terminal_error = match operation {
            "close" => manager.close_completed().err().expect("configured close failure fires"),
            "interrupt-empty" => manager.finish_interrupted("").expect_err("empty signal is rejected"),
            "interrupt" => manager.finish_interrupted("SIGTERM").expect_err("completed output cannot interrupt"),
            "abort-empty" => manager.abort("").expect_err("empty failure reason is rejected"),
            "abort" => manager.abort("requested").expect_err("completed output cannot abort"),
            _ => unreachable!("completed no-op failure operations are exhaustive"),
        };
        drop(lifecycle_guard);
        let crate::OutputTerminalFailureParts { source, post_session_cleanup } = terminal_error.into_parts();
        assert!(source.to_string().contains(expected_error), "unexpected {operation} error: {source}");
        let mut cleanup = post_session_cleanup.expect("terminal failure returns cleanup authority");
        assert!(output_root.join(".g-output/session.claim.json").is_file());
        cleanup.cleanup().expect("terminal failure cleanup releases after diagnostics close");
        cleanup.cleanup().expect("terminal failure cleanup is idempotent");
        assert_owner_authority_released(&output_root);
    }

    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Profile);
    let covered = OutputManager::open(resume_plan, "# completed no-op finish failure\n".to_string())
        .expect("completed resume plans")
        .claim(single_phenotype_headers(&inputs, &chunk_ranges), &chunk_ranges, true)
        .expect("completed resume claims")
        .activate_with_deferred_completed_noop_cleanup(test_association_implementation())
        .expect("completed resume activates")
        .close_completed()
        .expect("completed resume first verification succeeds");
    let completed_attempt_id = attempt_identifier(&output_root, None);
    let manifest_path =
        output_root.join("attempts").join(completed_attempt_id).join(OUTPUT_DIRECTORY_NAME).join("run_manifest.json");
    let mut manifest = read_json(&manifest_path);
    manifest["runtime"]["device"] = Value::String("invalid".to_string());
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest).expect("corrupt manifest serializes"))
        .expect("completed manifest is corrupted after close");
    let finish_error = covered.finish().expect_err("final completed revalidation fails");
    let crate::OutputTerminalFailureParts { source, post_session_cleanup } = finish_error.into_parts();
    assert!(source.to_string().contains("runtime field 'device'"));
    let mut cleanup = post_session_cleanup.expect("finish failure returns cleanup authority");
    cleanup.cleanup().expect("finish failure cleanup releases after diagnostics close");
    cleanup.cleanup().expect("finish failure cleanup is idempotent");
    assert_owner_authority_released(&output_root);
}

#[test]
fn immediate_completed_noop_finish_preserves_cleanup_authority_on_failure() {
    let directory = TestDirectory::new("completed-noop-immediate-cleanup-failure");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let initial_manager = initialize_manager(initial_plan, &inputs, &chunk_ranges);
    let delivery_token =
        initial_manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    write_chunk(&delivery_token, &metadata_store(2), 0..2);
    drop(delivery_token);
    assert_no_post_session_cleanup(
        initial_manager
            .close_completed()
            .expect("initial exact coverage closes")
            .finish()
            .expect("initial output completes"),
    );

    let output_root = directory.path.join("results");
    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Profile);
    let covered =
        initialize_manager(resume_plan, &inputs, &chunk_ranges).close_completed().expect("completed no-op reverifies");
    let failure_guard = crate::manager::install_terminal_cleanup_failure_for_test("before_post_session_owner_release");
    let finish_error = covered.finish().expect_err("immediate post-session cleanup failure is returned");
    drop(failure_guard);
    let crate::OutputTerminalFailureParts { source, post_session_cleanup } = finish_error.into_parts();
    assert!(source.to_string().contains("before_post_session_owner_release"));
    let mut cleanup = post_session_cleanup.expect("immediate cleanup failure preserves retry authority");
    assert!(output_root.join(".g-output/session.claim.json").is_file());
    cleanup.cleanup().expect("preserved immediate cleanup retries successfully");
    cleanup.cleanup().expect("preserved immediate cleanup is idempotent");
    assert_owner_authority_released(&output_root);
}

#[test]
fn completed_noop_cleanup_racing_fenced_takeover_preserves_successor_staging() {
    let directory = TestDirectory::new("completed-noop-cleanup-takeover");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let initial_manager = initialize_manager(initial_plan, &inputs, &chunk_ranges);
    let token =
        initial_manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    write_chunk(&token, &metadata_store(2), 0..2);
    drop(token);
    assert_no_post_session_cleanup(
        initial_manager
            .close_completed()
            .expect("initial exact coverage closes")
            .finish()
            .expect("initial output completes"),
    );

    let output_root = directory.path.join("results");
    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Profile);
    let claimed_manager = OutputManager::open(Arc::clone(&resume_plan), "# completed predecessor\n".to_string())
        .expect("completed predecessor plans")
        .claim(single_phenotype_headers(&inputs, &chunk_ranges), &chunk_ranges, true)
        .expect("completed predecessor claims");
    let predecessor_diagnostics =
        claimed_manager.diagnostics_directory().expect("predecessor diagnostics exist").to_path_buf();
    let predecessor_claim_id = owner_claim_identifier(&output_root);
    let completed_manager = claimed_manager
        .activate_with_deferred_completed_noop_cleanup(test_association_implementation())
        .expect("completed predecessor activates");
    let completion = completed_manager
        .close_completed()
        .expect("completed predecessor reverifies")
        .finish()
        .expect("completed predecessor finalizes read-only");
    let mut predecessor_cleanup = completion.post_session_cleanup.expect("post-session cleanup exists");
    let cleanup_control = predecessor_cleanup.install_cleanup_pause_for_test();

    let successor_plan = authorize_fenced_owner_claim(resume_plan, predecessor_claim_id.clone());
    let successor_manager =
        OutputManager::open(successor_plan, "# successor\n".to_string()).expect("successor manager plans");
    let successor_manager = std::thread::scope(|scope| {
        let cleanup_thread = scope.spawn(move || predecessor_cleanup.cleanup());
        cleanup_control.wait_until_reached().expect("predecessor cleanup reaches its bounded test pause");
        let successor_result = successor_manager.claim(vec![header(&inputs, 2)], &single_chunk_ranges(2), false);
        cleanup_control.resume();
        let cleanup_result = cleanup_thread.join().expect("predecessor cleanup thread joins");
        cleanup_result.expect("predecessor cleanup tolerates the successor sweep");
        successor_result.expect("fenced successor claims while predecessor cleanup is paused")
    });
    let successor_claim_id = owner_claim_identifier(&output_root);
    assert_ne!(successor_claim_id, predecessor_claim_id);
    let successor_diagnostics_directory =
        successor_manager.diagnostics_directory().expect("successor diagnostics path exists").to_path_buf();
    assert!(!predecessor_diagnostics.exists());
    assert!(successor_diagnostics_directory.is_dir());
    successor_manager.abort_before_activation().expect("successor staging cleans and releases");
    assert_owner_authority_released(&output_root);
}

#[test]
fn fenced_successor_sweeps_dropped_completed_noop_cleanup() {
    let directory = TestDirectory::new("completed-noop-dropped-cleanup");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let initial_manager = initialize_manager(initial_plan, &inputs, &chunk_ranges);
    let token =
        initial_manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    write_chunk(&token, &metadata_store(2), 0..2);
    drop(token);
    assert_no_post_session_cleanup(
        initial_manager
            .close_completed()
            .expect("initial exact coverage closes")
            .finish()
            .expect("initial output completes"),
    );

    let output_root = directory.path.join("results");
    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Profile);
    let claimed_manager = OutputManager::open(Arc::clone(&resume_plan), "# dropped cleanup\n".to_string())
        .expect("completed resume plans")
        .claim(single_phenotype_headers(&inputs, &chunk_ranges), &chunk_ranges, true)
        .expect("completed resume claims");
    let lineage_paths = crate::persistence::lineage::OutputLineagePaths::new(&output_root);
    let diagnostics_directory =
        claimed_manager.diagnostics_directory().expect("completed claim diagnostics exist").to_path_buf();
    std::fs::write(diagnostics_directory.join("events.jsonl"), b"{\"schema_version\":0}\n")
        .expect("test telemetry is written");
    let predecessor_claim_id = owner_claim_identifier(&output_root);
    let staging_attempt_id = lineage_paths
        .owner_staging_attempt(&predecessor_claim_id)
        .expect("predecessor staging reads")
        .expect("predecessor staging exists");
    let completed_manager = claimed_manager
        .activate_with_deferred_completed_noop_cleanup(test_association_implementation())
        .expect("completed output activates read-only");
    let completion = completed_manager
        .close_completed()
        .expect("completed output reverifies")
        .finish()
        .expect("completed output returns cleanup authority");
    drop(completion.post_session_cleanup.expect("completed no-op cleanup exists"));

    assert!(diagnostics_directory.is_dir(), "dropped cleanup retains claim-scoped diagnostics");
    assert_eq!(
        lineage_paths.owner_staging_attempt(&predecessor_claim_id).expect("retained staging reads"),
        Some(staging_attempt_id.clone())
    );
    let contender_error = claim_error(
        Arc::clone(&resume_plan),
        &inputs,
        &chunk_ranges,
        "dropped cleanup must retain the exact owner claim",
    );
    assert_surviving_owner_claim(contender_error, &output_root);

    let contender_plan = authorize_fenced_owner_claim(resume_plan, predecessor_claim_id.clone());
    let contender = OutputManager::open(contender_plan, "# fenced contender\n".to_string())
        .expect("fenced contender plans")
        .claim(single_phenotype_headers(&inputs, &chunk_ranges), &chunk_ranges, false)
        .expect("fenced contender takes over");
    assert!(!diagnostics_directory.exists());
    assert!(!lineage_paths.attempt_directory(&staging_attempt_id).exists());
    assert_eq!(lineage_paths.owner_staging_attempt(&predecessor_claim_id).expect("staging absence reads"), None);
    assert!(
        contender.diagnostics_directory().expect("contender diagnostics path remains").is_dir(),
        "the successor sweep must preserve its own staging"
    );
    contender.abort_before_activation().expect("contender staging cleans and releases");
    assert_owner_authority_released(&output_root);
}

#[test]
fn completed_preactivation_failure_returns_only_rollback_authority() {
    let directory = TestDirectory::new("completed-preactivation-rollback");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let initial_manager = initialize_manager(initial_plan, &inputs, &chunk_ranges);
    let token =
        initial_manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    write_chunk(&token, &metadata_store(2), 0..2);
    drop(token);
    assert_no_post_session_cleanup(
        initial_manager
            .close_completed()
            .expect("initial exact coverage closes")
            .finish()
            .expect("initial output completes"),
    );

    let output_root = directory.path.join("results");
    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Profile);
    let claimed_manager = OutputManager::open(resume_plan, "# completed rollback\n".to_string())
        .expect("completed resume plans")
        .claim(single_phenotype_headers(&inputs, &chunk_ranges), &chunk_ranges, true)
        .expect("completed resume claims");
    let diagnostics_directory =
        claimed_manager.diagnostics_directory().expect("completed claim diagnostics exist").to_path_buf();
    let failure_guard = crate::manager::install_initialization_failure_for_test("after_owner_claim");
    let activation_error = claimed_manager
        .activate_with_deferred_completed_noop_cleanup(test_association_implementation())
        .err()
        .expect("prepublication activation failure is injected");
    drop(failure_guard);
    let crate::OutputActivationFailureParts { source, rollback } = activation_error.into_parts();
    assert!(source.to_string().contains("after_owner_claim"));
    assert!(diagnostics_directory.is_dir());
    rollback
        .expect("unpublished activation returns rollback authority")
        .abort_before_activation()
        .expect("rollback cleans after diagnostics close");
    assert!(!diagnostics_directory.exists());
    assert_owner_authority_released(&output_root);
}

#[test]
fn existing_lineage_is_preflighted_before_owner_claim_acquisition() {
    let mismatch_directory = TestDirectory::new("preflight-mismatch");
    let mismatch_inputs = test_inputs(&mismatch_directory);
    let initial_plan = run_plan(&mismatch_directory, &mismatch_inputs, false, None, g_plan::TelemetryMode::Off);
    initialize_manager(initial_plan, &mismatch_inputs, &single_chunk_ranges(2))
        .finish_interrupted("SIGTERM")
        .expect("initial attempt is interrupted");
    let mismatch_plan = run_plan(&mismatch_directory, &mismatch_inputs, true, None, g_plan::TelemetryMode::Off);
    let mismatch_manager =
        OutputManager::open(mismatch_plan, "# test configuration\n".to_string()).expect("resume manager plans");
    let failure_guard = crate::manager::install_initialization_failure_for_test("after_owner_claim");
    let mismatch_error = mismatch_manager
        .initialize(
            vec![header(&mismatch_inputs, 1)],
            &single_chunk_ranges(1),
            false,
            test_association_implementation(),
        )
        .err()
        .expect("mismatched chunk plan fails");
    drop(failure_guard);
    assert!(mismatch_error.to_string().contains("lineage contract does not match"));
    assert!(!mismatch_error.to_string().contains("Injected"));
    assert_owner_authority_released(&mismatch_directory.path.join("results"));

    let corruption_directory = TestDirectory::new("preflight-corruption");
    let corruption_inputs = test_inputs(&corruption_directory);
    let initial_plan = run_plan(&corruption_directory, &corruption_inputs, false, None, g_plan::TelemetryMode::Off);
    initialize_manager(initial_plan, &corruption_inputs, &single_chunk_ranges(2))
        .finish_interrupted("SIGTERM")
        .expect("initial attempt is interrupted");
    let output_root = corruption_directory.path.join("results");
    let attempt = attempt_identifier(&output_root, None);
    let manifest_path =
        output_root.join("attempts").join(attempt).join(OUTPUT_DIRECTORY_NAME).join("run_manifest.json");
    let mut manifest = read_json(&manifest_path);
    manifest["failure_reason"] = Value::String("tampered".to_string());
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest).expect("tampered manifest serializes"))
        .expect("terminal manifest is corrupted");

    let corruption_plan = run_plan(&corruption_directory, &corruption_inputs, true, None, g_plan::TelemetryMode::Off);
    let corruption_manager =
        OutputManager::open(corruption_plan, "# test configuration\n".to_string()).expect("resume manager plans");
    let failure_guard = crate::manager::install_initialization_failure_for_test("after_owner_claim");
    let corruption_error = corruption_manager
        .initialize(
            vec![header(&corruption_inputs, 2)],
            &single_chunk_ranges(2),
            false,
            test_association_implementation(),
        )
        .err()
        .expect("corrupt lineage fails");
    drop(failure_guard);
    assert!(!corruption_error.to_string().contains("Injected"));
    assert_owner_authority_released(&output_root);
}

#[test]
fn resume_rejects_noncanonical_attempt_manifest_schema_versions_without_a_successor() {
    for (label, schema_version) in [
        ("missing", None),
        ("one", Some(Value::from(1_u32))),
        ("floating-zero", Some(Value::from(0.0_f64))),
        ("boolean", Some(Value::Bool(false))),
    ] {
        let directory = TestDirectory::new(&format!("attempt-schema-{label}"));
        let inputs = test_inputs(&directory);
        let chunk_ranges = single_chunk_ranges(2);
        let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
        initialize_manager(initial_plan, &inputs, &chunk_ranges)
            .finish_interrupted("SIGTERM")
            .expect("initial attempt is interrupted");

        let output_root = directory.path.join("results");
        let attempt = attempt_identifier(&output_root, None);
        let manifest_path =
            output_root.join("attempts").join(&attempt).join(OUTPUT_DIRECTORY_NAME).join("run_manifest.json");
        let mut manifest = read_json(&manifest_path);
        let manifest_object = manifest.as_object_mut().expect("attempt manifest is an object");
        match schema_version {
            Some(schema_version) => {
                manifest_object.insert("attempt_manifest_schema_version".to_string(), schema_version);
            }
            None => {
                manifest_object.remove("attempt_manifest_schema_version");
            }
        }
        std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest).expect("manifest serializes"))
            .expect("manifest schema version is replaced");

        let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off);
        let error = OutputManager::open(resume_plan, "# invalid schema resume\n".to_string())
            .expect("resume manager plans")
            .initialize(vec![header(&inputs, 2)], &chunk_ranges, false, test_association_implementation())
            .err()
            .expect("noncanonical attempt manifest schema is rejected");
        assert!(error.to_string().contains("unsupported schema version"));
        assert!(!output_root.join(".g-output/successors").join(format!("{attempt}.json")).exists());
        assert_eq!(std::fs::read_dir(output_root.join("attempts")).expect("attempt root reads").count(), 1);
        assert_owner_authority_released(&output_root);
    }
}

struct InvalidAttemptManifestCase {
    label: &'static str,
    manifest_bytes: Vec<u8>,
    expected_error: &'static str,
}

fn invalid_attempt_manifest_case(
    label: &'static str,
    manifest: &Value,
    expected_error: &'static str,
) -> InvalidAttemptManifestCase {
    InvalidAttemptManifestCase {
        label,
        manifest_bytes: serde_json::to_vec_pretty(manifest).expect("invalid manifest serializes"),
        expected_error,
    }
}

fn nonexact_attempt_manifest_cases(baseline_manifest: &Value) -> Vec<InvalidAttemptManifestCase> {
    let mut invalid_manifests = Vec::new();
    let mut unknown_top_level = baseline_manifest.clone();
    unknown_top_level["unknown"] = Value::Null;
    invalid_manifests.push(invalid_attempt_manifest_case(
        "unknown-top-level",
        &unknown_top_level,
        "contains unknown field 'unknown'",
    ));

    let mut missing_command = baseline_manifest.clone();
    missing_command.as_object_mut().expect("manifest is an object").remove("command");
    invalid_manifests.push(invalid_attempt_manifest_case(
        "missing-command",
        &missing_command,
        "field 'command' is missing",
    ));

    let mut wrong_command = baseline_manifest.clone();
    wrong_command["command"] = Value::Bool(false);
    invalid_manifests.push(invalid_attempt_manifest_case(
        "wrong-command-type",
        &wrong_command,
        "command must contain an object",
    ));

    let mut missing_runtime = baseline_manifest.clone();
    missing_runtime.as_object_mut().expect("manifest is an object").remove("runtime");
    invalid_manifests.push(invalid_attempt_manifest_case(
        "missing-runtime",
        &missing_runtime,
        "field 'runtime' is missing",
    ));

    let mut wrong_runtime = baseline_manifest.clone();
    wrong_runtime["runtime"] = Value::Bool(false);
    invalid_manifests.push(invalid_attempt_manifest_case(
        "wrong-runtime-type",
        &wrong_runtime,
        "runtime must contain an object",
    ));

    let mut wrong_interrupted_signal = baseline_manifest.clone();
    wrong_interrupted_signal["interrupted_signal"] = Value::Bool(false);
    invalid_manifests.push(invalid_attempt_manifest_case(
        "wrong-interrupted-signal-type",
        &wrong_interrupted_signal,
        "expected a string",
    ));

    let mut missing_interrupted_signal = baseline_manifest.clone();
    missing_interrupted_signal.as_object_mut().expect("manifest is an object").remove("interrupted_signal");
    invalid_manifests.push(invalid_attempt_manifest_case(
        "missing-interrupted-signal",
        &missing_interrupted_signal,
        "field 'interrupted_signal' is missing",
    ));

    for status in ["running", "completed"] {
        let mut inapplicable_detail = baseline_manifest.clone();
        inapplicable_detail["status"] = Value::String(status.to_string());
        invalid_manifests.push(invalid_attempt_manifest_case(
            status,
            &inapplicable_detail,
            "contains unknown field 'interrupted_signal'",
        ));
    }

    let mut failed_with_wrong_detail = baseline_manifest.clone();
    failed_with_wrong_detail["status"] = Value::String("failed".to_string());
    failed_with_wrong_detail.as_object_mut().expect("manifest is an object").remove("interrupted_signal");
    failed_with_wrong_detail["failure_reason"] = Value::Bool(false);
    invalid_manifests.push(invalid_attempt_manifest_case(
        "wrong-failure-reason-type",
        &failed_with_wrong_detail,
        "expected a string",
    ));

    let compact_manifest = serde_json::to_string(&baseline_manifest).expect("baseline manifest serializes");
    let duplicate_status = compact_manifest.replacen(
        "\"status\":\"interrupted\"",
        "\"status\":\"interrupted\",\"status\":\"interrupted\"",
        1,
    );
    assert_ne!(duplicate_status, compact_manifest);
    invalid_manifests.push(InvalidAttemptManifestCase {
        label: "duplicate-status",
        manifest_bytes: duplicate_status.into_bytes(),
        expected_error: "duplicate object key 'status'",
    });
    invalid_manifests
}

#[test]
fn resume_rejects_nonexact_attempt_manifest_schema_without_a_successor() {
    let directory = TestDirectory::new("attempt-schema-exact");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    initialize_manager(initial_plan, &inputs, &chunk_ranges)
        .finish_interrupted("SIGTERM")
        .expect("initial attempt is interrupted");

    let output_root = directory.path.join("results");
    let attempt = attempt_identifier(&output_root, None);
    let manifest_path =
        output_root.join("attempts").join(&attempt).join(OUTPUT_DIRECTORY_NAME).join("run_manifest.json");
    let baseline_manifest = read_json(&manifest_path);
    for invalid_manifest in nonexact_attempt_manifest_cases(&baseline_manifest) {
        let InvalidAttemptManifestCase { label, manifest_bytes, expected_error } = invalid_manifest;
        std::fs::write(&manifest_path, manifest_bytes).expect("invalid manifest is installed");
        let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off);
        let error = OutputManager::open(resume_plan, format!("# invalid schema {label}\n"))
            .expect("resume manager plans")
            .initialize(vec![header(&inputs, 2)], &chunk_ranges, false, test_association_implementation())
            .err()
            .expect("nonexact attempt manifest schema is rejected");
        assert!(error.to_string().contains(expected_error), "{label} returned unexpected error: {error}");
        assert!(!output_root.join(".g-output/successors").join(format!("{attempt}.json")).exists());
        assert_eq!(std::fs::read_dir(output_root.join("attempts")).expect("attempt root reads").count(), 1);
        assert_owner_authority_released(&output_root);
    }
}

#[test]
fn invalid_terminal_arguments_publish_failed_and_release_the_claim() {
    for (operation, expected_error) in [("interrupt", "interruption signal name"), ("abort", "failure reason")] {
        let directory = TestDirectory::new(operation);
        let inputs = test_inputs(&directory);
        let plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
        let manager = initialize_manager(plan, &inputs, &single_chunk_ranges(2));
        let error = match operation {
            "interrupt" => manager.finish_interrupted("").expect_err("empty signal is rejected"),
            "abort" => manager.abort("  ").expect_err("empty failure reason is rejected"),
            _ => unreachable!("terminal operations are exhaustive"),
        };
        assert!(error.to_string().contains(expected_error));

        let output_root = directory.path.join("results");
        assert_owner_authority_released(&output_root);
        let attempt = attempt_identifier(&output_root, None);
        let manifest = read_json(
            &output_root.join("attempts").join(&attempt).join(OUTPUT_DIRECTORY_NAME).join("run_manifest.json"),
        );
        assert_eq!(manifest["status"], "failed");
        assert!(output_root.join(".g-output/terminal-finalizations").join(format!("{attempt}.json")).is_file());
    }
}

#[test]
fn consuming_terminal_failures_report_primary_and_cleanup_outcomes() {
    let close_directory = TestDirectory::new("close-cleanup-conflict");
    let close_inputs = test_inputs(&close_directory);
    let close_plan = run_plan(&close_directory, &close_inputs, false, None, g_plan::TelemetryMode::Off);
    let close_manager = initialize_manager(close_plan, &close_inputs, &single_chunk_ranges(2));
    let close_token =
        close_manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    write_chunk(&close_token, &metadata_store(2), 0..2);
    drop(close_token);
    let lifecycle_guard = crate::manager::install_lifecycle_failure_for_test("close_completed");
    let cleanup_guard = crate::manager::install_terminal_cleanup_failure_for_test("owner_claim_release_conflict");
    let close_error = close_manager.close_completed().err().expect("close and cleanup failures are both reported");
    drop(cleanup_guard);
    drop(lifecycle_guard);
    let close_error_text = close_error.to_string();
    assert!(close_error_text.contains("close_completed"));
    assert!(close_error_text.contains("survives from process"));
    let close_root = close_directory.path.join("results");
    assert!(close_root.join(".g-output/session.claim.json").is_file());

    let interrupted_directory = TestDirectory::new("interrupted-terminal-cleanup");
    let interrupted_inputs = test_inputs(&interrupted_directory);
    let interrupted_plan =
        run_plan(&interrupted_directory, &interrupted_inputs, false, None, g_plan::TelemetryMode::Off);
    let interrupted_manager = initialize_manager(interrupted_plan, &interrupted_inputs, &single_chunk_ranges(2));
    let lifecycle_guard = crate::manager::install_lifecycle_failure_for_test("finish_interrupted");
    let cleanup_guard = crate::manager::install_terminal_cleanup_failure_for_test("before_terminal_publication");
    let interrupted_error = interrupted_manager
        .finish_interrupted("SIGTERM")
        .expect_err("flush and terminal-publication failures are both reported");
    drop(cleanup_guard);
    drop(lifecycle_guard);
    let interrupted_error_text = interrupted_error.to_string();
    assert!(interrupted_error_text.contains("finish_interrupted"));
    assert!(interrupted_error_text.contains("before_terminal_publication"));
    assert!(interrupted_error_text.contains("remains authoritative"));
    let interrupted_root = interrupted_directory.path.join("results");
    assert!(interrupted_root.join(".g-output/session.claim.json").is_file());

    let noop_directory = TestDirectory::new("completed-noop-cleanup-conflict");
    let noop_inputs = test_inputs(&noop_directory);
    let initial_plan = run_plan(&noop_directory, &noop_inputs, false, None, g_plan::TelemetryMode::Off);
    let initial_manager = initialize_manager(initial_plan, &noop_inputs, &single_chunk_ranges(2));
    let token =
        initial_manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    write_chunk(&token, &metadata_store(2), 0..2);
    drop(token);
    assert_no_post_session_cleanup(
        initial_manager.close_completed().expect("exact coverage closes").finish().expect("initial output completes"),
    );
    let resume_plan = run_plan(&noop_directory, &noop_inputs, true, None, g_plan::TelemetryMode::Off);
    let noop_manager = initialize_manager(resume_plan, &noop_inputs, &single_chunk_ranges(2));
    let lifecycle_guard = crate::manager::install_lifecycle_failure_for_test("close_completed_noop");
    let cleanup_guard = crate::manager::install_terminal_cleanup_failure_for_test("before_post_session_owner_release");
    let noop_error = noop_manager.close_completed().err().expect("completed no-op cleanup failure is reported");
    drop(cleanup_guard);
    drop(lifecycle_guard);
    let crate::OutputTerminalFailureParts { source, post_session_cleanup } = noop_error.into_parts();
    let noop_error_text = source.to_string();
    assert!(noop_error_text.contains("close_completed_noop"));
    assert!(noop_error_text.contains("before_post_session_owner_release"));
    let noop_root = noop_directory.path.join("results");
    assert!(noop_root.join(".g-output/session.claim.json").is_file());
    let mut post_session_cleanup =
        post_session_cleanup.expect("failed immediate cleanup preserves the same retry authority");
    post_session_cleanup.cleanup().expect("preserved cleanup authority retries after diagnostics close");
    post_session_cleanup.cleanup().expect("retried cleanup is idempotent");
    assert_owner_authority_released(&noop_root);
}

#[test]
fn consuming_terminal_release_retries_or_reports_transition_durability() {
    let retry_directory = TestDirectory::new("claim-release-sync-retry");
    let retry_inputs = test_inputs(&retry_directory);
    let retry_plan = run_plan(&retry_directory, &retry_inputs, false, None, g_plan::TelemetryMode::Off);
    let retry_manager = initialize_manager(retry_plan, &retry_inputs, &single_chunk_ranges(2));
    crate::persistence::io::fail_owner_publication_syncs_for_test(3);
    retry_manager.abort("release retry test").expect("three directory-sync failures are retried");
    assert_owner_authority_released(&retry_directory.path.join("results"));

    let unresolved_directory = TestDirectory::new("claim-release-sync-unresolved");
    let unresolved_inputs = test_inputs(&unresolved_directory);
    let unresolved_plan = run_plan(&unresolved_directory, &unresolved_inputs, false, None, g_plan::TelemetryMode::Off);
    let unresolved_manager = initialize_manager(unresolved_plan, &unresolved_inputs, &single_chunk_ranges(2));
    crate::persistence::io::fail_owner_publication_syncs_for_test(4);
    let error = unresolved_manager
        .abort("release durability test")
        .expect_err("four directory-sync failures report unresolved durability");
    let crate::OutputTerminalFailureParts { source, post_session_cleanup } = error.into_parts();
    assert!(post_session_cleanup.is_none());
    assert!(matches!(source, crate::OutputError::PublishedOutputOwnerClaimReleaseDurability { .. }));
    let error_text = source.to_string();
    assert!(error_text.contains("visible graceful-release transition"));
    assert!(!error_text.contains("remains authoritative"));
    let unresolved_root = unresolved_directory.path.join("results");
    assert!(unresolved_root.join(".g-output/session.claim.json").is_file());
    File::open(unresolved_root.join(".g-output"))
        .expect("control directory opens")
        .sync_all()
        .expect("test resolves the injected directory-sync uncertainty");
}

#[test]
fn completion_faults_do_not_strand_a_claim_across_terminal_publication() {
    for (failure_point, expected_status) in
        [("before_completed_terminal_publication", "failed"), ("after_completed_terminal_finalization", "completed")]
    {
        let directory = TestDirectory::new(failure_point);
        let inputs = test_inputs(&directory);
        let chunk_ranges = single_chunk_ranges(2);
        let plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
        let manager = initialize_manager(plan, &inputs, &chunk_ranges);
        let token =
            manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
        write_chunk(&token, &metadata_store(2), 0..2);
        drop(token);
        let covered = manager.close_completed().expect("exact coverage closes");

        let failure_guard = crate::manager::install_completion_failure_for_test(failure_point);
        let error = covered.finish().expect_err("configured completion stage fails");
        drop(failure_guard);
        assert!(error.to_string().contains("Injected output completion failure"));

        let output_root = directory.path.join("results");
        assert_owner_authority_released(&output_root);
        let attempt = attempt_identifier(&output_root, None);
        assert_eq!(
            read_json(
                &output_root.join("attempts").join(&attempt).join(OUTPUT_DIRECTORY_NAME).join("run_manifest.json")
            )["status"],
            expected_status
        );

        let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off);
        let resumed = initialize_manager(resume_plan, &inputs, &chunk_ranges);
        if expected_status == "completed" {
            assert_no_post_session_cleanup(
                resumed
                    .close_completed()
                    .expect("completed terminal reverifies")
                    .finish()
                    .expect("completed terminal returns read-only outputs"),
            );
        } else {
            resumed.abort("completion fault recovery test").expect("failed terminal resumes safely");
        }
        assert_owner_authority_released(&output_root);
    }
}

#[test]
fn output_transaction_subprocess_helper() {
    let Ok(mode) = std::env::var(TRANSACTION_HELPER_MODE_ENVIRONMENT) else {
        return;
    };
    let root = std::env::var(TRANSACTION_HELPER_ROOT_ENVIRONMENT).expect("transaction helper root is configured");
    let directory = std::mem::ManuallyDrop::new(TestDirectory { path: PathBuf::from(root) });
    let inputs = existing_test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    match mode.as_str() {
        "genesis_claim" => {
            let plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
            let _manager = initialize_manager(plan, &inputs, &chunk_ranges);
        }
        "successor_claim" => {
            let plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off);
            let _manager = initialize_manager(plan, &inputs, &chunk_ranges);
        }
        "hold_active" => {
            let plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
            let _manager = initialize_manager(plan, &inputs, &chunk_ranges);
            let ready_path =
                std::env::var(TRANSACTION_HELPER_READY_ENVIRONMENT).expect("live-owner ready path is configured");
            std::fs::write(ready_path, b"ready").expect("live owner reports readiness");
            loop {
                std::thread::park_timeout(std::time::Duration::from_mins(1));
            }
        }
        "terminal_claim" => {
            let plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Profile);
            let manager = initialize_manager(plan, &inputs, &chunk_ranges);
            let token =
                manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
            write_chunk(&token, &metadata_store(2), 0..2);
            drop(token);
            assert_no_post_session_cleanup(
                manager
                    .close_completed()
                    .expect("exact coverage closes")
                    .finish()
                    .expect("completed terminal publishes"),
            );
        }
        "two_phenotype_interrupted_terminal" => {
            let plan = two_phenotype_run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
            let manager = OutputManager::open(plan, "# partial two phenotype terminal\n".to_string())
                .expect("two phenotype transaction helper plans")
                .initialize(two_phenotype_headers(&inputs, 2), &chunk_ranges, false, test_association_implementation())
                .expect("two phenotype transaction helper initializes");
            manager.finish_interrupted("SIGTERM").expect("two phenotype interrupted terminal publishes");
        }
        "panic_active" => {
            let plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
            let _manager = initialize_manager(plan, &inputs, &chunk_ranges);
            panic!("injected active output manager panic");
        }
        unsupported_mode => panic!("unsupported transaction helper mode '{unsupported_mode}'"),
    }
    panic!("transaction helper did not reach its configured crash point");
}

#[test]
fn genesis_claim_crash_recovers_from_its_authorized_attempt_directory() {
    let directory = TestDirectory::new("genesis-claim-crash");
    let inputs = test_inputs(&directory);
    let crash = run_crashing_transaction_helper(&directory, "genesis_claim", "after_genesis_claim");
    assert_expected_crash(&crash);

    let output_root = directory.path.join("results");
    let claimed_attempt = attempt_identifier(&output_root, None);
    assert!(output_root.join("attempts").join(&claimed_attempt).is_dir());
    assert_eq!(std::fs::read_dir(output_root.join("attempts")).expect("attempt root exists").count(), 1);

    let resume_plan = run_plan(&directory, &inputs, true, Some(claimed_attempt.clone()), g_plan::TelemetryMode::Off);
    let blocked_error = claim_error(
        Arc::clone(&resume_plan),
        &inputs,
        &single_chunk_ranges(2),
        "surviving owner claim blocks automatic recovery",
    );
    assert_surviving_owner_claim(blocked_error, &output_root);
    let fenced_claim_id = owner_claim_identifier(&output_root);
    let manager = initialize_manager(
        authorize_fenced_owner_claim(resume_plan, fenced_claim_id),
        &inputs,
        &single_chunk_ranges(2),
    );
    let recovery_attempt = attempt_identifier(&output_root, Some(&claimed_attempt));
    assert_ne!(recovery_attempt, claimed_attempt);
    assert!(output_root.join("attempts").join(&recovery_attempt).is_dir());
    assert!(output_root.join("attempts").join(&claimed_attempt).is_dir());
    manager.abort("genesis claim crash recovery test").expect("recovery attempt terminates");
}

#[test]
fn fenced_successor_preserves_referenced_diagnostics_after_publication_crash() {
    let directory = TestDirectory::new("genesis-publication-staging-crash");
    let inputs = test_inputs(&directory);
    let crash = run_crashing_transaction_helper(
        &directory,
        "genesis_claim",
        "after_genesis_publication_before_staging_retirement",
    );
    assert_expected_crash(&crash);

    let output_root = directory.path.join("results");
    let lineage_paths = crate::persistence::lineage::OutputLineagePaths::new(&output_root);
    let published_attempt_id = attempt_identifier(&output_root, None);
    let predecessor_claim_id = owner_claim_identifier(&output_root);
    let staged_attempt_id = lineage_paths
        .owner_staging_attempt(&predecessor_claim_id)
        .expect("published predecessor staging reads")
        .expect("published predecessor staging remains");
    assert_eq!(staged_attempt_id.as_str(), published_attempt_id);
    let predecessor_diagnostics =
        lineage_paths.attempt_directory(&staged_attempt_id).join("diagnostics").join(&predecessor_claim_id);
    assert!(predecessor_diagnostics.is_dir());

    let recovery_plan = authorize_fenced_owner_claim(
        run_plan(&directory, &inputs, true, Some(published_attempt_id), g_plan::TelemetryMode::Off),
        predecessor_claim_id.clone(),
    );
    let recovery_manager = OutputManager::open(recovery_plan, "# exact recovery\n".to_string())
        .expect("exact recovery plans")
        .claim(vec![header(&inputs, 2)], &single_chunk_ranges(2), false)
        .expect("exact fence claims a recovery attempt");
    assert!(predecessor_diagnostics.is_dir(), "referenced writable diagnostics must survive fencing");
    assert_eq!(
        lineage_paths.owner_staging_attempt(&predecessor_claim_id).expect("retired predecessor staging reads"),
        None
    );
    assert!(recovery_manager.diagnostics_directory().expect("recovery diagnostics exist").is_dir());
    recovery_manager.abort_before_activation().expect("recovery staging cleans and releases");
    assert_owner_authority_released(&output_root);
}

#[test]
fn fenced_recovery_preserves_successor_diagnostics_before_staging_retirement() {
    let directory = TestDirectory::new("successor-publication-staging-crash");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    initialize_manager(initial_plan, &inputs, &chunk_ranges)
        .finish_interrupted("SIGTERM")
        .expect("predecessor interrupts");
    let output_root = directory.path.join("results");
    let parent_attempt_id = attempt_identifier(&output_root, None);

    let crash = run_crashing_transaction_helper(
        &directory,
        "successor_claim",
        "after_successor_publication_before_staging_retirement",
    );
    assert_expected_crash(&crash);

    let lineage_paths = crate::persistence::lineage::OutputLineagePaths::new(&output_root);
    let successor_attempt_id = attempt_identifier(&output_root, Some(&parent_attempt_id));
    let predecessor_claim_id = owner_claim_identifier(&output_root);
    let staged_attempt_id = lineage_paths
        .owner_staging_attempt(&predecessor_claim_id)
        .expect("published successor staging reads")
        .expect("published successor staging remains");
    assert_eq!(staged_attempt_id.as_str(), successor_attempt_id);
    let predecessor_diagnostics =
        lineage_paths.attempt_directory(&staged_attempt_id).join("diagnostics").join(&predecessor_claim_id);
    assert!(predecessor_diagnostics.is_dir());

    let recovery_plan = authorize_fenced_owner_claim(
        run_plan(&directory, &inputs, true, Some(successor_attempt_id), g_plan::TelemetryMode::Off),
        predecessor_claim_id.clone(),
    );
    let recovery_manager = OutputManager::open(recovery_plan, "# exact successor recovery\n".to_string())
        .expect("exact successor recovery plans")
        .claim(single_phenotype_headers(&inputs, &chunk_ranges), &chunk_ranges, false)
        .expect("exact fence claims another recovery attempt");
    assert!(predecessor_diagnostics.is_dir(), "referenced successor diagnostics must survive fencing");
    assert_eq!(
        lineage_paths.owner_staging_attempt(&predecessor_claim_id).expect("retired predecessor staging reads"),
        None
    );
    recovery_manager.abort_before_activation().expect("recovery staging cleans and releases");
    assert_owner_authority_released(&output_root);
}

#[test]
fn owner_claim_publication_crash_windows_remain_fail_closed_and_recoverable() {
    for (failpoint, claim_published) in [("before_owner_claim_link", false), ("after_owner_claim_link", true)] {
        let directory = TestDirectory::new(failpoint);
        let inputs = test_inputs(&directory);
        let crash = run_crashing_transaction_helper(&directory, "genesis_claim", failpoint);
        assert_expected_crash(&crash);

        let output_root = directory.path.join("results");
        assert_eq!(owner_claim_candidate_temporary_paths(&output_root).len(), 1);
        assert_eq!(output_root.join(".g-output/session.claim.json").exists(), claim_published);
        assert!(!output_root.join(".g-output/genesis.json").exists());
        let fresh_plan = if claim_published {
            let blocked_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
            let blocked_error = claim_error(
                blocked_plan,
                &inputs,
                &single_chunk_ranges(2),
                "post-link crash leaves a typed surviving claim",
            );
            assert_surviving_owner_claim(blocked_error, &output_root);
            authorize_fenced_owner_claim(
                run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off),
                owner_claim_identifier(&output_root),
            )
        } else {
            run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off)
        };

        let manager = initialize_manager(fresh_plan, &inputs, &single_chunk_ranges(2));
        manager.abort("owner claim publication crash recovery test").expect("fresh attempt terminates");
        assert_owner_authority_released(&output_root);
        assert_eq!(
            owner_claim_candidate_temporary_paths(&output_root).len(),
            1,
            "non-authoritative crashed candidates remain inspectable but do not block ownership"
        );
    }
}

#[test]
fn panic_unwind_leaves_a_fail_closed_owner_claim() {
    let directory = TestDirectory::new("panic-active");
    let inputs = test_inputs(&directory);
    let output = run_crashing_transaction_helper(&directory, "panic_active", "unused");
    assert!(!output.status.success());
    assert_ne!(output.status.code(), Some(86));
    assert!(String::from_utf8_lossy(&output.stderr).contains("injected active output manager panic"));

    let output_root = directory.path.join("results");
    let attempt = attempt_identifier(&output_root, None);
    assert!(output_root.join(".g-output/session.claim.json").is_file());
    assert!(!output_root.join(".g-output/terminal-finalizations").join(format!("{attempt}.json")).exists());
    let recovery_plan = run_plan(&directory, &inputs, true, Some(attempt), g_plan::TelemetryMode::Off);
    let blocked_error = claim_error(
        Arc::clone(&recovery_plan),
        &inputs,
        &single_chunk_ranges(2),
        "panic claim blocks automatic recovery",
    );
    assert_surviving_owner_claim(blocked_error, &output_root);

    let recovery_plan = authorize_fenced_owner_claim(recovery_plan, owner_claim_identifier(&output_root));
    let manager = initialize_manager(recovery_plan, &inputs, &single_chunk_ranges(2));
    manager.abort("panic recovery test").expect("externally fenced panic recovery terminates");
}

#[test]
fn nonterminal_recovery_rejects_a_dangling_manifest_symlink_as_present_but_invalid() {
    let directory = TestDirectory::new("nonterminal-dangling-manifest");
    let inputs = test_inputs(&directory);
    let output = run_crashing_transaction_helper(&directory, "panic_active", "unused");
    assert!(!output.status.success());
    assert_ne!(output.status.code(), Some(86));

    let output_root = directory.path.join("results");
    let attempt = attempt_identifier(&output_root, None);
    let manifest_path =
        output_root.join("attempts").join(&attempt).join(OUTPUT_DIRECTORY_NAME).join("run_manifest.json");
    std::fs::remove_file(&manifest_path).expect("running manifest removes");
    std::os::unix::fs::symlink("missing-manifest-target", &manifest_path).expect("dangling manifest symlink installs");

    let recovery_plan = run_plan(&directory, &inputs, true, Some(attempt), g_plan::TelemetryMode::Off);
    let recovery_plan = authorize_fenced_owner_claim(recovery_plan, owner_claim_identifier(&output_root));
    let error = OutputManager::open(recovery_plan, "# dangling manifest recovery\n".to_string())
        .expect("recovery manager plans")
        .initialize(vec![header(&inputs, 2)], &single_chunk_ranges(2), false, test_association_implementation())
        .err()
        .expect("dangling nonterminal manifest symlink is rejected");
    assert!(error.to_string().contains("must not be a symbolic link"));
    let surviving_owner_error = crate::persistence::lineage::OutputLineagePaths::new(&output_root)
        .reject_surviving_owner_claim()
        .expect_err("invalid pre-owner manifest validation leaves the prior owner fail-closed");
    assert_surviving_owner_claim(surviving_owner_error, &output_root);
}

#[test]
fn successor_claim_crash_is_exactly_recoverable_without_an_orphan_candidate() {
    let directory = TestDirectory::new("successor-claim-crash");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let manager = initialize_manager(initial_plan, &inputs, &chunk_ranges);
    manager.finish_interrupted("SIGTERM").expect("initial attempt is interrupted");
    let output_root = directory.path.join("results");
    let initial_attempt = attempt_identifier(&output_root, None);

    let crash = run_crashing_transaction_helper(&directory, "successor_claim", "after_successor_claim");
    assert_expected_crash(&crash);
    let claimed_successor = attempt_identifier(&output_root, Some(&initial_attempt));
    assert!(output_root.join("attempts").join(&claimed_successor).is_dir());

    let exact_recovery_plan =
        run_plan(&directory, &inputs, true, Some(claimed_successor.clone()), g_plan::TelemetryMode::Off);
    let blocked_error = claim_error(
        Arc::clone(&exact_recovery_plan),
        &inputs,
        &chunk_ranges,
        "surviving owner claim blocks exact recovery",
    );
    assert_surviving_owner_claim(blocked_error, &output_root);
    let exact_recovery_plan = authorize_fenced_owner_claim(exact_recovery_plan, owner_claim_identifier(&output_root));
    let manager = initialize_manager(exact_recovery_plan, &inputs, &chunk_ranges);
    let recovery_attempt = attempt_identifier(&output_root, Some(&claimed_successor));
    assert_ne!(recovery_attempt, claimed_successor);
    assert!(output_root.join("attempts").join(&recovery_attempt).is_dir());
    assert!(output_root.join("attempts").join(&claimed_successor).is_dir());
    manager.abort("successor claim crash recovery test").expect("exact recovery attempt terminates");
}

#[test]
fn reused_receipt_crash_retains_exact_implementation_authority() {
    let directory = TestDirectory::new("reused-receipt-implementation-crash");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let initial_manager = initialize_manager(initial_plan, &inputs, &chunk_ranges);
    let token =
        initial_manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    write_chunk(&token, &metadata_store(2), 0..2);
    drop(token);
    initial_manager.finish_interrupted("SIGTERM").expect("initial attempt is interrupted");

    let output_root = directory.path.join("results");
    let initial_attempt = attempt_identifier(&output_root, None);
    let crash =
        run_crashing_transaction_helper(&directory, "successor_claim", "after_reused_receipts_before_manifest_update");
    assert_expected_crash(&crash);
    let crashed_successor = attempt_identifier(&output_root, Some(&initial_attempt));
    let crashed_run_directory = output_root.join("attempts").join(&crashed_successor).join(OUTPUT_DIRECTORY_NAME);
    let crashed_manifest = read_json(&crashed_run_directory.join("run_manifest.json"));
    assert_eq!(crashed_manifest["status"], "running");
    assert_eq!(crashed_manifest["runtime"]["association_implementation"]["jax_version"], "test-jax");
    assert!(
        std::fs::read_dir(crashed_run_directory.join("commits"))
            .expect("reused receipt directory reads")
            .next()
            .is_some(),
        "the crash point must follow durable receipt reuse"
    );

    let lineage_paths = crate::persistence::lineage::OutputLineagePaths::new(&output_root);
    let crashed_attempt_identifier = crate::persistence::identifier::AttemptIdentifier::parse(&crashed_successor)
        .expect("crashed successor identifier parses");
    let next_successor_path = lineage_paths.outcome_path(&crashed_attempt_identifier);
    let recovery_plan = authorize_fenced_owner_claim(
        run_plan(&directory, &inputs, true, Some(crashed_successor.clone()), g_plan::TelemetryMode::Off),
        owner_claim_identifier(&output_root),
    );
    let claimed_manager = OutputManager::open(recovery_plan, "# mismatched crash recovery\n".to_string())
        .expect("mismatched recovery plans")
        .claim(vec![header(&inputs, 2)], &chunk_ranges, false)
        .expect("exact fence claims recovery");
    let activation_error = claimed_manager
        .activate_with_deferred_completed_noop_cleanup(test_raw_cuda_association_implementation_with_digests('a', 'c'))
        .err()
        .expect("changed implementation is rejected");
    let crate::OutputActivationFailureParts { source, rollback } = activation_error.into_parts();
    assert!(matches!(source, crate::OutputError::CurrentAssociationImplementationMismatch));
    assert!(!next_successor_path.exists(), "mismatched recovery cannot publish another successor");
    let mut rollback = rollback.expect("mismatched recovery retains rollback authority");
    rollback.abort_before_activation().expect("mismatched recovery rolls back");
    assert_owner_authority_released(&output_root);

    let matching_plan =
        run_plan(&directory, &inputs, true, Some(crashed_successor.clone()), g_plan::TelemetryMode::Off);
    let matching_manager = initialize_manager(matching_plan, &inputs, &chunk_ranges);
    let recovery_attempt = attempt_identifier(&output_root, Some(&crashed_successor));
    assert_ne!(recovery_attempt, crashed_successor);
    matching_manager.abort("matching implementation crash recovery").expect("matching recovery terminates");
    assert_owner_authority_released(&output_root);
}

#[test]
fn nonterminal_receipts_without_an_implementation_manifest_fail_closed() {
    let directory = TestDirectory::new("manifestless-reused-receipts");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let initial_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let initial_manager = initialize_manager(initial_plan, &inputs, &chunk_ranges);
    let token =
        initial_manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("delivery token builds");
    write_chunk(&token, &metadata_store(2), 0..2);
    drop(token);
    initial_manager.finish_interrupted("SIGTERM").expect("initial attempt is interrupted");

    let output_root = directory.path.join("results");
    let initial_attempt = attempt_identifier(&output_root, None);
    let crash =
        run_crashing_transaction_helper(&directory, "successor_claim", "after_reused_receipts_before_manifest_update");
    assert_expected_crash(&crash);
    let crashed_successor = attempt_identifier(&output_root, Some(&initial_attempt));
    let crashed_run_directory = output_root.join("attempts").join(&crashed_successor).join(OUTPUT_DIRECTORY_NAME);
    assert!(
        std::fs::read_dir(crashed_run_directory.join("commits"))
            .expect("reused receipt directory reads")
            .next()
            .is_some(),
        "the invalid fixture must contain a durable reused receipt"
    );
    let crashed_manifest = crashed_run_directory.join("run_manifest.json");
    std::fs::remove_file(&crashed_manifest).expect("implementation-bearing manifest removes");

    let lineage_paths = crate::persistence::lineage::OutputLineagePaths::new(&output_root);
    let crashed_attempt_identifier = crate::persistence::identifier::AttemptIdentifier::parse(&crashed_successor)
        .expect("crashed successor identifier parses");
    let next_successor_path = lineage_paths.outcome_path(&crashed_attempt_identifier);
    let recovery_plan = authorize_fenced_owner_claim(
        run_plan(&directory, &inputs, true, Some(crashed_successor), g_plan::TelemetryMode::Off),
        owner_claim_identifier(&output_root),
    );
    let error = claim_error(recovery_plan, &inputs, &chunk_ranges, "manifestless durable receipts must fail closed");
    let crate::OutputError::InvalidInput(message) = error else {
        panic!("expected invalid manifest state");
    };
    assert!(message.contains("has durable receipts but is missing its implementation-bearing manifest"));
    assert!(!next_successor_path.exists(), "invalid recovery cannot publish another successor");
    assert_owner_authority_released(&output_root);
}

#[test]
fn live_manager_claim_blocks_a_second_process_and_survives_owner_kill() {
    use std::process::Stdio;

    let directory = TestDirectory::new("live-manager-claim");
    let inputs = test_inputs(&directory);
    let ready_path = directory.path.join("live-owner.ready");
    let mut owner = std::process::Command::new(std::env::current_exe().expect("current test executable resolves"))
        .args(["--exact", TRANSACTION_HELPER_TEST_NAME, "--nocapture"])
        .env(TRANSACTION_HELPER_MODE_ENVIRONMENT, "hold_active")
        .env(TRANSACTION_HELPER_ROOT_ENVIRONMENT, &directory.path)
        .env(TRANSACTION_HELPER_READY_ENVIRONMENT, &ready_path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("live output owner starts");
    let ready_deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !ready_path.exists() && std::time::Instant::now() < ready_deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(ready_path.exists(), "live output owner did not report readiness");

    let output_root = directory.path.join("results");
    let active_attempt = attempt_identifier(&output_root, None);
    let active_claim_id = owner_claim_identifier(&output_root);
    let before_contender = regular_file_snapshot(&output_root);
    let contender_plan = run_plan(&directory, &inputs, true, Some(active_attempt.clone()), g_plan::TelemetryMode::Off);
    let contender_error =
        claim_error(contender_plan, &inputs, &single_chunk_ranges(2), "live owner claim blocks contender");
    assert_surviving_owner_claim(contender_error, &output_root);
    assert_eq!(regular_file_snapshot(&output_root), before_contender);

    owner.kill().expect("live output owner is killed");
    owner.wait().expect("killed output owner is reaped");
    let recovery_plan = run_plan(&directory, &inputs, true, Some(active_attempt.clone()), g_plan::TelemetryMode::Off);
    let stale_error = claim_error(
        Arc::clone(&recovery_plan),
        &inputs,
        &single_chunk_ranges(2),
        "killed owner leaves a fail-closed claim",
    );
    assert_surviving_owner_claim(stale_error, &output_root);
    assert_eq!(regular_file_snapshot(&output_root), before_contender);

    let wrong_fence_plan = authorize_fenced_owner_claim(
        run_plan(&directory, &inputs, true, Some(active_attempt.clone()), g_plan::TelemetryMode::Off),
        "owner-wrong-fence".to_string(),
    );
    let wrong_fence_error = claim_error(
        wrong_fence_plan,
        &inputs,
        &single_chunk_ranges(2),
        "mismatched external fence cannot remove the surviving claim",
    );
    assert_surviving_owner_claim(wrong_fence_error, &output_root);
    assert_eq!(regular_file_snapshot(&output_root), before_contender);

    let recovery_plan = authorize_fenced_owner_claim(recovery_plan, active_claim_id);
    let manager = initialize_manager(recovery_plan, &inputs, &single_chunk_ranges(2));
    manager.abort("externally fenced killed-owner recovery test").expect("externally fenced recovery succeeds");
}

#[cfg(unix)]
#[test]
fn path_aliases_to_one_output_root_fail_before_owner_transition() {
    let directory = TestDirectory::new("owner-path-alias");
    let inputs = test_inputs(&directory);
    let owner_plan = run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off);
    let owner = initialize_manager(owner_plan, &inputs, &single_chunk_ranges(2));
    let output_root = directory.path.join("results");
    let attempt = attempt_identifier(&output_root, None);
    let before_contender = regular_file_snapshot(&output_root);

    let alias_path = directory.path.join("root-alias");
    std::os::unix::fs::symlink(&directory.path, &alias_path).expect("test output-root alias is created");
    let alias_directory = std::mem::ManuallyDrop::new(TestDirectory { path: alias_path });
    let contender_plan = run_plan(&alias_directory, &inputs, true, Some(attempt), g_plan::TelemetryMode::Off);
    let alias_output_root = alias_directory.path.join("results");
    let error = claim_error(
        contender_plan,
        &inputs,
        &single_chunk_ranges(2),
        "path alias fails strict pre-owner manifest validation",
    );
    assert!(error.to_string().contains("effective_config"));
    assert_eq!(
        output_root.join(".g-output/session.claim.json").metadata().expect("owner claim metadata reads").ino(),
        alias_output_root
            .join(".g-output/session.claim.json")
            .metadata()
            .expect("aliased owner claim metadata reads")
            .ino()
    );
    assert_eq!(regular_file_snapshot(&output_root), before_contender);

    owner.abort("path alias claim test").expect("owner terminates normally");
}

#[test]
fn partially_materialized_interrupted_terminal_hints_and_recovers_both_phenotypes() {
    let directory = TestDirectory::new("partial-two-phenotype-terminal");
    let inputs = test_inputs(&directory);
    let crash = run_crashing_transaction_helper(
        &directory,
        "two_phenotype_interrupted_terminal",
        "after_terminal_run_materialization",
    );
    assert_expected_crash(&crash);

    let output_root = directory.path.join("results");
    let attempt_id = attempt_identifier(&output_root, None);
    let attempt_directory = output_root.join("attempts").join(&attempt_id);
    let first_manifest_path = attempt_directory.join(OUTPUT_DIRECTORY_NAME).join("run_manifest.json");
    let second_manifest_path = attempt_directory.join(SECOND_OUTPUT_DIRECTORY_NAME).join("run_manifest.json");
    assert_eq!(read_json(&first_manifest_path)["status"], "interrupted");
    assert_eq!(read_json(&second_manifest_path)["status"], "running");

    let resume_plan = two_phenotype_run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off);
    let hint_manager = OutputManager::open(Arc::clone(&resume_plan), "# partial terminal agreement\n".to_string())
        .expect("partial terminal resume plans");
    assert_eq!(
        hint_manager.existing_output_resume_agreement().expect("partially materialized manifests validate"),
        Some(expected_resume_agreement(&inputs, g_plan::GpuGenotypeFormat::Packed8))
    );
    drop(hint_manager);

    let recovery_plan = authorize_fenced_owner_claim(resume_plan, owner_claim_identifier(&output_root));
    let recovery_manager = OutputManager::open(recovery_plan, "# fenced partial terminal recovery\n".to_string())
        .expect("fenced recovery plans")
        .initialize(
            two_phenotype_headers(&inputs, 2),
            &single_chunk_ranges(2),
            false,
            test_association_implementation(),
        )
        .expect("fenced recovery initializes a successor");
    assert_eq!(read_json(&first_manifest_path)["status"], "interrupted");
    assert_eq!(read_json(&second_manifest_path)["status"], "interrupted");
    assert!(output_root.join(".g-output/terminal-finalizations").join(format!("{attempt_id}.json")).is_file());
    recovery_manager.abort("partial terminal recovery test").expect("recovery successor terminates");
    assert_owner_authority_released(&output_root);
}

#[test]
fn whole_plan_resume_agreement_pauses_and_reinspects_once_after_all_manifest_reads() {
    let directory = TestDirectory::new("resume-agreement-lineage-race");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    OutputManager::open(
        two_phenotype_run_plan(&directory, &inputs, false, None, g_plan::TelemetryMode::Off),
        "# two phenotype interrupted terminal\n".to_string(),
    )
    .expect("initial manager plans")
    .initialize(two_phenotype_headers(&inputs, 2), &chunk_ranges, false, test_association_implementation())
    .expect("initial manager activates")
    .finish_interrupted("SIGTERM")
    .expect("interrupted terminal publishes");

    let resume_plan = two_phenotype_run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off);
    let mut hint_manager = OutputManager::open(Arc::clone(&resume_plan), "# paused resume agreement\n".to_string())
        .expect("agreement manager plans");
    let hint_control = hint_manager.install_manifest_hint_pause_for_test().expect("hint pause installs");
    std::thread::scope(|scope| {
        let hint_thread = scope.spawn(move || hint_manager.existing_output_resume_agreement());
        hint_control.wait_until_reached().expect("hint pauses before lineage reinspection");
        let successor = OutputManager::open(resume_plan, "# successor\n".to_string())
            .expect("successor plans")
            .initialize(two_phenotype_headers(&inputs, 2), &chunk_ranges, false, test_association_implementation())
            .expect("successor initializes");
        hint_control.resume();

        let hint_error = hint_thread
            .join()
            .expect("hint thread does not panic")
            .expect_err("lineage successor invalidates the prior hint read");
        assert!(matches!(hint_error, crate::OutputError::ConcurrentLineageUpdate { .. }));
        assert_eq!(hint_control.reach_count(), 1);
        assert_eq!(hint_control.final_inspect_count(), 1);
        successor.abort("manifest hint race cleanup").expect("successor terminates");
    });
    assert_owner_authority_released(&directory.path.join("results"));
}

#[test]
fn pending_completed_terminal_is_finalized_after_a_process_crash() {
    let directory = TestDirectory::new("terminal-claim-crash");
    let inputs = test_inputs(&directory);
    let crash = run_crashing_transaction_helper(&directory, "terminal_claim", "after_terminal_claim");
    assert_expected_crash(&crash);

    let output_root = directory.path.join("results");
    let attempt = attempt_identifier(&output_root, None);
    let run_directory = output_root.join("attempts").join(&attempt).join(OUTPUT_DIRECTORY_NAME);
    assert_eq!(read_json(&run_directory.join("run_manifest.json"))["status"], "running");
    assert!(run_directory.join("output_stage_timings.json").is_file());
    assert!(output_root.join(".g-output/outcomes").join(format!("{attempt}.json")).is_file());
    assert!(!output_root.join(".g-output/terminal-finalizations").join(format!("{attempt}.json")).exists());

    let resume_plan = run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Profile);
    let hint_manager = OutputManager::open(Arc::clone(&resume_plan), "# pending-terminal agreement\n".to_string())
        .expect("hint plans");
    assert_eq!(
        hint_manager.existing_output_resume_agreement().expect("bound running manifest gives agreement"),
        Some(expected_resume_agreement(&inputs, g_plan::GpuGenotypeFormat::Packed8))
    );
    drop(hint_manager);
    let blocked_error = claim_error(
        Arc::clone(&resume_plan),
        &inputs,
        &single_chunk_ranges(2),
        "surviving owner claim blocks pending-terminal recovery",
    );
    assert_surviving_owner_claim(blocked_error, &output_root);
    let resume_plan = authorize_fenced_owner_claim(resume_plan, owner_claim_identifier(&output_root));
    let manager = initialize_manager(resume_plan, &inputs, &single_chunk_ranges(2));
    let token = manager.delivery_token_for_phenotypes(&[PHENOTYPE_NAME.to_string()]).expect("completed token builds");
    assert!(token.is_read_only());
    drop(token);
    assert_no_post_session_cleanup(
        manager
            .close_completed()
            .expect("recovered completion reverifies")
            .finish()
            .expect("recovered terminal finishes"),
    );

    assert_eq!(read_json(&run_directory.join("run_manifest.json"))["status"], "completed");
    assert!(output_root.join(".g-output/terminal-finalizations").join(format!("{attempt}.json")).is_file());
    assert!(!output_root.join(".g-output/successors").join(format!("{attempt}.json")).exists());
    assert_eq!(std::fs::read_dir(output_root.join("attempts")).expect("attempt root reads").count(), 1);
}

#[test]
fn pending_terminal_claim_can_roll_back_before_activation() {
    let directory = TestDirectory::new("pending-terminal-preactivation-rollback");
    let inputs = test_inputs(&directory);
    let chunk_ranges = single_chunk_ranges(2);
    let crash = run_crashing_transaction_helper(&directory, "terminal_claim", "after_terminal_claim");
    assert_expected_crash(&crash);

    let output_root = directory.path.join("results");
    let source_attempt_id = attempt_identifier(&output_root, None);
    let resume_plan = authorize_fenced_owner_claim(
        run_plan(&directory, &inputs, true, None, g_plan::TelemetryMode::Off),
        owner_claim_identifier(&output_root),
    );
    let claimed_manager = OutputManager::open(resume_plan, "# pending terminal rejection\n".to_string())
        .expect("pending-terminal recovery plans")
        .claim(single_phenotype_headers(&inputs, &chunk_ranges), &chunk_ranges, false)
        .expect("pending-terminal recovery claims");
    let activation_error =
        claimed_manager.reject_activation(crate::OutputError::Runtime("backend projection rejected".to_string()));
    let crate::OutputActivationFailureParts { source, rollback } = activation_error.into_parts();
    assert!(source.to_string().contains("backend projection rejected"));
    rollback
        .expect("pending-terminal rejection returns rollback authority")
        .abort_before_activation()
        .expect("pending-terminal rejection rolls back against the absent successor path");

    assert!(output_root.join(".g-output/outcomes").join(format!("{source_attempt_id}.json")).is_file());
    assert!(!output_root.join(".g-output/terminal-finalizations").join(format!("{source_attempt_id}.json")).exists());
    assert!(!output_root.join(".g-output/successors").join(format!("{source_attempt_id}.json")).exists());
    assert_owner_authority_released(&output_root);
}
