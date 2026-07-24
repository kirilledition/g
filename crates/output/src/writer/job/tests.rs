use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{ArrayRef, Float32Array};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use g_genotype_contracts::{
    ChunkOutputStatistics, NullableFloat32Column, VariantMetadataColumns, VariantMetadataStore,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::{UnpublishedPart, transaction_temporary_part_path, write_regenie_step2_chunk_job_with_io};
use crate::chunk::{NativeChunkHandle, NativeVariantMetadataHandle};
use crate::persistence::io::{FileIntegrity, NoReplacePublication, OutputIo, file_size_and_sha256};
use crate::persistence::model::{OutputChunkCommit, OutputPartBinding, OutputTransactionIdentifier};
use crate::persistence::receipt::{OutputPartFooter, read_part_receipt, receipt_path};
use crate::writer::{OutputPartPublication, RegenieStep2ChunkJob, RegenieStep2ChunkWriteBatch, streams};

const CHUNK_FILE_NAME: &str = "part_000000000.parquet";
const TEST_TRANSACTION_IDENTIFIER: &str = "test-transaction";
const PARQUET_MAGIC: &[u8] = b"PAR1";

struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn new(label: &str) -> Self {
        static DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);
        let sequence = DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).expect("test time is after Unix epoch").as_nanos();
        let path = std::env::temp_dir()
            .join(format!("g-output-part-publication-{label}-{}-{timestamp}-{sequence}", std::process::id()));
        std::fs::create_dir_all(&path).expect("test directory is created");
        Self { path }
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FaultPoint {
    CreateNew,
    FirstPhysicalParquetWrite,
    ParquetFinalize,
    FileSync,
    Rename,
    AfterFinalLink,
    DirectorySync,
    Metadata,
    Cleanup,
}

impl FaultPoint {
    const fn label(self) -> &'static str {
        match self {
            Self::CreateNew => "create-new",
            Self::FirstPhysicalParquetWrite => "first-physical-Parquet-write",
            Self::ParquetFinalize => "Parquet-finalize",
            Self::FileSync => "file-sync",
            Self::Rename => "rename",
            Self::AfterFinalLink => "after-final-link",
            Self::DirectorySync => "directory-sync",
            Self::Metadata => "metadata",
            Self::Cleanup => "cleanup",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum RecordedOperation {
    CreateNew(PathBuf),
    ParquetHeader(PathBuf),
    ParquetFinalize(PathBuf),
    FileSync(PathBuf),
    Rename { source_path: PathBuf, destination_path: PathBuf },
    DirectorySync(PathBuf),
    Metadata(PathBuf),
    Cleanup(PathBuf),
}

struct PublicationFailureCase {
    fault_point: FaultPoint,
    expected_operation: &'static str,
    final_exists: bool,
}

struct ExpectedPartPaths {
    temporary_path: PathBuf,
    final_path: PathBuf,
}

struct RecordingState {
    operations: Mutex<Vec<RecordedOperation>>,
    fault_points: Vec<FaultPoint>,
}

struct RecordingOutputIo {
    state: Arc<RecordingState>,
}

impl RecordingOutputIo {
    fn new(fault_points: &[FaultPoint]) -> Self {
        Self {
            state: Arc::new(RecordingState { operations: Mutex::new(Vec::new()), fault_points: fault_points.to_vec() }),
        }
    }

    fn operations(&self) -> Vec<RecordedOperation> {
        self.state.operations.lock().expect("operation recorder remains available").clone()
    }
}

impl RecordingState {
    fn record(&self, operation: RecordedOperation) {
        self.operations.lock().expect("operation recorder remains available").push(operation);
    }

    fn should_fail(&self, fault_point: FaultPoint) -> bool {
        self.fault_points.contains(&fault_point)
    }
}

struct RecordingFile {
    file: File,
    path: PathBuf,
    state: Arc<RecordingState>,
    observed_parquet_header: bool,
    observed_parquet_finalize: bool,
}

impl Write for RecordingFile {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let header_was_already_observed = self.observed_parquet_header;
        if !self.observed_parquet_header && buffer.starts_with(PARQUET_MAGIC) {
            self.observed_parquet_header = true;
            self.state.record(RecordedOperation::ParquetHeader(self.path.clone()));
        }
        if buffer.starts_with(PARQUET_MAGIC) && self.state.should_fail(FaultPoint::FirstPhysicalParquetWrite) {
            return Err(injected_error(FaultPoint::FirstPhysicalParquetWrite));
        }
        let contains_final_magic =
            buffer.ends_with(PARQUET_MAGIC) && (header_was_already_observed || buffer.len() > PARQUET_MAGIC.len());
        if contains_final_magic {
            if !self.observed_parquet_finalize {
                self.observed_parquet_finalize = true;
                self.state.record(RecordedOperation::ParquetFinalize(self.path.clone()));
            }
            if self.state.should_fail(FaultPoint::ParquetFinalize) {
                return Err(injected_error(FaultPoint::ParquetFinalize));
            }
        }
        self.file.write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

impl OutputIo for RecordingOutputIo {
    type File = RecordingFile;

    fn create_new_file(&self, path: &Path) -> io::Result<Self::File> {
        self.state.record(RecordedOperation::CreateNew(path.to_path_buf()));
        if self.state.should_fail(FaultPoint::CreateNew) {
            return Err(injected_error(FaultPoint::CreateNew));
        }
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        Ok(RecordingFile {
            file,
            path: path.to_path_buf(),
            state: Arc::clone(&self.state),
            observed_parquet_header: false,
            observed_parquet_finalize: false,
        })
    }

    fn sync_file(&self, file: &Self::File, path: &Path) -> io::Result<()> {
        self.state.record(RecordedOperation::FileSync(path.to_path_buf()));
        if self.state.should_fail(FaultPoint::FileSync) {
            return Err(injected_error(FaultPoint::FileSync));
        }
        file.file.sync_all()
    }

    fn publish_file_no_replace(&self, source_path: &Path, destination_path: &Path) -> io::Result<NoReplacePublication> {
        self.state.record(RecordedOperation::Rename {
            source_path: source_path.to_path_buf(),
            destination_path: destination_path.to_path_buf(),
        });
        if self.state.should_fail(FaultPoint::Rename) {
            return Err(injected_error(FaultPoint::Rename));
        }
        match std::fs::hard_link(source_path, destination_path) {
            Ok(()) => {
                if self.state.should_fail(FaultPoint::AfterFinalLink) {
                    return Err(injected_error(FaultPoint::AfterFinalLink));
                }
                std::fs::remove_file(source_path)?;
                Ok(NoReplacePublication::Created)
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(NoReplacePublication::AlreadyExists),
            Err(error) => Err(error),
        }
    }

    fn sync_directory(&self, path: &Path) -> io::Result<()> {
        self.state.record(RecordedOperation::DirectorySync(path.to_path_buf()));
        if self.state.should_fail(FaultPoint::DirectorySync) {
            return Err(injected_error(FaultPoint::DirectorySync));
        }
        File::open(path)?.sync_all()
    }

    fn file_integrity(&self, path: &Path) -> io::Result<FileIntegrity> {
        self.state.record(RecordedOperation::Metadata(path.to_path_buf()));
        if self.state.should_fail(FaultPoint::Metadata) {
            return Err(injected_error(FaultPoint::Metadata));
        }
        file_size_and_sha256(path).map_err(io::Error::other)
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.state.record(RecordedOperation::Cleanup(path.to_path_buf()));
        if self.state.should_fail(FaultPoint::Cleanup) {
            return Err(injected_error(FaultPoint::Cleanup));
        }
        std::fs::remove_file(path)
    }
}

fn injected_error(fault_point: FaultPoint) -> io::Error {
    io::Error::other(format!("injected {} failure", fault_point.label()))
}

fn test_publication(parts_directory: &Path) -> OutputPartPublication {
    test_publication_with_identifier(parts_directory, TEST_TRANSACTION_IDENTIFIER)
}

fn test_publication_with_identifier(parts_directory: &Path, temporary_identifier: &str) -> OutputPartPublication {
    OutputPartPublication {
        parts_directory: parts_directory.to_path_buf(),
        commits_directory: parts_directory.join("commits"),
        temporary_identifier: OutputTransactionIdentifier::for_test(temporary_identifier),
        binding: OutputPartBinding {
            run_set_id: "run-set-test".to_string(),
            attempt_id: OutputTransactionIdentifier::for_test("attempt-test"),
            phenotype_name: "trait-a".to_string(),
            execution_plan_sha256: "a".repeat(64),
            chunk_plan_sha256: "b".repeat(64),
        },
    }
}

fn test_part_footer() -> OutputPartFooter {
    OutputPartFooter::new(
        &test_publication(Path::new("unused")).binding,
        CHUNK_FILE_NAME.to_string(),
        vec![OutputChunkCommit {
            chunk_identifier: 0,
            variant_start_index: 0,
            variant_stop_index: 1,
            row_count: 1,
            chunk_file_name: CHUNK_FILE_NAME.to_string(),
        }],
    )
    .expect("test part footer builds")
}

fn test_write_batch() -> RegenieStep2ChunkWriteBatch {
    let dictionary: Box<[Arc<str>]> = ["22", "A", "C"].map(Arc::<str>::from).into();
    let metadata_store = Arc::new(
        VariantMetadataStore::from_parts(
            dictionary,
            vec![0_u32].into_boxed_slice(),
            "variant-0".to_string().into_boxed_str(),
            vec![0_u32, 9].into_boxed_slice(),
            vec![100_i64].into_boxed_slice(),
            vec![1_u32].into_boxed_slice(),
            vec![2_u32].into_boxed_slice(),
        )
        .expect("test metadata store is valid"),
    );
    let metadata = VariantMetadataColumns::new(metadata_store, 0..1).expect("test metadata range is valid");
    let metadata_handle = NativeVariantMetadataHandle::try_new(&metadata).expect("test metadata handle is valid");
    let chunk_handle = NativeChunkHandle::try_new(
        metadata_handle,
        ChunkOutputStatistics {
            allele_one_frequency: vec![0.25],
            observation_count: vec![12],
            info_score: NullableFloat32Column { values: vec![0.9], validity_bytes: vec![1] },
        },
        0,
    )
    .expect("test chunk is valid");
    RegenieStep2ChunkWriteBatch {
        chunk_file_name: CHUNK_FILE_NAME.to_string(),
        chunks: vec![RegenieStep2ChunkJob {
            chunk_handle,
            beta: float_array(0.5),
            se: float_array(0.25),
            chisq: float_array(4.0),
            log10p: float_array(2.0),
            correction_code: None,
        }],
    }
}

fn float_array(value: f32) -> ArrayRef {
    Arc::new(Float32Array::from(vec![value]))
}

fn expected_paths(parts_directory: &Path) -> ExpectedPartPaths {
    let publication = test_publication(parts_directory);
    ExpectedPartPaths {
        temporary_path: transaction_temporary_part_path(
            parts_directory,
            CHUNK_FILE_NAME,
            &publication.temporary_identifier,
        ),
        final_path: parts_directory.join(CHUNK_FILE_NAME),
    }
}

fn assert_readable_parquet_part(final_path: &Path) {
    let input_file = File::open(final_path).expect("published part opens");
    let builder = ParquetRecordBatchReaderBuilder::try_new(input_file).expect("published footer is readable");
    assert_eq!(builder.schema().fields(), crate::schema::REGENIE_STEP2_CHUNK_SCHEMA.fields());
    let footer_metadata = builder.metadata().file_metadata().key_value_metadata().expect("footer metadata exists");
    assert!(
        footer_metadata.iter().all(|entry| entry.key != "g.output.chunk_commits"),
        "legacy unvalidated chunk metadata must not be emitted"
    );
    let part_binding_metadata = footer_metadata
        .iter()
        .find(|entry| entry.key == crate::schema::PART_BINDING_METADATA_KEY)
        .and_then(|entry| entry.value.as_deref())
        .expect("bound part metadata exists");
    let part_footer: OutputPartFooter =
        serde_json::from_str(part_binding_metadata).expect("bound part metadata is valid JSON");
    assert_eq!(part_footer.part_file_name, CHUNK_FILE_NAME);
    assert_eq!(part_footer.receipt_id, "part_000000000");
    let batches =
        builder.build().expect("published part reader builds").collect::<Result<Vec<_>, _>>().expect("part reads");
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    let beta = batches[0]
        .column_by_name("BETA")
        .expect("BETA exists")
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("BETA is Float32");
    assert!((beta.value(0) - 0.5).abs() < f32::EPSILON);
}

fn count_files_with_extension(directory: &Path, extension: &str) -> usize {
    std::fs::read_dir(directory)
        .expect("test output directory reads")
        .map(|entry| entry.expect("test output entry reads").path())
        .filter(|path| path.extension().is_some_and(|observed| observed == extension))
        .count()
}

fn expected_fault_operations(
    fault_point: FaultPoint,
    temporary_path: &Path,
    final_path: &Path,
    parts_directory: &Path,
) -> Vec<RecordedOperation> {
    let mut operations = vec![RecordedOperation::CreateNew(temporary_path.to_path_buf())];
    if fault_point == FaultPoint::CreateNew {
        return operations;
    }
    operations.push(RecordedOperation::ParquetHeader(temporary_path.to_path_buf()));
    operations.push(RecordedOperation::ParquetFinalize(temporary_path.to_path_buf()));
    if fault_point == FaultPoint::ParquetFinalize {
        operations.push(RecordedOperation::Cleanup(temporary_path.to_path_buf()));
        return operations;
    }
    operations.push(RecordedOperation::FileSync(temporary_path.to_path_buf()));
    if fault_point == FaultPoint::FileSync {
        operations.push(RecordedOperation::Cleanup(temporary_path.to_path_buf()));
        return operations;
    }
    operations.push(RecordedOperation::Metadata(temporary_path.to_path_buf()));
    if fault_point == FaultPoint::Metadata {
        operations.push(RecordedOperation::Cleanup(temporary_path.to_path_buf()));
        return operations;
    }
    operations.push(RecordedOperation::Rename {
        source_path: temporary_path.to_path_buf(),
        destination_path: final_path.to_path_buf(),
    });
    if fault_point == FaultPoint::Rename {
        operations.push(RecordedOperation::Cleanup(temporary_path.to_path_buf()));
        return operations;
    }
    operations.push(RecordedOperation::DirectorySync(parts_directory.to_path_buf()));
    if fault_point == FaultPoint::DirectorySync {
        return operations;
    }
    panic!("cleanup is tested only as a secondary fault")
}

#[test]
fn durable_publication_orders_every_boundary_before_commit_exposure() {
    let directory = TestDirectory::new("ordered-success");
    let output_io = RecordingOutputIo::new(&[]);
    let publication = test_publication(&directory.path);
    let ExpectedPartPaths { temporary_path, final_path } = expected_paths(&directory.path);

    let write_result = write_regenie_step2_chunk_job_with_io(&output_io, &publication, test_write_batch(), true)
        .expect("durable publication succeeds");

    assert_eq!(
        output_io.operations(),
        [
            RecordedOperation::CreateNew(temporary_path.clone()),
            RecordedOperation::ParquetHeader(temporary_path.clone()),
            RecordedOperation::ParquetFinalize(temporary_path.clone()),
            RecordedOperation::FileSync(temporary_path.clone()),
            RecordedOperation::Metadata(temporary_path.clone()),
            RecordedOperation::Rename { source_path: temporary_path.clone(), destination_path: final_path.clone() },
            RecordedOperation::DirectorySync(directory.path.clone()),
        ]
    );
    assert_eq!(
        write_result.part_receipt.footer.chunks,
        [OutputChunkCommit {
            chunk_identifier: 0,
            variant_start_index: 0,
            variant_stop_index: 1,
            row_count: 1,
            chunk_file_name: CHUNK_FILE_NAME.to_string(),
        }]
    );
    assert!(!temporary_path.exists());
    assert_readable_parquet_part(&final_path);
    let receipt = read_part_receipt(
        &receipt_path(&publication.commits_directory, &write_result.part_receipt.footer.receipt_id)
            .expect("receipt path builds"),
    )
    .expect("receipt reads");
    assert_eq!(receipt, write_result.part_receipt);
}

#[test]
fn every_publication_failure_returns_no_commit_and_preserves_operation_context() {
    let cases = [
        PublicationFailureCase {
            fault_point: FaultPoint::CreateNew,
            expected_operation: "create new temporary Parquet part",
            final_exists: false,
        },
        PublicationFailureCase {
            fault_point: FaultPoint::ParquetFinalize,
            expected_operation: "finalize temporary Parquet part",
            final_exists: false,
        },
        PublicationFailureCase {
            fault_point: FaultPoint::FileSync,
            expected_operation: "synchronize temporary Parquet part",
            final_exists: false,
        },
        PublicationFailureCase {
            fault_point: FaultPoint::Rename,
            expected_operation: "publish temporary Parquet part",
            final_exists: false,
        },
        PublicationFailureCase {
            fault_point: FaultPoint::DirectorySync,
            expected_operation: "synchronize Parquet parts directory",
            final_exists: true,
        },
        PublicationFailureCase {
            fault_point: FaultPoint::Metadata,
            expected_operation: "hash temporary Parquet part",
            final_exists: false,
        },
    ];

    for failure_case in cases {
        let fault_point = failure_case.fault_point;
        let directory = TestDirectory::new(fault_point.label());
        let output_io = RecordingOutputIo::new(&[fault_point]);
        let publication = test_publication(&directory.path);
        let ExpectedPartPaths { temporary_path, final_path } = expected_paths(&directory.path);

        let error = write_regenie_step2_chunk_job_with_io(&output_io, &publication, test_write_batch(), true)
            .err()
            .expect("faulted publication cannot expose commits");

        let error_message = error.to_string();
        assert!(error_message.contains(failure_case.expected_operation), "unexpected error: {error_message}");
        assert!(error_message.contains("injected"), "original failure is missing: {error_message}");
        let expected_error_path =
            if fault_point == FaultPoint::DirectorySync { &directory.path } else { &temporary_path };
        assert!(
            error_message.contains(&expected_error_path.display().to_string()),
            "failing path is missing: {error_message}"
        );
        if fault_point == FaultPoint::Rename {
            assert!(error_message.contains(&final_path.display().to_string()));
        }
        assert_eq!(final_path.exists(), failure_case.final_exists);
        assert_eq!(
            output_io.operations(),
            expected_fault_operations(fault_point, &temporary_path, &final_path, &directory.path)
        );
        if failure_case.final_exists {
            assert_readable_parquet_part(&final_path);
        } else {
            assert!(!temporary_path.exists(), "failed unpublished temp must be cleaned");
        }
        assert!(
            !publication.commits_directory.exists()
                || std::fs::read_dir(&publication.commits_directory)
                    .expect("existing receipt directory reads")
                    .next()
                    .is_none(),
            "a failed publication boundary must not expose a receipt"
        );
    }
}

#[test]
fn parquet_writer_initialization_failure_cleans_the_transaction_temp_without_publishing() {
    let directory = TestDirectory::new("Parquet-writer-initialize");
    let output_io = RecordingOutputIo::new(&[]);
    let ExpectedPartPaths { temporary_path, final_path } = expected_paths(&directory.path);
    let foreign_temp_path = directory.path.join(".foreign-part.foreign-transaction.tmp");
    std::fs::write(&foreign_temp_path, b"foreign temp").expect("foreign temp is created");
    let output_file = output_io.create_new_file(&temporary_path).expect("transaction temp is created");
    let unsupported_schema =
        Arc::new(Schema::new(vec![Field::new("unsupported-empty-struct", DataType::Struct(Fields::empty()), false)]));

    let error = {
        let _unpublished_part = UnpublishedPart::new(&output_io, temporary_path.clone());
        streams::write_regenie_step2_chunks_to_parquet_file(
            output_file,
            streams::RegenieStep2ParquetStreamRequest {
                chunks: Vec::new(),
                chunk_schema: &unsupported_schema,
                parquet_record_batch_schema: &crate::schema::REGENIE_STEP2_PARQUET_RECORD_BATCH_FLOAT32_SCHEMA,
                chunk_file_path: &temporary_path,
                part_footer: &test_part_footer(),
                file_create_seconds: 0.0,
                collect_stage_timings: false,
            },
        )
    }
    .err()
    .expect("rejected Parquet schema cannot expose commits");

    let error_message = error.to_string();
    assert!(error_message.contains("initialize temporary Parquet part"), "unexpected error: {error_message}");
    assert!(error_message.contains("Parquet does not support writing empty structs"));
    assert!(error_message.contains(&temporary_path.display().to_string()));
    assert_eq!(
        output_io.operations(),
        [RecordedOperation::CreateNew(temporary_path.clone()), RecordedOperation::Cleanup(temporary_path.clone()),]
    );
    assert!(!temporary_path.exists());
    assert!(!final_path.exists());
    assert_eq!(std::fs::read(&foreign_temp_path).expect("foreign temp remains"), b"foreign temp");
}

#[test]
fn first_physical_parquet_write_failure_during_finalize_cleans_the_transaction_temp_without_publishing() {
    let directory = TestDirectory::new("first-physical-Parquet-write");
    let output_io = RecordingOutputIo::new(&[FaultPoint::FirstPhysicalParquetWrite]);
    let publication = test_publication(&directory.path);
    let ExpectedPartPaths { temporary_path, final_path } = expected_paths(&directory.path);

    let error = write_regenie_step2_chunk_job_with_io(&output_io, &publication, test_write_batch(), true)
        .err()
        .expect("failed first physical Parquet write cannot expose commits");

    let error_message = error.to_string();
    assert!(error_message.contains("finalize temporary Parquet part"), "unexpected error: {error_message}");
    assert!(error_message.contains("injected first-physical-Parquet-write failure"));
    assert!(error_message.contains(&temporary_path.display().to_string()));
    assert_eq!(
        output_io.operations(),
        [
            RecordedOperation::CreateNew(temporary_path.clone()),
            RecordedOperation::ParquetHeader(temporary_path.clone()),
            RecordedOperation::Cleanup(temporary_path.clone()),
        ]
    );
    assert!(!temporary_path.exists());
    assert!(!final_path.exists());
}

#[test]
fn invalid_record_batch_cleans_only_the_transaction_temp_without_publishing() {
    let directory = TestDirectory::new("invalid-record-batch");
    let output_io = RecordingOutputIo::new(&[]);
    let publication = test_publication(&directory.path);
    let ExpectedPartPaths { temporary_path, final_path } = expected_paths(&directory.path);
    let foreign_temp_path = directory.path.join(".foreign-part.foreign-transaction.tmp");
    std::fs::write(&foreign_temp_path, b"foreign temp").expect("foreign temp is created");
    let mut write_batch = test_write_batch();
    write_batch.chunks[0].beta = Arc::new(Float32Array::from(vec![0.5_f32, 0.75_f32]));

    let error = write_regenie_step2_chunk_job_with_io(&output_io, &publication, write_batch, true)
        .err()
        .expect("invalid record-batch construction cannot expose commits");

    let error_message = error.to_string();
    assert_eq!(error_message, "Invalid argument error: all columns in a record batch must have the same length");
    assert_eq!(
        output_io.operations(),
        [
            RecordedOperation::CreateNew(temporary_path.clone()),
            RecordedOperation::ParquetHeader(temporary_path.clone()),
            RecordedOperation::Cleanup(temporary_path.clone()),
        ]
    );
    assert!(!temporary_path.exists());
    assert!(!final_path.exists());
    assert_eq!(std::fs::read(&foreign_temp_path).expect("foreign temp remains"), b"foreign temp");
}

#[test]
fn create_new_collision_preserves_the_foreign_temp_and_never_cleans_it() {
    let directory = TestDirectory::new("create-new-collision");
    let output_io = RecordingOutputIo::new(&[]);
    let publication = test_publication(&directory.path);
    let ExpectedPartPaths { temporary_path, final_path } = expected_paths(&directory.path);
    std::fs::write(&temporary_path, b"foreign transaction data").expect("colliding temp is created");

    let error = write_regenie_step2_chunk_job_with_io(&output_io, &publication, test_write_batch(), true)
        .err()
        .expect("create-new collision is rejected");

    assert!(error.to_string().contains("create new temporary Parquet part"));
    assert!(error.to_string().contains(&temporary_path.display().to_string()));
    assert_eq!(std::fs::read(&temporary_path).expect("foreign temp remains"), b"foreign transaction data");
    assert!(!final_path.exists());
    assert_eq!(output_io.operations(), [RecordedOperation::CreateNew(temporary_path)]);
}

#[test]
fn cleanup_removes_only_the_current_unpublished_temp() {
    let directory = TestDirectory::new("cleanup-scope");
    let output_io = RecordingOutputIo::new(&[FaultPoint::FileSync]);
    let publication = test_publication(&directory.path);
    let ExpectedPartPaths { temporary_path, final_path } = expected_paths(&directory.path);
    let foreign_temp_path = directory.path.join(".foreign-part.foreign-transaction.tmp");
    std::fs::write(&foreign_temp_path, b"foreign temp").expect("foreign temp is created");
    std::fs::write(&final_path, b"existing final").expect("existing final is created");

    let error = write_regenie_step2_chunk_job_with_io(&output_io, &publication, test_write_batch(), true)
        .err()
        .expect("file sync fault rejects publication");

    assert!(error.to_string().contains("injected file-sync failure"));
    assert!(!temporary_path.exists());
    assert_eq!(std::fs::read(&foreign_temp_path).expect("foreign temp remains"), b"foreign temp");
    assert_eq!(std::fs::read(&final_path).expect("existing final remains"), b"existing final");
}

#[test]
fn failed_footer_is_never_published_and_cleanup_failure_cannot_replace_the_primary_error() {
    let directory = TestDirectory::new("missing-footer");
    let output_io = RecordingOutputIo::new(&[FaultPoint::ParquetFinalize, FaultPoint::Cleanup]);
    let publication = test_publication(&directory.path);
    let ExpectedPartPaths { temporary_path, final_path } = expected_paths(&directory.path);

    let error = write_regenie_step2_chunk_job_with_io(&output_io, &publication, test_write_batch(), true)
        .err()
        .expect("failed footer cannot publish");

    let error_message = error.to_string();
    assert!(error_message.contains("injected Parquet-finalize failure"));
    assert!(!error_message.contains("injected cleanup failure"));
    assert!(!final_path.exists());
    assert!(temporary_path.exists(), "injected cleanup failure retains only the transaction temp");
    let incomplete_file = File::open(&temporary_path).expect("incomplete temp remains for inspection");
    assert!(
        ParquetRecordBatchReaderBuilder::try_new(incomplete_file).is_err(),
        "a temp without its footer must be unreadable"
    );
}

#[test]
fn restart_reconciles_final_link_created_before_source_removal() {
    let directory = TestDirectory::new("restart-after-final-link");
    let first_output_io = RecordingOutputIo::new(&[FaultPoint::AfterFinalLink]);
    let first_publication = test_publication(&directory.path);
    let final_path = expected_paths(&directory.path).final_path;

    let first_error =
        write_regenie_step2_chunk_job_with_io(&first_output_io, &first_publication, test_write_batch(), true)
            .err()
            .expect("injected link boundary interrupts the first publication");
    assert!(first_error.to_string().contains("injected after-final-link failure"));
    assert_readable_parquet_part(&final_path);
    assert!(!first_publication.commits_directory.exists());

    let second_output_io = RecordingOutputIo::new(&[]);
    let second_publication = test_publication_with_identifier(&directory.path, "restart-transaction");
    let result =
        write_regenie_step2_chunk_job_with_io(&second_output_io, &second_publication, test_write_batch(), true)
            .expect("restart verifies the final and publishes its receipt");

    assert_eq!(count_files_with_extension(&directory.path, "parquet"), 1);
    assert_eq!(count_files_with_extension(&second_publication.commits_directory, "json"), 1);
    let receipt = read_part_receipt(
        &receipt_path(&second_publication.commits_directory, &result.part_receipt.footer.receipt_id)
            .expect("receipt path builds"),
    )
    .expect("reconstructed receipt reads");
    assert_eq!(receipt, result.part_receipt);
}

#[test]
fn restart_reconciles_final_part_created_before_receipt() {
    let directory = TestDirectory::new("restart-before-receipt");
    let first_publication = test_publication(&directory.path);
    std::fs::write(&first_publication.commits_directory, b"blocks receipt directory")
        .expect("receipt path blocker is created");
    let final_path = expected_paths(&directory.path).final_path;

    let first_error = write_regenie_step2_chunk_job_with_io(
        &RecordingOutputIo::new(&[]),
        &first_publication,
        test_write_batch(),
        true,
    )
    .err()
    .expect("blocked receipt publication interrupts the first publication");
    assert!(first_error.to_string().contains("non-directory ancestor"));
    assert_readable_parquet_part(&final_path);

    std::fs::remove_file(&first_publication.commits_directory).expect("receipt path blocker is removed");
    std::fs::create_dir(&first_publication.commits_directory).expect("receipt directory is created");
    let second_publication = test_publication_with_identifier(&directory.path, "receipt-restart");
    let result = write_regenie_step2_chunk_job_with_io(
        &RecordingOutputIo::new(&[]),
        &second_publication,
        test_write_batch(),
        true,
    )
    .expect("restart verifies the final and publishes its receipt");

    assert_eq!(count_files_with_extension(&directory.path, "parquet"), 1);
    assert_eq!(count_files_with_extension(&second_publication.commits_directory, "json"), 1);
    assert_eq!(result.part_receipt.footer.chunks.len(), 1);
}

#[test]
fn existing_final_with_conflicting_footer_is_rejected() {
    let directory = TestDirectory::new("conflicting-final-footer");
    let first_publication = test_publication(&directory.path);
    write_regenie_step2_chunk_job_with_io(&RecordingOutputIo::new(&[]), &first_publication, test_write_batch(), false)
        .expect("first immutable part publishes");

    let mut conflicting_publication = test_publication_with_identifier(&directory.path, "conflicting-transaction");
    conflicting_publication.binding.phenotype_name = "trait-b".to_string();
    let error = write_regenie_step2_chunk_job_with_io(
        &RecordingOutputIo::new(&[]),
        &conflicting_publication,
        test_write_batch(),
        false,
    )
    .err()
    .expect("conflicting footer cannot reuse an existing final");

    assert!(error.to_string().contains("conflicts with the expected immutable footer"));
    assert_eq!(count_files_with_extension(&directory.path, "parquet"), 1);
    assert_eq!(count_files_with_extension(&first_publication.commits_directory, "json"), 1);
}
