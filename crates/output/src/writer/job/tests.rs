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
use crate::persistence::io::OutputIo;
use crate::persistence::model::{OutputChunkCommit, OutputTransactionIdentifier};
use crate::writer::{RegenieStep2ChunkJob, RegenieStep2ChunkWriteBatch, streams};

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

    fn rename_file(&self, source_path: &Path, destination_path: &Path) -> io::Result<()> {
        self.state.record(RecordedOperation::Rename {
            source_path: source_path.to_path_buf(),
            destination_path: destination_path.to_path_buf(),
        });
        if self.state.should_fail(FaultPoint::Rename) {
            return Err(injected_error(FaultPoint::Rename));
        }
        std::fs::rename(source_path, destination_path)
    }

    fn sync_directory(&self, path: &Path) -> io::Result<()> {
        self.state.record(RecordedOperation::DirectorySync(path.to_path_buf()));
        if self.state.should_fail(FaultPoint::DirectorySync) {
            return Err(injected_error(FaultPoint::DirectorySync));
        }
        File::open(path)?.sync_all()
    }

    fn file_size(&self, path: &Path) -> io::Result<u64> {
        self.state.record(RecordedOperation::Metadata(path.to_path_buf()));
        if self.state.should_fail(FaultPoint::Metadata) {
            return Err(injected_error(FaultPoint::Metadata));
        }
        std::fs::metadata(path).map(|metadata| metadata.len())
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

fn test_transaction_identifier() -> OutputTransactionIdentifier {
    OutputTransactionIdentifier::for_test(TEST_TRANSACTION_IDENTIFIER)
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

fn expected_paths(parts_directory: &Path) -> (PathBuf, PathBuf) {
    let transaction_identifier = test_transaction_identifier();
    (
        transaction_temporary_part_path(parts_directory, CHUNK_FILE_NAME, &transaction_identifier),
        parts_directory.join(CHUNK_FILE_NAME),
    )
}

fn assert_readable_parquet_part(final_path: &Path) {
    let input_file = File::open(final_path).expect("published part opens");
    let builder = ParquetRecordBatchReaderBuilder::try_new(input_file).expect("published footer is readable");
    assert_eq!(builder.schema().fields(), crate::schema::REGENIE_STEP2_CHUNK_SCHEMA.fields());
    let footer_metadata = builder.metadata().file_metadata().key_value_metadata().expect("footer metadata exists");
    let chunk_commit_metadata = footer_metadata
        .iter()
        .find(|entry| entry.key == crate::schema::CHUNK_COMMITS_METADATA_KEY)
        .and_then(|entry| entry.value.as_deref())
        .expect("chunk commit metadata exists");
    let chunk_commit_values: serde_json::Value =
        serde_json::from_str(chunk_commit_metadata).expect("chunk commit metadata is valid JSON");
    assert_eq!(chunk_commit_values[0]["chunk_file_name"], CHUNK_FILE_NAME);
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
    if fault_point == FaultPoint::Metadata {
        operations.push(RecordedOperation::Metadata(final_path.to_path_buf()));
        return operations;
    }
    panic!("cleanup is tested only as a secondary fault")
}

#[test]
fn durable_publication_orders_every_boundary_before_commit_exposure() {
    let directory = TestDirectory::new("ordered-success");
    let output_io = RecordingOutputIo::new(&[]);
    let transaction_identifier = test_transaction_identifier();
    let (temporary_path, final_path) = expected_paths(&directory.path);

    let write_result = write_regenie_step2_chunk_job_with_io(
        &output_io,
        &directory.path,
        &transaction_identifier,
        test_write_batch(),
        true,
    )
    .expect("durable publication succeeds");

    assert_eq!(
        output_io.operations(),
        [
            RecordedOperation::CreateNew(temporary_path.clone()),
            RecordedOperation::ParquetHeader(temporary_path.clone()),
            RecordedOperation::ParquetFinalize(temporary_path.clone()),
            RecordedOperation::FileSync(temporary_path.clone()),
            RecordedOperation::Rename { source_path: temporary_path.clone(), destination_path: final_path.clone() },
            RecordedOperation::DirectorySync(directory.path.clone()),
            RecordedOperation::Metadata(final_path.clone()),
        ]
    );
    assert_eq!(
        write_result.chunk_commits,
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
            expected_operation: "rename temporary Parquet part",
            final_exists: false,
        },
        PublicationFailureCase {
            fault_point: FaultPoint::DirectorySync,
            expected_operation: "synchronize Parquet parts directory",
            final_exists: true,
        },
        PublicationFailureCase {
            fault_point: FaultPoint::Metadata,
            expected_operation: "read published Parquet part metadata",
            final_exists: true,
        },
    ];

    for failure_case in cases {
        let fault_point = failure_case.fault_point;
        let directory = TestDirectory::new(fault_point.label());
        let output_io = RecordingOutputIo::new(&[fault_point]);
        let transaction_identifier = test_transaction_identifier();
        let (temporary_path, final_path) = expected_paths(&directory.path);

        let error = write_regenie_step2_chunk_job_with_io(
            &output_io,
            &directory.path,
            &transaction_identifier,
            test_write_batch(),
            true,
        )
        .err()
        .expect("faulted publication cannot expose commits");

        let error_message = error.to_string();
        assert!(error_message.contains(failure_case.expected_operation), "unexpected error: {error_message}");
        assert!(error_message.contains("injected"), "original failure is missing: {error_message}");
        let expected_error_path = match fault_point {
            FaultPoint::DirectorySync => &directory.path,
            FaultPoint::Metadata => &final_path,
            _ => &temporary_path,
        };
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
    }
}

#[test]
fn parquet_writer_initialization_failure_cleans_the_transaction_temp_without_publishing() {
    let directory = TestDirectory::new("Parquet-writer-initialize");
    let output_io = RecordingOutputIo::new(&[]);
    let (temporary_path, final_path) = expected_paths(&directory.path);
    let foreign_temp_path = directory.path.join(".foreign-part.foreign-transaction.tmp");
    std::fs::write(&foreign_temp_path, b"foreign temp").expect("foreign temp is created");
    let output_file = output_io.create_new_file(&temporary_path).expect("transaction temp is created");
    let unsupported_schema =
        Arc::new(Schema::new(vec![Field::new("unsupported-empty-struct", DataType::Struct(Fields::empty()), false)]));

    let error = {
        let _unpublished_part = UnpublishedPart::new(&output_io, temporary_path.clone());
        streams::write_regenie_step2_chunks_to_parquet_file(
            output_file,
            Vec::new(),
            &unsupported_schema,
            &crate::schema::REGENIE_STEP2_PARQUET_RECORD_BATCH_FLOAT32_SCHEMA,
            &temporary_path,
            &[],
            0.0,
            false,
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
    let transaction_identifier = test_transaction_identifier();
    let (temporary_path, final_path) = expected_paths(&directory.path);

    let error = write_regenie_step2_chunk_job_with_io(
        &output_io,
        &directory.path,
        &transaction_identifier,
        test_write_batch(),
        true,
    )
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
    let transaction_identifier = test_transaction_identifier();
    let (temporary_path, final_path) = expected_paths(&directory.path);
    let foreign_temp_path = directory.path.join(".foreign-part.foreign-transaction.tmp");
    std::fs::write(&foreign_temp_path, b"foreign temp").expect("foreign temp is created");
    let mut write_batch = test_write_batch();
    write_batch.chunks[0].beta = Arc::new(Float32Array::from(vec![0.5_f32, 0.75_f32]));

    let error =
        write_regenie_step2_chunk_job_with_io(&output_io, &directory.path, &transaction_identifier, write_batch, true)
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
    let transaction_identifier = test_transaction_identifier();
    let (temporary_path, final_path) = expected_paths(&directory.path);
    std::fs::write(&temporary_path, b"foreign transaction data").expect("colliding temp is created");

    let error = write_regenie_step2_chunk_job_with_io(
        &output_io,
        &directory.path,
        &transaction_identifier,
        test_write_batch(),
        true,
    )
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
    let transaction_identifier = test_transaction_identifier();
    let (temporary_path, final_path) = expected_paths(&directory.path);
    let foreign_temp_path = directory.path.join(".foreign-part.foreign-transaction.tmp");
    std::fs::write(&foreign_temp_path, b"foreign temp").expect("foreign temp is created");
    std::fs::write(&final_path, b"existing final").expect("existing final is created");

    let error = write_regenie_step2_chunk_job_with_io(
        &output_io,
        &directory.path,
        &transaction_identifier,
        test_write_batch(),
        true,
    )
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
    let transaction_identifier = test_transaction_identifier();
    let (temporary_path, final_path) = expected_paths(&directory.path);

    let error = write_regenie_step2_chunk_job_with_io(
        &output_io,
        &directory.path,
        &transaction_identifier,
        test_write_batch(),
        true,
    )
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
