#![allow(clippy::missing_errors_doc)]
#![allow(clippy::needless_pass_by_value)]

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;

use arrow::array::{
    ArrayRef, Float32Array, Int32Array, Int64Array, RecordBatch, StringArray, StringDictionaryBuilder, new_null_array,
};
use arrow::datatypes::{DataType, Field, Int8Type, Int32Type, Schema};
use arrow::ipc::CompressionType;
use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use crossbeam_channel::{Receiver, Sender, bounded};
#[cfg(feature = "python")]
use numpy::{PyReadonlyArray1, PyUntypedArrayMethods};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;
#[cfg(feature = "python")]
use pyo3::exceptions::{PyRuntimeError, PyValueError};
#[cfg(feature = "python")]
use pyo3::prelude::*;
use thiserror::Error;

use crate::genotype::common::VariantMetadataColumns;

#[cfg(feature = "python")]
use arrow::array::Array;
#[cfg(feature = "python")]
use std::collections::BTreeSet;

const REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE: usize = 122_880;
const REGENIE_STEP2_CHUNKS_PER_ARROW_FILE: usize = 4;

#[derive(Clone)]
struct RustOutputWriterConfig {
    run_directory: PathBuf,
    chunks_directory: PathBuf,
    association_mode: String,
    finalize_parquet: bool,
}

#[derive(Debug, Error)]
pub enum OutputError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Runtime(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Arrow(#[from] arrow::error::ArrowError),
    #[error(transparent)]
    Parquet(#[from] parquet::errors::ParquetError),
}

impl From<String> for OutputError {
    fn from(value: String) -> Self {
        Self::Runtime(value)
    }
}

#[derive(Clone, Debug)]
pub struct NativeRegenieStep2Chunk {
    pub variant_start_index: i64,
    pub variant_stop_index: i64,
    pub metadata: VariantMetadataColumns,
    pub allele_one_frequency: Vec<f32>,
    pub observation_count: Vec<i32>,
    pub beta: Vec<f32>,
    pub standard_error: Vec<f32>,
    pub chi_squared: Vec<f32>,
    pub log10_p_value: Vec<f32>,
    pub extra_code: Option<Vec<i32>>,
}

struct RegenieStep2ChunkJob {
    chunk_identifier: i64,
    variant_start_index: i64,
    variant_stop_index: i64,
    chrom: Vec<String>,
    genpos: Vec<i64>,
    id: Vec<String>,
    allele0: Vec<String>,
    allele1: Vec<String>,
    a1freq: Vec<f32>,
    n: Vec<i32>,
    beta: Vec<f32>,
    se: Vec<f32>,
    chisq: Vec<f32>,
    log10p: Vec<f32>,
    extra_code: Option<Vec<i32>>,
}

struct RegenieStep2ChunkWriteBatch {
    chunk_file_name: String,
    chunks: Vec<RegenieStep2ChunkJob>,
}

enum OutputCoordinatorJob {
    RegenieStep2(Box<RegenieStep2ChunkJob>),
    Finish,
    Abort,
}

enum OutputWriteJob {
    RegenieStep2(Box<RegenieStep2ChunkWriteBatch>),
    Shutdown,
}

pub struct NativeOutputWriterSession {
    sender: Mutex<Option<Sender<OutputCoordinatorJob>>>,
    coordinator_handle: Mutex<Option<JoinHandle<()>>>,
    worker_handles: Mutex<Vec<JoinHandle<()>>>,
    worker_errors: Arc<Mutex<Vec<String>>>,
    config: RustOutputWriterConfig,
}

#[cfg(feature = "python")]
#[pyclass]
pub struct OutputWriterSession {
    inner: NativeOutputWriterSession,
}

impl NativeOutputWriterSession {
    pub fn new(
        run_directory: PathBuf,
        chunks_directory: PathBuf,
        association_mode: String,
        writer_thread_count: usize,
        writer_queue_depth: usize,
        finalize_parquet: bool,
    ) -> Result<Self, OutputError> {
        if writer_thread_count == 0 {
            return Err(OutputError::InvalidInput("Writer thread count must be at least 1.".to_string()));
        }
        if association_mode != "regenie2_linear" && association_mode != "regenie2_binary" {
            return Err(OutputError::InvalidInput(
                "Rust output backend only supports REGENIE step 2 quantitative and binary output.".to_string(),
            ));
        }
        std::fs::create_dir_all(&run_directory)?;
        std::fs::create_dir_all(&chunks_directory)?;
        let config = RustOutputWriterConfig { run_directory, chunks_directory, association_mode, finalize_parquet };
        let (sender, receiver) = bounded(writer_queue_depth.max(1));
        let (writer_sender, writer_receiver) = bounded(writer_queue_depth.max(1));
        let worker_errors = Arc::new(Mutex::new(Vec::new()));
        let mut worker_handles = Vec::with_capacity(writer_thread_count);
        for _ in 0..writer_thread_count {
            let receiver_clone = writer_receiver.clone();
            let config_clone = config.clone();
            let worker_errors_clone = Arc::clone(&worker_errors);
            worker_handles.push(std::thread::spawn(move || {
                run_output_writer_worker(receiver_clone, config_clone, worker_errors_clone);
            }));
        }
        let coordinator_worker_errors = Arc::clone(&worker_errors);
        let coordinator_handle = std::thread::spawn(move || {
            run_output_writer_coordinator(receiver, writer_sender, writer_thread_count, coordinator_worker_errors);
        });
        Ok(Self {
            sender: Mutex::new(Some(sender)),
            coordinator_handle: Mutex::new(Some(coordinator_handle)),
            worker_handles: Mutex::new(worker_handles),
            worker_errors,
            config,
        })
    }

    pub fn write_regenie2_chunk(&self, chunk: NativeRegenieStep2Chunk) -> Result<(), OutputError> {
        let row_count = chunk.metadata.position.len();
        let observed_lengths = [
            chunk.metadata.chromosome.len(),
            chunk.metadata.variant_identifier.len(),
            chunk.metadata.allele_two.len(),
            chunk.metadata.allele_one.len(),
            chunk.allele_one_frequency.len(),
            chunk.observation_count.len(),
            chunk.beta.len(),
            chunk.standard_error.len(),
            chunk.chi_squared.len(),
            chunk.log10_p_value.len(),
        ];
        validate_column_lengths(row_count, observed_lengths.as_slice())?;
        if let Some(extra_code_values) = chunk.extra_code.as_ref() {
            validate_column_lengths(row_count, &[extra_code_values.len()])?;
        }
        let job = RegenieStep2ChunkJob {
            chunk_identifier: chunk.variant_start_index,
            variant_start_index: chunk.variant_start_index,
            variant_stop_index: chunk.variant_stop_index,
            chrom: chunk.metadata.chromosome,
            genpos: chunk.metadata.position,
            id: chunk.metadata.variant_identifier,
            allele0: chunk.metadata.allele_two,
            allele1: chunk.metadata.allele_one,
            a1freq: chunk.allele_one_frequency,
            n: chunk.observation_count,
            beta: chunk.beta,
            se: chunk.standard_error,
            chisq: chunk.chi_squared,
            log10p: chunk.log10_p_value,
            extra_code: chunk.extra_code,
        };
        self.raise_if_worker_failed()?;
        let sender_guard = self
            .sender
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer sender lock was poisoned.".to_string()))?;
        let sender = sender_guard
            .as_ref()
            .ok_or_else(|| OutputError::Runtime("Rust output writer session is already closed.".to_string()))?;
        sender
            .send(OutputCoordinatorJob::RegenieStep2(Box::new(job)))
            .map_err(|error| OutputError::Runtime(error.to_string()))?;
        Ok(())
    }

    pub fn finish(&self) -> Result<Option<PathBuf>, OutputError> {
        close_writer_sender(&self.sender, OutputCoordinatorJob::Finish)?;
        join_coordinator_thread(&self.coordinator_handle)?;
        join_writer_threads(&self.worker_handles)?;
        self.raise_if_worker_failed()?;
        if !self.config.finalize_parquet {
            return Ok(None);
        }
        let final_parquet_path = self.config.run_directory.join("final.parquet");
        write_final_parquet_from_chunk_files(
            &self.config.chunks_directory,
            &final_parquet_path,
            &self.config.association_mode,
        )?;
        Ok(Some(final_parquet_path))
    }

    pub fn abort(&self) -> Result<(), OutputError> {
        close_writer_sender(&self.sender, OutputCoordinatorJob::Abort)?;
        join_coordinator_thread(&self.coordinator_handle)?;
        join_writer_threads(&self.worker_handles)
    }

    fn raise_if_worker_failed(&self) -> Result<(), OutputError> {
        let worker_errors = self
            .worker_errors
            .lock()
            .map_err(|_| OutputError::Runtime("Rust output writer error lock was poisoned.".to_string()))?;
        if let Some(first_error) = worker_errors.first() {
            return Err(OutputError::Runtime(first_error.clone()));
        }
        Ok(())
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl OutputWriterSession {
    #[new]
    #[pyo3(signature = (run_directory, chunks_directory, association_mode, writer_thread_count=1, writer_queue_depth=1, finalize_parquet=true))]
    fn new(
        run_directory: String,
        chunks_directory: String,
        association_mode: String,
        writer_thread_count: usize,
        writer_queue_depth: usize,
        finalize_parquet: bool,
    ) -> PyResult<Self> {
        let inner = NativeOutputWriterSession::new(
            PathBuf::from(run_directory),
            PathBuf::from(chunks_directory),
            association_mode,
            writer_thread_count,
            writer_queue_depth,
            finalize_parquet,
        )
        .map_err(convert_output_error)?;
        Ok(Self { inner })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (metadata, allele_one_frequency, observation_count, beta, standard_error, chi_squared, log10_p_value, extra_code=None))]
    fn write_regenie2_chunk(
        &self,
        metadata: &Bound<'_, PyAny>,
        allele_one_frequency: PyReadonlyArray1<'_, f32>,
        observation_count: PyReadonlyArray1<'_, i32>,
        beta: PyReadonlyArray1<'_, f32>,
        standard_error: PyReadonlyArray1<'_, f32>,
        chi_squared: PyReadonlyArray1<'_, f32>,
        log10_p_value: PyReadonlyArray1<'_, f32>,
        extra_code: Option<PyReadonlyArray1<'_, i32>>,
    ) -> PyResult<()> {
        let variant_start_index = metadata.getattr("variant_start_index")?.extract::<i64>()?;
        let variant_stop_index = metadata.getattr("variant_stop_index")?.extract::<i64>()?;
        let chromosome = extract_string_column(metadata, "chromosome")?;
        let position_object = metadata.getattr("position")?;
        let position = position_object.extract::<PyReadonlyArray1<'_, i64>>()?;
        let variant_identifier = extract_string_column(metadata, "variant_identifiers")?;
        let allele_one = extract_string_column(metadata, "allele_one")?;
        let allele_zero = extract_string_column(metadata, "allele_two")?;

        let row_count = position.len();
        let observed_lengths = [
            chromosome.len(),
            variant_identifier.len(),
            allele_zero.len(),
            allele_one.len(),
            allele_one_frequency.len(),
            observation_count.len(),
            beta.len(),
            standard_error.len(),
            chi_squared.len(),
            log10_p_value.len(),
        ];
        validate_column_lengths(row_count, observed_lengths.as_slice()).map_err(convert_output_error)?;
        if let Some(extra_code_values) = extra_code.as_ref() {
            validate_column_lengths(row_count, &[extra_code_values.len()]).map_err(convert_output_error)?;
        }
        let chunk = NativeRegenieStep2Chunk {
            variant_start_index,
            variant_stop_index,
            metadata: VariantMetadataColumns {
                chromosome,
                variant_identifier,
                position: position.as_slice()?.to_vec(),
                allele_one,
                allele_two: allele_zero,
            },
            allele_one_frequency: allele_one_frequency.as_slice()?.to_vec(),
            observation_count: observation_count.as_slice()?.to_vec(),
            beta: beta.as_slice()?.to_vec(),
            standard_error: standard_error.as_slice()?.to_vec(),
            chi_squared: chi_squared.as_slice()?.to_vec(),
            log10_p_value: log10_p_value.as_slice()?.to_vec(),
            extra_code: extra_code
                .map(|extra_code_array| extra_code_array.as_slice().map(<[i32]>::to_vec))
                .transpose()?,
        };
        self.inner.write_regenie2_chunk(chunk).map_err(convert_output_error)
    }

    fn finish(&self) -> PyResult<Option<String>> {
        self.inner
            .finish()
            .map(|maybe_path| maybe_path.map(|path| path.display().to_string()))
            .map_err(convert_output_error)
    }

    fn abort(&self) -> PyResult<()> {
        self.inner.abort().map_err(convert_output_error)
    }
}

#[cfg(feature = "python")]
impl OutputWriterSession {
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::missing_errors_doc)]
    pub fn write_regenie2_chunk_values(
        &self,
        variant_start_index: i64,
        variant_stop_index: i64,
        metadata: VariantMetadataColumns,
        allele_one_frequency: Vec<f32>,
        observation_count: Vec<i32>,
        beta: Vec<f32>,
        standard_error: Vec<f32>,
        chi_squared: Vec<f32>,
        log10_p_value: Vec<f32>,
        extra_code: Option<Vec<i32>>,
    ) -> PyResult<()> {
        let chunk = NativeRegenieStep2Chunk {
            variant_start_index,
            variant_stop_index,
            metadata,
            allele_one_frequency,
            observation_count,
            beta,
            standard_error,
            chi_squared,
            log10_p_value,
            extra_code,
        };
        self.inner.write_regenie2_chunk(chunk).map_err(convert_output_error)
    }
}

#[cfg(feature = "python")]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::missing_errors_doc)]
pub fn finalize_output_run_chunks(
    run_directory: String,
    chunks_directory: String,
    association_mode: String,
) -> PyResult<String> {
    let final_parquet_path = PathBuf::from(run_directory).join("final.parquet");
    write_final_parquet_from_chunk_files(Path::new(&chunks_directory), &final_parquet_path, &association_mode)
        .map_err(convert_output_error)?;
    Ok(final_parquet_path.display().to_string())
}

pub fn finalize_native_output_run_chunks(
    run_directory: &Path,
    chunks_directory: &Path,
    association_mode: &str,
) -> Result<PathBuf, OutputError> {
    let final_parquet_path = run_directory.join("final.parquet");
    write_final_parquet_from_chunk_files(chunks_directory, &final_parquet_path, association_mode)?;
    Ok(final_parquet_path)
}

#[cfg(feature = "python")]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::missing_errors_doc)]
pub fn scan_committed_chunk_identifiers(chunks_directory: String) -> PyResult<Vec<i64>> {
    scan_committed_chunk_identifiers_from_arrow_files(Path::new(&chunks_directory)).map_err(convert_output_error)
}

#[cfg(feature = "python")]
fn convert_output_error(error: OutputError) -> PyErr {
    match error {
        OutputError::InvalidInput(message) => PyValueError::new_err(message),
        OutputError::Runtime(message) => PyRuntimeError::new_err(message),
        OutputError::Io(source) => PyRuntimeError::new_err(source.to_string()),
        OutputError::Arrow(source) => PyRuntimeError::new_err(source.to_string()),
        OutputError::Parquet(source) => PyRuntimeError::new_err(source.to_string()),
    }
}

#[cfg(feature = "python")]
fn extract_string_column(metadata: &Bound<'_, PyAny>, attribute_name: &str) -> PyResult<Vec<String>> {
    let column_object = metadata.getattr(attribute_name)?;
    if let Ok(values) = column_object.extract::<Vec<String>>() {
        return Ok(values);
    }
    column_object.call_method0("tolist")?.extract::<Vec<String>>()
}

fn close_writer_sender(
    sender: &Mutex<Option<Sender<OutputCoordinatorJob>>>,
    close_job: OutputCoordinatorJob,
) -> Result<(), OutputError> {
    let mut sender_guard =
        sender.lock().map_err(|_| OutputError::Runtime("Rust output writer sender lock was poisoned.".to_string()))?;
    if let Some(active_sender) = sender_guard.take() {
        active_sender.send(close_job).map_err(|error| OutputError::Runtime(error.to_string()))?;
    }
    Ok(())
}

fn join_coordinator_thread(coordinator_handle: &Mutex<Option<JoinHandle<()>>>) -> Result<(), OutputError> {
    let mut coordinator_handle_guard = coordinator_handle
        .lock()
        .map_err(|_| OutputError::Runtime("Rust output writer coordinator handle lock was poisoned.".to_string()))?;
    if let Some(handle) = coordinator_handle_guard.take() {
        handle
            .join()
            .map_err(|_| OutputError::Runtime("Rust output writer coordinator thread panicked.".to_string()))?;
    }
    Ok(())
}

fn join_writer_threads(worker_handles: &Mutex<Vec<JoinHandle<()>>>) -> Result<(), OutputError> {
    let mut worker_handles_guard = worker_handles
        .lock()
        .map_err(|_| OutputError::Runtime("Rust output writer handle lock was poisoned.".to_string()))?;
    while let Some(worker_handle) = worker_handles_guard.pop() {
        worker_handle
            .join()
            .map_err(|_| OutputError::Runtime("Rust output writer worker thread panicked.".to_string()))?;
    }
    Ok(())
}

fn validate_column_lengths(expected_row_count: usize, observed_lengths: &[usize]) -> Result<(), OutputError> {
    if observed_lengths.iter().all(|observed_length| *observed_length == expected_row_count) {
        return Ok(());
    }
    Err(OutputError::InvalidInput(
        "Rust output writer batch column lengths do not all match the expected row count.".to_string(),
    ))
}

fn run_output_writer_coordinator(
    receiver: Receiver<OutputCoordinatorJob>,
    writer_sender: Sender<OutputWriteJob>,
    writer_thread_count: usize,
    worker_errors: Arc<Mutex<Vec<String>>>,
) {
    let mut pending_chunks = Vec::with_capacity(REGENIE_STEP2_CHUNKS_PER_ARROW_FILE);
    while let Ok(job) = receiver.recv() {
        match job {
            OutputCoordinatorJob::RegenieStep2(chunk_job) => {
                pending_chunks.push(*chunk_job);
                if pending_chunks.len() >= REGENIE_STEP2_CHUNKS_PER_ARROW_FILE
                    && flush_pending_regenie_step2_chunks(&writer_sender, &mut pending_chunks, &worker_errors).is_err()
                {
                    break;
                }
            }
            OutputCoordinatorJob::Finish => {
                let _ = flush_pending_regenie_step2_chunks(&writer_sender, &mut pending_chunks, &worker_errors);
                break;
            }
            OutputCoordinatorJob::Abort => break,
        }
    }
    for _ in 0..writer_thread_count {
        if let Err(error) = writer_sender.send(OutputWriteJob::Shutdown) {
            push_worker_error(&worker_errors, error.to_string());
            return;
        }
    }
}

fn flush_pending_regenie_step2_chunks(
    writer_sender: &Sender<OutputWriteJob>,
    pending_chunks: &mut Vec<RegenieStep2ChunkJob>,
    worker_errors: &Arc<Mutex<Vec<String>>>,
) -> Result<(), ()> {
    if pending_chunks.is_empty() {
        return Ok(());
    }
    let first_chunk_identifier = pending_chunks.first().map_or(0, |chunk_job| chunk_job.chunk_identifier);
    let last_chunk_identifier =
        pending_chunks.last().map_or(first_chunk_identifier, |chunk_job| chunk_job.chunk_identifier);
    let chunk_file_name = build_chunk_file_name(first_chunk_identifier, last_chunk_identifier);
    let write_batch = RegenieStep2ChunkWriteBatch { chunk_file_name, chunks: std::mem::take(pending_chunks) };
    writer_sender.send(OutputWriteJob::RegenieStep2(Box::new(write_batch))).map_err(|error| {
        push_worker_error(worker_errors, error.to_string());
    })
}

fn push_worker_error(worker_errors: &Arc<Mutex<Vec<String>>>, error: String) {
    if let Ok(mut worker_errors_guard) = worker_errors.lock() {
        worker_errors_guard.push(error);
    }
}

fn run_output_writer_worker(
    receiver: Receiver<OutputWriteJob>,
    config: RustOutputWriterConfig,
    worker_errors: Arc<Mutex<Vec<String>>>,
) {
    while let Ok(job) = receiver.recv() {
        let write_result = match job {
            OutputWriteJob::RegenieStep2(regenie_step2_job) => {
                write_regenie_step2_chunk_job(&config, *regenie_step2_job)
            }
            OutputWriteJob::Shutdown => return,
        };
        if let Err(error) = write_result {
            push_worker_error(&worker_errors, error);
            return;
        }
    }
}

fn write_regenie_step2_chunk_job(
    config: &RustOutputWriterConfig,
    job: RegenieStep2ChunkWriteBatch,
) -> Result<(), String> {
    let chunk_file_path = config.chunks_directory.join(&job.chunk_file_name);
    let temporary_chunk_file_path = chunk_file_path.with_extension("arrow.tmp");
    let record_batch = build_regenie_step2_record_batch(job)?;
    write_record_batch_to_arrow_file(&record_batch, &temporary_chunk_file_path)?;
    std::fs::rename(&temporary_chunk_file_path, &chunk_file_path).map_err(|error| error.to_string())?;
    Ok(())
}

fn build_chunk_file_name(first_chunk_identifier: i64, last_chunk_identifier: i64) -> String {
    if first_chunk_identifier == last_chunk_identifier {
        return format!("chunk_{first_chunk_identifier:09}.arrow");
    }
    format!("chunk_{first_chunk_identifier:09}_{last_chunk_identifier:09}.arrow")
}

fn build_regenie_step2_record_batch(job: RegenieStep2ChunkWriteBatch) -> Result<RecordBatch, String> {
    let schema = get_regenie_step2_chunk_schema();
    let row_count = job.chunks.iter().map(|chunk_job| chunk_job.genpos.len()).sum();
    let mut chunk_identifier = Vec::with_capacity(row_count);
    let mut variant_start_index = Vec::with_capacity(row_count);
    let mut variant_stop_index = Vec::with_capacity(row_count);
    let mut chrom = Vec::with_capacity(row_count);
    let mut genpos = Vec::with_capacity(row_count);
    let mut id = Vec::with_capacity(row_count);
    let mut allele0 = Vec::with_capacity(row_count);
    let mut allele1 = Vec::with_capacity(row_count);
    let mut a1freq = Vec::with_capacity(row_count);
    let mut n = Vec::with_capacity(row_count);
    let mut beta = Vec::with_capacity(row_count);
    let mut se = Vec::with_capacity(row_count);
    let mut chisq = Vec::with_capacity(row_count);
    let mut log10p = Vec::with_capacity(row_count);
    let mut extra_code = Vec::with_capacity(row_count);

    for chunk_job in job.chunks {
        let chunk_row_count = chunk_job.genpos.len();
        chunk_identifier.extend(std::iter::repeat_n(chunk_job.chunk_identifier, chunk_row_count));
        variant_start_index.extend(std::iter::repeat_n(chunk_job.variant_start_index, chunk_row_count));
        variant_stop_index.extend(std::iter::repeat_n(chunk_job.variant_stop_index, chunk_row_count));
        chrom.extend(chunk_job.chrom);
        genpos.extend(chunk_job.genpos);
        id.extend(chunk_job.id);
        allele0.extend(chunk_job.allele0);
        allele1.extend(chunk_job.allele1);
        a1freq.extend(chunk_job.a1freq);
        n.extend(chunk_job.n);
        beta.extend(chunk_job.beta);
        se.extend(chunk_job.se);
        chisq.extend(chunk_job.chisq);
        log10p.extend(chunk_job.log10p);
        match chunk_job.extra_code {
            None => extra_code.extend(std::iter::repeat_n(None, chunk_row_count)),
            Some(extra_code_values) => {
                extra_code.extend(extra_code_values.into_iter().map(Some));
            }
        }
    }
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(chunk_identifier)),
        Arc::new(Int64Array::from(variant_start_index)),
        Arc::new(Int64Array::from(variant_stop_index)),
        Arc::new(build_dictionary_string_array(&chrom)?),
        Arc::new(Int64Array::from(genpos)),
        Arc::new(StringArray::from(id)),
        Arc::new(build_dictionary_string_array(&allele0)?),
        Arc::new(build_dictionary_string_array(&allele1)?),
        Arc::new(Float32Array::from(a1freq)),
        new_null_array(&DataType::Float32, row_count),
        Arc::new(Int32Array::from(n)),
        Arc::new(build_constant_dictionary_string_array(row_count, "ADD")?),
        Arc::new(Float32Array::from(beta)),
        Arc::new(Float32Array::from(se)),
        Arc::new(Float32Array::from(chisq)),
        Arc::new(Float32Array::from(log10p)),
        Arc::new(build_extra_string_array(extra_code)?),
    ];
    RecordBatch::try_new(Arc::clone(schema), columns).map_err(|error| error.to_string())
}

fn build_regenie_step2_chunk_schema() -> Schema {
    let large_dictionary_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    let small_dictionary_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
    Schema::new(vec![
        Field::new("chunk_identifier", DataType::Int64, true),
        Field::new("variant_start_index", DataType::Int64, true),
        Field::new("variant_stop_index", DataType::Int64, true),
        Field::new("CHROM", large_dictionary_type.clone(), true),
        Field::new("GENPOS", DataType::Int64, true),
        Field::new("ID", DataType::Utf8, true),
        Field::new("ALLELE0", large_dictionary_type.clone(), true),
        Field::new("ALLELE1", large_dictionary_type.clone(), true),
        Field::new("A1FREQ", DataType::Float32, true),
        Field::new("INFO", DataType::Float32, true),
        Field::new("N", DataType::Int32, true),
        Field::new("TEST", small_dictionary_type.clone(), true),
        Field::new("BETA", DataType::Float32, true),
        Field::new("SE", DataType::Float32, true),
        Field::new("CHISQ", DataType::Float32, true),
        Field::new("LOG10P", DataType::Float32, true),
        Field::new("EXTRA", DataType::Utf8, true),
    ])
}

fn get_regenie_step2_chunk_schema() -> &'static Arc<Schema> {
    static REGENIE_STEP2_CHUNK_SCHEMA: OnceLock<Arc<Schema>> = OnceLock::new();
    REGENIE_STEP2_CHUNK_SCHEMA.get_or_init(|| Arc::new(build_regenie_step2_chunk_schema()))
}

fn build_dictionary_string_array(values: &[String]) -> Result<arrow::array::DictionaryArray<Int32Type>, String> {
    let mut builder = StringDictionaryBuilder::<Int32Type>::new();
    for value in values {
        builder.append(value).map_err(|error| error.to_string())?;
    }
    Ok(builder.finish())
}

fn build_constant_dictionary_string_array(
    row_count: usize,
    value: &str,
) -> Result<arrow::array::DictionaryArray<Int8Type>, String> {
    let mut builder = StringDictionaryBuilder::<Int8Type>::new();
    for _ in 0..row_count {
        builder.append(value).map_err(|error| error.to_string())?;
    }
    Ok(builder.finish())
}

fn build_extra_string_array(extra_code: Vec<Option<i32>>) -> Result<StringArray, String> {
    let mut values: Vec<Option<&str>> = Vec::with_capacity(extra_code.len());
    for maybe_extra_code_value in extra_code {
        match maybe_extra_code_value {
            None | Some(0) => values.push(None),
            Some(1) => values.push(Some("FIRTH")),
            Some(2) => values.push(Some("SPA")),
            Some(3) => values.push(Some("TEST_FAIL")),
            Some(extra_code_value) => return Err(format!("Unsupported REGENIE step 2 extra code: {extra_code_value}")),
        }
    }
    Ok(StringArray::from(values))
}

fn write_record_batch_to_arrow_file(record_batch: &RecordBatch, chunk_file_path: &Path) -> Result<(), String> {
    let output_file = File::create(chunk_file_path).map_err(|error| error.to_string())?;
    let write_options = get_regenie_step2_ipc_write_options().clone();
    let mut writer = FileWriter::try_new_with_options(output_file, &record_batch.schema(), write_options)
        .map_err(|error| error.to_string())?;
    writer.write(record_batch).map_err(|error| error.to_string())?;
    writer.finish().map_err(|error| error.to_string())
}

fn build_regenie_step2_ipc_write_options() -> Result<IpcWriteOptions, String> {
    IpcWriteOptions::default().try_with_compression(Some(CompressionType::ZSTD)).map_err(|error| error.to_string())
}

fn get_regenie_step2_ipc_write_options() -> &'static IpcWriteOptions {
    static REGENIE_STEP2_IPC_WRITE_OPTIONS: OnceLock<IpcWriteOptions> = OnceLock::new();
    REGENIE_STEP2_IPC_WRITE_OPTIONS.get_or_init(|| {
        build_regenie_step2_ipc_write_options()
            .expect("REGENIE step 2 IPC write options should support zstd compression")
    })
}

fn write_final_parquet_from_chunk_files(
    chunks_directory: &Path,
    final_parquet_path: &Path,
    association_mode: &str,
) -> Result<(), OutputError> {
    if association_mode != "regenie2_linear" && association_mode != "regenie2_binary" {
        return Err(OutputError::InvalidInput(format!(
            "Unsupported association mode for Rust output writer finalization: {association_mode}",
        )));
    }
    let mut chunk_file_paths = std::fs::read_dir(chunks_directory)
        .map_err(OutputError::Io)?
        .filter_map(|directory_entry| directory_entry.ok().map(|entry| entry.path()))
        .filter(|chunk_file_path| chunk_file_path.extension().is_some_and(|extension| extension == "arrow"))
        .collect::<Vec<_>>();
    chunk_file_paths.sort();
    let writer_properties = get_regenie_step2_parquet_writer_properties().clone();
    let output_file = File::create(final_parquet_path).map_err(OutputError::Io)?;
    let final_schema = Arc::clone(get_regenie_step2_final_schema());
    let mut parquet_writer = ArrowWriter::try_new(output_file, final_schema, Some(writer_properties))?;
    let chunk_file_count = chunk_file_paths.len();
    let mut output_row_count = 0usize;
    for chunk_file_path in chunk_file_paths {
        let input_file = File::open(&chunk_file_path).map_err(OutputError::Io)?;
        let file_reader = ArrowFileReader::try_new(input_file, None)?;
        for maybe_batch in file_reader {
            let batch = maybe_batch?;
            let projected_batch = project_chunk_batch_to_final_batch(batch)?;
            output_row_count += projected_batch.num_rows();
            parquet_writer.write(&projected_batch)?;
        }
    }
    append_output_footer_metadata(&mut parquet_writer, association_mode, chunk_file_count, output_row_count);
    parquet_writer.close()?;
    Ok(())
}

fn build_regenie_step2_parquet_writer_properties() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .set_max_row_group_row_count(Some(REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE))
        .set_dictionary_enabled(false)
        .set_column_dictionary_enabled(ColumnPath::from("CHROM"), true)
        .set_column_dictionary_enabled(ColumnPath::from("ALLELE0"), true)
        .set_column_dictionary_enabled(ColumnPath::from("ALLELE1"), true)
        .set_column_dictionary_enabled(ColumnPath::from("N"), true)
        .set_column_dictionary_enabled(ColumnPath::from("TEST"), true)
        .set_column_dictionary_enabled(ColumnPath::from("EXTRA"), true)
        .build()
}

fn get_regenie_step2_parquet_writer_properties() -> &'static WriterProperties {
    static REGENIE_STEP2_PARQUET_WRITER_PROPERTIES: OnceLock<WriterProperties> = OnceLock::new();
    REGENIE_STEP2_PARQUET_WRITER_PROPERTIES.get_or_init(build_regenie_step2_parquet_writer_properties)
}

fn append_output_footer_metadata(
    parquet_writer: &mut ArrowWriter<File>,
    association_mode: &str,
    chunk_file_count: usize,
    row_count: usize,
) {
    let metadata_values = [
        ("g.output.schema_version", "1".to_string()),
        ("g.output.association_mode", association_mode.to_string()),
        ("g.output.chunk_file_count", chunk_file_count.to_string()),
        ("g.output.row_count", row_count.to_string()),
        ("g.output.writer", "rust".to_string()),
    ];
    for (key, value) in metadata_values {
        parquet_writer.append_key_value_metadata(KeyValue { key: key.to_string(), value: Some(value) });
    }
}

fn build_regenie_step2_final_schema() -> Schema {
    let large_dictionary_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    let small_dictionary_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
    Schema::new(vec![
        Field::new("CHROM", large_dictionary_type.clone(), true),
        Field::new("GENPOS", DataType::Int64, true),
        Field::new("ID", DataType::Utf8, true),
        Field::new("ALLELE0", large_dictionary_type.clone(), true),
        Field::new("ALLELE1", large_dictionary_type.clone(), true),
        Field::new("A1FREQ", DataType::Float32, true),
        Field::new("INFO", DataType::Float32, true),
        Field::new("N", DataType::Int32, true),
        Field::new("TEST", small_dictionary_type.clone(), true),
        Field::new("BETA", DataType::Float32, true),
        Field::new("SE", DataType::Float32, true),
        Field::new("CHISQ", DataType::Float32, true),
        Field::new("LOG10P", DataType::Float32, true),
        Field::new("EXTRA", DataType::Utf8, true),
    ])
}

fn get_regenie_step2_final_schema() -> &'static Arc<Schema> {
    static REGENIE_STEP2_FINAL_SCHEMA: OnceLock<Arc<Schema>> = OnceLock::new();
    REGENIE_STEP2_FINAL_SCHEMA.get_or_init(|| Arc::new(build_regenie_step2_final_schema()))
}

fn project_chunk_batch_to_final_batch(batch: RecordBatch) -> Result<RecordBatch, OutputError> {
    let final_column_names = [
        "CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1", "A1FREQ", "INFO", "N", "TEST", "BETA", "SE", "CHISQ", "LOG10P",
        "EXTRA",
    ];
    let projected_columns = final_column_names
        .iter()
        .map(|column_name| batch.column_by_name(column_name).cloned())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            OutputError::Runtime("Rust output writer could not project chunk batch to final schema.".to_string())
        })?;
    RecordBatch::try_new(Arc::clone(get_regenie_step2_final_schema()), projected_columns).map_err(OutputError::Arrow)
}

#[cfg(feature = "python")]
fn scan_committed_chunk_identifiers_from_arrow_files(chunks_directory: &Path) -> Result<Vec<i64>, OutputError> {
    if !chunks_directory.exists() {
        return Ok(Vec::new());
    }
    let mut committed_identifiers = BTreeSet::new();
    let mut chunk_file_paths = std::fs::read_dir(chunks_directory)
        .map_err(OutputError::Io)?
        .filter_map(|directory_entry| directory_entry.ok().map(|entry| entry.path()))
        .filter(|chunk_file_path| chunk_file_path.extension().is_some_and(|extension| extension == "arrow"))
        .collect::<Vec<_>>();
    chunk_file_paths.sort();
    for chunk_file_path in chunk_file_paths {
        if let Some((first_chunk_identifier, None)) = parse_chunk_file_name(&chunk_file_path) {
            committed_identifiers.insert(first_chunk_identifier);
            continue;
        }
        let input_file = File::open(&chunk_file_path).map_err(OutputError::Io)?;
        let file_reader = ArrowFileReader::try_new(input_file, None)?;
        for maybe_batch in file_reader {
            let batch = maybe_batch?;
            let chunk_identifier_array = batch
                .column_by_name("chunk_identifier")
                .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
                .ok_or_else(|| {
                    OutputError::Runtime(
                        "Rust output writer could not read chunk identifiers from Arrow chunk.".to_string(),
                    )
                })?;
            for row_index in 0..chunk_identifier_array.len() {
                if !chunk_identifier_array.is_null(row_index) {
                    committed_identifiers.insert(chunk_identifier_array.value(row_index));
                }
            }
        }
    }
    Ok(committed_identifiers.into_iter().collect())
}

#[cfg(feature = "python")]
fn parse_chunk_file_name(chunk_file_path: &Path) -> Option<(i64, Option<i64>)> {
    let file_name = chunk_file_path.file_name()?.to_str()?;
    let chunk_name = file_name.strip_prefix("chunk_")?.strip_suffix(".arrow")?;
    let chunk_parts = chunk_name.split('_').collect::<Vec<_>>();
    match chunk_parts.as_slice() {
        [first_chunk_identifier] => first_chunk_identifier.parse::<i64>().ok().map(|identifier| (identifier, None)),
        [first_chunk_identifier, last_chunk_identifier] => {
            first_chunk_identifier.parse::<i64>().ok().zip(last_chunk_identifier.parse::<i64>().ok().map(Some))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use parquet::file::reader::{FileReader as ParquetFileReader, SerializedFileReader};

    use super::*;

    fn build_test_chunk(chunk_identifier: i64, extra_code: Option<Vec<i32>>) -> RegenieStep2ChunkJob {
        RegenieStep2ChunkJob {
            chunk_identifier,
            variant_start_index: chunk_identifier,
            variant_stop_index: chunk_identifier + 1,
            chrom: vec!["22".to_string()],
            genpos: vec![100 + chunk_identifier],
            id: vec![format!("variant{chunk_identifier}")],
            allele0: vec!["G".to_string()],
            allele1: vec!["A".to_string()],
            a1freq: vec![0.5],
            n: vec![100],
            beta: vec![0.1],
            se: vec![0.01],
            chisq: vec![10.0],
            log10p: vec![5.0],
            extra_code,
        }
    }

    fn build_test_batch(chunks: Vec<RegenieStep2ChunkJob>) -> RegenieStep2ChunkWriteBatch {
        let first_chunk_identifier = chunks.first().map_or(0, |chunk_job| chunk_job.chunk_identifier);
        let last_chunk_identifier =
            chunks.last().map_or(first_chunk_identifier, |chunk_job| chunk_job.chunk_identifier);
        RegenieStep2ChunkWriteBatch {
            chunk_file_name: build_chunk_file_name(first_chunk_identifier, last_chunk_identifier),
            chunks,
        }
    }

    fn create_test_directory() -> PathBuf {
        let unique_suffix =
            SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after Unix epoch").as_nanos();
        let directory_path = std::env::temp_dir().join(format!("g-output-rust-test-{unique_suffix}"));
        std::fs::create_dir_all(&directory_path).expect("test directory should be created");
        directory_path
    }

    #[test]
    fn linear_record_batch_uses_shared_schema_and_null_extra() {
        let record_batch = build_regenie_step2_record_batch(build_test_batch(vec![build_test_chunk(0, None)]))
            .expect("linear record batch should build");

        assert_eq!(record_batch.schema().fields().len(), 17);
        assert!(record_batch.schema().field_with_name("INFO").expect("INFO field should exist").is_nullable());
        assert!(record_batch.schema().field_with_name("EXTRA").expect("EXTRA field should exist").is_nullable());
        assert_eq!(record_batch.num_rows(), 1);
        assert_eq!(record_batch.column_by_name("EXTRA").expect("EXTRA column should exist").null_count(), 1);
    }

    #[test]
    fn binary_record_batch_maps_extra_code_with_same_schema() {
        let linear_record_batch = build_regenie_step2_record_batch(build_test_batch(vec![build_test_chunk(0, None)]))
            .expect("linear record batch should build");
        let binary_record_batch =
            build_regenie_step2_record_batch(build_test_batch(vec![build_test_chunk(1, Some(vec![1]))]))
                .expect("binary record batch should build");

        assert_eq!(linear_record_batch.schema(), binary_record_batch.schema());
        let extra_array = binary_record_batch
            .column_by_name("EXTRA")
            .expect("EXTRA column should exist")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("EXTRA column should be a string array");
        assert_eq!(extra_array.value(0), "FIRTH");
    }

    #[test]
    fn finalization_writes_footer_metadata() {
        let run_directory = create_test_directory();
        let chunks_directory = run_directory.join("chunks");
        std::fs::create_dir_all(&chunks_directory).expect("chunk directory should be created");
        let config = RustOutputWriterConfig {
            run_directory: run_directory.clone(),
            chunks_directory: chunks_directory.clone(),
            association_mode: "regenie2_binary".to_string(),
            finalize_parquet: true,
        };
        write_regenie_step2_chunk_job(
            &config,
            build_test_batch(vec![build_test_chunk(0, Some(vec![1])), build_test_chunk(1, Some(vec![0]))]),
        )
        .expect("chunk batch should write");

        let final_parquet_path = run_directory.join("final.parquet");
        write_final_parquet_from_chunk_files(&chunks_directory, &final_parquet_path, "regenie2_binary")
            .expect("final parquet should write");

        let parquet_file = File::open(final_parquet_path).expect("final parquet should open");
        let parquet_reader = SerializedFileReader::new(parquet_file).expect("parquet reader should open");
        let key_value_metadata =
            parquet_reader.metadata().file_metadata().key_value_metadata().expect("footer metadata should exist");
        let metadata_value = |key: &str| {
            key_value_metadata.iter().find(|entry| entry.key == key).and_then(|entry| entry.value.as_deref())
        };
        assert_eq!(metadata_value("g.output.schema_version"), Some("1"));
        assert_eq!(metadata_value("g.output.association_mode"), Some("regenie2_binary"));
        assert_eq!(metadata_value("g.output.chunk_file_count"), Some("1"));
        assert_eq!(metadata_value("g.output.row_count"), Some("2"));
        assert_eq!(metadata_value("g.output.writer"), Some("rust"));

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }
}
