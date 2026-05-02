#![allow(clippy::needless_pass_by_value)]

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;

use arrow::array::{
    ArrayRef, Float32Array, Int32Array, Int64Array, RecordBatch, StringArray,
    StringDictionaryBuilder,
};
use arrow::datatypes::{DataType, Field, Int8Type, Int32Type, Schema};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use arrow::ipc::CompressionType;
use crossbeam_channel::{bounded, Receiver, Sender};
use numpy::{PyReadonlyArray1, PyUntypedArrayMethods};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

const BINARY_PARQUET_MAX_ROW_GROUP_SIZE: usize = 122_880;

#[derive(Clone)]
struct RustOutputWriterConfig {
    run_directory: PathBuf,
    chunks_directory: PathBuf,
    association_mode: String,
    finalize_parquet: bool,
}

#[derive(Clone)]
struct BinaryChunkWriteJob {
    chunk_file_name: String,
    chunk_identifier: Vec<i64>,
    variant_start_index: Vec<i64>,
    variant_stop_index: Vec<i64>,
    chromosome: Vec<String>,
    position: Vec<i64>,
    variant_identifier: Vec<String>,
    allele_zero: Vec<String>,
    allele_one: Vec<String>,
    allele_one_frequency: Vec<f32>,
    observation_count: Vec<i32>,
    beta: Vec<f32>,
    standard_error: Vec<f32>,
    chi_squared: Vec<f32>,
    log10_p_value: Vec<f32>,
    extra_code: Vec<i32>,
}

enum OutputWriteJob {
    Binary(Box<BinaryChunkWriteJob>),
    Shutdown,
}

#[pyclass]
pub struct PyOutputWriterSession {
    sender: Mutex<Option<Sender<OutputWriteJob>>>,
    worker_handles: Mutex<Vec<JoinHandle<()>>>,
    worker_errors: Arc<Mutex<Vec<String>>>,
    config: RustOutputWriterConfig,
    worker_thread_count: usize,
}

#[pymethods]
impl PyOutputWriterSession {
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
        if writer_thread_count == 0 {
            return Err(PyValueError::new_err("Writer thread count must be at least 1."));
        }
        let config = RustOutputWriterConfig {
            run_directory: PathBuf::from(run_directory),
            chunks_directory: PathBuf::from(chunks_directory),
            association_mode,
            finalize_parquet,
        };
        let (sender, receiver) = bounded(writer_queue_depth.max(1));
        let worker_errors = Arc::new(Mutex::new(Vec::new()));
        let mut worker_handles = Vec::with_capacity(writer_thread_count);
        for _ in 0..writer_thread_count {
            let receiver_clone = receiver.clone();
            let config_clone = config.clone();
            let worker_errors_clone = Arc::clone(&worker_errors);
            worker_handles.push(std::thread::spawn(move || {
                run_output_writer_worker(receiver_clone, config_clone, worker_errors_clone);
            }));
        }
        Ok(Self {
            sender: Mutex::new(Some(sender)),
            worker_handles: Mutex::new(worker_handles),
            worker_errors,
            config,
            worker_thread_count: writer_thread_count,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (chunk_file_name, chunk_identifier, variant_start_index, variant_stop_index, chromosome, position, variant_identifier, allele_zero, allele_one, allele_one_frequency, observation_count, beta, standard_error, chi_squared, log10_p_value, extra_code))]
    fn enqueue_binary_chunk_batch(
        &self,
        chunk_file_name: String,
        chunk_identifier: PyReadonlyArray1<'_, i64>,
        variant_start_index: PyReadonlyArray1<'_, i64>,
        variant_stop_index: PyReadonlyArray1<'_, i64>,
        chromosome: Vec<String>,
        position: PyReadonlyArray1<'_, i64>,
        variant_identifier: Vec<String>,
        allele_zero: Vec<String>,
        allele_one: Vec<String>,
        allele_one_frequency: PyReadonlyArray1<'_, f32>,
        observation_count: PyReadonlyArray1<'_, i32>,
        beta: PyReadonlyArray1<'_, f32>,
        standard_error: PyReadonlyArray1<'_, f32>,
        chi_squared: PyReadonlyArray1<'_, f32>,
        log10_p_value: PyReadonlyArray1<'_, f32>,
        extra_code: PyReadonlyArray1<'_, i32>,
    ) -> PyResult<()> {
        if self.config.association_mode != "regenie2_binary" {
            return Err(PyValueError::new_err(
                "Rust output backend currently supports only REGENIE step 2 binary output.",
            ));
        }
        let row_count = position.len();
        let observed_lengths = [
            chunk_identifier.len(),
            variant_start_index.len(),
            variant_stop_index.len(),
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
            extra_code.len(),
        ];
        validate_column_lengths(
            row_count,
            observed_lengths.as_slice(),
        )?;
        let job = BinaryChunkWriteJob {
            chunk_file_name,
            chunk_identifier: chunk_identifier.as_slice()?.to_vec(),
            variant_start_index: variant_start_index.as_slice()?.to_vec(),
            variant_stop_index: variant_stop_index.as_slice()?.to_vec(),
            chromosome,
            position: position.as_slice()?.to_vec(),
            variant_identifier,
            allele_zero,
            allele_one,
            allele_one_frequency: allele_one_frequency.as_slice()?.to_vec(),
            observation_count: observation_count.as_slice()?.to_vec(),
            beta: beta.as_slice()?.to_vec(),
            standard_error: standard_error.as_slice()?.to_vec(),
            chi_squared: chi_squared.as_slice()?.to_vec(),
            log10_p_value: log10_p_value.as_slice()?.to_vec(),
            extra_code: extra_code.as_slice()?.to_vec(),
        };
        self.raise_if_worker_failed()?;
        let sender_guard = self
            .sender
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Rust output writer sender lock was poisoned."))?;
        let sender = sender_guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Rust output writer session is already closed."))?;
        sender
            .send(OutputWriteJob::Binary(Box::new(job)))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(())
    }

    fn finish(&self) -> PyResult<Option<String>> {
        close_writer_sender(&self.sender, self.worker_thread_count)?;
        join_writer_threads(&self.worker_handles)?;
        self.raise_if_worker_failed()?;
        if !self.config.finalize_parquet {
            return Ok(None);
        }
        let final_parquet_path = self.config.run_directory.join("final.parquet");
        write_final_parquet_from_chunk_files(&self.config.chunks_directory, &final_parquet_path)?;
        Ok(Some(final_parquet_path.display().to_string()))
    }

    fn abort(&self) -> PyResult<()> {
        close_writer_sender(&self.sender, self.worker_thread_count)?;
        join_writer_threads(&self.worker_handles)?;
        Ok(())
    }
}

impl PyOutputWriterSession {
    fn raise_if_worker_failed(&self) -> PyResult<()> {
        let worker_errors = self
            .worker_errors
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Rust output writer error lock was poisoned."))?;
        if let Some(first_error) = worker_errors.first() {
            return Err(PyRuntimeError::new_err(first_error.clone()));
        }
        Ok(())
    }
}

fn close_writer_sender(
    sender: &Mutex<Option<Sender<OutputWriteJob>>>,
    worker_thread_count: usize,
) -> PyResult<()> {
    let mut sender_guard = sender
        .lock()
        .map_err(|_| PyRuntimeError::new_err("Rust output writer sender lock was poisoned."))?;
    if let Some(active_sender) = sender_guard.take() {
        for _ in 0..worker_thread_count {
            active_sender
                .send(OutputWriteJob::Shutdown)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        }
    }
    Ok(())
}

fn join_writer_threads(worker_handles: &Mutex<Vec<JoinHandle<()>>>) -> PyResult<()> {
    let mut worker_handles_guard = worker_handles
        .lock()
        .map_err(|_| PyRuntimeError::new_err("Rust output writer handle lock was poisoned."))?;
    while let Some(worker_handle) = worker_handles_guard.pop() {
        worker_handle
            .join()
            .map_err(|_| PyRuntimeError::new_err("Rust output writer worker thread panicked."))?;
    }
    Ok(())
}

fn validate_column_lengths(expected_row_count: usize, observed_lengths: &[usize]) -> PyResult<()> {
    if observed_lengths.iter().all(|observed_length| *observed_length == expected_row_count) {
        return Ok(());
    }
    Err(PyValueError::new_err(
        "Rust output writer batch column lengths do not all match the expected row count.",
    ))
}

fn run_output_writer_worker(
    receiver: Receiver<OutputWriteJob>,
    config: RustOutputWriterConfig,
    worker_errors: Arc<Mutex<Vec<String>>>,
) {
    while let Ok(job) = receiver.recv() {
        let write_result = match job {
            OutputWriteJob::Binary(binary_job) => write_binary_chunk_job(&config, *binary_job),
            OutputWriteJob::Shutdown => return,
        };
        if let Err(error) = write_result {
            if let Ok(mut worker_errors_guard) = worker_errors.lock() {
                worker_errors_guard.push(error.clone());
            }
            return;
        }
    }
}

fn write_binary_chunk_job(config: &RustOutputWriterConfig, job: BinaryChunkWriteJob) -> Result<(), String> {
    let chunk_file_path = config.chunks_directory.join(&job.chunk_file_name);
    let temporary_chunk_file_path = chunk_file_path.with_extension("arrow.tmp");
    let record_batch = build_binary_record_batch(job)?;
    write_record_batch_to_arrow_file(&record_batch, &temporary_chunk_file_path)?;
    std::fs::rename(&temporary_chunk_file_path, &chunk_file_path).map_err(|error| error.to_string())?;
    Ok(())
}

fn build_binary_record_batch(job: BinaryChunkWriteJob) -> Result<RecordBatch, String> {
    let row_count = job.position.len();
    let info = vec![1.0_f32; row_count];
    let schema = get_binary_output_schema();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(job.chunk_identifier)),
        Arc::new(Int64Array::from(job.variant_start_index)),
        Arc::new(Int64Array::from(job.variant_stop_index)),
        Arc::new(build_dictionary_string_array(&job.chromosome)?),
        Arc::new(Int64Array::from(job.position)),
        Arc::new(StringArray::from(job.variant_identifier)),
        Arc::new(build_dictionary_string_array(&job.allele_zero)?),
        Arc::new(build_dictionary_string_array(&job.allele_one)?),
        Arc::new(Float32Array::from(job.allele_one_frequency)),
        Arc::new(Float32Array::from(info)),
        Arc::new(Int32Array::from(job.observation_count)),
        Arc::new(build_small_constant_dictionary_string_array("ADD", row_count)?),
        Arc::new(Float32Array::from(job.beta)),
        Arc::new(Float32Array::from(job.standard_error)),
        Arc::new(Float32Array::from(job.chi_squared)),
        Arc::new(Float32Array::from(job.log10_p_value)),
        Arc::new(build_small_extra_dictionary_array(&job.extra_code)?),
    ];
    RecordBatch::try_new(Arc::clone(schema), columns).map_err(|error| error.to_string())
}

fn build_binary_output_schema() -> Schema {
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
        Field::new("EXTRA", small_dictionary_type, true),
    ])
}

fn get_binary_output_schema() -> &'static Arc<Schema> {
    static BINARY_OUTPUT_SCHEMA: OnceLock<Arc<Schema>> = OnceLock::new();
    BINARY_OUTPUT_SCHEMA.get_or_init(|| Arc::new(build_binary_output_schema()))
}

fn build_dictionary_string_array(values: &[String]) -> Result<arrow::array::DictionaryArray<Int32Type>, String> {
    let mut builder = StringDictionaryBuilder::<Int32Type>::new();
    for value in values {
        builder.append(value).map_err(|error| error.to_string())?;
    }
    Ok(builder.finish())
}

fn build_small_constant_dictionary_string_array(
    value: &str,
    row_count: usize,
) -> Result<arrow::array::DictionaryArray<Int8Type>, String> {
    let mut builder = StringDictionaryBuilder::<Int8Type>::new();
    for _row_index in 0..row_count {
        builder.append(value).map_err(|error| error.to_string())?;
    }
    Ok(builder.finish())
}

fn build_small_extra_dictionary_array(extra_codes: &[i32]) -> Result<arrow::array::DictionaryArray<Int8Type>, String> {
    let mut builder = StringDictionaryBuilder::<Int8Type>::new();
    for extra_code in extra_codes {
        let label = match *extra_code {
            1 => "FIRTH",
            2 => "SPA",
            3 => "TEST_FAIL",
            _ => "NA",
        };
        builder.append(label).map_err(|error| error.to_string())?;
    }
    Ok(builder.finish())
}

fn write_record_batch_to_arrow_file(record_batch: &RecordBatch, chunk_file_path: &Path) -> Result<(), String> {
    let output_file = File::create(chunk_file_path).map_err(|error| error.to_string())?;
    let write_options = get_binary_ipc_write_options().clone();
    let mut writer = FileWriter::try_new_with_options(output_file, &record_batch.schema(), write_options)
        .map_err(|error| error.to_string())?;
    writer.write(record_batch).map_err(|error| error.to_string())?;
    writer.finish().map_err(|error| error.to_string())
}

fn build_binary_ipc_write_options() -> Result<IpcWriteOptions, String> {
    IpcWriteOptions::default()
        .try_with_compression(Some(CompressionType::ZSTD))
        .map_err(|error| error.to_string())
}

fn get_binary_ipc_write_options() -> &'static IpcWriteOptions {
    static BINARY_IPC_WRITE_OPTIONS: OnceLock<IpcWriteOptions> = OnceLock::new();
    BINARY_IPC_WRITE_OPTIONS.get_or_init(|| {
        build_binary_ipc_write_options().expect("binary IPC write options should support zstd compression")
    })
}

fn write_final_parquet_from_chunk_files(chunks_directory: &Path, final_parquet_path: &Path) -> PyResult<()> {
    let mut chunk_file_paths = std::fs::read_dir(chunks_directory)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .filter_map(|directory_entry| directory_entry.ok().map(|entry| entry.path()))
        .filter(|chunk_file_path| chunk_file_path.extension().is_some_and(|extension| extension == "arrow"))
        .collect::<Vec<_>>();
    chunk_file_paths.sort();
    let writer_properties = get_binary_parquet_writer_properties().clone();
    let output_file = File::create(final_parquet_path).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let schema = Arc::clone(get_binary_output_schema());
    let mut parquet_writer =
        ArrowWriter::try_new(output_file, schema, Some(writer_properties)).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    for chunk_file_path in chunk_file_paths {
        let input_file = File::open(&chunk_file_path).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let file_reader =
            FileReader::try_new(input_file, None).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        for maybe_batch in file_reader {
            let batch = maybe_batch.map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            parquet_writer.write(&batch).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        }
    }
    parquet_writer.close().map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(())
}

fn build_binary_parquet_writer_properties() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .set_max_row_group_size(BINARY_PARQUET_MAX_ROW_GROUP_SIZE)
        .set_dictionary_enabled(false)
        .set_column_dictionary_enabled(ColumnPath::from("chunk_identifier"), true)
        .set_column_dictionary_enabled(ColumnPath::from("variant_start_index"), true)
        .set_column_dictionary_enabled(ColumnPath::from("variant_stop_index"), true)
        .set_column_dictionary_enabled(ColumnPath::from("CHROM"), true)
        .set_column_dictionary_enabled(ColumnPath::from("ALLELE0"), true)
        .set_column_dictionary_enabled(ColumnPath::from("ALLELE1"), true)
        .set_column_dictionary_enabled(ColumnPath::from("N"), true)
        .set_column_dictionary_enabled(ColumnPath::from("TEST"), true)
        .set_column_dictionary_enabled(ColumnPath::from("EXTRA"), true)
        .build()
}

fn get_binary_parquet_writer_properties() -> &'static WriterProperties {
    static BINARY_PARQUET_WRITER_PROPERTIES: OnceLock<WriterProperties> = OnceLock::new();
    BINARY_PARQUET_WRITER_PROPERTIES.get_or_init(build_binary_parquet_writer_properties)
}
