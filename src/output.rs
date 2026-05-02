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
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use crossbeam_channel::{Receiver, Sender, bounded};
use numpy::{PyReadonlyArray1, PyUntypedArrayMethods};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

const REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE: usize = 122_880;

#[derive(Clone)]
struct RustOutputWriterConfig {
    run_directory: PathBuf,
    chunks_directory: PathBuf,
    association_mode: String,
    finalize_parquet: bool,
}

#[derive(Clone)]
struct RegenieStep2ChunkWriteJob {
    chunk_file_name: String,
    chunk_identifier: Vec<i64>,
    variant_start_index: Vec<i64>,
    variant_stop_index: Vec<i64>,
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

enum OutputWriteJob {
    RegenieStep2(Box<RegenieStep2ChunkWriteJob>),
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
    #[pyo3(signature = (chunk_file_name, chunk_identifier, variant_start_index, variant_stop_index, chrom, genpos, variant_id, allele0, allele1, a1freq, n, beta, se, chisq, log10p, extra_code=None))]
    fn enqueue_regenie_step2_chunk_batch(
        &self,
        chunk_file_name: String,
        chunk_identifier: PyReadonlyArray1<'_, i64>,
        variant_start_index: PyReadonlyArray1<'_, i64>,
        variant_stop_index: PyReadonlyArray1<'_, i64>,
        chrom: Vec<String>,
        genpos: PyReadonlyArray1<'_, i64>,
        variant_id: Vec<String>,
        allele0: Vec<String>,
        allele1: Vec<String>,
        a1freq: PyReadonlyArray1<'_, f32>,
        n: PyReadonlyArray1<'_, i32>,
        beta: PyReadonlyArray1<'_, f32>,
        se: PyReadonlyArray1<'_, f32>,
        chisq: PyReadonlyArray1<'_, f32>,
        log10p: PyReadonlyArray1<'_, f32>,
        extra_code: Option<PyReadonlyArray1<'_, i32>>,
    ) -> PyResult<()> {
        if self.config.association_mode != "regenie2_linear" && self.config.association_mode != "regenie2_binary" {
            return Err(PyValueError::new_err(
                "Rust output backend only supports REGENIE step 2 quantitative and binary output.",
            ));
        }
        let row_count = genpos.len();
        let observed_lengths = [
            chunk_identifier.len(),
            variant_start_index.len(),
            variant_stop_index.len(),
            chrom.len(),
            variant_id.len(),
            allele0.len(),
            allele1.len(),
            a1freq.len(),
            n.len(),
            beta.len(),
            se.len(),
            chisq.len(),
            log10p.len(),
        ];
        validate_column_lengths(row_count, observed_lengths.as_slice())?;
        let job = RegenieStep2ChunkWriteJob {
            chunk_file_name,
            chunk_identifier: chunk_identifier.as_slice()?.to_vec(),
            variant_start_index: variant_start_index.as_slice()?.to_vec(),
            variant_stop_index: variant_stop_index.as_slice()?.to_vec(),
            chrom,
            genpos: genpos.as_slice()?.to_vec(),
            id: variant_id,
            allele0,
            allele1,
            a1freq: a1freq.as_slice()?.to_vec(),
            n: n.as_slice()?.to_vec(),
            beta: beta.as_slice()?.to_vec(),
            se: se.as_slice()?.to_vec(),
            chisq: chisq.as_slice()?.to_vec(),
            log10p: log10p.as_slice()?.to_vec(),
            extra_code: extra_code.map(|extra_code_array| extra_code_array.as_slice().map(<[i32]>::to_vec)).transpose()?,
        };
        self.raise_if_worker_failed()?;
        let sender_guard =
            self.sender.lock().map_err(|_| PyRuntimeError::new_err("Rust output writer sender lock was poisoned."))?;
        let sender = sender_guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Rust output writer session is already closed."))?;
        sender
            .send(OutputWriteJob::RegenieStep2(Box::new(job)))
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
        write_final_parquet_from_chunk_files(
            &self.config.chunks_directory,
            &final_parquet_path,
            &self.config.association_mode,
        )?;
        Ok(Some(final_parquet_path.display().to_string()))
    }

    fn abort(&self) -> PyResult<()> {
        close_writer_sender(&self.sender, self.worker_thread_count)?;
        join_writer_threads(&self.worker_handles)?;
        Ok(())
    }
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::missing_errors_doc)]
pub fn finalize_output_run_chunks(
    run_directory: String,
    chunks_directory: String,
    association_mode: String,
) -> PyResult<String> {
    let final_parquet_path = PathBuf::from(run_directory).join("final.parquet");
    write_final_parquet_from_chunk_files(Path::new(&chunks_directory), &final_parquet_path, &association_mode)?;
    Ok(final_parquet_path.display().to_string())
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

fn close_writer_sender(sender: &Mutex<Option<Sender<OutputWriteJob>>>, worker_thread_count: usize) -> PyResult<()> {
    let mut sender_guard =
        sender.lock().map_err(|_| PyRuntimeError::new_err("Rust output writer sender lock was poisoned."))?;
    if let Some(active_sender) = sender_guard.take() {
        for _ in 0..worker_thread_count {
            active_sender.send(OutputWriteJob::Shutdown).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        }
    }
    Ok(())
}

fn join_writer_threads(worker_handles: &Mutex<Vec<JoinHandle<()>>>) -> PyResult<()> {
    let mut worker_handles_guard =
        worker_handles.lock().map_err(|_| PyRuntimeError::new_err("Rust output writer handle lock was poisoned."))?;
    while let Some(worker_handle) = worker_handles_guard.pop() {
        worker_handle.join().map_err(|_| PyRuntimeError::new_err("Rust output writer worker thread panicked."))?;
    }
    Ok(())
}

fn validate_column_lengths(expected_row_count: usize, observed_lengths: &[usize]) -> PyResult<()> {
    if observed_lengths.iter().all(|observed_length| *observed_length == expected_row_count) {
        return Ok(());
    }
    Err(PyValueError::new_err("Rust output writer batch column lengths do not all match the expected row count."))
}

fn run_output_writer_worker(
    receiver: Receiver<OutputWriteJob>,
    config: RustOutputWriterConfig,
    worker_errors: Arc<Mutex<Vec<String>>>,
) {
    while let Ok(job) = receiver.recv() {
        let write_result = match job {
            OutputWriteJob::RegenieStep2(regenie_step2_job) => write_regenie_step2_chunk_job(&config, *regenie_step2_job),
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

fn write_regenie_step2_chunk_job(config: &RustOutputWriterConfig, job: RegenieStep2ChunkWriteJob) -> Result<(), String> {
    let chunk_file_path = config.chunks_directory.join(&job.chunk_file_name);
    let temporary_chunk_file_path = chunk_file_path.with_extension("arrow.tmp");
    let record_batch = build_regenie_step2_record_batch(job)?;
    write_record_batch_to_arrow_file(&record_batch, &temporary_chunk_file_path)?;
    std::fs::rename(&temporary_chunk_file_path, &chunk_file_path).map_err(|error| error.to_string())?;
    Ok(())
}

fn build_regenie_step2_record_batch(job: RegenieStep2ChunkWriteJob) -> Result<RecordBatch, String> {
    let schema = get_regenie_step2_chunk_schema();
    let row_count = job.genpos.len();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(job.chunk_identifier)),
        Arc::new(Int64Array::from(job.variant_start_index)),
        Arc::new(Int64Array::from(job.variant_stop_index)),
        Arc::new(build_dictionary_string_array(&job.chrom)?),
        Arc::new(Int64Array::from(job.genpos)),
        Arc::new(StringArray::from(job.id)),
        Arc::new(build_dictionary_string_array(&job.allele0)?),
        Arc::new(build_dictionary_string_array(&job.allele1)?),
        Arc::new(Float32Array::from(job.a1freq)),
        new_null_array(&DataType::Float32, row_count),
        Arc::new(Int32Array::from(job.n)),
        Arc::new(build_constant_dictionary_string_array(row_count, "ADD")?),
        Arc::new(Float32Array::from(job.beta)),
        Arc::new(Float32Array::from(job.se)),
        Arc::new(Float32Array::from(job.chisq)),
        Arc::new(Float32Array::from(job.log10p)),
        Arc::new(build_extra_string_array(job.extra_code, row_count)?),
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

fn build_extra_string_array(extra_code: Option<Vec<i32>>, row_count: usize) -> Result<StringArray, String> {
    let mut values: Vec<Option<&str>> = Vec::with_capacity(row_count);
    match extra_code {
        None => {
            for _ in 0..row_count {
                values.push(None);
            }
        }
        Some(extra_code_values) => {
            if extra_code_values.len() != row_count {
                return Err(format!(
                    "Rust output writer extra-code length {} did not match row count {row_count}.",
                    extra_code_values.len(),
                ));
            }
            for extra_code_value in extra_code_values {
                match extra_code_value {
                    0 => values.push(None),
                    1 => values.push(Some("FIRTH")),
                    2 => values.push(Some("SPA")),
                    3 => values.push(Some("TEST_FAIL")),
                    _ => return Err(format!("Unsupported REGENIE step 2 extra code: {extra_code_value}")),
                }
            }
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
        build_regenie_step2_ipc_write_options().expect("REGENIE step 2 IPC write options should support zstd compression")
    })
}

fn write_final_parquet_from_chunk_files(
    chunks_directory: &Path,
    final_parquet_path: &Path,
    association_mode: &str,
) -> PyResult<()> {
    if association_mode != "regenie2_linear" && association_mode != "regenie2_binary" {
        return Err(PyRuntimeError::new_err(format!(
            "Unsupported association mode for Rust output writer finalization: {association_mode}",
        )));
    }
    let mut chunk_file_paths = std::fs::read_dir(chunks_directory)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .filter_map(|directory_entry| directory_entry.ok().map(|entry| entry.path()))
        .filter(|chunk_file_path| chunk_file_path.extension().is_some_and(|extension| extension == "arrow"))
        .collect::<Vec<_>>();
    chunk_file_paths.sort();
    let writer_properties = get_regenie_step2_parquet_writer_properties().clone();
    let output_file = File::create(final_parquet_path).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let final_schema = Arc::clone(get_regenie_step2_final_schema());
    let mut parquet_writer = ArrowWriter::try_new(output_file, final_schema, Some(writer_properties))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    for chunk_file_path in chunk_file_paths {
        let input_file = File::open(&chunk_file_path).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        let file_reader =
            FileReader::try_new(input_file, None).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        for maybe_batch in file_reader {
            let batch = maybe_batch.map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            parquet_writer
                .write(&project_chunk_batch_to_final_batch(batch)?)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        }
    }
    parquet_writer.close().map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(())
}

fn build_regenie_step2_parquet_writer_properties() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .set_max_row_group_size(REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE)
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

fn project_chunk_batch_to_final_batch(batch: RecordBatch) -> PyResult<RecordBatch> {
    let final_column_names = [
        "CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1", "A1FREQ", "INFO", "N", "TEST", "BETA", "SE", "CHISQ",
        "LOG10P", "EXTRA",
    ];
    let projected_columns = final_column_names
        .iter()
        .map(|column_name| batch.column_by_name(column_name).cloned())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| PyRuntimeError::new_err("Rust output writer could not project chunk batch to final schema."))?;
    RecordBatch::try_new(Arc::clone(get_regenie_step2_final_schema()), projected_columns)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}
