use std::collections::BTreeSet;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use arrow::array::{ArrayRef, RecordBatch};
use arrow::ipc::reader::FileReader as ArrowFileReader;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;
use serde_json::Value;

use crate::error::OutputError;
use crate::manifest;
use crate::schema;
use crate::writer::OutputFileFormat;

const REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE: usize = 122_880;

#[derive(Clone, Copy, Default)]
pub(crate) struct RegenieStep2FinalizationTiming {
    pub(crate) chunk_file_count: u64,
    pub(crate) batch_count: u64,
    pub(crate) row_count: u64,
    pub(crate) list_chunk_files_seconds: f64,
    pub(crate) parquet_writer_properties_seconds: f64,
    pub(crate) parquet_file_create_seconds: f64,
    pub(crate) parquet_writer_init_seconds: f64,
    pub(crate) arrow_file_open_seconds: f64,
    pub(crate) arrow_reader_init_seconds: f64,
    pub(crate) arrow_batch_read_seconds: f64,
    pub(crate) read_arrow_seconds: f64,
    pub(crate) project_batch_seconds: f64,
    pub(crate) write_parquet_seconds: f64,
    pub(crate) footer_metadata_seconds: f64,
    pub(crate) close_writer_seconds: f64,
    pub(crate) manifest_update_seconds: f64,
    pub(crate) arrow_file_bytes: u64,
    pub(crate) parquet_file_bytes: u64,
    pub(crate) total_seconds: f64,
}

pub(crate) fn write_final_parquet_from_chunk_files(
    chunks_directory: &Path,
    final_parquet_path: &Path,
    association_mode: &str,
    output_format: OutputFileFormat,
) -> Result<(), OutputError> {
    write_final_parquet_from_chunk_files_with_timing(
        chunks_directory,
        final_parquet_path,
        association_mode,
        output_format,
    )
    .map(|_| ())
}

pub(crate) fn write_final_parquet_from_chunk_files_with_timing(
    chunks_directory: &Path,
    final_parquet_path: &Path,
    association_mode: &str,
    output_format: OutputFileFormat,
) -> Result<RegenieStep2FinalizationTiming, OutputError> {
    write_final_parquet_from_chunk_files_with_optional_dtype(
        chunks_directory,
        final_parquet_path,
        association_mode,
        output_format,
        None,
    )
}

pub(crate) fn write_final_parquet_from_chunk_files_with_timing_for_dtype(
    chunks_directory: &Path,
    final_parquet_path: &Path,
    association_mode: &str,
    output_format: OutputFileFormat,
    output_statistic_dtype: schema::OutputStatisticDtype,
) -> Result<RegenieStep2FinalizationTiming, OutputError> {
    write_final_parquet_from_chunk_files_with_optional_dtype(
        chunks_directory,
        final_parquet_path,
        association_mode,
        output_format,
        Some(output_statistic_dtype),
    )
}

fn write_final_parquet_from_chunk_files_with_optional_dtype(
    chunks_directory: &Path,
    final_parquet_path: &Path,
    association_mode: &str,
    output_format: OutputFileFormat,
    output_statistic_dtype_override: Option<schema::OutputStatisticDtype>,
) -> Result<RegenieStep2FinalizationTiming, OutputError> {
    let total_start_time = Instant::now();
    if association_mode != "regenie2_linear" && association_mode != "regenie2_binary" {
        return Err(OutputError::InvalidInput(format!(
            "Unsupported association mode for Rust output writer finalization: {association_mode}",
        )));
    }

    let list_chunk_files_start_time = Instant::now();
    let run_directory = final_parquet_path
        .parent()
        .ok_or_else(|| OutputError::InvalidInput("Final Parquet path must have a parent run directory.".to_string()))?;
    let manifest_commits = manifest::read_run_manifest_chunk_commits(run_directory)?;
    let chunk_file_paths = manifest_output_chunk_file_paths(chunks_directory, output_format, &manifest_commits)?;
    let list_chunk_files_seconds = list_chunk_files_start_time.elapsed().as_secs_f64();

    let parquet_writer_properties_start_time = Instant::now();
    let writer_properties = get_regenie_step2_parquet_writer_properties().clone();
    let parquet_writer_properties_seconds = parquet_writer_properties_start_time.elapsed().as_secs_f64();

    let parquet_file_create_start_time = Instant::now();
    let output_file = File::create(final_parquet_path).map_err(OutputError::runtime)?;
    let parquet_file_create_seconds = parquet_file_create_start_time.elapsed().as_secs_f64();

    let output_statistic_dtype = match output_statistic_dtype_override {
        Some(output_statistic_dtype) => output_statistic_dtype,
        None => read_output_statistic_dtype_from_manifest(run_directory)?,
    };
    let final_schema = Arc::clone(schema::get_regenie_step2_final_schema(output_statistic_dtype));
    let parquet_writer_init_start_time = Instant::now();
    let mut parquet_writer =
        ArrowWriter::try_new(output_file, final_schema, Some(writer_properties)).map_err(OutputError::runtime)?;
    let parquet_writer_init_seconds = parquet_writer_init_start_time.elapsed().as_secs_f64();

    let chunk_file_count = chunk_file_paths.len();
    let mut output_row_count = 0usize;
    let mut batch_count = 0u64;
    let mut arrow_file_open_seconds = 0.0;
    let mut arrow_reader_init_seconds = 0.0;
    let mut arrow_batch_read_seconds = 0.0;
    let mut read_arrow_seconds = 0.0;
    let mut project_batch_seconds = 0.0;
    let mut write_parquet_seconds = 0.0;
    let mut arrow_file_bytes = 0u64;
    for chunk_file_path in chunk_file_paths {
        arrow_file_bytes =
            arrow_file_bytes.saturating_add(std::fs::metadata(&chunk_file_path).map_err(OutputError::runtime)?.len());
        for maybe_batch in read_output_chunk_file_batches(
            &chunk_file_path,
            output_format,
            &mut arrow_file_open_seconds,
            &mut arrow_reader_init_seconds,
            &mut read_arrow_seconds,
        )? {
            let read_batch_start_time = Instant::now();
            let batch = maybe_batch.map_err(OutputError::runtime)?;
            let current_arrow_batch_read_seconds = read_batch_start_time.elapsed().as_secs_f64();
            arrow_batch_read_seconds += current_arrow_batch_read_seconds;
            read_arrow_seconds += current_arrow_batch_read_seconds;

            let project_batch_start_time = Instant::now();
            let projected_batch = prepare_chunk_batch_for_final_writer(batch)?;
            project_batch_seconds += project_batch_start_time.elapsed().as_secs_f64();
            output_row_count += projected_batch.num_rows();

            let write_parquet_start_time = Instant::now();
            parquet_writer.write(&projected_batch).map_err(OutputError::runtime)?;
            write_parquet_seconds += write_parquet_start_time.elapsed().as_secs_f64();
            batch_count += 1;
        }
    }

    let footer_metadata_start_time = Instant::now();
    append_output_footer_metadata(&mut parquet_writer, association_mode, chunk_file_count, output_row_count);
    let footer_metadata_seconds = footer_metadata_start_time.elapsed().as_secs_f64();

    let close_writer_start_time = Instant::now();
    parquet_writer.close().map_err(OutputError::runtime)?;
    let close_writer_seconds = close_writer_start_time.elapsed().as_secs_f64();
    let parquet_file_bytes = std::fs::metadata(final_parquet_path).map_err(OutputError::runtime)?.len();

    let manifest_update_start_time = Instant::now();
    manifest::mark_run_manifest_finalized(final_parquet_path, output_row_count, chunk_file_count)?;
    let manifest_update_seconds = manifest_update_start_time.elapsed().as_secs_f64();

    Ok(RegenieStep2FinalizationTiming {
        chunk_file_count: u64::try_from(chunk_file_count).map_err(OutputError::runtime)?,
        batch_count,
        row_count: u64::try_from(output_row_count).map_err(OutputError::runtime)?,
        list_chunk_files_seconds,
        parquet_writer_properties_seconds,
        parquet_file_create_seconds,
        parquet_writer_init_seconds,
        arrow_file_open_seconds,
        arrow_reader_init_seconds,
        arrow_batch_read_seconds,
        read_arrow_seconds,
        project_batch_seconds,
        write_parquet_seconds,
        footer_metadata_seconds,
        close_writer_seconds,
        manifest_update_seconds,
        arrow_file_bytes,
        parquet_file_bytes,
        total_seconds: total_start_time.elapsed().as_secs_f64(),
    })
}

pub(crate) fn manifest_output_chunk_file_paths(
    chunks_directory: &Path,
    output_format: OutputFileFormat,
    manifest_commits: &[manifest::RunManifestChunkCommit],
) -> Result<Vec<PathBuf>, OutputError> {
    let expected_output_format = output_format_name(output_format);
    let manifest_file_names = manifest_commits
        .iter()
        .filter(|chunk_commit| chunk_commit.output_format == expected_output_format)
        .map(|chunk_commit| chunk_commit.chunk_file_name.clone())
        .collect::<BTreeSet<_>>();
    reject_unmanifested_output_chunk_files(chunks_directory, output_format, &manifest_file_names)?;
    let mut observed_file_names = BTreeSet::new();
    let mut chunk_file_paths = Vec::new();
    for chunk_commit in manifest_commits {
        if chunk_commit.output_format != expected_output_format {
            return Err(OutputError::InvalidInput(format!(
                "Run manifest chunk {} has output_format={}, expected {expected_output_format}.",
                chunk_commit.chunk_identifier, chunk_commit.output_format
            )));
        }
        if observed_file_names.insert(chunk_commit.chunk_file_name.clone()) {
            let chunk_file_path = chunks_directory.join(&chunk_commit.chunk_file_name);
            if !chunk_file_path.exists() {
                return Err(OutputError::InvalidInput(format!(
                    "Run manifest references missing chunk file: {}",
                    chunk_file_path.display()
                )));
            }
            chunk_file_paths.push(chunk_file_path);
        }
    }
    Ok(chunk_file_paths)
}

fn reject_unmanifested_output_chunk_files(
    chunks_directory: &Path,
    output_format: OutputFileFormat,
    manifest_file_names: &BTreeSet<String>,
) -> Result<(), OutputError> {
    for chunk_file_path in sorted_output_chunk_file_paths(chunks_directory, output_format)? {
        let Some(file_name) = chunk_file_path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !manifest_file_names.contains(file_name) {
            return Err(OutputError::InvalidInput(format!(
                "Output chunk file is not recorded in run manifest: {}",
                chunk_file_path.display()
            )));
        }
    }
    Ok(())
}

pub(crate) fn output_format_name(output_format: OutputFileFormat) -> &'static str {
    match output_format {
        OutputFileFormat::Arrow => "arrow",
        OutputFileFormat::Parquet => "parquet",
        OutputFileFormat::Regenie => "regenie",
    }
}

pub(crate) fn sorted_output_chunk_file_paths(
    chunks_directory: &Path,
    output_format: OutputFileFormat,
) -> Result<Vec<PathBuf>, OutputError> {
    let mut chunk_file_paths = std::fs::read_dir(chunks_directory)
        .map_err(OutputError::runtime)?
        .filter_map(|directory_entry| directory_entry.ok().map(|entry| entry.path()))
        .filter(|chunk_file_path| is_output_chunk_file_path(chunk_file_path, output_format))
        .collect::<Vec<_>>();
    chunk_file_paths.sort();
    Ok(chunk_file_paths)
}

fn is_output_chunk_file_path(chunk_file_path: &Path, output_format: OutputFileFormat) -> bool {
    let Some(file_name) = chunk_file_path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let extension_matches = chunk_file_path.extension().and_then(|extension| extension.to_str()).is_some_and(
        |extension| match output_format {
            OutputFileFormat::Arrow => extension.eq_ignore_ascii_case("arrow"),
            OutputFileFormat::Parquet => extension.eq_ignore_ascii_case("parquet"),
            OutputFileFormat::Regenie => extension.eq_ignore_ascii_case("regenie"),
        },
    );
    match output_format {
        OutputFileFormat::Arrow => file_name.starts_with("chunk_") && extension_matches,
        OutputFileFormat::Parquet | OutputFileFormat::Regenie => file_name.starts_with("part_") && extension_matches,
    }
}

fn read_output_chunk_file_batches(
    chunk_file_path: &Path,
    output_format: OutputFileFormat,
    arrow_file_open_seconds: &mut f64,
    arrow_reader_init_seconds: &mut f64,
    read_arrow_seconds: &mut f64,
) -> Result<Box<dyn Iterator<Item = Result<RecordBatch, arrow::error::ArrowError>>>, OutputError> {
    let arrow_file_open_start_time = Instant::now();
    let input_file = File::open(chunk_file_path).map_err(OutputError::runtime)?;
    let current_arrow_file_open_seconds = arrow_file_open_start_time.elapsed().as_secs_f64();
    *arrow_file_open_seconds += current_arrow_file_open_seconds;
    *read_arrow_seconds += current_arrow_file_open_seconds;

    let arrow_reader_init_start_time = Instant::now();
    let batch_reader: Box<dyn Iterator<Item = Result<RecordBatch, arrow::error::ArrowError>>> = match output_format {
        OutputFileFormat::Arrow => Box::new(ArrowFileReader::try_new(input_file, None).map_err(OutputError::runtime)?),
        OutputFileFormat::Parquet => {
            let parquet_reader = ParquetRecordBatchReaderBuilder::try_new(input_file)
                .map_err(OutputError::runtime)?
                .build()
                .map_err(OutputError::runtime)?;
            Box::new(parquet_reader)
        }
        OutputFileFormat::Regenie => {
            return Err(OutputError::InvalidInput(
                "REGENIE text chunks cannot be read by the Parquet finalizer.".to_string(),
            ));
        }
    };
    let current_arrow_reader_init_seconds = arrow_reader_init_start_time.elapsed().as_secs_f64();
    *arrow_reader_init_seconds += current_arrow_reader_init_seconds;
    *read_arrow_seconds += current_arrow_reader_init_seconds;
    Ok(batch_reader)
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
        .set_column_dictionary_enabled(ColumnPath::from("CORRECTION_METHOD"), true)
        .set_column_dictionary_enabled(ColumnPath::from("CORRECTION_STATUS"), true)
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
        ("g.output.schema_version", schema::OUTPUT_SCHEMA_VERSION.to_string()),
        ("g.output.association_mode", association_mode.to_string()),
        ("g.output.chunk_file_count", chunk_file_count.to_string()),
        ("g.output.row_count", row_count.to_string()),
        ("g.output.writer", "rust".to_string()),
    ];
    for (key, value) in metadata_values {
        parquet_writer.append_key_value_metadata(KeyValue { key: key.to_string(), value: Some(value) });
    }
}

pub(crate) fn prepare_chunk_batch_for_final_writer(batch: RecordBatch) -> Result<RecordBatch, OutputError> {
    let output_statistic_dtype = schema::output_statistic_dtype_from_schema(batch.schema().as_ref())?;
    let final_schema = schema::get_regenie_step2_final_schema(output_statistic_dtype);
    if batch.schema().fields() == final_schema.fields() {
        return RecordBatch::try_new(Arc::clone(final_schema), batch.columns().to_vec()).map_err(OutputError::runtime);
    }
    project_chunk_batch_to_final_batch(batch, output_statistic_dtype)
}

pub(crate) fn project_chunk_batch_to_final_batch(
    batch: RecordBatch,
    output_statistic_dtype: schema::OutputStatisticDtype,
) -> Result<RecordBatch, OutputError> {
    let final_column_names = [
        "CHROM",
        "GENPOS",
        "ID",
        "ALLELE0",
        "ALLELE1",
        "A1FREQ",
        "INFO",
        "N",
        "TEST",
        "BETA",
        "SE",
        "CHISQ",
        "LOG10P",
        "EXTRA",
        "CORRECTION_METHOD",
        "CORRECTION_STATUS",
    ];
    let projected_columns = final_column_names
        .iter()
        .map(|column_name| batch.column_by_name(column_name).cloned())
        .collect::<Option<Vec<ArrayRef>>>()
        .ok_or_else(|| {
            OutputError::Runtime("Rust output writer could not project chunk batch to final schema.".to_string())
        })?;
    RecordBatch::try_new(Arc::clone(schema::get_regenie_step2_final_schema(output_statistic_dtype)), projected_columns)
        .map_err(OutputError::runtime)
}

fn read_output_statistic_dtype_from_manifest(
    run_directory: &Path,
) -> Result<schema::OutputStatisticDtype, OutputError> {
    let Some(manifest_json) = manifest::load_run_manifest_json(run_directory)? else {
        return Ok(schema::OutputStatisticDtype::default());
    };
    let manifest_value = serde_json::from_str::<Value>(&manifest_json).map_err(OutputError::runtime)?;
    let output_statistic_dtype_text = manifest_value
        .pointer("/output_writer/result_statistic_dtype")
        .or_else(|| manifest_value.pointer("/execution_plan/output_writer/result_statistic_dtype"))
        .and_then(Value::as_str)
        .unwrap_or_else(|| schema::OutputStatisticDtype::default().as_str());
    schema::OutputStatisticDtype::parse(output_statistic_dtype_text)
}
