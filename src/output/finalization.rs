#![allow(clippy::missing_errors_doc)]
#![allow(clippy::needless_pass_by_value)]

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

use crate::output::manifest;
use crate::output::schema;
use crate::output::writer::{OutputFileFormat, OutputWriterError};

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

pub fn finalize_output_run_chunks(
    run_directory: &Path,
    chunks_directory: &Path,
    association_mode: &str,
    output_format: OutputFileFormat,
) -> Result<PathBuf, OutputWriterError> {
    let final_parquet_path = run_directory.join("final.parquet");
    write_final_parquet_from_chunk_files(chunks_directory, &final_parquet_path, association_mode, output_format)?;
    Ok(final_parquet_path)
}

pub(crate) fn write_final_parquet_from_chunk_files(
    chunks_directory: &Path,
    final_parquet_path: &Path,
    association_mode: &str,
    output_format: OutputFileFormat,
) -> Result<(), OutputWriterError> {
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
) -> Result<RegenieStep2FinalizationTiming, OutputWriterError> {
    let total_start_time = Instant::now();
    if association_mode != "regenie2_linear" && association_mode != "regenie2_binary" {
        return Err(OutputWriterError::InvalidInput(format!(
            "Unsupported association mode for Rust output writer finalization: {association_mode}",
        )));
    }

    let list_chunk_files_start_time = Instant::now();
    let chunk_file_paths = sorted_output_chunk_file_paths(chunks_directory, output_format)?;
    let list_chunk_files_seconds = list_chunk_files_start_time.elapsed().as_secs_f64();

    let parquet_writer_properties_start_time = Instant::now();
    let writer_properties = get_regenie_step2_parquet_writer_properties().clone();
    let parquet_writer_properties_seconds = parquet_writer_properties_start_time.elapsed().as_secs_f64();

    let parquet_file_create_start_time = Instant::now();
    let output_file = File::create(final_parquet_path).map_err(OutputWriterError::runtime)?;
    let parquet_file_create_seconds = parquet_file_create_start_time.elapsed().as_secs_f64();

    let final_schema = Arc::clone(schema::get_regenie_step2_final_schema());
    let parquet_writer_init_start_time = Instant::now();
    let mut parquet_writer =
        ArrowWriter::try_new(output_file, final_schema, Some(writer_properties)).map_err(OutputWriterError::runtime)?;
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
        arrow_file_bytes = arrow_file_bytes
            .saturating_add(std::fs::metadata(&chunk_file_path).map_err(OutputWriterError::runtime)?.len());
        for maybe_batch in read_output_chunk_file_batches(
            &chunk_file_path,
            output_format,
            &mut arrow_file_open_seconds,
            &mut arrow_reader_init_seconds,
            &mut read_arrow_seconds,
        )? {
            let read_batch_start_time = Instant::now();
            let batch = maybe_batch.map_err(OutputWriterError::runtime)?;
            let current_arrow_batch_read_seconds = read_batch_start_time.elapsed().as_secs_f64();
            arrow_batch_read_seconds += current_arrow_batch_read_seconds;
            read_arrow_seconds += current_arrow_batch_read_seconds;

            let project_batch_start_time = Instant::now();
            let projected_batch = project_chunk_batch_to_final_batch(batch)?;
            project_batch_seconds += project_batch_start_time.elapsed().as_secs_f64();
            output_row_count += projected_batch.num_rows();

            let write_parquet_start_time = Instant::now();
            parquet_writer.write(&projected_batch).map_err(OutputWriterError::runtime)?;
            write_parquet_seconds += write_parquet_start_time.elapsed().as_secs_f64();
            batch_count += 1;
        }
    }

    let footer_metadata_start_time = Instant::now();
    append_output_footer_metadata(&mut parquet_writer, association_mode, chunk_file_count, output_row_count);
    let footer_metadata_seconds = footer_metadata_start_time.elapsed().as_secs_f64();

    let close_writer_start_time = Instant::now();
    parquet_writer.close().map_err(OutputWriterError::runtime)?;
    let close_writer_seconds = close_writer_start_time.elapsed().as_secs_f64();
    let parquet_file_bytes = std::fs::metadata(final_parquet_path).map_err(OutputWriterError::runtime)?.len();

    let manifest_update_start_time = Instant::now();
    manifest::mark_run_manifest_finalized(final_parquet_path, output_row_count, chunk_file_count)
        .map_err(OutputWriterError::runtime)?;
    let manifest_update_seconds = manifest_update_start_time.elapsed().as_secs_f64();

    Ok(RegenieStep2FinalizationTiming {
        chunk_file_count: u64::try_from(chunk_file_count).map_err(OutputWriterError::runtime)?,
        batch_count,
        row_count: u64::try_from(output_row_count).map_err(OutputWriterError::runtime)?,
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

fn sorted_output_chunk_file_paths(
    chunks_directory: &Path,
    output_format: OutputFileFormat,
) -> Result<Vec<PathBuf>, OutputWriterError> {
    let mut chunk_file_paths = std::fs::read_dir(chunks_directory)
        .map_err(OutputWriterError::runtime)?
        .filter_map(|directory_entry| directory_entry.ok().map(|entry| entry.path()))
        .filter(|chunk_file_path| {
            chunk_file_path.extension().is_some_and(|extension| match output_format {
                OutputFileFormat::Arrow => extension == "arrow",
                OutputFileFormat::Parquet => extension == "parquet",
            })
        })
        .collect::<Vec<_>>();
    chunk_file_paths.sort();
    Ok(chunk_file_paths)
}

fn read_output_chunk_file_batches(
    chunk_file_path: &Path,
    output_format: OutputFileFormat,
    arrow_file_open_seconds: &mut f64,
    arrow_reader_init_seconds: &mut f64,
    read_arrow_seconds: &mut f64,
) -> Result<Box<dyn Iterator<Item = Result<RecordBatch, arrow::error::ArrowError>>>, OutputWriterError> {
    let arrow_file_open_start_time = Instant::now();
    let input_file = File::open(chunk_file_path).map_err(OutputWriterError::runtime)?;
    let current_arrow_file_open_seconds = arrow_file_open_start_time.elapsed().as_secs_f64();
    *arrow_file_open_seconds += current_arrow_file_open_seconds;
    *read_arrow_seconds += current_arrow_file_open_seconds;

    let arrow_reader_init_start_time = Instant::now();
    let batch_reader: Box<dyn Iterator<Item = Result<RecordBatch, arrow::error::ArrowError>>> = match output_format {
        OutputFileFormat::Arrow => {
            Box::new(ArrowFileReader::try_new(input_file, None).map_err(OutputWriterError::runtime)?)
        }
        OutputFileFormat::Parquet => {
            let parquet_reader = ParquetRecordBatchReaderBuilder::try_new(input_file)
                .map_err(OutputWriterError::runtime)?
                .build()
                .map_err(OutputWriterError::runtime)?;
            Box::new(parquet_reader)
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

fn project_chunk_batch_to_final_batch(batch: RecordBatch) -> Result<RecordBatch, OutputWriterError> {
    let final_column_names = [
        "CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1", "A1FREQ", "INFO", "N", "TEST", "BETA", "SE", "CHISQ", "LOG10P",
        "EXTRA",
    ];
    let projected_columns = final_column_names
        .iter()
        .map(|column_name| batch.column_by_name(column_name).cloned())
        .collect::<Option<Vec<ArrayRef>>>()
        .ok_or_else(|| {
            OutputWriterError::Runtime("Rust output writer could not project chunk batch to final schema.".to_string())
        })?;
    RecordBatch::try_new(Arc::clone(schema::get_regenie_step2_final_schema()), projected_columns)
        .map_err(OutputWriterError::runtime)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};

    use crate::output::writer::OutputFileFormat;

    use super::*;

    fn create_test_directory() -> PathBuf {
        let unique_suffix =
            SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after Unix epoch").as_nanos();
        let directory_path = std::env::temp_dir().join(format!("g-output-finalization-test-{unique_suffix}"));
        std::fs::create_dir_all(&directory_path).expect("test directory should be created");
        directory_path
    }

    #[test]
    fn finalization_rejects_unsupported_association_mode_before_reading_chunks() {
        let chunks_directory = create_test_directory();
        let final_parquet_path = chunks_directory.join("final.parquet");

        let error = write_final_parquet_from_chunk_files_with_timing(
            &chunks_directory,
            &final_parquet_path,
            "unsupported",
            OutputFileFormat::Arrow,
        )
        .err()
        .expect("unsupported association mode should fail")
        .to_string();
        assert!(error.contains("Unsupported association mode"));

        std::fs::remove_dir_all(chunks_directory).expect("test directory should be removed");
    }

    #[test]
    fn finalization_projection_reports_missing_final_columns() {
        let schema = Arc::new(Schema::new(vec![Field::new("CHROM", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vec!["22"]))])
            .expect("record batch should build");

        let error = project_chunk_batch_to_final_batch(batch)
            .expect_err("missing final columns should fail projection")
            .to_string();
        assert!(error.contains("project chunk batch"));
    }
}
