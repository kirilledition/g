#![allow(clippy::missing_errors_doc)]
#![allow(clippy::needless_pass_by_value)]

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use arrow::array::{ArrayRef, RecordBatch};
use arrow::ipc::reader::FileReader as ArrowFileReader;
use arrow_cast::display::array_value_to_string;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;

use crate::output::manifest;
use crate::output::schema;
use crate::output::writer::OutputWriterError;

const REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE: usize = 122_880;

pub fn finalize_output_run_chunks(
    run_directory: &Path,
    chunks_directory: &Path,
    association_mode: &str,
) -> Result<PathBuf, OutputWriterError> {
    let final_parquet_path = run_directory.join("final.parquet");
    write_final_parquet_from_chunk_files(chunks_directory, &final_parquet_path, association_mode)?;
    Ok(final_parquet_path)
}

pub fn finalize_output_run_chunks_to_regenie_text(
    chunks_directory: &Path,
    regenie_text_path: &Path,
) -> Result<(), OutputWriterError> {
    let mut chunk_file_paths = sorted_arrow_chunk_file_paths(chunks_directory)?;
    let mut output_file = File::create(regenie_text_path).map_err(OutputWriterError::runtime)?;
    let mut wrote_header = false;
    for chunk_file_path in chunk_file_paths.drain(..) {
        let input_file = File::open(&chunk_file_path).map_err(OutputWriterError::runtime)?;
        let file_reader = ArrowFileReader::try_new(input_file, None).map_err(OutputWriterError::runtime)?;
        for maybe_batch in file_reader {
            let batch = maybe_batch.map_err(OutputWriterError::runtime)?;
            let projected_batch = project_chunk_batch_to_final_batch(batch)?;
            if !wrote_header {
                write_regenie_text_header(&mut output_file, &projected_batch)?;
                wrote_header = true;
            }
            write_regenie_text_batch(&mut output_file, &projected_batch)?;
        }
    }
    Ok(())
}

pub(crate) fn write_final_parquet_from_chunk_files(
    chunks_directory: &Path,
    final_parquet_path: &Path,
    association_mode: &str,
) -> Result<(), OutputWriterError> {
    if association_mode != "regenie2_linear" && association_mode != "regenie2_binary" {
        return Err(OutputWriterError::InvalidInput(format!(
            "Unsupported association mode for Rust output writer finalization: {association_mode}",
        )));
    }
    let chunk_file_paths = sorted_arrow_chunk_file_paths(chunks_directory)?;
    let writer_properties = get_regenie_step2_parquet_writer_properties().clone();
    let output_file = File::create(final_parquet_path).map_err(OutputWriterError::runtime)?;
    let final_schema = Arc::clone(schema::get_regenie_step2_final_schema());
    let mut parquet_writer =
        ArrowWriter::try_new(output_file, final_schema, Some(writer_properties)).map_err(OutputWriterError::runtime)?;
    let chunk_file_count = chunk_file_paths.len();
    let mut output_row_count = 0usize;
    for chunk_file_path in chunk_file_paths {
        let input_file = File::open(&chunk_file_path).map_err(OutputWriterError::runtime)?;
        let file_reader = ArrowFileReader::try_new(input_file, None).map_err(OutputWriterError::runtime)?;
        for maybe_batch in file_reader {
            let batch = maybe_batch.map_err(OutputWriterError::runtime)?;
            let projected_batch = project_chunk_batch_to_final_batch(batch)?;
            output_row_count += projected_batch.num_rows();
            parquet_writer.write(&projected_batch).map_err(OutputWriterError::runtime)?;
        }
    }
    append_output_footer_metadata(&mut parquet_writer, association_mode, chunk_file_count, output_row_count);
    parquet_writer.close().map_err(OutputWriterError::runtime)?;
    manifest::mark_run_manifest_finalized(final_parquet_path, output_row_count, chunk_file_count)
        .map_err(OutputWriterError::runtime)?;
    Ok(())
}

fn sorted_arrow_chunk_file_paths(chunks_directory: &Path) -> Result<Vec<PathBuf>, OutputWriterError> {
    let mut chunk_file_paths = std::fs::read_dir(chunks_directory)
        .map_err(OutputWriterError::runtime)?
        .filter_map(|directory_entry| directory_entry.ok().map(|entry| entry.path()))
        .filter(|chunk_file_path| chunk_file_path.extension().is_some_and(|extension| extension == "arrow"))
        .collect::<Vec<_>>();
    chunk_file_paths.sort();
    Ok(chunk_file_paths)
}

fn write_regenie_text_header(output_file: &mut File, batch: &RecordBatch) -> Result<(), OutputWriterError> {
    let header = batch.schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>().join("\t");
    writeln!(output_file, "{header}").map_err(OutputWriterError::runtime)
}

fn write_regenie_text_batch(output_file: &mut File, batch: &RecordBatch) -> Result<(), OutputWriterError> {
    for row_index in 0..batch.num_rows() {
        for column_index in 0..batch.num_columns() {
            if column_index > 0 {
                write!(output_file, "\t").map_err(OutputWriterError::runtime)?;
            }
            let column = batch.column(column_index);
            let value = array_value_to_string(column.as_ref(), row_index).map_err(OutputWriterError::runtime)?;
            write!(output_file, "{value}").map_err(OutputWriterError::runtime)?;
        }
        writeln!(output_file).map_err(OutputWriterError::runtime)?;
    }
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
