#![allow(clippy::missing_errors_doc)]
#![allow(clippy::needless_pass_by_value)]

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
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
use crate::output::writer::{self, OutputFileFormat, OutputWriterError};

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
    match output_format {
        OutputFileFormat::Arrow | OutputFileFormat::Parquet => {
            let final_parquet_path = run_directory.join("final.parquet");
            write_final_parquet_from_chunk_files(
                chunks_directory,
                &final_parquet_path,
                association_mode,
                output_format,
            )?;
            Ok(final_parquet_path)
        }
        OutputFileFormat::Regenie => {
            let final_regenie_path = run_directory.join("final.regenie");
            write_final_regenie_from_chunk_files(
                chunks_directory,
                &final_regenie_path,
                association_mode,
                output_format,
            )?;
            Ok(final_regenie_path)
        }
    }
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
    let run_directory = final_parquet_path.parent().ok_or_else(|| {
        OutputWriterError::InvalidInput("Final Parquet path must have a parent run directory.".to_string())
    })?;
    let manifest_commits =
        manifest::read_run_manifest_chunk_commits(run_directory).map_err(OutputWriterError::runtime)?;
    let chunk_file_paths = manifest_output_chunk_file_paths(chunks_directory, output_format, &manifest_commits)?;
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
            let projected_batch = prepare_chunk_batch_for_final_writer(batch)?;
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

pub(crate) fn write_final_regenie_from_chunk_files(
    chunks_directory: &Path,
    final_regenie_path: &Path,
    association_mode: &str,
    output_format: OutputFileFormat,
) -> Result<(), OutputWriterError> {
    write_final_regenie_from_chunk_files_with_timing(
        chunks_directory,
        final_regenie_path,
        association_mode,
        output_format,
    )
    .map(|_| ())
}

pub(crate) fn write_final_regenie_from_chunk_files_with_timing(
    chunks_directory: &Path,
    final_regenie_path: &Path,
    association_mode: &str,
    output_format: OutputFileFormat,
) -> Result<RegenieStep2FinalizationTiming, OutputWriterError> {
    let total_start_time = Instant::now();
    if association_mode != "regenie2_linear" && association_mode != "regenie2_binary" {
        return Err(OutputWriterError::InvalidInput(format!(
            "Unsupported association mode for Rust output writer finalization: {association_mode}",
        )));
    }
    if output_format != OutputFileFormat::Regenie {
        return Err(OutputWriterError::InvalidInput(
            "REGENIE text finalization requires output_format=regenie.".to_string(),
        ));
    }

    let list_chunk_files_start_time = Instant::now();
    let run_directory = final_regenie_path.parent().ok_or_else(|| {
        OutputWriterError::InvalidInput("Final REGENIE text path must have a parent run directory.".to_string())
    })?;
    let manifest_commits =
        manifest::read_run_manifest_chunk_commits(run_directory).map_err(OutputWriterError::runtime)?;
    let chunk_file_paths = manifest_output_chunk_file_paths(chunks_directory, output_format, &manifest_commits)?;
    let list_chunk_files_seconds = list_chunk_files_start_time.elapsed().as_secs_f64();

    let parquet_file_create_start_time = Instant::now();
    let temporary_final_path = final_regenie_path.with_extension("regenie.tmp");
    let output_file = File::create(&temporary_final_path).map_err(OutputWriterError::runtime)?;
    let parquet_file_create_seconds = parquet_file_create_start_time.elapsed().as_secs_f64();

    let parquet_writer_init_start_time = Instant::now();
    let mut output_writer = BufWriter::new(output_file);
    output_writer.write_all(writer::REGENIE_STEP2_TEXT_HEADER.as_bytes()).map_err(OutputWriterError::runtime)?;
    let parquet_writer_init_seconds = parquet_writer_init_start_time.elapsed().as_secs_f64();

    let chunk_file_count = chunk_file_paths.len();
    let mut output_row_count = 0usize;
    let mut batch_count = 0u64;
    let mut arrow_file_open_seconds = 0.0;
    let mut arrow_batch_read_seconds = 0.0;
    let mut read_arrow_seconds = 0.0;
    let mut write_parquet_seconds = 0.0;
    let mut arrow_file_bytes = 0u64;
    for chunk_file_path in chunk_file_paths {
        arrow_file_bytes = arrow_file_bytes
            .saturating_add(std::fs::metadata(&chunk_file_path).map_err(OutputWriterError::runtime)?.len());
        let append_result = append_regenie_text_part_rows(
            &chunk_file_path,
            &mut output_writer,
            &mut arrow_file_open_seconds,
            &mut arrow_batch_read_seconds,
            &mut read_arrow_seconds,
            &mut write_parquet_seconds,
        )?;
        output_row_count += append_result;
        batch_count += 1;
    }

    let close_writer_start_time = Instant::now();
    output_writer.flush().map_err(OutputWriterError::runtime)?;
    drop(output_writer);
    let close_writer_seconds = close_writer_start_time.elapsed().as_secs_f64();
    std::fs::rename(&temporary_final_path, final_regenie_path).map_err(OutputWriterError::runtime)?;
    let parquet_file_bytes = std::fs::metadata(final_regenie_path).map_err(OutputWriterError::runtime)?.len();

    let manifest_update_start_time = Instant::now();
    manifest::mark_run_manifest_finalized_output(
        final_regenie_path,
        output_row_count,
        chunk_file_count,
        output_format_name(output_format),
    )
    .map_err(OutputWriterError::runtime)?;
    let manifest_update_seconds = manifest_update_start_time.elapsed().as_secs_f64();

    Ok(RegenieStep2FinalizationTiming {
        chunk_file_count: u64::try_from(chunk_file_count).map_err(OutputWriterError::runtime)?,
        batch_count,
        row_count: u64::try_from(output_row_count).map_err(OutputWriterError::runtime)?,
        list_chunk_files_seconds,
        parquet_writer_properties_seconds: 0.0,
        parquet_file_create_seconds,
        parquet_writer_init_seconds,
        arrow_file_open_seconds,
        arrow_reader_init_seconds: 0.0,
        arrow_batch_read_seconds,
        read_arrow_seconds,
        project_batch_seconds: 0.0,
        write_parquet_seconds,
        footer_metadata_seconds: 0.0,
        close_writer_seconds,
        manifest_update_seconds,
        arrow_file_bytes,
        parquet_file_bytes,
        total_seconds: total_start_time.elapsed().as_secs_f64(),
    })
}

fn append_regenie_text_part_rows(
    chunk_file_path: &Path,
    output_writer: &mut BufWriter<File>,
    arrow_file_open_seconds: &mut f64,
    arrow_batch_read_seconds: &mut f64,
    read_arrow_seconds: &mut f64,
    write_parquet_seconds: &mut f64,
) -> Result<usize, OutputWriterError> {
    let arrow_file_open_start_time = Instant::now();
    let input_file = File::open(chunk_file_path).map_err(OutputWriterError::runtime)?;
    let current_arrow_file_open_seconds = arrow_file_open_start_time.elapsed().as_secs_f64();
    *arrow_file_open_seconds += current_arrow_file_open_seconds;
    *read_arrow_seconds += current_arrow_file_open_seconds;

    let mut input_reader = BufReader::new(input_file);
    let mut header_line = String::new();
    let header_read_start_time = Instant::now();
    input_reader.read_line(&mut header_line).map_err(OutputWriterError::runtime)?;
    let header_read_seconds = header_read_start_time.elapsed().as_secs_f64();
    *arrow_batch_read_seconds += header_read_seconds;
    *read_arrow_seconds += header_read_seconds;
    validate_regenie_text_header(&header_line, chunk_file_path)?;

    let mut row_count = 0usize;
    let mut row_line = String::new();
    loop {
        row_line.clear();
        let row_read_start_time = Instant::now();
        let read_byte_count = input_reader.read_line(&mut row_line).map_err(OutputWriterError::runtime)?;
        let row_read_seconds = row_read_start_time.elapsed().as_secs_f64();
        *arrow_batch_read_seconds += row_read_seconds;
        *read_arrow_seconds += row_read_seconds;
        if read_byte_count == 0 {
            break;
        }
        validate_regenie_text_row(&row_line, chunk_file_path)?;
        let write_start_time = Instant::now();
        output_writer.write_all(row_line.as_bytes()).map_err(OutputWriterError::runtime)?;
        if !row_line.ends_with('\n') {
            output_writer.write_all(b"\n").map_err(OutputWriterError::runtime)?;
        }
        *write_parquet_seconds += write_start_time.elapsed().as_secs_f64();
        row_count += 1;
    }
    Ok(row_count)
}

fn validate_regenie_text_header(header_line: &str, chunk_file_path: &Path) -> Result<(), OutputWriterError> {
    let observed_header = header_line.trim_end_matches(['\r', '\n']);
    let expected_header = writer::REGENIE_STEP2_TEXT_HEADER.trim_end_matches('\n');
    if observed_header == expected_header {
        return Ok(());
    }
    Err(OutputWriterError::InvalidInput(format!(
        "REGENIE text part has an unexpected header: {}",
        chunk_file_path.display()
    )))
}

fn validate_regenie_text_row(row_line: &str, chunk_file_path: &Path) -> Result<(), OutputWriterError> {
    let row = row_line.trim_end_matches(['\r', '\n']);
    let expected_column_count = writer::REGENIE_STEP2_TEXT_HEADER.trim_end_matches('\n').split('\t').count();
    if row.split('\t').count() == expected_column_count {
        return Ok(());
    }
    Err(OutputWriterError::InvalidInput(format!(
        "REGENIE text part has a row with an unexpected column count: {}",
        chunk_file_path.display()
    )))
}

fn manifest_output_chunk_file_paths(
    chunks_directory: &Path,
    output_format: OutputFileFormat,
    manifest_commits: &[manifest::RunManifestChunkCommit],
) -> Result<Vec<PathBuf>, OutputWriterError> {
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
            return Err(OutputWriterError::InvalidInput(format!(
                "Run manifest chunk {} has output_format={}, expected {expected_output_format}.",
                chunk_commit.chunk_identifier, chunk_commit.output_format
            )));
        }
        if observed_file_names.insert(chunk_commit.chunk_file_name.clone()) {
            let chunk_file_path = chunks_directory.join(&chunk_commit.chunk_file_name);
            if !chunk_file_path.exists() {
                return Err(OutputWriterError::InvalidInput(format!(
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
) -> Result<(), OutputWriterError> {
    for chunk_file_path in sorted_output_chunk_file_paths(chunks_directory, output_format)? {
        let Some(file_name) = chunk_file_path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !manifest_file_names.contains(file_name) {
            return Err(OutputWriterError::InvalidInput(format!(
                "Output chunk file is not recorded in run manifest: {}",
                chunk_file_path.display()
            )));
        }
    }
    Ok(())
}

fn output_format_name(output_format: OutputFileFormat) -> &'static str {
    match output_format {
        OutputFileFormat::Arrow => "arrow",
        OutputFileFormat::Parquet => "parquet",
        OutputFileFormat::Regenie => "regenie",
    }
}

fn sorted_output_chunk_file_paths(
    chunks_directory: &Path,
    output_format: OutputFileFormat,
) -> Result<Vec<PathBuf>, OutputWriterError> {
    let mut chunk_file_paths = std::fs::read_dir(chunks_directory)
        .map_err(OutputWriterError::runtime)?
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
        OutputFileFormat::Regenie => {
            return Err(OutputWriterError::InvalidInput(
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

fn prepare_chunk_batch_for_final_writer(batch: RecordBatch) -> Result<RecordBatch, OutputWriterError> {
    let final_schema = schema::get_regenie_step2_final_schema();
    if batch.schema().fields() == final_schema.fields() {
        return RecordBatch::try_new(Arc::clone(final_schema), batch.columns().to_vec())
            .map_err(OutputWriterError::runtime);
    }
    project_chunk_batch_to_final_batch(batch)
}

fn project_chunk_batch_to_final_batch(batch: RecordBatch) -> Result<RecordBatch, OutputWriterError> {
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
            OutputWriterError::Runtime("Rust output writer could not project chunk batch to final schema.".to_string())
        })?;
    RecordBatch::try_new(Arc::clone(schema::get_regenie_step2_final_schema()), projected_columns)
        .map_err(OutputWriterError::runtime)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use arrow::array::{ArrayRef, Float32Array, Int32Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};

    use crate::output::writer::{self as output_writer, OutputFileFormat};

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

    #[test]
    fn sorted_chunk_files_ignore_stale_final_parquet_outputs() {
        let chunks_directory = create_test_directory();
        let part_file_path = chunks_directory.join("part_000000000.parquet");
        let final_file_path = chunks_directory.join("final.parquet");
        let temporary_part_file_path = chunks_directory.join("part_000000001.parquet.tmp");
        std::fs::write(&part_file_path, b"part").expect("part marker should be written");
        std::fs::write(final_file_path, b"final").expect("final marker should be written");
        std::fs::write(temporary_part_file_path, b"temporary").expect("temporary marker should be written");

        let chunk_file_paths = sorted_output_chunk_file_paths(&chunks_directory, OutputFileFormat::Parquet)
            .expect("chunk files should be listed");

        assert_eq!(chunk_file_paths, vec![part_file_path]);
        std::fs::remove_dir_all(chunks_directory).expect("test directory should be removed");
    }

    #[test]
    fn manifest_chunk_paths_reject_unmanifested_matching_files() {
        let chunks_directory = create_test_directory();
        let part_file_path = chunks_directory.join("part_000000000.parquet");
        let stale_part_file_path = chunks_directory.join("part_000000001.parquet");
        std::fs::write(&part_file_path, b"part").expect("part marker should be written");
        std::fs::write(&stale_part_file_path, b"stale").expect("stale marker should be written");
        let manifest_commits = vec![manifest::RunManifestChunkCommit {
            chunk_identifier: 0,
            output_format: "parquet".to_string(),
            compression: "none".to_string(),
            variant_start_index: 0,
            variant_stop_index: 2,
            row_count: 2,
            chunk_file_name: "part_000000000.parquet".to_string(),
        }];

        let error = manifest_output_chunk_file_paths(&chunks_directory, OutputFileFormat::Parquet, &manifest_commits)
            .expect_err("unmanifested chunk file should fail")
            .to_string();

        assert!(error.contains("not recorded in run manifest"));
        std::fs::remove_dir_all(chunks_directory).expect("test directory should be removed");
    }

    #[test]
    fn finalization_prepares_ordered_final_schema_without_column_projection() {
        let chromosome_array: ArrayRef = Arc::new(StringArray::from(vec!["22"]));
        let position_array: ArrayRef = Arc::new(Int64Array::from(vec![100_i64]));
        let identifier_array: ArrayRef = Arc::new(StringArray::from(vec!["variant0"]));
        let allele_zero_array: ArrayRef = Arc::new(StringArray::from(vec!["G"]));
        let allele_one_array: ArrayRef = Arc::new(StringArray::from(vec!["A"]));
        let allele_frequency_array: ArrayRef = Arc::new(Float32Array::from(vec![0.5_f32]));
        let info_array: ArrayRef = Arc::new(Float32Array::from(vec![Some(0.9_f32)]));
        let observation_count_array: ArrayRef = Arc::new(Int32Array::from(vec![100_i32]));
        let test_array: ArrayRef = Arc::new(StringArray::from(vec!["ADD"]));
        let beta_array: ArrayRef = Arc::new(Float32Array::from(vec![0.1_f32]));
        let standard_error_array: ArrayRef = Arc::new(Float32Array::from(vec![0.01_f32]));
        let chi_squared_array: ArrayRef = Arc::new(Float32Array::from(vec![10.0_f32]));
        let log10_p_value_array: ArrayRef = Arc::new(Float32Array::from(vec![5.0_f32]));
        let extra_array: ArrayRef = Arc::new(StringArray::from(vec![None::<&str>]));
        let correction_method_array: ArrayRef = Arc::new(StringArray::from(vec!["score"]));
        let correction_status_array: ArrayRef = Arc::new(StringArray::from(vec!["success"]));
        let columns = vec![
            chromosome_array,
            position_array,
            identifier_array,
            allele_zero_array,
            allele_one_array,
            allele_frequency_array,
            info_array,
            observation_count_array,
            test_array,
            beta_array,
            standard_error_array,
            chi_squared_array,
            log10_p_value_array,
            extra_array,
            correction_method_array,
            correction_status_array,
        ];
        let batch = RecordBatch::try_new(Arc::clone(schema::get_regenie_step2_final_schema()), columns.clone())
            .expect("ordered final batch should build");

        let prepared_batch =
            prepare_chunk_batch_for_final_writer(batch).expect("ordered final batch should be prepared");

        assert_eq!(prepared_batch.schema().fields(), schema::get_regenie_step2_final_schema().fields());
        assert!(Arc::ptr_eq(prepared_batch.column(0), &columns[0]));
        assert!(Arc::ptr_eq(prepared_batch.column(13), &columns[13]));
        assert!(Arc::ptr_eq(prepared_batch.column(15), &columns[15]));
    }

    #[test]
    fn finalization_concatenates_regenie_text_parts_with_one_header() {
        let run_directory = create_test_directory();
        let regenie_directory = run_directory.join("regenie");
        std::fs::create_dir_all(&regenie_directory).expect("regenie directory should be created");
        let first_part_path = regenie_directory.join("part_000000000.regenie");
        let second_part_path = regenie_directory.join("part_000000002.regenie");
        std::fs::write(
            &first_part_path,
            format!(
                "{}22\t100\tvariant0\tG\tA\t0.5\t0.9\t100\tADD\t0.1\t0.01\t10\t5\tNA\tscore\tsuccess\n",
                output_writer::REGENIE_STEP2_TEXT_HEADER
            ),
        )
        .expect("first REGENIE text part should be written");
        std::fs::write(
            &second_part_path,
            format!(
                "{}22\t102\tvariant2\tG\tA\t0.5\t0.9\t100\tADD\t0.1\t0.01\t10\t5\tTEST_FAIL\tfirth_approximate\tfailed\n",
                output_writer::REGENIE_STEP2_TEXT_HEADER
            ),
        )
        .expect("second REGENIE text part should be written");
        std::fs::write(
            run_directory.join("run_manifest.json"),
            r#"{
              "committed_chunks": [
                {"chunk_identifier":0,"output_format":"regenie","compression":"none","variant_start_index":0,"variant_stop_index":1,"row_count":1,"chunk_file_name":"part_000000000.regenie"},
                {"chunk_identifier":2,"output_format":"regenie","compression":"none","variant_start_index":2,"variant_stop_index":3,"row_count":1,"chunk_file_name":"part_000000002.regenie"}
              ]
            }"#,
        )
        .expect("manifest should be written");

        let final_regenie_path = run_directory.join("final.regenie");
        write_final_regenie_from_chunk_files(
            &regenie_directory,
            &final_regenie_path,
            "regenie2_binary",
            OutputFileFormat::Regenie,
        )
        .expect("final REGENIE text should write");

        let final_lines = std::fs::read_to_string(&final_regenie_path)
            .expect("final REGENIE text should be readable")
            .lines()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert_eq!(final_lines.len(), 3);
        assert_eq!(
            final_lines[0],
            "CHROM\tGENPOS\tID\tALLELE0\tALLELE1\tA1FREQ\tINFO\tN\tTEST\tBETA\tSE\tCHISQ\tLOG10P\tEXTRA\tCORRECTION_METHOD\tCORRECTION_STATUS"
        );
        assert_eq!(
            final_lines[2],
            "22\t102\tvariant2\tG\tA\t0.5\t0.9\t100\tADD\t0.1\t0.01\t10\t5\tTEST_FAIL\tfirth_approximate\tfailed"
        );
        let manifest_text =
            std::fs::read_to_string(run_directory.join("run_manifest.json")).expect("manifest should be readable");
        let manifest = serde_json::from_str::<serde_json::Value>(&manifest_text).expect("manifest should parse");
        assert_eq!(manifest.get("final_output_format").and_then(serde_json::Value::as_str), Some("regenie"));
        assert_eq!(manifest.get("final_row_count").and_then(serde_json::Value::as_i64), Some(2));

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }
}
