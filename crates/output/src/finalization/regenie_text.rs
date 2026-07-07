use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use crate::error::OutputError;
use crate::manifest;
use crate::writer::{self, OutputFileFormat};

use super::{RegenieStep2FinalizationTiming, manifest_output_chunk_file_paths, output_format_name};

pub(crate) fn write_final_regenie_from_chunk_files(
    chunks_directory: &Path,
    final_regenie_path: &Path,
    association_mode: &str,
    output_format: OutputFileFormat,
) -> Result<(), OutputError> {
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
) -> Result<RegenieStep2FinalizationTiming, OutputError> {
    let total_start_time = Instant::now();
    if association_mode != "regenie2_linear" && association_mode != "regenie2_binary" {
        return Err(OutputError::InvalidInput(format!(
            "Unsupported association mode for Rust output writer finalization: {association_mode}",
        )));
    }
    if output_format != OutputFileFormat::Regenie {
        return Err(OutputError::InvalidInput("REGENIE text finalization requires output_format=regenie.".to_string()));
    }

    let list_chunk_files_start_time = Instant::now();
    let run_directory = final_regenie_path.parent().ok_or_else(|| {
        OutputError::InvalidInput("Final REGENIE text path must have a parent run directory.".to_string())
    })?;
    let manifest_commits = manifest::read_run_manifest_chunk_commits(run_directory)?;
    let chunk_file_paths = manifest_output_chunk_file_paths(chunks_directory, output_format, &manifest_commits)?;
    let list_chunk_files_seconds = list_chunk_files_start_time.elapsed().as_secs_f64();

    let parquet_file_create_start_time = Instant::now();
    let temporary_final_path = final_regenie_path.with_extension("regenie.tmp");
    let output_file = File::create(&temporary_final_path).map_err(OutputError::runtime)?;
    let parquet_file_create_seconds = parquet_file_create_start_time.elapsed().as_secs_f64();

    let parquet_writer_init_start_time = Instant::now();
    let mut output_writer = BufWriter::new(output_file);
    output_writer.write_all(writer::REGENIE_STEP2_TEXT_HEADER.as_bytes()).map_err(OutputError::runtime)?;
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
        arrow_file_bytes =
            arrow_file_bytes.saturating_add(std::fs::metadata(&chunk_file_path).map_err(OutputError::runtime)?.len());
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
    output_writer.flush().map_err(OutputError::runtime)?;
    drop(output_writer);
    let close_writer_seconds = close_writer_start_time.elapsed().as_secs_f64();
    std::fs::rename(&temporary_final_path, final_regenie_path).map_err(OutputError::runtime)?;
    let parquet_file_bytes = std::fs::metadata(final_regenie_path).map_err(OutputError::runtime)?.len();

    let manifest_update_start_time = Instant::now();
    manifest::mark_run_manifest_finalized_output(
        final_regenie_path,
        output_row_count,
        chunk_file_count,
        output_format_name(output_format),
    )?;
    let manifest_update_seconds = manifest_update_start_time.elapsed().as_secs_f64();

    Ok(RegenieStep2FinalizationTiming {
        chunk_file_count: u64::try_from(chunk_file_count).map_err(OutputError::runtime)?,
        batch_count,
        row_count: u64::try_from(output_row_count).map_err(OutputError::runtime)?,
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
) -> Result<usize, OutputError> {
    let arrow_file_open_start_time = Instant::now();
    let input_file = File::open(chunk_file_path).map_err(OutputError::runtime)?;
    let current_arrow_file_open_seconds = arrow_file_open_start_time.elapsed().as_secs_f64();
    *arrow_file_open_seconds += current_arrow_file_open_seconds;
    *read_arrow_seconds += current_arrow_file_open_seconds;

    let mut input_reader = BufReader::new(input_file);
    let mut header_line = String::new();
    let header_read_start_time = Instant::now();
    input_reader.read_line(&mut header_line).map_err(OutputError::runtime)?;
    let header_read_seconds = header_read_start_time.elapsed().as_secs_f64();
    *arrow_batch_read_seconds += header_read_seconds;
    *read_arrow_seconds += header_read_seconds;
    validate_regenie_text_header(&header_line, chunk_file_path)?;

    let mut row_count = 0usize;
    let mut row_line = String::new();
    loop {
        row_line.clear();
        let row_read_start_time = Instant::now();
        let read_byte_count = input_reader.read_line(&mut row_line).map_err(OutputError::runtime)?;
        let row_read_seconds = row_read_start_time.elapsed().as_secs_f64();
        *arrow_batch_read_seconds += row_read_seconds;
        *read_arrow_seconds += row_read_seconds;
        if read_byte_count == 0 {
            break;
        }
        validate_regenie_text_row(&row_line, chunk_file_path)?;
        let write_start_time = Instant::now();
        output_writer.write_all(row_line.as_bytes()).map_err(OutputError::runtime)?;
        if !row_line.ends_with('\n') {
            output_writer.write_all(b"\n").map_err(OutputError::runtime)?;
        }
        *write_parquet_seconds += write_start_time.elapsed().as_secs_f64();
        row_count += 1;
    }
    Ok(row_count)
}

fn validate_regenie_text_header(header_line: &str, chunk_file_path: &Path) -> Result<(), OutputError> {
    let observed_header = header_line.trim_end_matches(['\r', '\n']);
    let expected_header = writer::REGENIE_STEP2_TEXT_HEADER.trim_end_matches('\n');
    if observed_header == expected_header {
        return Ok(());
    }
    Err(OutputError::InvalidInput(format!("REGENIE text part has an unexpected header: {}", chunk_file_path.display())))
}

fn validate_regenie_text_row(row_line: &str, chunk_file_path: &Path) -> Result<(), OutputError> {
    let row = row_line.trim_end_matches(['\r', '\n']);
    let expected_column_count = writer::REGENIE_STEP2_TEXT_HEADER.trim_end_matches('\n').split('\t').count();
    if row.split('\t').count() == expected_column_count {
        return Ok(());
    }
    Err(OutputError::InvalidInput(format!(
        "REGENIE text part has a row with an unexpected column count: {}",
        chunk_file_path.display()
    )))
}
