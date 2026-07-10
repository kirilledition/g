use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::Schema;

use crate::error::OutputError;
use crate::manifest;

use super::chunk_manifest::{build_chunk_commit_metadata_text, build_regenie_text_metadata_sidecar_path};
use super::record_batch::{
    RegenieStep2CorrectionArrayEncoding, RegenieStep2RecordBatchArrayCache, build_regenie_step2_record_batch,
};
use super::{
    OutputResult, RegenieStep2ArrowFileWriteTiming, RegenieStep2ChunkJob, RegenieStep2ChunkStreamWriteResult,
    RegenieStep2RecordBatchBuildTiming,
};

pub(crate) const REGENIE_STEP2_TEXT_HEADER: &str = "CHROM\tGENPOS\tID\tALLELE0\tALLELE1\tA1FREQ\tINFO\tN\tTEST\tBETA\tSE\tCHISQ\tLOG10P\tEXTRA\tCORRECTION_METHOD\tCORRECTION_STATUS\n";
const REGENIE_STEP2_TEXT_MISSING_VALUE: &str = "NA";

pub(super) fn write_regenie_step2_chunks_to_regenie_text_file(
    chunks: Vec<RegenieStep2ChunkJob>,
    chunk_schema: &Arc<Schema>,
    chunk_file_path: &Path,
) -> OutputResult<RegenieStep2ChunkStreamWriteResult> {
    let file_create_start_time = Instant::now();
    let output_file = File::create(chunk_file_path).map_err(OutputError::runtime)?;
    let file_create = file_create_start_time.elapsed().as_secs_f64();

    let writer_init_start_time = Instant::now();
    let mut output_writer = BufWriter::new(output_file);
    output_writer.write_all(REGENIE_STEP2_TEXT_HEADER.as_bytes()).map_err(OutputError::runtime)?;
    let writer_init = writer_init_start_time.elapsed().as_secs_f64();

    let mut record_batch_build_timing = RegenieStep2RecordBatchBuildTiming::default();
    let mut record_batch_build_seconds = 0.0;
    let mut array_cache = RegenieStep2RecordBatchArrayCache::default();
    let mut batch_write = 0.0;
    for chunk_job in chunks {
        let record_batch_build_start_time = Instant::now();
        let record_batch_build_result = build_regenie_step2_record_batch(
            chunk_job,
            Arc::clone(chunk_schema),
            &mut array_cache,
            RegenieStep2CorrectionArrayEncoding::String,
        )?;
        record_batch_build_seconds += record_batch_build_start_time.elapsed().as_secs_f64();
        record_batch_build_timing.add(record_batch_build_result.timing);

        let batch_write_start_time = Instant::now();
        write_regenie_step2_text_record_batch(&mut output_writer, &record_batch_build_result.record_batch)?;
        batch_write += batch_write_start_time.elapsed().as_secs_f64();
    }
    let writer_finish_start_time = Instant::now();
    output_writer.flush().map_err(OutputError::runtime)?;
    let writer_finish = writer_finish_start_time.elapsed().as_secs_f64();
    Ok(RegenieStep2ChunkStreamWriteResult {
        record_batch_build_timing,
        record_batch_build_seconds,
        arrow_file_write_timing: RegenieStep2ArrowFileWriteTiming {
            file_create,
            writer_init,
            batch_write,
            writer_finish,
        },
    })
}

pub(super) fn write_regenie_text_metadata_sidecar(
    chunk_file_path: &Path,
    chunk_commits: &[manifest::RunManifestChunkCommit],
) -> OutputResult<()> {
    let sidecar_path = build_regenie_text_metadata_sidecar_path(chunk_file_path);
    let temporary_sidecar_path = sidecar_path.with_extension("json.tmp");
    let metadata_text = build_chunk_commit_metadata_text(chunk_commits)?;
    std::fs::write(&temporary_sidecar_path, format!("{metadata_text}\n")).map_err(OutputError::runtime)?;
    std::fs::rename(&temporary_sidecar_path, &sidecar_path).map_err(OutputError::runtime)
}

fn write_regenie_step2_text_record_batch(
    output_writer: &mut BufWriter<File>,
    record_batch: &RecordBatch,
) -> OutputResult<()> {
    let chromosome_array = required_string_column(record_batch, "CHROM")?;
    let position_array = required_int64_column(record_batch, "GENPOS")?;
    let variant_identifier_array = required_string_column(record_batch, "ID")?;
    let allele_zero_array = required_string_column(record_batch, "ALLELE0")?;
    let allele_one_array = required_string_column(record_batch, "ALLELE1")?;
    let allele_one_frequency_array = required_float32_column(record_batch, "A1FREQ")?;
    let info_score_array = required_float32_column(record_batch, "INFO")?;
    let observation_count_array = required_int32_column(record_batch, "N")?;
    let test_array = required_string_column(record_batch, "TEST")?;
    let beta_array = required_statistic_column(record_batch, "BETA")?;
    let standard_error_array = required_statistic_column(record_batch, "SE")?;
    let chi_squared_array = required_statistic_column(record_batch, "CHISQ")?;
    let log10_p_value_array = required_statistic_column(record_batch, "LOG10P")?;
    let extra_array = required_string_column(record_batch, "EXTRA")?;
    let correction_method_array = required_string_column(record_batch, "CORRECTION_METHOD")?;
    let correction_status_array = required_string_column(record_batch, "CORRECTION_STATUS")?;
    for row_index in 0..record_batch.num_rows() {
        write_regenie_text_string_value(output_writer, chromosome_array, row_index, "CHROM")?;
        output_writer.write_all(b"\t").map_err(OutputError::runtime)?;
        write_regenie_text_int64_value(output_writer, position_array, row_index)?;
        output_writer.write_all(b"\t").map_err(OutputError::runtime)?;
        write_regenie_text_string_value(output_writer, variant_identifier_array, row_index, "ID")?;
        output_writer.write_all(b"\t").map_err(OutputError::runtime)?;
        write_regenie_text_string_value(output_writer, allele_zero_array, row_index, "ALLELE0")?;
        output_writer.write_all(b"\t").map_err(OutputError::runtime)?;
        write_regenie_text_string_value(output_writer, allele_one_array, row_index, "ALLELE1")?;
        output_writer.write_all(b"\t").map_err(OutputError::runtime)?;
        write_regenie_text_float32_value(output_writer, allele_one_frequency_array, row_index)?;
        output_writer.write_all(b"\t").map_err(OutputError::runtime)?;
        write_regenie_text_float32_value(output_writer, info_score_array, row_index)?;
        output_writer.write_all(b"\t").map_err(OutputError::runtime)?;
        write_regenie_text_int32_value(output_writer, observation_count_array, row_index)?;
        output_writer.write_all(b"\t").map_err(OutputError::runtime)?;
        write_regenie_text_string_value(output_writer, test_array, row_index, "TEST")?;
        output_writer.write_all(b"\t").map_err(OutputError::runtime)?;
        write_regenie_text_statistic_value(output_writer, beta_array, row_index)?;
        output_writer.write_all(b"\t").map_err(OutputError::runtime)?;
        write_regenie_text_statistic_value(output_writer, standard_error_array, row_index)?;
        output_writer.write_all(b"\t").map_err(OutputError::runtime)?;
        write_regenie_text_statistic_value(output_writer, chi_squared_array, row_index)?;
        output_writer.write_all(b"\t").map_err(OutputError::runtime)?;
        write_regenie_text_statistic_value(output_writer, log10_p_value_array, row_index)?;
        output_writer.write_all(b"\t").map_err(OutputError::runtime)?;
        write_regenie_text_string_value(output_writer, extra_array, row_index, "EXTRA")?;
        output_writer.write_all(b"\t").map_err(OutputError::runtime)?;
        write_regenie_text_string_value(output_writer, correction_method_array, row_index, "CORRECTION_METHOD")?;
        output_writer.write_all(b"\t").map_err(OutputError::runtime)?;
        write_regenie_text_string_value(output_writer, correction_status_array, row_index, "CORRECTION_STATUS")?;
        output_writer.write_all(b"\n").map_err(OutputError::runtime)?;
    }
    Ok(())
}

fn required_string_column<'a>(record_batch: &'a RecordBatch, column_name: &str) -> OutputResult<&'a StringArray> {
    record_batch
        .column_by_name(column_name)
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| {
            OutputError::InvalidInput(format!("REGENIE text writer could not read string column {column_name}."))
        })
}

fn required_float32_column<'a>(record_batch: &'a RecordBatch, column_name: &str) -> OutputResult<&'a Float32Array> {
    record_batch
        .column_by_name(column_name)
        .and_then(|column| column.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| {
            OutputError::InvalidInput(format!("REGENIE text writer could not read float32 column {column_name}."))
        })
}

#[derive(Clone, Copy)]
enum StatisticColumnRef<'a> {
    Float32(&'a Float32Array),
    Float64(&'a Float64Array),
}

fn required_statistic_column<'a>(
    record_batch: &'a RecordBatch,
    column_name: &str,
) -> OutputResult<StatisticColumnRef<'a>> {
    let Some(column) = record_batch.column_by_name(column_name) else {
        return Err(OutputError::InvalidInput(format!(
            "REGENIE text writer could not read statistic column {column_name}."
        )));
    };
    if let Some(float32_column) = column.as_any().downcast_ref::<Float32Array>() {
        return Ok(StatisticColumnRef::Float32(float32_column));
    }
    if let Some(float64_column) = column.as_any().downcast_ref::<Float64Array>() {
        return Ok(StatisticColumnRef::Float64(float64_column));
    }
    Err(OutputError::InvalidInput(format!(
        "REGENIE text writer could not read float32/float64 statistic column {column_name}."
    )))
}

fn required_int32_column<'a>(record_batch: &'a RecordBatch, column_name: &str) -> OutputResult<&'a Int32Array> {
    record_batch.column_by_name(column_name).and_then(|column| column.as_any().downcast_ref::<Int32Array>()).ok_or_else(
        || OutputError::InvalidInput(format!("REGENIE text writer could not read int32 column {column_name}.")),
    )
}

fn required_int64_column<'a>(record_batch: &'a RecordBatch, column_name: &str) -> OutputResult<&'a Int64Array> {
    record_batch.column_by_name(column_name).and_then(|column| column.as_any().downcast_ref::<Int64Array>()).ok_or_else(
        || OutputError::InvalidInput(format!("REGENIE text writer could not read int64 column {column_name}.")),
    )
}

fn write_regenie_text_string_value(
    output_writer: &mut BufWriter<File>,
    array: &StringArray,
    row_index: usize,
    column_name: &str,
) -> OutputResult<()> {
    if array.is_null(row_index) {
        return output_writer.write_all(REGENIE_STEP2_TEXT_MISSING_VALUE.as_bytes()).map_err(OutputError::runtime);
    }
    let value = array.value(row_index);
    if value.contains('\t') || value.contains('\n') || value.contains('\r') {
        return Err(OutputError::InvalidInput(format!(
            "REGENIE text writer found an unsupported separator in {column_name}."
        )));
    }
    output_writer.write_all(value.as_bytes()).map_err(OutputError::runtime)
}

fn write_regenie_text_float32_value(
    output_writer: &mut BufWriter<File>,
    array: &Float32Array,
    row_index: usize,
) -> OutputResult<()> {
    if array.is_null(row_index) {
        return output_writer.write_all(REGENIE_STEP2_TEXT_MISSING_VALUE.as_bytes()).map_err(OutputError::runtime);
    }
    let value = array.value(row_index);
    if !value.is_finite() {
        return output_writer.write_all(REGENIE_STEP2_TEXT_MISSING_VALUE.as_bytes()).map_err(OutputError::runtime);
    }
    write!(output_writer, "{value}").map_err(OutputError::runtime)
}

fn write_regenie_text_statistic_value(
    output_writer: &mut BufWriter<File>,
    array: StatisticColumnRef<'_>,
    row_index: usize,
) -> OutputResult<()> {
    match array {
        StatisticColumnRef::Float32(float32_array) => {
            write_regenie_text_float32_value(output_writer, float32_array, row_index)
        }
        StatisticColumnRef::Float64(float64_array) => {
            write_regenie_text_float64_value(output_writer, float64_array, row_index)
        }
    }
}

fn write_regenie_text_float64_value(
    output_writer: &mut BufWriter<File>,
    array: &Float64Array,
    row_index: usize,
) -> OutputResult<()> {
    if array.is_null(row_index) {
        return output_writer.write_all(REGENIE_STEP2_TEXT_MISSING_VALUE.as_bytes()).map_err(OutputError::runtime);
    }
    let value = array.value(row_index);
    if !value.is_finite() {
        return output_writer.write_all(REGENIE_STEP2_TEXT_MISSING_VALUE.as_bytes()).map_err(OutputError::runtime);
    }
    write!(output_writer, "{value}").map_err(OutputError::runtime)
}

fn write_regenie_text_int32_value(
    output_writer: &mut BufWriter<File>,
    array: &Int32Array,
    row_index: usize,
) -> OutputResult<()> {
    if array.is_null(row_index) {
        return output_writer.write_all(REGENIE_STEP2_TEXT_MISSING_VALUE.as_bytes()).map_err(OutputError::runtime);
    }
    write!(output_writer, "{}", array.value(row_index)).map_err(OutputError::runtime)
}

fn write_regenie_text_int64_value(
    output_writer: &mut BufWriter<File>,
    array: &Int64Array,
    row_index: usize,
) -> OutputResult<()> {
    if array.is_null(row_index) {
        return output_writer.write_all(REGENIE_STEP2_TEXT_MISSING_VALUE.as_bytes()).map_err(OutputError::runtime);
    }
    write!(output_writer, "{}", array.value(row_index)).map_err(OutputError::runtime)
}
