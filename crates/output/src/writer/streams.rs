use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::datatypes::Schema;
use arrow::ipc::CompressionType;
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;

use crate::error::OutputError;
use crate::manifest;
use crate::schema;

use super::chunk_manifest;
use super::record_batch::{
    RegenieStep2CorrectionArrayEncoding, RegenieStep2RecordBatchArrayCache,
    build_regenie_step2_parquet_record_batch_schema, build_regenie_step2_record_batch,
};
use super::{
    OutputResult, RegenieStep2ArrowFileWriteTiming, RegenieStep2ChunkJob, RegenieStep2ChunkStreamWriteResult,
    RegenieStep2RecordBatchBuildTiming,
};

const REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE: usize = 122_880;

pub(super) fn write_regenie_step2_chunks_to_arrow_file(
    chunks: Vec<RegenieStep2ChunkJob>,
    chunk_schema: &Arc<Schema>,
    chunk_file_path: &Path,
    arrow_compression: &str,
) -> OutputResult<RegenieStep2ChunkStreamWriteResult> {
    let file_create_start_time = Instant::now();
    let output_file = File::create(chunk_file_path).map_err(OutputError::runtime)?;
    let file_create = file_create_start_time.elapsed().as_secs_f64();

    let writer_init_start_time = Instant::now();
    let write_options = build_regenie_step2_ipc_write_options(arrow_compression)?;
    let mut writer =
        FileWriter::try_new_with_options(output_file, chunk_schema, write_options).map_err(OutputError::runtime)?;
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
        writer.write(&record_batch_build_result.record_batch).map_err(OutputError::runtime)?;
        batch_write += batch_write_start_time.elapsed().as_secs_f64();
    }
    let writer_finish_start_time = Instant::now();
    writer.finish().map_err(OutputError::runtime)?;
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

pub(super) fn write_regenie_step2_chunks_to_parquet_file(
    chunks: Vec<RegenieStep2ChunkJob>,
    chunk_schema: &Arc<Schema>,
    chunk_file_path: &Path,
    parquet_compression: &str,
    chunk_commits: &[manifest::RunManifestChunkCommit],
) -> OutputResult<RegenieStep2ChunkStreamWriteResult> {
    let file_create_start_time = Instant::now();
    let output_file = File::create(chunk_file_path).map_err(OutputError::runtime)?;
    let file_create = file_create_start_time.elapsed().as_secs_f64();

    let writer_init_start_time = Instant::now();
    let writer_properties = build_regenie_step2_parquet_writer_properties(parquet_compression)?;
    let mut writer = ArrowWriter::try_new(output_file, Arc::clone(chunk_schema), Some(writer_properties))
        .map_err(OutputError::runtime)?;
    writer.append_key_value_metadata(KeyValue {
        key: schema::CHUNK_COMMITS_METADATA_KEY.to_string(),
        value: Some(chunk_manifest::build_chunk_commit_metadata_text(chunk_commits)?),
    });
    let writer_init = writer_init_start_time.elapsed().as_secs_f64();

    let mut record_batch_build_timing = RegenieStep2RecordBatchBuildTiming::default();
    let mut record_batch_build_seconds = 0.0;
    let mut array_cache = RegenieStep2RecordBatchArrayCache::default();
    let parquet_record_batch_schema = build_regenie_step2_parquet_record_batch_schema(chunk_schema);
    let mut batch_write = 0.0;
    for chunk_job in chunks {
        let record_batch_build_start_time = Instant::now();
        let record_batch_build_result = build_regenie_step2_record_batch(
            chunk_job,
            Arc::clone(&parquet_record_batch_schema),
            &mut array_cache,
            RegenieStep2CorrectionArrayEncoding::Dictionary,
        )?;
        record_batch_build_seconds += record_batch_build_start_time.elapsed().as_secs_f64();
        record_batch_build_timing.add(record_batch_build_result.timing);

        let batch_write_start_time = Instant::now();
        writer.write(&record_batch_build_result.record_batch).map_err(OutputError::runtime)?;
        batch_write += batch_write_start_time.elapsed().as_secs_f64();
    }
    let writer_finish_start_time = Instant::now();
    writer.close().map_err(OutputError::runtime)?;
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

fn build_regenie_step2_ipc_write_options(arrow_compression: &str) -> OutputResult<IpcWriteOptions> {
    match arrow_compression.to_ascii_lowercase().as_str() {
        "zstd" => {
            IpcWriteOptions::default().try_with_compression(Some(CompressionType::ZSTD)).map_err(OutputError::runtime)
        }
        "none" => Ok(IpcWriteOptions::default()),
        unsupported_compression => Err(OutputError::InvalidInput(format!(
            "Arrow compression must be 'zstd' or 'none', observed '{unsupported_compression}'."
        ))),
    }
}

fn build_regenie_step2_parquet_writer_properties(parquet_compression: &str) -> OutputResult<WriterProperties> {
    let compression = match parquet_compression.to_ascii_lowercase().as_str() {
        "zstd" => Compression::ZSTD(ZstdLevel::default()),
        "none" => Compression::UNCOMPRESSED,
        unsupported_compression => {
            return Err(OutputError::InvalidInput(format!(
                "Parquet compression must be 'zstd' or 'none', observed '{unsupported_compression}'."
            )));
        }
    };
    Ok(WriterProperties::builder()
        .set_compression(compression)
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
        .build())
}
