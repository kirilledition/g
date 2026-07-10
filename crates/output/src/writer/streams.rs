use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::datatypes::Schema;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;

use crate::error::OutputError;
use crate::manifest;
use crate::schema;

use super::chunk_manifest;
use super::record_batch::{RegenieStep2RecordBatchArrayCache, build_regenie_step2_record_batch};
use super::{
    OutputResult, RegenieStep2ChunkJob, RegenieStep2ChunkStreamWriteResult, RegenieStep2ParquetFileWriteTiming,
    RegenieStep2RecordBatchBuildTiming,
};

const REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE: usize = 122_880;

pub(super) fn write_regenie_step2_chunks_to_parquet_file(
    chunks: Vec<RegenieStep2ChunkJob>,
    chunk_schema: &Arc<Schema>,
    parquet_record_batch_schema: &Arc<Schema>,
    chunk_file_path: &Path,
    parquet_compression: g_plan::ParquetCompression,
    chunk_commits: &[manifest::RunManifestChunkCommit],
) -> OutputResult<RegenieStep2ChunkStreamWriteResult> {
    let file_create_start_time = Instant::now();
    let output_file = File::create(chunk_file_path).map_err(OutputError::runtime)?;
    let file_create = file_create_start_time.elapsed().as_secs_f64();

    let writer_init_start_time = Instant::now();
    let writer_properties = build_regenie_step2_parquet_writer_properties(parquet_compression);
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
    let mut batch_write = 0.0;
    for chunk_job in chunks {
        let record_batch_build_start_time = Instant::now();
        let record_batch_build_result =
            build_regenie_step2_record_batch(chunk_job, Arc::clone(parquet_record_batch_schema), &mut array_cache)?;
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
        parquet_file_write_timing: RegenieStep2ParquetFileWriteTiming {
            file_create,
            writer_init,
            batch_write,
            writer_finish,
        },
    })
}

fn build_regenie_step2_parquet_writer_properties(parquet_compression: g_plan::ParquetCompression) -> WriterProperties {
    let compression = match parquet_compression {
        g_plan::ParquetCompression::Zstd => Compression::ZSTD(ZstdLevel::default()),
        g_plan::ParquetCompression::None => Compression::UNCOMPRESSED,
    };
    WriterProperties::builder()
        .set_compression(compression)
        .set_max_row_group_row_count(Some(REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE))
        .set_dictionary_enabled(false)
        .set_column_dictionary_enabled(ColumnPath::from("CHROM"), true)
        .set_column_dictionary_enabled(ColumnPath::from("ALLELE0"), true)
        .set_column_dictionary_enabled(ColumnPath::from("ALLELE1"), true)
        .set_column_dictionary_enabled(ColumnPath::from("N"), true)
        .set_column_dictionary_enabled(ColumnPath::from("CORRECTION_METHOD"), true)
        .set_column_dictionary_enabled(ColumnPath::from("CORRECTION_STATUS"), true)
        .build()
}
