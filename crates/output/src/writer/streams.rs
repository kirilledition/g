use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::Schema;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;

use crate::persistence::io::path_operation_error;
use crate::persistence::receipt::OutputPartFooter;
use crate::schema;
use crate::timing::start_optional_timing;

use super::record_batch::{RegenieStep2RecordBatchArrayCache, build_regenie_step2_record_batch};
use super::{
    OutputResult, REGENIE_STEP2_PARQUET_FLOAT_ENCODING, REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE,
    REGENIE_STEP2_PARQUET_WRITE_BATCH_SIZE, REGENIE_STEP2_PARQUET_WRITER_VERSION, RegenieStep2ChunkJob,
    RegenieStep2ChunkStreamWriteResult, RegenieStep2ParquetFileWriteTiming, RegenieStep2RecordBatchBuildTiming,
};

pub(super) struct RegenieStep2ParquetStreamRequest<'request> {
    pub(super) chunks: Vec<RegenieStep2ChunkJob>,
    pub(super) chunk_schema: &'request Arc<Schema>,
    pub(super) parquet_record_batch_schema: &'request Arc<Schema>,
    pub(super) chunk_file_path: &'request Path,
    pub(super) part_footer: &'request OutputPartFooter,
    pub(super) file_create_seconds: f64,
    pub(super) collect_stage_timings: bool,
}

pub(super) fn write_regenie_step2_chunks_to_parquet_file<OutputFile: Write + Send>(
    output_file: OutputFile,
    request: RegenieStep2ParquetStreamRequest<'_>,
) -> OutputResult<RegenieStep2ChunkStreamWriteResult<OutputFile>> {
    let writer_init_start_time = start_optional_timing(request.collect_stage_timings);
    let writer_properties = build_regenie_step2_parquet_writer_properties();
    let mut writer = ArrowWriter::try_new(output_file, Arc::clone(request.chunk_schema), Some(writer_properties))
        .map_err(|error| path_operation_error("initialize temporary Parquet part", request.chunk_file_path, &error))?;
    writer.append_key_value_metadata(KeyValue {
        key: schema::PART_BINDING_METADATA_KEY.to_string(),
        value: Some(request.part_footer.to_metadata_text()?),
    });
    let writer_init = writer_init_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64());

    let mut record_batch_build_timing = RegenieStep2RecordBatchBuildTiming::default();
    let mut record_batch_build_seconds = 0.0;
    let mut array_cache = RegenieStep2RecordBatchArrayCache::default();
    let mut batch_write = 0.0;
    for chunk_job in request.chunks {
        let record_batch_build_start_time = start_optional_timing(request.collect_stage_timings);
        let record_batch_build_result = build_regenie_step2_record_batch(
            chunk_job,
            Arc::clone(request.parquet_record_batch_schema),
            &mut array_cache,
            request.collect_stage_timings,
        )?;
        record_batch_build_seconds +=
            record_batch_build_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64());
        record_batch_build_timing.add(record_batch_build_result.timing);

        let batch_write_start_time = start_optional_timing(request.collect_stage_timings);
        writer
            .write(&record_batch_build_result.record_batch)
            .map_err(|error| path_operation_error("write temporary Parquet part", request.chunk_file_path, &error))?;
        batch_write += batch_write_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64());
    }
    let writer_finish_start_time = start_optional_timing(request.collect_stage_timings);
    let output_file = writer
        .into_inner()
        .map_err(|error| path_operation_error("finalize temporary Parquet part", request.chunk_file_path, &error))?;
    let writer_finish = writer_finish_start_time.map_or(0.0, |start_time| start_time.elapsed().as_secs_f64());
    Ok(RegenieStep2ChunkStreamWriteResult {
        output_file,
        record_batch_build_timing,
        record_batch_build_seconds,
        parquet_file_write_timing: RegenieStep2ParquetFileWriteTiming {
            file_create: request.file_create_seconds,
            writer_init,
            batch_write,
            writer_finish,
        },
    })
}

fn build_regenie_step2_parquet_writer_properties() -> WriterProperties {
    let compression = Compression::ZSTD(ZstdLevel::default());
    let mut properties = WriterProperties::builder()
        .set_writer_version(REGENIE_STEP2_PARQUET_WRITER_VERSION)
        .set_compression(compression)
        .set_write_batch_size(REGENIE_STEP2_PARQUET_WRITE_BATCH_SIZE)
        .set_max_row_group_row_count(Some(REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE))
        .set_dictionary_enabled(false)
        .set_column_dictionary_enabled(ColumnPath::from("CHROM"), true)
        .set_column_dictionary_enabled(ColumnPath::from("ALLELE0"), true)
        .set_column_dictionary_enabled(ColumnPath::from("ALLELE1"), true)
        .set_column_dictionary_enabled(ColumnPath::from("N"), true)
        .set_column_dictionary_enabled(ColumnPath::from("CORRECTION_METHOD"), true)
        .set_column_dictionary_enabled(ColumnPath::from("CORRECTION_STATUS"), true);
    for column_name in ["A1FREQ", "INFO", "BETA", "SE", "CHISQ", "LOG10P"] {
        properties =
            properties.set_column_encoding(ColumnPath::from(column_name), REGENIE_STEP2_PARQUET_FLOAT_ENCODING);
    }
    properties.build()
}
