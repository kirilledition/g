#![allow(clippy::needless_pass_by_value)]

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::ipc::CompressionType;
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use thiserror::Error;

use crate::output::NativeChunkHandle;
use crate::output::manifest;
use crate::output::schema;

#[derive(Debug, Error)]
pub enum OutputWriterError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Runtime(String),
}

impl OutputWriterError {
    pub(crate) fn runtime(error: impl ToString) -> Self {
        Self::Runtime(error.to_string())
    }
}

pub(crate) struct RegenieStep2ChunkJob {
    pub(crate) chunk_handle: NativeChunkHandle,
    pub(crate) beta: Vec<f32>,
    pub(crate) se: Vec<f32>,
    pub(crate) chisq: Vec<f32>,
    pub(crate) log10p: Vec<f32>,
    pub(crate) extra_code: Option<Vec<i32>>,
}

pub(crate) struct RegenieStep2ChunkWriteBatch {
    pub(crate) chunk_file_name: String,
    pub(crate) chunks: Vec<RegenieStep2ChunkJob>,
}

pub(crate) fn write_regenie_step2_chunk_job(
    run_directory: &Path,
    chunks_directory: &Path,
    job: RegenieStep2ChunkWriteBatch,
    arrow_compression: &str,
) -> Result<(), String> {
    let chunk_file_path = chunks_directory.join(&job.chunk_file_name);
    let temporary_chunk_file_path = chunk_file_path.with_extension("arrow.tmp");
    let chunk_commits = build_run_manifest_chunk_commits(&job)?;
    let record_batch = build_regenie_step2_record_batch(job)?;
    write_record_batch_to_arrow_file(&record_batch, &temporary_chunk_file_path, arrow_compression)?;
    std::fs::rename(&temporary_chunk_file_path, &chunk_file_path).map_err(|error| error.to_string())?;
    manifest::record_run_manifest_chunk_commits(run_directory, chunk_commits)?;
    Ok(())
}

fn build_run_manifest_chunk_commits(
    job: &RegenieStep2ChunkWriteBatch,
) -> Result<Vec<manifest::RunManifestChunkCommit>, String> {
    job.chunks
        .iter()
        .map(|chunk_job| {
            let variant_stop_index = chunk_job.chunk_handle.variant_stop_index().map_err(|error| error.to_string())?;
            Ok(manifest::RunManifestChunkCommit {
                chunk_identifier: chunk_job.chunk_handle.chunk_identifier,
                variant_start_index: chunk_job.chunk_handle.variant_start_index(),
                variant_stop_index,
                row_count: chunk_job.chunk_handle.row_count(),
                chunk_file_name: job.chunk_file_name.clone(),
            })
        })
        .collect()
}

pub(crate) fn build_chunk_file_name(first_chunk_identifier: i64, last_chunk_identifier: i64) -> String {
    if first_chunk_identifier == last_chunk_identifier {
        return format!("chunk_{first_chunk_identifier:09}.arrow");
    }
    format!("chunk_{first_chunk_identifier:09}_{last_chunk_identifier:09}.arrow")
}

fn build_regenie_step2_record_batch(job: RegenieStep2ChunkWriteBatch) -> Result<RecordBatch, String> {
    let schema = schema::get_regenie_step2_chunk_schema();
    let row_count = job.chunks.iter().map(|chunk_job| chunk_job.chunk_handle.row_count()).sum();
    let mut chunk_identifier = Vec::with_capacity(row_count);
    let mut variant_start_index = Vec::with_capacity(row_count);
    let mut variant_stop_index = Vec::with_capacity(row_count);
    let mut chrom = Vec::with_capacity(row_count);
    let mut genpos = Vec::with_capacity(row_count);
    let mut id = Vec::with_capacity(row_count);
    let mut allele0 = Vec::with_capacity(row_count);
    let mut allele1 = Vec::with_capacity(row_count);
    let mut a1freq = Vec::with_capacity(row_count);
    let mut info = Vec::with_capacity(row_count);
    let mut n = Vec::with_capacity(row_count);
    let mut beta = Vec::with_capacity(row_count);
    let mut se = Vec::with_capacity(row_count);
    let mut chisq = Vec::with_capacity(row_count);
    let mut log10p = Vec::with_capacity(row_count);
    let mut extra_code = Vec::with_capacity(row_count);

    for chunk_job in job.chunks {
        let chunk_row_count = chunk_job.chunk_handle.row_count();
        let chunk_variant_stop_index =
            chunk_job.chunk_handle.variant_stop_index().map_err(|error| error.to_string())?;
        chunk_identifier.extend(std::iter::repeat_n(chunk_job.chunk_handle.chunk_identifier, chunk_row_count));
        variant_start_index.extend(std::iter::repeat_n(chunk_job.chunk_handle.variant_start_index(), chunk_row_count));
        variant_stop_index.extend(std::iter::repeat_n(chunk_variant_stop_index, chunk_row_count));
        chrom.extend(chunk_job.chunk_handle.metadata.chromosome.iter().cloned());
        genpos.extend_from_slice(&chunk_job.chunk_handle.metadata.position);
        id.extend(chunk_job.chunk_handle.metadata.variant_identifier.iter().cloned());
        allele0.extend(chunk_job.chunk_handle.metadata.allele_two.iter().cloned());
        allele1.extend(chunk_job.chunk_handle.metadata.allele_one.iter().cloned());
        a1freq.extend_from_slice(&chunk_job.chunk_handle.stats.allele_one_frequency);
        info.extend(chunk_job.chunk_handle.stats.info_score.iter().copied());
        n.extend_from_slice(&chunk_job.chunk_handle.stats.observation_count);
        beta.extend(chunk_job.beta);
        se.extend(chunk_job.se);
        chisq.extend(chunk_job.chisq);
        log10p.extend(chunk_job.log10p);
        match chunk_job.extra_code {
            None => extra_code.extend(std::iter::repeat_n(None, chunk_row_count)),
            Some(extra_code_values) => {
                extra_code.extend(extra_code_values.into_iter().map(Some));
            }
        }
    }
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(chunk_identifier)),
        Arc::new(Int64Array::from(variant_start_index)),
        Arc::new(Int64Array::from(variant_stop_index)),
        Arc::new(schema::build_dictionary_string_array(&chrom)?),
        Arc::new(Int64Array::from(genpos)),
        Arc::new(StringArray::from(id)),
        Arc::new(schema::build_dictionary_string_array(&allele0)?),
        Arc::new(schema::build_dictionary_string_array(&allele1)?),
        Arc::new(Float32Array::from(a1freq)),
        Arc::new(Float32Array::from(info)),
        Arc::new(Int32Array::from(n)),
        Arc::new(schema::build_constant_dictionary_string_array(row_count, "ADD")?),
        Arc::new(Float32Array::from(beta)),
        Arc::new(Float32Array::from(se)),
        Arc::new(Float32Array::from(chisq)),
        Arc::new(Float32Array::from(log10p)),
        Arc::new(schema::build_extra_string_array(extra_code)?),
    ];
    RecordBatch::try_new(Arc::clone(schema), columns).map_err(|error| error.to_string())
}

fn write_record_batch_to_arrow_file(
    record_batch: &RecordBatch,
    chunk_file_path: &Path,
    arrow_compression: &str,
) -> Result<(), String> {
    let output_file = File::create(chunk_file_path).map_err(|error| error.to_string())?;
    let write_options = build_regenie_step2_ipc_write_options(arrow_compression)?;
    let mut writer = FileWriter::try_new_with_options(output_file, &record_batch.schema(), write_options)
        .map_err(|error| error.to_string())?;
    writer.write(record_batch).map_err(|error| error.to_string())?;
    writer.finish().map_err(|error| error.to_string())
}

fn build_regenie_step2_ipc_write_options(arrow_compression: &str) -> Result<IpcWriteOptions, String> {
    match arrow_compression.to_ascii_lowercase().as_str() {
        "zstd" => IpcWriteOptions::default()
            .try_with_compression(Some(CompressionType::ZSTD))
            .map_err(|error| error.to_string()),
        "none" => Ok(IpcWriteOptions::default()),
        unsupported_compression => {
            Err(format!("Arrow compression must be 'zstd' or 'none', observed '{unsupported_compression}'."))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use parquet::file::reader::{FileReader as ParquetFileReader, SerializedFileReader};

    use crate::genotype::common::{ChunkStats, VariantMetadataColumns};
    use crate::output::finalization;

    use super::*;

    fn build_test_chunk_handle(chunk_identifier: i64) -> NativeChunkHandle {
        NativeChunkHandle::new(
            Arc::new(VariantMetadataColumns {
                chromosome: vec!["22".to_string()],
                variant_identifier: vec![format!("variant{chunk_identifier}")],
                position: vec![100 + chunk_identifier],
                allele_one: vec!["A".to_string()],
                allele_two: vec!["G".to_string()],
            }),
            Arc::new(ChunkStats {
                allele_one_frequency: vec![0.5],
                observation_count: vec![100],
                has_missing_values: false,
                dosage_sum: vec![0.0],
                dosage_square_sum: vec![0.0],
                imputed_dosage_square_sum: vec![0.0],
                dosage_variance_numerator: vec![0.0],
                info_score: vec![Some(0.9)],
                allele_count: vec![0.0],
                minor_allele_count: vec![0.0],
                zero_count: vec![0],
                nonzero_count: vec![0],
                homozygous_reference_count: vec![0],
                heterozygous_count: vec![0],
                homozygous_alternate_count: vec![0],
                is_sparse_candidate: vec![false],
                is_rare_sparse_firth_candidate: vec![false],
            }),
            chunk_identifier,
        )
    }

    fn build_test_chunk(chunk_identifier: i64, extra_code: Option<Vec<i32>>) -> RegenieStep2ChunkJob {
        RegenieStep2ChunkJob {
            chunk_handle: build_test_chunk_handle(chunk_identifier),
            beta: vec![0.1],
            se: vec![0.01],
            chisq: vec![10.0],
            log10p: vec![5.0],
            extra_code,
        }
    }

    fn build_test_batch(chunks: Vec<RegenieStep2ChunkJob>) -> RegenieStep2ChunkWriteBatch {
        let first_chunk_identifier = chunks.first().map_or(0, |chunk_job| chunk_job.chunk_handle.chunk_identifier);
        let last_chunk_identifier =
            chunks.last().map_or(first_chunk_identifier, |chunk_job| chunk_job.chunk_handle.chunk_identifier);
        RegenieStep2ChunkWriteBatch {
            chunk_file_name: build_chunk_file_name(first_chunk_identifier, last_chunk_identifier),
            chunks,
        }
    }

    fn create_test_directory() -> PathBuf {
        let unique_suffix =
            SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after Unix epoch").as_nanos();
        let directory_path = std::env::temp_dir().join(format!("g-output-rust-test-{unique_suffix}"));
        std::fs::create_dir_all(&directory_path).expect("test directory should be created");
        directory_path
    }

    #[test]
    fn linear_record_batch_uses_shared_schema_and_null_extra() {
        let record_batch = build_regenie_step2_record_batch(build_test_batch(vec![build_test_chunk(0, None)]))
            .expect("linear record batch should build");

        assert_eq!(record_batch.schema().fields().len(), 17);
        assert!(record_batch.schema().field_with_name("INFO").expect("INFO field should exist").is_nullable());
        assert!(record_batch.schema().field_with_name("EXTRA").expect("EXTRA field should exist").is_nullable());
        assert_eq!(record_batch.num_rows(), 1);
        let info_array = record_batch
            .column_by_name("INFO")
            .expect("INFO column should exist")
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("INFO column should be a float32 array");
        assert!((info_array.value(0) - 0.9).abs() < f32::EPSILON);
        assert_eq!(record_batch.column_by_name("EXTRA").expect("EXTRA column should exist").null_count(), 1);
    }

    #[test]
    fn binary_record_batch_maps_extra_codes_with_same_schema() {
        let linear_record_batch = build_regenie_step2_record_batch(build_test_batch(vec![build_test_chunk(0, None)]))
            .expect("linear record batch should build");
        let binary_record_batch = build_regenie_step2_record_batch(build_test_batch(vec![
            build_test_chunk(1, Some(vec![1])),
            build_test_chunk(2, Some(vec![2])),
            build_test_chunk(3, Some(vec![3])),
        ]))
        .expect("binary record batch should build");

        assert_eq!(linear_record_batch.schema(), binary_record_batch.schema());
        let extra_array = binary_record_batch
            .column_by_name("EXTRA")
            .expect("EXTRA column should exist")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("EXTRA column should be a string array");
        assert_eq!(extra_array.value(0), "FIRTH");
        assert_eq!(extra_array.value(1), "SPA");
        assert_eq!(extra_array.value(2), "TEST_FAIL");
    }

    #[test]
    fn finalization_writes_footer_metadata() {
        let run_directory = create_test_directory();
        let chunks_directory = run_directory.join("chunks");
        std::fs::create_dir_all(&chunks_directory).expect("chunk directory should be created");
        write_regenie_step2_chunk_job(
            &run_directory,
            &chunks_directory,
            build_test_batch(vec![build_test_chunk(0, Some(vec![1])), build_test_chunk(1, Some(vec![0]))]),
            "zstd",
        )
        .expect("chunk batch should write");

        let final_parquet_path = run_directory.join("final.parquet");
        finalization::write_final_parquet_from_chunk_files(&chunks_directory, &final_parquet_path, "regenie2_binary")
            .expect("final parquet should write");

        let parquet_file = File::open(final_parquet_path).expect("final parquet should open");
        let parquet_reader = SerializedFileReader::new(parquet_file).expect("parquet reader should open");
        let key_value_metadata =
            parquet_reader.metadata().file_metadata().key_value_metadata().expect("footer metadata should exist");
        let metadata_value = |key: &str| {
            key_value_metadata.iter().find(|entry| entry.key == key).and_then(|entry| entry.value.as_deref())
        };
        assert_eq!(metadata_value("g.output.schema_version"), Some("1"));
        assert_eq!(metadata_value("g.output.association_mode"), Some("regenie2_binary"));
        assert_eq!(metadata_value("g.output.chunk_file_count"), Some("1"));
        assert_eq!(metadata_value("g.output.row_count"), Some("2"));
        assert_eq!(metadata_value("g.output.writer"), Some("rust"));

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }
}
