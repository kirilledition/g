#![allow(clippy::missing_errors_doc)]
#![allow(clippy::needless_pass_by_value)]

use std::path::{Path, PathBuf};

use crate::error::OutputError;
use crate::writer::OutputFileFormat;

mod parquet;
mod regenie_text;

pub(crate) use parquet::{
    RegenieStep2FinalizationTiming, manifest_output_chunk_file_paths, output_format_name,
    write_final_parquet_from_chunk_files_with_timing, write_final_parquet_from_chunk_files_with_timing_for_dtype,
};
#[cfg(test)]
pub(crate) use parquet::{
    prepare_chunk_batch_for_final_writer, project_chunk_batch_to_final_batch, sorted_output_chunk_file_paths,
};
pub(crate) use regenie_text::write_final_regenie_from_chunk_files_with_timing;

#[cfg(test)]
use crate::{manifest, schema};

pub fn finalize_output_run_chunks(
    run_directory: &Path,
    chunks_directory: &Path,
    association_mode: &str,
    output_format: OutputFileFormat,
) -> Result<PathBuf, OutputError> {
    match output_format {
        OutputFileFormat::Arrow | OutputFileFormat::Parquet => {
            let final_parquet_path = run_directory.join("final.parquet");
            write_final_parquet_from_chunk_files_with_timing(
                chunks_directory,
                &final_parquet_path,
                association_mode,
                output_format,
            )
            .map(|_| ())?;
            Ok(final_parquet_path)
        }
        OutputFileFormat::Regenie => {
            let final_regenie_path = run_directory.join("final.regenie");
            write_final_regenie_from_chunk_files_with_timing(
                chunks_directory,
                &final_regenie_path,
                association_mode,
                output_format,
            )
            .map(|_| ())?;
            Ok(final_regenie_path)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use arrow::array::{ArrayRef, Float32Array, Int32Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};

    use crate::writer::{self as output_writer, OutputFileFormat};

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

        let error = project_chunk_batch_to_final_batch(batch, schema::OutputStatisticDtype::Float32)
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
        let batch = RecordBatch::try_new(
            Arc::clone(schema::get_regenie_step2_final_schema(schema::OutputStatisticDtype::Float32)),
            columns.clone(),
        )
        .expect("ordered final batch should build");

        let prepared_batch =
            prepare_chunk_batch_for_final_writer(batch).expect("ordered final batch should be prepared");

        assert_eq!(
            prepared_batch.schema().fields(),
            schema::get_regenie_step2_final_schema(schema::OutputStatisticDtype::Float32).fields(),
        );
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
        write_final_regenie_from_chunk_files_with_timing(
            &regenie_directory,
            &final_regenie_path,
            "regenie2_binary",
            OutputFileFormat::Regenie,
        )
        .map(|_| ())
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
