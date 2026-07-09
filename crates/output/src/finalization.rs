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
pub(crate) use regenie_text::write_final_regenie_from_chunk_files_with_timing;

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
