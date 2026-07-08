#[derive(Debug, thiserror::Error)]
pub enum PipelineResumeCompatibilityError {
    #[error(
        "Resume compatibility input counts must match: chunks_directory_count={chunks_directory_count}, \
         manifest_count={manifest_count}, header_count={header_count}."
    )]
    MismatchedInputCounts { chunks_directory_count: usize, manifest_count: usize, header_count: usize },
    #[error(
        "Pipeline output run directory count must match chunks directory count: run_directory_count={run_directory_count}, \
         chunks_directory_count={chunks_directory_count}."
    )]
    MismatchedOutputRunDirectoryCount { run_directory_count: usize, chunks_directory_count: usize },
    #[error("Resume requires run_manifest.json.")]
    MissingManifest,
    #[error(transparent)]
    Output(#[from] g_output::OutputError),
}
