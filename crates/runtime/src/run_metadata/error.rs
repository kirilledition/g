use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RunMetadataError {
    ArtifactSequenceLengthMismatch {
        output_run_directory_count: usize,
        chunks_directory_count: usize,
        effective_config_count: usize,
        phenotype_name_count: usize,
        final_output_path_count: usize,
    },
}

impl fmt::Display for RunMetadataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArtifactSequenceLengthMismatch { .. } => {
                formatter.write_str("execution artifact sequence lengths must match")
            }
        }
    }
}

impl std::error::Error for RunMetadataError {}
