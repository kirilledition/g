//! Shared error boundary for engine planning policy.

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ScheduleError {
    #[error(
        "Committed chunk identifier set count ({committed_set_count}) must match writer session count ({writer_session_count})."
    )]
    MultiTraitCommittedChunkSetCountMismatch { writer_session_count: usize, committed_set_count: usize },
    #[error("Unsupported GPU genotype format: {gpu_genotype_format}")]
    UnsupportedGpuGenotypeFormat { gpu_genotype_format: String },
    #[error("Unsupported JAX device: {jax_device}")]
    UnsupportedJaxDevice { jax_device: String },
}
