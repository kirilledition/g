//! Native run-preparation policies.

mod batch;
mod error;
mod initialization;
mod output_runs;
mod runtime_output;
mod validation;

pub use batch::PipelineOutputPreparationBatch;
pub use error::PipelineResumeCompatibilityError;
pub use initialization::PipelineOutputInitialization;
pub use output_runs::{initialize_pipeline_output_run_batch, initialize_pipeline_output_runs};
pub use runtime_output::{
    PipelineOutputPreparationError, RuntimeOutputGroup, RuntimeOutputGroupInput, RuntimeOutputPhenotypeComputeGroup,
    RuntimeOutputPlan, RuntimeOutputPreparationGroup, RuntimeOutputPreparedRun,
    build_output_resume_committed_chunk_diagnostic_payloads, build_runtime_output_preparation_group,
};
pub use validation::validate_pipeline_resume_compatibility;
