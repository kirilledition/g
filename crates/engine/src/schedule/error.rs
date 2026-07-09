//! Scheduler error boundary.

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ScheduleError {
    #[error("staging_depth must be positive.")]
    NonPositiveStagingDepth,
    #[error("native_callback_batch_size must be positive.")]
    NonPositiveCallbackBatchSize,
    #[error("native_callback_batch_size > 1 is not supported for packed8 BGEN delivery.")]
    Packed8CallbackBatchSize,
    #[error("native_callback_batch_size > 1 is not supported for grouped union BGEN delivery.")]
    GroupedUnionCallbackBatchSize,
    #[error("native_callback_batch_size must not exceed the effective dosage_buffer_limit ({dosage_buffer_limit}).")]
    CallbackBatchSizeExceedsDosageBufferLimit { dosage_buffer_limit: usize },
    #[error("result_in_flight_limit must be positive when provided.")]
    NonPositiveResultInFlightLimit,
    #[error("dosage_buffer_limit must be positive when provided.")]
    NonPositiveDosageBufferLimit,
    #[error("staging_depth exceeds platform capacity: {staging_depth}")]
    StagingDepthOverflow { staging_depth: i64 },
    #[error("native_callback_batch_size exceeds platform capacity: {callback_batch_size}")]
    CallbackBatchSizeOverflow { callback_batch_size: i64 },
    #[error("result_in_flight_limit exceeds platform capacity: {result_in_flight_limit}")]
    ResultInFlightLimitOverflow { result_in_flight_limit: i64 },
    #[error("dosage_buffer_limit exceeds platform capacity: {dosage_buffer_limit}")]
    DosageBufferLimitOverflow { dosage_buffer_limit: i64 },
    #[error("queue limit default for staging_depth exceeds platform capacity: {staging_depth}")]
    QueueLimitDefaultOverflow { staging_depth: usize },
    #[error("Variant-major dosage batch inputs must have identical lengths.")]
    VariantMajorDosageBatchLengthMismatch,
    #[error("Variant-major dosage batch must contain at least one chunk.")]
    EmptyVariantMajorDosageBatch,
    #[error("Dosage work handoff must contain at least one chunk.")]
    EmptyDosageWorkHandoff,
    #[error(
        "Committed chunk identifier set count ({committed_set_count}) must match writer session count ({writer_session_count})."
    )]
    MultiTraitCommittedChunkSetCountMismatch { writer_session_count: usize, committed_set_count: usize },
    #[error("Unsupported GPU genotype format: {gpu_genotype_format}")]
    UnsupportedGpuGenotypeFormat { gpu_genotype_format: String },
    #[error("Unsupported JAX device: {jax_device}")]
    UnsupportedJaxDevice { jax_device: String },
    #[error("Unsupported callback queue stage operation: {queue_name}.{operation_name}")]
    UnsupportedCallbackQueueStageOperation { queue_name: String, operation_name: String },
    #[error("Unsupported callback queue operation: {queue_name}.{operation_name}")]
    UnsupportedCallbackQueueOperation { queue_name: String, operation_name: String },
    #[error("Unsupported result write item kind: {result_work_item_kind}")]
    UnsupportedResultWriteItemKind { result_work_item_kind: String },
    #[error("Unsupported dosage work item kind: {dosage_work_item_kind}")]
    UnsupportedDosageWorkItemKind { dosage_work_item_kind: String },
    #[error("Dosage work item stage duration attribution requires at least one chunk.")]
    EmptyDosageWorkItemStageDuration,
    #[error("Cannot record stage duration for a dosage work stop signal.")]
    DosageWorkItemStageDurationStopSignal,
    #[error("Dosage work item kind {dosage_work_item_kind} must attribute to exactly one chunk, got {chunk_count}.")]
    DosageWorkItemStageDurationChunkCountMismatch { dosage_work_item_kind: String, chunk_count: usize },
    #[error("Dosage work item stage duration chunk count exceeds floating-point attribution capacity: {chunk_count}.")]
    DosageWorkItemStageDurationChunkCountOverflow { chunk_count: usize },
}
