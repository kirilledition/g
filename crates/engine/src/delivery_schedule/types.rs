#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuGenotypeFormatResolutionPlan {
    pub requested_gpu_genotype_format: String,
    pub resolved_gpu_genotype_format: Option<String>,
    pub resolution_reason: Option<String>,
    pub fallback_error: Option<String>,
    pub requires_trusted_validation: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BgenDeliveryCleanupOutcome {
    Failure,
    Interrupted,
    InterruptedCleanupFailure,
    Success,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BgenDeliveryErrorKind {
    Failure,
    Interrupted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BgenDeliveryCleanupAction {
    AbortCallback,
    AbortWriterSessions,
    DrainCallback,
    FinishInterruptedWriterSessions,
    FinishWriterSessions,
    WriteStageTimingSnapshot,
}

impl BgenDeliveryCleanupAction {
    #[must_use]
    pub const fn as_value(self) -> &'static str {
        match self {
            Self::AbortCallback => "abort_callback",
            Self::AbortWriterSessions => "abort_writer_sessions",
            Self::DrainCallback => "drain_callback",
            Self::FinishInterruptedWriterSessions => "finish_interrupted_writer_sessions",
            Self::FinishWriterSessions => "finish_writer_sessions",
            Self::WriteStageTimingSnapshot => "write_stage_timing_snapshot",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BgenDeliveryCleanupPlan {
    pub(super) cleanup_actions: Vec<BgenDeliveryCleanupAction>,
}

impl BgenDeliveryCleanupPlan {
    #[must_use]
    pub fn cleanup_actions(&self) -> &[BgenDeliveryCleanupAction] {
        &self.cleanup_actions
    }

    #[must_use]
    pub fn drain_callback(&self) -> bool {
        self.contains_cleanup_action(BgenDeliveryCleanupAction::DrainCallback)
    }

    #[must_use]
    pub fn finish_writer_sessions(&self) -> bool {
        self.contains_cleanup_action(BgenDeliveryCleanupAction::FinishWriterSessions)
    }

    #[must_use]
    pub fn finish_interrupted_writer_sessions(&self) -> bool {
        self.contains_cleanup_action(BgenDeliveryCleanupAction::FinishInterruptedWriterSessions)
    }

    #[must_use]
    pub fn abort_callback(&self) -> bool {
        self.contains_cleanup_action(BgenDeliveryCleanupAction::AbortCallback)
    }

    #[must_use]
    pub fn abort_writer_sessions(&self) -> bool {
        self.contains_cleanup_action(BgenDeliveryCleanupAction::AbortWriterSessions)
    }

    #[must_use]
    pub fn write_stage_timing_snapshot(&self) -> bool {
        self.contains_cleanup_action(BgenDeliveryCleanupAction::WriteStageTimingSnapshot)
    }

    fn contains_cleanup_action(&self, cleanup_action: BgenDeliveryCleanupAction) -> bool {
        self.cleanup_actions.contains(&cleanup_action)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BgenDeliveryMethod {
    DosageNativeMultiAlignedSamples,
    DosageNativeAlignedSamples,
    DosageSampleIndices,
    Packed8NativeMultiAlignedSamples,
    Packed8NativeAlignedSamples,
    Packed8SampleIndices,
}

impl BgenDeliveryMethod {
    #[must_use]
    pub const fn as_value(self) -> &'static str {
        match self {
            Self::DosageNativeMultiAlignedSamples => "dosage_native_multi_aligned_samples",
            Self::DosageNativeAlignedSamples => "dosage_native_aligned_samples",
            Self::DosageSampleIndices => "dosage_sample_indices",
            Self::Packed8NativeMultiAlignedSamples => "packed8_native_multi_aligned_samples",
            Self::Packed8NativeAlignedSamples => "packed8_native_aligned_samples",
            Self::Packed8SampleIndices => "packed8_sample_indices",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BgenDeliveryInvocationPlan {
    pub delivery_method: BgenDeliveryMethod,
    pub callback_batch_size: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BgenDeliveryAttemptPlan {
    pub committed_chunk_count: usize,
    pub invocation_plan: BgenDeliveryInvocationPlan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BgenDeliveryErrorHandlingPlan {
    pub cleanup_outcome: BgenDeliveryCleanupOutcome,
    pub fallback_cleanup_outcome: Option<BgenDeliveryCleanupOutcome>,
}
