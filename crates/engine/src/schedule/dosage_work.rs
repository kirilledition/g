use super::ScheduleError;

const DOSAGE_WORK_ITEM_KIND_SAMPLE_MAJOR_DOSAGE: &str = "sample_major_dosage";
const DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE: &str = "variant_major_dosage";
const DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH: &str = "variant_major_dosage_batch";
const DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_PACKED8_PROBABILITY_PAIR: &str = "variant_major_packed8_probability_pair";
const DOSAGE_WORK_ITEM_KIND_STOP_SIGNAL: &str = "stop_signal";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DosageWorkItemKind {
    SampleMajorDosage,
    StopSignal,
    VariantMajorDosage,
    VariantMajorDosageBatch,
    VariantMajorPacked8ProbabilityPair,
}

impl DosageWorkItemKind {
    #[must_use]
    pub const fn as_value(self) -> &'static str {
        match self {
            Self::SampleMajorDosage => DOSAGE_WORK_ITEM_KIND_SAMPLE_MAJOR_DOSAGE,
            Self::StopSignal => DOSAGE_WORK_ITEM_KIND_STOP_SIGNAL,
            Self::VariantMajorDosage => DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE,
            Self::VariantMajorDosageBatch => DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH,
            Self::VariantMajorPacked8ProbabilityPair => DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_PACKED8_PROBABILITY_PAIR,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DosageWorkDrainCompletionPlan {
    pub should_stop: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DosageWorkItemDispatchPlan {
    dosage_work_item_kind: DosageWorkItemKind,
    processing_path: Option<DosageWorkItemKind>,
    pub error_message: Option<String>,
}

impl DosageWorkItemDispatchPlan {
    #[must_use]
    pub const fn dosage_work_item_kind(&self) -> DosageWorkItemKind {
        self.dosage_work_item_kind
    }

    #[must_use]
    pub fn should_process_sample_major_dosage(&self) -> bool {
        self.processing_path == Some(DosageWorkItemKind::SampleMajorDosage)
    }

    #[must_use]
    pub fn should_process_variant_major_dosage(&self) -> bool {
        self.processing_path == Some(DosageWorkItemKind::VariantMajorDosage)
    }

    #[must_use]
    pub fn should_process_variant_major_dosage_batch(&self) -> bool {
        self.processing_path == Some(DosageWorkItemKind::VariantMajorDosageBatch)
    }

    #[must_use]
    pub fn should_process_variant_major_packed8_probability_pair(&self) -> bool {
        self.processing_path == Some(DosageWorkItemKind::VariantMajorPacked8ProbabilityPair)
    }

    #[must_use]
    pub fn has_dispatch_error(&self) -> bool {
        self.error_message.is_some()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DosageWorkItemStageDurationPlan {
    pub chunk_count: usize,
    pub duration_per_chunk: f64,
}

/// Plan which dosage processing path should consume a dequeued work item.
///
/// # Errors
///
/// Returns an error when the work-item kind is unsupported.
pub fn plan_dosage_work_item_dispatch(
    dosage_work_item_kind: &str,
) -> Result<DosageWorkItemDispatchPlan, ScheduleError> {
    let dosage_work_item_kind = parse_dosage_work_item_kind(dosage_work_item_kind)?;

    if dosage_work_item_kind == DosageWorkItemKind::StopSignal {
        return Ok(DosageWorkItemDispatchPlan {
            dosage_work_item_kind,
            processing_path: None,
            error_message: Some("Native dosage work dispatch plan continued without a work item.".to_owned()),
        });
    }

    Ok(DosageWorkItemDispatchPlan {
        dosage_work_item_kind,
        processing_path: Some(dosage_work_item_kind),
        error_message: None,
    })
}

fn parse_dosage_work_item_kind(dosage_work_item_kind: &str) -> Result<DosageWorkItemKind, ScheduleError> {
    match dosage_work_item_kind {
        DOSAGE_WORK_ITEM_KIND_SAMPLE_MAJOR_DOSAGE => Ok(DosageWorkItemKind::SampleMajorDosage),
        DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE => Ok(DosageWorkItemKind::VariantMajorDosage),
        DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH => Ok(DosageWorkItemKind::VariantMajorDosageBatch),
        DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_PACKED8_PROBABILITY_PAIR => {
            Ok(DosageWorkItemKind::VariantMajorPacked8ProbabilityPair)
        }
        DOSAGE_WORK_ITEM_KIND_STOP_SIGNAL => Ok(DosageWorkItemKind::StopSignal),
        _ => Err(ScheduleError::UnsupportedDosageWorkItemKind {
            dosage_work_item_kind: dosage_work_item_kind.to_owned(),
        }),
    }
}

/// Plan chunk-level timing attribution for one dosage work item.
///
/// # Errors
///
/// Returns an error when the work-item kind or chunk count is invalid.
pub fn plan_dosage_work_item_stage_duration(
    dosage_work_item_kind: &str,
    chunk_count: usize,
    elapsed_seconds: f64,
) -> Result<DosageWorkItemStageDurationPlan, ScheduleError> {
    let dosage_work_item_kind = parse_dosage_work_item_kind(dosage_work_item_kind)?;
    if dosage_work_item_kind == DosageWorkItemKind::StopSignal {
        return Err(ScheduleError::DosageWorkItemStageDurationStopSignal);
    }
    if chunk_count == 0 {
        return Err(ScheduleError::EmptyDosageWorkItemStageDuration);
    }
    if dosage_work_item_kind != DosageWorkItemKind::VariantMajorDosageBatch && chunk_count != 1 {
        return Err(ScheduleError::DosageWorkItemStageDurationChunkCountMismatch {
            dosage_work_item_kind: dosage_work_item_kind.as_value().to_owned(),
            chunk_count,
        });
    }
    let chunk_count_for_duration = u32::try_from(chunk_count)
        .map_err(|_| ScheduleError::DosageWorkItemStageDurationChunkCountOverflow { chunk_count })?;
    Ok(DosageWorkItemStageDurationPlan {
        chunk_count,
        duration_per_chunk: elapsed_seconds / f64::from(chunk_count_for_duration),
    })
}
