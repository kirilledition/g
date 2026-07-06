use super::{
    DOSAGE_WORK_ITEM_KIND_SAMPLE_MAJOR_DOSAGE, DOSAGE_WORK_ITEM_KIND_STOP_SIGNAL,
    DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE, DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH,
    DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_PACKED8_PROBABILITY_PAIR, ScheduleError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DosageWorkDrainCompletionPlan {
    pub should_stop: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DosageWorkItemDispatchPlan {
    pub dosage_work_item_kind: String,
    pub processing_path: Option<String>,
    pub error_message: Option<String>,
}

impl DosageWorkItemDispatchPlan {
    #[must_use]
    pub fn should_process_sample_major_dosage(&self) -> bool {
        self.processing_path.as_deref() == Some(DOSAGE_WORK_ITEM_KIND_SAMPLE_MAJOR_DOSAGE)
    }

    #[must_use]
    pub fn should_process_variant_major_dosage(&self) -> bool {
        self.processing_path.as_deref() == Some(DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE)
    }

    #[must_use]
    pub fn should_process_variant_major_dosage_batch(&self) -> bool {
        self.processing_path.as_deref() == Some(DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH)
    }

    #[must_use]
    pub fn should_process_variant_major_packed8_probability_pair(&self) -> bool {
        self.processing_path.as_deref() == Some(DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_PACKED8_PROBABILITY_PAIR)
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
    validate_dosage_work_item_kind(dosage_work_item_kind)?;

    if dosage_work_item_kind == DOSAGE_WORK_ITEM_KIND_STOP_SIGNAL {
        return Ok(DosageWorkItemDispatchPlan {
            dosage_work_item_kind: dosage_work_item_kind.to_owned(),
            processing_path: None,
            error_message: Some("Native dosage work dispatch plan continued without a work item.".to_owned()),
        });
    }

    Ok(DosageWorkItemDispatchPlan {
        dosage_work_item_kind: dosage_work_item_kind.to_owned(),
        processing_path: Some(dosage_work_item_kind.to_owned()),
        error_message: None,
    })
}

fn validate_dosage_work_item_kind(dosage_work_item_kind: &str) -> Result<(), ScheduleError> {
    match dosage_work_item_kind {
        DOSAGE_WORK_ITEM_KIND_SAMPLE_MAJOR_DOSAGE
        | DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE
        | DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH
        | DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_PACKED8_PROBABILITY_PAIR
        | DOSAGE_WORK_ITEM_KIND_STOP_SIGNAL => Ok(()),
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
    validate_dosage_work_item_kind(dosage_work_item_kind)?;
    if dosage_work_item_kind == DOSAGE_WORK_ITEM_KIND_STOP_SIGNAL {
        return Err(ScheduleError::DosageWorkItemStageDurationStopSignal);
    }
    if chunk_count == 0 {
        return Err(ScheduleError::EmptyDosageWorkItemStageDuration);
    }
    if dosage_work_item_kind != DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH && chunk_count != 1 {
        return Err(ScheduleError::DosageWorkItemStageDurationChunkCountMismatch {
            dosage_work_item_kind: dosage_work_item_kind.to_owned(),
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
