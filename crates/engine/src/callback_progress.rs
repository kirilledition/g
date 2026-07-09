//! Native callback progress state and chunk identity policy.

const CALLBACK_PROGRESS_EVENT_LEVEL: &str = "info";
const CHROMOSOME_COMPLETED_EVENT_NAME: &str = "chromosome_completed";
const CHROMOSOME_STARTED_EVENT_NAME: &str = "chromosome_started";

#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(clippy::struct_field_names)]
pub struct CallbackChunkIdentity {
    pub chunk_identifier: i64,
    pub chromosome: String,
    pub variant_start_index: i64,
    pub variant_stop_index: i64,
    pub variant_count: i64,
}

impl CallbackChunkIdentity {
    #[must_use]
    pub fn new(chromosome: String, variant_start_index: i64, variant_stop_index: i64) -> Self {
        Self {
            chunk_identifier: variant_start_index,
            chromosome,
            variant_start_index,
            variant_stop_index,
            variant_count: variant_stop_index - variant_start_index,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackProgressUpdate {
    pub processed_chunk_count: i64,
    pub completed_chromosome: Option<String>,
    pub completed_processed_chunk_count: Option<i64>,
    pub started_chromosome: Option<String>,
    pub chunk_identity: CallbackChunkIdentity,
}

impl CallbackProgressUpdate {
    #[must_use]
    pub fn telemetry_plan(&self) -> CallbackProgressTelemetryPlan {
        let mut events = Vec::new();
        if let (Some(completed_chromosome), Some(completed_processed_chunk_count)) =
            (self.completed_chromosome.as_ref(), self.completed_processed_chunk_count)
        {
            events.push(CallbackProgressTelemetryEvent {
                event_name: CHROMOSOME_COMPLETED_EVENT_NAME.to_string(),
                level: CALLBACK_PROGRESS_EVENT_LEVEL.to_string(),
                chromosome: completed_chromosome.clone(),
                processed_chunk_count: completed_processed_chunk_count,
            });
        }
        if let Some(started_chromosome) = self.started_chromosome.as_ref() {
            events.push(CallbackProgressTelemetryEvent {
                event_name: CHROMOSOME_STARTED_EVENT_NAME.to_string(),
                level: CALLBACK_PROGRESS_EVENT_LEVEL.to_string(),
                chromosome: started_chromosome.clone(),
                processed_chunk_count: self.processed_chunk_count,
            });
        }
        CallbackProgressTelemetryPlan {
            events,
            progress: CallbackProgressTelemetryRecord {
                processed_chunk_count: self.processed_chunk_count,
                chromosome: self.chunk_identity.chromosome.clone(),
                chunk_identifier: self.chunk_identity.chunk_identifier,
                variant_start_index: self.chunk_identity.variant_start_index,
                variant_stop_index: self.chunk_identity.variant_stop_index,
                variant_count: self.chunk_identity.variant_count,
            },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackProgressCompletion {
    pub chromosome: String,
    pub processed_chunk_count: i64,
}

impl CallbackProgressCompletion {
    #[must_use]
    pub fn telemetry_event(&self) -> CallbackProgressTelemetryEvent {
        CallbackProgressTelemetryEvent {
            event_name: CHROMOSOME_COMPLETED_EVENT_NAME.to_string(),
            level: CALLBACK_PROGRESS_EVENT_LEVEL.to_string(),
            chromosome: self.chromosome.clone(),
            processed_chunk_count: self.processed_chunk_count,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackProgressTelemetryEvent {
    pub event_name: String,
    pub level: String,
    pub chromosome: String,
    pub processed_chunk_count: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackProgressTelemetryRecord {
    pub processed_chunk_count: i64,
    pub chromosome: String,
    pub chunk_identifier: i64,
    pub variant_start_index: i64,
    pub variant_stop_index: i64,
    pub variant_count: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackProgressTelemetryPlan {
    pub events: Vec<CallbackProgressTelemetryEvent>,
    pub progress: CallbackProgressTelemetryRecord,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CallbackProgressState {
    processed_chunk_count: i64,
    current_progress_chromosome: Option<String>,
}

impl CallbackProgressState {
    #[must_use]
    pub const fn new() -> Self {
        Self { processed_chunk_count: 0, current_progress_chromosome: None }
    }

    #[must_use]
    pub const fn processed_chunk_count(&self) -> i64 {
        self.processed_chunk_count
    }

    #[must_use]
    pub fn current_progress_chromosome(&self) -> Option<&str> {
        self.current_progress_chromosome.as_deref()
    }

    pub fn record_processed_chunk(&mut self, chunk_identity: CallbackChunkIdentity) -> CallbackProgressUpdate {
        self.processed_chunk_count += 1;
        let mut completed_chromosome = None;
        let mut completed_processed_chunk_count = None;
        let mut started_chromosome = None;
        if self.current_progress_chromosome.as_deref() != Some(chunk_identity.chromosome.as_str()) {
            completed_chromosome = self.current_progress_chromosome.replace(chunk_identity.chromosome.clone());
            completed_processed_chunk_count = completed_chromosome.as_ref().map(|_| self.processed_chunk_count - 1);
            started_chromosome = Some(chunk_identity.chromosome.clone());
        }
        CallbackProgressUpdate {
            processed_chunk_count: self.processed_chunk_count,
            completed_chromosome,
            completed_processed_chunk_count,
            started_chromosome,
            chunk_identity,
        }
    }

    pub fn record_processed_chunk_without_progress(&mut self) {
        self.processed_chunk_count += 1;
    }

    pub fn finish_progress(&mut self) -> Option<CallbackProgressCompletion> {
        self.current_progress_chromosome.take().map(|chromosome| CallbackProgressCompletion {
            chromosome,
            processed_chunk_count: self.processed_chunk_count,
        })
    }
}
