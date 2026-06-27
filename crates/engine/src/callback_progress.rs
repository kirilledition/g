//! Native callback progress state and chunk identity policy.

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackProgressCompletion {
    pub chromosome: String,
    pub processed_chunk_count: i64,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_callback_chunk_identity_from_metadata_fields() {
        assert_eq!(
            CallbackChunkIdentity::new("chr2".to_string(), 64, 96),
            CallbackChunkIdentity {
                chunk_identifier: 64,
                chromosome: "chr2".to_string(),
                variant_start_index: 64,
                variant_stop_index: 96,
                variant_count: 32,
            },
        );
    }

    #[test]
    fn tracks_callback_progress_chromosome_transitions() {
        let mut state = CallbackProgressState::new();

        let first_update = state.record_processed_chunk(CallbackChunkIdentity::new("chr1".to_string(), 0, 8));
        assert_eq!(first_update.processed_chunk_count, 1);
        assert_eq!(first_update.completed_chromosome, None);
        assert_eq!(first_update.completed_processed_chunk_count, None);
        assert_eq!(first_update.started_chromosome, Some("chr1".to_string()));
        assert_eq!(state.current_progress_chromosome(), Some("chr1"));

        let same_chromosome_update =
            state.record_processed_chunk(CallbackChunkIdentity::new("chr1".to_string(), 8, 16));
        assert_eq!(same_chromosome_update.processed_chunk_count, 2);
        assert_eq!(same_chromosome_update.completed_chromosome, None);
        assert_eq!(same_chromosome_update.started_chromosome, None);

        let chromosome_transition_update =
            state.record_processed_chunk(CallbackChunkIdentity::new("chr2".to_string(), 16, 24));
        assert_eq!(chromosome_transition_update.processed_chunk_count, 3);
        assert_eq!(chromosome_transition_update.completed_chromosome, Some("chr1".to_string()));
        assert_eq!(chromosome_transition_update.completed_processed_chunk_count, Some(2));
        assert_eq!(chromosome_transition_update.started_chromosome, Some("chr2".to_string()));
        assert_eq!(state.current_progress_chromosome(), Some("chr2"));
    }

    #[test]
    fn finishes_active_progress_chromosome_once() {
        let mut state = CallbackProgressState::new();

        assert_eq!(state.finish_progress(), None);

        state.record_processed_chunk(CallbackChunkIdentity::new("chr3".to_string(), 5, 9));
        assert_eq!(
            state.finish_progress(),
            Some(CallbackProgressCompletion { chromosome: "chr3".to_string(), processed_chunk_count: 1 }),
        );
        assert_eq!(state.finish_progress(), None);
    }

    #[test]
    fn increments_processed_count_without_opening_progress_chromosome() {
        let mut state = CallbackProgressState::new();

        state.record_processed_chunk_without_progress();

        assert_eq!(state.processed_chunk_count(), 1);
        assert_eq!(state.current_progress_chromosome(), None);
        assert_eq!(state.finish_progress(), None);
    }
}
