//! Bounded two-stage association batch scheduler.

use std::any::Any;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::backend::{
    AssociationBackend, GenotypeBatchInput, GenotypeBatchStatistics, HostAssociationBatch, OwnedGenotypeBuffer,
};
use crossbeam_channel::{Receiver, SendTimeoutError, Sender, TryRecvError, TrySendError};
use g_genotype::ChunkStats;
use g_genotype_contracts::ChunkOutputStatistics;
use g_output::NativeVariantMetadataHandle;

use crate::output_schedule::ActiveTraitSelection;

const COMPUTE_WORKER_NAME: &str = "g-association-compute";
const MATERIALIZATION_WORKER_NAME: &str = "g-association-materialization";
const DECODED_BATCH_QUEUE: &str = "decoded batch";
const DEVICE_RESULT_QUEUE: &str = "device result";
const COMPLETED_BATCH_QUEUE: &str = "completed batch";
const COMPUTE_WORKER: &str = "compute";
const MATERIALIZATION_WORKER: &str = "materialization";

/// One owned association batch ready for device compute.
#[derive(Debug)]
pub(crate) struct ScheduledAssociationBatch {
    pub(crate) variant_start_index: usize,
    pub(crate) variant_count: usize,
    pub(crate) sample_count: usize,
    pub(crate) metadata: NativeVariantMetadataHandle,
    pub(crate) statistics: ChunkStats,
    pub(crate) genotype_buffer: OwnedGenotypeBuffer,
    pub(crate) active_trait_selection: ActiveTraitSelection,
}

impl ScheduledAssociationBatch {
    fn validate<BackendError>(&self) -> SchedulerResult<(), BackendError> {
        let genotype_value_count =
            self.variant_count.checked_mul(self.sample_count).ok_or_else(|| SchedulerError::InvalidBatch {
                message: "variant and sample counts overflow the host index width".to_string(),
            })?;
        let expected_genotype_value_count = match &self.genotype_buffer {
            OwnedGenotypeBuffer::Dosage(_) => genotype_value_count,
            OwnedGenotypeBuffer::Packed8(_) => {
                genotype_value_count.checked_mul(2).ok_or_else(|| SchedulerError::InvalidBatch {
                    message: "packed8 genotype dimensions overflow the host index width".to_string(),
                })?
            }
        };
        let observed_genotype_value_count = match &self.genotype_buffer {
            OwnedGenotypeBuffer::Dosage(values) => values.len(),
            OwnedGenotypeBuffer::Packed8(values) => values.len(),
        };
        if observed_genotype_value_count != expected_genotype_value_count {
            return Err(SchedulerError::InvalidBatch {
                message: format!(
                    "genotype buffer contains {observed_genotype_value_count} values, expected {expected_genotype_value_count}"
                ),
            });
        }
        validate_variant_column_length("metadata", self.metadata.row_count(), self.variant_count)?;
        validate_variant_column_length(
            "allele one frequency",
            self.statistics.output.allele_one_frequency.len(),
            self.variant_count,
        )?;
        validate_variant_column_length(
            "INFO score",
            self.statistics.output.info_score.values.len(),
            self.variant_count,
        )?;
        validate_variant_column_length(
            "genotype mean",
            self.statistics.compute.genotype_mean.len(),
            self.variant_count,
        )?;
        validate_variant_column_length(
            "observation count",
            self.statistics.output.observation_count.len(),
            self.variant_count,
        )?;
        if let Some(imputed_dosage_square_sum) = self.statistics.compute.imputed_dosage_square_sum.as_ref() {
            validate_variant_column_length(
                "imputed dosage square sum",
                imputed_dosage_square_sum.len(),
                self.variant_count,
            )?;
        }
        if let Some(sparse_candidate_mask) = self.statistics.compute.sparse_candidate_mask.as_ref() {
            validate_variant_column_length(
                "rare sparse Firth candidate mask",
                sparse_candidate_mask.len(),
                self.variant_count,
            )?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct AssociationBatchOutput {
    pub variant_start_index: usize,
    pub metadata: NativeVariantMetadataHandle,
    pub statistics: ChunkOutputStatistics,
    pub active_trait_selection: ActiveTraitSelection,
}

/// One completed host result and the resources that produced it.
#[derive(Debug)]
pub(crate) struct CompletedAssociationBatch {
    pub(crate) group_index: usize,
    pub(crate) output: AssociationBatchOutput,
    pub(crate) result: HostAssociationBatch,
}

#[derive(Debug)]
// Keeping completed batches inline avoids one allocation on every hot-path
// result; the small release event occurs only at drained transitions.
#[allow(clippy::large_enum_variant)]
enum AssociationSchedulerEvent {
    Completed(CompletedAssociationBatch),
    ChromosomeReleased { group_index: usize },
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum SchedulerError<BackendError> {
    #[error("association scheduler {queue} capacity must be positive, observed {capacity}")]
    InvalidCapacity { queue: &'static str, capacity: usize },
    #[error("invalid scheduled association batch: {message}")]
    InvalidBatch { message: String },
    #[error("association backend failed during {stage}: {source}")]
    Backend {
        stage: &'static str,
        #[source]
        source: BackendError,
    },
    #[error("association scheduler {worker} worker could not start: {message}")]
    WorkerSpawn { worker: &'static str, message: String },
    #[error("association scheduler {worker} worker panicked: {message}")]
    WorkerPanicked { worker: &'static str, message: String },
    #[error("association scheduler {queue} channel disconnected unexpectedly")]
    ChannelDisconnected { queue: &'static str },
    #[error("association scheduler is closed")]
    Closed,
    #[error("association scheduler requires a prepared chromosome before batch submission")]
    ChromosomeNotPrepared,
    #[error("association scheduler group index {group_index} is not registered")]
    UnknownGroup { group_index: usize },
    #[error("association scheduler group {group_index} already has a prepared chromosome")]
    ChromosomeAlreadyPrepared { group_index: usize },
    #[error("association scheduler group {group_index} chromosome release is already pending")]
    ChromosomeReleasePending { group_index: usize },
    #[error(
        "association scheduler cannot change chromosomes with batches pending: {submitted} submitted, {completed} completed"
    )]
    ChromosomeTransitionPending { submitted: usize, completed: usize },
    #[error("association scheduler has no submitted batch available to receive")]
    NoPendingBatch,
    #[error("association scheduler submitted-batch counter overflowed")]
    BatchCounterOverflow,
    #[error("association scheduler submission must be closed before joining workers")]
    SubmissionOpen,
    #[error(
        "association scheduler batches must be drained before joining workers: {submitted} submitted, {completed} completed"
    )]
    PendingBatches { submitted: usize, completed: usize },
    #[error("association scheduler was aborted")]
    Aborted,
}

type SchedulerResult<Value, BackendError> = Result<Value, SchedulerError<BackendError>>;

struct DeviceAssociationBatch<DeviceResult> {
    group_index: usize,
    output: AssociationBatchOutput,
    device_result: DeviceResult,
}

// Keeping batches inline lets the bounded channel allocate its storage once;
// boxing the hot variant would add one allocation to every submitted batch.
#[allow(clippy::large_enum_variant)]
enum ComputeCommand<ChromosomeState> {
    PrepareChromosome { group_index: usize, state: ChromosomeState },
    ReleaseChromosome { group_index: usize },
    ComputeBatch { group_index: usize, batch: ScheduledAssociationBatch },
}

#[derive(Debug, Default)]
struct GroupSchedulerState {
    submitted_batch_count: usize,
    completed_batch_count: usize,
    chromosome_prepared: bool,
    chromosome_release_pending: bool,
}

#[derive(Debug)]
struct SchedulerControl<BackendError> {
    aborted: std::sync::atomic::AtomicBool,
    first_error: Mutex<Option<SchedulerError<BackendError>>>,
}

impl<BackendError> SchedulerControl<BackendError> {
    const fn new() -> Self {
        Self { aborted: std::sync::atomic::AtomicBool::new(false), first_error: Mutex::new(None) }
    }

    fn abort(&self) {
        self.aborted.store(true, std::sync::atomic::Ordering::Release);
    }

    fn is_aborted(&self) -> bool {
        self.aborted.load(std::sync::atomic::Ordering::Acquire)
    }

    fn record_error(&self, error: SchedulerError<BackendError>) {
        {
            let mut first_error = self.first_error.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if first_error.is_none() {
                *first_error = Some(error);
            }
        }
        self.abort();
    }

    fn take_error(&self) -> Option<SchedulerError<BackendError>> {
        self.first_error.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take()
    }
}

/// One bounded decoded-to-device-to-host pipeline shared by phenotype groups.
pub(crate) struct AssociationBatchPipeline<Backend>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    compute_sender: Option<Sender<ComputeCommand<Backend::ChromosomeState>>>,
    event_receiver: Receiver<AssociationSchedulerEvent>,
    compute_worker: Option<thread::JoinHandle<()>>,
    materialization_worker: Option<thread::JoinHandle<()>>,
    control: Arc<SchedulerControl<Backend::Error>>,
    groups: Vec<GroupSchedulerState>,
}

impl<Backend> AssociationBatchPipeline<Backend>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    /// Start one shared two-stage pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error when either capacity is zero or a worker thread cannot
    /// be started.
    pub(crate) fn new(
        backend: Arc<Backend>,
        staging_depth: usize,
        result_in_flight_limit: usize,
    ) -> SchedulerResult<Self, Backend::Error> {
        validate_capacity(DECODED_BATCH_QUEUE, staging_depth)?;
        validate_capacity(DEVICE_RESULT_QUEUE, result_in_flight_limit)?;

        let (compute_sender, compute_receiver) = crossbeam_channel::bounded(staging_depth);
        let (device_sender, device_receiver) = crossbeam_channel::bounded(result_in_flight_limit);
        let (event_sender, event_receiver) = crossbeam_channel::bounded(result_in_flight_limit);
        let control = Arc::new(SchedulerControl::new());

        let materialization_backend = Arc::clone(&backend);
        let materialization_control = Arc::clone(&control);
        let materialization_event_sender = event_sender.clone();
        let materialization_worker = thread::Builder::new()
            .name(MATERIALIZATION_WORKER_NAME.to_string())
            .spawn(move || {
                let worker_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_materialization_worker(
                        materialization_backend.as_ref(),
                        &device_receiver,
                        &materialization_event_sender,
                        &materialization_control,
                    );
                }));
                if let Err(payload) = worker_result {
                    materialization_control.record_error(SchedulerError::WorkerPanicked {
                        worker: MATERIALIZATION_WORKER,
                        message: panic_message(payload.as_ref()),
                    });
                }
            })
            .map_err(|source| SchedulerError::WorkerSpawn {
                worker: MATERIALIZATION_WORKER,
                message: source.to_string(),
            })?;

        let compute_control = Arc::clone(&control);
        let compute_event_sender = event_sender;
        let compute_worker = match thread::Builder::new().name(COMPUTE_WORKER_NAME.to_string()).spawn(move || {
            let worker_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_compute_worker(
                    backend.as_ref(),
                    &compute_receiver,
                    &device_sender,
                    &compute_event_sender,
                    &compute_control,
                );
            }));
            if let Err(payload) = worker_result {
                compute_control.record_error(SchedulerError::WorkerPanicked {
                    worker: COMPUTE_WORKER,
                    message: panic_message(payload.as_ref()),
                });
            }
        }) {
            Ok(worker) => worker,
            Err(source) => {
                control.abort();
                drop(compute_sender);
                let _ = materialization_worker.join();
                return Err(SchedulerError::WorkerSpawn { worker: COMPUTE_WORKER, message: source.to_string() });
            }
        };

        Ok(Self {
            compute_sender: Some(compute_sender),
            event_receiver,
            compute_worker: Some(compute_worker),
            materialization_worker: Some(materialization_worker),
            control,
            groups: Vec::new(),
        })
    }

    /// Register one independently counted phenotype group.
    ///
    /// # Errors
    ///
    /// Returns an error when the pipeline has been closed, aborted, or failed.
    pub(crate) fn register_group(&mut self) -> SchedulerResult<usize, Backend::Error> {
        self.ensure_running()?;
        let group_index = self.groups.len();
        self.groups.push(GroupSchedulerState::default());
        Ok(group_index)
    }

    /// Install an owned chromosome state after the shared pipeline is drained.
    ///
    /// # Errors
    ///
    /// Returns an error when prior batches remain pending or the pipeline has
    /// been closed, aborted, or failed.
    pub(crate) fn prepare_chromosome(
        &mut self,
        group_index: usize,
        chromosome_state: Backend::ChromosomeState,
    ) -> SchedulerResult<(), Backend::Error> {
        self.ensure_running()?;
        if !self.is_drained() {
            let (submitted, completed) = self.batch_counts();
            return Err(SchedulerError::ChromosomeTransitionPending { submitted, completed });
        }
        let group = self.group(group_index)?;
        if group.chromosome_release_pending {
            return Err(SchedulerError::ChromosomeReleasePending { group_index });
        }
        if group.chromosome_prepared {
            return Err(SchedulerError::ChromosomeAlreadyPrepared { group_index });
        }
        let sender = self.compute_sender.as_ref().ok_or(SchedulerError::Closed)?;
        sender
            .send(ComputeCommand::PrepareChromosome { group_index, state: chromosome_state })
            .map_err(|_| self.current_or_channel_error(DECODED_BATCH_QUEUE))?;
        self.group_mut(group_index)?.chromosome_prepared = true;
        Ok(())
    }

    /// Release one prepared chromosome and wait until backend destruction has completed.
    ///
    /// # Errors
    ///
    /// Returns an error unless every submitted group batch has been received,
    /// or when the pipeline has been closed, aborted, or failed.
    pub(crate) fn release_chromosome(&mut self, group_index: usize) -> SchedulerResult<(), Backend::Error> {
        self.ensure_running()?;
        if !self.is_drained() {
            let (submitted, completed) = self.batch_counts();
            return Err(SchedulerError::ChromosomeTransitionPending { submitted, completed });
        }
        let group = self.group(group_index)?;
        if group.chromosome_release_pending {
            return Err(SchedulerError::ChromosomeReleasePending { group_index });
        }
        if !group.chromosome_prepared {
            return Ok(());
        }
        let sender = self.compute_sender.as_ref().ok_or(SchedulerError::Closed)?;
        sender
            .send(ComputeCommand::ReleaseChromosome { group_index })
            .map_err(|_| self.current_or_channel_error(DECODED_BATCH_QUEUE))?;
        let group = self.group_mut(group_index)?;
        group.chromosome_prepared = false;
        group.chromosome_release_pending = true;
        match self.receive_scheduler_event()? {
            AssociationSchedulerEvent::ChromosomeReleased { group_index: released_group_index }
                if released_group_index == group_index =>
            {
                Ok(())
            }
            AssociationSchedulerEvent::ChromosomeReleased { group_index: released_group_index } => {
                Err(SchedulerError::InvalidBatch {
                    message: format!(
                        "received chromosome release for group {released_group_index} while waiting for group {group_index}"
                    ),
                })
            }
            AssociationSchedulerEvent::Completed(_) => Err(SchedulerError::InvalidBatch {
                message: "received a completed batch after the drained chromosome-release barrier".to_string(),
            }),
        }
    }

    /// Release every registered chromosome after the shared pipeline is drained.
    ///
    /// # Errors
    ///
    /// Returns an error when a release fails or pending batches remain.
    pub(crate) fn release_all_chromosomes(&mut self) -> SchedulerResult<(), Backend::Error> {
        for group_index in 0..self.groups.len() {
            self.release_chromosome(group_index)?;
        }
        Ok(())
    }

    /// Try to submit one owned decoded batch without blocking behind work.
    ///
    /// The first batch after a chromosome transition may briefly block until
    /// the compute worker consumes the preceding state command. Subsequent
    /// backpressure returns ownership to the caller so it can drain completed
    /// output before retrying.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch shape is invalid, the pipeline has been
    /// closed or aborted, or a worker has failed.
    pub(crate) fn try_submit(
        &mut self,
        group_index: usize,
        batch: ScheduledAssociationBatch,
    ) -> SchedulerResult<Option<ScheduledAssociationBatch>, Backend::Error> {
        batch.validate::<Backend::Error>()?;
        self.ensure_running()?;
        let group = self.group(group_index)?;
        if !group.chromosome_prepared || group.chromosome_release_pending {
            return Err(SchedulerError::ChromosomeNotPrepared);
        }
        let next_submitted_batch_count =
            group.submitted_batch_count.checked_add(1).ok_or(SchedulerError::BatchCounterOverflow)?;
        let sender = self.compute_sender.as_ref().ok_or(SchedulerError::Closed)?;
        let command = ComputeCommand::ComputeBatch { group_index, batch };
        if self.is_drained() {
            sender.send(command).map_err(|_| self.current_or_channel_error(DECODED_BATCH_QUEUE))?;
            self.group_mut(group_index)?.submitted_batch_count = next_submitted_batch_count;
            return Ok(None);
        }
        match sender.try_send(command) {
            Ok(()) => {
                self.group_mut(group_index)?.submitted_batch_count = next_submitted_batch_count;
                Ok(None)
            }
            Err(TrySendError::Full(ComputeCommand::ComputeBatch { batch, .. })) => Ok(Some(batch)),
            Err(TrySendError::Full(
                ComputeCommand::PrepareChromosome { .. } | ComputeCommand::ReleaseChromosome { .. },
            )) => {
                unreachable!("batch submission cannot return a chromosome command")
            }
            Err(TrySendError::Disconnected(_)) => Err(self.current_or_channel_error(DECODED_BATCH_QUEUE)),
        }
    }

    /// Receive the next submitted batch.
    ///
    /// # Errors
    ///
    /// Returns the first worker or backend error, or reports an explicit abort.
    pub(crate) fn receive(&mut self) -> SchedulerResult<CompletedAssociationBatch, Backend::Error> {
        match self.receive_scheduler_event()? {
            AssociationSchedulerEvent::Completed(completed_batch) => Ok(completed_batch),
            AssociationSchedulerEvent::ChromosomeReleased { group_index } => Err(SchedulerError::InvalidBatch {
                message: format!("unexpected chromosome release for group {group_index} outside a drained transition"),
            }),
        }
    }

    /// Try to receive one completed batch without blocking.
    ///
    /// # Errors
    ///
    /// Returns the first worker or backend error, or reports an explicit abort.
    pub(crate) fn try_receive(&mut self) -> SchedulerResult<Option<CompletedAssociationBatch>, Backend::Error> {
        match self.try_receive_scheduler_event()? {
            Some(AssociationSchedulerEvent::Completed(completed_batch)) => Ok(Some(completed_batch)),
            Some(AssociationSchedulerEvent::ChromosomeReleased { group_index }) => Err(SchedulerError::InvalidBatch {
                message: format!("unexpected chromosome release for group {group_index} outside a drained transition"),
            }),
            None => Ok(None),
        }
    }

    fn receive_scheduler_event(&mut self) -> SchedulerResult<AssociationSchedulerEvent, Backend::Error> {
        if let Some(error) = self.control.take_error() {
            return Err(error);
        }
        if self.control.is_aborted() {
            return Err(SchedulerError::Aborted);
        }
        if !self.has_pending_event() {
            return Err(SchedulerError::NoPendingBatch);
        }
        match self.event_receiver.recv() {
            Ok(event) => self.record_event(event),
            Err(_) => match self.control.take_error() {
                Some(error) => Err(error),
                None if self.control.is_aborted() => Err(SchedulerError::Aborted),
                None => Err(SchedulerError::ChannelDisconnected { queue: COMPLETED_BATCH_QUEUE }),
            },
        }
    }

    fn try_receive_scheduler_event(&mut self) -> SchedulerResult<Option<AssociationSchedulerEvent>, Backend::Error> {
        if let Some(error) = self.control.take_error() {
            return Err(error);
        }
        if self.control.is_aborted() {
            return Err(SchedulerError::Aborted);
        }
        match self.event_receiver.try_recv() {
            Ok(event) => self.record_event(event).map(Some),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => match self.control.take_error() {
                Some(error) => Err(error),
                None if self.control.is_aborted() => Err(SchedulerError::Aborted),
                None if !self.has_pending_event() && self.compute_sender.is_none() => Ok(None),
                None => Err(SchedulerError::ChannelDisconnected { queue: COMPLETED_BATCH_QUEUE }),
            },
        }
    }

    /// Return whether every submitted batch has been received by the caller.
    #[must_use]
    pub(crate) fn is_drained(&self) -> bool {
        self.groups.iter().all(|group| group.submitted_batch_count == group.completed_batch_count)
    }

    /// Close the decoded-batch input so queued work can complete.
    pub(crate) fn close_submission(&mut self) {
        for group in &mut self.groups {
            group.chromosome_prepared = false;
            group.chromosome_release_pending = false;
        }
        self.compute_sender.take();
    }

    /// Join compute followed by materialization after all results were drained.
    ///
    /// # Errors
    ///
    /// Returns the first worker or backend error, including a worker panic.
    pub(crate) fn join(&mut self) -> SchedulerResult<(), Backend::Error> {
        if self.compute_sender.is_some() {
            return Err(SchedulerError::SubmissionOpen);
        }
        if !self.is_drained() {
            let (submitted, completed) = self.batch_counts();
            return Err(SchedulerError::PendingBatches { submitted, completed });
        }
        self.join_workers();
        match self.control.take_error() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Cancel queued work, close submission, and join both workers.
    ///
    /// # Errors
    ///
    /// Returns a worker or backend error observed before or during shutdown.
    pub(crate) fn abort(&mut self) -> SchedulerResult<(), Backend::Error> {
        self.control.abort();
        for group in &mut self.groups {
            group.chromosome_prepared = false;
            group.chromosome_release_pending = false;
        }
        self.compute_sender.take();
        self.join_workers();
        match self.control.take_error() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn ensure_running(&self) -> SchedulerResult<(), Backend::Error> {
        if let Some(error) = self.control.take_error() {
            return Err(error);
        }
        if self.control.is_aborted() {
            return Err(SchedulerError::Aborted);
        }
        if self.compute_sender.is_none() {
            return Err(SchedulerError::Closed);
        }
        Ok(())
    }

    fn current_or_channel_error(&self, queue: &'static str) -> SchedulerError<Backend::Error> {
        self.control.take_error().unwrap_or(SchedulerError::ChannelDisconnected { queue })
    }

    fn group(&self, group_index: usize) -> SchedulerResult<&GroupSchedulerState, Backend::Error> {
        self.groups.get(group_index).ok_or(SchedulerError::UnknownGroup { group_index })
    }

    fn group_mut(&mut self, group_index: usize) -> SchedulerResult<&mut GroupSchedulerState, Backend::Error> {
        self.groups.get_mut(group_index).ok_or(SchedulerError::UnknownGroup { group_index })
    }

    fn batch_counts(&self) -> (usize, usize) {
        self.groups.iter().fold((0_usize, 0_usize), |(submitted, completed), group| {
            (
                submitted.saturating_add(group.submitted_batch_count),
                completed.saturating_add(group.completed_batch_count),
            )
        })
    }

    fn has_pending_event(&self) -> bool {
        self.groups
            .iter()
            .any(|group| group.submitted_batch_count != group.completed_batch_count || group.chromosome_release_pending)
    }

    fn record_event(
        &mut self,
        event: AssociationSchedulerEvent,
    ) -> SchedulerResult<AssociationSchedulerEvent, Backend::Error> {
        match &event {
            AssociationSchedulerEvent::Completed(batch) => {
                let group = self.group_mut(batch.group_index)?;
                if group.completed_batch_count >= group.submitted_batch_count {
                    return Err(SchedulerError::InvalidBatch {
                        message: format!(
                            "group {} produced more completed batches than were submitted",
                            batch.group_index
                        ),
                    });
                }
                group.completed_batch_count =
                    group.completed_batch_count.checked_add(1).ok_or(SchedulerError::BatchCounterOverflow)?;
            }
            AssociationSchedulerEvent::ChromosomeReleased { group_index } => {
                let group = self.group_mut(*group_index)?;
                if !group.chromosome_release_pending {
                    return Err(SchedulerError::InvalidBatch {
                        message: format!("unexpected chromosome release for group {group_index}"),
                    });
                }
                group.chromosome_release_pending = false;
            }
        }
        Ok(event)
    }

    fn join_workers(&mut self) {
        if let Some(worker) = self.compute_worker.take()
            && let Err(payload) = worker.join()
        {
            self.control.record_error(SchedulerError::WorkerPanicked {
                worker: COMPUTE_WORKER,
                message: panic_message(payload.as_ref()),
            });
        }
        if let Some(worker) = self.materialization_worker.take()
            && let Err(payload) = worker.join()
        {
            self.control.record_error(SchedulerError::WorkerPanicked {
                worker: MATERIALIZATION_WORKER,
                message: panic_message(payload.as_ref()),
            });
        }
    }
}

impl<Backend> Drop for AssociationBatchPipeline<Backend>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    fn drop(&mut self) {
        self.control.abort();
        self.compute_sender.take();
        self.join_workers();
    }
}

fn run_compute_worker<Backend>(
    backend: &Backend,
    compute_receiver: &Receiver<ComputeCommand<Backend::ChromosomeState>>,
    device_sender: &Sender<DeviceAssociationBatch<Backend::DeviceResult>>,
    event_sender: &Sender<AssociationSchedulerEvent>,
    control: &SchedulerControl<Backend::Error>,
) where
    Backend: AssociationBackend,
{
    let mut chromosome_states = Vec::<Option<Backend::ChromosomeState>>::new();
    while let Ok(command) = compute_receiver.recv() {
        if control.is_aborted() {
            release_command_state(backend, command);
            break;
        }
        let (group_index, scheduled_batch) = match command {
            ComputeCommand::PrepareChromosome { group_index, state } => {
                if chromosome_states.len() <= group_index {
                    chromosome_states.resize_with(group_index + 1, || None);
                }
                if chromosome_states[group_index].is_some() {
                    backend.release_chromosome(state);
                    control.record_error(SchedulerError::ChromosomeAlreadyPrepared { group_index });
                    break;
                }
                chromosome_states[group_index] = Some(state);
                continue;
            }
            ComputeCommand::ReleaseChromosome { group_index } => {
                let Some(chromosome_state) = chromosome_states.get_mut(group_index).and_then(Option::take) else {
                    control.record_error(SchedulerError::ChromosomeNotPrepared);
                    break;
                };
                backend.release_chromosome(chromosome_state);
                if !send_scheduler_event(
                    event_sender,
                    AssociationSchedulerEvent::ChromosomeReleased { group_index },
                    control,
                ) {
                    break;
                }
                continue;
            }
            ComputeCommand::ComputeBatch { group_index, batch } => (group_index, batch),
        };
        let Some(active_chromosome_state) = chromosome_states.get(group_index).and_then(Option::as_ref) else {
            control.record_error(SchedulerError::ChromosomeNotPrepared);
            break;
        };
        let ScheduledAssociationBatch {
            variant_start_index,
            variant_count,
            sample_count,
            metadata,
            statistics,
            genotype_buffer,
            active_trait_selection,
        } = scheduled_batch;
        let ChunkStats { output: output_statistics, compute: compute_statistics } = statistics;
        let input = GenotypeBatchInput {
            variant_count,
            sample_count,
            genotypes: genotype_buffer,
            statistics: GenotypeBatchStatistics {
                genotype_mean: compute_statistics.genotype_mean,
                imputed_dosage_square_sum: compute_statistics.imputed_dosage_square_sum,
                sparse_candidate_mask: compute_statistics.sparse_candidate_mask,
            },
        };
        let device_result = match backend.compute_batch(active_chromosome_state, input) {
            Ok(result) => result,
            Err(source) => {
                control.record_error(SchedulerError::Backend { stage: COMPUTE_WORKER, source });
                break;
            }
        };
        if control.is_aborted() {
            break;
        }
        let output = AssociationBatchOutput {
            variant_start_index,
            metadata,
            statistics: output_statistics,
            active_trait_selection,
        };
        if device_sender.send(DeviceAssociationBatch { group_index, output, device_result }).is_err() {
            if !control.is_aborted() {
                control.record_error(SchedulerError::ChannelDisconnected { queue: DEVICE_RESULT_QUEUE });
            }
            break;
        }
    }
    for chromosome_state in chromosome_states.into_iter().flatten() {
        backend.release_chromosome(chromosome_state);
    }
    while let Ok(command) = compute_receiver.try_recv() {
        release_command_state(backend, command);
    }
}

fn run_materialization_worker<Backend>(
    backend: &Backend,
    device_receiver: &Receiver<DeviceAssociationBatch<Backend::DeviceResult>>,
    event_sender: &Sender<AssociationSchedulerEvent>,
    control: &SchedulerControl<Backend::Error>,
) where
    Backend: AssociationBackend,
{
    while let Ok(device_batch) = device_receiver.recv() {
        if control.is_aborted() {
            break;
        }
        let active_trait_indices = match &device_batch.output.active_trait_selection {
            ActiveTraitSelection::All => None,
            ActiveTraitSelection::Indices(indices) => Some(indices.as_slice()),
        };
        let result = match backend.materialize_batch(device_batch.device_result, active_trait_indices) {
            Ok(result) => result,
            Err(source) => {
                control.record_error(SchedulerError::Backend { stage: MATERIALIZATION_WORKER, source });
                break;
            }
        };
        let event = AssociationSchedulerEvent::Completed(CompletedAssociationBatch {
            group_index: device_batch.group_index,
            output: device_batch.output,
            result,
        });
        if !send_scheduler_event(event_sender, event, control) {
            return;
        }
    }
}

fn send_scheduler_event<BackendError>(
    sender: &Sender<AssociationSchedulerEvent>,
    event: AssociationSchedulerEvent,
    control: &SchedulerControl<BackendError>,
) -> bool {
    let mut pending_event = event;
    loop {
        if control.is_aborted() {
            return false;
        }
        match sender.send_timeout(pending_event, Duration::from_millis(10)) {
            Ok(()) => return true,
            Err(SendTimeoutError::Timeout(returned_event)) => pending_event = returned_event,
            Err(SendTimeoutError::Disconnected(_)) => {
                if !control.is_aborted() {
                    control.record_error(SchedulerError::ChannelDisconnected { queue: COMPLETED_BATCH_QUEUE });
                }
                return false;
            }
        }
    }
}

fn release_command_state<Backend>(backend: &Backend, command: ComputeCommand<Backend::ChromosomeState>)
where
    Backend: AssociationBackend,
{
    if let ComputeCommand::PrepareChromosome { state, .. } = command {
        backend.release_chromosome(state);
    }
}

fn validate_capacity<BackendError>(queue: &'static str, capacity: usize) -> SchedulerResult<(), BackendError> {
    if capacity == 0 {
        return Err(SchedulerError::InvalidCapacity { queue, capacity });
    }
    Ok(())
}

fn validate_variant_column_length<BackendError>(
    name: &str,
    observed: usize,
    expected: usize,
) -> SchedulerResult<(), BackendError> {
    if observed != expected {
        return Err(SchedulerError::InvalidBatch {
            message: format!("{name} contains {observed} values, expected {expected}"),
        });
    }
    Ok(())
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    payload.downcast_ref::<&str>().map_or_else(
        || payload.downcast_ref::<String>().cloned().unwrap_or_else(|| "unknown panic payload".to_string()),
        |message| (*message).to_string(),
    )
}
