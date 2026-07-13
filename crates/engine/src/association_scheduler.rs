//! Bounded two-stage association batch scheduler.

use std::any::Any;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::backend::{AssociationBackend, GenotypeBatchInput};
use crossbeam_channel::{Receiver, RecvTimeoutError, SendTimeoutError, Sender, TryRecvError, TrySendError};
use g_genotype::{ChunkStats, DecodedGenotypeBatch, OwnedGenotypeBuffer};
use g_genotype_contracts::ChunkOutputStatistics;
use g_output::NativeVariantMetadataHandle;

use crate::output_schedule::ActiveTraitSelection;

const COMPUTE_WORKER_NAME: &str = "g-association-compute";
const MATERIALIZATION_WORKER_NAME: &str = "g-association-materialization";
const DECODED_BATCH_CAPACITY: usize = 1;
const DEVICE_RESULT_CAPACITY: usize = 2;
const DECODED_BATCH_QUEUE: &str = "decoded batch";
const DEVICE_RESULT_QUEUE: &str = "device result";
const COMPLETED_BATCH_QUEUE: &str = "completed batch";
const COMPUTE_WORKER: &str = "compute";
const MATERIALIZATION_WORKER: &str = "materialization";
const CHANNEL_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// One owned association batch ready for device compute.
#[derive(Debug)]
pub(crate) struct ScheduledAssociationBatch {
    pub(crate) decoded: DecodedGenotypeBatch,
    pub(crate) metadata: NativeVariantMetadataHandle,
    pub(crate) active_trait_selection: ActiveTraitSelection,
}

impl ScheduledAssociationBatch {
    fn validate<BackendError>(&self) -> SchedulerResult<(), BackendError> {
        let decoded = &self.decoded;
        if self.metadata.row_count() != decoded.logical_variant_count {
            return Err(SchedulerError::InvalidBatch {
                message: format!(
                    "metadata contains {} variants, decoded batch contains {}",
                    self.metadata.row_count(),
                    decoded.logical_variant_count
                ),
            });
        }
        if decoded.compute_variant_count < decoded.logical_variant_count {
            return Err(SchedulerError::InvalidBatch {
                message: format!(
                    "compute variant count {} is smaller than logical variant count {}",
                    decoded.compute_variant_count, decoded.logical_variant_count
                ),
            });
        }
        let genotype_value_count =
            decoded.compute_variant_count.checked_mul(decoded.sample_count).ok_or_else(|| {
                SchedulerError::InvalidBatch {
                    message: "variant and sample counts overflow the host index width".to_string(),
                }
            })?;
        let expected_genotype_value_count = match &decoded.genotypes {
            OwnedGenotypeBuffer::Dosage(_) => genotype_value_count,
            OwnedGenotypeBuffer::Packed8(_) => {
                genotype_value_count.checked_mul(2).ok_or_else(|| SchedulerError::InvalidBatch {
                    message: "packed8 genotype dimensions overflow the host index width".to_string(),
                })?
            }
        };
        let observed_genotype_value_count = match &decoded.genotypes {
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
        validate_variant_column_length(
            "genotype mean",
            decoded.statistics.compute.genotype_mean.len(),
            decoded.compute_variant_count,
        )?;
        if let Some(imputed_dosage_square_sum) = decoded.statistics.compute.imputed_dosage_square_sum.as_ref() {
            validate_variant_column_length(
                "imputed dosage square sum",
                imputed_dosage_square_sum.len(),
                decoded.compute_variant_count,
            )?;
        }
        if let Some(sparse_candidate_mask) = decoded.statistics.compute.sparse_candidate_mask.as_ref() {
            validate_variant_column_length(
                "rare sparse Firth candidate mask",
                sparse_candidate_mask.len(),
                decoded.compute_variant_count,
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
    pub(crate) output: AssociationBatchOutput,
    pub(crate) result: g_output::Regenie2StatisticBatch,
}

#[derive(Debug)]
// Keeping completed batches inline avoids one allocation on every hot-path
// result; the small release event occurs only at drained transitions.
#[allow(clippy::large_enum_variant)]
enum AssociationSchedulerEvent {
    Completed(CompletedAssociationBatch),
    ChromosomeReleased,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum SchedulerError<BackendError> {
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
    #[error("association scheduler already has a prepared chromosome")]
    ChromosomeAlreadyPrepared,
    #[error("association scheduler chromosome release is already pending")]
    ChromosomeReleasePending,
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
    output: AssociationBatchOutput,
    device_result: DeviceResult,
}

// Keeping batches inline lets the bounded channel allocate its storage once;
// boxing the hot variant would add one allocation to every submitted batch.
#[allow(clippy::large_enum_variant)]
enum ComputeCommand<ChromosomeState> {
    PrepareChromosome { state: ChromosomeState },
    ReleaseChromosome,
    ComputeBatch { batch: ScheduledAssociationBatch },
}

#[derive(Debug, Default)]
struct PipelineSchedulerState {
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

/// One bounded decoded-to-device-to-host delivery pipeline.
pub(crate) struct AssociationBatchPipeline<Backend>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    backend: Arc<Backend>,
    compute_sender: Option<Sender<ComputeCommand<Backend::ChromosomeState>>>,
    event_receiver: Receiver<AssociationSchedulerEvent>,
    compute_worker: Option<thread::JoinHandle<()>>,
    materialization_worker: Option<thread::JoinHandle<()>>,
    control: Arc<SchedulerControl<Backend::Error>>,
    state: PipelineSchedulerState,
}

impl<Backend> AssociationBatchPipeline<Backend>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    /// Start one two-stage pipeline with the engine's bounded capacities.
    ///
    /// # Errors
    ///
    /// Returns an error when a worker thread cannot be started.
    pub(crate) fn new(backend: Arc<Backend>) -> SchedulerResult<Self, Backend::Error> {
        let (compute_sender, compute_receiver) = crossbeam_channel::bounded(DECODED_BATCH_CAPACITY);
        let (device_sender, device_receiver) = crossbeam_channel::bounded(DEVICE_RESULT_CAPACITY);
        let (event_sender, event_receiver) = crossbeam_channel::bounded(DEVICE_RESULT_CAPACITY);
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
        let pipeline_backend = Arc::clone(&backend);
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
                drain_command_states(backend.as_ref(), &compute_receiver);
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
            backend: pipeline_backend,
            compute_sender: Some(compute_sender),
            event_receiver,
            compute_worker: Some(compute_worker),
            materialization_worker: Some(materialization_worker),
            control,
            state: PipelineSchedulerState::default(),
        })
    }

    /// Install an owned chromosome state after the shared pipeline is drained.
    ///
    /// # Errors
    ///
    /// Returns an error when prior batches remain pending or the pipeline has
    /// been closed, aborted, or failed.
    pub(crate) fn prepare_chromosome(
        &mut self,
        chromosome_state: Backend::ChromosomeState,
    ) -> SchedulerResult<(), Backend::Error> {
        self.ensure_running()?;
        if !self.is_drained() {
            let (submitted, completed) = self.batch_counts();
            return Err(SchedulerError::ChromosomeTransitionPending { submitted, completed });
        }
        if self.state.chromosome_release_pending {
            return Err(SchedulerError::ChromosomeReleasePending);
        }
        if self.state.chromosome_prepared {
            return Err(SchedulerError::ChromosomeAlreadyPrepared);
        }
        let sender = self.compute_sender.as_ref().ok_or(SchedulerError::Closed)?;
        if let Err(error) = sender.send(ComputeCommand::PrepareChromosome { state: chromosome_state }) {
            let scheduler_error = self.current_or_channel_error(DECODED_BATCH_QUEUE);
            release_command_state(self.backend.as_ref(), error.0);
            return Err(scheduler_error);
        }
        self.state.chromosome_prepared = true;
        Ok(())
    }

    /// Release one prepared chromosome and wait until backend destruction has completed.
    ///
    /// # Errors
    ///
    /// Returns an error unless every submitted group batch has been received,
    /// or when the pipeline has been closed, aborted, or failed.
    pub(crate) fn release_chromosome(&mut self) -> SchedulerResult<(), Backend::Error> {
        self.ensure_running()?;
        if !self.is_drained() {
            let (submitted, completed) = self.batch_counts();
            return Err(SchedulerError::ChromosomeTransitionPending { submitted, completed });
        }
        if self.state.chromosome_release_pending {
            return Err(SchedulerError::ChromosomeReleasePending);
        }
        if !self.state.chromosome_prepared {
            return Ok(());
        }
        let sender = self.compute_sender.as_ref().ok_or(SchedulerError::Closed)?;
        sender
            .send(ComputeCommand::ReleaseChromosome)
            .map_err(|_| self.current_or_channel_error(DECODED_BATCH_QUEUE))?;
        self.state.chromosome_prepared = false;
        self.state.chromosome_release_pending = true;
        match self.receive_scheduler_event()? {
            AssociationSchedulerEvent::ChromosomeReleased => Ok(()),
            AssociationSchedulerEvent::Completed(_) => Err(SchedulerError::InvalidBatch {
                message: "received a completed batch after the drained chromosome-release barrier".to_string(),
            }),
        }
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
        batch: ScheduledAssociationBatch,
    ) -> SchedulerResult<Option<ScheduledAssociationBatch>, Backend::Error> {
        batch.validate::<Backend::Error>()?;
        self.ensure_running()?;
        if !self.state.chromosome_prepared || self.state.chromosome_release_pending {
            return Err(SchedulerError::ChromosomeNotPrepared);
        }
        let next_submitted_batch_count =
            self.state.submitted_batch_count.checked_add(1).ok_or(SchedulerError::BatchCounterOverflow)?;
        let sender = self.compute_sender.as_ref().ok_or(SchedulerError::Closed)?;
        let command = ComputeCommand::ComputeBatch { batch };
        if self.is_drained() {
            sender.send(command).map_err(|_| self.current_or_channel_error(DECODED_BATCH_QUEUE))?;
            self.state.submitted_batch_count = next_submitted_batch_count;
            return Ok(None);
        }
        match sender.try_send(command) {
            Ok(()) => {
                self.state.submitted_batch_count = next_submitted_batch_count;
                Ok(None)
            }
            Err(TrySendError::Full(ComputeCommand::ComputeBatch { batch })) => Ok(Some(batch)),
            Err(TrySendError::Full(ComputeCommand::PrepareChromosome { .. } | ComputeCommand::ReleaseChromosome)) => {
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
            AssociationSchedulerEvent::ChromosomeReleased => Err(SchedulerError::InvalidBatch {
                message: "unexpected chromosome release outside a drained transition".to_string(),
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
            Some(AssociationSchedulerEvent::ChromosomeReleased) => Err(SchedulerError::InvalidBatch {
                message: "unexpected chromosome release outside a drained transition".to_string(),
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
        loop {
            match self.event_receiver.recv_timeout(CHANNEL_POLL_INTERVAL) {
                Ok(event) => return self.record_event(event),
                Err(RecvTimeoutError::Timeout) => {
                    if let Some(error) = self.control.take_error() {
                        return Err(error);
                    }
                    if self.control.is_aborted() {
                        return Err(SchedulerError::Aborted);
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return match self.control.take_error() {
                        Some(error) => Err(error),
                        None if self.control.is_aborted() => Err(SchedulerError::Aborted),
                        None => Err(SchedulerError::ChannelDisconnected { queue: COMPLETED_BATCH_QUEUE }),
                    };
                }
            }
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
        self.state.submitted_batch_count == self.state.completed_batch_count
    }

    /// Close the decoded-batch input so queued work can complete.
    pub(crate) fn close_submission(&mut self) {
        self.state.chromosome_prepared = false;
        self.state.chromosome_release_pending = false;
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

    fn batch_counts(&self) -> (usize, usize) {
        (self.state.submitted_batch_count, self.state.completed_batch_count)
    }

    fn has_pending_event(&self) -> bool {
        self.state.submitted_batch_count != self.state.completed_batch_count || self.state.chromosome_release_pending
    }

    fn record_event(
        &mut self,
        event: AssociationSchedulerEvent,
    ) -> SchedulerResult<AssociationSchedulerEvent, Backend::Error> {
        match &event {
            AssociationSchedulerEvent::Completed(_) => {
                if self.state.completed_batch_count >= self.state.submitted_batch_count {
                    return Err(SchedulerError::InvalidBatch {
                        message: "pipeline produced more completed batches than were submitted".to_string(),
                    });
                }
                self.state.completed_batch_count =
                    self.state.completed_batch_count.checked_add(1).ok_or(SchedulerError::BatchCounterOverflow)?;
            }
            AssociationSchedulerEvent::ChromosomeReleased => {
                if !self.state.chromosome_release_pending {
                    return Err(SchedulerError::InvalidBatch { message: "unexpected chromosome release".to_string() });
                }
                self.state.chromosome_release_pending = false;
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
    let mut chromosome = ChromosomeStateGuard { backend, state: None };
    while let Ok(command) = compute_receiver.recv() {
        if control.is_aborted() {
            release_command_state(backend, command);
            break;
        }
        let scheduled_batch = match command {
            ComputeCommand::PrepareChromosome { state } => {
                if chromosome.state.is_some() {
                    backend.release_chromosome(state);
                    control.record_error(SchedulerError::ChromosomeAlreadyPrepared);
                    break;
                }
                chromosome.state = Some(state);
                continue;
            }
            ComputeCommand::ReleaseChromosome => {
                let Some(chromosome_state) = chromosome.state.take() else {
                    control.record_error(SchedulerError::ChromosomeNotPrepared);
                    break;
                };
                backend.release_chromosome(chromosome_state);
                if !send_scheduler_event(event_sender, AssociationSchedulerEvent::ChromosomeReleased, control) {
                    break;
                }
                continue;
            }
            ComputeCommand::ComputeBatch { batch } => batch,
        };
        let Some(active_chromosome_state) = chromosome.state.as_ref() else {
            control.record_error(SchedulerError::ChromosomeNotPrepared);
            break;
        };
        let ScheduledAssociationBatch { decoded, metadata, active_trait_selection } = scheduled_batch;
        let DecodedGenotypeBatch {
            variant_start_index,
            logical_variant_count: _,
            compute_variant_count,
            sample_count,
            genotypes,
            statistics,
        } = decoded;
        let ChunkStats { output: output_statistics, compute: compute_statistics } = statistics;
        let input = GenotypeBatchInput {
            variant_count: compute_variant_count,
            sample_count,
            genotypes,
            statistics: compute_statistics,
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
        if device_sender.send(DeviceAssociationBatch { output, device_result }).is_err() {
            if !control.is_aborted() {
                control.record_error(SchedulerError::ChannelDisconnected { queue: DEVICE_RESULT_QUEUE });
            }
            break;
        }
    }
    drop(chromosome);
    drain_command_states(backend, compute_receiver);
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
        let logical_variant_count = device_batch.output.metadata.row_count();
        let result =
            match backend.materialize_batch(device_batch.device_result, active_trait_indices, logical_variant_count) {
                Ok(result) => result,
                Err(source) => {
                    control.record_error(SchedulerError::Backend { stage: MATERIALIZATION_WORKER, source });
                    break;
                }
            };
        let event =
            AssociationSchedulerEvent::Completed(CompletedAssociationBatch { output: device_batch.output, result });
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
        match sender.send_timeout(pending_event, CHANNEL_POLL_INTERVAL) {
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
    if let ComputeCommand::PrepareChromosome { state } = command {
        backend.release_chromosome(state);
    }
}

fn drain_command_states<Backend>(backend: &Backend, receiver: &Receiver<ComputeCommand<Backend::ChromosomeState>>)
where
    Backend: AssociationBackend,
{
    while let Ok(command) = receiver.try_recv() {
        release_command_state(backend, command);
    }
}

struct ChromosomeStateGuard<'backend, Backend>
where
    Backend: AssociationBackend,
{
    backend: &'backend Backend,
    state: Option<Backend::ChromosomeState>,
}

impl<Backend> Drop for ChromosomeStateGuard<'_, Backend>
where
    Backend: AssociationBackend,
{
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            self.backend.release_chromosome(state);
        }
    }
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
