//! Bounded two-stage association batch scheduler.

use std::any::Any;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[cfg(test)]
use std::sync::Condvar;

use crate::backend::{AssociationBackend, MaterializedAssociationBatch, MaterializedGenotypeStatistics};
use crossbeam_channel::{Receiver, RecvTimeoutError, SendTimeoutError, Sender, TryRecvError};
use g_genotype::{GenotypeBatch, GenotypeBatchPayload, OwnedGenotypeBuffer};
use g_genotype_contracts::ChunkOutputStatistics;
use g_output::NativeVariantMetadataHandle;

use crate::output_schedule::ActiveTraitSelection;

const COMPUTE_WORKER_NAME: &str = "g-association-compute";
const MATERIALIZATION_WORKER_NAME: &str = "g-association-materialization";
const TRANSFERRED_BATCH_CAPACITY: usize = 1;
const DEVICE_RESULT_CAPACITY: usize = 2;
const TRANSFERRED_BATCH_QUEUE: &str = "transferred batch";
const DEVICE_RESULT_QUEUE: &str = "device result";
const COMPLETED_BATCH_QUEUE: &str = "completed batch";
const TRANSFER_STAGE: &str = "device transfer";
const COMPUTE_WORKER: &str = "compute";
const MATERIALIZATION_WORKER: &str = "materialization";
const CHANNEL_POLL_INTERVAL: Duration = Duration::from_millis(10);
#[cfg(test)]
const TEST_QUIESCENCE_TIMEOUT: Duration = Duration::from_secs(5);

/// One owned association batch ready for device compute.
#[derive(Debug)]
pub(crate) struct ScheduledAssociationBatch {
    pub(crate) genotypes: GenotypeBatch,
    pub(crate) metadata: NativeVariantMetadataHandle,
    pub(crate) active_trait_selection: ActiveTraitSelection,
}

impl ScheduledAssociationBatch {
    fn validate<BackendError>(&self) -> SchedulerResult<(), BackendError> {
        let genotypes = &self.genotypes;
        if self.metadata.row_count() != genotypes.logical_variant_count {
            return Err(SchedulerError::InvalidBatch {
                message: format!(
                    "metadata contains {} variants, genotype batch contains {}",
                    self.metadata.row_count(),
                    genotypes.logical_variant_count
                ),
            });
        }
        if genotypes.compute_variant_count < genotypes.logical_variant_count {
            return Err(SchedulerError::InvalidBatch {
                message: format!(
                    "compute variant count {} is smaller than logical variant count {}",
                    genotypes.compute_variant_count, genotypes.logical_variant_count
                ),
            });
        }
        if let GenotypeBatchPayload::Decoded { genotypes: values, statistics } = &genotypes.payload {
            validate_decoded_batch::<BackendError>(genotypes, values, statistics)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct AssociationBatchContext {
    pub(crate) variant_start_index: usize,
    pub(crate) metadata: NativeVariantMetadataHandle,
    pub(crate) active_trait_selection: ActiveTraitSelection,
}

/// One completed host result and the resources that produced it.
#[derive(Debug)]
pub(crate) struct CompletedAssociationBatch {
    pub(crate) context: AssociationBatchContext,
    pub(crate) statistics: ChunkOutputStatistics,
    pub(crate) result: g_output::Regenie2StatisticBatch,
}

#[derive(Debug)]
// Keeping completed batches inline avoids one allocation on every hot-path
// result; the small release event occurs only at drained transitions.
#[allow(clippy::large_enum_variant)]
enum AssociationSchedulerEvent {
    Completed(CompletedAssociationBatch),
    ChromosomePrepared { null_logistic_converged: Option<Vec<bool>> },
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
    #[error(transparent)]
    Genotype(#[from] g_genotype::GenotypeError),
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
    #[error("association scheduler chromosome preparation is already pending")]
    ChromosomePreparePending,
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
    context: AssociationBatchContext,
    device_result: DeviceResult,
}

struct TransferredAssociationBatch<TransferredInput> {
    context: AssociationBatchContext,
    input: TransferredInput,
}

struct MaterializationQuiescenceGuard<Value> {
    device_receiver: Option<Receiver<Value>>,
    materialization_quiesced_sender: Option<Sender<()>>,
}

impl<Value> MaterializationQuiescenceGuard<Value> {
    const fn new(device_receiver: Receiver<Value>, materialization_quiesced_sender: Sender<()>) -> Self {
        Self {
            device_receiver: Some(device_receiver),
            materialization_quiesced_sender: Some(materialization_quiesced_sender),
        }
    }

    fn device_receiver(&self) -> &Receiver<Value> {
        self.device_receiver.as_ref().expect("materialization worker retains its device-result receiver")
    }
}

impl<Value> Drop for MaterializationQuiescenceGuard<Value> {
    fn drop(&mut self) {
        drop(self.device_receiver.take());
        if let Some(materialization_quiesced_sender) = self.materialization_quiesced_sender.take() {
            let _ = materialization_quiesced_sender.try_send(());
        }
    }
}

// Keeping batches inline lets the bounded channel allocate its storage once;
// boxing the hot variant would add one allocation to every submitted batch.
enum ComputeCommand<TransferredInput> {
    PrepareChromosome { predictions: g_input::ChromosomePredictionMatrix },
    ReleaseChromosome,
    ComputeBatch { batch: TransferredAssociationBatch<TransferredInput> },
}

#[derive(Debug, Default)]
struct PipelineSchedulerState {
    submitted_batch_count: usize,
    completed_batch_count: usize,
    chromosome_prepare_pending: bool,
    chromosome_prepared: bool,
    chromosome_release_pending: bool,
}

#[derive(Debug)]
struct SchedulerControl<BackendError> {
    aborted: std::sync::atomic::AtomicBool,
    failure: Mutex<SchedulerFailure<BackendError>>,
    #[cfg(test)]
    materialization_quiescence_waiting: Mutex<bool>,
    #[cfg(test)]
    materialization_quiescence_waiting_changed: Condvar,
}

#[derive(Debug)]
struct SchedulerFailure<BackendError> {
    first_error: Option<SchedulerError<BackendError>>,
    recorded: bool,
}

impl<BackendError> SchedulerControl<BackendError> {
    fn new() -> Self {
        Self {
            aborted: std::sync::atomic::AtomicBool::new(false),
            failure: Mutex::new(SchedulerFailure { first_error: None, recorded: false }),
            #[cfg(test)]
            materialization_quiescence_waiting: Mutex::new(false),
            #[cfg(test)]
            materialization_quiescence_waiting_changed: Condvar::new(),
        }
    }

    fn abort(&self) {
        self.aborted.store(true, std::sync::atomic::Ordering::Release);
    }

    fn is_aborted(&self) -> bool {
        self.aborted.load(std::sync::atomic::Ordering::Acquire)
    }

    fn record_error(&self, error: SchedulerError<BackendError>) {
        let mut failure = self.failure.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !failure.recorded {
            failure.first_error = Some(error);
            failure.recorded = true;
        }
        self.abort();
    }

    #[cfg(test)]
    fn take_recorded_error(&self) -> Option<SchedulerError<BackendError>> {
        self.failure.lock().unwrap_or_else(std::sync::PoisonError::into_inner).first_error.take()
    }

    fn check_failure(&self) -> Option<SchedulerError<BackendError>> {
        let mut failure = self.failure.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        failure.first_error.take().or_else(|| self.is_aborted().then_some(SchedulerError::Aborted))
    }

    #[cfg(test)]
    fn record_materialization_quiescence_wait(&self) {
        *self.materialization_quiescence_waiting.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        self.materialization_quiescence_waiting_changed.notify_all();
    }

    #[cfg(test)]
    fn wait_for_materialization_quiescence(&self) {
        let waiting = self.materialization_quiescence_waiting.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let (waiting, _) = self
            .materialization_quiescence_waiting_changed
            .wait_timeout_while(waiting, TEST_QUIESCENCE_TIMEOUT, |waiting| !*waiting)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(*waiting, "compute worker did not reach materialization quiescence");
    }
}

/// One bounded decoded-to-device-to-host delivery pipeline.
pub(crate) struct AssociationBatchPipeline<Backend>
where
    Backend: AssociationBackend + 'static,
{
    backend: Arc<Backend>,
    group: Arc<Backend::GroupState>,
    compute_sender: Option<Sender<ComputeCommand<Backend::TransferredInput>>>,
    event_receiver: Receiver<AssociationSchedulerEvent>,
    compute_worker: Option<thread::JoinHandle<()>>,
    materialization_worker: Option<thread::JoinHandle<()>>,
    control: Arc<SchedulerControl<Backend::Error>>,
    state: PipelineSchedulerState,
}

impl<Backend> AssociationBatchPipeline<Backend>
where
    Backend: AssociationBackend + 'static,
{
    /// Start one two-stage pipeline with the engine's bounded capacities.
    ///
    /// # Errors
    ///
    /// Returns an error when a worker thread cannot be started.
    pub(crate) fn new(backend: Arc<Backend>, group: Arc<Backend::GroupState>) -> SchedulerResult<Self, Backend::Error> {
        let (compute_sender, compute_receiver) = crossbeam_channel::bounded(TRANSFERRED_BATCH_CAPACITY);
        let (device_sender, device_receiver) = crossbeam_channel::bounded(DEVICE_RESULT_CAPACITY);
        let (event_sender, event_receiver) = crossbeam_channel::bounded(DEVICE_RESULT_CAPACITY);
        let (materialization_quiesced_sender, materialization_quiesced_receiver) = crossbeam_channel::bounded(1);
        let control = Arc::new(SchedulerControl::new());

        let materialization_backend = Arc::clone(&backend);
        let materialization_control = Arc::clone(&control);
        let materialization_event_sender = event_sender.clone();
        let materialization_worker = thread::Builder::new()
            .name(MATERIALIZATION_WORKER_NAME.to_string())
            .spawn(move || {
                let materialization_quiescence =
                    MaterializationQuiescenceGuard::new(device_receiver, materialization_quiesced_sender);
                let worker_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_materialization_worker(
                        materialization_backend.as_ref(),
                        materialization_quiescence.device_receiver(),
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
                drop(materialization_quiescence);
            })
            .map_err(|source| SchedulerError::WorkerSpawn {
                worker: MATERIALIZATION_WORKER,
                message: source.to_string(),
            })?;

        let compute_control = Arc::clone(&control);
        let compute_event_sender = event_sender;
        let pipeline_backend = Arc::clone(&backend);
        let pipeline_group = Arc::clone(&group);
        let compute_worker = match thread::Builder::new().name(COMPUTE_WORKER_NAME.to_string()).spawn(move || {
            run_compute_worker(
                backend.as_ref(),
                group.as_ref(),
                &compute_receiver,
                device_sender,
                &compute_event_sender,
                &compute_control,
                &materialization_quiesced_receiver,
            );
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
            group: pipeline_group,
            compute_sender: Some(compute_sender),
            event_receiver,
            compute_worker: Some(compute_worker),
            materialization_worker: Some(materialization_worker),
            control,
            state: PipelineSchedulerState::default(),
        })
    }

    /// Prepare and install chromosome state on the backend execution worker.
    ///
    /// # Errors
    ///
    /// Returns an error when prior batches remain pending or the pipeline has
    /// been closed, aborted, or failed.
    pub(crate) fn prepare_chromosome(
        &mut self,
        predictions: g_input::ChromosomePredictionMatrix,
    ) -> SchedulerResult<Option<Vec<bool>>, Backend::Error> {
        self.ensure_running()?;
        if !self.is_drained() {
            let (submitted, completed) = self.batch_counts();
            return Err(SchedulerError::ChromosomeTransitionPending { submitted, completed });
        }
        if self.state.chromosome_prepare_pending {
            return Err(SchedulerError::ChromosomePreparePending);
        }
        if self.state.chromosome_release_pending {
            return Err(SchedulerError::ChromosomeReleasePending);
        }
        if self.state.chromosome_prepared {
            return Err(SchedulerError::ChromosomeAlreadyPrepared);
        }
        let sender = self.compute_sender.as_ref().ok_or(SchedulerError::Closed)?;
        sender
            .send(ComputeCommand::PrepareChromosome { predictions })
            .map_err(|_| self.current_or_channel_error(TRANSFERRED_BATCH_QUEUE))?;
        self.state.chromosome_prepare_pending = true;
        match self.receive_scheduler_event()? {
            AssociationSchedulerEvent::ChromosomePrepared { null_logistic_converged } => Ok(null_logistic_converged),
            AssociationSchedulerEvent::Completed(_) | AssociationSchedulerEvent::ChromosomeReleased => {
                Err(SchedulerError::InvalidBatch {
                    message: "received an unexpected event at the chromosome-prepare barrier".to_string(),
                })
            }
        }
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
            .map_err(|_| self.current_or_channel_error(TRANSFERRED_BATCH_QUEUE))?;
        self.state.chromosome_prepared = false;
        self.state.chromosome_release_pending = true;
        match self.receive_scheduler_event()? {
            AssociationSchedulerEvent::ChromosomeReleased => Ok(()),
            AssociationSchedulerEvent::Completed(_) | AssociationSchedulerEvent::ChromosomePrepared { .. } => {
                Err(SchedulerError::InvalidBatch {
                    message: "received an unexpected event at the drained chromosome-release barrier".to_string(),
                })
            }
        }
    }

    /// Try to transfer and submit one decoded batch without blocking behind work.
    ///
    /// The first batch after a chromosome transition may briefly block until
    /// the compute worker consumes the preceding state command. Subsequent
    /// backpressure returns the decoded batch before transfer so the caller can
    /// drain completed output without increasing device residency.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch is invalid, the pipeline has been closed
    /// or aborted, transfer fails, or a worker has failed.
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
        if !self.is_drained() && sender.is_full() {
            return Ok(Some(batch));
        }
        let ScheduledAssociationBatch { genotypes, metadata, active_trait_selection } = batch;
        let variant_start_index = genotypes.variant_start_index;
        let input = self
            .backend
            .transfer_batch(self.group.as_ref(), genotypes)
            .map_err(|source| SchedulerError::Backend { stage: TRANSFER_STAGE, source })?;
        let context = AssociationBatchContext { variant_start_index, metadata, active_trait_selection };
        sender
            .send(ComputeCommand::ComputeBatch { batch: TransferredAssociationBatch { context, input } })
            .map_err(|_| self.current_or_channel_error(TRANSFERRED_BATCH_QUEUE))?;
        self.state.submitted_batch_count = next_submitted_batch_count;
        Ok(None)
    }

    /// Receive the next submitted batch.
    ///
    /// # Errors
    ///
    /// Returns the first worker or backend error, or reports an explicit abort.
    pub(crate) fn receive(&mut self) -> SchedulerResult<CompletedAssociationBatch, Backend::Error> {
        match self.receive_scheduler_event()? {
            AssociationSchedulerEvent::Completed(completed_batch) => Ok(completed_batch),
            AssociationSchedulerEvent::ChromosomePrepared { .. } | AssociationSchedulerEvent::ChromosomeReleased => {
                Err(SchedulerError::InvalidBatch {
                    message: "unexpected chromosome lifecycle event outside a drained transition".to_string(),
                })
            }
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
            Some(
                AssociationSchedulerEvent::ChromosomePrepared { .. } | AssociationSchedulerEvent::ChromosomeReleased,
            ) => Err(SchedulerError::InvalidBatch {
                message: "unexpected chromosome lifecycle event outside a drained transition".to_string(),
            }),
            None => Ok(None),
        }
    }

    fn receive_scheduler_event(&mut self) -> SchedulerResult<AssociationSchedulerEvent, Backend::Error> {
        self.check_failure()?;
        if !self.has_pending_event() {
            return Err(SchedulerError::NoPendingBatch);
        }
        loop {
            match self.event_receiver.recv_timeout(CHANNEL_POLL_INTERVAL) {
                Ok(event) => return self.record_event(event),
                Err(RecvTimeoutError::Timeout) => {
                    self.check_failure()?;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return match self.check_failure() {
                        Err(error) => Err(error),
                        Ok(()) => Err(SchedulerError::ChannelDisconnected { queue: COMPLETED_BATCH_QUEUE }),
                    };
                }
            }
        }
    }

    fn try_receive_scheduler_event(&mut self) -> SchedulerResult<Option<AssociationSchedulerEvent>, Backend::Error> {
        self.check_failure()?;
        match self.event_receiver.try_recv() {
            Ok(event) => self.record_event(event).map(Some),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => match self.check_failure() {
                Err(error) => Err(error),
                Ok(()) if !self.has_pending_event() && self.compute_sender.is_none() => Ok(None),
                Ok(()) => Err(SchedulerError::ChannelDisconnected { queue: COMPLETED_BATCH_QUEUE }),
            },
        }
    }

    /// Return whether every submitted batch has been received by the caller.
    #[must_use]
    pub(crate) fn is_drained(&self) -> bool {
        self.state.submitted_batch_count == self.state.completed_batch_count
    }

    /// Return the scheduler's recorded failure or explicit abort state.
    ///
    /// A normally closed pipeline has no failure.
    ///
    /// # Errors
    ///
    /// Returns and consumes the first concrete scheduler failure, or reports an
    /// explicit abort when no concrete failure was recorded.
    pub(crate) fn check_failure(&self) -> SchedulerResult<(), Backend::Error> {
        self.control.check_failure().map_or(Ok(()), Err)
    }

    #[cfg(test)]
    pub(crate) fn wait_for_materialization_quiescence_for_test(&self) {
        self.control.wait_for_materialization_quiescence();
    }

    /// Close the transferred-batch input so queued work can complete.
    pub(crate) fn close_submission(&mut self) {
        self.state.chromosome_prepare_pending = false;
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
        self.check_failure()
    }

    fn ensure_running(&self) -> SchedulerResult<(), Backend::Error> {
        self.check_failure()?;
        if self.compute_sender.is_none() {
            return Err(SchedulerError::Closed);
        }
        Ok(())
    }

    fn current_or_channel_error(&self, queue: &'static str) -> SchedulerError<Backend::Error> {
        self.check_failure().err().unwrap_or(SchedulerError::ChannelDisconnected { queue })
    }

    fn batch_counts(&self) -> (usize, usize) {
        (self.state.submitted_batch_count, self.state.completed_batch_count)
    }

    fn has_pending_event(&self) -> bool {
        self.state.submitted_batch_count != self.state.completed_batch_count
            || self.state.chromosome_prepare_pending
            || self.state.chromosome_release_pending
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
            AssociationSchedulerEvent::ChromosomePrepared { .. } => {
                if !self.state.chromosome_prepare_pending {
                    return Err(SchedulerError::InvalidBatch {
                        message: "unexpected chromosome preparation".to_string(),
                    });
                }
                self.state.chromosome_prepare_pending = false;
                self.state.chromosome_prepared = true;
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
{
    fn drop(&mut self) {
        self.control.abort();
        self.compute_sender.take();
        self.join_workers();
    }
}

fn run_compute_worker<Backend>(
    backend: &Backend,
    group: &Backend::GroupState,
    compute_receiver: &Receiver<ComputeCommand<Backend::TransferredInput>>,
    device_sender: Sender<DeviceAssociationBatch<Backend::DeviceResult>>,
    event_sender: &Sender<AssociationSchedulerEvent>,
    control: &SchedulerControl<Backend::Error>,
    materialization_quiesced_receiver: &Receiver<()>,
) where
    Backend: AssociationBackend,
{
    let mut chromosome_state = None;
    let worker_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_compute_worker_commands(
            backend,
            group,
            compute_receiver,
            &device_sender,
            event_sender,
            control,
            &mut chromosome_state,
        );
    }));
    if let Err(payload) = worker_result {
        control.record_error(SchedulerError::WorkerPanicked {
            worker: COMPUTE_WORKER,
            message: panic_message(payload.as_ref()),
        });
    }
    let release_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_after_materialization_quiescence(
            device_sender,
            materialization_quiesced_receiver,
            || {
                #[cfg(test)]
                control.record_materialization_quiescence_wait();
            },
            || {
                release_active_chromosome(backend, &mut chromosome_state);
            },
        );
    }));
    if let Err(payload) = release_result {
        record_chromosome_release_panic(control, payload.as_ref());
    }
}

fn record_chromosome_release_panic<BackendError>(control: &SchedulerControl<BackendError>, payload: &(dyn Any + Send)) {
    let panic_message = panic_message(payload);
    // Preserve the scheduler outcome before invoking the fallible observation boundary.
    control.record_error(SchedulerError::WorkerPanicked {
        worker: COMPUTE_WORKER,
        message: format!("chromosome release panicked: {panic_message}"),
    });
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tracing::error!(
            worker = COMPUTE_WORKER,
            cleanup_stage = "release_chromosome",
            panic_message,
            "association backend cleanup panicked"
        );
    }));
}

fn run_after_materialization_quiescence<Value, BeforeWait, AfterQuiescence>(
    device_sender: Sender<Value>,
    materialization_quiesced_receiver: &Receiver<()>,
    before_wait: BeforeWait,
    after_quiescence: AfterQuiescence,
) where
    BeforeWait: FnOnce(),
    AfterQuiescence: FnOnce(),
{
    drop(device_sender);
    before_wait();
    let _ = materialization_quiesced_receiver.recv();
    after_quiescence();
}

fn run_compute_worker_commands<Backend>(
    backend: &Backend,
    group: &Backend::GroupState,
    compute_receiver: &Receiver<ComputeCommand<Backend::TransferredInput>>,
    device_sender: &Sender<DeviceAssociationBatch<Backend::DeviceResult>>,
    event_sender: &Sender<AssociationSchedulerEvent>,
    control: &SchedulerControl<Backend::Error>,
    chromosome_state: &mut Option<Backend::ChromosomeState>,
) where
    Backend: AssociationBackend,
{
    while let Ok(command) = compute_receiver.recv() {
        if control.is_aborted() {
            break;
        }
        let transferred_batch = match command {
            ComputeCommand::PrepareChromosome { predictions } => {
                if chromosome_state.is_some() {
                    control.record_error(SchedulerError::ChromosomeAlreadyPrepared);
                    break;
                }
                let prepared_chromosome = match backend.prepare_chromosome(group, predictions) {
                    Ok(prepared_chromosome) => prepared_chromosome,
                    Err(source) => {
                        control.record_error(SchedulerError::Backend { stage: "prepare chromosome", source });
                        break;
                    }
                };
                *chromosome_state = Some(prepared_chromosome.state);
                if !send_scheduler_event(
                    event_sender,
                    AssociationSchedulerEvent::ChromosomePrepared {
                        null_logistic_converged: prepared_chromosome.null_logistic_converged,
                    },
                    control,
                ) {
                    break;
                }
                continue;
            }
            ComputeCommand::ReleaseChromosome => {
                let Some(active_chromosome_state) = chromosome_state.take() else {
                    control.record_error(SchedulerError::ChromosomeNotPrepared);
                    break;
                };
                backend.release_chromosome(active_chromosome_state);
                if !send_scheduler_event(event_sender, AssociationSchedulerEvent::ChromosomeReleased, control) {
                    break;
                }
                continue;
            }
            ComputeCommand::ComputeBatch { batch } => batch,
        };
        let Some(active_chromosome_state) = chromosome_state.as_ref() else {
            control.record_error(SchedulerError::ChromosomeNotPrepared);
            break;
        };
        let TransferredAssociationBatch { context, input } = transferred_batch;
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
        if device_sender.send(DeviceAssociationBatch { context, device_result }).is_err() {
            if !control.is_aborted() {
                control.record_error(SchedulerError::ChannelDisconnected { queue: DEVICE_RESULT_QUEUE });
            }
            break;
        }
    }
}

fn release_active_chromosome<Backend>(backend: &Backend, chromosome_state: &mut Option<Backend::ChromosomeState>)
where
    Backend: AssociationBackend,
{
    if let Some(active_chromosome_state) = chromosome_state.take() {
        backend.release_chromosome(active_chromosome_state);
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
        let active_trait_indices = match &device_batch.context.active_trait_selection {
            ActiveTraitSelection::All => None,
            ActiveTraitSelection::Indices(indices) => Some(indices.as_slice()),
        };
        let logical_variant_count = device_batch.context.metadata.row_count();
        let materialized =
            match backend.materialize_batch(device_batch.device_result, active_trait_indices, logical_variant_count) {
                Ok(result) => result,
                Err(source) => {
                    control.record_error(SchedulerError::Backend { stage: MATERIALIZATION_WORKER, source });
                    break;
                }
            };
        let MaterializedAssociationBatch { association, genotype_statistics } = materialized;
        let statistics = match genotype_statistics {
            MaterializedGenotypeStatistics::Ready(statistics) => statistics,
            MaterializedGenotypeStatistics::Packed8Raw(raw_statistics) => {
                match raw_statistics.into_output_statistics() {
                    Ok(statistics) => statistics,
                    Err(source) => {
                        control.record_error(SchedulerError::Genotype(source));
                        break;
                    }
                }
            }
        };
        let event = AssociationSchedulerEvent::Completed(CompletedAssociationBatch {
            context: device_batch.context,
            statistics,
            result: association,
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

fn validate_decoded_batch<BackendError>(
    batch: &GenotypeBatch,
    genotypes: &OwnedGenotypeBuffer,
    statistics: &g_genotype::ChunkStats,
) -> SchedulerResult<(), BackendError> {
    let genotype_value_count = batch.compute_variant_count.checked_mul(batch.sample_count).ok_or_else(|| {
        SchedulerError::InvalidBatch { message: "variant and sample counts overflow the host index width".to_string() }
    })?;
    let expected_genotype_value_count = match genotypes {
        OwnedGenotypeBuffer::Dosage(_) => genotype_value_count,
        OwnedGenotypeBuffer::Packed8(_) => {
            genotype_value_count.checked_mul(2).ok_or_else(|| SchedulerError::InvalidBatch {
                message: "packed8 genotype dimensions overflow the host index width".to_string(),
            })?
        }
    };
    let observed_genotype_value_count = match genotypes {
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
        "allele-one frequency",
        statistics.output.allele_one_frequency.len(),
        batch.logical_variant_count,
    )?;
    validate_variant_column_length(
        "observation count",
        statistics.output.observation_count.len(),
        batch.logical_variant_count,
    )?;
    validate_variant_column_length(
        "INFO score",
        statistics.output.info_score.values.len(),
        batch.logical_variant_count,
    )?;
    validate_variant_column_length(
        "INFO validity bitmap",
        statistics.output.info_score.validity_bytes.len(),
        batch.logical_variant_count.div_ceil(8),
    )?;
    validate_variant_column_length(
        "genotype mean",
        statistics.compute.genotype_mean.len(),
        batch.compute_variant_count,
    )?;
    if let Some(imputed_dosage_square_sum) = statistics.compute.imputed_dosage_square_sum.as_ref() {
        validate_variant_column_length(
            "imputed dosage square sum",
            imputed_dosage_square_sum.len(),
            batch.compute_variant_count,
        )?;
    }
    if let Some(sparse_candidate_mask) = statistics.compute.sparse_candidate_mask.as_ref() {
        validate_variant_column_length(
            "rare sparse Firth candidate mask",
            sparse_candidate_mask.len(),
            batch.compute_variant_count,
        )?;
    }
    Ok(())
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    payload.downcast_ref::<&str>().map_or_else(
        || payload.downcast_ref::<String>().cloned().unwrap_or_else(|| "unknown panic payload".to_string()),
        |message| (*message).to_string(),
    )
}

#[cfg(test)]
pub(crate) fn assert_disconnected_event_send_for_test() {
    let (event_sender, event_receiver) = crossbeam_channel::bounded(1);
    drop(event_receiver);
    let control = SchedulerControl::<std::convert::Infallible>::new();

    assert!(!send_scheduler_event(&event_sender, AssociationSchedulerEvent::ChromosomeReleased, &control,));
    assert!(matches!(
        control.take_recorded_error(),
        Some(SchedulerError::ChannelDisconnected { queue: COMPLETED_BATCH_QUEUE })
    ));
    assert!(control.is_aborted());
}

#[cfg(test)]
pub(crate) fn assert_consumed_first_failure_for_test() {
    let control = SchedulerControl::<std::convert::Infallible>::new();
    control.record_error(SchedulerError::InvalidBatch { message: "first".to_string() });
    assert!(matches!(
        control.take_recorded_error(),
        Some(SchedulerError::InvalidBatch { message }) if message == "first"
    ));

    control.record_error(SchedulerError::ChannelDisconnected { queue: DEVICE_RESULT_QUEUE });
    assert!(control.take_recorded_error().is_none());
    assert!(control.is_aborted());
}

#[cfg(test)]
pub(crate) fn assert_materialization_quiescence_handshake_for_test() {
    #[derive(Debug)]
    struct DeviceResultDropProbe {
        dropped: Arc<std::sync::atomic::AtomicBool>,
    }

    impl Drop for DeviceResultDropProbe {
        fn drop(&mut self) {
            self.dropped.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    let device_result_dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (queued_device_sender, queued_device_receiver) = crossbeam_channel::bounded(1);
    queued_device_sender
        .send(DeviceResultDropProbe { dropped: Arc::clone(&device_result_dropped) })
        .expect("test device-result queue accepts one result");
    let (quiesced_sender, quiesced_receiver) = crossbeam_channel::bounded(1);
    drop(queued_device_sender);
    drop(MaterializationQuiescenceGuard::new(queued_device_receiver, quiesced_sender));
    quiesced_receiver.recv().expect("materialization quiescence is acknowledged");
    assert!(device_result_dropped.load(std::sync::atomic::Ordering::SeqCst));

    let (device_sender, device_receiver) = crossbeam_channel::bounded::<()>(1);
    let (quiesced_sender, quiesced_receiver) = crossbeam_channel::bounded(1);
    let (cleanup_sender, cleanup_receiver) = crossbeam_channel::bounded(1);
    let cleanup_worker = thread::spawn(move || {
        run_after_materialization_quiescence(
            device_sender,
            &quiesced_receiver,
            || {},
            || {
                cleanup_sender.send(()).expect("cleanup observation receiver remains connected");
            },
        );
    });

    assert!(device_receiver.recv().is_err());
    assert!(matches!(cleanup_receiver.try_recv(), Err(TryRecvError::Empty)));
    quiesced_sender.send(()).expect("compute cleanup waits for quiescence");
    cleanup_receiver.recv().expect("cleanup runs after materialization quiescence");
    cleanup_worker.join().expect("cleanup test worker exits");
}

#[cfg(test)]
mod cleanup_logging_tests {
    use std::sync::Arc;

    struct PanickingEventSubscriber {
        event_count: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl tracing::Subscriber for PanickingEventSubscriber {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }

        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}

        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

        fn event(&self, _: &tracing::Event<'_>) {
            self.event_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            panic!("cleanup event subscriber panicked");
        }

        fn enter(&self, _: &tracing::span::Id) {}

        fn exit(&self, _: &tracing::span::Id) {}
    }

    #[test]
    fn panicking_cleanup_subscriber_preserves_primary_scheduler_failure() {
        let event_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cleanup_panic_payload: Box<dyn std::any::Any + Send> = Box::new("cleanup backend panic");
        let cleanup_control = super::SchedulerControl::<std::convert::Infallible>::new();

        tracing::subscriber::with_default(PanickingEventSubscriber { event_count: Arc::clone(&event_count) }, || {
            super::record_chromosome_release_panic(&cleanup_control, cleanup_panic_payload.as_ref());
        });

        assert!(matches!(
            cleanup_control.take_recorded_error(),
            Some(super::SchedulerError::WorkerPanicked { worker, message })
                if worker == super::COMPUTE_WORKER
                    && message == "chromosome release panicked: cleanup backend panic"
        ));

        let primary_control = super::SchedulerControl::<std::convert::Infallible>::new();
        primary_control
            .record_error(super::SchedulerError::InvalidBatch { message: "primary scheduler failure".to_string() });
        tracing::subscriber::with_default(PanickingEventSubscriber { event_count: Arc::clone(&event_count) }, || {
            super::record_chromosome_release_panic(&primary_control, cleanup_panic_payload.as_ref());
        });

        assert_eq!(event_count.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert!(matches!(
            primary_control.take_recorded_error(),
            Some(super::SchedulerError::InvalidBatch { message }) if message == "primary scheduler failure"
        ));
        assert!(primary_control.is_aborted());
    }
}
