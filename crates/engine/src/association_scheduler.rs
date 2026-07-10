//! Bounded two-stage association batch scheduler.

use std::any::Any;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::thread;

use crossbeam_channel::{Receiver, Sender, TryRecvError};
use g_genotype::{ChunkStats, VariantMetadataColumns};
use g_plan::FloatingPointDtype;

use crate::backend::{
    AssociationBackend, BackendError, GenotypeBatchInput, GenotypeBatchStatisticsView, GenotypeMatrixView,
    HostAssociationBatch, MaterializationInput, VariantMajorDosageMatrixView, VariantMajorPacked8MatrixView,
};

const COMPUTE_WORKER_NAME: &str = "g-association-compute";
const MATERIALIZATION_WORKER_NAME: &str = "g-association-materialization";
const DECODED_BATCH_QUEUE: &str = "decoded batch";
const DEVICE_RESULT_QUEUE: &str = "device result";
const COMPLETED_BATCH_QUEUE: &str = "completed batch";
const COMPUTE_WORKER: &str = "compute";
const MATERIALIZATION_WORKER: &str = "materialization";

/// Owned host genotype allocation carried through the scheduler.
#[derive(Debug, PartialEq)]
pub enum OwnedGenotypeBuffer {
    Dosage(Vec<f32>),
    Packed8(Vec<u8>),
}

/// One owned association batch ready for device compute.
#[derive(Debug, PartialEq)]
pub struct ScheduledAssociationBatch {
    pub variant_start_index: usize,
    pub variant_count: usize,
    pub sample_count: usize,
    pub metadata: VariantMetadataColumns,
    pub statistics: ChunkStats,
    pub genotype_buffer: OwnedGenotypeBuffer,
    pub active_trait_indices: Vec<usize>,
    pub output_statistic_dtype: FloatingPointDtype,
}

impl ScheduledAssociationBatch {
    fn genotype_input(&self) -> GenotypeBatchInput<'_> {
        let genotypes = match &self.genotype_buffer {
            OwnedGenotypeBuffer::Dosage(values) => GenotypeMatrixView::Dosage(VariantMajorDosageMatrixView {
                values,
                variant_count: self.variant_count,
                sample_count: self.sample_count,
            }),
            OwnedGenotypeBuffer::Packed8(values) => GenotypeMatrixView::Packed8(VariantMajorPacked8MatrixView {
                values,
                variant_count: self.variant_count,
                sample_count: self.sample_count,
            }),
        };
        GenotypeBatchInput {
            variant_start_index: self.variant_start_index,
            genotypes,
            statistics: GenotypeBatchStatisticsView {
                dosage_sum: &self.statistics.dosage_sum,
                observation_count: &self.statistics.observation_count,
                imputed_dosage_square_sum: Some(&self.statistics.imputed_dosage_square_sum),
                sparse_candidate_mask: Some(&self.statistics.is_rare_sparse_firth_candidate),
            },
        }
    }

    fn validate(&self) -> SchedulerResult<()> {
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
        validate_variant_column_length("metadata chromosome", self.metadata.chromosome.len(), self.variant_count)?;
        validate_variant_column_length(
            "metadata variant identifier",
            self.metadata.variant_identifier.len(),
            self.variant_count,
        )?;
        validate_variant_column_length("metadata position", self.metadata.position.len(), self.variant_count)?;
        validate_variant_column_length("metadata allele one", self.metadata.allele_one.len(), self.variant_count)?;
        validate_variant_column_length("metadata allele two", self.metadata.allele_two.len(), self.variant_count)?;
        validate_variant_column_length("dosage sum", self.statistics.dosage_sum.len(), self.variant_count)?;
        validate_variant_column_length(
            "observation count",
            self.statistics.observation_count.len(),
            self.variant_count,
        )?;
        validate_variant_column_length(
            "imputed dosage square sum",
            self.statistics.imputed_dosage_square_sum.len(),
            self.variant_count,
        )?;
        validate_variant_column_length(
            "rare sparse Firth candidate mask",
            self.statistics.is_rare_sparse_firth_candidate.len(),
            self.variant_count,
        )
    }
}

/// One completed host result and the resources that produced it.
#[derive(Debug, PartialEq)]
pub struct CompletedAssociationBatch {
    pub variant_start_index: usize,
    pub variant_count: usize,
    pub sample_count: usize,
    pub metadata: VariantMetadataColumns,
    pub statistics: ChunkStats,
    pub genotype_buffer: OwnedGenotypeBuffer,
    pub result: HostAssociationBatch,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SchedulerError {
    #[error("association scheduler {queue} capacity must be positive, observed {capacity}")]
    InvalidCapacity { queue: &'static str, capacity: usize },
    #[error("invalid scheduled association batch: {message}")]
    InvalidBatch { message: String },
    #[error("association backend failed during {stage}: {source}")]
    Backend { stage: &'static str, source: BackendError },
    #[error("association scheduler {worker} worker could not start: {message}")]
    WorkerSpawn { worker: &'static str, message: String },
    #[error("association scheduler {worker} worker panicked: {message}")]
    WorkerPanicked { worker: &'static str, message: String },
    #[error("association scheduler {queue} channel disconnected unexpectedly")]
    ChannelDisconnected { queue: &'static str },
    #[error("association scheduler is closed")]
    Closed,
    #[error("association scheduler was aborted")]
    Aborted,
}

type SchedulerResult<Value> = Result<Value, SchedulerError>;

struct DeviceAssociationBatch<DeviceResult> {
    scheduled_batch: ScheduledAssociationBatch,
    device_result: DeviceResult,
}

#[derive(Debug)]
struct SchedulerControl {
    aborted: std::sync::atomic::AtomicBool,
    first_error: Mutex<Option<SchedulerError>>,
}

impl SchedulerControl {
    const fn new() -> Self {
        Self { aborted: std::sync::atomic::AtomicBool::new(false), first_error: Mutex::new(None) }
    }

    fn abort(&self) {
        self.aborted.store(true, std::sync::atomic::Ordering::Release);
    }

    fn is_aborted(&self) -> bool {
        self.aborted.load(std::sync::atomic::Ordering::Acquire)
    }

    fn record_error(&self, error: SchedulerError) {
        {
            let mut first_error = self.first_error.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if first_error.is_none() {
                *first_error = Some(error);
            }
        }
        self.abort();
    }

    fn error(&self) -> Option<SchedulerError> {
        self.first_error.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }
}

/// Bounded decoded-to-device-to-host pipeline for one prepared chromosome.
pub struct AssociationBatchPipeline<Backend>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    decoded_sender: Option<Sender<ScheduledAssociationBatch>>,
    completed_receiver: Receiver<CompletedAssociationBatch>,
    compute_worker: Option<thread::JoinHandle<()>>,
    materialization_worker: Option<thread::JoinHandle<()>>,
    control: Arc<SchedulerControl>,
    backend_marker: PhantomData<Backend>,
}

impl<Backend> AssociationBatchPipeline<Backend>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    /// Start a two-stage pipeline for one prepared chromosome state.
    ///
    /// # Errors
    ///
    /// Returns an error when either capacity is zero or a worker thread cannot
    /// be started.
    pub fn new(
        backend: Arc<Backend>,
        chromosome_state: Backend::ChromosomeState,
        staging_depth: usize,
        result_in_flight_limit: usize,
    ) -> SchedulerResult<Self> {
        validate_capacity(DECODED_BATCH_QUEUE, staging_depth)?;
        validate_capacity(DEVICE_RESULT_QUEUE, result_in_flight_limit)?;

        let (decoded_sender, decoded_receiver) = crossbeam_channel::bounded(staging_depth);
        let (device_sender, device_receiver) = crossbeam_channel::bounded(result_in_flight_limit);
        let (completed_sender, completed_receiver) = crossbeam_channel::unbounded();
        let control = Arc::new(SchedulerControl::new());

        let materialization_backend = Arc::clone(&backend);
        let materialization_control = Arc::clone(&control);
        let materialization_worker = thread::Builder::new()
            .name(MATERIALIZATION_WORKER_NAME.to_string())
            .spawn(move || {
                let worker_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_materialization_worker(
                        materialization_backend.as_ref(),
                        &device_receiver,
                        &completed_sender,
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
        let compute_worker = match thread::Builder::new().name(COMPUTE_WORKER_NAME.to_string()).spawn(move || {
            let worker_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_compute_worker(
                    backend.as_ref(),
                    &chromosome_state,
                    &decoded_receiver,
                    &device_sender,
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
                drop(decoded_sender);
                let _ = materialization_worker.join();
                return Err(SchedulerError::WorkerSpawn { worker: COMPUTE_WORKER, message: source.to_string() });
            }
        };

        Ok(Self {
            decoded_sender: Some(decoded_sender),
            completed_receiver,
            compute_worker: Some(compute_worker),
            materialization_worker: Some(materialization_worker),
            control,
            backend_marker: PhantomData,
        })
    }

    /// Submit one owned decoded batch, blocking while the staging queue is full.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch shape is invalid, the pipeline has been
    /// closed or aborted, or a worker has failed.
    pub fn submit(&self, batch: ScheduledAssociationBatch) -> SchedulerResult<()> {
        batch.validate()?;
        self.ensure_running()?;
        let sender = self.decoded_sender.as_ref().ok_or(SchedulerError::Closed)?;
        sender.send(batch).map_err(|_| self.current_or_channel_error(DECODED_BATCH_QUEUE))
    }

    /// Receive the next completed batch, or `None` after all workers close.
    ///
    /// # Errors
    ///
    /// Returns the first worker or backend error, or reports an explicit abort.
    pub fn receive(&self) -> SchedulerResult<Option<CompletedAssociationBatch>> {
        if let Some(error) = self.control.error() {
            return Err(error);
        }
        if self.control.is_aborted() {
            return Err(SchedulerError::Aborted);
        }
        match self.completed_receiver.recv() {
            Ok(batch) => Ok(Some(batch)),
            Err(_) => match self.control.error() {
                Some(error) => Err(error),
                None if self.control.is_aborted() => Err(SchedulerError::Aborted),
                None => Ok(None),
            },
        }
    }

    /// Try to receive one completed batch without blocking.
    ///
    /// # Errors
    ///
    /// Returns the first worker or backend error, or reports an explicit abort.
    pub fn try_receive(&self) -> SchedulerResult<Option<CompletedAssociationBatch>> {
        if let Some(error) = self.control.error() {
            return Err(error);
        }
        if self.control.is_aborted() {
            return Err(SchedulerError::Aborted);
        }
        match self.completed_receiver.try_recv() {
            Ok(batch) => Ok(Some(batch)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => match self.control.error() {
                Some(error) => Err(error),
                None if self.control.is_aborted() => Err(SchedulerError::Aborted),
                None => Ok(None),
            },
        }
    }

    /// Close submission and join compute followed by materialization.
    ///
    /// # Errors
    ///
    /// Returns the first worker or backend error, including a worker panic.
    pub fn finish(&mut self) -> SchedulerResult<()> {
        self.decoded_sender.take();
        self.join_workers();
        match self.control.error() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Cancel queued work, close submission, and join both workers.
    ///
    /// # Errors
    ///
    /// Returns a worker or backend error observed before or during shutdown.
    pub fn abort(&mut self) -> SchedulerResult<()> {
        self.control.abort();
        self.decoded_sender.take();
        self.join_workers();
        match self.control.error() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn ensure_running(&self) -> SchedulerResult<()> {
        if let Some(error) = self.control.error() {
            return Err(error);
        }
        if self.control.is_aborted() {
            return Err(SchedulerError::Aborted);
        }
        if self.decoded_sender.is_none() {
            return Err(SchedulerError::Closed);
        }
        Ok(())
    }

    fn current_or_channel_error(&self, queue: &'static str) -> SchedulerError {
        self.control.error().unwrap_or(SchedulerError::ChannelDisconnected { queue })
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
        self.decoded_sender.take();
        self.join_workers();
    }
}

fn run_compute_worker<Backend>(
    backend: &Backend,
    chromosome_state: &Backend::ChromosomeState,
    decoded_receiver: &Receiver<ScheduledAssociationBatch>,
    device_sender: &Sender<DeviceAssociationBatch<Backend::DeviceResult>>,
    control: &SchedulerControl,
) where
    Backend: AssociationBackend,
{
    while let Ok(scheduled_batch) = decoded_receiver.recv() {
        if control.is_aborted() {
            break;
        }
        let device_result = match backend.compute_batch(chromosome_state, scheduled_batch.genotype_input()) {
            Ok(result) => result,
            Err(source) => {
                control.record_error(SchedulerError::Backend { stage: COMPUTE_WORKER, source });
                break;
            }
        };
        if control.is_aborted() {
            break;
        }
        if device_sender.send(DeviceAssociationBatch { scheduled_batch, device_result }).is_err() {
            if !control.is_aborted() {
                control.record_error(SchedulerError::ChannelDisconnected { queue: DEVICE_RESULT_QUEUE });
            }
            break;
        }
    }
}

fn run_materialization_worker<Backend>(
    backend: &Backend,
    device_receiver: &Receiver<DeviceAssociationBatch<Backend::DeviceResult>>,
    completed_sender: &Sender<CompletedAssociationBatch>,
    control: &SchedulerControl,
) where
    Backend: AssociationBackend,
{
    while let Ok(device_batch) = device_receiver.recv() {
        if control.is_aborted() {
            break;
        }
        let materialization_input = MaterializationInput {
            active_trait_indices: &device_batch.scheduled_batch.active_trait_indices,
            output_statistic_dtype: device_batch.scheduled_batch.output_statistic_dtype,
        };
        let result = match backend.materialize_batch(device_batch.device_result, materialization_input) {
            Ok(result) => result,
            Err(source) => {
                control.record_error(SchedulerError::Backend { stage: MATERIALIZATION_WORKER, source });
                break;
            }
        };
        let scheduled_batch = device_batch.scheduled_batch;
        let completed_batch = CompletedAssociationBatch {
            variant_start_index: scheduled_batch.variant_start_index,
            variant_count: scheduled_batch.variant_count,
            sample_count: scheduled_batch.sample_count,
            metadata: scheduled_batch.metadata,
            statistics: scheduled_batch.statistics,
            genotype_buffer: scheduled_batch.genotype_buffer,
            result,
        };
        if completed_sender.send(completed_batch).is_err() {
            if !control.is_aborted() {
                control.record_error(SchedulerError::ChannelDisconnected { queue: COMPLETED_BATCH_QUEUE });
            }
            break;
        }
    }
}

fn validate_capacity(queue: &'static str, capacity: usize) -> SchedulerResult<()> {
    if capacity == 0 {
        return Err(SchedulerError::InvalidCapacity { queue, capacity });
    }
    Ok(())
}

fn validate_variant_column_length(name: &str, observed: usize, expected: usize) -> SchedulerResult<()> {
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
