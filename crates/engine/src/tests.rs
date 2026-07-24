use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use g_genotype::{ChunkComputeStatistics, ChunkStats, GenotypeBatch, GenotypeBatchPayload, OwnedGenotypeBuffer};
use g_genotype_contracts::{
    BgenSourceIdentity, ChunkOutputStatistics, NullableFloat32Column, VariantMetadataColumns, VariantMetadataStore,
};
use g_output::{ManifestFileFingerprintCache, NativeVariantMetadataHandle, Regenie2StatisticBatch};

use crate::association_scheduler::{
    AssociationBatchPipeline, ScheduledAssociationBatch, SchedulerError, assert_consumed_first_failure_for_test,
    assert_disconnected_event_send_for_test, assert_materialization_quiescence_handshake_for_test,
};
use crate::backend::{
    AssociationBackend, GenotypeDeliveryCapability, GroupPreparationInput, MaterializedAssociationBatch,
    MaterializedGenotypeStatistics, PreparedChromosome,
};
use crate::delivery_execution::{
    DeliveryError, prepare_group_after_interruption_check, retry_pending_batch, transition_drained_chromosome,
    try_submit_after_interruption_check,
};
use crate::genotype_buffer::homogeneous_chunk_chromosome;
use crate::null_logistic_policy::{
    NullLogisticNonconvergenceAction, NullLogisticPolicyError, plan_null_logistic_nonconvergence,
};
use crate::output_manifest::build_prediction_loco_file_fingerprints_with_cache;
use crate::output_schedule::{
    ActiveTraitSelection, active_trait_selection_for_chunk, intersect_committed_chunk_identifier_sets,
};
use crate::preflight::{PreflightError, validate_jax_index_capacity, validate_multi_trait_preflight_values};
use crate::preparation::{
    PipelineOutputPreparationError, RuntimeOutputGroupInput, RuntimeOutputPlan, build_runtime_output_initializations,
};
use crate::run::{RunPreparationError, validate_jax_integer_domain};

const TEST_SAMPLE_COUNT: usize = 3;
const TEST_SYNCHRONIZATION_TIMEOUT: Duration = Duration::from_secs(5);
const TEST_VARIANT_COUNT: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TestFailureStage {
    None,
    Prepare,
    Transfer,
    Compute,
    Materialize,
    ComputePanic,
    MaterializePanic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("test backend failed during {0}")]
struct TestBackendError(&'static str);

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("test delivery interruption")]
struct TestInterruption;

struct TestDeviceResult {
    variant_start_index: usize,
    logical_variant_count: usize,
    statistics: ChunkOutputStatistics,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BackendThreadEvent {
    operation: &'static str,
    thread_identifier: thread::ThreadId,
}

struct ComputeGateRelease {
    sender: Option<Sender<()>>,
}

impl ComputeGateRelease {
    const fn new(sender: Sender<()>) -> Self {
        Self { sender: Some(sender) }
    }

    fn release(&mut self) {
        self.sender
            .as_ref()
            .expect("compute gate has not already been released")
            .send_timeout((), TEST_SYNCHRONIZATION_TIMEOUT)
            .expect("compute gate accepts its release signal before the timeout");
        self.sender.take();
    }
}

impl Drop for ComputeGateRelease {
    fn drop(&mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.try_send(());
        }
    }
}

struct TestBackend {
    failure_stage: TestFailureStage,
    events: Mutex<Vec<String>>,
    backend_thread_events: Mutex<Vec<BackendThreadEvent>>,
    transfer_count: AtomicUsize,
    chromosome_release_count: AtomicUsize,
    materialized_trait_indices: Mutex<Vec<Option<Vec<usize>>>>,
    compute_started_sender: Option<Sender<usize>>,
    compute_gate_receiver: Option<Receiver<()>>,
    block_first_compute: AtomicBool,
    compute_failure_variant_start_index: Option<usize>,
    materialization_started_sender: Option<Sender<usize>>,
    materialization_gate_receiver: Option<Receiver<()>>,
    block_first_materialization: AtomicBool,
    chromosome_released_sender: Option<Sender<()>>,
    release_panics: bool,
}

impl TestBackend {
    fn new(failure_stage: TestFailureStage) -> Self {
        Self {
            failure_stage,
            events: Mutex::new(Vec::new()),
            backend_thread_events: Mutex::new(Vec::new()),
            transfer_count: AtomicUsize::new(0),
            chromosome_release_count: AtomicUsize::new(0),
            materialized_trait_indices: Mutex::new(Vec::new()),
            compute_started_sender: None,
            compute_gate_receiver: None,
            block_first_compute: AtomicBool::new(false),
            compute_failure_variant_start_index: None,
            materialization_started_sender: None,
            materialization_gate_receiver: None,
            block_first_materialization: AtomicBool::new(false),
            chromosome_released_sender: None,
            release_panics: false,
        }
    }

    fn with_first_compute_gate(compute_started_sender: Sender<usize>, compute_gate_receiver: Receiver<()>) -> Self {
        Self {
            compute_started_sender: Some(compute_started_sender),
            compute_gate_receiver: Some(compute_gate_receiver),
            block_first_compute: AtomicBool::new(true),
            ..Self::new(TestFailureStage::None)
        }
    }

    fn with_materialization_gate_and_compute_failure(
        compute_failure_variant_start_index: usize,
        materialization_started_sender: Sender<usize>,
        materialization_gate_receiver: Receiver<()>,
        chromosome_released_sender: Sender<()>,
    ) -> Self {
        Self {
            compute_failure_variant_start_index: Some(compute_failure_variant_start_index),
            materialization_started_sender: Some(materialization_started_sender),
            materialization_gate_receiver: Some(materialization_gate_receiver),
            block_first_materialization: AtomicBool::new(true),
            chromosome_released_sender: Some(chromosome_released_sender),
            ..Self::new(TestFailureStage::None)
        }
    }

    fn with_release_panic(failure_stage: TestFailureStage) -> Self {
        Self { release_panics: true, ..Self::new(failure_stage) }
    }

    fn record_event(&self, event: String) {
        self.events.lock().expect("test event lock is available").push(event);
    }

    fn record_backend_thread(&self, operation: &'static str) {
        self.backend_thread_events
            .lock()
            .expect("test backend-thread lock is available")
            .push(BackendThreadEvent { operation, thread_identifier: thread::current().id() });
    }
}

impl AssociationBackend for TestBackend {
    type GroupState = ();
    type ChromosomeState = usize;
    type TransferredInput = GenotypeBatch;
    type DeviceResult = TestDeviceResult;
    type Error = TestBackendError;

    fn genotype_delivery_capability(&self) -> GenotypeDeliveryCapability {
        GenotypeDeliveryCapability::HostOnly
    }

    fn prepare_group(&self, _input: GroupPreparationInput) -> Result<Self::GroupState, Self::Error> {
        self.record_event("prepare_group".to_string());
        Ok(())
    }

    fn prepare_chromosome(
        &self,
        _group: &Self::GroupState,
        predictions: g_input::ChromosomePredictionMatrix,
    ) -> Result<PreparedChromosome<Self::ChromosomeState>, Self::Error> {
        self.record_backend_thread("prepare");
        self.record_event("prepare_chromosome".to_string());
        if self.failure_stage == TestFailureStage::Prepare {
            return Err(TestBackendError("prepare"));
        }
        Ok(PreparedChromosome { state: predictions.sample_count, null_logistic_converged: None })
    }

    fn release_chromosome(&self, chromosome: Self::ChromosomeState) {
        self.record_backend_thread("release");
        self.chromosome_release_count.fetch_add(1, Ordering::SeqCst);
        self.record_event(format!("release:{chromosome}"));
        if let Some(chromosome_released_sender) = self.chromosome_released_sender.as_ref() {
            let _ = chromosome_released_sender.try_send(());
        }
        assert!(!self.release_panics, "intentional release panic");
    }

    fn transfer_batch(
        &self,
        _group: &Self::GroupState,
        input: GenotypeBatch,
    ) -> Result<Self::TransferredInput, Self::Error> {
        self.transfer_count.fetch_add(1, Ordering::SeqCst);
        self.record_event(format!("transfer:{}", input.variant_start_index));
        if self.failure_stage == TestFailureStage::Transfer {
            return Err(TestBackendError("transfer"));
        }
        Ok(input)
    }

    fn compute_batch(
        &self,
        chromosome: &Self::ChromosomeState,
        input: Self::TransferredInput,
    ) -> Result<Self::DeviceResult, Self::Error> {
        self.record_backend_thread("compute");
        let variant_start_index = input.variant_start_index;
        self.record_event(format!("compute:{chromosome}:{variant_start_index}"));
        assert!(self.failure_stage != TestFailureStage::ComputePanic, "intentional compute panic");
        if self.failure_stage == TestFailureStage::Compute {
            return Err(TestBackendError("compute"));
        }
        if self.compute_failure_variant_start_index == Some(variant_start_index) {
            return Err(TestBackendError("compute"));
        }
        if self.block_first_compute.swap(false, Ordering::SeqCst) {
            self.compute_started_sender
                .as_ref()
                .expect("gated test has a start sender")
                .send_timeout(variant_start_index, TEST_SYNCHRONIZATION_TIMEOUT)
                .expect("test receives the compute-start notification before the timeout");
            self.compute_gate_receiver
                .as_ref()
                .expect("gated test has a receiver")
                .recv_timeout(TEST_SYNCHRONIZATION_TIMEOUT)
                .expect("test releases the compute gate before the timeout");
        }
        let logical_variant_count = input.logical_variant_count;
        let GenotypeBatchPayload::Decoded { statistics, .. } = input.payload else {
            return Err(TestBackendError("unexpected compressed input"));
        };
        Ok(TestDeviceResult { variant_start_index, logical_variant_count, statistics: statistics.output })
    }

    fn materialize_batch(
        &self,
        result: Self::DeviceResult,
        active_trait_indices: Option<&[usize]>,
        logical_variant_count: usize,
    ) -> Result<MaterializedAssociationBatch, Self::Error> {
        self.record_event(format!("materialize:{}", result.variant_start_index));
        self.materialized_trait_indices
            .lock()
            .expect("test materialization lock is available")
            .push(active_trait_indices.map(<[usize]>::to_vec));
        if self.block_first_materialization.swap(false, Ordering::SeqCst) {
            self.record_event(format!("materialize_start:{}", result.variant_start_index));
            self.materialization_started_sender
                .as_ref()
                .expect("gated materialization test has a start sender")
                .send_timeout(result.variant_start_index, TEST_SYNCHRONIZATION_TIMEOUT)
                .expect("test receives the materialization-start notification before the timeout");
            self.materialization_gate_receiver
                .as_ref()
                .expect("gated materialization test has a receiver")
                .recv_timeout(TEST_SYNCHRONIZATION_TIMEOUT)
                .expect("test releases the materialization gate before the timeout");
            self.record_event(format!("materialize_end:{}", result.variant_start_index));
        }
        assert!(self.failure_stage != TestFailureStage::MaterializePanic, "intentional materialization panic");
        if self.failure_stage == TestFailureStage::Materialize {
            return Err(TestBackendError("materialize"));
        }
        if logical_variant_count != result.logical_variant_count {
            return Err(TestBackendError("logical variant count"));
        }
        let trait_count = active_trait_indices.map_or(1, <[usize]>::len);
        let value_count = trait_count * logical_variant_count;
        let start_value = f32::from(
            u16::try_from(result.variant_start_index).expect("test variant start index fits into float fixture range"),
        );
        Ok(MaterializedAssociationBatch {
            association: Regenie2StatisticBatch {
                trait_count,
                variant_count: logical_variant_count,
                beta: vec![start_value; value_count],
                standard_error: vec![1.0; value_count],
                chi_squared: vec![2.0; value_count],
                log10_p_value: vec![3.0; value_count],
                correction_code: None,
            },
            genotype_statistics: MaterializedGenotypeStatistics::Ready(result.statistics),
        })
    }
}

fn build_metadata(variant_count: usize, chromosome: &str) -> VariantMetadataColumns {
    let text_dictionary: Box<[Arc<str>]> = [Arc::from(chromosome), Arc::from("A"), Arc::from("G")].into();
    let variant_identifier_text = "v".repeat(variant_count);
    let variant_identifier_offsets = (0..=variant_count)
        .map(|index| u32::try_from(index).expect("test variant count fits u32"))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let positions = (0..variant_count)
        .map(|index| i64::try_from(index + 1).expect("test position fits i64"))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let store = Arc::new(
        VariantMetadataStore::from_parts(
            text_dictionary,
            vec![0_u32; variant_count].into_boxed_slice(),
            variant_identifier_text.into_boxed_str(),
            variant_identifier_offsets,
            positions,
            vec![1_u32; variant_count].into_boxed_slice(),
            vec![2_u32; variant_count].into_boxed_slice(),
        )
        .expect("test metadata store should satisfy its invariants"),
    );
    VariantMetadataColumns::new(store, 0..variant_count).expect("test metadata range should be valid")
}

fn build_scheduled_batch(
    variant_start_index: usize,
    logical_variant_count: usize,
    compute_variant_count: usize,
    active_trait_selection: ActiveTraitSelection,
) -> ScheduledAssociationBatch {
    let metadata = build_metadata(logical_variant_count, "22");
    ScheduledAssociationBatch {
        genotypes: GenotypeBatch {
            variant_start_index,
            logical_variant_count,
            compute_variant_count,
            sample_count: TEST_SAMPLE_COUNT,
            payload: GenotypeBatchPayload::Decoded {
                genotypes: OwnedGenotypeBuffer::Dosage(vec![0.0; compute_variant_count * TEST_SAMPLE_COUNT]),
                statistics: ChunkStats {
                    output: build_output_statistics(logical_variant_count),
                    compute: ChunkComputeStatistics {
                        genotype_mean: vec![0.0; compute_variant_count],
                        imputed_dosage_square_sum: Some(vec![0.0; compute_variant_count]),
                        sparse_candidate_mask: Some(vec![false; compute_variant_count]),
                    },
                },
            },
        },
        metadata: NativeVariantMetadataHandle::try_new(&metadata).expect("test metadata is valid"),
        active_trait_selection,
    }
}

fn build_output_statistics(variant_count: usize) -> ChunkOutputStatistics {
    ChunkOutputStatistics {
        allele_one_frequency: vec![0.25; variant_count],
        observation_count: vec![3; variant_count],
        info_score: NullableFloat32Column {
            values: vec![0.75; variant_count],
            validity_bytes: vec![u8::MAX; variant_count.div_ceil(8)],
        },
    }
}

fn build_prediction_matrix(chromosome_state: usize) -> g_input::ChromosomePredictionMatrix {
    g_input::ChromosomePredictionMatrix {
        trait_count: 1,
        sample_count: chromosome_state,
        prediction_values: vec![0.0; chromosome_state],
    }
}

fn drain_pipeline(
    pipeline: &mut AssociationBatchPipeline<TestBackend>,
) -> Vec<crate::association_scheduler::CompletedAssociationBatch> {
    let mut completed_batches = Vec::new();
    while !pipeline.is_drained() {
        completed_batches.push(pipeline.receive().expect("test batch completes"));
    }
    completed_batches
}

fn finish_pipeline(pipeline: &mut AssociationBatchPipeline<TestBackend>) {
    pipeline.release_chromosome().expect("test chromosome is released");
    pipeline.close_submission();
    pipeline.join().expect("test workers join");
}

#[test]
fn scheduler_records_disconnected_event_send() {
    assert_disconnected_event_send_for_test();
}

#[test]
fn scheduler_consumed_first_failure_cannot_be_overwritten() {
    assert_consumed_first_failure_for_test();
}

#[test]
fn scheduler_waits_for_materialization_quiescence_before_cleanup() {
    assert_materialization_quiescence_handshake_for_test();
}

#[test]
fn scheduler_preserves_batch_order_context_and_chromosome_lifecycle() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::None));
    let group = Arc::new(());
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), group).expect("scheduler starts");

    assert!(matches!(pipeline.receive(), Err(SchedulerError::NoPendingBatch)));
    assert!(matches!(
        pipeline.try_submit(build_scheduled_batch(0, 2, 2, ActiveTraitSelection::All)),
        Err(SchedulerError::ChromosomeNotPrepared)
    ));
    pipeline.prepare_chromosome(build_prediction_matrix(17)).expect("chromosome is prepared");
    assert!(matches!(
        pipeline.prepare_chromosome(build_prediction_matrix(18)),
        Err(SchedulerError::ChromosomeAlreadyPrepared)
    ));

    let mut pending_batches = vec![
        build_scheduled_batch(0, 2, 2, ActiveTraitSelection::All),
        build_scheduled_batch(2, 2, 2, ActiveTraitSelection::Indices(vec![1, 0])),
    ];
    let mut completed_batches = Vec::new();
    while let Some(batch) = pending_batches.pop() {
        let mut pending_batch = batch;
        loop {
            match pipeline.try_submit(pending_batch).expect("batch submission succeeds") {
                None => break,
                Some(returned_batch) => {
                    completed_batches.push(pipeline.receive().expect("backpressure drains one batch"));
                    pending_batch = returned_batch;
                }
            }
        }
    }
    completed_batches.extend(drain_pipeline(&mut pipeline));
    completed_batches.sort_by_key(|batch| batch.context.variant_start_index);

    assert_eq!(completed_batches.len(), 2);
    assert_eq!(completed_batches[0].context.variant_start_index, 0);
    assert_eq!(completed_batches[1].context.variant_start_index, 2);
    assert!((completed_batches[0].result.beta[0] - 0.0).abs() < f32::EPSILON);
    assert!((completed_batches[1].result.beta[0] - 2.0).abs() < f32::EPSILON);
    assert_eq!(completed_batches[0].statistics.observation_count, vec![3, 3]);
    assert_eq!(completed_batches[1].context.metadata.row_count(), TEST_VARIANT_COUNT);

    finish_pipeline(&mut pipeline);
    assert_eq!(backend.chromosome_release_count.load(Ordering::SeqCst), 1);
    assert_eq!(backend.transfer_count.load(Ordering::SeqCst), 2);
    assert_eq!(
        *backend.materialized_trait_indices.lock().expect("materialization observations are available"),
        vec![Some(vec![1, 0]), None]
    );
    assert!(matches!(pipeline.try_receive(), Ok(None)));
}

#[test]
fn scheduler_applies_bounded_backpressure_before_a_third_transfer() {
    let (compute_started_sender, compute_started_receiver) = crossbeam_channel::bounded(1);
    let (compute_gate_sender, compute_gate_receiver) = crossbeam_channel::bounded(1);
    let backend = Arc::new(TestBackend::with_first_compute_gate(compute_started_sender, compute_gate_receiver));
    let group = Arc::new(());
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), group).expect("scheduler starts");
    pipeline.prepare_chromosome(build_prediction_matrix(22)).expect("chromosome is prepared");
    let mut compute_gate_release = ComputeGateRelease::new(compute_gate_sender);

    assert!(
        pipeline
            .try_submit(build_scheduled_batch(0, 2, 2, ActiveTraitSelection::All))
            .expect("first batch is submitted")
            .is_none()
    );
    assert_eq!(
        compute_started_receiver
            .recv_timeout(TEST_SYNCHRONIZATION_TIMEOUT)
            .expect("first compute starts before the timeout"),
        0
    );
    assert!(
        pipeline
            .try_submit(build_scheduled_batch(2, 2, 2, ActiveTraitSelection::All))
            .expect("second batch is queued")
            .is_none()
    );
    let returned_batch = pipeline
        .try_submit(build_scheduled_batch(4, 2, 2, ActiveTraitSelection::All))
        .expect("backpressure is reported")
        .expect("third batch is returned before transfer");
    assert_eq!(backend.transfer_count.load(Ordering::SeqCst), 2);

    compute_gate_release.release();
    let mut completed_batches = Vec::new();
    let mut pending_batch = returned_batch;
    loop {
        match pipeline.try_submit(pending_batch).expect("returned batch is retried") {
            None => break,
            Some(returned_again) => {
                completed_batches.push(pipeline.receive().expect("one pending batch completes"));
                pending_batch = returned_again;
            }
        }
    }
    completed_batches.extend(drain_pipeline(&mut pipeline));
    completed_batches.sort_by_key(|batch| batch.context.variant_start_index);
    assert_eq!(
        completed_batches.iter().map(|batch| batch.context.variant_start_index).collect::<Vec<_>>(),
        vec![0, 2, 4]
    );
    assert_eq!(backend.transfer_count.load(Ordering::SeqCst), 3);
    finish_pipeline(&mut pipeline);
}

#[test]
fn delivery_interruption_precedes_group_preparation() {
    let backend = TestBackend::new(TestFailureStage::None);
    let mut input_built = false;
    let error =
        prepare_group_after_interruption_check(&backend, &mut || Err(TestInterruption), || -> GroupPreparationInput {
            input_built = true;
            panic!("group input must not be built after interruption")
        })
        .expect_err("interruption must stop delivery before group preparation");

    assert!(matches!(error, DeliveryError::Interrupted(TestInterruption)));
    assert!(!input_built);
    assert!(!backend.events.lock().expect("test events are available").iter().any(|event| event == "prepare_group"));
}

#[test]
fn delivery_interruption_after_decode_prevents_any_transfer() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::None));
    let group = Arc::new(());
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), group).expect("scheduler starts");
    pipeline.prepare_chromosome(build_prediction_matrix(22)).expect("chromosome is prepared");

    let error = try_submit_after_interruption_check(
        &mut pipeline,
        build_scheduled_batch(0, 2, 2, ActiveTraitSelection::All),
        &mut || Err(TestInterruption),
    )
    .expect_err("interruption after decode must stop the transfer boundary");

    assert!(matches!(error, DeliveryError::Interrupted(TestInterruption)));
    assert_eq!(backend.transfer_count.load(Ordering::SeqCst), 0);
    assert!(
        !backend.events.lock().expect("test events are available").iter().any(|event| event.starts_with("transfer:"))
    );
}

#[test]
fn delivery_interruption_after_full_queue_receive_stops_pending_batch_launch() {
    let (compute_started_sender, compute_started_receiver) = crossbeam_channel::bounded(1);
    let (compute_gate_sender, compute_gate_receiver) = crossbeam_channel::bounded(1);
    let backend = Arc::new(TestBackend::with_first_compute_gate(compute_started_sender, compute_gate_receiver));
    let group = Arc::new(());
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), group).expect("scheduler starts");
    pipeline.prepare_chromosome(build_prediction_matrix(22)).expect("chromosome is prepared");
    let mut compute_gate_release = ComputeGateRelease::new(compute_gate_sender);

    assert!(
        pipeline
            .try_submit(build_scheduled_batch(0, 2, 2, ActiveTraitSelection::All))
            .expect("first batch is submitted")
            .is_none()
    );
    assert_eq!(
        compute_started_receiver
            .recv_timeout(TEST_SYNCHRONIZATION_TIMEOUT)
            .expect("first compute starts before the timeout"),
        0
    );
    assert!(
        pipeline
            .try_submit(build_scheduled_batch(2, 2, 2, ActiveTraitSelection::All))
            .expect("second batch is queued")
            .is_none()
    );
    let pending_batch = pipeline
        .try_submit(build_scheduled_batch(4, 2, 2, ActiveTraitSelection::All))
        .expect("full queue reports backpressure")
        .expect("third batch remains pending before transfer");
    assert_eq!(backend.transfer_count.load(Ordering::SeqCst), 2);

    compute_gate_release.release();
    let mut interruption_check_count = 0_usize;
    let error = retry_pending_batch(
        &mut pipeline,
        pending_batch,
        &mut || {
            interruption_check_count += 1;
            Err(TestInterruption)
        },
        &mut |_completed_batch| -> Result<(), DeliveryError<TestBackendError, TestInterruption>> {
            panic!("interruption must precede completed-batch acceptance")
        },
    )
    .expect_err("interruption after the receive stops the pending submission");
    assert!(matches!(error, DeliveryError::Interrupted(TestInterruption)));
    assert_eq!(interruption_check_count, 1);

    drop(pipeline);
    assert_eq!(backend.transfer_count.load(Ordering::SeqCst), 2);
    let events = backend.events.lock().expect("test events are available");
    assert!(!events.iter().any(|event| event == "transfer:4" || event.ends_with(":4")));
}

#[test]
fn delivery_interruption_at_chromosome_boundary_stops_next_prepare() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::None));
    let group = Arc::new(());
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), group).expect("scheduler starts");
    pipeline.prepare_chromosome(build_prediction_matrix(21)).expect("first chromosome is prepared");
    let mut interruption_check_count = 0_usize;
    let mut next_prepare_launched = false;

    let error = transition_drained_chromosome(
        &mut pipeline,
        &mut || {
            interruption_check_count += 1;
            Err(TestInterruption)
        },
        |_pipeline| -> Result<(), DeliveryError<TestBackendError, TestInterruption>> {
            next_prepare_launched = true;
            Ok(())
        },
    )
    .expect_err("interruption prevents the next chromosome preparation");

    assert!(matches!(error, DeliveryError::Interrupted(TestInterruption)));
    assert_eq!(interruption_check_count, 1);
    assert!(!next_prepare_launched);
    assert_eq!(backend.chromosome_release_count.load(Ordering::SeqCst), 1);
    assert_eq!(
        backend
            .events
            .lock()
            .expect("test events are available")
            .iter()
            .filter(|event| *event == "prepare_chromosome")
            .count(),
        1
    );
    pipeline.close_submission();
    pipeline.join().expect("interrupted boundary leaves workers joinable");
}

#[test]
fn scheduler_reuses_steady_state_workers_across_drained_chromosome_lifecycles() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::None));
    let group = Arc::new(());
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), group).expect("scheduler starts");

    for (chromosome_state, variant_start_index) in [(21, 0), (22, 2)] {
        pipeline.prepare_chromosome(build_prediction_matrix(chromosome_state)).expect("chromosome is prepared");
        assert!(
            pipeline
                .try_submit(build_scheduled_batch(
                    variant_start_index,
                    TEST_VARIANT_COUNT,
                    TEST_VARIANT_COUNT,
                    ActiveTraitSelection::All,
                ))
                .expect("batch is submitted")
                .is_none()
        );
        let completed = pipeline.receive().expect("batch completes");
        assert_eq!(completed.context.variant_start_index, variant_start_index);
        pipeline.release_chromosome().expect("drained chromosome is released");
    }

    pipeline.close_submission();
    pipeline.join().expect("reused workers join");
    assert_eq!(backend.chromosome_release_count.load(Ordering::SeqCst), 2);
    let events = backend.events.lock().expect("test events are available");
    let compute_events = events.iter().filter(|event| event.starts_with("compute:")).collect::<Vec<_>>();
    assert_eq!(compute_events, vec!["compute:21:0", "compute:22:2"]);
}

#[test]
fn scheduler_runs_chromosome_lifecycle_on_one_backend_worker() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::None));
    let group = Arc::new(());
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), group).expect("scheduler starts");
    let caller_thread_identifier = thread::current().id();

    pipeline.prepare_chromosome(build_prediction_matrix(22)).expect("chromosome is prepared");
    assert!(
        pipeline
            .try_submit(build_scheduled_batch(0, TEST_VARIANT_COUNT, TEST_VARIANT_COUNT, ActiveTraitSelection::All,))
            .expect("batch is submitted")
            .is_none()
    );
    pipeline.receive().expect("batch completes");
    finish_pipeline(&mut pipeline);

    let backend_thread_events =
        backend.backend_thread_events.lock().expect("test backend-thread observations are available");
    assert_eq!(
        backend_thread_events.iter().map(|event| event.operation).collect::<Vec<_>>(),
        vec!["prepare", "compute", "release"]
    );
    let execution_thread_identifier = backend_thread_events[0].thread_identifier;
    assert_ne!(execution_thread_identifier, caller_thread_identifier);
    assert!(backend_thread_events.iter().all(|event| event.thread_identifier == execution_thread_identifier));
}

#[test]
fn scheduler_accepts_tail_padded_packed8_batches_without_optional_compute_columns() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::None));
    let group = Arc::new(());
    let mut pipeline = AssociationBatchPipeline::new(backend, group).expect("scheduler starts");
    pipeline.prepare_chromosome(build_prediction_matrix(22)).expect("chromosome is prepared");
    let mut batch = build_scheduled_batch(8, 1, 2, ActiveTraitSelection::All);
    let GenotypeBatchPayload::Decoded { genotypes, statistics } = &mut batch.genotypes.payload else {
        unreachable!("test builds decoded genotypes")
    };
    *genotypes = OwnedGenotypeBuffer::Packed8(vec![0_u8; 2 * TEST_SAMPLE_COUNT * 2].into());
    statistics.compute.imputed_dosage_square_sum = None;
    statistics.compute.sparse_candidate_mask = None;
    assert!(pipeline.try_submit(batch).expect("packed8 batch is submitted").is_none());
    let completed = pipeline.receive().expect("packed8 batch completes");
    assert_eq!(completed.context.variant_start_index, 8);
    assert_eq!(completed.context.metadata.row_count(), 1);
    finish_pipeline(&mut pipeline);
}

#[test]
fn scheduler_propagates_prepare_failure_without_releasing_uncreated_state() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::Prepare));
    let group = Arc::new(());
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), group).expect("scheduler starts");

    let error = pipeline.prepare_chromosome(build_prediction_matrix(1)).expect_err("prepare failure reaches caller");
    assert!(matches!(
        error,
        SchedulerError::Backend { stage: "prepare chromosome", source: TestBackendError("prepare") }
    ));
    pipeline.close_submission();
    assert!(matches!(pipeline.join(), Err(SchedulerError::Aborted)));
    drop(pipeline);
    assert_eq!(backend.chromosome_release_count.load(Ordering::SeqCst), 0);
}

#[test]
fn scheduler_propagates_each_backend_failure_stage() {
    let failure_cases = [
        (TestFailureStage::Transfer, "device transfer", false),
        (TestFailureStage::Compute, "compute", true),
        (TestFailureStage::Materialize, "materialization", true),
    ];
    for (failure_stage, expected_stage, asynchronously_submitted) in failure_cases {
        let backend = Arc::new(TestBackend::new(failure_stage));
        let group = Arc::new(());
        let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), group).expect("scheduler starts");
        pipeline.prepare_chromosome(build_prediction_matrix(1)).expect("chromosome is prepared");
        let submission = pipeline.try_submit(build_scheduled_batch(0, 2, 2, ActiveTraitSelection::All));
        let error = if asynchronously_submitted {
            assert!(submission.expect("asynchronous batch submission succeeds").is_none());
            pipeline.receive().expect_err("worker failure reaches receiver")
        } else {
            submission.expect_err("transfer failure reaches submitter")
        };
        assert!(matches!(
            error,
            SchedulerError::Backend { stage, source: TestBackendError(_) }
                if stage == expected_stage
        ));
        drop(pipeline);
        assert_eq!(backend.chromosome_release_count.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn scheduler_waits_for_in_flight_materialization_before_failure_cleanup() {
    let (materialization_started_sender, materialization_started_receiver) = crossbeam_channel::bounded(1);
    let (materialization_gate_sender, materialization_gate_receiver) = crossbeam_channel::bounded(1);
    let (chromosome_released_sender, chromosome_released_receiver) = crossbeam_channel::bounded(1);
    let backend = Arc::new(TestBackend::with_materialization_gate_and_compute_failure(
        2,
        materialization_started_sender,
        materialization_gate_receiver,
        chromosome_released_sender,
    ));
    let group = Arc::new(());
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), group).expect("scheduler starts");
    pipeline.prepare_chromosome(build_prediction_matrix(7)).expect("chromosome is prepared");
    assert!(
        pipeline
            .try_submit(build_scheduled_batch(0, 2, 2, ActiveTraitSelection::All))
            .expect("first batch is submitted")
            .is_none()
    );
    assert_eq!(
        materialization_started_receiver
            .recv_timeout(TEST_SYNCHRONIZATION_TIMEOUT)
            .expect("first batch enters materialization before the timeout"),
        0
    );
    assert!(
        pipeline
            .try_submit(build_scheduled_batch(2, 2, 2, ActiveTraitSelection::All))
            .expect("failing second batch is submitted")
            .is_none()
    );

    let error = pipeline.receive().expect_err("second-batch compute failure reaches the receiver");
    assert!(matches!(error, SchedulerError::Backend { stage: "compute", source: TestBackendError("compute") }));
    pipeline.wait_for_materialization_quiescence_for_test();
    assert!(matches!(chromosome_released_receiver.try_recv(), Err(crossbeam_channel::TryRecvError::Empty)));
    assert_eq!(backend.chromosome_release_count.load(Ordering::SeqCst), 0);

    materialization_gate_sender
        .send_timeout((), TEST_SYNCHRONIZATION_TIMEOUT)
        .expect("materialization gate accepts its release signal before the timeout");
    chromosome_released_receiver
        .recv_timeout(TEST_SYNCHRONIZATION_TIMEOUT)
        .expect("chromosome release follows materialization quiescence");
    drop(pipeline);

    assert_eq!(backend.chromosome_release_count.load(Ordering::SeqCst), 1);
    let events = backend.events.lock().expect("test events are available");
    let materialization_end_index =
        events.iter().position(|event| event == "materialize_end:0").expect("materialization completion is recorded");
    let chromosome_release_index =
        events.iter().position(|event| event == "release:7").expect("chromosome release is recorded");
    assert!(materialization_end_index < chromosome_release_index);
    let backend_thread_events =
        backend.backend_thread_events.lock().expect("test backend-thread observations are available");
    let prepare_thread_identifier = backend_thread_events
        .iter()
        .find(|event| event.operation == "prepare")
        .expect("prepare thread is recorded")
        .thread_identifier;
    let release_thread_identifier = backend_thread_events
        .iter()
        .find(|event| event.operation == "release")
        .expect("release thread is recorded")
        .thread_identifier;
    assert_eq!(release_thread_identifier, prepare_thread_identifier);
}

#[test]
fn scheduler_reports_worker_panics_and_releases_owned_chromosome_state() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::ComputePanic));
    let group = Arc::new(());
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), group).expect("scheduler starts");
    pipeline.prepare_chromosome(build_prediction_matrix(9)).expect("chromosome is prepared");
    assert!(
        pipeline
            .try_submit(build_scheduled_batch(0, 2, 2, ActiveTraitSelection::All))
            .expect("batch is submitted before worker panic")
            .is_none()
    );
    let error = pipeline.receive().expect_err("worker panic reaches receiver");
    assert!(matches!(
        error,
        SchedulerError::WorkerPanicked { worker: "compute", message }
            if message.contains("intentional compute panic")
    ));
    drop(pipeline);
    assert_eq!(backend.chromosome_release_count.load(Ordering::SeqCst), 1);
}

#[test]
fn scheduler_preserves_compute_panic_when_release_also_panics() {
    let backend = Arc::new(TestBackend::with_release_panic(TestFailureStage::ComputePanic));
    let group = Arc::new(());
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), group).expect("scheduler starts");
    pipeline.prepare_chromosome(build_prediction_matrix(9)).expect("chromosome is prepared");
    assert!(
        pipeline
            .try_submit(build_scheduled_batch(0, 2, 2, ActiveTraitSelection::All))
            .expect("batch is submitted before worker panic")
            .is_none()
    );

    let error = pipeline.receive().expect_err("worker panic reaches receiver");
    assert!(matches!(
        error,
        SchedulerError::WorkerPanicked { worker: "compute", message }
            if message.contains("intentional compute panic")
    ));
    drop(pipeline);
    assert_eq!(backend.chromosome_release_count.load(Ordering::SeqCst), 1);
}

#[test]
fn scheduler_releases_chromosome_after_materialization_panic() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::MaterializePanic));
    let group = Arc::new(());
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), group).expect("scheduler starts");
    pipeline.prepare_chromosome(build_prediction_matrix(7)).expect("chromosome is prepared");
    assert!(
        pipeline
            .try_submit(build_scheduled_batch(0, 2, 2, ActiveTraitSelection::All))
            .expect("batch is submitted before worker panic")
            .is_none()
    );

    let error = pipeline.receive().expect_err("materialization panic reaches receiver");
    assert!(matches!(
        error,
        SchedulerError::WorkerPanicked { worker: "materialization", message }
            if message.contains("intentional materialization panic")
    ));
    drop(pipeline);
    assert_eq!(backend.chromosome_release_count.load(Ordering::SeqCst), 1);
    let backend_thread_events =
        backend.backend_thread_events.lock().expect("test backend-thread observations are available");
    let prepare_thread_identifier = backend_thread_events
        .iter()
        .find(|event| event.operation == "prepare")
        .expect("prepare thread is recorded")
        .thread_identifier;
    let release_thread_identifier = backend_thread_events
        .iter()
        .find(|event| event.operation == "release")
        .expect("release thread is recorded")
        .thread_identifier;
    assert_eq!(release_thread_identifier, prepare_thread_identifier);
}

#[test]
fn dropping_scheduler_cancels_workers_and_releases_chromosome_state() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::None));
    let group = Arc::new(());
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), group).expect("scheduler starts");
    pipeline.prepare_chromosome(build_prediction_matrix(5)).expect("chromosome is prepared");
    drop(pipeline);
    assert_eq!(backend.chromosome_release_count.load(Ordering::SeqCst), 1);
}

#[test]
fn scheduler_rejects_invalid_lifecycle_transitions() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::None));
    let group = Arc::new(());
    let mut pipeline = AssociationBatchPipeline::new(backend, group).expect("scheduler starts");
    assert!(matches!(pipeline.join(), Err(SchedulerError::SubmissionOpen)));
    pipeline.prepare_chromosome(build_prediction_matrix(3)).expect("chromosome is prepared");
    assert!(
        pipeline
            .try_submit(build_scheduled_batch(0, 2, 2, ActiveTraitSelection::All))
            .expect("batch is submitted")
            .is_none()
    );
    assert!(matches!(
        pipeline.release_chromosome(),
        Err(SchedulerError::ChromosomeTransitionPending { submitted: 1, completed: 0 })
    ));
    pipeline.close_submission();
    assert!(matches!(pipeline.join(), Err(SchedulerError::PendingBatches { submitted: 1, completed: 0 })));
    let completed = pipeline.receive().expect("submitted batch can still drain after close");
    assert_eq!(completed.context.variant_start_index, 0);
    pipeline.join().expect("drained closed scheduler joins");
    assert!(matches!(
        pipeline.try_submit(build_scheduled_batch(2, 2, 2, ActiveTraitSelection::All)),
        Err(SchedulerError::Closed)
    ));
}

fn assert_invalid_scheduled_batch(batch: ScheduledAssociationBatch, expected_message_fragment: &str) {
    let backend = Arc::new(TestBackend::new(TestFailureStage::None));
    let group = Arc::new(());
    let mut pipeline = AssociationBatchPipeline::new(backend, group).expect("scheduler starts");
    let error = pipeline.try_submit(batch).expect_err("invalid batch is rejected before submission");
    let SchedulerError::InvalidBatch { message } = error else {
        panic!("expected invalid-batch error, observed {error}");
    };
    assert!(
        message.contains(expected_message_fragment),
        "expected error containing {expected_message_fragment:?}, observed {message:?}"
    );
    pipeline.close_submission();
    pipeline.join().expect("idle workers join");
}

#[test]
fn scheduler_validates_batch_shapes_before_transfer() {
    let mut metadata_mismatch = build_scheduled_batch(0, 2, 2, ActiveTraitSelection::All);
    metadata_mismatch.metadata =
        NativeVariantMetadataHandle::try_new(&build_metadata(1, "22")).expect("test metadata is valid");
    assert_invalid_scheduled_batch(metadata_mismatch, "metadata contains 1 variants");

    assert_invalid_scheduled_batch(
        build_scheduled_batch(0, 2, 1, ActiveTraitSelection::All),
        "compute variant count 1 is smaller",
    );

    let mut wrong_genotype_count = build_scheduled_batch(0, 2, 2, ActiveTraitSelection::All);
    let GenotypeBatchPayload::Decoded { genotypes, .. } = &mut wrong_genotype_count.genotypes.payload else {
        unreachable!("test builds decoded genotypes")
    };
    *genotypes = OwnedGenotypeBuffer::Dosage(vec![0.0; 5]);
    assert_invalid_scheduled_batch(wrong_genotype_count, "genotype buffer contains 5 values, expected 6");

    let mut wrong_info_bitmap = build_scheduled_batch(0, 2, 2, ActiveTraitSelection::All);
    let GenotypeBatchPayload::Decoded { statistics, .. } = &mut wrong_info_bitmap.genotypes.payload else {
        unreachable!("test builds decoded statistics")
    };
    statistics.output.info_score.validity_bytes.clear();
    assert_invalid_scheduled_batch(wrong_info_bitmap, "INFO validity bitmap contains 0 values, expected 1");

    let metadata = build_metadata(1, "22");
    let overflow_batch = ScheduledAssociationBatch {
        genotypes: GenotypeBatch {
            variant_start_index: 0,
            logical_variant_count: 1,
            compute_variant_count: usize::MAX,
            sample_count: 2,
            payload: GenotypeBatchPayload::Decoded {
                genotypes: OwnedGenotypeBuffer::Dosage(Vec::new()),
                statistics: ChunkStats {
                    output: build_output_statistics(1),
                    compute: ChunkComputeStatistics {
                        genotype_mean: Vec::new(),
                        imputed_dosage_square_sum: None,
                        sparse_candidate_mask: None,
                    },
                },
            },
        },
        metadata: NativeVariantMetadataHandle::try_new(&metadata).expect("test metadata is valid"),
        active_trait_selection: ActiveTraitSelection::All,
    };
    assert_invalid_scheduled_batch(overflow_batch, "variant and sample counts overflow");
}

#[test]
fn preflight_accepts_full_rank_finite_quantitative_and_binary_inputs() {
    let covariates = vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
    validate_multi_trait_preflight_values(2, 4, &[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], 4, 2, &covariates, false)
        .expect("finite quantitative input passes preflight");
    validate_multi_trait_preflight_values(2, 4, &[-0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], 4, 2, &covariates, true)
        .expect("binary input with both classes per trait passes preflight");
    validate_multi_trait_preflight_values(1, 2, &[0.0, 1.0], 2, 0, &[], true)
        .expect("an empty covariate design does not invoke an empty-matrix SVD");
}

#[test]
fn preflight_rejects_shape_finiteness_rank_and_binary_contract_violations() {
    let valid_covariates = [1.0, 0.0, 1.0, 1.0];
    let cases = [
        validate_multi_trait_preflight_values(0, 2, &[], 2, 0, &[], false),
        validate_multi_trait_preflight_values(1, 0, &[], 0, 0, &[], false),
        validate_multi_trait_preflight_values(1, 2, &[0.0, 1.0], 1, 0, &[], false),
        validate_multi_trait_preflight_values(1, 2, &[0.0, 1.0], 2, 2, &valid_covariates, false),
        validate_multi_trait_preflight_values(usize::MAX, 2, &[], 2, 0, &[], false),
        validate_multi_trait_preflight_values(1, 2, &[0.0], 2, 0, &[], false),
        validate_multi_trait_preflight_values(1, 2, &[0.0, 1.0], 2, 1, &[1.0], false),
        validate_multi_trait_preflight_values(1, 2, &[0.0, f32::NAN], 2, 0, &[], false),
        validate_multi_trait_preflight_values(1, 2, &[0.0, 1.0], 2, 1, &[1.0, f32::INFINITY], false),
        validate_multi_trait_preflight_values(1, 3, &[0.0, 1.0, 2.0], 3, 2, &[1.0; 6], false),
        validate_multi_trait_preflight_values(1, 2, &[0.0, 0.5], 2, 0, &[], true),
        validate_multi_trait_preflight_values(1, 2, &[1.0, 1.0], 2, 0, &[], true),
    ];
    let expected_errors = [
        PreflightError::EmptyPhenotypeTraitSet,
        PreflightError::EmptyPhenotypeSampleSet,
        PreflightError::CovariateSampleCountMismatch,
        PreflightError::NonPositiveResidualDegreesOfFreedom,
        PreflightError::PhenotypeMatrixShapeOverflow,
        PreflightError::PhenotypeMatrixValueCountMismatch,
        PreflightError::CovariateMatrixValueCountMismatch,
        PreflightError::NonFiniteArray { label: "Phenotype matrix".to_string() },
        PreflightError::NonFiniteArray { label: "Covariate matrix".to_string() },
        PreflightError::CovariateMatrixRankDeficient,
        PreflightError::BinaryPhenotypeCoding,
        PreflightError::BinaryPhenotypeMissingClass,
    ];
    for (case, expected_error) in cases.into_iter().zip(expected_errors) {
        assert_eq!(case.expect_err("invalid preflight case is rejected"), expected_error);
    }
}

#[test]
fn jax_capacity_validation_covers_scalar_flattened_and_padded_domains() {
    validate_jax_index_capacity(4, 500_000, 16_384, 1_024, 512, true)
        .expect("production-scale dimensions fit JAX index domain");
    validate_jax_index_capacity(4, 500_000, 16_384, 1_024, 512, false)
        .expect("quantitative dimensions skip Firth capacity checks");

    let maximum_index_count = usize::try_from(i32::MAX).expect("64-bit target represents i32::MAX");
    assert_eq!(
        validate_jax_index_capacity(maximum_index_count + 1, 1, 1, 1, 1, false)
            .expect_err("trait count above int32 is rejected"),
        PreflightError::JaxIndexCapacityExceeded { label: "trait count" }
    );
    assert_eq!(
        validate_jax_index_capacity(50_000, 1, 50_000, 1, 1, false)
            .expect_err("flattened lanes above int32 are rejected"),
        PreflightError::JaxIndexCapacityExceeded { label: "flattened trait-by-chunk lane count" }
    );
    assert_eq!(
        validate_jax_index_capacity(1, 1, maximum_index_count, maximum_index_count, 2, true)
            .expect_err("padded candidates above int32 are rejected"),
        PreflightError::JaxIndexCapacityExceeded { label: "padded Firth candidate capacity" }
    );
}

#[test]
fn null_logistic_policy_handles_convergence_warnings_failures_and_names() {
    let names = vec!["trait-a".to_string(), "trait-b".to_string(), "trait-c".to_string()];
    let converged = plan_null_logistic_nonconvergence("22", &[true, true, true], false, Some(&names), "fail")
        .expect("converged multi-trait plan succeeds");
    assert_eq!(converged.action, NullLogisticNonconvergenceAction::Continue);
    assert!(converged.failed_trait_indices.is_empty());
    assert!(converged.message.is_none());

    let warning = plan_null_logistic_nonconvergence("22", &[false, true, false], false, Some(&names), "warn")
        .expect("warning plan succeeds");
    assert_eq!(warning.action, NullLogisticNonconvergenceAction::Warn);
    assert_eq!(warning.failed_trait_indices, vec![0, 2]);
    assert_eq!(warning.nonconverged_count, 2);
    assert_eq!(warning.total_fit_count, 3);
    assert!(warning.message.as_deref().is_some_and(|message| message.contains("trait-a, trait-c")));
    assert!(warning.warning_message.as_deref().is_some_and(|message| message.contains("Continuing")));

    let failure =
        plan_null_logistic_nonconvergence("7", &[false], true, None, "fail").expect("scalar failure plan succeeds");
    assert_eq!(failure.action, NullLogisticNonconvergenceAction::Fail);
    assert!(failure.scalar_convergence);
    assert_eq!(failure.failed_trait_indices, vec![0]);
    assert!(failure.message.as_deref().is_some_and(|message| message.contains("chromosome 7")));
}

#[test]
fn null_logistic_policy_rejects_invalid_flag_and_policy_shapes() {
    let names = vec!["trait-a".to_string()];
    assert_eq!(
        plan_null_logistic_nonconvergence("1", &[], false, None, "fail")
            .expect_err("empty convergence flags are rejected"),
        NullLogisticPolicyError::EmptyConvergenceFlags
    );
    assert_eq!(
        plan_null_logistic_nonconvergence("1", &[true, false], true, None, "fail")
            .expect_err("scalar convergence requires one flag"),
        NullLogisticPolicyError::ScalarConvergenceFlagCount { observed_count: 2 }
    );
    assert_eq!(
        plan_null_logistic_nonconvergence("1", &[true, false], false, Some(&names), "warn")
            .expect_err("phenotype names must match flags"),
        NullLogisticPolicyError::PhenotypeNameCountMismatch { phenotype_name_count: 1, convergence_flag_count: 2 }
    );
    assert_eq!(
        plan_null_logistic_nonconvergence("1", &[true], false, None, "ignore").expect_err("unknown policy is rejected"),
        NullLogisticPolicyError::UnsupportedNullLogisticPolicy { policy: "ignore".to_string() }
    );
}

#[test]
fn output_schedule_intersects_resume_state_and_selects_active_traits() {
    let committed_sets = vec![
        Arc::new(BTreeSet::from([0, 2, 4, 6])),
        Arc::new(BTreeSet::from([2, 4])),
        Arc::new(BTreeSet::from([1, 2, 4, 5])),
    ];
    assert_eq!(intersect_committed_chunk_identifier_sets(&committed_sets), BTreeSet::from([2, 4]));
    assert!(intersect_committed_chunk_identifier_sets::<usize>(&[]).is_empty());

    assert!(matches!(
        active_trait_selection_for_chunk(3, 3, &committed_sets).expect("uncommitted chunk is planned"),
        ActiveTraitSelection::All
    ));
    assert!(matches!(
        active_trait_selection_for_chunk(3, 0, &committed_sets).expect("partly committed chunk is planned"),
        ActiveTraitSelection::Indices(indices) if indices == vec![1, 2]
    ));
    assert!(matches!(
        active_trait_selection_for_chunk(3, 2, &committed_sets).expect("fully committed chunk is recognized"),
        ActiveTraitSelection::Indices(indices) if indices.is_empty()
    ));
    let error =
        active_trait_selection_for_chunk(2, 0, &committed_sets).expect_err("writer and commit-set counts must match");
    assert!(error.contains("set count (3) must match writer session count (2)"));
}

#[test]
fn homogeneous_chunk_validation_preserves_shared_chromosome_ownership() {
    let metadata = build_metadata(3, "chr22");
    let chromosome = homogeneous_chunk_chromosome(&metadata, 3).expect("homogeneous metadata is accepted");
    assert_eq!(chromosome.as_ref(), "chr22");
    assert!(homogeneous_chunk_chromosome(&metadata, 0).is_err());
    assert!(homogeneous_chunk_chromosome(&metadata, 2).is_err());
}

fn valid_run_plan() -> g_plan::RunPlan {
    g_plan::RunPlan {
        association_mode: g_plan::AssociationMode::Regenie2Binary,
        chunk_size: 16_384,
        input: g_plan::InputPlan {
            bgen_path: "input.bgen".to_string(),
            bgen_content_sha256: None,
            sample_path: "input.sample".to_string(),
            phenotype_path: "phenotype.tsv".to_string(),
            prediction_list_path: "predictions.list".to_string(),
            covariate_path: None,
            covariate_names: vec!["intercept".to_string()],
        },
        compute: g_plan::ComputePlan {
            device: g_plan::Device::Gpu,
            cpu_thread_count: None,
            jax_cache_directory: None,
            multi_phenotype_sample_mode: g_plan::MultiPhenotypeSampleMode::CompleteCase,
            kernels: g_plan::KernelPlan {
                linear: g_plan::LinearKernelPlan {
                    minimum_variance: g_plan::PositiveF32::try_from(1.0e-8).expect("positive test value"),
                    relative_variance_tolerance: g_plan::PositiveF32::try_from(1.0e-5).expect("positive test value"),
                },
                binary_null: g_plan::BinaryNullKernelPlan {
                    maximum_iterations: 100,
                    coefficient_tolerance: g_plan::PositiveF32::try_from(1.0e-5).expect("positive test value"),
                    nonconvergence_policy: g_plan::NullLogisticNonconvergencePolicy::Fail,
                    minimum_probability: g_plan::ProbabilityFloor::try_from(1.0e-7)
                        .expect("probability floor test value"),
                    minimum_variance: g_plan::PositiveF32::try_from(1.0e-8).expect("positive test value"),
                    relative_variance_tolerance: g_plan::PositiveF32::try_from(1.0e-5).expect("positive test value"),
                },
                firth: g_plan::FirthKernelPlan {
                    batch_size: 512,
                    candidate_capacity: 1_024,
                    maximum_iterations: 100,
                    gradient_tolerance: g_plan::PositiveF64::try_from(1.0e-6).expect("positive test value"),
                    maximum_step_size: g_plan::PositiveF64::try_from(5.0).expect("positive test value"),
                    pseudo_maximum_iterations: 100,
                    pseudo_inner_maximum_iterations: 100,
                    line_search_maximum_attempts: 25,
                    sparse_carrier_dosage_threshold: g_plan::DosageThreshold::try_from(0.5)
                        .expect("dosage threshold test value"),
                },
                null_firth: g_plan::NullFirthKernelPlan {
                    maximum_iterations: 100,
                    gradient_tolerance: g_plan::PositiveF64::try_from(1.0e-6).expect("positive test value"),
                    maximum_step_size: g_plan::PositiveF64::try_from(5.0).expect("positive test value"),
                    fallback_iteration_multiplier: 2,
                    fallback_step_divisor: g_plan::PositiveF64::try_from(2.0).expect("positive test value"),
                    line_search_maximum_attempts: 25,
                    step_halving_scale: g_plan::StepScale::try_from(0.5).expect("step scale test value"),
                },
            },
        },
        correction: g_plan::CorrectionPlan {
            method: g_plan::BinaryFallbackMethod::FirthApproximate,
            p_threshold: g_plan::Probability::try_from(0.05).expect("probability test value"),
            firth_se: false,
        },
        output: g_plan::OutputPlan {
            output_run_root: "output".to_string(),
            resume: false,
            recover_attempt: None,
            fenced_owner_claim_id: None,
            writer_thread_count: 8,
        },
        telemetry: g_plan::TelemetryMode::Off,
        phenotype_runs: vec![g_plan::PhenotypeRunPlan {
            phenotype_name: "trait".to_string(),
            output_directory_name: "0001-trait".to_string(),
        }],
    }
}

#[test]
fn run_plan_jax_integer_validation_accepts_production_values_and_rejects_boundaries() {
    validate_jax_integer_domain(&valid_run_plan()).expect("production-sized plan fits JAX integers");

    let mut zero_chunk_plan = valid_run_plan();
    zero_chunk_plan.chunk_size = 0;
    assert!(matches!(
        validate_jax_integer_domain(&zero_chunk_plan),
        Err(RunPreparationError::NonPositiveCapacity { field_name: "analysis chunk size" })
    ));

    let mut oversized_batch_plan = valid_run_plan();
    oversized_batch_plan.compute.kernels.firth.batch_size = i32::MAX.cast_unsigned() + 1;
    assert!(matches!(
        validate_jax_integer_domain(&oversized_batch_plan),
        Err(RunPreparationError::JaxIntegerOverflow { field_name: "Firth batch size" })
    ));

    let mut fallback_product_plan = valid_run_plan();
    fallback_product_plan.compute.kernels.null_firth.maximum_iterations = i32::MAX.cast_unsigned();
    fallback_product_plan.compute.kernels.null_firth.fallback_iteration_multiplier = 2;
    assert!(matches!(
        validate_jax_integer_domain(&fallback_product_plan),
        Err(RunPreparationError::JaxIntegerOverflow { field_name: "null Firth fallback iteration limit" })
    ));
}

fn test_phenotype_compute_group(indices: Vec<u32>, names: Vec<&str>) -> g_plan::PhenotypeComputeGroup {
    g_plan::PhenotypeComputeGroup {
        group_mode: g_plan::PhenotypeComputeGroupMode::CompleteCase,
        phenotype_indices: indices,
        phenotype_names: names.into_iter().map(str::to_string).collect(),
        sample_mode: g_plan::MultiPhenotypeSampleMode::CompleteCase,
        sample_set_fingerprint: "sample-set".to_string(),
        covariate_design_fingerprint: "covariates".to_string(),
        phenotype_design_fingerprint: "phenotypes".to_string(),
        prediction_alignment_fingerprint: "predictions".to_string(),
    }
}

#[test]
fn runtime_output_preparation_reuses_identity_fingerprints_and_validates_subsets() {
    let temporary_root = std::env::temp_dir().join(format!("g-engine-preparation-{}", std::process::id()));
    std::fs::create_dir_all(&temporary_root).expect("test temporary directory is created");
    let first_path = temporary_root.join("first.loco");
    let second_path = temporary_root.join("second.loco");
    std::fs::write(&first_path, b"first").expect("first prediction fixture is written");
    std::fs::write(&second_path, b"second").expect("second prediction fixture is written");
    let mut fingerprint_cache = ManifestFileFingerprintCache::default();
    let fingerprints: Arc<[g_output::PredictionLocoFileFingerprint]> = vec![
        fingerprint_cache
            .build_prediction_loco_file_fingerprint(Arc::from("trait-a"), &first_path)
            .expect("first fingerprint is built"),
        fingerprint_cache
            .build_prediction_loco_file_fingerprint(Arc::from("trait-b"), &second_path)
            .expect("second fingerprint is built"),
    ]
    .into();
    let runtime_plan = RuntimeOutputPlan {
        variant_count: 418_943,
        resolved_gpu_genotype_format: g_plan::GpuGenotypeFormat::Packed8,
        bgen_source_identity: Arc::new(BgenSourceIdentity {
            configured_path: "input.bgen".into(),
            canonical_path: None,
            device_identifier: 1,
            inode_identifier: 2,
            change_time_nanoseconds: 3,
            modification_time_nanoseconds: 4,
            file_size: 5,
        }),
    };
    let identity_group = test_phenotype_compute_group(vec![0, 1], vec!["trait-a", "trait-b"]);
    let identity_initializations = build_runtime_output_initializations(
        &RuntimeOutputGroupInput {
            phenotype_group: &identity_group,
            covariate_names: &["age".to_string(), "sex".to_string()],
            sample_count: 500_000,
        },
        &runtime_plan,
        &fingerprints,
    )
    .expect("identity output preparation succeeds");
    assert_eq!(identity_initializations.len(), 2);
    assert!(Arc::ptr_eq(&identity_initializations[0].prediction_loco_files, &fingerprints));
    assert_eq!(identity_initializations[0].sample_count, 500_000);
    assert_eq!(identity_initializations[0].variant_count, 418_943);
    assert_eq!(identity_initializations[1].phenotype_name, "trait-b");

    let subset_group = test_phenotype_compute_group(vec![1], vec!["trait-b"]);
    let subset_initializations = build_runtime_output_initializations(
        &RuntimeOutputGroupInput { phenotype_group: &subset_group, covariate_names: &[], sample_count: 10 },
        &runtime_plan,
        &fingerprints,
    )
    .expect("subset output preparation succeeds");
    assert_eq!(subset_initializations[0].prediction_loco_files.len(), 1);
    assert!(!Arc::ptr_eq(&subset_initializations[0].prediction_loco_files, &fingerprints));

    let missing_group = test_phenotype_compute_group(vec![2], vec!["missing"]);
    assert!(matches!(
        build_runtime_output_initializations(
            &RuntimeOutputGroupInput { phenotype_group: &missing_group, covariate_names: &[], sample_count: 1 },
            &runtime_plan,
            &fingerprints,
        ),
        Err(PipelineOutputPreparationError::MissingPredictionLocoFile { phenotype_index: 2 })
    ));
    std::fs::remove_dir_all(&temporary_root).expect("test temporary directory is removed");
}

#[test]
fn prediction_manifest_fingerprints_cover_every_input_and_reuse_cache() {
    let temporary_root = std::env::temp_dir().join(format!("g-engine-manifest-{}", std::process::id()));
    std::fs::create_dir_all(&temporary_root).expect("test temporary directory is created");
    let first_path = temporary_root.join("first.loco");
    let second_path = temporary_root.join("second.loco");
    std::fs::write(&first_path, b"first").expect("first prediction fixture is written");
    std::fs::write(&second_path, b"second").expect("second prediction fixture is written");
    let resolved_paths = vec![
        g_input::PredictionLocoPath { phenotype_name: Arc::from("trait-b"), loco_file_path: second_path },
        g_input::PredictionLocoPath { phenotype_name: Arc::from("trait-a"), loco_file_path: first_path },
    ];
    let mut fingerprint_cache = ManifestFileFingerprintCache::default();
    let fingerprints = build_prediction_loco_file_fingerprints_with_cache(&resolved_paths, &mut fingerprint_cache)
        .expect("resolved prediction files are fingerprinted");
    assert_eq!(fingerprints.len(), 2);
    let repeated_fingerprints =
        build_prediction_loco_file_fingerprints_with_cache(&resolved_paths, &mut fingerprint_cache)
            .expect("cached prediction fingerprints are reusable");
    assert_eq!(repeated_fingerprints.len(), fingerprints.len());
    std::fs::remove_dir_all(&temporary_root).expect("test temporary directory is removed");
}
