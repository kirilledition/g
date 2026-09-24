use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use g_genotype::{ChunkComputeStatistics, ChunkStats, GenotypeBatch, GenotypeBatchPayload, OwnedGenotypeBuffer};
use g_genotype_contracts::{
    BgenSourceIdentity, ChunkOutputStatistics, NullableFloat32Column, VariantMetadataColumns, VariantMetadataStore,
};
use g_output::{ManifestFileFingerprintCache, NativeVariantMetadataHandle, Regenie2StatisticBatch};

use crate::association_scheduler::{AssociationBatchPipeline, ScheduledAssociationBatch, SchedulerError};
use crate::backend::{
    AssociationBackend, GenotypeDeliveryCapability, GroupPreparationInput, MaterializedAssociationBatch,
    MaterializedGenotypeStatistics, PreparedChromosome,
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
    Transfer,
    Compute,
    Materialize,
    ComputePanic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("test backend failed during {0}")]
struct TestBackendError(&'static str);

struct TestDeviceResult {
    variant_start_index: usize,
    logical_variant_count: usize,
    statistics: ChunkOutputStatistics,
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
    transfer_count: AtomicUsize,
    chromosome_release_count: AtomicUsize,
    materialized_trait_indices: Mutex<Vec<Option<Vec<usize>>>>,
    compute_started_sender: Option<Sender<usize>>,
    compute_gate_receiver: Option<Receiver<()>>,
    block_first_compute: AtomicBool,
}

impl TestBackend {
    fn new(failure_stage: TestFailureStage) -> Self {
        Self {
            failure_stage,
            events: Mutex::new(Vec::new()),
            transfer_count: AtomicUsize::new(0),
            chromosome_release_count: AtomicUsize::new(0),
            materialized_trait_indices: Mutex::new(Vec::new()),
            compute_started_sender: None,
            compute_gate_receiver: None,
            block_first_compute: AtomicBool::new(false),
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

    fn record_event(&self, event: String) {
        self.events.lock().expect("test event lock is available").push(event);
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
        _predictions: g_input::ChromosomePredictionMatrix,
    ) -> Result<PreparedChromosome<Self::ChromosomeState>, Self::Error> {
        self.record_event("prepare_chromosome".to_string());
        Ok(PreparedChromosome { state: 0, null_logistic_converged: None })
    }

    fn release_chromosome(&self, chromosome: Self::ChromosomeState) {
        self.chromosome_release_count.fetch_add(1, Ordering::SeqCst);
        self.record_event(format!("release:{chromosome}"));
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
        let variant_start_index = input.variant_start_index;
        self.record_event(format!("compute:{chromosome}:{variant_start_index}"));
        assert!(self.failure_stage != TestFailureStage::ComputePanic, "intentional compute panic");
        if self.failure_stage == TestFailureStage::Compute {
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

fn drain_pipeline(
    pipeline: &mut AssociationBatchPipeline<'_, TestBackend>,
) -> Vec<crate::association_scheduler::CompletedAssociationBatch> {
    let mut completed_batches = Vec::new();
    while !pipeline.is_drained() {
        completed_batches.push(pipeline.receive().expect("test batch completes"));
    }
    completed_batches
}

fn finish_pipeline(pipeline: &mut AssociationBatchPipeline<'_, TestBackend>) {
    pipeline.release_chromosome().expect("test chromosome is released");
    pipeline.close_submission();
    pipeline.join().expect("test workers join");
}

#[test]
fn scheduler_preserves_batch_order_context_and_chromosome_lifecycle() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::None));
    let group = ();
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), &group).expect("scheduler starts");

    assert!(matches!(pipeline.receive(), Err(SchedulerError::NoPendingBatch)));
    assert!(matches!(
        pipeline.try_submit(build_scheduled_batch(0, 2, 2, ActiveTraitSelection::All)),
        Err(SchedulerError::ChromosomeNotPrepared)
    ));
    pipeline.prepare_chromosome(17).expect("chromosome is prepared");
    assert!(matches!(pipeline.prepare_chromosome(18), Err(SchedulerError::ChromosomeAlreadyPrepared)));

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
    let group = ();
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), &group).expect("scheduler starts");
    pipeline.prepare_chromosome(22).expect("chromosome is prepared");
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
fn scheduler_reuses_steady_state_workers_across_drained_chromosome_lifecycles() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::None));
    let group = ();
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), &group).expect("scheduler starts");

    for (chromosome_state, variant_start_index) in [(21, 0), (22, 2)] {
        pipeline.prepare_chromosome(chromosome_state).expect("chromosome is prepared");
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
fn scheduler_accepts_tail_padded_packed8_batches_without_optional_compute_columns() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::None));
    let group = ();
    let mut pipeline = AssociationBatchPipeline::new(backend, &group).expect("scheduler starts");
    pipeline.prepare_chromosome(22).expect("chromosome is prepared");
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
fn scheduler_propagates_each_backend_failure_stage() {
    let failure_cases = [
        (TestFailureStage::Transfer, "device transfer", false),
        (TestFailureStage::Compute, "compute", true),
        (TestFailureStage::Materialize, "materialization", true),
    ];
    for (failure_stage, expected_stage, asynchronously_submitted) in failure_cases {
        let backend = Arc::new(TestBackend::new(failure_stage));
        let group = ();
        let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), &group).expect("scheduler starts");
        pipeline.prepare_chromosome(1).expect("chromosome is prepared");
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
fn scheduler_reports_worker_panics_and_releases_owned_chromosome_state() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::ComputePanic));
    let group = ();
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), &group).expect("scheduler starts");
    pipeline.prepare_chromosome(9).expect("chromosome is prepared");
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
fn dropping_scheduler_cancels_workers_and_releases_chromosome_state() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::None));
    let group = ();
    let mut pipeline = AssociationBatchPipeline::new(Arc::clone(&backend), &group).expect("scheduler starts");
    pipeline.prepare_chromosome(5).expect("chromosome is prepared");
    drop(pipeline);
    assert_eq!(backend.chromosome_release_count.load(Ordering::SeqCst), 1);
}

#[test]
fn scheduler_rejects_invalid_lifecycle_transitions() {
    let backend = Arc::new(TestBackend::new(TestFailureStage::None));
    let group = ();
    let mut pipeline = AssociationBatchPipeline::new(backend, &group).expect("scheduler starts");
    assert!(matches!(pipeline.join(), Err(SchedulerError::SubmissionOpen)));
    pipeline.prepare_chromosome(3).expect("chromosome is prepared");
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
    let group = ();
    let mut pipeline = AssociationBatchPipeline::new(backend, &group).expect("scheduler starts");
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
fn preflight_accepts_shifted_and_scaled_covariates_for_both_trait_modes() {
    for covariate_values in [
        [1_000_000.0_f32, 1_000_001.0, 1_000_002.0, 1_000_003.0],
        [1.0e-12, 2.0e-12, 3.0e-12, 4.0e-12],
        [1.0e12, 2.0e12, 3.0e12, 4.0e12],
    ] {
        for intercept_value in [1.0, 2.0] {
            let covariates = covariate_values.iter().flat_map(|value| [intercept_value, *value]).collect::<Vec<_>>();
            for is_binary_trait in [false, true] {
                validate_multi_trait_preflight_values(1, 4, &[0.0, 1.0, 0.0, 1.0], 4, 2, &covariates, is_binary_trait)
                    .expect("a full-rank design should not depend on covariate units or intercept scale");
            }
        }
    }
}

#[test]
fn preflight_rejects_dependent_covariates_after_conditioning() {
    for dependent_column in [[0.0_f32; 5], [1.0; 5], [0.0, 2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 7.0, 9.0]] {
        let covariates = [0.0, 1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .zip(dependent_column)
            .flat_map(|(independent_value, dependent_value)| [1.0, independent_value, dependent_value])
            .collect::<Vec<_>>();
        for is_binary_trait in [false, true] {
            assert_eq!(
                validate_multi_trait_preflight_values(
                    1,
                    5,
                    &[0.0, 1.0, 0.0, 1.0, 0.0],
                    5,
                    3,
                    &covariates,
                    is_binary_trait,
                )
                .expect_err("zero, constant, proportional, and affine-dependent columns remain rank deficient"),
                PreflightError::CovariateMatrixRankDeficient
            );
        }
    }
}

#[test]
fn preflight_preserves_rank_without_an_explicit_intercept() {
    let covariates = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0];
    for is_binary_trait in [false, true] {
        validate_multi_trait_preflight_values(1, 4, &[0.0, 1.0, 0.0, 1.0], 4, 2, &covariates, is_binary_trait)
            .expect("x and x + 1 span a full-rank design when no constant column is present");
    }
}

#[test]
fn preflight_accepts_large_cohorts_with_ordinary_covariates() {
    let sample_count = 100_000_usize;
    let phenotypes = (0..sample_count)
        .map(|sample_index| if sample_index.is_multiple_of(2) { 0.0 } else { 1.0 })
        .collect::<Vec<_>>();
    let covariates = (0..sample_count)
        .flat_map(|sample_index| [1.0, if sample_index.is_multiple_of(2) { 40.0 } else { 60.0 }])
        .collect::<Vec<_>>();
    for is_binary_trait in [false, true] {
        validate_multi_trait_preflight_values(
            1,
            sample_count,
            &phenotypes,
            sample_count,
            2,
            &covariates,
            is_binary_trait,
        )
        .expect("cohort size should not make an ordinary full-rank age covariate appear deficient");
    }
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
        output: g_plan::OutputPlan { output_run_root: "output".to_string(), resume: false, writer_thread_count: 8 },
        telemetry: g_plan::TelemetryMode::Off,
        phenotype_runs: vec![g_plan::PhenotypeRunPlan {
            phenotype_name: "trait".to_string(),
            output_directory_name: "0001-trait".to_string(),
        }],
    }
}

struct RunPreparationFixture {
    directory: std::path::PathBuf,
}

impl RunPreparationFixture {
    fn new() -> Self {
        static NEXT_FIXTURE_IDENTIFIER: AtomicUsize = AtomicUsize::new(0);
        let fixture_identifier = NEXT_FIXTURE_IDENTIFIER.fetch_add(1, Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("g-engine-run-preparation-{}-{fixture_identifier}", std::process::id()));
        std::fs::create_dir(&directory).expect("run preparation fixture directory is created");
        let fixture = Self { directory };
        fixture.write_bgen();
        fixture.write(
            "input.sample",
            "ID_1 ID_2\n0 0\nfamily-1 individual-1\nfamily-2 individual-2\nfamily-3 individual-3\nfamily-4 individual-4\n",
        );
        fixture.write(
            "phenotypes.tsv",
            "FID\tIID\ttrait-a\ttrait-b\nfamily-1\tindividual-1\t1\tNA\nfamily-2\tindividual-2\t2\t1\nfamily-3\tindividual-3\t1\t2\nfamily-4\tindividual-4\tNA\t1\n",
        );
        fixture.write(
            "covariates.tsv",
            "FID\tIID\tage\tinvalid\nfamily-1\tindividual-1\t10\tinf\nfamily-2\tindividual-2\t20\tinf\nfamily-3\tindividual-3\t30\tinf\nfamily-4\tindividual-4\t40\tinf\n",
        );
        fixture.write("predictions.list", "trait-a predictions.loco\ntrait-b predictions.loco\n");
        fixture.write(
            "predictions.loco",
            "FID_IID family-1_individual-1 family-2_individual-2 family-3_individual-3 family-4_individual-4\n22 0.1 0.2 0.3 0.4\n",
        );
        fixture
    }

    fn write(&self, name: &str, contents: &str) {
        std::fs::write(self.directory.join(name), contents).expect("run preparation input fixture is written");
    }

    fn path_text(&self, name: &str) -> String {
        self.directory.join(name).to_str().expect("test fixture paths are UTF-8").to_string()
    }

    fn write_bgen(&self) {
        // One uncompressed layout-2 variant with four diploid, unphased samples.
        let mut bytes = Vec::new();
        for header_value in [20_u32, 20, 1, 4] {
            bytes.extend_from_slice(&header_value.to_le_bytes());
        }
        bytes.extend_from_slice(b"bgen");
        bytes.extend_from_slice(&(2_u32 << 2).to_le_bytes());
        for identifier in ["variant-1", "rs-1", "22"] {
            bytes.extend_from_slice(&u16::try_from(identifier.len()).expect("short BGEN identifier").to_le_bytes());
            bytes.extend_from_slice(identifier.as_bytes());
        }
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        for allele in *b"AG" {
            bytes.extend_from_slice(&1_u32.to_le_bytes());
            bytes.push(allele);
        }
        let mut probabilities = Vec::new();
        probabilities.extend_from_slice(&4_u32.to_le_bytes());
        probabilities.extend_from_slice(&2_u16.to_le_bytes());
        probabilities.extend_from_slice(&[2, 2, 2, 2, 2, 2, 0, 8]);
        probabilities.extend_from_slice(&[0, 0, 255, 0, 0, 255, 0, 0]);
        bytes.extend_from_slice(&u32::try_from(probabilities.len()).expect("small probability block").to_le_bytes());
        bytes.extend_from_slice(&probabilities);
        std::fs::write(self.directory.join("input.bgen"), bytes).expect("minimal valid BGEN fixture is written");
    }

    fn run_plan(&self) -> g_plan::RunPlan {
        let mut run_plan = valid_run_plan();
        run_plan.chunk_size = 2;
        run_plan.input = g_plan::InputPlan {
            bgen_path: self.path_text("input.bgen"),
            sample_path: self.path_text("input.sample"),
            phenotype_path: self.path_text("phenotypes.tsv"),
            prediction_list_path: self.path_text("predictions.list"),
            covariate_path: None,
            covariate_names: Vec::new(),
        };
        run_plan.compute.device = g_plan::Device::Cpu;
        run_plan.compute.cpu_thread_count = Some(1);
        run_plan.compute.multi_phenotype_sample_mode = g_plan::MultiPhenotypeSampleMode::PerPhenotype;
        run_plan.output.output_run_root = self.path_text("output");
        run_plan.output.writer_thread_count = 1;
        run_plan.phenotype_runs = ["trait-a", "trait-b"]
            .into_iter()
            .map(|name| g_plan::PhenotypeRunPlan {
                phenotype_name: name.to_string(),
                output_directory_name: format!("{name}.run"),
            })
            .collect();
        run_plan
    }

    fn manifest(&self, phenotype_name: &str) -> serde_json::Value {
        let manifest_path = self.directory.join("output").join(format!("{phenotype_name}.run/run_manifest.json"));
        let manifest_text = std::fs::read_to_string(manifest_path).expect("prepared run writes its manifest");
        serde_json::from_str(&manifest_text).expect("prepared run manifest is valid JSON")
    }
}

impl Drop for RunPreparationFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn compressed_layout_reuse_preserves_exact_resume_geometry_and_source_checks() {
    let fixture = RunPreparationFixture::new();
    let bgen_path = fixture.directory.join("input.bgen");
    let original_bytes = std::fs::read(&bgen_path).expect("fixture BGEN is readable");
    // The fixture's final block contains a four-byte length and 22 probability bytes.
    let metadata_end = original_bytes.len() - 26;
    let mut variant = original_bytes[24..metadata_end].to_vec();
    let compressed_probabilities =
        [120_u8, 156, 99, 97, 96, 96, 96, 98, 96, 2, 3, 6, 14, 6, 134, 255, 96, 4, 0, 10, 115, 2, 25];
    variant.extend_from_slice(&27_u32.to_le_bytes());
    variant.extend_from_slice(&22_u32.to_le_bytes());
    variant.extend_from_slice(&compressed_probabilities);
    let mut bytes = original_bytes[..24].to_vec();
    bytes[8..12].copy_from_slice(&2_u32.to_le_bytes());
    bytes[20..24].copy_from_slice(&9_u32.to_le_bytes());
    bytes.extend_from_slice(&variant);
    bytes.extend_from_slice(&variant);
    std::fs::write(&bgen_path, &bytes).expect("two compressed fixture variants are written");
    let reader = g_genotype::BgenReaderCore::open(&bgen_path).expect("compressed fixture opens");
    assert_eq!(
        reader.packed8_compatibility_with_cache().expect("compressed fixture validates"),
        g_genotype::Packed8Compatibility::Compatible
    );
    let genotype_input = crate::delivery::PreparedGenotypeInput::new(reader, 1);
    let full_plan = [
        g_genotype::ChunkSpec { variant_start_index: 0, variant_stop_index: 1 },
        g_genotype::ChunkSpec { variant_start_index: 1, variant_stop_index: 2 },
    ];
    let first = genotype_input.compressed_layout_for_chunks(&full_plan).expect("initial plan succeeds").unwrap();
    let repeated = genotype_input.compressed_layout_for_chunks(&full_plan).expect("identical plan succeeds").unwrap();
    assert!(Arc::ptr_eq(&first, &repeated));
    let resumed = genotype_input.compressed_layout_for_chunks(&full_plan[1..]).expect("resume plan succeeds").unwrap();
    assert!(!Arc::ptr_eq(&first, &resumed));
    let reversed_plan = [
        g_genotype::ChunkSpec { variant_start_index: 1, variant_stop_index: 2 },
        g_genotype::ChunkSpec { variant_start_index: 0, variant_stop_index: 1 },
    ];
    let reversed =
        genotype_input.compressed_layout_for_chunks(&reversed_plan).expect("reordered plan succeeds").unwrap();
    assert!(!Arc::ptr_eq(&resumed, &reversed));
    let invalid_plan = [g_genotype::ChunkSpec { variant_start_index: 0, variant_stop_index: 3 }];
    assert!(genotype_input.compressed_layout_for_chunks(&invalid_plan).is_err());
    let after_failure = genotype_input
        .compressed_layout_for_chunks(&reversed_plan)
        .expect("failed plans do not replace cache")
        .unwrap();
    assert!(Arc::ptr_eq(&reversed, &after_failure));
    let session = genotype_input.reader.read_session(&[0, 1, 2, 3]).expect("sample selection is valid");
    let batch = session.pack_compressed_packed8_batch(&first, 0, 1).expect("evicted layout remains usable while owned");
    assert_eq!(batch.member_metadata().len(), 3);
    session.finish().expect("unchanged source validates");
    bytes.push(0);
    std::fs::write(&bgen_path, bytes).expect("source is changed between groups");
    assert!(genotype_input.reader.read_session(&[0, 1, 2, 3]).is_err());
}

#[test]
fn run_preparation_accepts_intercept_only_binary_and_quantitative_inputs() {
    for association_mode in [g_plan::AssociationMode::Regenie2Binary, g_plan::AssociationMode::Regenie2Linear] {
        for sample_mode in
            [g_plan::MultiPhenotypeSampleMode::PerPhenotype, g_plan::MultiPhenotypeSampleMode::CompleteCase]
        {
            let fixture = RunPreparationFixture::new();
            let mut run_plan = fixture.run_plan();
            run_plan.association_mode = association_mode;
            run_plan.compute.multi_phenotype_sample_mode = sample_mode;
            let prepared_run = crate::run::RunEngine::open(run_plan, String::new())
                .expect("valid run plan opens")
                .prepare()
                .expect("omitted covariates should prepare an intercept-only design");
            assert_eq!(prepared_run.resolved_gpu_genotype_format(), g_plan::GpuGenotypeFormat::Dosage);
            let expected_sample_count =
                if sample_mode == g_plan::MultiPhenotypeSampleMode::CompleteCase { 2 } else { 3 };
            for phenotype_name in ["trait-a", "trait-b"] {
                let manifest = fixture.manifest(phenotype_name);
                let execution_plan = &manifest["execution_plan"];
                assert_eq!(execution_plan["association_mode"], association_mode.as_str());
                assert_eq!(execution_plan["sample_count"], expected_sample_count);
                assert_eq!(execution_plan["multi_phenotype_sample_mode"], sample_mode.as_str());
                assert_eq!(execution_plan["covariate_names"], serde_json::json!(["intercept"]));
                assert!(execution_plan["covariate_file"].is_null());
            }
            drop(prepared_run);
        }
    }
}

#[test]
fn run_preparation_rejects_covariate_names_without_a_file() {
    let fixture = RunPreparationFixture::new();
    let mut run_plan = fixture.run_plan();
    run_plan.input.covariate_names = vec!["age".to_string()];
    let result = crate::run::RunEngine::open(run_plan, String::new()).expect("valid output plan opens").prepare();
    assert!(matches!(
        result,
        Err(RunPreparationError::Input(g_input::InputError::SampleAlignment(message)))
            if message == "Covariate names cannot be provided without a covariate table."
    ));
}

#[test]
fn run_preparation_preserves_explicit_empty_and_named_covariate_selections() {
    for covariate_names in [Vec::new(), vec!["age".to_string()]] {
        let fixture = RunPreparationFixture::new();
        let mut run_plan = fixture.run_plan();
        run_plan.input.covariate_path = Some(fixture.path_text("covariates.tsv"));
        run_plan.input.covariate_names.clone_from(&covariate_names);
        let prepared_run = crate::run::RunEngine::open(run_plan, String::new())
            .expect("valid output plan opens")
            .prepare()
            .expect("unselected invalid covariates must not be inferred or parsed");
        let mut expected_covariate_names = vec!["intercept".to_string()];
        expected_covariate_names.extend(covariate_names);
        for phenotype_name in ["trait-a", "trait-b"] {
            let manifest = fixture.manifest(phenotype_name);
            assert_eq!(manifest["execution_plan"]["covariate_names"], serde_json::json!(expected_covariate_names));
            assert_eq!(manifest["execution_plan"]["sample_count"], 3);
            assert!(manifest["execution_plan"]["covariate_file"].is_object());
        }
        drop(prepared_run);
    }
}

#[test]
fn run_preparation_validates_explicitly_selected_covariate_values() {
    let fixture = RunPreparationFixture::new();
    let mut run_plan = fixture.run_plan();
    run_plan.input.covariate_path = Some(fixture.path_text("covariates.tsv"));
    run_plan.input.covariate_names = vec!["invalid".to_string()];
    let result = crate::run::RunEngine::open(run_plan, String::new()).expect("valid output plan opens").prepare();
    assert!(matches!(
        result,
        Err(RunPreparationError::Input(g_input::InputError::NonFiniteCovariateValue { covariate_name, value }))
            if covariate_name == "invalid" && value == "inf"
    ));
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

mod compressed_layout_benchmark {
    use std::collections::BTreeSet;
    use std::hint::black_box;
    use std::path::PathBuf;
    use std::time::Instant;

    use crate::delivery::PreparedGenotypeInput;

    const CHUNK_SIZE: usize = 16_384;
    const REPETITIONS: usize = 31;
    const WARMUP_PAIRS: usize = 3;
    const HIT_BUNDLE_SIZE: u32 = 10_000;

    #[derive(Clone, Copy)]
    enum PlanningMethod {
        Uncached,
        Reused,
    }

    #[derive(Clone, Copy)]
    enum TimingScope {
        Guarded,
        PlanningOnly,
    }

    struct LayoutBenchmark {
        input: PreparedGenotypeInput,
        chunks: Vec<g_genotype::ChunkSpec>,
        reversed_chunks: Vec<g_genotype::ChunkSpec>,
        sample_groups: Vec<Vec<usize>>,
    }

    impl LayoutBenchmark {
        fn open() -> Self {
            let path = PathBuf::from(
                std::env::var_os("GWAS_ENGINE_BGEN_BENCHMARK_PATH")
                    .expect("set GWAS_ENGINE_BGEN_BENCHMARK_PATH to the full chr22 BGEN"),
            );
            let reader = g_genotype::BgenReaderCore::open(&path).expect("benchmark BGEN opens");
            assert_eq!(reader.variant_count(), 418_943, "benchmark requires the full chr22 geometry");
            assert_eq!(reader.sample_count(), 2_504, "benchmark requires the full chr22 sample set");
            assert_eq!(
                reader.packed8_compatibility_with_cache().expect("packed8 compatibility resolves outside timing"),
                g_genotype::Packed8Compatibility::Compatible,
            );
            let chunks = reader
                .plan_chromosome_homogeneous_chunks(CHUNK_SIZE, &BTreeSet::new())
                .expect("full chromosome-aware chunk plan builds outside timing");
            assert!(chunks.len() > 1, "reversing the plan must change its ordered cache key");
            let reversed_chunks = chunks
                .iter()
                .rev()
                .map(|chunk| g_genotype::ChunkSpec {
                    variant_start_index: chunk.variant_start_index,
                    variant_stop_index: chunk.variant_stop_index,
                })
                .collect();
            let source_sample_count = reader.sample_count();
            let missing_sample_count = source_sample_count / 20;
            let sample_groups = (0..32)
                .map(|group_index| {
                    let missing_start_index = group_index * 73;
                    (0..source_sample_count)
                        .filter(|sample_index| {
                            (sample_index + source_sample_count - missing_start_index) % source_sample_count
                                >= missing_sample_count
                        })
                        .collect()
                })
                .collect();
            Self { input: PreparedGenotypeInput::new(reader, CHUNK_SIZE), chunks, reversed_chunks, sample_groups }
        }

        fn prime_miss(&self) {
            let session = self.input.reader.read_session(&self.sample_groups[0]).expect("priming source guard opens");
            black_box(
                self.input
                    .compressed_layout_for_chunks(black_box(&self.reversed_chunks))
                    .expect("different ordered geometry evicts the full-plan cache")
                    .expect("benchmark source must use zlib"),
            );
            // Both timed arms end their identical preamble with this complete
            // scan, so forcing a candidate miss does not uniquely warm its data.
            black_box(
                self.input
                    .reader
                    .plan_compressed_packed8_batch_layout(black_box(&self.chunks))
                    .expect("uncached priming scan succeeds")
                    .expect("benchmark source must use zlib"),
            );
            session.finish().expect("priming source guard closes");
        }

        fn plan(&self, method: PlanningMethod) {
            match method {
                PlanningMethod::Uncached => {
                    black_box(
                        self.input
                            .reader
                            .plan_compressed_packed8_batch_layout(black_box(&self.chunks))
                            .expect("uncached planning succeeds")
                            .expect("benchmark source must use zlib"),
                    );
                }
                PlanningMethod::Reused => {
                    black_box(
                        self.input
                            .compressed_layout_for_chunks(black_box(&self.chunks))
                            .expect("cached planning succeeds")
                            .expect("benchmark source must use zlib"),
                    );
                }
            }
        }

        fn measure(&self, group_count: usize, method: PlanningMethod, scope: TimingScope) -> f64 {
            let outer_session = match scope {
                TimingScope::Guarded => None,
                TimingScope::PlanningOnly => {
                    Some(self.input.reader.read_session(&self.sample_groups[0]).expect("outer source guard opens"))
                }
            };
            let started_at = Instant::now();
            for sample_indices in &self.sample_groups[..group_count] {
                let session = match scope {
                    TimingScope::Guarded => Some(
                        self.input.reader.read_session(black_box(sample_indices)).expect("group source guard opens"),
                    ),
                    TimingScope::PlanningOnly => None,
                };
                self.plan(method);
                if let Some(session) = session {
                    session.finish().expect("group source guard closes");
                }
            }
            let elapsed_seconds = started_at.elapsed().as_secs_f64();
            if let Some(session) = outer_session {
                session.finish().expect("outer source guard closes");
            }
            elapsed_seconds
        }

        fn measure_pairs(&self, group_count: usize, scope: TimingScope) -> serde_json::Value {
            let mut pairs = Vec::with_capacity(REPETITIONS);
            for pair_index in 0..WARMUP_PAIRS + REPETITIONS {
                let uncached_first = pair_index % 2 == 0;
                let methods = if uncached_first {
                    [PlanningMethod::Uncached, PlanningMethod::Reused]
                } else {
                    [PlanningMethod::Reused, PlanningMethod::Uncached]
                };
                let mut uncached_seconds = 0.0;
                let mut reused_seconds = 0.0;
                for method in methods {
                    self.prime_miss();
                    let elapsed_seconds = self.measure(group_count, method, scope);
                    match method {
                        PlanningMethod::Uncached => uncached_seconds = elapsed_seconds,
                        PlanningMethod::Reused => reused_seconds = elapsed_seconds,
                    }
                }
                if pair_index >= WARMUP_PAIRS {
                    pairs.push(serde_json::json!({
                        "uncached_first": uncached_first,
                        "uncached_seconds": uncached_seconds,
                        "reused_seconds": reused_seconds,
                        "saved_seconds": uncached_seconds - reused_seconds,
                    }));
                }
            }
            serde_json::json!({
                "group_count": group_count,
                "candidate_misses_per_trial": 1,
                "candidate_hits_per_trial": group_count - 1,
                "pairs": pairs,
            })
        }

        fn measure_hit_bundles(&self) -> Vec<f64> {
            self.plan(PlanningMethod::Reused);
            (0..REPETITIONS)
                .map(|_| {
                    let session = self.input.reader.read_session(&self.sample_groups[0]).expect("hit guard opens");
                    let started_at = Instant::now();
                    for _ in 0..HIT_BUNDLE_SIZE {
                        self.plan(PlanningMethod::Reused);
                    }
                    let seconds_per_hit = started_at.elapsed().as_secs_f64() / f64::from(HIT_BUNDLE_SIZE);
                    session.finish().expect("hit guard closes");
                    seconds_per_hit
                })
                .collect()
        }
    }

    #[test]
    #[ignore = "real chr22 stage benchmark; run only in a CPU allocation with an isolated release target"]
    fn real_chr22_compressed_layout_reuse() {
        let benchmark = LayoutBenchmark::open();
        let guarded: Vec<_> = [1, 8, 32]
            .into_iter()
            .map(|group_count| benchmark.measure_pairs(group_count, TimingScope::Guarded))
            .collect();
        let planning_only: Vec<_> = [1, 8, 32]
            .into_iter()
            .map(|group_count| benchmark.measure_pairs(group_count, TimingScope::PlanningOnly))
            .collect();
        let source = benchmark.input.reader.source_identity();
        let report = serde_json::json!({
            "scope": "compressed_layout_planning_only",
            "debug_assertions": cfg!(debug_assertions),
            "variant_count": benchmark.input.reader.variant_count(),
            "source_sample_count": benchmark.input.reader.sample_count(),
            "selected_sample_count_per_group": benchmark.sample_groups[0].len(),
            "chunk_size": CHUNK_SIZE,
            "chunks": benchmark.chunks.iter().map(|chunk| serde_json::json!({
                "start": chunk.variant_start_index, "stop": chunk.variant_stop_index,
            })).collect::<Vec<_>>(),
            "source_identity": {
                "configured_path": source.configured_path,
                "canonical_path": source.canonical_path,
                "device": source.device_identifier,
                "inode": source.inode_identifier,
                "change_time_nanoseconds": source.change_time_nanoseconds,
                "modification_time_nanoseconds": source.modification_time_nanoseconds,
                "file_size": source.file_size,
            },
            "warmup_pairs_per_case": WARMUP_PAIRS,
            "repetitions": REPETITIONS,
            "guarded": guarded,
            "planning_only": planning_only,
            "hit_bundle_size": HIT_BUNDLE_SIZE,
            "planning_only_seconds_per_hit": benchmark.measure_hit_bundles(),
        });
        let report_path = std::env::var_os("GWAS_ENGINE_LAYOUT_BENCHMARK_REPORT")
            .expect("set GWAS_ENGINE_LAYOUT_BENCHMARK_REPORT to an ignored result path");
        std::fs::write(report_path, serde_json::to_vec_pretty(&report).expect("benchmark report serializes"))
            .expect("benchmark report is written");
    }
}

#[test]
fn coordinated_resume_initializes_backend_only_for_pending_phenotypes() {
    struct GroupInterruptionHooks {
        backend: Arc<TestBackend>,
        interrupt_second_group: bool,
    }

    impl crate::run::RunHooks for GroupInterruptionHooks {
        type BackendError = TestBackendError;
        type Error = TestBackendError;

        fn check_interruption(&mut self) -> Result<(), Self::Error> {
            let prepared_group_count = self
                .backend
                .events
                .lock()
                .expect("test events remain available")
                .iter()
                .filter(|event| event.as_str() == "prepare_group")
                .count();
            if self.interrupt_second_group && prepared_group_count == 2 {
                Err(TestBackendError("interrupt"))
            } else {
                Ok(())
            }
        }

        fn interruption_signal_name(error: &Self::Error) -> Option<&str> {
            (error.0 == "interrupt").then_some("SIGINT")
        }

        fn backend_interruption_error(_error: &Self::BackendError) -> Option<Self::Error> {
            None
        }
    }

    let fixture = RunPreparationFixture::new();
    let make_run_plan = |resume| {
        let mut run_plan = fixture.run_plan();
        run_plan.association_mode = g_plan::AssociationMode::Regenie2Linear;
        run_plan.output.resume = resume;
        run_plan
    };
    let backend = Arc::new(TestBackend::new(TestFailureStage::None));
    let mut hooks = GroupInterruptionHooks { backend: Arc::clone(&backend), interrupt_second_group: true };
    let interrupted = crate::execute_coordinated_run(
        make_run_plan(false),
        String::new(),
        |_, _, _| Ok(Arc::clone(&backend)),
        &mut hooks,
        &g_runtime::TelemetryRunSession::default(),
        "resume-test",
        None,
    );
    assert!(matches!(interrupted, Err(crate::EngineRunError::Interrupted(TestBackendError("interrupt")))));
    assert_eq!(fixture.manifest("trait-a")["committed_chunks"].as_array().expect("commit list").len(), 1);
    assert!(fixture.manifest("trait-b")["committed_chunks"].as_array().expect("commit list").is_empty());
    let preserved_part = std::fs::read_dir(fixture.directory.join("output/trait-a.run/parts"))
        .expect("committed phenotype has output")
        .next()
        .expect("committed phenotype has a part")
        .expect("part entry is readable")
        .path();
    let preserved_bytes = std::fs::read(&preserved_part).expect("first phenotype output is readable");

    let resumed_backend = Arc::new(TestBackend::new(TestFailureStage::None));
    hooks = GroupInterruptionHooks { backend: Arc::clone(&resumed_backend), interrupt_second_group: false };
    let mut initialization_count = 0;
    let artifacts = crate::execute_coordinated_run(
        make_run_plan(true),
        String::new(),
        |_, _, _| {
            initialization_count += 1;
            Ok(Arc::clone(&resumed_backend))
        },
        &mut hooks,
        &g_runtime::TelemetryRunSession::default(),
        "resume-test",
        None,
    )
    .expect("asymmetric phenotype coverage resumes");
    assert_eq!(artifacts.len(), 2);
    assert_eq!(initialization_count, 1);
    assert_eq!(resumed_backend.transfer_count.load(Ordering::SeqCst), 1);
    assert_eq!(std::fs::read(&preserved_part).expect("preserved part remains readable"), preserved_bytes);
    assert_eq!(fixture.manifest("trait-a")["status"], "completed");
    assert_eq!(fixture.manifest("trait-b")["status"], "completed");

    crate::execute_coordinated_run::<TestBackend, _, _>(
        make_run_plan(true),
        String::new(),
        |_, _, _| panic!("fully committed outputs must not initialize a backend"),
        &mut hooks,
        &g_runtime::TelemetryRunSession::default(),
        "resume-test",
        None,
    )
    .expect("fully committed phenotypes finish without a backend");
    assert_eq!(resumed_backend.transfer_count.load(Ordering::SeqCst), 1);
}
