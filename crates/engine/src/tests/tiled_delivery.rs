//! Shared-source delivery regression tests through the real output lifecycle.

use std::fmt::Write;

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SharedFailure {
    None,
    PrepareSource,
    MissingSource,
    SelectSecondGroup,
    MissingSelection,
    ComputeSecondGroup,
    MaterializeSecondGroup,
}

struct SharedGroup {
    identifier: usize,
    sample_count: usize,
    trait_count: usize,
}

struct SharedChromosome {
    group_identifier: usize,
    prediction_marker: u32,
}

struct SharedSource {
    variant_start_index: usize,
    logical_variant_count: usize,
    compute_variant_count: usize,
}

struct SharedInput {
    group_identifier: usize,
    variant_start_index: usize,
    logical_variant_count: usize,
    sample_count: usize,
    trait_count: usize,
}

#[derive(Debug, Eq, PartialEq)]
struct SharedMaterialization {
    group_identifier: usize,
    variant_start_index: usize,
    trait_count: usize,
    active_trait_indices: Option<Vec<usize>>,
}

struct PreservedPart {
    path: std::path::PathBuf,
    bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SharedSynchronizationMode {
    ReleaseDuringAbort,
    ReleaseWhenSecondComputeStarts,
}

struct SharedSynchronization {
    mode: SharedSynchronizationMode,
    entered_sender: Sender<()>,
    entered_receiver: Receiver<()>,
    release_sender: Sender<()>,
    release_receiver: Receiver<()>,
    materialization_pending: AtomicBool,
}

impl SharedSynchronization {
    fn new(mode: SharedSynchronizationMode) -> Self {
        let (entered_sender, entered_receiver) = crossbeam_channel::bounded(1);
        let (release_sender, release_receiver) = crossbeam_channel::bounded(1);
        Self {
            mode,
            entered_sender,
            entered_receiver,
            release_sender,
            release_receiver,
            materialization_pending: AtomicBool::new(false),
        }
    }
}

struct SharedMaterializationGuard<'backend> {
    backend: &'backend SharedBackend,
}

impl Drop for SharedMaterializationGuard<'_> {
    fn drop(&mut self) {
        self.backend
            .synchronization
            .as_ref()
            .expect("guard owns a synchronized materialization")
            .materialization_pending
            .store(false, Ordering::SeqCst);
        self.backend.record("first_materialization_finished".to_string());
    }
}

struct SharedBackend {
    failure: SharedFailure,
    supported: bool,
    events: Mutex<Vec<String>>,
    prepared_group_count: AtomicUsize,
    materialized_count: AtomicUsize,
    materializations: Mutex<Vec<SharedMaterialization>>,
    source_mutation_path: Option<std::path::PathBuf>,
    synchronization: Option<SharedSynchronization>,
}

impl SharedBackend {
    fn new(failure: SharedFailure) -> Self {
        Self {
            failure,
            supported: true,
            events: Mutex::new(Vec::new()),
            prepared_group_count: AtomicUsize::new(0),
            materialized_count: AtomicUsize::new(0),
            materializations: Mutex::new(Vec::new()),
            source_mutation_path: None,
            synchronization: None,
        }
    }

    fn record(&self, event: String) {
        self.events.lock().expect("shared backend event lock is available").push(event);
    }

    fn events(&self) -> Vec<String> {
        self.events.lock().expect("shared backend event lock is available").clone()
    }

    fn with_synchronization(failure: SharedFailure, mode: SharedSynchronizationMode) -> Self {
        Self { synchronization: Some(SharedSynchronization::new(mode)), ..Self::new(failure) }
    }

    fn hold_first_materialization(
        &self,
        input: &SharedInput,
    ) -> Result<Option<SharedMaterializationGuard<'_>>, TestBackendError> {
        let Some(synchronization) =
            self.synchronization.as_ref().filter(|_| input.group_identifier == 0 && input.variant_start_index == 0)
        else {
            return Ok(None);
        };
        synchronization.materialization_pending.store(true, Ordering::SeqCst);
        let guard = SharedMaterializationGuard { backend: self };
        self.record("first_materialization_waiting".to_string());
        synchronization
            .entered_sender
            .send_timeout((), TEST_SYNCHRONIZATION_TIMEOUT)
            .map_err(|_| TestBackendError("first materialization entry notification timed out"))?;
        synchronization
            .release_receiver
            .recv_timeout(TEST_SYNCHRONIZATION_TIMEOUT)
            .map_err(|_| TestBackendError("first materialization release timed out"))?;
        Ok(Some(guard))
    }
}

impl AssociationBackend for SharedBackend {
    type GroupState = SharedGroup;
    type ChromosomeState = SharedChromosome;
    type TransferredInput = SharedInput;
    type SharedSourceBatch = SharedSource;
    type DeviceResult = SharedInput;
    type Error = TestBackendError;

    fn genotype_delivery_capability(&self) -> GenotypeDeliveryCapability {
        GenotypeDeliveryCapability::RawDeflatePacked8
    }

    fn supports_shared_source_batches(&self) -> bool {
        self.supported
    }

    fn prepare_shared_source(&self, input: GenotypeBatch) -> Result<Option<SharedSource>, Self::Error> {
        assert!(matches!(input.payload, GenotypeBatchPayload::CompressedPacked8(_)));
        assert_eq!(input.sample_count, 4, "shared decoding must preserve every source sample");
        self.record(format!("source:{}", input.variant_start_index));
        match self.failure {
            SharedFailure::PrepareSource => return Err(TestBackendError("prepare_shared_source")),
            SharedFailure::MissingSource => return Ok(None),
            _ => {}
        }
        Ok(Some(SharedSource {
            variant_start_index: input.variant_start_index,
            logical_variant_count: input.logical_variant_count,
            compute_variant_count: input.compute_variant_count,
        }))
    }

    fn select_shared_source(
        &self,
        group: &SharedGroup,
        source: &SharedSource,
    ) -> Result<Option<SharedInput>, Self::Error> {
        assert!(source.logical_variant_count <= source.compute_variant_count);
        self.record(format!("select:{}:{}", group.identifier, source.variant_start_index));
        if group.identifier == 1 {
            if let Some(synchronization) = &self.synchronization {
                synchronization
                    .entered_receiver
                    .recv_timeout(TEST_SYNCHRONIZATION_TIMEOUT)
                    .map_err(|_| TestBackendError("second selection did not observe pending materialization"))?;
                assert!(synchronization.materialization_pending.load(Ordering::SeqCst));
                self.record("second_selection_with_pending_first".to_string());
            }
            match self.failure {
                SharedFailure::SelectSecondGroup => return Err(TestBackendError("select_shared_source")),
                SharedFailure::MissingSelection => return Ok(None),
                _ => {}
            }
        }
        Ok(Some(SharedInput {
            group_identifier: group.identifier,
            variant_start_index: source.variant_start_index,
            logical_variant_count: source.logical_variant_count,
            sample_count: group.sample_count,
            trait_count: group.trait_count,
        }))
    }

    fn release_shared_source(&self, source: SharedSource) {
        assert!(
            !self
                .synchronization
                .as_ref()
                .is_some_and(|synchronization| synchronization.materialization_pending.load(Ordering::SeqCst)),
            "source must outlive pending materialization"
        );
        self.record(format!("release_source:{}", source.variant_start_index));
    }

    fn prepare_group(&self, input: GroupPreparationInput) -> Result<SharedGroup, Self::Error> {
        let identifier = self.prepared_group_count.fetch_add(1, Ordering::SeqCst);
        let crate::backend::GenotypeTransferPreparation::CompressedPacked8(transfer) = input.genotype_transfer else {
            return Err(TestBackendError("expected compressed group transfer"));
        };
        assert_eq!(transfer.file_sample_count, 4);
        assert_eq!(transfer.selected_sample_count, 3, "each fixture group excludes a different source sample");
        assert_eq!(transfer.selected_sample_count, input.phenotypes.sample_count);
        self.record(format!("group:{identifier}"));
        Ok(SharedGroup {
            identifier,
            sample_count: input.phenotypes.sample_count,
            trait_count: input.phenotypes.trait_count,
        })
    }

    fn release_group(&self, group: SharedGroup) {
        self.record(format!("release_group:{}", group.identifier));
    }

    fn prepare_chromosome(
        &self,
        group: &SharedGroup,
        predictions: g_input::ChromosomePredictionMatrix,
    ) -> Result<PreparedChromosome<SharedChromosome>, Self::Error> {
        assert_eq!(predictions.sample_count, group.sample_count);
        assert_eq!(predictions.trait_count, group.trait_count);
        // Fixture rows contain a chromosome-specific constant for every sample.
        let prediction_marker = predictions.prediction_values[0].to_bits();
        self.record(format!("chromosome:{}:{prediction_marker}", group.identifier));
        Ok(PreparedChromosome {
            state: SharedChromosome { group_identifier: group.identifier, prediction_marker },
            null_logistic_converged: None,
        })
    }

    fn release_chromosome(&self, chromosome: SharedChromosome) {
        self.record(format!("release_chromosome:{}:{}", chromosome.group_identifier, chromosome.prediction_marker));
        if chromosome.group_identifier == 0
            && let Some(synchronization) = self
                .synchronization
                .as_ref()
                .filter(|synchronization| synchronization.mode == SharedSynchronizationMode::ReleaseDuringAbort)
        {
            synchronization.release_sender.try_send(()).expect("abort releases the pending first materialization");
        }
    }

    fn transfer_batch(&self, group: &SharedGroup, input: GenotypeBatch) -> Result<SharedInput, Self::Error> {
        assert!(matches!(input.payload, GenotypeBatchPayload::CompressedPacked8(_)));
        assert_eq!(input.sample_count, group.sample_count);
        self.record(format!("transfer:{}:{}", group.identifier, input.variant_start_index));
        Ok(SharedInput {
            group_identifier: group.identifier,
            variant_start_index: input.variant_start_index,
            logical_variant_count: input.logical_variant_count,
            sample_count: group.sample_count,
            trait_count: group.trait_count,
        })
    }

    fn compute_batch(&self, chromosome: &SharedChromosome, input: SharedInput) -> Result<SharedInput, Self::Error> {
        assert_eq!(chromosome.group_identifier, input.group_identifier);
        self.record(format!("compute:{}:{}", input.group_identifier, input.variant_start_index));
        if input.group_identifier == 1
            && let Some(synchronization) = self.synchronization.as_ref().filter(|synchronization| {
                synchronization.mode == SharedSynchronizationMode::ReleaseWhenSecondComputeStarts
            })
        {
            self.record("second_compute_releases_first_materialization".to_string());
            synchronization
                .release_sender
                .send_timeout((), TEST_SYNCHRONIZATION_TIMEOUT)
                .map_err(|_| TestBackendError("second compute could not release first materialization"))?;
        }
        if self.failure == SharedFailure::ComputeSecondGroup && input.group_identifier == 1 {
            return Err(TestBackendError("compute"));
        }
        Ok(input)
    }

    fn materialize_batch(
        &self,
        result: SharedInput,
        active_trait_indices: Option<&[usize]>,
        logical_variant_count: usize,
    ) -> Result<MaterializedAssociationBatch, Self::Error> {
        self.record(format!("materialize:{}:{}", result.group_identifier, result.variant_start_index));
        let _materialization_guard = self.hold_first_materialization(&result)?;
        self.materializations.lock().expect("materialization observations remain available").push(
            SharedMaterialization {
                group_identifier: result.group_identifier,
                variant_start_index: result.variant_start_index,
                trait_count: result.trait_count,
                active_trait_indices: active_trait_indices.map(<[usize]>::to_vec),
            },
        );
        if self.failure == SharedFailure::MaterializeSecondGroup && result.group_identifier == 1 {
            return Err(TestBackendError("materialize"));
        }
        assert_eq!(logical_variant_count, result.logical_variant_count);
        if self.materialized_count.fetch_add(1, Ordering::SeqCst) == 0
            && let Some(path) = &self.source_mutation_path
        {
            // Replace the path while the original mapping remains safely readable.
            let replacement_path = path.with_extension("replacement.bgen");
            let mut replacement = std::fs::read(path).expect("source fixture remains readable");
            replacement.push(0);
            std::fs::write(&replacement_path, replacement).expect("replacement source is written");
            std::fs::rename(replacement_path, path).expect("source identity changes after first consumer");
        }
        let trait_count = active_trait_indices.map_or(result.trait_count, <[usize]>::len);
        let value_count = trait_count * logical_variant_count;
        let mut statistics = build_output_statistics(logical_variant_count);
        statistics.observation_count =
            vec![i32::try_from(result.sample_count).expect("fixture sample count fits i32"); logical_variant_count];
        Ok(MaterializedAssociationBatch {
            association: Regenie2StatisticBatch {
                trait_count,
                variant_count: logical_variant_count,
                beta: vec![1.0; value_count],
                standard_error: vec![1.0; value_count],
                chi_squared: vec![2.0; value_count],
                log10_p_value: vec![3.0; value_count],
                correction_code: None,
            },
            genotype_statistics: MaterializedGenotypeStatistics::Ready(statistics),
        })
    }
}

struct SharedHooks {
    backend: Arc<SharedBackend>,
    interrupt_after_materialized: Option<usize>,
}

impl crate::run::RunHooks for SharedHooks {
    type BackendError = TestBackendError;
    type Error = TestBackendError;

    fn check_interruption(&mut self) -> Result<(), Self::Error> {
        if self
            .interrupt_after_materialized
            .is_some_and(|limit| self.backend.materialized_count.load(Ordering::SeqCst) >= limit)
        {
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

fn compressed_fixture(chromosomes: &[&str]) -> RunPreparationFixture {
    let fixture = RunPreparationFixture::new();
    let mut bytes = Vec::new();
    for header_value in [20_u32, 20, u32::try_from(chromosomes.len()).expect("small fixture"), 4] {
        bytes.extend_from_slice(&header_value.to_le_bytes());
    }
    bytes.extend_from_slice(b"bgen");
    bytes.extend_from_slice(&9_u32.to_le_bytes());
    for (variant_index, chromosome) in chromosomes.iter().enumerate() {
        let variant_identifier = format!("variant-{variant_index}");
        let reference_identifier = format!("rs-{variant_index}");
        for identifier in [variant_identifier.as_str(), reference_identifier.as_str(), *chromosome] {
            bytes.extend_from_slice(&u16::try_from(identifier.len()).expect("short BGEN identifier").to_le_bytes());
            bytes.extend_from_slice(identifier.as_bytes());
        }
        bytes.extend_from_slice(&u32::try_from(variant_index + 1).expect("small variant index").to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        for allele in *b"AG" {
            bytes.extend_from_slice(&1_u32.to_le_bytes());
            bytes.push(allele);
        }
        bytes.extend_from_slice(&27_u32.to_le_bytes());
        bytes.extend_from_slice(&22_u32.to_le_bytes());
        bytes.extend_from_slice(&[
            120_u8, 156, 99, 97, 96, 96, 96, 98, 96, 2, 3, 6, 14, 6, 134, 255, 96, 4, 0, 10, 115, 2, 25,
        ]);
    }
    std::fs::write(fixture.directory.join("input.bgen"), bytes).expect("compressed source fixture is written");
    let mut predictions =
        "FID_IID family-1_individual-1 family-2_individual-2 family-3_individual-3 family-4_individual-4\n".to_string();
    for (chromosome_index, chromosome) in chromosomes.iter().copied().collect::<BTreeSet<_>>().into_iter().enumerate() {
        let marker = chromosome_index + 1;
        writeln!(predictions, "{chromosome} {marker} {marker} {marker} {marker}")
            .expect("fixture prediction row formats into a string");
    }
    fixture.write("predictions.loco", &predictions);
    fixture
}

fn shared_run_plan(fixture: &RunPreparationFixture) -> g_plan::RunPlan {
    let mut plan = fixture.run_plan();
    plan.association_mode = g_plan::AssociationMode::Regenie2Linear;
    plan.compute.device = g_plan::Device::Gpu;
    plan.compute.cpu_thread_count = None;
    plan.chunk_size = 1;
    plan
}

fn execute_shared(
    plan: g_plan::RunPlan,
    backend: &Arc<SharedBackend>,
    interrupt_after_materialized: Option<usize>,
) -> Result<Vec<crate::PhenotypeRunArtifact>, crate::EngineRunError<TestBackendError>> {
    let mut hooks = SharedHooks { backend: Arc::clone(backend), interrupt_after_materialized };
    crate::execute_coordinated_run(
        plan,
        String::new(),
        |_, _, _| Ok(Arc::clone(backend)),
        &mut hooks,
        &g_runtime::TelemetryRunSession::default(),
        "shared-source-test",
        None,
    )
}

fn event_count(events: &[String], prefix: &str) -> usize {
    events.iter().filter(|event| event.starts_with(prefix)).count()
}

fn event_position(events: &[String], value: &str) -> usize {
    events.iter().position(|event| event == value).unwrap_or_else(|| panic!("missing event {value}: {events:?}"))
}

fn assert_released_groups(events: &[String], group_count: usize) {
    assert_eq!(event_count(events, "release_group:"), group_count, "{events:?}");
    for identifier in 0..group_count {
        let release_position = event_position(events, &format!("release_group:{identifier}"));
        assert_eq!(events.iter().filter(|event| **event == format!("release_group:{identifier}")).count(), 1);
        assert!(
            events.iter().enumerate().all(|(position, event)| {
                !(event.starts_with("release_chromosome:") || event.starts_with("release_source:"))
                    || position < release_position
            }),
            "device owners must be released before group state: {events:?}"
        );
    }
}

#[test]
fn overlapping_groups_decode_each_source_once_and_finish_real_outputs() {
    let fixture = compressed_fixture(&["1", "1", "2"]);
    let backend = Arc::new(SharedBackend::new(SharedFailure::None));
    let mut plan = shared_run_plan(&fixture);
    plan.chunk_size = 2;
    let artifacts = execute_shared(plan, &backend, None).expect("two groups complete with a padded final chunk");
    assert_eq!(artifacts.len(), 2);
    let events = backend.events();
    assert_eq!(event_count(&events, "source:"), 2, "{events:?}");
    assert_eq!(event_count(&events, "select:"), 4, "{events:?}");
    assert_eq!(event_count(&events, "transfer:"), 0, "{events:?}");
    assert_eq!(event_count(&events, "release_source:"), 2, "{events:?}");
    for variant_index in [0, 2] {
        let released = event_position(&events, &format!("release_source:{variant_index}"));
        for identifier in 0..2 {
            assert!(event_position(&events, &format!("materialize:{identifier}:{variant_index}")) < released);
        }
    }
    assert_released_groups(&events, 2);
    for phenotype_name in ["trait-a", "trait-b"] {
        let manifest = fixture.manifest(phenotype_name);
        assert_eq!(manifest["status"], "completed");
        assert_eq!(manifest["committed_chunks"].as_array().expect("commit array").len(), 2);
    }
}

#[test]
fn singleton_and_unsupported_backends_use_existing_compressed_transfer() {
    for single_group in [false, true] {
        let fixture = compressed_fixture(&["1", "2"]);
        let mut plan = shared_run_plan(&fixture);
        if single_group {
            plan.phenotype_runs.truncate(1);
        }
        let backend = Arc::new(SharedBackend { supported: single_group, ..SharedBackend::new(SharedFailure::None) });
        execute_shared(plan, &backend, None).expect("ordinary compressed delivery remains supported");
        let events = backend.events();
        let group_count = if single_group { 1 } else { 2 };
        assert_eq!(event_count(&events, "source:"), 0, "{events:?}");
        assert_eq!(event_count(&events, "select:"), 0, "{events:?}");
        assert_eq!(event_count(&events, "transfer:"), group_count * 2, "{events:?}");
        assert_eq!(event_count(&events, "release_group:"), group_count, "{events:?}");
    }
}

#[test]
fn second_consumer_failures_release_chromosomes_before_source_and_groups() {
    for failure in [
        SharedFailure::SelectSecondGroup,
        SharedFailure::MissingSelection,
        SharedFailure::ComputeSecondGroup,
        SharedFailure::MaterializeSecondGroup,
    ] {
        let fixture = compressed_fixture(&["1"]);
        let backend = Arc::new(SharedBackend::new(failure));
        let error = execute_shared(shared_run_plan(&fixture), &backend, None)
            .expect_err("the failed second consumer must reject the run");
        if failure == SharedFailure::MissingSelection {
            assert!(
                matches!(error, crate::EngineRunError::Failure { message } if message.contains("Backend advertised shared-source support but did not select a batch"))
            );
        }
        let events = backend.events();
        assert_eq!(event_count(&events, "source:"), 1, "{failure:?}: {events:?}");
        assert_eq!(event_count(&events, "release_source:"), 1, "{failure:?}: {events:?}");
        assert_eq!(event_count(&events, "transfer:"), 0, "advertised support must not fall back: {events:?}");
        if matches!(failure, SharedFailure::ComputeSecondGroup | SharedFailure::MaterializeSecondGroup) {
            assert!(event_position(&events, "materialize:0:0") < event_position(&events, "release_source:0"));
        }
        assert!(
            events.iter().enumerate().all(|(position, event)| !event.starts_with("release_chromosome:")
                || position < event_position(&events, "release_source:0")),
            "{failure:?}: {events:?}"
        );
        assert_eq!(event_count(&events, "chromosome:"), event_count(&events, "release_chromosome:"));
        assert_released_groups(&events, 2);
    }
}

#[test]
fn second_selection_failure_joins_pending_materialization_before_source_release() {
    let fixture = compressed_fixture(&["1"]);
    let backend = Arc::new(SharedBackend::with_synchronization(
        SharedFailure::SelectSecondGroup,
        SharedSynchronizationMode::ReleaseDuringAbort,
    ));
    let error = execute_shared(shared_run_plan(&fixture), &backend, None)
        .expect_err("second selection fails while the first materialization remains pending");
    assert!(matches!(error, crate::EngineRunError::Failure { message } if message.contains("select_shared_source")));
    let events = backend.events();
    assert_eq!(event_count(&events, "select:"), 2, "{events:?}");
    assert_eq!(event_count(&events, "compute:"), 1, "{events:?}");
    assert_eq!(event_count(&events, "materialize:"), 1, "{events:?}");
    assert_eq!(event_count(&events, "release_source:"), 1, "{events:?}");
    let waiting = event_position(&events, "first_materialization_waiting");
    let rejected_selection = event_position(&events, "second_selection_with_pending_first");
    let abort_release = event_position(&events, &format!("release_chromosome:0:{}", 1_f32.to_bits()));
    let finished = event_position(&events, "first_materialization_finished");
    let source_release = event_position(&events, "release_source:0");
    assert!(waiting < rejected_selection && rejected_selection < abort_release, "{events:?}");
    assert!(abort_release < finished && finished < source_release, "{events:?}");
    assert_eq!(event_count(&events, "release_chromosome:"), 2, "{events:?}");
    assert_released_groups(&events, 2);
    for phenotype_name in ["trait-a", "trait-b"] {
        assert!(fixture.manifest(phenotype_name)["committed_chunks"].as_array().expect("commit array").is_empty());
    }
}

#[test]
fn both_groups_submit_before_waiting_for_first_materialization() {
    let fixture = compressed_fixture(&["1"]);
    let backend = Arc::new(SharedBackend::with_synchronization(
        SharedFailure::None,
        SharedSynchronizationMode::ReleaseWhenSecondComputeStarts,
    ));
    execute_shared(shared_run_plan(&fixture), &backend, None)
        .expect("second compute starts while the first materialization is held");
    let events = backend.events();
    assert_eq!(event_count(&events, "source:"), 1, "{events:?}");
    assert_eq!(event_count(&events, "select:"), 2, "{events:?}");
    assert_eq!(event_count(&events, "compute:"), 2, "{events:?}");
    assert_eq!(event_count(&events, "transfer:"), 0, "{events:?}");
    let waiting = event_position(&events, "first_materialization_waiting");
    let second_selection = event_position(&events, "second_selection_with_pending_first");
    let second_compute = event_position(&events, "second_compute_releases_first_materialization");
    let finished = event_position(&events, "first_materialization_finished");
    assert!(waiting < second_selection && second_selection < second_compute && second_compute < finished, "{events:?}");
    assert!(finished < event_position(&events, "release_source:0"), "{events:?}");
    assert_released_groups(&events, 2);
    for phenotype_name in ["trait-a", "trait-b"] {
        let manifest = fixture.manifest(phenotype_name);
        assert_eq!(manifest["status"], "completed");
        assert_eq!(manifest["committed_chunks"].as_array().expect("commit array").len(), 1);
    }
}

#[test]
fn failed_or_missing_shared_source_does_not_fall_back_after_admission() {
    for failure in [SharedFailure::PrepareSource, SharedFailure::MissingSource] {
        let fixture = compressed_fixture(&["1"]);
        let backend = Arc::new(SharedBackend::new(failure));
        let error = execute_shared(shared_run_plan(&fixture), &backend, None)
            .expect_err("the failed source preparation must reject the run");
        if failure == SharedFailure::MissingSource {
            assert!(
                matches!(error, crate::EngineRunError::Failure { message } if message.contains("Backend advertised shared-source support but did not prepare a source"))
            );
        }
        let events = backend.events();
        assert_eq!(event_count(&events, "source:"), 1, "{events:?}");
        assert_eq!(event_count(&events, "release_source:"), 0, "no source owner was returned: {events:?}");
        assert_eq!(event_count(&events, "select:"), 0, "{events:?}");
        assert_eq!(event_count(&events, "transfer:"), 0, "{events:?}");
        assert_released_groups(&events, 2);
    }
}

#[test]
fn interruption_between_consumers_flushes_first_output_and_releases_device_owners() {
    let fixture = compressed_fixture(&["1"]);
    let backend = Arc::new(SharedBackend::with_synchronization(
        SharedFailure::None,
        SharedSynchronizationMode::ReleaseWhenSecondComputeStarts,
    ));
    let result = execute_shared(shared_run_plan(&fixture), &backend, Some(1));
    assert!(matches!(result, Err(crate::EngineRunError::Interrupted(TestBackendError("interrupt")))));
    let events = backend.events();
    assert_eq!(event_count(&events, "select:"), 2, "{events:?}");
    assert_eq!(event_count(&events, "release_source:"), 1, "{events:?}");
    assert!(
        events.iter().enumerate().all(|(position, event)| !event.starts_with("release_chromosome:")
            || position < event_position(&events, "release_source:0")),
        "{events:?}"
    );
    assert_released_groups(&events, 2);
    assert_eq!(fixture.manifest("trait-a")["committed_chunks"].as_array().expect("commit array").len(), 1);
    assert!(fixture.manifest("trait-b")["committed_chunks"].as_array().expect("commit array").is_empty());

    let resumed_backend = Arc::new(SharedBackend::new(SharedFailure::None));
    let mut plan = shared_run_plan(&fixture);
    plan.output.resume = true;
    execute_shared(plan, &resumed_backend, None).expect("only the missing group resumes");
    let resumed_events = resumed_backend.events();
    assert_eq!(event_count(&resumed_events, "source:"), 0, "{resumed_events:?}");
    assert_eq!(event_count(&resumed_events, "transfer:"), 1, "{resumed_events:?}");
    for phenotype_name in ["trait-a", "trait-b"] {
        assert_eq!(fixture.manifest(phenotype_name)["status"], "completed");
    }
}

#[test]
fn shared_delivery_detects_source_replacement_before_success() {
    let fixture = compressed_fixture(&["1", "1"]);
    let backend = Arc::new(SharedBackend {
        source_mutation_path: Some(fixture.directory.join("input.bgen")),
        ..SharedBackend::new(SharedFailure::None)
    });
    let result = execute_shared(shared_run_plan(&fixture), &backend, None);
    assert!(matches!(result, Err(crate::EngineRunError::Failure { .. })), "source replacement must reject the run");
    let events = backend.events();
    assert!(event_count(&events, "source:") > 0, "{events:?}");
    assert_eq!(event_count(&events, "source:"), event_count(&events, "release_source:"), "{events:?}");
    assert_released_groups(&events, 2);
    for phenotype_name in ["trait-a", "trait-b"] {
        assert_ne!(fixture.manifest(phenotype_name)["status"], "completed");
    }
}

fn seed_committed_chunks(plan: g_plan::RunPlan, committed_chunks: &[&[usize]]) {
    let plan = Arc::new(plan);
    let reader = g_genotype::BgenReaderCore::open(std::path::Path::new(&plan.input.bgen_path))
        .expect("compressed source opens for resume seeding");
    let phenotype_names =
        plan.phenotype_runs.iter().map(|phenotype| phenotype.phenotype_name.clone()).collect::<Vec<_>>();
    let sample_identifiers = g_input::load_sample_identifier_data_from_sample_file(
        std::path::Path::new(&plan.input.sample_path),
        reader.sample_count(),
    )
    .expect("fixture sample identifiers load");
    let prediction_paths = g_input::resolve_prediction_loco_paths(
        std::path::Path::new(&plan.input.prediction_list_path),
        &phenotype_names,
    )
    .expect("fixture prediction catalog resolves");
    let groups = g_input::load_aligned_phenotype_groups(&g_input::PhenotypeGroupLoadRequest {
        sample_identifiers: &sample_identifiers,
        phenotype_path: &plan.input.phenotype_path,
        prediction_loco_paths: &prediction_paths,
        phenotype_names: &phenotype_names,
        covariate_path: None,
        covariate_names: None,
        is_binary_trait: false,
        sample_mode: plan.compute.multi_phenotype_sample_mode,
    })
    .expect("fixture groups align before seeding");
    let prediction_fingerprints: Arc<[g_output::PredictionLocoFileFingerprint]> =
        build_prediction_loco_file_fingerprints_with_cache(
            &prediction_paths,
            &mut ManifestFileFingerprintCache::default(),
        )
        .expect("fixture prediction files are fingerprinted")
        .into();
    let output_plan = RuntimeOutputPlan {
        variant_count: reader.variant_count(),
        resolved_gpu_genotype_format: g_plan::GpuGenotypeFormat::Packed8,
        bgen_source_identity: Arc::new(reader.source_identity().clone()),
    };
    let initializations = groups
        .iter()
        .flat_map(|group| {
            build_runtime_output_initializations(
                &RuntimeOutputGroupInput {
                    phenotype_group: &group.phenotype_group,
                    covariate_names: &group.covariate_names,
                    sample_count: group.sample_indices.len(),
                },
                &output_plan,
                &prediction_fingerprints,
            )
            .expect("seeding uses real runtime output headers")
        })
        .collect();
    let mut manager = g_output::OutputManager::open(plan, String::new()).expect("seeding opens output lifecycle");
    let chunk_ranges = (0..reader.variant_count()).map(|start| start..start + 1).collect::<Vec<_>>();
    manager.initialize(initializations, &chunk_ranges, false).expect("seeding initializes every planned chunk");
    assert_eq!(phenotype_names.len(), committed_chunks.len());
    for (phenotype_name, chunks) in phenotype_names.iter().zip(committed_chunks) {
        let delivery = manager
            .delivery_state_for_phenotypes(std::slice::from_ref(phenotype_name))
            .expect("seeding selects an existing writer");
        for &variant_start_index in *chunks {
            let metadata = reader
                .variant_metadata_slice(variant_start_index, variant_start_index + 1)
                .expect("seeding retains actual chromosome metadata");
            crate::output_write::write_host_association_batch(
                &delivery.writer_sessions,
                None,
                variant_start_index,
                NativeVariantMetadataHandle::try_new(&metadata).expect("fixture metadata is valid"),
                build_output_statistics(1),
                Regenie2StatisticBatch {
                    trait_count: 1,
                    variant_count: 1,
                    beta: vec![1.0],
                    standard_error: vec![1.0],
                    chi_squared: vec![2.0],
                    log10_p_value: vec![3.0],
                    correction_code: None,
                },
            )
            .expect("seeded chunk is accepted by the real writer");
        }
    }
    manager.finish_interrupted("SIGINT").expect("seeded chunks flush with valid resume provenance");
}

#[test]
fn asymmetric_resume_preserves_skipped_chromosome_state_and_reentry() {
    let fixture = compressed_fixture(&["1", "2", "1"]);
    seed_committed_chunks(shared_run_plan(&fixture), &[&[1], &[]]);
    let mut plan = shared_run_plan(&fixture);
    plan.output.resume = true;
    let backend = Arc::new(SharedBackend::new(SharedFailure::None));
    execute_shared(plan, &backend, None).expect("overlapping resumed groups complete");
    let events = backend.events();
    assert_eq!(event_count(&events, "source:"), 2, "{events:?}");
    assert_eq!(event_count(&events, "select:"), 4, "{events:?}");
    assert_eq!(event_count(&events, "transfer:"), 1, "{events:?}");
    assert_eq!(event_count(&events, "compute:"), 5, "{events:?}");
    assert!(events.contains(&"transfer:1:1".to_string()), "{events:?}");
    let chromosome_one = 1_f32.to_bits();
    let chromosome_two = 2_f32.to_bits();
    assert_eq!(
        event_count(&events, &format!("chromosome:0:{chromosome_one}")),
        1,
        "group zero must keep its chromosome state across its skipped chunk: {events:?}"
    );
    assert_eq!(event_count(&events, &format!("chromosome:0:{chromosome_two}")), 0, "{events:?}");
    assert_eq!(
        event_count(&events, &format!("chromosome:1:{chromosome_one}")),
        2,
        "group one must reload chromosome one after chromosome two: {events:?}"
    );
    assert_eq!(event_count(&events, &format!("chromosome:1:{chromosome_two}")), 1, "{events:?}");
    assert_eq!(event_count(&events, "release_chromosome:0:"), 1, "{events:?}");
    assert_eq!(event_count(&events, "release_chromosome:1:"), 3, "{events:?}");
    assert_released_groups(&events, 2);
    for phenotype_name in ["trait-a", "trait-b"] {
        let manifest = fixture.manifest(phenotype_name);
        assert_eq!(manifest["status"], "completed");
        let identifiers = manifest["committed_chunks"]
            .as_array()
            .expect("commit array")
            .iter()
            .map(|chunk| chunk["chunk_identifier"].as_u64().expect("positive chunk identifier"))
            .collect::<BTreeSet<_>>();
        assert_eq!(identifiers, BTreeSet::from([0, 1, 2]));
    }
}

#[test]
fn disjoint_resume_uses_each_groups_original_pending_chunks() {
    let fixture = compressed_fixture(&["1", "2", "1"]);
    seed_committed_chunks(shared_run_plan(&fixture), &[&[1], &[0, 2]]);
    let mut plan = shared_run_plan(&fixture);
    plan.output.resume = true;
    let backend = Arc::new(SharedBackend::new(SharedFailure::None));
    execute_shared(plan, &backend, None).expect("disjoint resumed groups complete");
    let events = backend.events();
    assert_eq!(event_count(&events, "source:"), 0, "{events:?}");
    assert_eq!(event_count(&events, "select:"), 0, "{events:?}");
    let transfers =
        events.iter().filter(|event| event.starts_with("transfer:")).map(String::as_str).collect::<Vec<_>>();
    assert_eq!(transfers, ["transfer:0:0", "transfer:0:2", "transfer:1:1"]);
    assert_eq!(event_count(&events, "release_group:"), 2, "{events:?}");
    for phenotype_name in ["trait-a", "trait-b"] {
        assert_eq!(fixture.manifest(phenotype_name)["status"], "completed");
    }
}

fn three_trait_fixture() -> RunPreparationFixture {
    let fixture = compressed_fixture(&["1", "1", "1"]);
    fixture.write(
        "phenotypes.tsv",
        "FID\tIID\ttrait-a\ttrait-a2\ttrait-b\nfamily-1\tindividual-1\t1\t2\tNA\nfamily-2\tindividual-2\t2\t1\t1\nfamily-3\tindividual-3\t1\t3\t2\nfamily-4\tindividual-4\tNA\tNA\t1\n",
    );
    fixture
        .write("predictions.list", "trait-a predictions.loco\ntrait-a2 predictions.loco\ntrait-b predictions.loco\n");
    fixture
}

fn three_trait_run_plan(fixture: &RunPreparationFixture) -> g_plan::RunPlan {
    let mut plan = shared_run_plan(fixture);
    plan.phenotype_runs = ["trait-a", "trait-a2", "trait-b"]
        .into_iter()
        .map(|name| g_plan::PhenotypeRunPlan {
            phenotype_name: name.to_string(),
            output_directory_name: format!("{name}.run"),
        })
        .collect();
    plan
}

fn preserved_seeded_parts(fixture: &RunPreparationFixture) -> Vec<PreservedPart> {
    ["trait-a", "trait-a2"]
        .into_iter()
        .map(|phenotype_name| {
            let manifest = fixture.manifest(phenotype_name);
            let commits = manifest["committed_chunks"].as_array().expect("seeded commit array");
            assert_eq!(commits.len(), 1);
            let file_name = commits[0]["chunk_file_name"].as_str().expect("seeded part file name");
            let path = fixture.directory.join("output").join(format!("{phenotype_name}.run/parts")).join(file_name);
            let bytes = std::fs::read(&path).expect("seeded part is readable before resume");
            PreservedPart { path, bytes }
        })
        .collect()
}

#[test]
fn tiled_resume_selects_remaining_traits_and_preserves_committed_parts() {
    let fixture = three_trait_fixture();
    seed_committed_chunks(three_trait_run_plan(&fixture), &[&[0], &[1], &[]]);
    let preserved_parts = preserved_seeded_parts(&fixture);
    let mut plan = three_trait_run_plan(&fixture);
    plan.output.resume = true;
    let backend = Arc::new(SharedBackend::new(SharedFailure::None));
    let artifacts = execute_shared(plan, &backend, None).expect("partial traits resume through shared sources");
    assert_eq!(artifacts.len(), 3);
    let events = backend.events();
    assert_eq!(event_count(&events, "group:"), 2, "A and A2 must share a sample group: {events:?}");
    assert_eq!(event_count(&events, "source:"), 3, "{events:?}");
    assert_eq!(event_count(&events, "select:"), 6, "{events:?}");
    assert_eq!(
        event_count(&events, "transfer:"),
        0,
        "all consumers must use pretransferred shared selections: {events:?}"
    );
    assert_eq!(event_count(&events, "release_source:"), 3, "{events:?}");
    assert_released_groups(&events, 2);
    let materializations = backend.materializations.lock().expect("materialization observations remain available");
    assert_eq!(materializations.len(), 6);
    let grouped = materializations.iter().filter(|observation| observation.group_identifier == 0).collect::<Vec<_>>();
    assert_eq!(
        grouped,
        [
            &SharedMaterialization {
                group_identifier: 0,
                variant_start_index: 0,
                trait_count: 2,
                active_trait_indices: Some(vec![1])
            },
            &SharedMaterialization {
                group_identifier: 0,
                variant_start_index: 1,
                trait_count: 2,
                active_trait_indices: Some(vec![0])
            },
            &SharedMaterialization {
                group_identifier: 0,
                variant_start_index: 2,
                trait_count: 2,
                active_trait_indices: None
            },
        ]
    );
    let ungrouped = materializations.iter().filter(|observation| observation.group_identifier == 1).collect::<Vec<_>>();
    assert_eq!(ungrouped.len(), 3);
    for (variant_start_index, observation) in ungrouped.iter().enumerate() {
        assert_eq!(observation.variant_start_index, variant_start_index);
        assert_eq!(observation.trait_count, 1);
        assert_eq!(observation.active_trait_indices, None);
    }
    for preserved in preserved_parts {
        assert_eq!(std::fs::read(&preserved.path).expect("seeded part survives resume"), preserved.bytes);
    }
    for phenotype_name in ["trait-a", "trait-a2", "trait-b"] {
        let manifest = fixture.manifest(phenotype_name);
        assert_eq!(manifest["status"], "completed");
        let commits = manifest["committed_chunks"].as_array().expect("completed commit array");
        assert_eq!(commits.len(), 3);
        let identifiers = commits
            .iter()
            .map(|chunk| chunk["chunk_identifier"].as_u64().expect("positive chunk identifier"))
            .collect::<BTreeSet<_>>();
        assert_eq!(identifiers, BTreeSet::from([0, 1, 2]));
    }
}
