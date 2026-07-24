use std::collections::VecDeque;
use std::convert::Infallible;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use g_engine::{
    AssociationBackend, GenotypeDeliveryCapability, GroupPreparationInput, MaterializedAssociationBatch,
    PreparedChromosome,
};
use g_plan::{
    BinaryFallbackMethod, BinaryNullKernelPlan, ComputePlan, CorrectionPlan, Device, DosageThreshold, FirthKernelPlan,
    InputPlan, KernelPlan, LinearKernelPlan, MultiPhenotypeSampleMode, NullFirthKernelPlan,
    NullLogisticNonconvergencePolicy, OutputPlan, PhenotypeRunPlan, PositiveF32, PositiveF64, Probability,
    ProbabilityFloor, RunPlan, StepScale, TelemetryMode,
};

use crate::{
    JaxAssociationBackendPlan, JaxDevice, JaxRuntimeConfigUpdate, NativeRunFailure, NativeRunHost,
    NativeRunInterruption,
};

static NEXT_FIXTURE_IDENTIFIER: AtomicU64 = AtomicU64::new(0);

pub(crate) struct TemporaryRunFixture {
    root_path: PathBuf,
}

impl TemporaryRunFixture {
    pub(crate) fn new() -> Self {
        let fixture_identifier = NEXT_FIXTURE_IDENTIFIER.fetch_add(1, Ordering::Relaxed);
        let root_path = std::env::temp_dir().join(format!("g-runner-test-{}-{fixture_identifier}", std::process::id()));
        std::fs::create_dir_all(&root_path).expect("runner fixture directory should be created");
        for file_name in ["dataset.bgen", "dataset.sample", "phenotypes.tsv", "predictions.list"] {
            std::fs::write(root_path.join(file_name), []).expect("runner fixture input should be written");
        }
        Self { root_path }
    }

    pub(crate) fn root_path(&self) -> &Path {
        &self.root_path
    }

    pub(crate) fn run_plan(&self, association_mode: g_plan::AssociationMode) -> RunPlan {
        run_plan(self.root_path(), association_mode)
    }
}

pub(crate) fn execute_isolated_test_body(test_name: &str, child_environment_variable: &str) -> bool {
    if let Some(handshake_path) = std::env::var_os(child_environment_variable) {
        std::fs::write(handshake_path, test_name).expect("isolated runner test child should record its handshake");
        return true;
    }
    let fixture_identifier = NEXT_FIXTURE_IDENTIFIER.fetch_add(1, Ordering::Relaxed);
    let handshake_path = std::env::temp_dir()
        .join(format!("g-runner-isolated-test-{}-{fixture_identifier}.handshake", std::process::id()));
    let test_executable = std::env::current_exe().expect("current runner test executable should be available");
    let status = std::process::Command::new(test_executable)
        .arg("--exact")
        .arg(test_name)
        .arg("--nocapture")
        .env(child_environment_variable, &handshake_path)
        .status()
        .expect("isolated runner test subprocess should start");
    let handshake = std::fs::read_to_string(&handshake_path);
    let _ = std::fs::remove_file(&handshake_path);
    assert!(status.success(), "isolated runner test subprocess should succeed: {status}");
    assert_eq!(handshake.expect("isolated runner test child should write its handshake"), test_name);
    false
}

impl Drop for TemporaryRunFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root_path);
    }
}

pub(crate) fn run_plan(root_path: &Path, association_mode: g_plan::AssociationMode) -> RunPlan {
    RunPlan {
        association_mode,
        chunk_size: 16,
        input: InputPlan {
            bgen_path: path_text(&root_path.join("dataset.bgen")),
            bgen_content_sha256: None,
            sample_path: path_text(&root_path.join("dataset.sample")),
            phenotype_path: path_text(&root_path.join("phenotypes.tsv")),
            prediction_list_path: path_text(&root_path.join("predictions.list")),
            covariate_path: None,
            covariate_names: Vec::new(),
        },
        compute: ComputePlan {
            device: Device::Cpu,
            cpu_thread_count: None,
            jax_cache_directory: None,
            multi_phenotype_sample_mode: MultiPhenotypeSampleMode::PerPhenotype,
            kernels: KernelPlan {
                linear: LinearKernelPlan {
                    minimum_variance: PositiveF32::try_from(1.0e-8).expect("minimum variance should be valid"),
                    relative_variance_tolerance: PositiveF32::try_from(1.0e-6)
                        .expect("relative variance tolerance should be valid"),
                },
                binary_null: BinaryNullKernelPlan {
                    maximum_iterations: 50,
                    coefficient_tolerance: PositiveF32::try_from(1.0e-6)
                        .expect("coefficient tolerance should be valid"),
                    nonconvergence_policy: NullLogisticNonconvergencePolicy::Fail,
                    minimum_probability: ProbabilityFloor::try_from(1.0e-6)
                        .expect("minimum probability should be valid"),
                    minimum_variance: PositiveF32::try_from(1.0e-8).expect("minimum variance should be valid"),
                    relative_variance_tolerance: PositiveF32::try_from(1.0e-6)
                        .expect("relative variance tolerance should be valid"),
                },
                firth: FirthKernelPlan {
                    batch_size: 8,
                    candidate_capacity: 16,
                    maximum_iterations: 25,
                    gradient_tolerance: PositiveF64::try_from(2.5e-4).expect("gradient tolerance should be valid"),
                    maximum_step_size: PositiveF64::try_from(5.0).expect("maximum step size should be valid"),
                    pseudo_maximum_iterations: 10,
                    pseudo_inner_maximum_iterations: 5,
                    line_search_maximum_attempts: 4,
                    sparse_carrier_dosage_threshold: DosageThreshold::try_from(1.0e-4)
                        .expect("dosage threshold should be valid"),
                },
                null_firth: NullFirthKernelPlan {
                    maximum_iterations: 50,
                    gradient_tolerance: PositiveF64::try_from(5.0e-5).expect("gradient tolerance should be valid"),
                    maximum_step_size: PositiveF64::try_from(25.0).expect("maximum step size should be valid"),
                    fallback_iteration_multiplier: 5,
                    fallback_step_divisor: PositiveF64::try_from(5.0).expect("fallback step divisor should be valid"),
                    line_search_maximum_attempts: 4,
                    step_halving_scale: StepScale::try_from(0.5).expect("step scale should be valid"),
                },
            },
        },
        correction: CorrectionPlan {
            method: BinaryFallbackMethod::ScoreOnly,
            p_threshold: Probability::try_from(0.05).expect("probability should be valid"),
            firth_se: false,
        },
        output: OutputPlan {
            output_run_root: path_text(&root_path.join("output")),
            resume: false,
            writer_thread_count: 1,
        },
        telemetry: TelemetryMode::Off,
        phenotype_runs: vec![PhenotypeRunPlan {
            phenotype_name: "trait".to_string(),
            output_directory_name: "trait".to_string(),
        }],
    }
}

fn path_text(path: &Path) -> String {
    path.to_str().expect("runner fixture path should be UTF-8").to_string()
}

pub(crate) struct TestAssociationBackend;

impl AssociationBackend for TestAssociationBackend {
    type GroupState = ();
    type ChromosomeState = ();
    type TransferredInput = ();
    type DeviceResult = ();
    type Error = Infallible;

    fn genotype_delivery_capability(&self) -> GenotypeDeliveryCapability {
        GenotypeDeliveryCapability::HostOnly
    }

    fn prepare_group(&self, _input: GroupPreparationInput) -> Result<Self::GroupState, Self::Error> {
        Ok(())
    }

    fn prepare_chromosome(
        &self,
        _group: &Self::GroupState,
        _predictions: g_input::ChromosomePredictionMatrix,
    ) -> Result<PreparedChromosome<Self::ChromosomeState>, Self::Error> {
        Ok(PreparedChromosome { state: (), null_logistic_converged: None })
    }

    fn transfer_batch(
        &self,
        _group: &Self::GroupState,
        _input: g_genotype::GenotypeBatch,
    ) -> Result<Self::TransferredInput, Self::Error> {
        Ok(())
    }

    fn compute_batch(
        &self,
        _chromosome: &Self::ChromosomeState,
        _input: Self::TransferredInput,
    ) -> Result<Self::DeviceResult, Self::Error> {
        Ok(())
    }

    fn materialize_batch(
        &self,
        _result: Self::DeviceResult,
        _active_trait_indices: Option<&[usize]>,
        _logical_variant_count: usize,
    ) -> Result<MaterializedAssociationBatch, Self::Error> {
        unreachable!("the invalid-input runner fixture must fail before association materialization")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TestErrorKind {
    Failure,
    Sigint,
    Sigterm,
    FlushedSigint,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TestHostError {
    pub(crate) kind: TestErrorKind,
    pub(crate) message: String,
}

impl TestHostError {
    pub(crate) fn failure(message: impl Into<String>) -> Self {
        Self { kind: TestErrorKind::Failure, message: message.into() }
    }

    pub(crate) fn sigint(message: impl Into<String>) -> Self {
        Self { kind: TestErrorKind::Sigint, message: message.into() }
    }
}

impl fmt::Display for TestHostError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(formatter)
    }
}

impl std::error::Error for TestHostError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BackendPlanKind {
    Linear,
    BinaryScore,
    BinaryFirth,
}

#[derive(Default)]
pub(crate) struct TestNativeRunHost {
    pub(crate) calls: Vec<&'static str>,
    pub(crate) backend_plan_kinds: Vec<BackendPlanKind>,
    pub(crate) config_update_names: Vec<String>,
    pub(crate) observed_devices: Vec<JaxDevice>,
    pub(crate) interruption_results: VecDeque<Result<(), TestHostError>>,
    pub(crate) current_thread_error: Option<TestHostError>,
}

impl NativeRunHost for TestNativeRunHost {
    type Backend = TestAssociationBackend;
    type Error = TestHostError;

    fn install_python_logging(&mut self) -> Result<(), Self::Error> {
        self.calls.push("install_python_logging");
        Ok(())
    }

    fn apply_jax_config_updates(&mut self, updates: &[JaxRuntimeConfigUpdate<'_>]) -> Result<(), Self::Error> {
        self.calls.push("apply_jax_config_updates");
        self.config_update_names.extend(updates.iter().map(|update| update.setting_name.to_string()));
        Ok(())
    }

    fn observe_jax_devices(&mut self) -> Result<Vec<JaxDevice>, Self::Error> {
        self.calls.push("observe_jax_devices");
        Ok(std::mem::take(&mut self.observed_devices))
    }

    fn create_backend(
        &mut self,
        _device: Device,
        plan: JaxAssociationBackendPlan<'_>,
    ) -> Result<Arc<Self::Backend>, Self::Error> {
        self.calls.push("create_backend");
        let plan_kind = match plan {
            JaxAssociationBackendPlan::Linear(_) => BackendPlanKind::Linear,
            JaxAssociationBackendPlan::BinaryScore(_) => BackendPlanKind::BinaryScore,
            JaxAssociationBackendPlan::BinaryFirth { .. } => BackendPlanKind::BinaryFirth,
        };
        self.backend_plan_kinds.push(plan_kind);
        Ok(Arc::new(TestAssociationBackend))
    }

    fn check_interruption(&mut self) -> Result<(), Self::Error> {
        self.calls.push("check_interruption");
        self.interruption_results.pop_front().unwrap_or(Ok(()))
    }

    fn sigterm_interruption_error(&mut self) -> Self::Error {
        TestHostError { kind: TestErrorKind::Sigterm, message: "SIGTERM".to_string() }
    }

    fn flushed_interruption_error(&mut self, error: Self::Error) -> Self::Error {
        TestHostError { kind: TestErrorKind::FlushedSigint, message: error.message }
    }

    fn interruption_signal_name(error: &Self::Error) -> Option<&str> {
        match error.kind {
            TestErrorKind::Sigint | TestErrorKind::FlushedSigint => Some("SIGINT"),
            TestErrorKind::Sigterm => Some("SIGTERM"),
            TestErrorKind::Failure => None,
        }
    }

    fn interruption_kind(&mut self, error: &Self::Error) -> Option<NativeRunInterruption> {
        match error.kind {
            TestErrorKind::Failure => None,
            TestErrorKind::Sigint => Some(NativeRunInterruption::Sigint),
            TestErrorKind::Sigterm => Some(NativeRunInterruption::Sigterm),
            TestErrorKind::FlushedSigint => Some(NativeRunInterruption::FlushedSigint),
        }
    }

    fn run_error(&mut self, message: String) -> Self::Error {
        TestHostError::failure(message)
    }

    fn failed_event(&mut self, error: &Self::Error) -> NativeRunFailure {
        NativeRunFailure { error_type: "TestHostError".to_string(), error_message: error.message.clone() }
    }

    fn current_thread_name(&mut self) -> Result<String, Self::Error> {
        self.calls.push("current_thread_name");
        match self.current_thread_error.take() {
            Some(error) => Err(error),
            None => Ok("runner-test-thread".to_string()),
        }
    }

    fn detach<T, Operation>(operation: Operation) -> T
    where
        T: Send,
        Operation: FnOnce() -> T + Send,
    {
        operation()
    }
}
