use std::num::NonZeroU32;

use g_plan as plan;
use serde::Serialize;

use super::partial::PartialConfig;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ConfigProvenance {
    pub(crate) trait_config: TraitConfigProvenance,
    pub(crate) binary: BinaryConfigProvenance,
}

impl ConfigProvenance {
    pub(crate) fn from_partial_config(partial_config: &PartialConfig) -> Self {
        Self {
            trait_config: TraitConfigProvenance {
                trait_type: partial_config.trait_config.trait_type.is_some(),
                qt: partial_config.trait_config.qt.is_some(),
                bt: partial_config.trait_config.bt.is_some(),
            },
            binary: BinaryConfigProvenance {
                fallback_method: partial_config.binary.fallback_method.is_some(),
                p_threshold: partial_config.binary.p_threshold.is_some(),
                firth_se: partial_config.binary.firth_se.is_some(),
            },
        }
    }

    pub(crate) fn overlay(&mut self, override_provenance: Self) {
        self.trait_config.overlay(override_provenance.trait_config);
        self.binary.overlay(override_provenance.binary);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TraitConfigProvenance {
    pub(crate) trait_type: bool,
    pub(crate) qt: bool,
    pub(crate) bt: bool,
}

impl TraitConfigProvenance {
    fn overlay(&mut self, override_provenance: Self) {
        self.trait_type |= override_provenance.trait_type;
        self.qt |= override_provenance.qt;
        self.bt |= override_provenance.bt;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BinaryConfigProvenance {
    pub(crate) fallback_method: bool,
    pub(crate) p_threshold: bool,
    pub(crate) firth_se: bool,
}

impl BinaryConfigProvenance {
    fn overlay(&mut self, override_provenance: Self) {
        self.fallback_method |= override_provenance.fallback_method;
        self.p_threshold |= override_provenance.p_threshold;
        self.firth_se |= override_provenance.firth_se;
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct InputConfigData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bgen: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pheno_file: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub pheno_columns: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub covar_file: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub covar_columns: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pred: Option<String>,
}

impl InputConfigData {
    pub(crate) fn is_empty(&self) -> bool {
        self.bgen.is_none()
            && self.sample.is_none()
            && self.pheno_file.is_none()
            && self.pheno_columns.is_empty()
            && self.covar_file.is_none()
            && self.covar_columns.is_empty()
            && self.pred.is_none()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct TraitConfigData {
    pub trait_type: plan::RegenieTraitType,
    pub bsize: NonZeroU32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct BinaryConfigData {
    pub fallback_method: plan::BinaryFallbackMethod,
    pub p_threshold: plan::Probability,
    pub firth_se: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct GComputeConfigData {
    pub device: plan::Device,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_threads: Option<NonZeroU32>,
    pub multi_phenotype_sample_mode: plan::MultiPhenotypeSampleMode,
    pub firth_batch_size: NonZeroU32,
    pub firth_candidate_capacity: NonZeroU32,
    pub binary_null_maximum_iterations: NonZeroU32,
    pub binary_null_coefficient_tolerance: plan::PositiveF32,
    pub null_logistic_nonconvergence_policy: plan::NullLogisticNonconvergencePolicy,
    pub binary_minimum_probability: plan::ProbabilityFloor,
    pub binary_minimum_variance: plan::PositiveF32,
    pub binary_relative_variance_tolerance: plan::PositiveF32,
    pub linear_minimum_variance: plan::PositiveF32,
    pub linear_relative_variance_tolerance: plan::PositiveF32,
    pub firth_maximum_iterations: NonZeroU32,
    pub firth_gradient_tolerance: plan::PositiveF64,
    pub firth_coefficient_tolerance: plan::PositiveF64,
    pub firth_likelihood_tolerance: plan::PositiveF64,
    pub firth_maximum_step_size: plan::PositiveF64,
    pub firth_pseudo_maximum_iterations: NonZeroU32,
    pub firth_pseudo_inner_maximum_iterations: NonZeroU32,
    pub firth_newton_raphson_zero_start_iterations: NonZeroU32,
    pub firth_line_search_maximum_attempts: NonZeroU32,
    pub firth_step_halving_maximum_attempts: NonZeroU32,
    pub firth_initial_response_scale: plan::PositiveF64,
    pub firth_sparse_carrier_dosage_threshold: plan::DosageThreshold,
    pub firth_step_halving_scale: plan::StepScale,
    pub null_firth_maximum_iterations: NonZeroU32,
    pub null_firth_gradient_tolerance: plan::PositiveF64,
    pub null_firth_maximum_step_size: plan::PositiveF64,
    pub null_firth_fallback_iteration_multiplier: NonZeroU32,
    pub null_firth_fallback_step_divisor: plan::PositiveF64,
    pub null_firth_line_search_maximum_attempts: NonZeroU32,
    pub null_firth_step_halving_scale: plan::StepScale,
    pub use_block_firth_math: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jax_cache_dir: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct GOutputConfigData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_run_directory: Option<String>,
    pub writer_threads: NonZeroU32,
    pub resume: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct GDiagnosticsConfigData {
    pub telemetry: plan::TelemetryMode,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct RegenieConfigData {
    #[serde(skip_serializing_if = "InputConfigData::is_empty")]
    pub input: InputConfigData,
    #[serde(rename = "trait")]
    pub trait_config: TraitConfigData,
    pub binary: BinaryConfigData,
    #[serde(rename = "compute")]
    pub g_compute: GComputeConfigData,
    #[serde(rename = "output")]
    pub g_output: GOutputConfigData,
    #[serde(rename = "diagnostics")]
    pub g_diagnostics: GDiagnosticsConfigData,
    #[serde(skip)]
    pub(crate) provenance: ConfigProvenance,
}
