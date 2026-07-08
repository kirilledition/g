//! Canonical requested-run planning contracts.

use serde::{Deserialize, Serialize};

use crate::enums::{
    ArrowCompression, AssociationMode, BinaryFallbackMethod, Device, FloatingPointDtype, GpuGenotypeFormat,
    JaxMatmulPrecision, MultiPhenotypeSampleMode, OutputFormat, ParquetCompression, PhenotypeComputeGroupMode,
    RegenieTraitType, ResumeMode, SampleKeyMode, TrustedBgenValidationMode,
};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RunRequest {
    pub association_mode: AssociationMode,
    pub input: InputRequest,
    pub trait_request: TraitRequest,
    pub compute: ComputeRequest,
    pub correction: CorrectionPlan,
    pub output: OutputWriterPlan,
    pub runtime: RuntimePlan,
    pub phenotype_runs: Vec<PhenotypeRunPlan>,
    pub phenotype_compute_groups: Vec<PhenotypeComputeGroup>,
    pub stage_timings_json: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct InputRequest {
    pub bgen_path: String,
    pub sample_path: Option<String>,
    pub phenotype_path: String,
    pub prediction_list_path: String,
    pub covariate_path: Option<String>,
    pub covariate_names: Vec<String>,
    pub sample_key_mode: SampleKeyMode,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct TraitRequest {
    pub trait_type: RegenieTraitType,
    pub chunk_size: u32,
    pub thread_count: Option<u32>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct ComputeRequest {
    pub device: Device,
    pub staging_depth: u32,
    pub native_callback_batch_size: u32,
    pub result_in_flight_limit: Option<u32>,
    pub dosage_buffer_limit: Option<u32>,
    pub variant_limit: Option<u32>,
    pub bgen_decode_tile_variant_count: u32,
    pub requested_gpu_genotype_format: GpuGenotypeFormat,
    pub trusted_no_missing_diploid: bool,
    pub trusted_bgen_validation_mode: TrustedBgenValidationMode,
    pub multi_phenotype_sample_mode: MultiPhenotypeSampleMode,
    pub score_dtype: FloatingPointDtype,
    pub firth_dtype: FloatingPointDtype,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CorrectionPlan {
    pub method: BinaryFallbackMethod,
    pub p_threshold: f64,
    pub firth_se: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct OutputWriterPlan {
    pub output_prefix: String,
    pub output_run_root: String,
    pub resume: bool,
    pub resume_mode: ResumeMode,
    pub finalize_parquet: bool,
    pub writer_thread_count: u32,
    pub writer_queue_depth: u32,
    pub chunks_per_arrow_file: u32,
    pub arrow_compression: ArrowCompression,
    pub parquet_compression: ParquetCompression,
    pub output_format: OutputFormat,
    pub output_statistic_dtype: FloatingPointDtype,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct RuntimePlan {
    pub jax_cache_directory: Option<String>,
    pub jax_matmul_precision: Option<JaxMatmulPrecision>,
    pub persistent_cache_enabled: bool,
    pub persistent_cache_min_entry_size_bytes: i64,
    pub persistent_cache_min_compile_time_seconds: u32,
    pub xla_autotune_cache_enabled: bool,
    pub transfer_guard_enabled: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct PhenotypeRunPlan {
    pub phenotype_index: u32,
    pub phenotype_name: String,
    pub output_directory_name: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct PhenotypeComputeGroup {
    pub group_mode: PhenotypeComputeGroupMode,
    pub phenotype_indices: Vec<u32>,
    pub phenotype_names: Vec<String>,
    pub sample_mode: MultiPhenotypeSampleMode,
    pub sample_set_fingerprint: Option<String>,
    pub covariate_design_fingerprint: Option<String>,
    pub prediction_alignment_fingerprint: Option<String>,
}
