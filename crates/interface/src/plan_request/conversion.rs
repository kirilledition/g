//! Conversion helpers from interface domains to plan domains.

use g_plan as plan;

use crate::ConfigError;
use crate::domain::{
    ArrowCompressionValue, DeviceValue, FloatingPointDtypeValue, GpuGenotypeFormatValue, JaxMatmulPrecisionValue,
    MultiPhenotypeSampleModeValue, OutputFormatValue, ParquetCompressionValue, RegenieTraitTypeValue, ResumeModeValue,
    SampleKeyModeValue, TrustedBgenValidationModeValue,
};

pub(super) fn plan_error_to_config_error(error: plan::HostPolicyError) -> ConfigError {
    match error {
        plan::HostPolicyError::NotImplemented(message) | plan::HostPolicyError::Value(message) => {
            ConfigError::new(message)
        }
    }
}

pub(super) fn plan_trait_type(value: RegenieTraitTypeValue) -> plan::RegenieTraitType {
    match value {
        RegenieTraitTypeValue::Quantitative => plan::RegenieTraitType::Quantitative,
        RegenieTraitTypeValue::Binary => plan::RegenieTraitType::Binary,
    }
}

pub(super) fn plan_device(value: DeviceValue) -> plan::Device {
    match value {
        DeviceValue::Cpu => plan::Device::Cpu,
        DeviceValue::Gpu => plan::Device::Gpu,
    }
}

pub(super) fn plan_trusted_bgen_validation_mode(
    value: TrustedBgenValidationModeValue,
) -> plan::TrustedBgenValidationMode {
    match value {
        TrustedBgenValidationModeValue::CacheOnMiss => plan::TrustedBgenValidationMode::CacheOnMiss,
        TrustedBgenValidationModeValue::ForceValidate => plan::TrustedBgenValidationMode::ForceValidate,
        TrustedBgenValidationModeValue::AssumeValidated => plan::TrustedBgenValidationMode::AssumeValidated,
    }
}

pub(super) fn plan_sample_key_mode(value: SampleKeyModeValue) -> plan::SampleKeyMode {
    match value {
        SampleKeyModeValue::Iid => plan::SampleKeyMode::Iid,
        SampleKeyModeValue::FidIid => plan::SampleKeyMode::FidIid,
    }
}

pub(super) fn plan_multi_phenotype_sample_mode(value: MultiPhenotypeSampleModeValue) -> plan::MultiPhenotypeSampleMode {
    match value {
        MultiPhenotypeSampleModeValue::PerPhenotype => plan::MultiPhenotypeSampleMode::PerPhenotype,
        MultiPhenotypeSampleModeValue::CompleteCase => plan::MultiPhenotypeSampleMode::CompleteCase,
    }
}

pub(super) fn plan_gpu_genotype_format(value: GpuGenotypeFormatValue) -> plan::GpuGenotypeFormat {
    match value {
        GpuGenotypeFormatValue::Auto => plan::GpuGenotypeFormat::Auto,
        GpuGenotypeFormatValue::Dosage => plan::GpuGenotypeFormat::Dosage,
        GpuGenotypeFormatValue::Packed8 => plan::GpuGenotypeFormat::Packed8,
    }
}

pub(super) fn plan_floating_point_dtype(value: FloatingPointDtypeValue) -> plan::FloatingPointDtype {
    match value {
        FloatingPointDtypeValue::Float32 => plan::FloatingPointDtype::Float32,
        FloatingPointDtypeValue::Float64 => plan::FloatingPointDtype::Float64,
    }
}

pub(super) fn plan_jax_matmul_precision(value: JaxMatmulPrecisionValue) -> plan::JaxMatmulPrecision {
    match value {
        JaxMatmulPrecisionValue::Float32 => plan::JaxMatmulPrecision::Float32,
        JaxMatmulPrecisionValue::TensorFloat32 => plan::JaxMatmulPrecision::TensorFloat32,
        JaxMatmulPrecisionValue::BrainFloat16 => plan::JaxMatmulPrecision::BrainFloat16,
        JaxMatmulPrecisionValue::Highest => plan::JaxMatmulPrecision::Highest,
    }
}

pub(super) fn plan_resume_mode(value: ResumeModeValue) -> plan::ResumeMode {
    match value {
        ResumeModeValue::Fast => plan::ResumeMode::Fast,
        ResumeModeValue::Strict => plan::ResumeMode::Strict,
    }
}

pub(super) fn plan_arrow_compression(value: ArrowCompressionValue) -> plan::ArrowCompression {
    match value {
        ArrowCompressionValue::Zstd => plan::ArrowCompression::Zstd,
        ArrowCompressionValue::None => plan::ArrowCompression::None,
    }
}

pub(super) fn plan_parquet_compression(value: ParquetCompressionValue) -> plan::ParquetCompression {
    match value {
        ParquetCompressionValue::Zstd => plan::ParquetCompression::Zstd,
        ParquetCompressionValue::None => plan::ParquetCompression::None,
    }
}

pub(super) fn plan_output_format(value: OutputFormatValue) -> plan::OutputFormat {
    match value {
        OutputFormatValue::Parquet => plan::OutputFormat::Parquet,
        OutputFormatValue::Arrow => plan::OutputFormat::Arrow,
        OutputFormatValue::Regenie => plan::OutputFormat::Regenie,
    }
}
