use std::fmt::Debug;
use std::str::FromStr;

use g_plan::{
    AssociationMode, BinaryFallbackMethod, Device, GpuGenotypeFormat, MultiPhenotypeSampleMode,
    NullLogisticNonconvergencePolicy, PhenotypeComputeGroupMode, PlanEnumParseError, RegenieTraitType, TelemetryMode,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

fn assert_string_enum_variant<EnumType>(variant: EnumType, actual_value: &str, expected_value: &str)
where
    EnumType: Copy + Debug + DeserializeOwned + Eq + FromStr<Err = PlanEnumParseError> + Serialize,
{
    assert_eq!(actual_value, expected_value);
    let serialized_value = serde_json::to_value(variant).expect("enum serialization should succeed");
    assert_eq!(serialized_value, serde_json::Value::String(expected_value.to_string()));

    let parsed_value = expected_value.parse::<EnumType>().expect("documented enum value should parse");
    assert_eq!(parsed_value, variant);

    let deserialized_value =
        serde_json::from_value::<EnumType>(serialized_value).expect("serialized enum value should deserialize");
    assert_eq!(deserialized_value, variant);
}

fn assert_invalid_string_enum_value<EnumType>()
where
    EnumType: Debug + DeserializeOwned + FromStr<Err = PlanEnumParseError>,
{
    let parse_error = "__invalid__".parse::<EnumType>().expect_err("unknown enum value should be rejected");
    assert_eq!(parse_error.to_string(), "invalid value \"__invalid__\"");
    assert_eq!(parse_error.raw_value(), "__invalid__");

    let serialized_value = serde_json::Value::String("__invalid__".to_string());
    assert!(serde_json::from_value::<EnumType>(serialized_value).is_err());
}

fn assert_error_contract<ErrorType>()
where
    ErrorType: std::error::Error + Send + Sync + 'static,
{
}

#[test]
fn enum_parse_errors_are_owned_standard_errors_with_type_context() {
    assert_error_contract::<PlanEnumParseError>();

    let parse_error = "__invalid__".parse::<Device>().expect_err("unknown device should be rejected");
    assert_eq!(parse_error.enum_name(), "Device");
    assert_eq!(parse_error.raw_value(), "__invalid__");
    assert_eq!(parse_error.to_string(), "invalid value \"__invalid__\"");
    assert!(std::error::Error::source(&parse_error).is_none());
}

#[test]
fn association_and_trait_enums_keep_their_string_contracts() {
    assert_string_enum_variant(
        AssociationMode::Regenie2Linear,
        AssociationMode::Regenie2Linear.as_str(),
        "regenie2_linear",
    );
    assert_string_enum_variant(
        AssociationMode::Regenie2Binary,
        AssociationMode::Regenie2Binary.as_str(),
        "regenie2_binary",
    );
    assert_invalid_string_enum_value::<AssociationMode>();

    assert_string_enum_variant(RegenieTraitType::Quantitative, RegenieTraitType::Quantitative.as_str(), "quantitative");
    assert_string_enum_variant(RegenieTraitType::Binary, RegenieTraitType::Binary.as_str(), "binary");
    assert_invalid_string_enum_value::<RegenieTraitType>();
}

#[test]
fn compute_enums_keep_their_string_contracts() {
    assert_string_enum_variant(Device::Cpu, Device::Cpu.as_str(), "cpu");
    assert_string_enum_variant(Device::Gpu, Device::Gpu.as_str(), "gpu");
    assert_invalid_string_enum_value::<Device>();

    assert_string_enum_variant(GpuGenotypeFormat::Dosage, GpuGenotypeFormat::Dosage.as_str(), "dosage");
    assert_string_enum_variant(GpuGenotypeFormat::Packed8, GpuGenotypeFormat::Packed8.as_str(), "packed8");
    assert_invalid_string_enum_value::<GpuGenotypeFormat>();

    assert_string_enum_variant(
        NullLogisticNonconvergencePolicy::Fail,
        NullLogisticNonconvergencePolicy::Fail.as_str(),
        "fail",
    );
    assert_string_enum_variant(
        NullLogisticNonconvergencePolicy::Warn,
        NullLogisticNonconvergencePolicy::Warn.as_str(),
        "warn",
    );
    assert_invalid_string_enum_value::<NullLogisticNonconvergencePolicy>();
}

#[test]
fn phenotype_grouping_enums_keep_their_string_contracts() {
    assert_string_enum_variant(
        MultiPhenotypeSampleMode::PerPhenotype,
        MultiPhenotypeSampleMode::PerPhenotype.as_str(),
        "per-phenotype",
    );
    assert_string_enum_variant(
        MultiPhenotypeSampleMode::CompleteCase,
        MultiPhenotypeSampleMode::CompleteCase.as_str(),
        "complete-case",
    );
    assert_invalid_string_enum_value::<MultiPhenotypeSampleMode>();

    assert_string_enum_variant(
        PhenotypeComputeGroupMode::SinglePhenotype,
        PhenotypeComputeGroupMode::SinglePhenotype.as_str(),
        "single-phenotype",
    );
    assert_string_enum_variant(
        PhenotypeComputeGroupMode::CompleteCase,
        PhenotypeComputeGroupMode::CompleteCase.as_str(),
        "complete-case",
    );
    assert_string_enum_variant(
        PhenotypeComputeGroupMode::PerPhenotypeCompatible,
        PhenotypeComputeGroupMode::PerPhenotypeCompatible.as_str(),
        "per-phenotype-compatible",
    );
    assert_invalid_string_enum_value::<PhenotypeComputeGroupMode>();
}

#[test]
fn correction_and_telemetry_enums_keep_their_string_contracts() {
    assert_string_enum_variant(BinaryFallbackMethod::ScoreOnly, BinaryFallbackMethod::ScoreOnly.as_str(), "score_only");
    assert_string_enum_variant(
        BinaryFallbackMethod::FirthApproximate,
        BinaryFallbackMethod::FirthApproximate.as_str(),
        "firth_approximate",
    );
    assert_invalid_string_enum_value::<BinaryFallbackMethod>();

    assert_string_enum_variant(TelemetryMode::Off, TelemetryMode::Off.as_str(), "off");
    assert_string_enum_variant(TelemetryMode::Progress, TelemetryMode::Progress.as_str(), "progress");
    assert_string_enum_variant(TelemetryMode::Profile, TelemetryMode::Profile.as_str(), "profile");
    assert_invalid_string_enum_value::<TelemetryMode>();
}

#[test]
fn string_enum_parsing_is_case_sensitive() {
    assert!("CPU".parse::<Device>().is_err());
    assert!("Packed8".parse::<GpuGenotypeFormat>().is_err());
    assert!("complete_case".parse::<MultiPhenotypeSampleMode>().is_err());
    assert!("PROFILE".parse::<TelemetryMode>().is_err());
}
