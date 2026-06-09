use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;

use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{ConfigError, ConfigResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PositiveU32(u32);

impl PositiveU32 {
    pub(crate) fn get(self) -> u32 {
        self.0
    }
}

impl TryFrom<i64> for PositiveU32 {
    type Error = String;

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        if value <= 0 {
            return Err("must be positive".to_string());
        }
        u32::try_from(value).map(Self).map_err(|_| "is too large".to_string())
    }
}

impl FromStr for PositiveU32 {
    type Err = String;

    fn from_str(raw_value: &str) -> Result<Self, Self::Err> {
        let value = raw_value.parse::<i64>().map_err(|_| "must be an integer".to_string())?;
        Self::try_from(value)
    }
}

impl<'de> Deserialize<'de> for PositiveU32 {
    fn deserialize<DeserializerType>(deserializer: DeserializerType) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        let value = i64::deserialize(deserializer)?;
        Self::try_from(value).map_err(de::Error::custom)
    }
}

impl Serialize for PositiveU32 {
    fn serialize<SerializerType>(&self, serializer: SerializerType) -> Result<SerializerType::Ok, SerializerType::Error>
    where
        SerializerType: Serializer,
    {
        serializer.serialize_u32(self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct NonNegativeU32(u32);

impl NonNegativeU32 {
    pub(crate) fn get(self) -> u32 {
        self.0
    }
}

impl TryFrom<i64> for NonNegativeU32 {
    type Error = String;

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        u32::try_from(value)
            .map(Self)
            .map_err(|_| if value < 0 { "must be non-negative".to_string() } else { "is too large".to_string() })
    }
}

impl FromStr for NonNegativeU32 {
    type Err = String;

    fn from_str(raw_value: &str) -> Result<Self, Self::Err> {
        let value = raw_value.parse::<i64>().map_err(|_| "must be an integer".to_string())?;
        Self::try_from(value)
    }
}

impl<'de> Deserialize<'de> for NonNegativeU32 {
    fn deserialize<DeserializerType>(deserializer: DeserializerType) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        let value = i64::deserialize(deserializer)?;
        Self::try_from(value).map_err(de::Error::custom)
    }
}

impl Serialize for NonNegativeU32 {
    fn serialize<SerializerType>(&self, serializer: SerializerType) -> Result<SerializerType::Ok, SerializerType::Error>
    where
        SerializerType: Serializer,
    {
        serializer.serialize_u32(self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PositiveF32(f32);

impl PositiveF32 {
    pub(crate) fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f64> for PositiveF32 {
    type Error = String;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        let narrowed_value = f32_from_f64(value)?;
        if narrowed_value <= 0.0 {
            return Err("must be positive".to_string());
        }
        Ok(Self(narrowed_value))
    }
}

impl FromStr for PositiveF32 {
    type Err = String;

    fn from_str(raw_value: &str) -> Result<Self, Self::Err> {
        let value = raw_value.parse::<f64>().map_err(|_| "must be a number".to_string())?;
        Self::try_from(value)
    }
}

impl<'de> Deserialize<'de> for PositiveF32 {
    fn deserialize<DeserializerType>(deserializer: DeserializerType) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        deserializer.deserialize_any(F32Visitor::new("a positive finite float32", Self::try_from))
    }
}

impl Serialize for PositiveF32 {
    fn serialize<SerializerType>(&self, serializer: SerializerType) -> Result<SerializerType::Ok, SerializerType::Error>
    where
        SerializerType: Serializer,
    {
        serializer.serialize_f32(self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Probability(f32);

impl Probability {
    pub(crate) fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f64> for Probability {
    type Error = String;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        let narrowed_value = f32_from_f64(value)?;
        if !(0.0..1.0).contains(&narrowed_value) {
            return Err("must be in (0, 1)".to_string());
        }
        Ok(Self(narrowed_value))
    }
}

impl FromStr for Probability {
    type Err = String;

    fn from_str(raw_value: &str) -> Result<Self, Self::Err> {
        let value = raw_value.parse::<f64>().map_err(|_| "must be a number".to_string())?;
        Self::try_from(value)
    }
}

impl<'de> Deserialize<'de> for Probability {
    fn deserialize<DeserializerType>(deserializer: DeserializerType) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        deserializer.deserialize_any(F32Visitor::new("a float32 in (0, 1)", Self::try_from))
    }
}

impl Serialize for Probability {
    fn serialize<SerializerType>(&self, serializer: SerializerType) -> Result<SerializerType::Ok, SerializerType::Error>
    where
        SerializerType: Serializer,
    {
        serializer.serialize_f32(self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ProbabilityFloor(f32);

impl ProbabilityFloor {
    pub(crate) fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f64> for ProbabilityFloor {
    type Error = String;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        let narrowed_value = PositiveF32::try_from(value)?.get();
        if narrowed_value >= 0.5 {
            return Err("must be less than 0.5".to_string());
        }
        Ok(Self(narrowed_value))
    }
}

impl FromStr for ProbabilityFloor {
    type Err = String;

    fn from_str(raw_value: &str) -> Result<Self, Self::Err> {
        let value = raw_value.parse::<f64>().map_err(|_| "must be a number".to_string())?;
        Self::try_from(value)
    }
}

impl<'de> Deserialize<'de> for ProbabilityFloor {
    fn deserialize<DeserializerType>(deserializer: DeserializerType) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        deserializer.deserialize_any(F32Visitor::new("a positive float32 less than 0.5", Self::try_from))
    }
}

impl Serialize for ProbabilityFloor {
    fn serialize<SerializerType>(&self, serializer: SerializerType) -> Result<SerializerType::Ok, SerializerType::Error>
    where
        SerializerType: Serializer,
    {
        serializer.serialize_f32(self.0)
    }
}

struct F32Visitor<ValueType, Validator>
where
    Validator: Fn(f64) -> Result<ValueType, String>,
{
    expectation: &'static str,
    validate: Validator,
    output: PhantomData<ValueType>,
}

impl<ValueType, Validator> F32Visitor<ValueType, Validator>
where
    Validator: Fn(f64) -> Result<ValueType, String>,
{
    fn new(expectation: &'static str, validate: Validator) -> Self {
        Self { expectation, validate, output: PhantomData }
    }
}

impl<ValueType, Validator> Visitor<'_> for F32Visitor<ValueType, Validator>
where
    Validator: Fn(f64) -> Result<ValueType, String>,
{
    type Value = ValueType;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.expectation)
    }

    fn visit_i64<ErrorType>(self, value: i64) -> Result<Self::Value, ErrorType>
    where
        ErrorType: de::Error,
    {
        let float_value = value.to_string().parse::<f64>().map_err(de::Error::custom)?;
        (self.validate)(float_value).map_err(de::Error::custom)
    }

    fn visit_u64<ErrorType>(self, value: u64) -> Result<Self::Value, ErrorType>
    where
        ErrorType: de::Error,
    {
        let float_value = value.to_string().parse::<f64>().map_err(de::Error::custom)?;
        (self.validate)(float_value).map_err(de::Error::custom)
    }

    fn visit_f64<ErrorType>(self, value: f64) -> Result<Self::Value, ErrorType>
    where
        ErrorType: de::Error,
    {
        (self.validate)(value).map_err(de::Error::custom)
    }
}

fn f32_from_f64(value: f64) -> Result<f32, String> {
    let narrowed_value = value.to_string().parse::<f32>().map_err(|_| "must fit in finite float32".to_string())?;
    if !narrowed_value.is_finite() {
        return Err("must be a finite float32".to_string());
    }
    Ok(narrowed_value)
}

macro_rules! string_enum {
    (
        $name:ident {
            $($variant:ident => $value:literal),+ $(,)?
        }
    ) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum $name {
            $($variant),+
        }

        impl $name {
            pub(crate) fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $value),+
                }
            }
        }

        impl FromStr for $name {
            type Err = String;

            fn from_str(raw_value: &str) -> Result<Self, Self::Err> {
                match raw_value {
                    $($value => Ok(Self::$variant),)+
                    _ => Err(format!("invalid value {raw_value:?}")),
                }
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<DeserializerType>(
                deserializer: DeserializerType,
            ) -> Result<Self, DeserializerType::Error>
            where
                DeserializerType: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::from_str(&value).map_err(de::Error::custom)
            }
        }

        impl Serialize for $name {
            fn serialize<SerializerType>(
                &self,
                serializer: SerializerType,
            ) -> Result<SerializerType::Ok, SerializerType::Error>
            where
                SerializerType: Serializer,
            {
                serializer.serialize_str(self.as_str())
            }
        }
    };
}

string_enum!(DeviceValue {
    Cpu => "cpu",
    Gpu => "gpu",
});

string_enum!(TrustedBgenValidationModeValue {
    CacheOnMiss => "cache_on_miss",
    ForceValidate => "force_validate",
    AssumeValidated => "assume_validated",
});

string_enum!(SampleKeyModeValue {
    Iid => "iid",
    FidIid => "fid_iid",
});

string_enum!(MultiPhenotypeSampleModeValue {
    PerPhenotype => "per-phenotype",
    CompleteCase => "complete-case",
});

string_enum!(OutputFormatValue {
    Parquet => "parquet",
    Arrow => "arrow",
    Regenie => "regenie",
});

string_enum!(ResumeModeValue {
    Fast => "fast",
    Strict => "strict",
});

string_enum!(NullLogisticNonconvergencePolicyValue {
    Fail => "fail",
    Warn => "warn",
});

string_enum!(GpuGenotypeFormatValue {
    Dosage => "dosage",
    Packed8 => "packed8",
});

string_enum!(FloatingPointDtypeValue {
    Float32 => "float32",
    Float64 => "float64",
});

string_enum!(JaxMatmulPrecisionValue {
    Float32 => "float32",
    TensorFloat32 => "tensorfloat32",
    BrainFloat16 => "bfloat16",
    Highest => "highest",
});

string_enum!(ArrowCompressionValue {
    Zstd => "zstd",
    None => "none",
});

string_enum!(ParquetCompressionValue {
    Zstd => "zstd",
    None => "none",
});

string_enum!(TelemetryModeValue {
    Off => "off",
    Progress => "progress",
    Profile => "profile",
    Trace => "trace",
});

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct NameList(Vec<String>);

impl NameList {
    pub(crate) fn into_vec(self) -> Vec<String> {
        self.0
    }

    fn from_text(raw_value: &str) -> Self {
        Self(raw_value.split(',').map(str::trim).filter(|name| !name.is_empty()).map(ToOwned::to_owned).collect())
    }

    fn from_values(values: Vec<String>) -> Self {
        Self(values.into_iter().map(|name| name.trim().to_string()).filter(|name| !name.is_empty()).collect())
    }
}

impl<'de> Deserialize<'de> for NameList {
    fn deserialize<DeserializerType>(deserializer: DeserializerType) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        deserializer.deserialize_any(NameListVisitor)
    }
}

impl Serialize for NameList {
    fn serialize<SerializerType>(&self, serializer: SerializerType) -> Result<SerializerType::Ok, SerializerType::Error>
    where
        SerializerType: Serializer,
    {
        serializer.serialize_str(&self.0.join(","))
    }
}

struct NameListVisitor;

impl<'de> Visitor<'de> for NameListVisitor {
    type Value = NameList;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a comma-delimited string or a list of strings")
    }

    fn visit_str<ErrorType>(self, value: &str) -> Result<Self::Value, ErrorType>
    where
        ErrorType: de::Error,
    {
        Ok(NameList::from_text(value))
    }

    fn visit_string<ErrorType>(self, value: String) -> Result<Self::Value, ErrorType>
    where
        ErrorType: de::Error,
    {
        Ok(NameList::from_text(&value))
    }

    fn visit_seq<SequenceAccess>(self, mut sequence: SequenceAccess) -> Result<Self::Value, SequenceAccess::Error>
    where
        SequenceAccess: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<String>()? {
            values.push(value);
        }
        Ok(NameList::from_values(values))
    }
}

pub(crate) fn parse_cli_option_value(option_name: &str, raw_value: &str) -> ConfigResult<()> {
    match option_name {
        "step" => parse_cli::<u8>(option_name, raw_value),
        "bsize"
        | "threads"
        | "staging_depth"
        | "variant_limit"
        | "firth_batch_size"
        | "firth_candidate_capacity"
        | "binary_null_maximum_iterations"
        | "firth_maximum_iterations"
        | "firth_pseudo_maximum_iterations"
        | "firth_pseudo_inner_maximum_iterations"
        | "firth_newton_raphson_zero_start_iterations"
        | "firth_line_search_maximum_attempts"
        | "firth_step_halving_maximum_attempts"
        | "null_firth_maximum_iterations"
        | "null_firth_fallback_iteration_multiplier"
        | "null_firth_line_search_maximum_attempts"
        | "bgen_decode_tile_variant_count"
        | "writer_threads"
        | "writer_queue_depth"
        | "chunks_per_arrow_file"
        | "progress_interval_chunks"
        | "log_queue_size" => parse_cli::<PositiveU32>(option_name, raw_value),
        "trace_event_cap" | "jax_persistent_cache_min_compile_time_seconds" => {
            parse_cli::<NonNegativeU32>(option_name, raw_value)
        }
        "binary_null_coefficient_tolerance"
        | "binary_minimum_variance"
        | "binary_relative_variance_tolerance"
        | "linear_minimum_variance"
        | "linear_relative_variance_tolerance"
        | "firth_gradient_tolerance"
        | "firth_coefficient_tolerance"
        | "firth_likelihood_tolerance"
        | "firth_maximum_step_size"
        | "firth_initial_response_scale"
        | "firth_sparse_carrier_dosage_threshold"
        | "firth_step_halving_scale"
        | "null_firth_gradient_tolerance"
        | "null_firth_maximum_step_size"
        | "null_firth_fallback_step_divisor"
        | "null_firth_step_halving_scale"
        | "progress_interval_seconds" => parse_cli::<PositiveF32>(option_name, raw_value),
        "pThresh" => parse_cli::<Probability>(option_name, raw_value),
        "binary_minimum_probability" => parse_cli::<ProbabilityFloor>(option_name, raw_value),
        "device" => parse_cli::<DeviceValue>(option_name, raw_value),
        "trusted_bgen_validation_mode" => parse_cli::<TrustedBgenValidationModeValue>(option_name, raw_value),
        "sample_key_mode" => parse_cli::<SampleKeyModeValue>(option_name, raw_value),
        "multi_phenotype_sample_mode" => parse_cli::<MultiPhenotypeSampleModeValue>(option_name, raw_value),
        "format" => parse_cli::<OutputFormatValue>(option_name, raw_value),
        "resume_mode" => parse_cli::<ResumeModeValue>(option_name, raw_value),
        "null_logistic_nonconvergence_policy" => {
            parse_cli::<NullLogisticNonconvergencePolicyValue>(option_name, raw_value)
        }
        "gpu_genotype_format" => parse_cli::<GpuGenotypeFormatValue>(option_name, raw_value),
        "score_dtype" | "firth_dtype" => parse_cli::<FloatingPointDtypeValue>(option_name, raw_value),
        "jax_matmul_precision" => parse_cli::<JaxMatmulPrecisionValue>(option_name, raw_value),
        "arrow_compression" => parse_cli::<ArrowCompressionValue>(option_name, raw_value),
        "parquet_compression" => parse_cli::<ParquetCompressionValue>(option_name, raw_value),
        "telemetry" => parse_cli::<TelemetryModeValue>(option_name, raw_value),
        "jax_persistent_cache_min_entry_size_bytes" => parse_cli::<i64>(option_name, raw_value),
        _ => Ok(()),
    }
}

fn parse_cli<ValueType>(option_name: &str, raw_value: &str) -> ConfigResult<()>
where
    ValueType: FromStr,
    ValueType::Err: fmt::Display,
{
    raw_value
        .parse::<ValueType>()
        .map(|_| ())
        .map_err(|error| ConfigError::new(format!("Invalid value for --{option_name}: {raw_value:?} ({error}).")))
}
