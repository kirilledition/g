use std::fmt;
use std::str::FromStr;

use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PositiveF32(f32);

impl PositiveF32 {
    pub(crate) fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for PositiveF32 {
    type Error = String;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        if !value.is_finite() {
            return Err("must be a finite float32".to_string());
        }
        if value <= 0.0 {
            return Err("must be positive".to_string());
        }
        Ok(Self(value))
    }
}

impl FromStr for PositiveF32 {
    type Err = String;

    fn from_str(raw_value: &str) -> Result<Self, Self::Err> {
        let value = raw_value.parse::<f32>().map_err(|_| "must be a number".to_string())?;
        Self::try_from(value)
    }
}

impl<'de> Deserialize<'de> for PositiveF32 {
    fn deserialize<DeserializerType>(deserializer: DeserializerType) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        let value = f32::deserialize(deserializer)?;
        Self::try_from(value).map_err(de::Error::custom)
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

impl TryFrom<f32> for Probability {
    type Error = String;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        if !value.is_finite() {
            return Err("must be a finite float32".to_string());
        }
        if !(0.0..1.0).contains(&value) {
            return Err("must be in (0, 1)".to_string());
        }
        Ok(Self(value))
    }
}

impl FromStr for Probability {
    type Err = String;

    fn from_str(raw_value: &str) -> Result<Self, Self::Err> {
        let value = raw_value.parse::<f32>().map_err(|_| "must be a number".to_string())?;
        Self::try_from(value)
    }
}

impl<'de> Deserialize<'de> for Probability {
    fn deserialize<DeserializerType>(deserializer: DeserializerType) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        let value = f32::deserialize(deserializer)?;
        Self::try_from(value).map_err(de::Error::custom)
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

impl TryFrom<f32> for ProbabilityFloor {
    type Error = String;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        let checked_value = PositiveF32::try_from(value)?.get();
        if checked_value >= 0.5 {
            return Err("must be less than 0.5".to_string());
        }
        Ok(Self(checked_value))
    }
}

impl FromStr for ProbabilityFloor {
    type Err = String;

    fn from_str(raw_value: &str) -> Result<Self, Self::Err> {
        let value = raw_value.parse::<f32>().map_err(|_| "must be a number".to_string())?;
        Self::try_from(value)
    }
}

impl<'de> Deserialize<'de> for ProbabilityFloor {
    fn deserialize<DeserializerType>(deserializer: DeserializerType) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        let value = f32::deserialize(deserializer)?;
        Self::try_from(value).map_err(de::Error::custom)
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

macro_rules! string_enum {
    (
        $name:ident {
            $($variant:ident => $value:literal),+ $(,)?
        }
    ) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
        pub enum $name {
            $(
                #[serde(rename = $value)]
                #[value(name = $value)]
                $variant
            ),+
        }

        impl $name {
            #[must_use]
            pub fn as_str(self) -> &'static str {
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

    };
}

string_enum!(RegenieTraitTypeValue {
    Quantitative => "quantitative",
    Binary => "binary",
});

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
    Auto => "auto",
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

    fn from_text(raw_value: &str) -> Result<Self, String> {
        Self::from_values(raw_value.split(',').map(ToOwned::to_owned).collect())
    }

    pub(crate) fn from_values(values: Vec<String>) -> Result<Self, String> {
        let mut normalized_values = Vec::with_capacity(values.len());
        for (zero_based_index, value) in values.into_iter().enumerate() {
            let normalized_value = value.trim();
            if normalized_value.is_empty() {
                return Err(format!("name list contains an empty entry at position {}.", zero_based_index + 1));
            }
            normalized_values.push(normalized_value.to_string());
        }
        if normalized_values.is_empty() {
            return Err("name list must contain at least one name.".to_string());
        }
        Ok(Self(normalized_values))
    }
}

impl FromStr for NameList {
    type Err = String;

    fn from_str(raw_value: &str) -> Result<Self, Self::Err> {
        Self::from_text(raw_value)
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
        NameList::from_text(value).map_err(de::Error::custom)
    }

    fn visit_string<ErrorType>(self, value: String) -> Result<Self::Value, ErrorType>
    where
        ErrorType: de::Error,
    {
        NameList::from_text(&value).map_err(de::Error::custom)
    }

    fn visit_seq<SequenceAccess>(self, mut sequence: SequenceAccess) -> Result<Self::Value, SequenceAccess::Error>
    where
        SequenceAccess: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<String>()? {
            values.push(value);
        }
        NameList::from_values(values).map_err(de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::NameList;

    #[test]
    fn name_list_trims_valid_names() {
        let names = NameList::from_str(" trait_a , trait_b ").expect("valid names should parse");

        assert_eq!(names.into_vec(), vec!["trait_a".to_string(), "trait_b".to_string()]);
    }

    #[test]
    fn name_list_rejects_empty_text_tokens() {
        let error = NameList::from_str("trait_a,,trait_b").expect_err("empty comma token should fail");

        assert!(error.contains("empty entry at position 2"));
    }

    #[test]
    fn name_list_rejects_empty_sequence_entries() {
        let error = NameList::from_values(vec!["trait_a".to_string(), " ".to_string(), "trait_b".to_string()])
            .expect_err("empty sequence entry should fail");

        assert!(error.contains("empty entry at position 2"));
    }

    #[test]
    fn name_list_rejects_empty_sequences() {
        let error = NameList::from_values(Vec::new()).expect_err("empty sequence should fail");

        assert!(error.contains("at least one name"));
    }
}
