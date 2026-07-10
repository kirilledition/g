//! Validated floating-point planning values.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

macro_rules! validated_f64 {
    ($name:ident, $expectation:literal, $predicate:expr) => {
        #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
        pub struct $name(f64);

        impl $name {
            #[must_use]
            pub fn get(self) -> f64 {
                self.0
            }
        }

        impl TryFrom<f64> for $name {
            type Error = String;

            fn try_from(value: f64) -> Result<Self, Self::Error> {
                if !value.is_finite() {
                    return Err("must be finite".to_string());
                }
                if !($predicate)(value) {
                    return Err($expectation.to_string());
                }
                Ok(Self(value))
            }
        }

        impl FromStr for $name {
            type Err = String;

            fn from_str(raw_value: &str) -> Result<Self, Self::Err> {
                let value = raw_value.parse::<f64>().map_err(|_| "must be a number".to_string())?;
                Self::try_from(value)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<DeserializerType>(deserializer: DeserializerType) -> Result<Self, DeserializerType::Error>
            where
                DeserializerType: Deserializer<'de>,
            {
                let value = f64::deserialize(deserializer)?;
                Self::try_from(value).map_err(de::Error::custom)
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
                serializer.serialize_f64(self.0)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }
    };
}

macro_rules! validated_f32 {
    ($name:ident, $expectation:literal, $predicate:expr) => {
        #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
        pub struct $name(f32);

        impl $name {
            #[must_use]
            pub fn get(self) -> f32 {
                self.0
            }
        }

        impl TryFrom<f32> for $name {
            type Error = String;

            fn try_from(value: f32) -> Result<Self, Self::Error> {
                if !value.is_finite() {
                    return Err("must be finite".to_string());
                }
                if !($predicate)(value) {
                    return Err($expectation.to_string());
                }
                Ok(Self(value))
            }
        }

        impl FromStr for $name {
            type Err = String;

            fn from_str(raw_value: &str) -> Result<Self, Self::Err> {
                let value = raw_value.parse::<f32>().map_err(|_| "must be a number".to_string())?;
                Self::try_from(value)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<DeserializerType>(deserializer: DeserializerType) -> Result<Self, DeserializerType::Error>
            where
                DeserializerType: Deserializer<'de>,
            {
                let value = f32::deserialize(deserializer)?;
                Self::try_from(value).map_err(de::Error::custom)
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
                serializer.serialize_f32(self.0)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }
    };
}

validated_f64!(PositiveF64, "must be positive", |value: f64| value > 0.0);
validated_f64!(StepScale, "must be in (0, 1)", |value: f64| value > 0.0 && value < 1.0);
validated_f64!(DosageThreshold, "must be in (0, 2]", |value: f64| value > 0.0 && value <= 2.0);
validated_f32!(PositiveF32, "must be positive", |value: f32| value > 0.0);
validated_f32!(Probability, "must be in (0, 1)", |value: f32| value > 0.0 && value < 1.0);
validated_f32!(ProbabilityFloor, "must be in (0, 0.5)", |value: f32| value > 0.0 && value < 0.5);
