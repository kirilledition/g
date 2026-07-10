use std::fmt;
use std::str::FromStr;

use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{ConfigError, ConfigResult};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct NameList(Vec<String>);

impl NameList {
    pub(crate) fn into_vec(self) -> Vec<String> {
        self.0
    }

    fn from_text(raw_value: &str) -> ConfigResult<Self> {
        Self::from_values(raw_value.split(',').map(ToOwned::to_owned).collect())
    }

    pub(crate) fn from_values(values: Vec<String>) -> ConfigResult<Self> {
        let mut normalized_values = Vec::with_capacity(values.len());
        for (zero_based_index, value) in values.into_iter().enumerate() {
            let normalized_value = value.trim();
            if normalized_value.is_empty() {
                return Err(ConfigError::new(format!(
                    "name list contains an empty entry at position {}.",
                    zero_based_index + 1
                )));
            }
            normalized_values.push(normalized_value.to_string());
        }
        if normalized_values.is_empty() {
            return Err(ConfigError::new("name list must contain at least one name."));
        }
        Ok(Self(normalized_values))
    }
}

impl FromStr for NameList {
    type Err = ConfigError;

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
