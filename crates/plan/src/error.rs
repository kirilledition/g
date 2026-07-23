//! Typed errors for validated planning values.

use std::error::Error;
use std::fmt;
use std::num::ParseFloatError;

/// Error returned when a closed planning enum cannot parse a string.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanEnumParseError {
    enum_name: &'static str,
    raw_value: Box<str>,
}

impl PlanEnumParseError {
    pub(crate) fn new(enum_name: &'static str, raw_value: &str) -> Self {
        Self { enum_name, raw_value: raw_value.into() }
    }

    /// Returns the Rust enum type that rejected the value.
    #[must_use]
    pub fn enum_name(&self) -> &'static str {
        self.enum_name
    }

    /// Returns the rejected string.
    #[must_use]
    pub fn raw_value(&self) -> &str {
        &self.raw_value
    }
}

impl fmt::Display for PlanEnumParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid value {:?}", self.raw_value)
    }
}

impl Error for PlanEnumParseError {}

/// Error returned when parsing or validating a numeric planning value.
#[derive(Debug)]
pub enum NumericValueError {
    /// The input was not a floating-point number.
    InvalidNumber { numeric_type_name: &'static str, source: ParseFloatError },
    /// The number violated the validated type's invariant.
    Validation { numeric_type_name: &'static str, expectation: &'static str },
}

impl NumericValueError {
    pub(crate) fn invalid_number(numeric_type_name: &'static str, source: ParseFloatError) -> Self {
        Self::InvalidNumber { numeric_type_name, source }
    }

    pub(crate) const fn validation(numeric_type_name: &'static str, expectation: &'static str) -> Self {
        Self::Validation { numeric_type_name, expectation }
    }

    /// Returns the validated planning type that could not be parsed.
    #[must_use]
    pub fn numeric_type_name(&self) -> &'static str {
        match self {
            Self::InvalidNumber { numeric_type_name, .. } | Self::Validation { numeric_type_name, .. } => {
                numeric_type_name
            }
        }
    }

    /// Returns the violated invariant, if the input was a number.
    #[must_use]
    pub fn expectation(&self) -> Option<&'static str> {
        match self {
            Self::InvalidNumber { .. } => None,
            Self::Validation { expectation, .. } => Some(expectation),
        }
    }
}

impl fmt::Display for NumericValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNumber { .. } => formatter.write_str("must be a number"),
            Self::Validation { expectation, .. } => formatter.write_str(expectation),
        }
    }
}

impl Error for NumericValueError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidNumber { source, .. } => Some(source),
            Self::Validation { .. } => None,
        }
    }
}
