use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{OutputError, OutputResult};

const MAXIMUM_IDENTIFIER_LENGTH: usize = 128;

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub(crate) struct AttemptIdentifier(String);

impl AttemptIdentifier {
    pub(crate) fn generate() -> Self {
        Self(generate_identifier("attempt"))
    }

    pub(crate) fn parse(identifier: &str) -> OutputResult<Self> {
        validate_identifier(identifier, "attempt")?;
        Ok(Self(identifier.to_string()))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }

    #[cfg(test)]
    pub(crate) fn for_test(identifier: &str) -> Self {
        Self::parse(identifier).expect("test attempt identifier is valid")
    }
}

pub(crate) fn generate_run_set_identifier() -> String {
    generate_identifier("run-set")
}

pub(crate) fn validate_run_set_identifier(identifier: &str) -> OutputResult<()> {
    validate_identifier(identifier, "run-set")
}

fn generate_identifier(domain: &str) -> String {
    format!("{domain}-{}", Uuid::new_v4().simple())
}

fn validate_identifier(identifier: &str, role: &str) -> OutputResult<()> {
    if identifier.is_empty()
        || identifier.len() > MAXIMUM_IDENTIFIER_LENGTH
        || !identifier.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(OutputError::InvalidInput(format!(
            "Output {role} identifier must be a non-empty path-safe identifier of at most {MAXIMUM_IDENTIFIER_LENGTH} ASCII characters."
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{AttemptIdentifier, generate_run_set_identifier, validate_run_set_identifier};

    #[test]
    fn identifiers_are_distinct_and_path_safe() {
        let first = AttemptIdentifier::generate();
        let second = AttemptIdentifier::generate();
        assert_ne!(first, second);
        for identifier in [first.as_str(), second.as_str(), &generate_run_set_identifier()] {
            assert!(identifier.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')));
        }
    }

    #[test]
    fn external_identifiers_reject_empty_long_or_path_syntax() {
        for identifier in ["", "../escape", "slash/value", &"a".repeat(129)] {
            assert!(AttemptIdentifier::parse(identifier).is_err());
            assert!(validate_run_set_identifier(identifier).is_err());
        }
    }
}
