use std::path::{Component, Path};

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

pub(crate) fn generate_owner_claim_identifier() -> String {
    generate_identifier("owner")
}

pub(crate) fn validate_owner_claim_identifier(identifier: &str) -> OutputResult<()> {
    validate_identifier(identifier, "owner claim")
}

pub(crate) fn validate_run_set_identifier(identifier: &str) -> OutputResult<()> {
    validate_identifier(identifier, "run-set")
}

pub(crate) fn validate_safe_path_component(component: &str, role: &str) -> OutputResult<()> {
    let mut components = Path::new(component).components();
    let is_one_normal_component =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    if component.is_empty()
        || component.len() > 255
        || component.contains('/')
        || component.contains('\\')
        || component.chars().any(|character| character.is_control() || matches!(character, '\u{2028}' | '\u{2029}'))
        || !is_one_normal_component
    {
        return Err(OutputError::InvalidInput(format!(
            "Output {role} must be one non-empty safe path component of at most 255 bytes with no CLI line separators."
        )));
    }
    Ok(())
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
    use super::{
        AttemptIdentifier, generate_owner_claim_identifier, generate_run_set_identifier,
        validate_owner_claim_identifier, validate_run_set_identifier, validate_safe_path_component,
    };

    #[test]
    fn identifiers_are_distinct_and_path_safe() {
        let first = AttemptIdentifier::generate();
        let second = AttemptIdentifier::generate();
        assert_ne!(first, second);
        for identifier in
            [first.as_str(), second.as_str(), &generate_run_set_identifier(), &generate_owner_claim_identifier()]
        {
            assert!(identifier.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')));
        }
    }

    #[test]
    fn external_identifiers_reject_empty_long_or_path_syntax() {
        for identifier in ["", "../escape", "slash/value", &"a".repeat(129)] {
            assert!(AttemptIdentifier::parse(identifier).is_err());
            assert!(validate_run_set_identifier(identifier).is_err());
            assert!(validate_owner_claim_identifier(identifier).is_err());
        }
    }

    #[test]
    fn output_names_require_one_safe_path_component() {
        for safe_name in ["trait_0001_height", "trait-β"] {
            validate_safe_path_component(safe_name, "directory").expect("safe component is accepted");
        }
        for unsafe_name in [
            "",
            ".",
            "..",
            "../escape",
            "nested/output",
            r"nested\\output",
            "line\nbreak",
            "line\u{2028}break",
            "line\u{2029}break",
            &"a".repeat(256),
        ] {
            assert!(validate_safe_path_component(unsafe_name, "directory").is_err());
        }
    }
}
