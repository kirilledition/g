use std::error::Error;

const ASSUME_VALIDATED_UNSAFE_MESSAGE: &str = concat!(
    "Trusted no-missing diploid validation mode 'assume_validated' is unsafe for calculation runs. ",
    "Use 'cache_on_miss' or 'force_validate' so BGEN compatibility is checked before decoding."
);

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrustedBgenValidationCacheLookupError {
    UnsafeAssumeValidatedMode,
    UnsupportedValidationMode(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrustedBgenValidationCacheDirectoryError {
    MissingHomeDirectory,
}

impl std::fmt::Display for TrustedBgenValidationCacheLookupError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsafeAssumeValidatedMode => formatter.write_str(ASSUME_VALIDATED_UNSAFE_MESSAGE),
            Self::UnsupportedValidationMode(validation_mode) => {
                write!(formatter, "Unsupported trusted BGEN validation mode: {validation_mode}")
            }
        }
    }
}

impl Error for TrustedBgenValidationCacheLookupError {}

impl std::fmt::Display for TrustedBgenValidationCacheDirectoryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingHomeDirectory => formatter.write_str(
                "Unable to resolve trusted BGEN validation cache directory: neither XDG_CACHE_HOME nor HOME is set.",
            ),
        }
    }
}

impl Error for TrustedBgenValidationCacheDirectoryError {}
