use g_genotype_contracts::VariantMetadataInvariantError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BgenError {
    #[error("{0}")]
    InvalidFormat(String),
    #[error("{0}")]
    UnsupportedFormat(String),
    #[error("{0}")]
    Range(String),
    #[error("I/O error while reading BGEN file: {0}")]
    Io(#[from] std::io::Error),
}

pub(super) fn contextualize_variant_metadata_invariant(
    context: &str,
    error: VariantMetadataInvariantError,
) -> BgenError {
    BgenError::InvalidFormat(format!("{context}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_invariants_map_to_contextual_invalid_format_errors() {
        let error = contextualize_variant_metadata_invariant(
            "Parsed BGEN variant metadata violates its invariants",
            VariantMetadataInvariantError::VariantIdentifierOffsetStartMismatch { observed_offset: 3 },
        );

        match error {
            BgenError::InvalidFormat(message) => assert_eq!(
                message,
                "Parsed BGEN variant metadata violates its invariants: first variant identifier offset must be zero, observed 3"
            ),
            other => panic!("expected an invalid-format error, observed {other:?}"),
        }
    }
}
