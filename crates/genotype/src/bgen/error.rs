use g_genotype_contracts::{BgenContentSha256, VariantMetadataInvariantError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BgenError {
    #[error("{0}")]
    InvalidFormat(String),
    #[error("{0}")]
    UnsupportedFormat(String),
    #[error("{0}")]
    Range(String),
    #[error("BGEN content SHA-256 mismatch: expected {expected}, observed {observed}.")]
    ContentSha256Mismatch { expected: BgenContentSha256, observed: BgenContentSha256 },
    #[error("BGEN content byte-count mismatch: expected {expected_byte_count}, observed {observed_byte_count}.")]
    ContentByteCountMismatch { expected_byte_count: u64, observed_byte_count: u64 },
    #[error(
        "Content-selected BGEN input contains {source_byte_count} bytes, exceeding the owned-snapshot limit of {maximum_snapshot_byte_count} bytes."
    )]
    ContentSelectionRequiresOwnedSnapshot { source_byte_count: u64, maximum_snapshot_byte_count: u64 },
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
