use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct OutputChunkCommit {
    pub(crate) chunk_identifier: i64,
    pub(crate) variant_start_index: i64,
    pub(crate) variant_stop_index: i64,
    pub(crate) row_count: i64,
    pub(crate) chunk_file_name: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OutputTransactionIdentifier(String);

impl OutputTransactionIdentifier {
    pub(crate) fn generate() -> Self {
        static NEXT_TRANSACTION_SEQUENCE: AtomicU64 = AtomicU64::new(0);

        let timestamp_nanoseconds =
            SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |duration| duration.as_nanos());
        let transaction_sequence = NEXT_TRANSACTION_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        Self(format!("{:08x}-{timestamp_nanoseconds:032x}-{transaction_sequence:016x}", std::process::id()))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }

    #[cfg(test)]
    pub(crate) fn for_test(identifier: &str) -> Self {
        Self(identifier.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::OutputTransactionIdentifier;

    #[test]
    fn generated_transaction_identifiers_are_distinct_and_path_safe() {
        let first = OutputTransactionIdentifier::generate();
        let second = OutputTransactionIdentifier::generate();

        assert_ne!(first, second);
        for identifier in [first, second] {
            assert!(identifier.as_str().chars().all(|character| character.is_ascii_hexdigit() || character == '-'));
        }
    }
}
