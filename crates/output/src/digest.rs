//! Shared syntax checks for persisted cryptographic digests.

pub(crate) fn is_canonical_sha256(digest: &str) -> bool {
    digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::is_canonical_sha256;

    #[test]
    fn canonical_sha256_requires_exact_lowercase_hexadecimal_syntax() {
        assert!(is_canonical_sha256(&"a".repeat(64)));
        assert!(!is_canonical_sha256(&"a".repeat(63)));
        assert!(!is_canonical_sha256(&"a".repeat(65)));
        assert!(!is_canonical_sha256(&"A".repeat(64)));
        assert!(!is_canonical_sha256(&"g".repeat(64)));
    }
}
