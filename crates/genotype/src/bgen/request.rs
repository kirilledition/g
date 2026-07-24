use std::path::Path;

use g_genotype_contracts::BgenContentSha256;

/// Authoritative content selection for one BGEN open request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BgenContentSelector {
    /// Required SHA-256 of the exact BGEN bytes.
    pub content_sha256: BgenContentSha256,
    /// Optional byte-count assertion in addition to the digest.
    pub expected_byte_count: Option<u64>,
}

/// One BGEN locator request with optional content selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BgenOpenRequest<'path> {
    /// Acquisition locator used only when selected content is not cached.
    pub locator: &'path Path,
    /// Authoritative selector enabling content-addressed snapshot reuse.
    pub content_selector: Option<BgenContentSelector>,
}
