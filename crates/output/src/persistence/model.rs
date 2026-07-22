#[derive(Debug, Eq, PartialEq)]
pub(crate) struct OutputChunkCommit {
    pub(crate) chunk_identifier: i64,
    pub(crate) variant_start_index: i64,
    pub(crate) variant_stop_index: i64,
    pub(crate) row_count: i64,
    pub(crate) chunk_file_name: String,
}
