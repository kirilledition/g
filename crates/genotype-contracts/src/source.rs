use std::path::PathBuf;

/// Identity of the exact BGEN file opened by the native reader.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BgenSourceIdentity {
    pub configured_path: PathBuf,
    pub canonical_path: Option<PathBuf>,
    pub device_identifier: u64,
    pub inode_identifier: u64,
    pub change_time_nanoseconds: i64,
    pub modification_time_nanoseconds: i64,
    pub file_size: u64,
}
