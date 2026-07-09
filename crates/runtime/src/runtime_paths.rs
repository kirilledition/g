//! Runtime path construction policy.

use std::path::{Path, PathBuf};

const UNKNOWN_USER_NAME: &str = "unknown";

#[must_use]
pub fn default_local_cache_directory(directory_name: &str) -> PathBuf {
    build_default_local_cache_directory(&std::env::temp_dir(), &default_local_cache_user_name(), directory_name)
}

#[must_use]
pub fn build_default_local_cache_directory(temporary_root: &Path, user_name: &str, directory_name: &str) -> PathBuf {
    let resolved_user_name = if user_name.is_empty() { UNKNOWN_USER_NAME } else { user_name };
    temporary_root.join(resolved_user_name).join(directory_name)
}

fn default_local_cache_user_name() -> String {
    std::env::var("USER")
        .ok()
        .filter(|user_name| !user_name.is_empty())
        .or_else(|| std::env::var("LOGNAME").ok().filter(|user_name| !user_name.is_empty()))
        .unwrap_or_else(|| UNKNOWN_USER_NAME.to_string())
}
