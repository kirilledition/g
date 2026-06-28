//! Runtime path construction policy.

use std::path::{Path, PathBuf};

const UNKNOWN_USER_NAME: &str = "unknown";

#[must_use]
pub fn build_default_local_cache_directory(temporary_root: &Path, user_name: &str, directory_name: &str) -> PathBuf {
    let resolved_user_name = if user_name.is_empty() { UNKNOWN_USER_NAME } else { user_name };
    temporary_root.join(resolved_user_name).join(directory_name)
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::build_default_local_cache_directory;

    #[test]
    fn builds_user_scoped_cache_directory() {
        let cache_directory = build_default_local_cache_directory(Path::new("/tmp"), "alice", "g-jax-cache");

        assert_eq!(cache_directory, PathBuf::from("/tmp/alice/g-jax-cache"));
    }

    #[test]
    fn defaults_empty_user_name_to_unknown() {
        let cache_directory = build_default_local_cache_directory(Path::new("/tmp"), "", "g-jax-cache");

        assert_eq!(cache_directory, PathBuf::from("/tmp/unknown/g-jax-cache"));
    }
}
