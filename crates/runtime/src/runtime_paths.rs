//! Runtime path construction policy.

use std::path::{Path, PathBuf};

const UNKNOWN_USER_NAME: &str = "unknown";

#[must_use]
pub fn default_local_temporary_root() -> PathBuf {
    std::env::temp_dir()
}

#[must_use]
pub fn default_local_cache_directory(directory_name: &str) -> PathBuf {
    build_default_local_cache_directory(
        &default_local_temporary_root(),
        &default_local_cache_user_name(),
        directory_name,
    )
}

#[must_use]
pub fn build_default_local_cache_directory(temporary_root: &Path, user_name: &str, directory_name: &str) -> PathBuf {
    let resolved_user_name = if user_name.is_empty() { UNKNOWN_USER_NAME } else { user_name };
    temporary_root.join(resolved_user_name).join(directory_name)
}

#[must_use]
pub fn expand_current_user_path(path: &str) -> String {
    expand_current_user_path_from_home(path, non_empty_environment_path("HOME").as_deref())
}

#[must_use]
pub fn expand_current_user_path_from_home(path: &str, home_directory: Option<&Path>) -> String {
    if path == "~" {
        return home_directory.map_or_else(|| path.to_string(), |home| home.to_string_lossy().into_owned());
    }
    if let Some(relative_path) = path.strip_prefix("~/") {
        return home_directory
            .map_or_else(|| path.to_string(), |home| home.join(relative_path).to_string_lossy().into_owned());
    }
    path.to_string()
}

fn default_local_cache_user_name() -> String {
    std::env::var("USER")
        .ok()
        .filter(|user_name| !user_name.is_empty())
        .or_else(|| std::env::var("LOGNAME").ok().filter(|user_name| !user_name.is_empty()))
        .unwrap_or_else(|| UNKNOWN_USER_NAME.to_string())
}

fn non_empty_environment_path(variable_name: &str) -> Option<PathBuf> {
    std::env::var_os(variable_name).filter(|environment_value| !environment_value.is_empty()).map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{build_default_local_cache_directory, expand_current_user_path_from_home};

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

    #[test]
    fn expands_current_user_home_path() {
        assert_eq!(
            expand_current_user_path_from_home("~/custom/g/cache", Some(Path::new("/home/alice"))),
            "/home/alice/custom/g/cache",
        );
        assert_eq!(expand_current_user_path_from_home("~", Some(Path::new("/home/alice"))), "/home/alice",);
    }

    #[test]
    fn leaves_non_current_user_paths_unchanged() {
        assert_eq!(
            expand_current_user_path_from_home("~other/custom/g/cache", Some(Path::new("/home/alice"))),
            "~other/custom/g/cache",
        );
        assert_eq!(expand_current_user_path_from_home("~/custom/g/cache", None), "~/custom/g/cache",);
        assert_eq!(
            expand_current_user_path_from_home("/tmp/custom/g/cache", Some(Path::new("/home/alice"))),
            "/tmp/custom/g/cache",
        );
    }
}
