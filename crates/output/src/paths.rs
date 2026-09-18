//! Read-only resolution of output directories through existing filesystem links.

use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};

use crate::error::{OutputError, OutputResult};

pub(crate) fn resolve_output_directory(path: &Path) -> OutputResult<PathBuf> {
    let absolute_path = std::path::absolute(path).map_err(OutputError::runtime)?;
    let mut resolved_path = PathBuf::new();
    for component in absolute_path.components() {
        match component {
            Component::CurDir => {}
            Component::Prefix(_) | Component::RootDir => resolved_path.push(component.as_os_str()),
            Component::ParentDir | Component::Normal(_) => {
                resolved_path.push(component.as_os_str());
                match resolved_path.canonicalize() {
                    Ok(canonical_path) => {
                        if !canonical_path.metadata().map_err(OutputError::runtime)?.is_dir() {
                            return Err(OutputError::InvalidInput(format!(
                                "Output directory '{}' traverses a non-directory at '{}'.",
                                path.display(),
                                resolved_path.display(),
                            )));
                        }
                        resolved_path = canonical_path;
                    }
                    Err(error)
                        if error.kind() == ErrorKind::NotFound
                            && !resolved_path
                                .symlink_metadata()
                                .is_ok_and(|metadata| metadata.file_type().is_symlink()) =>
                    {
                        // Missing suffixes may be normalized without creating them, but
                        // existing links must be resolved before a subsequent `..`.
                        if component == Component::ParentDir {
                            resolved_path.pop();
                            resolved_path.pop();
                        }
                    }
                    Err(error) => {
                        return Err(OutputError::InvalidInput(format!(
                            "Failed to resolve output directory '{}' at '{}': {error}",
                            path.display(),
                            resolved_path.display(),
                        )));
                    }
                }
            }
        }
    }
    Ok(resolved_path)
}
