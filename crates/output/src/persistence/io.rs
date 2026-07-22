use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

use crate::error::OutputError;

pub(crate) fn path_operation_error(operation: &str, path: &Path, error: &impl std::fmt::Display) -> OutputError {
    OutputError::Runtime(format!("Failed to {operation} '{}': {error}", path.display()))
}

pub(crate) trait OutputIo {
    type File: io::Write + Send;

    fn create_new_file(&self, path: &Path) -> io::Result<Self::File>;
    fn sync_file(&self, file: &Self::File, path: &Path) -> io::Result<()>;
    fn rename_file(&self, source_path: &Path, destination_path: &Path) -> io::Result<()>;
    fn sync_directory(&self, path: &Path) -> io::Result<()>;
    fn file_size(&self, path: &Path) -> io::Result<u64>;
    fn remove_file(&self, path: &Path) -> io::Result<()>;
}

pub(crate) struct StdOutputIo;

impl OutputIo for StdOutputIo {
    type File = File;

    fn create_new_file(&self, path: &Path) -> io::Result<Self::File> {
        OpenOptions::new().write(true).create_new(true).open(path)
    }

    fn sync_file(&self, file: &Self::File, _path: &Path) -> io::Result<()> {
        file.sync_all()
    }

    fn rename_file(&self, source_path: &Path, destination_path: &Path) -> io::Result<()> {
        std::fs::rename(source_path, destination_path)
    }

    fn sync_directory(&self, path: &Path) -> io::Result<()> {
        File::open(path)?.sync_all()
    }

    fn file_size(&self, path: &Path) -> io::Result<u64> {
        std::fs::metadata(path).map(|metadata| metadata.len())
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        std::fs::remove_file(path)
    }
}
