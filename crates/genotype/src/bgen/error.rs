use thiserror::Error;

#[derive(Error, Debug)]
pub enum BgenError {
    #[error("{0}")]
    InvalidFormat(String),
    #[error("{0}")]
    UnsupportedFormat(String),
    #[error("{0}")]
    Range(String),
    #[error("I/O error while reading BGEN file: {0}")]
    Io(#[from] std::io::Error),
}
