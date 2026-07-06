use thiserror::Error;

use crate::error::GenotypeError;

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

pub(super) fn convert_bgen_error_to_genotype_error(error: &BgenError) -> GenotypeError {
    GenotypeError::Reader(error.to_string())
}
