//! Public genotype error boundary.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GenotypeError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Reader(String),
}

pub type GenotypeResult<T> = Result<T, GenotypeError>;
