//! Public genotype error boundary.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GenotypeError {
    #[error("{0}")]
    InvalidInput(String),
}

pub type GenotypeResult<T> = Result<T, GenotypeError>;
