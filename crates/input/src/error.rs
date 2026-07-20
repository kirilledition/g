//! Public input error boundary.

use crate::regenie::PredictionError;

#[derive(Debug, thiserror::Error)]
pub enum InputError {
    #[error(transparent)]
    Prediction(#[from] PredictionError),
    #[error("Phenotype '{phenotype_name}' value '{value}' is not finite.")]
    NonFinitePhenotypeValue { phenotype_name: String, value: String },
    #[error("Covariate '{covariate_name}' value '{value}' is not finite.")]
    NonFiniteCovariateValue { covariate_name: String, value: String },
    #[error("{0}")]
    SampleAlignment(String),
}

impl From<String> for InputError {
    fn from(message: String) -> Self {
        Self::SampleAlignment(message)
    }
}

pub(crate) type InputResult<T> = Result<T, InputError>;
