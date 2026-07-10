//! Public input error boundary.

use crate::regenie::PredictionError;

#[derive(Debug, thiserror::Error)]
pub enum InputError {
    #[error(transparent)]
    Prediction(#[from] PredictionError),
    #[error("{0}")]
    SampleAlignment(String),
}

impl From<String> for InputError {
    fn from(message: String) -> Self {
        Self::SampleAlignment(message)
    }
}

pub type InputResult<T> = Result<T, InputError>;
