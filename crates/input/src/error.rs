//! Public input error boundary.

use crate::regenie::PredictionError;
use crate::sample::SampleAlignmentError;

#[derive(Debug, thiserror::Error)]
pub enum InputError {
    #[error(transparent)]
    Prediction(#[from] PredictionError),
    #[error(transparent)]
    SampleAlignment(#[from] SampleAlignmentError),
}

pub type InputResult<T> = Result<T, InputError>;
