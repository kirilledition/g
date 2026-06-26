use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::regenie::PredictionError;
use g_genotype::bgen::BgenError;
use g_genotype::common::GenotypeError;

pub(super) fn convert_bgen_error(operation: &str, error: BgenError) -> PyErr {
    let (error_class, message) = match &error {
        BgenError::InvalidFormat(message) | BgenError::UnsupportedFormat(message) | BgenError::Range(message) => {
            ("bgen_input", message.clone())
        }
        BgenError::Io(io_error) => ("bgen_io", io_error.to_string()),
    };
    tracing::warn!(
        target: "g.python",
        g_event = "native_boundary_error",
        subsystem = "bgen",
        operation = operation,
        error_class = error_class,
        error_message = %message,
        "Converting Rust BGEN error to Python."
    );
    match error {
        BgenError::InvalidFormat(message) | BgenError::UnsupportedFormat(message) | BgenError::Range(message) => {
            PyValueError::new_err(message)
        }
        BgenError::Io(io_error) => PyRuntimeError::new_err(io_error.to_string()),
    }
}

pub(super) fn convert_genotype_error(operation: &str, error: GenotypeError) -> PyErr {
    let (error_class, message) = match &error {
        GenotypeError::InvalidInput(message) => ("genotype_input", message.clone()),
        GenotypeError::Reader(message) => ("genotype_reader", message.clone()),
    };
    tracing::warn!(
        target: "g.python",
        g_event = "native_boundary_error",
        subsystem = "genotype",
        operation = operation,
        error_class = error_class,
        error_message = %message,
        "Converting Rust genotype error to Python."
    );
    match error {
        GenotypeError::InvalidInput(message) => PyValueError::new_err(message),
        GenotypeError::Reader(message) => PyRuntimeError::new_err(message),
    }
}

pub(super) fn convert_prediction_error(operation: &str, error: &PredictionError) -> PyErr {
    let error_message = match error {
        PredictionError::PredictionListNotFound(path) => {
            let message = format!("Prediction list file not found: {}", path.display());
            tracing::warn!(
                target: "g.python",
                g_event = "native_boundary_error",
                subsystem = "prediction",
                operation = operation,
                error_class = "prediction_list_not_found",
                error_message = %message,
                "Converting Rust prediction error to Python."
            );
            return pyo3::exceptions::PyFileNotFoundError::new_err(message);
        }
        PredictionError::LocoFileNotFound(path) => {
            let message = format!("LOCO file not found: {}", path.display());
            tracing::warn!(
                target: "g.python",
                g_event = "native_boundary_error",
                subsystem = "prediction",
                operation = operation,
                error_class = "loco_file_not_found",
                error_message = %message,
                "Converting Rust prediction error to Python."
            );
            return pyo3::exceptions::PyFileNotFoundError::new_err(message);
        }
        PredictionError::Io(io_error) => {
            let message = io_error.to_string();
            tracing::warn!(
                target: "g.python",
                g_event = "native_boundary_error",
                subsystem = "prediction",
                operation = operation,
                error_class = "prediction_io",
                error_message = %message,
                "Converting Rust prediction error to Python."
            );
            return PyRuntimeError::new_err(message);
        }
        other_error => other_error.to_string(),
    };
    tracing::warn!(
        target: "g.python",
        g_event = "native_boundary_error",
        subsystem = "prediction",
        operation = operation,
        error_class = "prediction_error",
        error_message = %error_message,
        "Converting Rust prediction error to Python."
    );
    PyValueError::new_err(error_message)
}
