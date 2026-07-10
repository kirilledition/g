use pyo3::exceptions::{PyKeyboardInterrupt, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::binding::engine::PyJaxBackendError;
use crate::binding::runtime;
use g_engine::TrustedBgenValidationError;
use g_engine::{
    BgenError, CoordinatedRunError, DeliveryError, GenotypeError, InputError, OutputError,
    PipelineOutputPreparationError, PredictionError, RunExecutionError, RunPreparationError, SchedulerError,
};

pub(super) fn convert_run_preparation_error(error: RunPreparationError) -> PyErr {
    match error {
        RunPreparationError::Bgen(error) => convert_bgen_error("prepare_run", error),
        RunPreparationError::Input(error) => convert_input_error("prepare_run", error),
        RunPreparationError::Output(error) => convert_output_error("prepare_run", error),
        RunPreparationError::OutputPreparation(error) => convert_pipeline_output_preparation_error(&error),
        RunPreparationError::Preflight(error) => PyValueError::new_err(error.to_string()),
        RunPreparationError::Prediction(error) => convert_prediction_error("prepare_run", &error),
        RunPreparationError::TrustedValidation(error) => convert_trusted_bgen_validation_error(error),
        RunPreparationError::TrustedValidationCacheDirectory(error) => PyRuntimeError::new_err(error.to_string()),
        error => PyValueError::new_err(error.to_string()),
    }
}

pub(super) fn convert_run_execution_error(error: RunExecutionError<PyJaxBackendError, PyErr>) -> PyErr {
    match error {
        RunExecutionError::Delivery(error) => convert_delivery_error(error),
        RunExecutionError::Interrupted(error) => Python::attach(|py| {
            if error.is_instance_of::<PyKeyboardInterrupt>(py) { runtime::flushed_interrupt_error() } else { error }
        }),
        RunExecutionError::InterruptedOutputFlush { interruption, output } => PyRuntimeError::new_err(format!(
            "Association interruption ({interruption}) was followed by an output flush failure: {output}"
        )),
        RunExecutionError::DeliveryAbort { delivery, output } => PyRuntimeError::new_err(format!(
            "Association delivery failed ({delivery}) and output abort also failed: {output}"
        )),
        RunExecutionError::OutputFinish(error) => convert_output_error("finish_run", error),
    }
}

pub(super) fn convert_coordinated_run_error(error: CoordinatedRunError<PyJaxBackendError, PyErr>) -> PyErr {
    match error {
        CoordinatedRunError::Preparation(error) => convert_run_preparation_error(error),
        CoordinatedRunError::Execution(error) => convert_run_execution_error(error),
        CoordinatedRunError::Telemetry(error) => PyRuntimeError::new_err(error.to_string()),
        CoordinatedRunError::Progress(error) => PyRuntimeError::new_err(error.to_string()),
        CoordinatedRunError::Diagnostic(error) => PyRuntimeError::new_err(error.to_string()),
        CoordinatedRunError::PhenotypeCountOutOfRange
        | CoordinatedRunError::ProcessedChunkCountOutOfRange
        | CoordinatedRunError::AssociationWarningCountOutOfRange
        | CoordinatedRunError::UnresolvedGpuGenotypeFormat
        | CoordinatedRunError::MissingPhenotypeOutput => PyValueError::new_err(error.to_string()),
    }
}

pub(super) fn convert_backend_error(operation: &str, error: PyJaxBackendError) -> PyErr {
    match error {
        PyJaxBackendError::InvalidInput(message) => {
            PyValueError::new_err(format!("Invalid association backend input during {operation}: {message}"))
        }
        PyJaxBackendError::Python(error) => error,
    }
}

pub(super) fn convert_scheduler_error(operation: &str, error: SchedulerError<PyJaxBackendError>) -> PyErr {
    match error {
        SchedulerError::Backend { source, .. } => convert_backend_error(operation, source),
        error => PyRuntimeError::new_err(format!("Association scheduler failed during {operation}: {error}")),
    }
}

pub(super) fn convert_delivery_error(error: DeliveryError<PyJaxBackendError, PyErr>) -> PyErr {
    match error {
        DeliveryError::Backend { stage, source } => convert_backend_error(stage, source),
        DeliveryError::Bgen(error) => convert_bgen_error("association_delivery", error),
        DeliveryError::Genotype(error) => convert_genotype_error("association_delivery", error),
        DeliveryError::Prediction(error) => convert_prediction_error("association_delivery", &error),
        DeliveryError::Input(error) => convert_input_error("association_delivery", error),
        DeliveryError::Output(error) => convert_output_error("association_delivery", error),
        DeliveryError::Progress(error) => PyRuntimeError::new_err(error.to_string()),
        DeliveryError::NullLogisticPolicy(error) => PyValueError::new_err(error.to_string()),
        DeliveryError::Interrupted(error) => error,
        DeliveryError::Scheduler(error) => convert_scheduler_error("association_delivery", error),
        DeliveryError::InvalidInput(message) => PyValueError::new_err(message),
        DeliveryError::NullLogisticNonconvergence(message) => PyRuntimeError::new_err(message),
    }
}

pub(super) fn convert_pipeline_output_preparation_error(error: &PipelineOutputPreparationError) -> PyErr {
    let (error_class, message) = match error {
        PipelineOutputPreparationError::Output(OutputError::InvalidInput(message)) => {
            ("output_invalid_input", message.clone())
        }
        PipelineOutputPreparationError::Output(OutputError::Runtime(message)) => ("output_runtime", message.clone()),
        PipelineOutputPreparationError::UnknownPlannedPhenotype { .. }
        | PipelineOutputPreparationError::UnresolvedGpuGenotypeFormat => ("output_preparation", error.to_string()),
    };
    tracing::warn!(
        target: "g.python",
        g_event = "native_boundary_error",
        subsystem = "engine",
        operation = "prepare_pipeline_output",
        error_class = error_class,
        error_message = %message,
        "Converting Rust pipeline output preparation error to Python."
    );
    match error {
        PipelineOutputPreparationError::Output(OutputError::Runtime(message)) => {
            PyRuntimeError::new_err(message.clone())
        }
        _ => PyValueError::new_err(message),
    }
}

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
    let GenotypeError::InvalidInput(message) = &error;
    tracing::warn!(
        target: "g.python",
        g_event = "native_boundary_error",
        subsystem = "genotype",
        operation = operation,
        error_class = "genotype_input",
        error_message = %message,
        "Converting Rust genotype error to Python."
    );
    let GenotypeError::InvalidInput(message) = error;
    PyValueError::new_err(message)
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

pub(super) fn convert_input_error(operation: &str, error: InputError) -> PyErr {
    match error {
        InputError::Prediction(prediction_error) => convert_prediction_error(operation, &prediction_error),
        InputError::SampleAlignment(sample_alignment_error) => {
            let error_message = sample_alignment_error.clone();
            tracing::warn!(
                target: "g.python",
                g_event = "native_boundary_error",
                subsystem = "input",
                operation = operation,
                error_class = "sample_alignment",
                error_message = %error_message,
                "Converting Rust input error to Python."
            );
            PyValueError::new_err(error_message)
        }
    }
}

pub(super) fn convert_trusted_bgen_validation_error(error: TrustedBgenValidationError) -> PyErr {
    match error {
        TrustedBgenValidationError::Bgen(error) => convert_bgen_error("validate_trusted_no_missing_diploid", error),
        TrustedBgenValidationError::Io(error) => PyOSError::new_err(error.to_string()),
        TrustedBgenValidationError::CacheLookup(error) => PyValueError::new_err(error.to_string()),
        TrustedBgenValidationError::SampleCountRange | TrustedBgenValidationError::VariantCountRange => {
            PyValueError::new_err(error.to_string())
        }
    }
}

pub(super) fn convert_output_error(operation: &str, error: OutputError) -> PyErr {
    let (error_class, message) = match &error {
        OutputError::InvalidInput(message) => ("output_invalid_input", message.clone()),
        OutputError::Runtime(message) => ("output_runtime", message.clone()),
    };
    tracing::warn!(
        target: "g.python",
        g_event = "native_boundary_error",
        subsystem = "output",
        operation = operation,
        error_class = error_class,
        error_message = %message,
        "Converting Rust output error to Python."
    );
    match error {
        OutputError::InvalidInput(message) => PyValueError::new_err(message),
        OutputError::Runtime(message) => PyRuntimeError::new_err(message),
    }
}
