use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use g_engine::debug::{
    CallbackDiagnosticsError, PipelineResumeCompatibilityError, PreflightError, ScheduleError,
    TrustedBgenValidationError,
};
use g_genotype::{BgenError, GenotypeError};
use g_input::{InputError, PredictionError};
use g_interface::ConfigError;
use g_output::OutputError;
use g_plan::{HostPolicyError, PreparedPlanError};
use g_runtime::debug::{
    LoggingSinkError, LoggingSinkInitializationError, RayonRuntimeError, RayonThreadPoolConfigurationError,
    TimingFileError, TransferMetadataError,
};

pub(super) fn convert_schedule_error(error: &ScheduleError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(super) fn convert_pipeline_resume_compatibility_error(error: &PipelineResumeCompatibilityError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(super) fn convert_preflight_error(error: &PreflightError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(super) fn convert_timing_file_error(error: &TimingFileError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

pub(super) fn convert_transfer_metadata_error(error: &TransferMetadataError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(super) fn convert_callback_diagnostics_error(error: &CallbackDiagnosticsError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(super) fn convert_rayon_thread_pool_configuration_error(error: &RayonThreadPoolConfigurationError) -> PyErr {
    let message = error.to_string();
    match error {
        RayonThreadPoolConfigurationError::RuntimeConfiguration {
            source: RayonRuntimeError::InvalidThreadCount,
            ..
        } => PyValueError::new_err(message),
        RayonThreadPoolConfigurationError::RuntimeCompatibility(_)
        | RayonThreadPoolConfigurationError::RuntimeConfiguration { .. } => PyRuntimeError::new_err(message),
    }
}

pub(super) fn convert_logging_sink_initialization_error(error: LoggingSinkInitializationError<PyErr>) -> PyErr {
    match error {
        LoggingSinkInitializationError::Sink(sink_error) => convert_logging_sink_error(&sink_error),
        LoggingSinkInitializationError::HostLogging(host_logging_error) => host_logging_error,
    }
}

pub(super) fn convert_logging_sink_error(error: &LoggingSinkError) -> PyErr {
    match error {
        LoggingSinkError::InvalidLogFilter { .. } | LoggingSinkError::InvalidTraceFilter { .. } => {
            PyValueError::new_err(error.to_string())
        }
        LoggingSinkError::Writer(_) | LoggingSinkError::LoggingGuardMutexPoisoned => {
            PyRuntimeError::new_err(error.to_string())
        }
    }
}

pub(super) fn convert_telemetry_writer_error(error: &std::io::Error) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
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

pub(super) fn convert_input_error(operation: &str, error: InputError) -> PyErr {
    match error {
        InputError::Prediction(prediction_error) => convert_prediction_error(operation, &prediction_error),
        InputError::SampleAlignment(sample_alignment_error) => {
            let error_message = sample_alignment_error.to_string();
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

pub(super) fn convert_config_error(operation: &str, error: &ConfigError) -> PyErr {
    let error_message = error.to_string();
    tracing::warn!(
        target: "g.python",
        g_event = "native_boundary_error",
        subsystem = "config",
        operation = operation,
        error_class = "config_error",
        error_message = %error_message,
        "Converting Rust config error to Python."
    );
    PyValueError::new_err(error_message)
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

pub(super) fn convert_prepared_plan_error(error: &PreparedPlanError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(super) fn convert_host_policy_error(error: HostPolicyError) -> PyErr {
    match error {
        HostPolicyError::NotImplemented(message) | HostPolicyError::Value(message) => PyValueError::new_err(message),
    }
}
