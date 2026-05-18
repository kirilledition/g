//! Direct CUDA quantitative REGENIE step 2 linear kernel.

#![allow(clippy::missing_errors_doc)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc;

use crate::native::linear::{self, LinearChromosomeState, LinearChunkResult, LinearError};

const KERNEL_FUNCTION_NAME: &str = "compute_regenie_linear_chunk";
const MAX_SCORE_ROW_COUNT: usize = 16;
const DEFAULT_DEVICE_ORDINAL: usize = 0;
const KERNEL_SOURCE: &str = r#"
extern "C" __global__ void compute_regenie_linear_chunk(
    const float* __restrict__ genotype_values,
    const float* __restrict__ stacked_score_matrix,
    float* __restrict__ beta_values,
    float* __restrict__ standard_error_values,
    float* __restrict__ chi_squared_values,
    int sample_count,
    int variant_count,
    int score_row_count,
    int covariate_count,
    float adjusted_residual_sum_squares,
    float degrees_of_freedom
) {
    int variant_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (variant_index >= variant_count) {
        return;
    }

    float score_values[16];
    for (int score_row_index = 0; score_row_index < score_row_count; ++score_row_index) {
        score_values[score_row_index] = 0.0f;
    }

    float genotype_sum_squares = 0.0f;
    for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
        float genotype_value = genotype_values[(sample_index * variant_count) + variant_index];
        genotype_sum_squares += genotype_value * genotype_value;
        for (int score_row_index = 0; score_row_index < score_row_count; ++score_row_index) {
            float score_value = stacked_score_matrix[(score_row_index * sample_count) + sample_index];
            score_values[score_row_index] += score_value * genotype_value;
        }
    }

    float projection_sum_squares = 0.0f;
    for (int covariate_index = 0; covariate_index < covariate_count; ++covariate_index) {
        float projection_coordinate = score_values[covariate_index];
        projection_sum_squares += projection_coordinate * projection_coordinate;
    }

    float genotype_residual_sum_squares = genotype_sum_squares - projection_sum_squares;
    if (genotype_residual_sum_squares < 0.0f) {
        genotype_residual_sum_squares = 0.0f;
    }

    float covariance_with_phenotype = score_values[covariate_count];
    float covariance_squared = covariance_with_phenotype * covariance_with_phenotype;
    bool positive_genotype_residual = genotype_residual_sum_squares > 0.0f;
    float genotype_residual_inverse = positive_genotype_residual ? (1.0f / genotype_residual_sum_squares) : 0.0f;
    float nan_value = 0.0f / 0.0f;

    beta_values[variant_index] = positive_genotype_residual
        ? covariance_with_phenotype * genotype_residual_inverse
        : nan_value;

    float residual_sum_squares_after =
        adjusted_residual_sum_squares - (covariance_squared * genotype_residual_inverse);
    if (residual_sum_squares_after < 0.0f) {
        residual_sum_squares_after = 0.0f;
    }

    bool valid_standard_error = positive_genotype_residual && residual_sum_squares_after > 0.0f;
    standard_error_values[variant_index] = valid_standard_error
        ? sqrtf(residual_sum_squares_after * genotype_residual_inverse / degrees_of_freedom)
        : nan_value;
    chi_squared_values[variant_index] = valid_standard_error
        ? covariance_squared * genotype_residual_inverse * degrees_of_freedom / residual_sum_squares_after
        : 0.0f;
}
"#;

#[derive(Clone, Debug)]
pub struct CudaLinearKernelConfig {
    pub block_size: u32,
}

impl Default for CudaLinearKernelConfig {
    fn default() -> Self {
        Self { block_size: 256 }
    }
}

pub struct CudaLinearChromosomeState {
    stacked_score_matrix: CudaSlice<f32>,
    adjusted_residual_sum_squares: f32,
    sample_count: usize,
    score_row_count: usize,
    covariate_count: usize,
    degrees_of_freedom: f32,
}

pub struct CudaLinearKernelSession {
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,
    function: CudaFunction,
    block_size: u32,
    genotype_values: Option<CudaSlice<f32>>,
    beta_values: Option<CudaSlice<f32>>,
    standard_error_values: Option<CudaSlice<f32>>,
    chi_squared_values: Option<CudaSlice<f32>>,
}

impl CudaLinearKernelSession {
    pub fn new(config: CudaLinearKernelConfig) -> Result<(Self, BTreeMap<String, f64>), LinearError> {
        validate_block_size(config.block_size)?;
        let mut timing_seconds = BTreeMap::new();
        let compile_start_time = Instant::now();
        let context = CudaContext::new(DEFAULT_DEVICE_ORDINAL).map_err(cuda_driver_error)?;
        let stream = context.default_stream();
        let ptx = nvrtc::compile_ptx(KERNEL_SOURCE).map_err(cuda_compile_error)?;
        let module = context.load_module(ptx).map_err(cuda_driver_error)?;
        let function = module.load_function(KERNEL_FUNCTION_NAME).map_err(cuda_driver_error)?;
        timing_seconds.insert("cuda_kernel_compile_load".to_string(), compile_start_time.elapsed().as_secs_f64());
        Ok((
            Self {
                stream,
                _module: module,
                function,
                block_size: config.block_size,
                genotype_values: None,
                beta_values: None,
                standard_error_values: None,
                chi_squared_values: None,
            },
            timing_seconds,
        ))
    }

    pub fn prepare_chromosome_state(
        &self,
        chromosome_state: &LinearChromosomeState,
    ) -> Result<CudaLinearChromosomeState, LinearError> {
        if chromosome_state.score_row_count > MAX_SCORE_ROW_COUNT {
            return Err(LinearError::InvalidInput(format!(
                "CUDA linear kernel supports at most {MAX_SCORE_ROW_COUNT} score rows, observed {}.",
                chromosome_state.score_row_count,
            )));
        }
        let stacked_score_matrix =
            self.stream.clone_htod(&chromosome_state.stacked_score_matrix).map_err(cuda_driver_error)?;
        Ok(CudaLinearChromosomeState {
            stacked_score_matrix,
            adjusted_residual_sum_squares: chromosome_state.adjusted_residual_sum_squares,
            sample_count: chromosome_state.sample_count,
            score_row_count: chromosome_state.score_row_count,
            covariate_count: chromosome_state.covariate_count,
            degrees_of_freedom: chromosome_state.degrees_of_freedom,
        })
    }

    pub fn compute_chunk(
        &mut self,
        chromosome_state: &CudaLinearChromosomeState,
        genotype_values: &[f32],
        variant_count: usize,
    ) -> Result<LinearChunkResult, LinearError> {
        linear::validate_genotype_shape(chromosome_state.sample_count, genotype_values.len(), variant_count)?;
        let mut timing_seconds = BTreeMap::new();

        let genotype_upload_start_time = Instant::now();
        ensure_device_buffer(&self.stream, &mut self.genotype_values, genotype_values.len())?;
        let genotype_device = self
            .genotype_values
            .as_mut()
            .ok_or_else(|| LinearError::Backend("CUDA genotype buffer was not allocated.".to_string()))?;
        {
            let mut genotype_view = genotype_device.slice_mut(0..genotype_values.len());
            self.stream.memcpy_htod(genotype_values, &mut genotype_view).map_err(cuda_driver_error)?;
        }
        timing_seconds.insert("cuda_genotype_h2d".to_string(), genotype_upload_start_time.elapsed().as_secs_f64());

        ensure_device_buffer(&self.stream, &mut self.beta_values, variant_count)?;
        ensure_device_buffer(&self.stream, &mut self.standard_error_values, variant_count)?;
        ensure_device_buffer(&self.stream, &mut self.chi_squared_values, variant_count)?;

        let kernel_start_time = Instant::now();
        let sample_count = i32::try_from(chromosome_state.sample_count).map_err(|error| {
            LinearError::InvalidInput(format!("Sample count is too large for CUDA kernel arguments: {error}"))
        })?;
        let variant_count_argument = i32::try_from(variant_count).map_err(|error| {
            LinearError::InvalidInput(format!("Variant count is too large for CUDA kernel arguments: {error}"))
        })?;
        let score_row_count = i32::try_from(chromosome_state.score_row_count).map_err(|error| {
            LinearError::InvalidInput(format!("Score row count is too large for CUDA kernel arguments: {error}"))
        })?;
        let covariate_count = i32::try_from(chromosome_state.covariate_count).map_err(|error| {
            LinearError::InvalidInput(format!("Covariate count is too large for CUDA kernel arguments: {error}"))
        })?;
        let grid_block_count = u32::try_from(variant_count)
            .map_err(|error| LinearError::InvalidInput(format!("Variant count is too large: {error}")))?
            .div_ceil(self.block_size);
        {
            let genotype_view = self
                .genotype_values
                .as_ref()
                .ok_or_else(|| LinearError::Backend("CUDA genotype buffer was not allocated.".to_string()))?
                .slice(0..genotype_values.len());
            let mut beta_view = self
                .beta_values
                .as_mut()
                .ok_or_else(|| LinearError::Backend("CUDA beta buffer was not allocated.".to_string()))?
                .slice_mut(0..variant_count);
            let mut standard_error_view = self
                .standard_error_values
                .as_mut()
                .ok_or_else(|| LinearError::Backend("CUDA standard-error buffer was not allocated.".to_string()))?
                .slice_mut(0..variant_count);
            let mut chi_squared_view = self
                .chi_squared_values
                .as_mut()
                .ok_or_else(|| LinearError::Backend("CUDA chi-square buffer was not allocated.".to_string()))?
                .slice_mut(0..variant_count);
            let launch_config = LaunchConfig {
                grid_dim: (grid_block_count, 1, 1),
                block_dim: (self.block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch_arguments = self.stream.launch_builder(&self.function);
            launch_arguments.arg(&genotype_view);
            launch_arguments.arg(&chromosome_state.stacked_score_matrix);
            launch_arguments.arg(&mut beta_view);
            launch_arguments.arg(&mut standard_error_view);
            launch_arguments.arg(&mut chi_squared_view);
            launch_arguments.arg(&sample_count);
            launch_arguments.arg(&variant_count_argument);
            launch_arguments.arg(&score_row_count);
            launch_arguments.arg(&covariate_count);
            launch_arguments.arg(&chromosome_state.adjusted_residual_sum_squares);
            launch_arguments.arg(&chromosome_state.degrees_of_freedom);
            unsafe { launch_arguments.launch(launch_config) }.map_err(cuda_driver_error)?;
            self.stream.synchronize().map_err(cuda_driver_error)?;
        }
        timing_seconds.insert("cuda_kernel_launch_sync".to_string(), kernel_start_time.elapsed().as_secs_f64());

        let output_start_time = Instant::now();
        let beta = copy_device_prefix(
            &self.stream,
            self.beta_values
                .as_ref()
                .ok_or_else(|| LinearError::Backend("CUDA beta buffer was not allocated.".to_string()))?,
            variant_count,
        )?;
        let standard_error = copy_device_prefix(
            &self.stream,
            self.standard_error_values
                .as_ref()
                .ok_or_else(|| LinearError::Backend("CUDA standard-error buffer was not allocated.".to_string()))?,
            variant_count,
        )?;
        let chi_squared = copy_device_prefix(
            &self.stream,
            self.chi_squared_values
                .as_ref()
                .ok_or_else(|| LinearError::Backend("CUDA chi-square buffer was not allocated.".to_string()))?,
            variant_count,
        )?;
        timing_seconds.insert("cuda_output_d2h".to_string(), output_start_time.elapsed().as_secs_f64());

        let cpu_start_time = Instant::now();
        let mut result = linear::build_linear_chunk_result(beta, standard_error, chi_squared);
        timing_seconds.insert("cuda_cpu_log10p_checksum".to_string(), cpu_start_time.elapsed().as_secs_f64());
        result.timing_seconds = timing_seconds;
        Ok(result)
    }
}

fn validate_block_size(block_size: u32) -> Result<(), LinearError> {
    match block_size {
        128 | 256 | 512 => Ok(()),
        _ => Err(LinearError::InvalidInput(format!(
            "CUDA block size must be one of 128, 256, or 512; observed {block_size}.",
        ))),
    }
}

fn ensure_device_buffer(
    stream: &Arc<CudaStream>,
    buffer: &mut Option<CudaSlice<f32>>,
    required_length: usize,
) -> Result<(), LinearError> {
    let needs_allocation = buffer.as_ref().is_none_or(|existing_buffer| existing_buffer.len() < required_length);
    if needs_allocation {
        *buffer = Some(stream.alloc_zeros(required_length).map_err(cuda_driver_error)?);
    }
    Ok(())
}

fn copy_device_prefix(
    stream: &Arc<CudaStream>,
    buffer: &CudaSlice<f32>,
    value_count: usize,
) -> Result<Vec<f32>, LinearError> {
    let buffer_view = buffer.slice(0..value_count);
    stream.clone_dtoh(&buffer_view).map_err(cuda_driver_error)
}

fn cuda_driver_error(error: cudarc::driver::DriverError) -> LinearError {
    LinearError::Backend(format!("CUDA driver error: {error}"))
}

fn cuda_compile_error(error: nvrtc::CompileError) -> LinearError {
    LinearError::Backend(format!("CUDA compile error: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_block_size_validation_rejects_unexpected_values() {
        assert!(validate_block_size(128).is_ok());
        assert!(validate_block_size(256).is_ok());
        assert!(validate_block_size(512).is_ok());
        assert!(validate_block_size(64).is_err());
    }
}
