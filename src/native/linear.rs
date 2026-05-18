//! Native quantitative REGENIE step 2 linear kernels.

#![allow(clippy::missing_errors_doc)]

use statrs::function::erf::erfc;
use thiserror::Error;

#[cfg(any(feature = "burn-wgpu", feature = "burn-cuda", feature = "cuda-kernel"))]
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct LinearState {
    pub whitened_covariate_transpose: Vec<f32>,
    pub phenotype_residual: Vec<f32>,
    pub sample_count: usize,
    pub covariate_count: usize,
    pub degrees_of_freedom: f32,
}

#[derive(Clone, Debug)]
pub struct LinearChromosomeState {
    pub stacked_score_matrix: Vec<f32>,
    pub adjusted_residual_sum_squares: f32,
    pub sample_count: usize,
    pub score_row_count: usize,
    pub covariate_count: usize,
    pub degrees_of_freedom: f32,
}

#[cfg(any(feature = "burn-wgpu", feature = "burn-cuda"))]
pub struct LinearDeviceChromosomeState<B: burn::tensor::backend::Backend> {
    stacked_score_tensor: burn::prelude::Tensor<B, 2>,
    adjusted_residual_sum_squares: f32,
    sample_count: usize,
    covariate_count: usize,
    degrees_of_freedom: f32,
}

#[derive(Clone, Debug, Default)]
pub struct LinearChunkResult {
    pub beta: Vec<f32>,
    pub standard_error: Vec<f32>,
    pub chi_squared: Vec<f32>,
    pub log10_p_value: Vec<f32>,
    pub checksum: f64,
    #[cfg(any(feature = "burn-wgpu", feature = "burn-cuda", feature = "cuda-kernel"))]
    pub timing_seconds: BTreeMap<String, f64>,
}

#[derive(Debug, Error)]
pub enum LinearError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Backend(String),
}

pub fn prepare_linear_state(
    covariate_values: &[f32],
    phenotype_values: &[f32],
    covariate_count: usize,
) -> Result<LinearState, LinearError> {
    let sample_count = phenotype_values.len();
    if covariate_count == 0 {
        return Err(LinearError::InvalidInput("Covariate count must be at least one.".to_string()));
    }
    if covariate_values.len() != sample_count * covariate_count {
        return Err(LinearError::InvalidInput(format!(
            "Covariate matrix shape mismatch: expected {} values, observed {}.",
            sample_count * covariate_count,
            covariate_values.len(),
        )));
    }
    if sample_count <= covariate_count + 1 {
        return Err(LinearError::InvalidInput(format!(
            "Sample count {sample_count} is too small for {covariate_count} covariates and one tested variant.",
        )));
    }

    let covariate_crossproduct = compute_crossproduct(covariate_values, sample_count, covariate_count);
    let cholesky_factor = cholesky_lower(&covariate_crossproduct, covariate_count)?;
    let covariate_transpose = transpose_row_major(covariate_values, sample_count, covariate_count);
    let whitened_covariate_transpose =
        solve_lower_triangular_matrix(&cholesky_factor, &covariate_transpose, covariate_count, sample_count)?;
    let phenotype_crossproduct =
        multiply_transpose_matrix_vector(covariate_values, phenotype_values, sample_count, covariate_count);
    let phenotype_projection =
        solve_positive_definite_from_cholesky(&cholesky_factor, &phenotype_crossproduct, covariate_count)?;
    let fitted_phenotype =
        multiply_matrix_vector(covariate_values, &phenotype_projection, sample_count, covariate_count);
    let phenotype_residual = phenotype_values
        .iter()
        .zip(fitted_phenotype.iter())
        .map(|(phenotype_value, fitted_value)| phenotype_value - fitted_value)
        .collect::<Vec<_>>();
    let degrees_of_freedom = sample_count - covariate_count - 1;
    #[allow(clippy::cast_precision_loss)]
    let degrees_of_freedom = degrees_of_freedom as f32;
    Ok(LinearState {
        whitened_covariate_transpose,
        phenotype_residual,
        sample_count,
        covariate_count,
        degrees_of_freedom,
    })
}

pub fn prepare_linear_chromosome_state(
    state: &LinearState,
    loco_predictions: &[f32],
) -> Result<LinearChromosomeState, LinearError> {
    if loco_predictions.len() != state.sample_count {
        return Err(LinearError::InvalidInput(format!(
            "LOCO prediction length mismatch: expected {}, observed {}.",
            state.sample_count,
            loco_predictions.len(),
        )));
    }
    let adjusted_residual = state
        .phenotype_residual
        .iter()
        .zip(loco_predictions.iter())
        .map(|(phenotype_residual, loco_prediction)| phenotype_residual - loco_prediction)
        .collect::<Vec<_>>();
    let adjusted_residual_sum_squares = adjusted_residual.iter().map(|value| value * value).sum::<f32>();
    let mut stacked_score_matrix = Vec::with_capacity((state.covariate_count + 1).saturating_mul(state.sample_count));
    stacked_score_matrix.extend_from_slice(&state.whitened_covariate_transpose);
    stacked_score_matrix.extend_from_slice(&adjusted_residual);
    Ok(LinearChromosomeState {
        stacked_score_matrix,
        adjusted_residual_sum_squares,
        sample_count: state.sample_count,
        score_row_count: state.covariate_count + 1,
        covariate_count: state.covariate_count,
        degrees_of_freedom: state.degrees_of_freedom,
    })
}

#[cfg(any(feature = "burn-wgpu", feature = "burn-cuda"))]
pub fn prepare_linear_device_chromosome_state<B: burn::tensor::backend::Backend>(
    chromosome_state: &LinearChromosomeState,
    device: &B::Device,
) -> LinearDeviceChromosomeState<B> {
    use burn::prelude::Tensor;
    use burn::tensor::TensorData;

    let stacked_score_tensor = Tensor::<B, 2>::from_data(
        TensorData::new(
            chromosome_state.stacked_score_matrix.clone(),
            [chromosome_state.score_row_count, chromosome_state.sample_count],
        ),
        device,
    );
    LinearDeviceChromosomeState {
        stacked_score_tensor,
        adjusted_residual_sum_squares: chromosome_state.adjusted_residual_sum_squares,
        sample_count: chromosome_state.sample_count,
        covariate_count: chromosome_state.covariate_count,
        degrees_of_freedom: chromosome_state.degrees_of_freedom,
    }
}

#[cfg(any(feature = "burn-wgpu", feature = "burn-cuda"))]
pub fn compute_linear_chunk_burn<B: burn::tensor::backend::Backend>(
    chromosome_state: &LinearDeviceChromosomeState<B>,
    genotype_values: Vec<f32>,
    variant_count: usize,
    device: &B::Device,
) -> Result<LinearChunkResult, LinearError> {
    use burn::prelude::Tensor;
    use burn::tensor::{TensorData, Transaction};
    use std::time::Instant;

    validate_genotype_shape(chromosome_state.sample_count, genotype_values.len(), variant_count)?;
    let mut timing_seconds = BTreeMap::new();
    let tensor_setup_start_time = Instant::now();
    let genotype_tensor = Tensor::<B, 2>::from_data(
        TensorData::new(genotype_values, [chromosome_state.sample_count, variant_count]),
        device,
    );
    timing_seconds.insert("burn_tensor_setup".to_string(), tensor_setup_start_time.elapsed().as_secs_f64());

    let graph_start_time = Instant::now();
    let (beta_tensor, standard_error_tensor, chi_squared_tensor) =
        compute_linear_chunk_tensors(chromosome_state, genotype_tensor, variant_count, device);
    timing_seconds.insert("burn_graph".to_string(), graph_start_time.elapsed().as_secs_f64());

    let materialization_start_time = Instant::now();
    let mut tensor_data_values = Transaction::<B>::default()
        .register(beta_tensor)
        .register(standard_error_tensor)
        .register(chi_squared_tensor)
        .try_execute()
        .map_err(|error| LinearError::Backend(error.to_string()))?;
    if tensor_data_values.len() != 3 {
        return Err(LinearError::Backend(format!(
            "Burn output transaction returned {} tensors instead of 3.",
            tensor_data_values.len(),
        )));
    }
    let chi_squared = tensor_data_values
        .pop()
        .ok_or_else(|| LinearError::Backend("Missing Burn chi-square output tensor.".to_string()))
        .and_then(|tensor_data| materialize_tensor_data(&tensor_data, variant_count, "chi-square"))?;
    let standard_error = tensor_data_values
        .pop()
        .ok_or_else(|| LinearError::Backend("Missing Burn standard-error output tensor.".to_string()))
        .and_then(|tensor_data| materialize_tensor_data(&tensor_data, variant_count, "standard-error"))?;
    let beta = tensor_data_values
        .pop()
        .ok_or_else(|| LinearError::Backend("Missing Burn beta output tensor.".to_string()))
        .and_then(|tensor_data| materialize_tensor_data(&tensor_data, variant_count, "beta"))?;
    timing_seconds
        .insert("burn_output_materialization".to_string(), materialization_start_time.elapsed().as_secs_f64());

    let cpu_log10_start_time = Instant::now();
    let mut result = build_linear_chunk_result(beta, standard_error, chi_squared);
    timing_seconds.insert("burn_cpu_log10p_checksum".to_string(), cpu_log10_start_time.elapsed().as_secs_f64());
    result.timing_seconds = timing_seconds;
    Ok(result)
}

#[cfg(any(feature = "burn-wgpu", feature = "burn-cuda"))]
fn compute_linear_chunk_tensors<B: burn::tensor::backend::Backend>(
    chromosome_state: &LinearDeviceChromosomeState<B>,
    genotype_tensor: burn::prelude::Tensor<B, 2>,
    variant_count: usize,
    device: &B::Device,
) -> (burn::prelude::Tensor<B, 1>, burn::prelude::Tensor<B, 1>, burn::prelude::Tensor<B, 1>) {
    use burn::prelude::Tensor;

    let score_tensor = chromosome_state.stacked_score_tensor.clone().matmul(genotype_tensor.clone());
    let genotype_sum_squares = (genotype_tensor.clone() * genotype_tensor).sum_dim(0).reshape([variant_count]);
    let projection_sum_squares = if chromosome_state.covariate_count == 0 {
        Tensor::<B, 1>::zeros([variant_count], device)
    } else {
        let projection_score_tensor =
            score_tensor.clone().slice([0..chromosome_state.covariate_count, 0..variant_count]);
        (projection_score_tensor.clone() * projection_score_tensor).sum_dim(0).reshape([variant_count])
    };
    let covariance_with_phenotype = score_tensor
        .slice([chromosome_state.covariate_count..chromosome_state.covariate_count + 1, 0..variant_count])
        .reshape([variant_count]);
    let covariance_squared = covariance_with_phenotype.clone() * covariance_with_phenotype.clone();
    let genotype_residual_sum_squares = (genotype_sum_squares - projection_sum_squares).clamp_min(0.0);
    let positive_genotype_mask = genotype_residual_sum_squares.clone().greater_elem(0.0);
    let safe_genotype_residual_sum_squares =
        genotype_residual_sum_squares.clone().mask_fill(positive_genotype_mask.clone().bool_not(), 1.0);
    let genotype_residual_inverse = Tensor::<B, 1>::zeros([variant_count], device)
        .mask_where(positive_genotype_mask.clone(), safe_genotype_residual_sum_squares.recip());
    let residual_sum_squares_after = (chromosome_state.adjusted_residual_sum_squares
        - (covariance_squared.clone() * genotype_residual_inverse.clone()))
    .clamp_min(0.0);
    let positive_residual_mask = residual_sum_squares_after.clone().greater_elem(0.0);
    let valid_standard_error_mask = positive_genotype_mask.clone().bool_and(positive_residual_mask);

    let beta_values = covariance_with_phenotype * genotype_residual_inverse.clone();
    let beta = Tensor::<B, 1>::full([variant_count], f32::NAN, device).mask_where(positive_genotype_mask, beta_values);

    let standard_error_values = (residual_sum_squares_after.clone() * genotype_residual_inverse.clone()
        / chromosome_state.degrees_of_freedom)
        .sqrt();
    let standard_error = Tensor::<B, 1>::full([variant_count], f32::NAN, device)
        .mask_where(valid_standard_error_mask.clone(), standard_error_values);

    let safe_residual_sum_squares_after =
        residual_sum_squares_after.mask_fill(valid_standard_error_mask.clone().bool_not(), 1.0);
    let chi_squared_values = covariance_squared * genotype_residual_inverse * chromosome_state.degrees_of_freedom
        / safe_residual_sum_squares_after;
    let chi_squared =
        Tensor::<B, 1>::zeros([variant_count], device).mask_where(valid_standard_error_mask, chi_squared_values);

    (beta, standard_error, chi_squared)
}

#[cfg(any(feature = "burn-wgpu", feature = "burn-cuda"))]
pub(crate) fn materialize_tensor_data(
    tensor_data: &burn::tensor::TensorData,
    expected_value_count: usize,
    tensor_name: &str,
) -> Result<Vec<f32>, LinearError> {
    let values = tensor_data
        .to_vec::<f32>()
        .map_err(|error| LinearError::Backend(format!("Failed to convert Burn {tensor_name} tensor data: {error}")))?;
    if values.len() != expected_value_count {
        return Err(LinearError::Backend(format!(
            "Burn {tensor_name} tensor length mismatch: expected {expected_value_count}, observed {}.",
            values.len(),
        )));
    }
    Ok(values)
}

pub fn compute_linear_chunk_cpu(
    chromosome_state: &LinearChromosomeState,
    genotype_values: &[f32],
    variant_count: usize,
) -> Result<LinearChunkResult, LinearError> {
    validate_genotype_shape(chromosome_state.sample_count, genotype_values.len(), variant_count)?;
    let mut score_values = vec![0.0_f32; chromosome_state.score_row_count * variant_count];
    for score_row_index in 0..chromosome_state.score_row_count {
        for sample_index in 0..chromosome_state.sample_count {
            let score_value =
                chromosome_state.stacked_score_matrix[(score_row_index * chromosome_state.sample_count) + sample_index];
            for variant_index in 0..variant_count {
                score_values[(score_row_index * variant_count) + variant_index] +=
                    score_value * genotype_values[(sample_index * variant_count) + variant_index];
            }
        }
    }
    let mut genotype_sum_squares = vec![0.0_f32; variant_count];
    for sample_index in 0..chromosome_state.sample_count {
        for variant_index in 0..variant_count {
            let genotype_value = genotype_values[(sample_index * variant_count) + variant_index];
            genotype_sum_squares[variant_index] += genotype_value * genotype_value;
        }
    }
    compute_linear_chunk_from_scores(chromosome_state, &score_values, &genotype_sum_squares, variant_count)
}

fn compute_linear_chunk_from_scores(
    chromosome_state: &LinearChromosomeState,
    score_values: &[f32],
    genotype_sum_squares: &[f32],
    variant_count: usize,
) -> Result<LinearChunkResult, LinearError> {
    if score_values.len() != chromosome_state.score_row_count * variant_count {
        return Err(LinearError::Backend("Burn score tensor returned an unexpected shape.".to_string()));
    }
    if genotype_sum_squares.len() != variant_count {
        return Err(LinearError::Backend(
            "Burn genotype sum-of-squares tensor returned an unexpected shape.".to_string(),
        ));
    }
    let mut result = LinearChunkResult {
        beta: Vec::with_capacity(variant_count),
        standard_error: Vec::with_capacity(variant_count),
        chi_squared: Vec::with_capacity(variant_count),
        log10_p_value: Vec::with_capacity(variant_count),
        checksum: 0.0,
        #[cfg(any(feature = "burn-wgpu", feature = "burn-cuda", feature = "cuda-kernel"))]
        timing_seconds: BTreeMap::new(),
    };
    for variant_index in 0..variant_count {
        let mut projection_sum_squares = 0.0_f32;
        for covariate_index in 0..chromosome_state.covariate_count {
            let projection_coordinate = score_values[(covariate_index * variant_count) + variant_index];
            projection_sum_squares += projection_coordinate * projection_coordinate;
        }
        let genotype_residual_sum_squares = (genotype_sum_squares[variant_index] - projection_sum_squares).max(0.0);
        let covariance_with_phenotype =
            score_values[(chromosome_state.covariate_count * variant_count) + variant_index];
        let covariance_squared = covariance_with_phenotype * covariance_with_phenotype;
        let positive_genotype_residual = genotype_residual_sum_squares > 0.0;
        let genotype_residual_inverse =
            if positive_genotype_residual { genotype_residual_sum_squares.recip() } else { 0.0 };
        let beta =
            if positive_genotype_residual { covariance_with_phenotype * genotype_residual_inverse } else { f32::NAN };
        let residual_sum_squares_after =
            (chromosome_state.adjusted_residual_sum_squares - covariance_squared * genotype_residual_inverse).max(0.0);
        let positive_residual_sum_squares = residual_sum_squares_after > 0.0;
        let standard_error = if positive_genotype_residual && positive_residual_sum_squares {
            (residual_sum_squares_after * genotype_residual_inverse / chromosome_state.degrees_of_freedom).sqrt()
        } else {
            f32::NAN
        };
        let chi_squared = if positive_genotype_residual && positive_residual_sum_squares {
            covariance_squared * genotype_residual_inverse * chromosome_state.degrees_of_freedom
                / residual_sum_squares_after
        } else {
            0.0
        };
        let log10_p_value = chi_squared_to_log10_p_value(chi_squared);
        result.checksum += f64::from(beta.to_bits());
        result.checksum += f64::from(standard_error.to_bits());
        result.checksum += f64::from(chi_squared.to_bits());
        result.checksum += f64::from(log10_p_value.to_bits());
        result.beta.push(beta);
        result.standard_error.push(standard_error);
        result.chi_squared.push(chi_squared);
        result.log10_p_value.push(log10_p_value);
    }
    Ok(result)
}

pub(crate) fn build_linear_chunk_result(
    beta: Vec<f32>,
    standard_error: Vec<f32>,
    chi_squared: Vec<f32>,
) -> LinearChunkResult {
    let variant_count = beta.len();
    let mut result = LinearChunkResult {
        beta: Vec::with_capacity(variant_count),
        standard_error: Vec::with_capacity(variant_count),
        chi_squared: Vec::with_capacity(variant_count),
        log10_p_value: Vec::with_capacity(variant_count),
        checksum: 0.0,
        #[cfg(any(feature = "burn-wgpu", feature = "burn-cuda", feature = "cuda-kernel"))]
        timing_seconds: BTreeMap::new(),
    };
    for ((beta_value, standard_error_value), chi_squared_value) in
        beta.into_iter().zip(standard_error.into_iter()).zip(chi_squared.into_iter())
    {
        let log10_p_value = chi_squared_to_log10_p_value(chi_squared_value);
        result.checksum += f64::from(beta_value.to_bits());
        result.checksum += f64::from(standard_error_value.to_bits());
        result.checksum += f64::from(chi_squared_value.to_bits());
        result.checksum += f64::from(log10_p_value.to_bits());
        result.beta.push(beta_value);
        result.standard_error.push(standard_error_value);
        result.chi_squared.push(chi_squared_value);
        result.log10_p_value.push(log10_p_value);
    }
    result
}

pub(crate) fn validate_genotype_shape(
    sample_count: usize,
    genotype_value_count: usize,
    variant_count: usize,
) -> Result<(), LinearError> {
    if genotype_value_count != sample_count * variant_count {
        return Err(LinearError::InvalidInput(format!(
            "Genotype matrix shape mismatch: expected {} values, observed {}.",
            sample_count * variant_count,
            genotype_value_count,
        )));
    }
    Ok(())
}

#[allow(clippy::cast_possible_truncation)]
fn chi_squared_to_log10_p_value(chi_squared: f32) -> f32 {
    let safe_chi_squared = f64::from(chi_squared.max(0.0));
    let p_value = erfc(safe_chi_squared.sqrt() / std::f64::consts::SQRT_2);
    if p_value <= 0.0 {
        return f32::INFINITY;
    }
    (-p_value.log10()) as f32
}

fn compute_crossproduct(covariate_values: &[f32], sample_count: usize, covariate_count: usize) -> Vec<f32> {
    let mut crossproduct = vec![0.0_f32; covariate_count * covariate_count];
    for sample_index in 0..sample_count {
        for left_index in 0..covariate_count {
            let left_value = covariate_values[(sample_index * covariate_count) + left_index];
            for right_index in 0..covariate_count {
                crossproduct[(left_index * covariate_count) + right_index] +=
                    left_value * covariate_values[(sample_index * covariate_count) + right_index];
            }
        }
    }
    crossproduct
}

fn transpose_row_major(values: &[f32], row_count: usize, column_count: usize) -> Vec<f32> {
    let mut transposed_values = vec![0.0_f32; values.len()];
    for row_index in 0..row_count {
        for column_index in 0..column_count {
            transposed_values[(column_index * row_count) + row_index] =
                values[(row_index * column_count) + column_index];
        }
    }
    transposed_values
}

fn cholesky_lower(matrix: &[f32], dimension: usize) -> Result<Vec<f32>, LinearError> {
    let mut factor = vec![0.0_f32; dimension * dimension];
    for row_index in 0..dimension {
        for column_index in 0..=row_index {
            let mut sum = matrix[(row_index * dimension) + column_index];
            for inner_index in 0..column_index {
                sum -= factor[(row_index * dimension) + inner_index] * factor[(column_index * dimension) + inner_index];
            }
            if row_index == column_index {
                if sum <= 0.0 {
                    return Err(LinearError::InvalidInput(
                        "Covariate crossproduct is not positive definite.".to_string(),
                    ));
                }
                factor[(row_index * dimension) + column_index] = sum.sqrt();
            } else {
                factor[(row_index * dimension) + column_index] =
                    sum / factor[(column_index * dimension) + column_index];
            }
        }
    }
    Ok(factor)
}

fn solve_lower_triangular_matrix(
    lower_factor: &[f32],
    right_hand_side: &[f32],
    dimension: usize,
    right_column_count: usize,
) -> Result<Vec<f32>, LinearError> {
    if right_hand_side.len() != dimension * right_column_count {
        return Err(LinearError::InvalidInput("Right-hand side matrix has an invalid shape.".to_string()));
    }
    let mut solution = vec![0.0_f32; right_hand_side.len()];
    for column_index in 0..right_column_count {
        for row_index in 0..dimension {
            let mut value = right_hand_side[(row_index * right_column_count) + column_index];
            for inner_index in 0..row_index {
                value -= lower_factor[(row_index * dimension) + inner_index]
                    * solution[(inner_index * right_column_count) + column_index];
            }
            solution[(row_index * right_column_count) + column_index] =
                value / lower_factor[(row_index * dimension) + row_index];
        }
    }
    Ok(solution)
}

fn solve_positive_definite_from_cholesky(
    lower_factor: &[f32],
    right_hand_side: &[f32],
    dimension: usize,
) -> Result<Vec<f32>, LinearError> {
    if right_hand_side.len() != dimension {
        return Err(LinearError::InvalidInput("Right-hand side vector has an invalid shape.".to_string()));
    }
    let mut intermediate = vec![0.0_f32; dimension];
    for row_index in 0..dimension {
        let mut value = right_hand_side[row_index];
        for inner_index in 0..row_index {
            value -= lower_factor[(row_index * dimension) + inner_index] * intermediate[inner_index];
        }
        intermediate[row_index] = value / lower_factor[(row_index * dimension) + row_index];
    }
    let mut solution = vec![0.0_f32; dimension];
    for reverse_row_index in 0..dimension {
        let row_index = dimension - reverse_row_index - 1;
        let mut value = intermediate[row_index];
        for inner_index in (row_index + 1)..dimension {
            value -= lower_factor[(inner_index * dimension) + row_index] * solution[inner_index];
        }
        solution[row_index] = value / lower_factor[(row_index * dimension) + row_index];
    }
    Ok(solution)
}

fn multiply_transpose_matrix_vector(matrix: &[f32], vector: &[f32], row_count: usize, column_count: usize) -> Vec<f32> {
    let mut product = vec![0.0_f32; column_count];
    for row_index in 0..row_count {
        for column_index in 0..column_count {
            product[column_index] += matrix[(row_index * column_count) + column_index] * vector[row_index];
        }
    }
    product
}

fn multiply_matrix_vector(matrix: &[f32], vector: &[f32], row_count: usize, column_count: usize) -> Vec<f32> {
    let mut product = vec![0.0_f32; row_count];
    for row_index in 0..row_count {
        for column_index in 0..column_count {
            product[row_index] += matrix[(row_index * column_count) + column_index] * vector[column_index];
        }
    }
    product
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_linear_chunk_matches_expected_single_covariate_shape() {
        let covariates = vec![1.0, 1.0, 1.0, 1.0];
        let phenotype = vec![1.0, 2.0, 3.0, 4.0];
        let state = prepare_linear_state(&covariates, &phenotype, 1).expect("state should build");
        let chromosome_state =
            prepare_linear_chromosome_state(&state, &[0.0, 0.0, 0.0, 0.0]).expect("chromosome state should build");
        let genotypes = vec![0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0];
        let result = compute_linear_chunk_cpu(&chromosome_state, &genotypes, 2).expect("chunk should compute");

        assert_eq!(result.beta.len(), 2);
        assert_eq!(result.standard_error.len(), 2);
        assert!(result.chi_squared.iter().all(|value| value.is_finite()));
        assert!(result.log10_p_value.iter().all(|value| value.is_finite()));
    }
}
