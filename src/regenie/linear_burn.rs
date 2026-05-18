//! Burn-backed quantitative REGENIE step 2 linear kernel.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]

use std::time::Instant;

use burn::backend::{Wgpu, wgpu::WgpuDevice};
use burn::tensor::{Tensor, TensorData, Transaction};
use statrs::function::erf::erfc;

type BurnBackend = Wgpu;

#[derive(Debug, Clone, Default)]
pub struct BurnLinearTimingProfile {
    pub tensor_upload_ns: u64,
    pub compute_ns: u64,
    pub materialization_ns: u64,
}

impl BurnLinearTimingProfile {
    pub fn add(&mut self, other: &Self) {
        self.tensor_upload_ns = self.tensor_upload_ns.saturating_add(other.tensor_upload_ns);
        self.compute_ns = self.compute_ns.saturating_add(other.compute_ns);
        self.materialization_ns = self.materialization_ns.saturating_add(other.materialization_ns);
    }
}

#[derive(Debug, Clone)]
pub struct LinearBurnState {
    whitened_covariate_transpose: Vec<f32>,
    phenotype_residual: Vec<f32>,
    sample_count: usize,
    covariate_count: usize,
    degrees_of_freedom: f32,
}

#[derive(Debug, Clone)]
pub struct LinearBurnChromosomeState {
    stacked_score_matrix: Vec<f32>,
    adjusted_residual_sum_squares: f32,
    sample_count: usize,
    score_row_count: usize,
    covariate_count: usize,
    degrees_of_freedom: f32,
}

#[derive(Debug, Clone)]
pub struct LinearBurnChunkResult {
    pub beta: Vec<f32>,
    pub standard_error: Vec<f32>,
    pub chi_squared: Vec<f32>,
    pub log10_p_value: Vec<f32>,
}

pub struct LinearBurnWgpuSession {
    device: WgpuDevice,
    chromosome_state: Option<LinearBurnDeviceChromosomeState>,
    timing_profile: BurnLinearTimingProfile,
}

struct LinearBurnDeviceChromosomeState {
    stacked_score_tensor: Tensor<BurnBackend, 2>,
    adjusted_residual_sum_squares: f32,
    sample_count: usize,
    covariate_count: usize,
    degrees_of_freedom: f32,
}

impl LinearBurnWgpuSession {
    #[must_use]
    pub fn new() -> Self {
        Self {
            device: WgpuDevice::default(),
            chromosome_state: None,
            timing_profile: BurnLinearTimingProfile::default(),
        }
    }

    pub fn set_chromosome_state(&mut self, chromosome_state: LinearBurnChromosomeState) {
        let upload_start = Instant::now();
        let stacked_score_tensor = Tensor::<BurnBackend, 2>::from_data(
            TensorData::new(
                chromosome_state.stacked_score_matrix,
                [chromosome_state.score_row_count, chromosome_state.sample_count],
            ),
            &self.device,
        );
        self.timing_profile.tensor_upload_ns =
            self.timing_profile.tensor_upload_ns.saturating_add(elapsed_ns(upload_start));
        self.chromosome_state = Some(LinearBurnDeviceChromosomeState {
            stacked_score_tensor,
            adjusted_residual_sum_squares: chromosome_state.adjusted_residual_sum_squares,
            sample_count: chromosome_state.sample_count,
            covariate_count: chromosome_state.covariate_count,
            degrees_of_freedom: chromosome_state.degrees_of_freedom,
        });
    }

    pub fn compute_chunk(
        &mut self,
        genotype_values: Vec<f32>,
        variant_count: usize,
    ) -> Result<LinearBurnChunkResult, String> {
        let chromosome_state = self
            .chromosome_state
            .as_ref()
            .ok_or_else(|| "Burn WGPU chromosome state was not initialized.".to_string())?;
        validate_matrix_shape(genotype_values.len(), chromosome_state.sample_count, variant_count, "genotype")?;

        let upload_start = Instant::now();
        let genotype_tensor = Tensor::<BurnBackend, 2>::from_data(
            TensorData::new(genotype_values, [chromosome_state.sample_count, variant_count]),
            &self.device,
        );
        self.timing_profile.tensor_upload_ns =
            self.timing_profile.tensor_upload_ns.saturating_add(elapsed_ns(upload_start));

        let compute_start = Instant::now();
        let (beta_tensor, standard_error_tensor, chi_squared_tensor) =
            compute_linear_chunk_tensors(chromosome_state, genotype_tensor, variant_count, &self.device);
        self.timing_profile.compute_ns = self.timing_profile.compute_ns.saturating_add(elapsed_ns(compute_start));

        let materialization_start = Instant::now();
        let mut tensor_data_values = Transaction::<BurnBackend>::default()
            .register(beta_tensor)
            .register(standard_error_tensor)
            .register(chi_squared_tensor)
            .try_execute()
            .map_err(|error| format!("Failed to materialize Burn linear output tensors: {error}"))?;
        if tensor_data_values.len() != 3 {
            return Err(format!(
                "Burn linear output transaction returned {} tensors instead of 3.",
                tensor_data_values.len(),
            ));
        }
        let chi_squared = tensor_data_values
            .pop()
            .ok_or_else(|| "Missing Burn chi-square output tensor.".to_string())
            .and_then(|tensor_data| materialize_tensor_data(&tensor_data, variant_count, "chi-square"))?;
        let standard_error = tensor_data_values
            .pop()
            .ok_or_else(|| "Missing Burn standard-error output tensor.".to_string())
            .and_then(|tensor_data| materialize_tensor_data(&tensor_data, variant_count, "standard-error"))?;
        let beta = tensor_data_values
            .pop()
            .ok_or_else(|| "Missing Burn beta output tensor.".to_string())
            .and_then(|tensor_data| materialize_tensor_data(&tensor_data, variant_count, "beta"))?;
        self.timing_profile.materialization_ns =
            self.timing_profile.materialization_ns.saturating_add(elapsed_ns(materialization_start));

        let log10_p_value = chi_squared.iter().map(|value| chi_squared_to_log10_p_value(*value)).collect();
        Ok(LinearBurnChunkResult { beta, standard_error, chi_squared, log10_p_value })
    }

    #[must_use]
    pub fn timing_profile(&self) -> BurnLinearTimingProfile {
        self.timing_profile.clone()
    }
}

impl Default for LinearBurnWgpuSession {
    fn default() -> Self {
        Self::new()
    }
}

pub fn prepare_linear_burn_state(
    covariate_matrix: &[f32],
    phenotype_vector: &[f32],
    sample_count: usize,
    covariate_count: usize,
) -> Result<LinearBurnState, String> {
    validate_matrix_shape(covariate_matrix.len(), sample_count, covariate_count, "covariate")?;
    if phenotype_vector.len() != sample_count {
        return Err(format!(
            "Phenotype vector length mismatch: expected {sample_count}, observed {}.",
            phenotype_vector.len(),
        ));
    }
    if sample_count <= covariate_count + 1 {
        return Err(format!(
            "Sample count must exceed covariate count plus one for linear step 2; samples={sample_count}, covariates={covariate_count}.",
        ));
    }

    let covariate_transpose = transpose_row_major(covariate_matrix, sample_count, covariate_count);
    let covariate_crossproduct =
        matmul_row_major(&covariate_transpose, covariate_matrix, covariate_count, sample_count, covariate_count);
    let cholesky_factor = cholesky_lower(&covariate_crossproduct, covariate_count)?;
    let whitened_covariate_transpose =
        solve_lower_triangular_matrix(&cholesky_factor, &covariate_transpose, covariate_count, sample_count)?;
    let covariate_phenotype_crossproduct =
        matvec_row_major(&covariate_transpose, phenotype_vector, covariate_count, sample_count);
    let phenotype_projection =
        solve_positive_definite_from_cholesky(&cholesky_factor, &covariate_phenotype_crossproduct, covariate_count)?;
    let fitted_phenotype = matvec_row_major(covariate_matrix, &phenotype_projection, sample_count, covariate_count);
    let phenotype_residual = phenotype_vector
        .iter()
        .zip(fitted_phenotype.iter())
        .map(|(phenotype_value, fitted_value)| phenotype_value - fitted_value)
        .collect();

    Ok(LinearBurnState {
        whitened_covariate_transpose,
        phenotype_residual,
        sample_count,
        covariate_count,
        degrees_of_freedom: (sample_count - covariate_count - 1) as f32,
    })
}

pub fn prepare_linear_burn_chromosome_state(
    state: &LinearBurnState,
    loco_predictions: &[f32],
) -> Result<LinearBurnChromosomeState, String> {
    if loco_predictions.len() != state.sample_count {
        return Err(format!(
            "LOCO prediction length mismatch: expected {}, observed {}.",
            state.sample_count,
            loco_predictions.len(),
        ));
    }
    let adjusted_residual: Vec<f32> = state
        .phenotype_residual
        .iter()
        .zip(loco_predictions.iter())
        .map(|(phenotype_residual, loco_prediction)| phenotype_residual - loco_prediction)
        .collect();
    let adjusted_residual_sum_squares = adjusted_residual.iter().map(|value| value * value).sum();
    let mut stacked_score_matrix = Vec::with_capacity((state.covariate_count + 1).saturating_mul(state.sample_count));
    stacked_score_matrix.extend_from_slice(&state.whitened_covariate_transpose);
    stacked_score_matrix.extend_from_slice(&adjusted_residual);

    Ok(LinearBurnChromosomeState {
        stacked_score_matrix,
        adjusted_residual_sum_squares,
        sample_count: state.sample_count,
        score_row_count: state.covariate_count + 1,
        covariate_count: state.covariate_count,
        degrees_of_freedom: state.degrees_of_freedom,
    })
}

pub fn compute_linear_burn_wgpu_chunk(
    chromosome_state: &LinearBurnChromosomeState,
    genotype_values: Vec<f32>,
    variant_count: usize,
) -> Result<(LinearBurnChunkResult, BurnLinearTimingProfile), String> {
    let mut session = LinearBurnWgpuSession::new();
    session.set_chromosome_state(chromosome_state.clone());
    let result = session.compute_chunk(genotype_values, variant_count)?;
    Ok((result, session.timing_profile()))
}

fn compute_linear_chunk_tensors(
    chromosome_state: &LinearBurnDeviceChromosomeState,
    genotype_tensor: Tensor<BurnBackend, 2>,
    variant_count: usize,
    device: &WgpuDevice,
) -> (Tensor<BurnBackend, 1>, Tensor<BurnBackend, 1>, Tensor<BurnBackend, 1>) {
    let score_tensor = chromosome_state.stacked_score_tensor.clone().matmul(genotype_tensor.clone());
    let genotype_sum_squares = (genotype_tensor.clone() * genotype_tensor).sum_dim(0).reshape([variant_count]);
    let projection_sum_squares = if chromosome_state.covariate_count == 0 {
        Tensor::<BurnBackend, 1>::zeros([variant_count], device)
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
    let genotype_residual_inverse = Tensor::<BurnBackend, 1>::zeros([variant_count], device)
        .mask_where(positive_genotype_mask.clone(), safe_genotype_residual_sum_squares.recip());
    let residual_sum_squares_after = (chromosome_state.adjusted_residual_sum_squares
        - (covariance_squared.clone() * genotype_residual_inverse.clone()))
    .clamp_min(0.0);
    let positive_residual_mask = residual_sum_squares_after.clone().greater_elem(0.0);
    let valid_standard_error_mask = positive_genotype_mask.clone().bool_and(positive_residual_mask);

    let beta_values = covariance_with_phenotype * genotype_residual_inverse.clone();
    let beta = Tensor::<BurnBackend, 1>::full([variant_count], f32::NAN, device)
        .mask_where(positive_genotype_mask, beta_values);

    let standard_error_values = (residual_sum_squares_after.clone() * genotype_residual_inverse.clone()
        / chromosome_state.degrees_of_freedom)
        .sqrt();
    let standard_error = Tensor::<BurnBackend, 1>::full([variant_count], f32::NAN, device)
        .mask_where(valid_standard_error_mask.clone(), standard_error_values);

    let safe_residual_sum_squares_after =
        residual_sum_squares_after.mask_fill(valid_standard_error_mask.clone().bool_not(), 1.0);
    let chi_squared_values = covariance_squared * genotype_residual_inverse * chromosome_state.degrees_of_freedom
        / safe_residual_sum_squares_after;
    let chi_squared = Tensor::<BurnBackend, 1>::zeros([variant_count], device)
        .mask_where(valid_standard_error_mask, chi_squared_values);

    (beta, standard_error, chi_squared)
}

fn materialize_tensor_data(
    tensor_data: &TensorData,
    expected_value_count: usize,
    tensor_name: &str,
) -> Result<Vec<f32>, String> {
    let values = tensor_data
        .to_vec::<f32>()
        .map_err(|error| format!("Failed to convert Burn {tensor_name} tensor data to f32 values: {error}"))?;
    if values.len() != expected_value_count {
        return Err(format!(
            "Burn {tensor_name} tensor length mismatch: expected {expected_value_count}, observed {}.",
            values.len(),
        ));
    }
    Ok(values)
}

#[allow(clippy::cast_possible_truncation)]
fn chi_squared_to_log10_p_value(chi_squared: f32) -> f32 {
    let safe_chi_squared = f64::from(chi_squared.max(0.0));
    if safe_chi_squared == 0.0 {
        return 0.0;
    }
    let z_score = safe_chi_squared.sqrt();
    let p_value = erfc(z_score / std::f64::consts::SQRT_2);
    let log_p_value = if p_value > 0.0 {
        p_value.ln()
    } else {
        0.5 * (2.0 / std::f64::consts::PI).ln() - z_score.ln() - (0.5 * safe_chi_squared)
    };
    (-log_p_value / std::f64::consts::LN_10) as f32
}

fn validate_matrix_shape(
    observed_value_count: usize,
    row_count: usize,
    column_count: usize,
    matrix_name: &str,
) -> Result<(), String> {
    let expected_value_count = row_count
        .checked_mul(column_count)
        .ok_or_else(|| format!("Integer overflow while validating {matrix_name} matrix shape."))?;
    if observed_value_count != expected_value_count {
        return Err(format!(
            "{matrix_name} matrix value count mismatch: expected {expected_value_count}, observed {observed_value_count}.",
        ));
    }
    Ok(())
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

fn matmul_row_major(
    left_values: &[f32],
    right_values: &[f32],
    left_row_count: usize,
    shared_dimension_count: usize,
    right_column_count: usize,
) -> Vec<f32> {
    let mut output_values = vec![0.0_f32; left_row_count * right_column_count];
    for left_row_index in 0..left_row_count {
        for shared_index in 0..shared_dimension_count {
            let left_value = left_values[(left_row_index * shared_dimension_count) + shared_index];
            for right_column_index in 0..right_column_count {
                output_values[(left_row_index * right_column_count) + right_column_index] +=
                    left_value * right_values[(shared_index * right_column_count) + right_column_index];
            }
        }
    }
    output_values
}

fn matvec_row_major(values: &[f32], vector: &[f32], row_count: usize, column_count: usize) -> Vec<f32> {
    let mut output_values = vec![0.0_f32; row_count];
    for row_index in 0..row_count {
        let mut total = 0.0_f32;
        for column_index in 0..column_count {
            total += values[(row_index * column_count) + column_index] * vector[column_index];
        }
        output_values[row_index] = total;
    }
    output_values
}

fn cholesky_lower(values: &[f32], dimension_count: usize) -> Result<Vec<f32>, String> {
    let mut factor_values = vec![0.0_f32; dimension_count * dimension_count];
    for row_index in 0..dimension_count {
        for column_index in 0..=row_index {
            let mut sum = values[(row_index * dimension_count) + column_index];
            for inner_index in 0..column_index {
                sum -= factor_values[(row_index * dimension_count) + inner_index]
                    * factor_values[(column_index * dimension_count) + inner_index];
            }
            if row_index == column_index {
                if sum <= 0.0 {
                    return Err("Covariate crossproduct is not positive definite.".to_string());
                }
                factor_values[(row_index * dimension_count) + column_index] = sum.sqrt();
            } else {
                factor_values[(row_index * dimension_count) + column_index] =
                    sum / factor_values[(column_index * dimension_count) + column_index];
            }
        }
    }
    Ok(factor_values)
}

fn solve_lower_triangular_matrix(
    lower_triangular_values: &[f32],
    right_hand_side: &[f32],
    dimension_count: usize,
    right_column_count: usize,
) -> Result<Vec<f32>, String> {
    let mut output_values = vec![0.0_f32; dimension_count * right_column_count];
    for column_index in 0..right_column_count {
        for row_index in 0..dimension_count {
            let mut value = right_hand_side[(row_index * right_column_count) + column_index];
            for inner_index in 0..row_index {
                value -= lower_triangular_values[(row_index * dimension_count) + inner_index]
                    * output_values[(inner_index * right_column_count) + column_index];
            }
            let diagonal_value = lower_triangular_values[(row_index * dimension_count) + row_index];
            if diagonal_value == 0.0 {
                return Err("Lower-triangular solve encountered a zero diagonal.".to_string());
            }
            output_values[(row_index * right_column_count) + column_index] = value / diagonal_value;
        }
    }
    Ok(output_values)
}

fn solve_positive_definite_from_cholesky(
    cholesky_factor: &[f32],
    right_hand_side: &[f32],
    dimension_count: usize,
) -> Result<Vec<f32>, String> {
    let forward_solution = solve_lower_triangular_vector(cholesky_factor, right_hand_side, dimension_count)?;
    solve_upper_triangular_vector_from_lower(cholesky_factor, &forward_solution, dimension_count)
}

fn solve_lower_triangular_vector(
    lower_triangular_values: &[f32],
    right_hand_side: &[f32],
    dimension_count: usize,
) -> Result<Vec<f32>, String> {
    let mut output_values = vec![0.0_f32; dimension_count];
    for row_index in 0..dimension_count {
        let mut value = right_hand_side[row_index];
        for inner_index in 0..row_index {
            value -= lower_triangular_values[(row_index * dimension_count) + inner_index] * output_values[inner_index];
        }
        let diagonal_value = lower_triangular_values[(row_index * dimension_count) + row_index];
        if diagonal_value == 0.0 {
            return Err("Lower-triangular solve encountered a zero diagonal.".to_string());
        }
        output_values[row_index] = value / diagonal_value;
    }
    Ok(output_values)
}

fn solve_upper_triangular_vector_from_lower(
    lower_triangular_values: &[f32],
    right_hand_side: &[f32],
    dimension_count: usize,
) -> Result<Vec<f32>, String> {
    let mut output_values = vec![0.0_f32; dimension_count];
    for row_index in (0..dimension_count).rev() {
        let mut value = right_hand_side[row_index];
        for inner_index in (row_index + 1)..dimension_count {
            value -= lower_triangular_values[(inner_index * dimension_count) + row_index] * output_values[inner_index];
        }
        let diagonal_value = lower_triangular_values[(row_index * dimension_count) + row_index];
        if diagonal_value == 0.0 {
            return Err("Upper-triangular solve encountered a zero diagonal.".to_string());
        }
        output_values[row_index] = value / diagonal_value;
    }
    Ok(output_values)
}

fn elapsed_ns(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{compute_linear_burn_wgpu_chunk, prepare_linear_burn_chromosome_state, prepare_linear_burn_state};

    #[test]
    fn burn_linear_chunk_matches_reference_values() {
        let sample_count = 8;
        let covariate_count = 2;
        let variant_count = 3;
        let covariate_matrix =
            vec![1.0, -1.0, 1.0, -0.5, 1.0, -0.25, 1.0, 0.0, 1.0, 0.25, 1.0, 0.5, 1.0, 0.75, 1.0, 1.0];
        let phenotype_vector = vec![0.1, 0.3, 0.4, 0.9, 1.3, 1.1, 1.8, 2.0];
        let loco_predictions = vec![0.02, -0.01, 0.0, 0.03, -0.02, 0.01, 0.0, 0.02];
        let genotype_values = vec![
            0.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 2.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 0.0, 2.0,
            1.0, 2.0, 0.0,
        ];

        let state = prepare_linear_burn_state(&covariate_matrix, &phenotype_vector, sample_count, covariate_count)
            .expect("state should prepare");
        let chromosome_state =
            prepare_linear_burn_chromosome_state(&state, &loco_predictions).expect("chromosome state should prepare");
        let (result, _timing_profile) =
            compute_linear_burn_wgpu_chunk(&chromosome_state, genotype_values, variant_count)
                .expect("Burn chunk should compute");

        assert_eq!(result.beta.len(), variant_count);
        assert_eq!(result.standard_error.len(), variant_count);
        assert!(result.chi_squared.iter().all(|value| value.is_finite() && *value >= 0.0));
        assert!(result.log10_p_value.iter().all(|value| value.is_finite() && *value >= 0.0));
    }
}
