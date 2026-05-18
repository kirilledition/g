//! Burn-native CubeCL linear regression kernel for quantitative REGENIE step 2.

#![allow(clippy::missing_errors_doc)]

use std::collections::BTreeMap;
use std::time::Instant;

use burn::prelude::Tensor as BurnTensor;
use burn::tensor::{TensorData, TensorPrimitive, Transaction};
use burn_backend::Shape;
use burn_backend::tensor::FloatTensor;
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::cubecl::std::tensor::layout::linear::LinearView;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, IntElement};

use crate::native::linear;

const MAX_CUBECL_SCORE_ROWS: usize = 8;

pub struct LinearCubeclDeviceChromosomeState<Backend: LinearCubeclBackend> {
    stacked_score_tensor: BurnTensor<Backend, 2>,
    adjusted_residual_sum_squares: f32,
    sample_count: usize,
    score_row_count: usize,
    covariate_count: usize,
    degrees_of_freedom: f32,
}

struct LinearCubeclOutputTensors<Backend: burn::tensor::backend::Backend> {
    beta_tensor: BurnTensor<Backend, 1>,
    standard_error_tensor: BurnTensor<Backend, 1>,
    chi_squared_tensor: BurnTensor<Backend, 1>,
}

pub trait LinearCubeclBackend: burn::tensor::backend::Backend {
    fn regenie_linear_chunk(
        stacked_score_tensor: FloatTensor<Self>,
        genotype_tensor: FloatTensor<Self>,
        adjusted_residual_sum_squares: f32,
        degrees_of_freedom: f32,
        sample_count: usize,
        variant_count: usize,
        score_row_count: usize,
        covariate_count: usize,
        cubecl_block_size: u32,
    ) -> LinearCubeclPrimitiveOutput<Self>;
}

pub struct LinearCubeclPrimitiveOutput<Backend: burn::tensor::backend::Backend> {
    beta_tensor: FloatTensor<Backend>,
    standard_error_tensor: FloatTensor<Backend>,
    chi_squared_tensor: FloatTensor<Backend>,
}

impl<Runtime, IntegerElement, BooleanElement> LinearCubeclBackend
    for CubeBackend<Runtime, f32, IntegerElement, BooleanElement>
where
    Runtime: CubeRuntime,
    IntegerElement: IntElement,
    BooleanElement: BoolElement,
{
    fn regenie_linear_chunk(
        stacked_score_tensor: FloatTensor<Self>,
        genotype_tensor: FloatTensor<Self>,
        adjusted_residual_sum_squares: f32,
        degrees_of_freedom: f32,
        sample_count: usize,
        variant_count: usize,
        score_row_count: usize,
        covariate_count: usize,
        cubecl_block_size: u32,
    ) -> LinearCubeclPrimitiveOutput<Self> {
        let stacked_score_tensor = into_contiguous(stacked_score_tensor);
        let genotype_tensor = into_contiguous(genotype_tensor);
        let client = genotype_tensor.client.clone();
        let device = genotype_tensor.device.clone();
        let output_shape = Shape::new([variant_count]);
        let beta_tensor =
            empty_device_dtype::<Runtime>(client.clone(), device.clone(), output_shape.clone(), genotype_tensor.dtype);
        let standard_error_tensor =
            empty_device_dtype::<Runtime>(client.clone(), device.clone(), output_shape.clone(), genotype_tensor.dtype);
        let chi_squared_tensor =
            empty_device_dtype::<Runtime>(client.clone(), device, output_shape, genotype_tensor.dtype);
        let cubecl_block_size = cubecl_block_size.max(1);
        let cube_count = cubecl::CubeCount::Static(variant_count.div_ceil(cubecl_block_size as usize) as u32, 1, 1);
        let cube_dim = cubecl::CubeDim::new_1d(cubecl_block_size);
        let address_type = [
            stacked_score_tensor.required_address_type(),
            genotype_tensor.required_address_type(),
            beta_tensor.required_address_type(),
            standard_error_tensor.required_address_type(),
            chi_squared_tensor.required_address_type(),
        ]
        .into_iter()
        .max()
        .unwrap_or_default();
        unsafe {
            regenie_linear_onepass_kernel::launch_unchecked::<Runtime>(
                &client,
                cube_count,
                cube_dim,
                address_type,
                stacked_score_tensor.into_linear_view(),
                genotype_tensor.into_linear_view(),
                beta_tensor.clone().into_linear_view(),
                standard_error_tensor.clone().into_linear_view(),
                chi_squared_tensor.clone().into_linear_view(),
                adjusted_residual_sum_squares,
                degrees_of_freedom,
                sample_count,
                variant_count,
                score_row_count,
                covariate_count,
            );
        }
        LinearCubeclPrimitiveOutput { beta_tensor, standard_error_tensor, chi_squared_tensor }
    }
}

pub fn prepare_linear_cubecl_device_chromosome_state<Backend: LinearCubeclBackend>(
    chromosome_state: &linear::LinearChromosomeState,
    device: &Backend::Device,
) -> LinearCubeclDeviceChromosomeState<Backend> {
    let stacked_score_tensor = BurnTensor::<Backend, 2>::from_data(
        TensorData::new(
            chromosome_state.stacked_score_matrix.clone(),
            [chromosome_state.score_row_count, chromosome_state.sample_count],
        ),
        device,
    );
    LinearCubeclDeviceChromosomeState {
        stacked_score_tensor,
        adjusted_residual_sum_squares: chromosome_state.adjusted_residual_sum_squares,
        sample_count: chromosome_state.sample_count,
        score_row_count: chromosome_state.score_row_count,
        covariate_count: chromosome_state.covariate_count,
        degrees_of_freedom: chromosome_state.degrees_of_freedom,
    }
}

pub fn compute_linear_chunk_cubecl<Backend: LinearCubeclBackend>(
    chromosome_state: &LinearCubeclDeviceChromosomeState<Backend>,
    genotype_values: Vec<f32>,
    variant_count: usize,
    device: &Backend::Device,
    cubecl_block_size: u32,
) -> Result<linear::LinearChunkResult, linear::LinearError> {
    linear::validate_genotype_shape(chromosome_state.sample_count, genotype_values.len(), variant_count)?;
    if chromosome_state.score_row_count > MAX_CUBECL_SCORE_ROWS {
        return Err(linear::LinearError::InvalidInput(format!(
            "CubeCL linear kernel supports at most {MAX_CUBECL_SCORE_ROWS} score rows, observed {}.",
            chromosome_state.score_row_count,
        )));
    }
    let mut timing_seconds = BTreeMap::new();
    let tensor_setup_start_time = Instant::now();
    let genotype_tensor = BurnTensor::<Backend, 2>::from_data(
        TensorData::new(genotype_values, [chromosome_state.sample_count, variant_count]),
        device,
    );
    timing_seconds.insert("cubecl_tensor_setup".to_string(), tensor_setup_start_time.elapsed().as_secs_f64());

    let graph_start_time = Instant::now();
    let output_tensors =
        compute_linear_chunk_tensors(chromosome_state, genotype_tensor, variant_count, cubecl_block_size);
    <Backend as burn::tensor::backend::Backend>::sync(device)
        .map_err(|error| linear::LinearError::Backend(error.to_string()))?;
    timing_seconds.insert("cubecl_kernel_launch".to_string(), graph_start_time.elapsed().as_secs_f64());

    let materialization_start_time = Instant::now();
    let mut tensor_data_values = Transaction::<Backend>::default()
        .register(output_tensors.beta_tensor)
        .register(output_tensors.standard_error_tensor)
        .register(output_tensors.chi_squared_tensor)
        .try_execute()
        .map_err(|error| linear::LinearError::Backend(error.to_string()))?;
    if tensor_data_values.len() != 3 {
        return Err(linear::LinearError::Backend(format!(
            "CubeCL output transaction returned {} tensors instead of 3.",
            tensor_data_values.len(),
        )));
    }
    let chi_squared = tensor_data_values
        .pop()
        .ok_or_else(|| linear::LinearError::Backend("Missing CubeCL chi-square output tensor.".to_string()))
        .and_then(|tensor_data| linear::materialize_tensor_data(&tensor_data, variant_count, "chi-square"))?;
    let standard_error = tensor_data_values
        .pop()
        .ok_or_else(|| linear::LinearError::Backend("Missing CubeCL standard-error output tensor.".to_string()))
        .and_then(|tensor_data| linear::materialize_tensor_data(&tensor_data, variant_count, "standard-error"))?;
    let beta = tensor_data_values
        .pop()
        .ok_or_else(|| linear::LinearError::Backend("Missing CubeCL beta output tensor.".to_string()))
        .and_then(|tensor_data| linear::materialize_tensor_data(&tensor_data, variant_count, "beta"))?;
    timing_seconds
        .insert("cubecl_output_materialization".to_string(), materialization_start_time.elapsed().as_secs_f64());

    let cpu_log10_start_time = Instant::now();
    let mut result = linear::build_linear_chunk_result(beta, standard_error, chi_squared);
    timing_seconds.insert("cubecl_cpu_log10p_checksum".to_string(), cpu_log10_start_time.elapsed().as_secs_f64());
    result.timing_seconds = timing_seconds;
    Ok(result)
}

fn compute_linear_chunk_tensors<Backend: LinearCubeclBackend>(
    chromosome_state: &LinearCubeclDeviceChromosomeState<Backend>,
    genotype_tensor: BurnTensor<Backend, 2>,
    variant_count: usize,
    cubecl_block_size: u32,
) -> LinearCubeclOutputTensors<Backend> {
    let primitive_output = Backend::regenie_linear_chunk(
        chromosome_state.stacked_score_tensor.clone().into_primitive().tensor(),
        genotype_tensor.into_primitive().tensor(),
        chromosome_state.adjusted_residual_sum_squares,
        chromosome_state.degrees_of_freedom,
        chromosome_state.sample_count,
        variant_count,
        chromosome_state.score_row_count,
        chromosome_state.covariate_count,
        cubecl_block_size,
    );
    LinearCubeclOutputTensors {
        beta_tensor: BurnTensor::<Backend, 1>::from_primitive(TensorPrimitive::Float(primitive_output.beta_tensor)),
        standard_error_tensor: BurnTensor::<Backend, 1>::from_primitive(TensorPrimitive::Float(
            primitive_output.standard_error_tensor,
        )),
        chi_squared_tensor: BurnTensor::<Backend, 1>::from_primitive(TensorPrimitive::Float(
            primitive_output.chi_squared_tensor,
        )),
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn regenie_linear_onepass_kernel(
    stacked_score_tensor: &LinearView<f32>,
    genotype_tensor: &LinearView<f32>,
    beta_tensor: &mut LinearView<f32, ReadWrite>,
    standard_error_tensor: &mut LinearView<f32, ReadWrite>,
    chi_squared_tensor: &mut LinearView<f32, ReadWrite>,
    adjusted_residual_sum_squares: f32,
    degrees_of_freedom: f32,
    sample_count: usize,
    variant_count: usize,
    score_row_count: usize,
    covariate_count: usize,
) {
    let variant_index = ABSOLUTE_POS;
    if variant_index >= variant_count {
        terminate!();
    }

    let zero_value = f32::new(0.0);
    let one_value = f32::new(1.0);
    let nan_value = f32::from_bits(0x7fc00000u32);
    let mut genotype_sum_squares = zero_value;
    let mut score_0 = zero_value;
    let mut score_1 = zero_value;
    let mut score_2 = zero_value;
    let mut score_3 = zero_value;
    let mut score_4 = zero_value;
    let mut score_5 = zero_value;
    let mut score_6 = zero_value;
    let mut score_7 = zero_value;
    let mut sample_index = 0usize;
    while sample_index < sample_count {
        let genotype_value = genotype_tensor[(sample_index * variant_count) + variant_index];
        genotype_sum_squares += genotype_value * genotype_value;
        if score_row_count > 0usize {
            score_0 += stacked_score_tensor[sample_index] * genotype_value;
        }
        if score_row_count > 1usize {
            score_1 += stacked_score_tensor[sample_count + sample_index] * genotype_value;
        }
        if score_row_count > 2usize {
            score_2 += stacked_score_tensor[(2usize * sample_count) + sample_index] * genotype_value;
        }
        if score_row_count > 3usize {
            score_3 += stacked_score_tensor[(3usize * sample_count) + sample_index] * genotype_value;
        }
        if score_row_count > 4usize {
            score_4 += stacked_score_tensor[(4usize * sample_count) + sample_index] * genotype_value;
        }
        if score_row_count > 5usize {
            score_5 += stacked_score_tensor[(5usize * sample_count) + sample_index] * genotype_value;
        }
        if score_row_count > 6usize {
            score_6 += stacked_score_tensor[(6usize * sample_count) + sample_index] * genotype_value;
        }
        if score_row_count > 7usize {
            score_7 += stacked_score_tensor[(7usize * sample_count) + sample_index] * genotype_value;
        }
        sample_index += 1;
    }

    let mut projection_sum_squares = zero_value;
    if covariate_count > 0usize {
        projection_sum_squares += score_0 * score_0;
    }
    if covariate_count > 1usize {
        projection_sum_squares += score_1 * score_1;
    }
    if covariate_count > 2usize {
        projection_sum_squares += score_2 * score_2;
    }
    if covariate_count > 3usize {
        projection_sum_squares += score_3 * score_3;
    }
    if covariate_count > 4usize {
        projection_sum_squares += score_4 * score_4;
    }
    if covariate_count > 5usize {
        projection_sum_squares += score_5 * score_5;
    }
    if covariate_count > 6usize {
        projection_sum_squares += score_6 * score_6;
    }
    if covariate_count > 7usize {
        projection_sum_squares += score_7 * score_7;
    }
    let phenotype_score_row_index = score_row_count - 1;
    let mut covariance_with_phenotype = score_0;
    if phenotype_score_row_index == 1usize {
        covariance_with_phenotype = score_1;
    }
    if phenotype_score_row_index == 2usize {
        covariance_with_phenotype = score_2;
    }
    if phenotype_score_row_index == 3usize {
        covariance_with_phenotype = score_3;
    }
    if phenotype_score_row_index == 4usize {
        covariance_with_phenotype = score_4;
    }
    if phenotype_score_row_index == 5usize {
        covariance_with_phenotype = score_5;
    }
    if phenotype_score_row_index == 6usize {
        covariance_with_phenotype = score_6;
    }
    if phenotype_score_row_index == 7usize {
        covariance_with_phenotype = score_7;
    }

    let genotype_residual_sum_squares_unclamped = genotype_sum_squares - projection_sum_squares;
    let genotype_residual_sum_squares = if genotype_residual_sum_squares_unclamped < zero_value {
        zero_value
    } else {
        genotype_residual_sum_squares_unclamped
    };
    let covariance_squared = covariance_with_phenotype * covariance_with_phenotype;
    let positive_genotype_residual = genotype_residual_sum_squares > zero_value;
    let genotype_residual_inverse =
        if positive_genotype_residual { one_value / genotype_residual_sum_squares } else { zero_value };
    let residual_sum_squares_after_unclamped =
        adjusted_residual_sum_squares - covariance_squared * genotype_residual_inverse;
    let residual_sum_squares_after = if residual_sum_squares_after_unclamped < zero_value {
        zero_value
    } else {
        residual_sum_squares_after_unclamped
    };
    let positive_residual_sum_squares = residual_sum_squares_after > zero_value;
    let valid_standard_error = positive_genotype_residual && positive_residual_sum_squares;

    beta_tensor[variant_index] =
        if positive_genotype_residual { covariance_with_phenotype * genotype_residual_inverse } else { nan_value };
    standard_error_tensor[variant_index] = if valid_standard_error {
        (residual_sum_squares_after * genotype_residual_inverse / degrees_of_freedom).sqrt()
    } else {
        nan_value
    };
    chi_squared_tensor[variant_index] = if valid_standard_error {
        covariance_squared * genotype_residual_inverse * degrees_of_freedom / residual_sum_squares_after
    } else {
        zero_value
    };
}
