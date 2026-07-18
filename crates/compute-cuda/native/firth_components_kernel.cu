extern "C" __global__ void compute_firth_components(
    const double* phenotype,
    const double* genotype,
    const double* offset,
    const bool* active_sample_mask,
    const double* non_active_deviance,
    const double* beta,
    const double* minimum_variance,
    unsigned long long lane_count,
    unsigned long long sample_count,
    double* genotype_information,
    double* score_adjustment,
    double* penalized_deviance,
    double* score,
    bool* valid) {
  constexpr unsigned int kernel_block_size = 256;
  constexpr unsigned int warp_size = 32;
  constexpr unsigned int warp_count = kernel_block_size / warp_size;
  constexpr unsigned int full_warp_mask = 0xffffffffu;
  const unsigned long long lane_index = blockIdx.x;
  const unsigned int thread_index = threadIdx.x;
  const unsigned int warp_index = thread_index / warp_size;
  const unsigned int warp_lane_index = thread_index % warp_size;
  if (lane_index >= lane_count) {
    return;
  }

  const unsigned long long row_offset = lane_index * sample_count;
  const double lane_beta = beta[lane_index];
  double information_sum = 0.0;
  double adjustment_numerator_sum = 0.0;
  double score_sum = 0.0;
  double deviance_sum = 0.0;
  for (unsigned long long sample_index = thread_index;
       sample_index < sample_count;
       sample_index += kernel_block_size) {
    const unsigned long long element_index = row_offset + sample_index;
    if (!active_sample_mask[element_index]) {
      continue;
    }
    const double genotype_value = genotype[element_index];
    const double phenotype_value = phenotype[element_index];
    const double linear_predictor =
        offset[element_index] + genotype_value * lane_beta;
    double probability;
    if (linear_predictor > 30.0) {
      probability = 0.9999999999999978;
    } else if (linear_predictor < -30.0) {
      probability = 2.2204460492503083e-15;
    } else {
      probability = 1.0 / (1.0 + exp(-linear_predictor));
    }
    const double weight = probability * (1.0 - probability);
    const double information_diagonal =
        genotype_value * genotype_value * weight;
    information_sum += information_diagonal;
    adjustment_numerator_sum +=
        genotype_value * information_diagonal * (0.5 - probability);
    score_sum += genotype_value * (phenotype_value - probability);
    deviance_sum += phenotype_value > 0.5
        ? -2.0 * log(probability)
        : -2.0 * log1p(-probability);
  }

  for (unsigned int offset_width = warp_size / 2;
       offset_width != 0;
       offset_width /= 2) {
    information_sum += __shfl_down_sync(
        full_warp_mask, information_sum, offset_width);
    adjustment_numerator_sum += __shfl_down_sync(
        full_warp_mask, adjustment_numerator_sum, offset_width);
    score_sum += __shfl_down_sync(
        full_warp_mask, score_sum, offset_width);
    deviance_sum += __shfl_down_sync(
        full_warp_mask, deviance_sum, offset_width);
  }

  __shared__ double warp_information[warp_count];
  __shared__ double warp_adjustment_numerator[warp_count];
  __shared__ double warp_score[warp_count];
  __shared__ double warp_deviance[warp_count];
  if (warp_lane_index == 0) {
    warp_information[warp_index] = information_sum;
    warp_adjustment_numerator[warp_index] = adjustment_numerator_sum;
    warp_score[warp_index] = score_sum;
    warp_deviance[warp_index] = deviance_sum;
  }
  __syncthreads();

  if (warp_index != 0) {
    return;
  }
  information_sum =
      warp_lane_index < warp_count ? warp_information[warp_lane_index] : 0.0;
  adjustment_numerator_sum = warp_lane_index < warp_count
      ? warp_adjustment_numerator[warp_lane_index]
      : 0.0;
  score_sum = warp_lane_index < warp_count ? warp_score[warp_lane_index] : 0.0;
  deviance_sum =
      warp_lane_index < warp_count ? warp_deviance[warp_lane_index] : 0.0;
  for (unsigned int offset_width = warp_size / 2;
       offset_width != 0;
       offset_width /= 2) {
    information_sum += __shfl_down_sync(
        full_warp_mask, information_sum, offset_width);
    adjustment_numerator_sum += __shfl_down_sync(
        full_warp_mask, adjustment_numerator_sum, offset_width);
    score_sum += __shfl_down_sync(
        full_warp_mask, score_sum, offset_width);
    deviance_sum += __shfl_down_sync(
        full_warp_mask, deviance_sum, offset_width);
  }
  if (warp_lane_index != 0) {
    return;
  }

  const double lane_information = information_sum;
  const double lane_score_adjustment =
      adjustment_numerator_sum / lane_information;
  const double lane_penalized_deviance =
      non_active_deviance[lane_index] + deviance_sum - log(lane_information);
  const double lane_score = score_sum + lane_score_adjustment;
  genotype_information[lane_index] = lane_information;
  score_adjustment[lane_index] = lane_score_adjustment;
  penalized_deviance[lane_index] = lane_penalized_deviance;
  score[lane_index] = lane_score;
  valid[lane_index] = isfinite(lane_information) &&
      lane_information > minimum_variance[lane_index] &&
      isfinite(lane_penalized_deviance) && isfinite(lane_score);
}
