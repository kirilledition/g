#define STATUS_NVCOMP 1u
#define STATUS_LENGTH 2u
#define STATUS_SAMPLE_COUNT 4u
#define STATUS_ALLELE_COUNT 8u
#define STATUS_PLOIDY_RANGE 16u
#define STATUS_SAMPLE_PLOIDY 32u
#define STATUS_PHASE 64u
#define STATUS_BIT_COUNT 128u
#define STATUS_PAIR_SUM 256u
#define STATUS_ADLER32 512u
#define STATUS_SAMPLE_INDEX 1024u
#define STATUS_DESCRIPTOR 2048u

extern "C" __global__ void build_nvcomp_descriptors(const unsigned char* compressed_slab,
                                                    unsigned long long compressed_slab_bytes,
                                                    const unsigned int* compressed_metadata,
                                                    unsigned long long input_alignment,
                                                    unsigned char* fallback_input,
                                                    unsigned char* output_slab,
                                                    unsigned long long output_stride,
                                                    const void** input_pointers,
                                                    unsigned long long* input_sizes,
                                                    void** output_pointers,
                                                    unsigned long long* output_capacities,
                                                    unsigned int* descriptor_statuses,
                                                    unsigned long long chunk_count) {
  const unsigned long long chunk_index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk_index >= chunk_count) {
    return;
  }
  if (chunk_index == 0) {
    // A final empty stored block is a valid raw-DEFLATE stream. Invalid
    // descriptors share this stream so nvCOMP never sees malformed bounds.
    fallback_input[0] = 0x01;
    fallback_input[1] = 0x00;
    fallback_input[2] = 0x00;
    fallback_input[3] = 0xff;
    fallback_input[4] = 0xff;
  }
  const unsigned long long input_offset = compressed_metadata[chunk_index * 3];
  const unsigned long long input_size = compressed_metadata[chunk_index * 3 + 1];
  const bool offset_in_bounds = input_offset <= compressed_slab_bytes;
  const bool size_in_bounds = offset_in_bounds && input_size <= compressed_slab_bytes - input_offset;
  const bool input_is_aligned = input_alignment != 0 && input_offset % input_alignment == 0;
  const bool descriptor_is_valid = input_size != 0 && size_in_bounds && input_is_aligned;

  descriptor_statuses[chunk_index] = descriptor_is_valid ? 0 : STATUS_DESCRIPTOR;
  input_pointers[chunk_index] = descriptor_is_valid ? compressed_slab + input_offset : fallback_input;
  input_sizes[chunk_index] = descriptor_is_valid ? input_size : 5;
  output_pointers[chunk_index] = output_slab + chunk_index * output_stride;
  output_capacities[chunk_index] = output_stride;
}

__device__ __forceinline__ unsigned int load_u32_little_endian(const unsigned char* bytes) {
  return (unsigned int)bytes[0] | ((unsigned int)bytes[1] << 8) | ((unsigned int)bytes[2] << 16) |
         ((unsigned int)bytes[3] << 24);
}

__device__ __forceinline__ float packed8_genotype_mean(unsigned long long raw_dosage_sum,
                                                       unsigned long long selected_sample_count) {
  // Match the host's two explicitly rounded f32 operations. The exact
  // reciprocal is f32(1 / 255); rounded intrinsics prevent reassociation.
  const float packed8_probability_scale = 0x1.010102p-8f;
  const float dosage_sum = __fmul_rn(__ull2float_rn(raw_dosage_sum), packed8_probability_scale);
  return __fdiv_rn(dosage_sum, __ull2float_rn(selected_sample_count));
}

extern "C" __global__ void finalize_packed8(const unsigned char* decompressed_slab,
                                            const unsigned long long* actual_sizes,
                                            const int* nvcomp_statuses,
                                            const unsigned int* compressed_metadata,
                                            const unsigned int* descriptor_statuses,
                                            const unsigned int* selected_sample_indices,
                                            long long selection_start,
                                            unsigned long long logical_variant_count,
                                            unsigned long long compute_variant_count,
                                            unsigned long long source_sample_count,
                                            unsigned long long selected_sample_count,
                                            unsigned long long output_stride,
                                            unsigned char* probabilities,
                                            unsigned long long* raw_dosage_sums,
                                            unsigned long long* raw_dosage_square_sums,
                                            unsigned int* zero_counts,
                                            unsigned int* homozygous_alternate_counts,
                                            unsigned int* statuses,
                                            float* genotype_means) {
  constexpr unsigned int adler_modulus = 65521;
  constexpr unsigned int kernel_block_size = 256;
  constexpr unsigned int warp_size = 32;
  constexpr unsigned int warp_count = kernel_block_size / warp_size;
  constexpr unsigned int full_warp_mask = 0xffffffffu;

  const unsigned long long variant_index = blockIdx.x;
  const unsigned int thread_index = threadIdx.x;
  const unsigned int lane_index = thread_index % warp_size;
  const unsigned int warp_index = thread_index / warp_size;
  if (variant_index >= compute_variant_count) {
    return;
  }

  unsigned char* probability_row = probabilities + variant_index * selected_sample_count * 2;
  if (variant_index >= logical_variant_count) {
    for (unsigned long long selected_index = thread_index; selected_index < selected_sample_count;
         selected_index += kernel_block_size) {
      probability_row[selected_index * 2] = 255;
      probability_row[selected_index * 2 + 1] = 0;
    }
    if (thread_index == 0) {
      raw_dosage_sums[variant_index] = 0;
      raw_dosage_square_sums[variant_index] = 0;
      zero_counts[variant_index] = 0;
      homozygous_alternate_counts[variant_index] = 0;
      statuses[variant_index] = 0;
      genotype_means[variant_index] = 0.0f;
    }
    return;
  }

  __shared__ unsigned int row_gate_status;
  if (thread_index == 0) {
    row_gate_status = descriptor_statuses[variant_index];
    if (row_gate_status == 0 && nvcomp_statuses[variant_index] != 0) {
      row_gate_status = STATUS_NVCOMP;
    }
    if (row_gate_status == 0 && actual_sizes[variant_index] != output_stride) {
      row_gate_status = STATUS_LENGTH;
    }
  }
  __syncthreads();

  if (row_gate_status != 0) {
    for (unsigned long long selected_index = thread_index; selected_index < selected_sample_count;
         selected_index += kernel_block_size) {
      probability_row[selected_index * 2] = 255;
      probability_row[selected_index * 2 + 1] = 0;
    }
    if (thread_index == 0) {
      raw_dosage_sums[variant_index] = 0;
      raw_dosage_square_sums[variant_index] = 0;
      zero_counts[variant_index] = 0;
      homozygous_alternate_counts[variant_index] = 0;
      statuses[variant_index] = row_gate_status;
      genotype_means[variant_index] = 0.0f;
    }
    return;
  }

  const unsigned char* row = decompressed_slab + variant_index * output_stride;
  const unsigned long long probability_offset = 10 + source_sample_count;
  const bool identity_selection = selection_start == 0 && selected_sample_count == source_sample_count;
  unsigned long long local_sum = 0;
  unsigned long long local_square_sum = 0;
  unsigned long long local_adler_sum = 0;
  unsigned long long local_adler_weighted_sum = 0;
  unsigned int local_zero_count = 0;
  unsigned int local_homozygous_alternate_count = 0;
  unsigned int local_status = 0;

  if (thread_index == 0) {
    const unsigned char header_bytes[8] = {row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]};
    if (load_u32_little_endian(header_bytes) != source_sample_count) {
      local_status |= STATUS_SAMPLE_COUNT;
    }
    if (((unsigned int)header_bytes[4] | ((unsigned int)header_bytes[5] << 8)) != 2) {
      local_status |= STATUS_ALLELE_COUNT;
    }
    if (header_bytes[6] != 2 || header_bytes[7] != 2) {
      local_status |= STATUS_PLOIDY_RANGE;
    }
    for (unsigned int header_index = 0; header_index < 8; ++header_index) {
      const unsigned long long byte_value = header_bytes[header_index];
      local_adler_sum += byte_value;
      local_adler_weighted_sum += (output_stride - header_index) * byte_value;
    }
    const unsigned long long phase_index = 8 + source_sample_count;
    const unsigned long long bit_count_index = phase_index + 1;
    const unsigned char phase = row[phase_index];
    const unsigned char bit_count = row[bit_count_index];
    if (phase != 0) {
      local_status |= STATUS_PHASE;
    }
    if (bit_count != 8) {
      local_status |= STATUS_BIT_COUNT;
    }
    local_adler_sum += phase + bit_count;
    local_adler_weighted_sum += (output_stride - phase_index) * phase + (output_stride - bit_count_index) * bit_count;
  }

  for (unsigned long long source_index = thread_index; source_index < source_sample_count;
       source_index += kernel_block_size) {
    const unsigned long long ploidy_index = 8 + source_index;
    const unsigned long long first_probability_index = probability_offset + source_index * 2;
    const unsigned long long second_probability_index = first_probability_index + 1;
    const unsigned int ploidy = row[ploidy_index];
    const unsigned int first_probability = row[first_probability_index];
    const unsigned int second_probability = row[second_probability_index];
    if (ploidy != 2) {
      local_status |= STATUS_SAMPLE_PLOIDY;
    }
    if (first_probability + second_probability > 255) {
      local_status |= STATUS_PAIR_SUM;
    }
    local_adler_sum += ploidy + first_probability + second_probability;
    local_adler_weighted_sum += (output_stride - ploidy_index) * ploidy +
                                (output_stride - first_probability_index) * first_probability +
                                (output_stride - second_probability_index) * second_probability;

    if (identity_selection) {
      probability_row[source_index * 2] = (unsigned char)first_probability;
      probability_row[source_index * 2 + 1] = (unsigned char)second_probability;
      const unsigned long long raw_dosage = 510 - 2 * first_probability - second_probability;
      local_sum += raw_dosage;
      local_square_sum += raw_dosage * raw_dosage;
      local_zero_count += raw_dosage == 0;
      local_homozygous_alternate_count += raw_dosage >= 383;
    }
  }

  if (!identity_selection) {
    for (unsigned long long selected_index = thread_index; selected_index < selected_sample_count;
         selected_index += kernel_block_size) {
      const unsigned long long source_index = selection_start >= 0
                                                  ? (unsigned long long)selection_start + selected_index
                                                  : selected_sample_indices[selected_index];
      unsigned int first_probability = 255;
      unsigned int second_probability = 0;
      if (source_index < source_sample_count) {
        first_probability = row[probability_offset + source_index * 2];
        second_probability = row[probability_offset + source_index * 2 + 1];
      } else {
        local_status |= STATUS_SAMPLE_INDEX;
      }
      probability_row[selected_index * 2] = (unsigned char)first_probability;
      probability_row[selected_index * 2 + 1] = (unsigned char)second_probability;
      const unsigned long long raw_dosage = 510 - 2 * first_probability - second_probability;
      local_sum += raw_dosage;
      local_square_sum += raw_dosage * raw_dosage;
      local_zero_count += raw_dosage == 0;
      local_homozygous_alternate_count += raw_dosage >= 383;
    }
  }

  for (unsigned int offset = warp_size / 2; offset != 0; offset /= 2) {
    local_sum += __shfl_down_sync(full_warp_mask, local_sum, offset);
    local_square_sum += __shfl_down_sync(full_warp_mask, local_square_sum, offset);
    local_adler_sum += __shfl_down_sync(full_warp_mask, local_adler_sum, offset);
    local_adler_weighted_sum += __shfl_down_sync(full_warp_mask, local_adler_weighted_sum, offset);
    local_zero_count += __shfl_down_sync(full_warp_mask, local_zero_count, offset);
    local_homozygous_alternate_count += __shfl_down_sync(full_warp_mask, local_homozygous_alternate_count, offset);
    local_status |= __shfl_down_sync(full_warp_mask, local_status, offset);
  }

  __shared__ unsigned long long warp_sums[warp_count];
  __shared__ unsigned long long warp_square_sums[warp_count];
  __shared__ unsigned long long warp_adler_sums[warp_count];
  __shared__ unsigned long long warp_adler_weighted_sums[warp_count];
  __shared__ unsigned int warp_zero_counts[warp_count];
  __shared__ unsigned int warp_homozygous_alternate_counts[warp_count];
  __shared__ unsigned int warp_statuses[warp_count];
  if (lane_index == 0) {
    warp_sums[warp_index] = local_sum;
    warp_square_sums[warp_index] = local_square_sum;
    warp_adler_sums[warp_index] = local_adler_sum;
    warp_adler_weighted_sums[warp_index] = local_adler_weighted_sum;
    warp_zero_counts[warp_index] = local_zero_count;
    warp_homozygous_alternate_counts[warp_index] = local_homozygous_alternate_count;
    warp_statuses[warp_index] = local_status;
  }
  __syncthreads();

  if (warp_index != 0) {
    return;
  }
  local_sum = lane_index < warp_count ? warp_sums[lane_index] : 0;
  local_square_sum = lane_index < warp_count ? warp_square_sums[lane_index] : 0;
  local_adler_sum = lane_index < warp_count ? warp_adler_sums[lane_index] : 0;
  local_adler_weighted_sum = lane_index < warp_count ? warp_adler_weighted_sums[lane_index] : 0;
  local_zero_count = lane_index < warp_count ? warp_zero_counts[lane_index] : 0;
  local_homozygous_alternate_count = lane_index < warp_count ? warp_homozygous_alternate_counts[lane_index] : 0;
  local_status = lane_index < warp_count ? warp_statuses[lane_index] : 0;
  for (unsigned int offset = warp_size / 2; offset != 0; offset /= 2) {
    local_sum += __shfl_down_sync(full_warp_mask, local_sum, offset);
    local_square_sum += __shfl_down_sync(full_warp_mask, local_square_sum, offset);
    local_adler_sum += __shfl_down_sync(full_warp_mask, local_adler_sum, offset);
    local_adler_weighted_sum += __shfl_down_sync(full_warp_mask, local_adler_weighted_sum, offset);
    local_zero_count += __shfl_down_sync(full_warp_mask, local_zero_count, offset);
    local_homozygous_alternate_count += __shfl_down_sync(full_warp_mask, local_homozygous_alternate_count, offset);
    local_status |= __shfl_down_sync(full_warp_mask, local_status, offset);
  }
  if (lane_index != 0) {
    return;
  }

  const unsigned long long first_adler_sum = (1 + local_adler_sum) % adler_modulus;
  const unsigned long long second_adler_sum = (output_stride + local_adler_weighted_sum) % adler_modulus;
  const unsigned int observed_adler32 = ((unsigned int)second_adler_sum << 16) | (unsigned int)first_adler_sum;
  if (observed_adler32 != compressed_metadata[variant_index * 3 + 2]) {
    local_status |= STATUS_ADLER32;
  }
  raw_dosage_sums[variant_index] = local_sum;
  raw_dosage_square_sums[variant_index] = local_square_sum;
  zero_counts[variant_index] = local_zero_count;
  homozygous_alternate_counts[variant_index] = local_homozygous_alternate_count;
  statuses[variant_index] = local_status;
  genotype_means[variant_index] = packed8_genotype_mean(local_sum, selected_sample_count);
}
