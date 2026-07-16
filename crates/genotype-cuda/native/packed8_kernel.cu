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

extern "C" __global__ void build_nvcomp_descriptors(
    const unsigned char* compressed_slab,
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
  const unsigned long long chunk_index =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
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
  const bool size_in_bounds =
      offset_in_bounds && input_size <= compressed_slab_bytes - input_offset;
  const bool input_is_aligned =
      input_alignment != 0 && input_offset % input_alignment == 0;
  const bool descriptor_is_valid =
      input_size != 0 && size_in_bounds && input_is_aligned;

  descriptor_statuses[chunk_index] =
      descriptor_is_valid ? 0 : STATUS_DESCRIPTOR;
  input_pointers[chunk_index] =
      descriptor_is_valid ? compressed_slab + input_offset : fallback_input;
  input_sizes[chunk_index] = descriptor_is_valid ? input_size : 5;
  output_pointers[chunk_index] = output_slab + chunk_index * output_stride;
  output_capacities[chunk_index] = output_stride;
}

__device__ __forceinline__ unsigned int load_u32_little_endian(
    const unsigned char* bytes) {
  return (unsigned int)bytes[0] |
      ((unsigned int)bytes[1] << 8) |
      ((unsigned int)bytes[2] << 16) |
      ((unsigned int)bytes[3] << 24);
}

__device__ __forceinline__ float packed8_genotype_mean(
    unsigned long long raw_dosage_sum,
    unsigned long long selected_sample_count) {
  // Match the host's two explicitly rounded f32 operations. The exact
  // reciprocal is f32(1 / 255); rounded intrinsics prevent reassociation.
  const float packed8_probability_scale = 0x1.010102p-8f;
  const float dosage_sum = __fmul_rn(
      __ull2float_rn(raw_dosage_sum), packed8_probability_scale);
  return __fdiv_rn(dosage_sum, __ull2float_rn(selected_sample_count));
}

extern "C" __global__ void finalize_packed8(
    const unsigned char* decompressed_slab,
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
  const unsigned long long variant_index = blockIdx.x;
  const unsigned int thread_index = threadIdx.x;
  if (variant_index >= compute_variant_count) {
    return;
  }

  unsigned char* probability_row =
      probabilities + variant_index * selected_sample_count * 2;
  if (variant_index >= logical_variant_count) {
    for (unsigned long long selected_index = thread_index;
         selected_index < selected_sample_count;
         selected_index += blockDim.x) {
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
    for (unsigned long long selected_index = thread_index;
         selected_index < selected_sample_count;
         selected_index += blockDim.x) {
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
  unsigned int local_status = 0;
  if (thread_index == 0) {
    if (load_u32_little_endian(row) != source_sample_count) {
      local_status |= STATUS_SAMPLE_COUNT;
    }
    if (((unsigned int)row[4] | ((unsigned int)row[5] << 8)) != 2) {
      local_status |= STATUS_ALLELE_COUNT;
    }
    if (row[6] != 2 || row[7] != 2) {
      local_status |= STATUS_PLOIDY_RANGE;
    }
    if (row[8 + source_sample_count] != 0) {
      local_status |= STATUS_PHASE;
    }
    if (row[9 + source_sample_count] != 8) {
      local_status |= STATUS_BIT_COUNT;
    }
  }

  const unsigned long long probability_offset = 10 + source_sample_count;
  for (unsigned long long source_index = thread_index;
       source_index < source_sample_count;
       source_index += blockDim.x) {
    if (row[8 + source_index] != 2) {
      local_status |= STATUS_SAMPLE_PLOIDY;
    }
    const unsigned int first_probability = row[probability_offset + source_index * 2];
    const unsigned int second_probability = row[probability_offset + source_index * 2 + 1];
    if (first_probability + second_probability > 255) {
      local_status |= STATUS_PAIR_SUM;
    }
  }

  unsigned long long local_sum = 0;
  unsigned long long local_square_sum = 0;
  unsigned int local_zero_count = 0;
  unsigned int local_homozygous_alternate_count = 0;
  for (unsigned long long selected_index = thread_index;
       selected_index < selected_sample_count;
       selected_index += blockDim.x) {
    const unsigned long long source_index = selection_start >= 0
        ? (unsigned long long)selection_start + selected_index
        : selected_sample_indices[selected_index];
    unsigned int first_probability = 255;
    unsigned int second_probability = 0;
    if (source_index < source_sample_count) {
      first_probability = row[probability_offset + (unsigned long long)source_index * 2];
      second_probability = row[probability_offset + (unsigned long long)source_index * 2 + 1];
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

  const unsigned long long segment_start = output_stride * thread_index / blockDim.x;
  const unsigned long long segment_end = output_stride * (thread_index + 1) / blockDim.x;
  unsigned long long segment_sum = 0;
  unsigned long long segment_weighted_sum = 0;
  for (unsigned long long byte_index = segment_start; byte_index < segment_end; ++byte_index) {
    segment_sum += row[byte_index];
    segment_weighted_sum += segment_sum;
  }

  // The host ABI launches exactly 256 threads; every shared reduction array
  // therefore has one element per participating thread.
  __shared__ unsigned long long reduction_sums[256];
  __shared__ unsigned long long reduction_square_sums[256];
  __shared__ unsigned int reduction_zero_counts[256];
  __shared__ unsigned int reduction_homozygous_alternate_counts[256];
  __shared__ unsigned int reduction_statuses[256];
  __shared__ unsigned int adler_sums[256];
  __shared__ unsigned int adler_weighted_sums[256];
  __shared__ unsigned long long adler_lengths[256];

  reduction_sums[thread_index] = local_sum;
  reduction_square_sums[thread_index] = local_square_sum;
  reduction_zero_counts[thread_index] = local_zero_count;
  reduction_homozygous_alternate_counts[thread_index] = local_homozygous_alternate_count;
  reduction_statuses[thread_index] = local_status;
  adler_sums[thread_index] = segment_sum % 65521;
  adler_weighted_sums[thread_index] = segment_weighted_sum % 65521;
  adler_lengths[thread_index] = segment_end - segment_start;
  __syncthreads();

  if (thread_index == 0) {
    unsigned long long first_sum = 1;
    unsigned long long second_sum = 0;
    for (unsigned int segment_index = 0; segment_index < blockDim.x; ++segment_index) {
      second_sum = (second_sum + adler_weighted_sums[segment_index] +
                    (unsigned long long)adler_lengths[segment_index] * first_sum) % 65521;
      first_sum = (first_sum + adler_sums[segment_index]) % 65521;
    }
    const unsigned int observed_adler32 =
        ((unsigned int)second_sum << 16) | (unsigned int)first_sum;
    if (observed_adler32 != compressed_metadata[variant_index * 3 + 2]) {
      reduction_statuses[0] |= STATUS_ADLER32;
    }
  }
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset != 0; offset /= 2) {
    if (thread_index < offset) {
      reduction_sums[thread_index] += reduction_sums[thread_index + offset];
      reduction_square_sums[thread_index] += reduction_square_sums[thread_index + offset];
      reduction_zero_counts[thread_index] += reduction_zero_counts[thread_index + offset];
      reduction_homozygous_alternate_counts[thread_index] +=
          reduction_homozygous_alternate_counts[thread_index + offset];
      reduction_statuses[thread_index] |= reduction_statuses[thread_index + offset];
    }
    __syncthreads();
  }

  if (thread_index == 0) {
    raw_dosage_sums[variant_index] = reduction_sums[0];
    raw_dosage_square_sums[variant_index] = reduction_square_sums[0];
    zero_counts[variant_index] = reduction_zero_counts[0];
    homozygous_alternate_counts[variant_index] = reduction_homozygous_alternate_counts[0];
    statuses[variant_index] = reduction_statuses[0];
    genotype_means[variant_index] = packed8_genotype_mean(
        reduction_sums[0], selected_sample_count);
  }
}
