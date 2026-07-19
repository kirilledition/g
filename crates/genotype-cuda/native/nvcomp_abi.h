#ifndef G_GENOTYPE_CUDA_NVCOMP_ABI_H_
#define G_GENOTYPE_CUDA_NVCOMP_ABI_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cuda_driver_abi.h"

namespace g::genotype_cuda::abi {

using NvcompStatus = std::int32_t;

struct NvcompProperties {
  std::uint32_t version;
  std::uint32_t cuda_runtime_version;
};

struct NvcompAlignmentRequirements {
  // Official 5.2.0.13 and 5.3.0.16 runtimes report {input=4, output=1, temporary=1}
  // for the CUDA DEFLATE backend. Rust passes its centralized member alignment
  // during initialization, and the runtime boundary rejects an uncovered value.
  std::size_t input;
  std::size_t output;
  std::size_t temporary;
};

struct NvcompDeflateDecompressOptions {
  std::int32_t backend;
  std::int32_t sort_before_hardware_decompression;
  char reserved[56];
};

inline constexpr NvcompStatus kNvcompSuccess = 0;
inline constexpr std::int32_t kNvcompCudaBackend = 2;
inline constexpr std::uint32_t kMinimumNvcompVersion = 5200;
inline constexpr std::uint32_t kMaximumNvcompVersion = 6000;

using NvcompGetProperties = NvcompStatus (*)(NvcompProperties* properties);
using NvcompGetStatusString = const char* (*)(NvcompStatus status);
using NvcompDeflateGetRequiredAlignments = NvcompStatus (*)(NvcompDeflateDecompressOptions options,
                                                            NvcompAlignmentRequirements* requirements);
using NvcompDeflateGetTemporarySize = NvcompStatus (*)(std::size_t chunk_count,
                                                       std::size_t maximum_uncompressed_chunk_bytes,
                                                       NvcompDeflateDecompressOptions options,
                                                       std::size_t* temporary_bytes,
                                                       std::size_t maximum_total_uncompressed_bytes);
using NvcompDeflateDecompress = NvcompStatus (*)(const void* const* device_compressed_chunk_pointers,
                                                 const std::size_t* device_compressed_chunk_bytes,
                                                 const std::size_t* device_uncompressed_buffer_bytes,
                                                 std::size_t* device_uncompressed_chunk_bytes,
                                                 std::size_t chunk_count,
                                                 void* device_temporary,
                                                 std::size_t temporary_bytes,
                                                 void* const* device_uncompressed_chunk_pointers,
                                                 NvcompDeflateDecompressOptions options,
                                                 NvcompStatus* device_statuses,
                                                 CudaStream stream);

static_assert(sizeof(NvcompStatus) == 4);
static_assert(sizeof(NvcompProperties) == 8);
static_assert(sizeof(NvcompAlignmentRequirements) == 3 * sizeof(std::size_t));
static_assert(sizeof(NvcompDeflateDecompressOptions) == 64);
static_assert(offsetof(NvcompDeflateDecompressOptions, reserved) == 8);
static_assert(std::is_standard_layout_v<NvcompDeflateDecompressOptions>);
static_assert(std::is_trivially_copyable_v<NvcompDeflateDecompressOptions>);

}  // namespace g::genotype_cuda::abi

#endif  // G_GENOTYPE_CUDA_NVCOMP_ABI_H_
