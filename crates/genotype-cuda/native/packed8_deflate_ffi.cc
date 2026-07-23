#include <xla/ffi/api/ffi.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

#include "cuda_driver.h"
#include "nvcomp_abi.h"

static_assert(XLA_FFI_API_MAJOR == 0);
static_assert(XLA_FFI_API_MINOR == 3);
static_assert(sizeof(std::size_t) == sizeof(std::uint64_t));

struct GGenotypeCudaCapability {
  std::uint32_t nvcomp_version;
  std::uint32_t nvcomp_cuda_runtime_version;
  std::int32_t cuda_driver_version;
  std::int32_t device_ordinal;
  std::int32_t compute_capability_major;
  std::int32_t compute_capability_minor;
  std::size_t nvcomp_input_alignment;
};

static_assert(sizeof(GGenotypeCudaCapability) == 32);
static_assert(std::is_standard_layout_v<GGenotypeCudaCapability>);

namespace {

using g::cuda_native::CudaContext;
using g::cuda_native::CudaFunction;
using g::cuda_native::CudaStream;
using g::cuda_native::DynamicLibrary;
using g::genotype_cuda::abi::NvcompAlignmentRequirements;
using g::genotype_cuda::abi::NvcompDeflateDecompress;
using g::genotype_cuda::abi::NvcompDeflateDecompressOptions;
using g::genotype_cuda::abi::NvcompDeflateGetRequiredAlignments;
using g::genotype_cuda::abi::NvcompDeflateGetTemporarySize;
using g::genotype_cuda::abi::NvcompGetProperties;
using g::genotype_cuda::abi::NvcompGetStatusString;
using g::genotype_cuda::abi::NvcompProperties;
using g::genotype_cuda::abi::NvcompStatus;
using xla::ffi::BufferR1;
using xla::ffi::BufferR2;
using xla::ffi::DataType;
using xla::ffi::Error;
using xla::ffi::ErrorCode;
using xla::ffi::ResultBufferR1;
using xla::ffi::ResultBufferR3;
using xla::ffi::ScratchAllocator;

constexpr char kDescriptorKernelName[] = "build_nvcomp_descriptors";
constexpr char kFinalizeKernelName[] = "finalize_packed8";
constexpr std::int32_t kMinimumCudaDriverVersion = 12020;
constexpr std::int32_t kMinimumComputeCapabilityMajor = 7;
constexpr unsigned int kKernelBlockSize = 256;
// For a row of n bytes, the unreduced Adler B numerator is bounded by
// n + 255*n*(n+1)/2. This sample limit keeps that expression within uint64_t
// for n = 3*source_sample_count + 10 while exceeding practical BGEN sizes.
constexpr std::size_t kMaximumExactAdlerSourceSampleCount = 126'789'562;
static_assert(kKernelBlockSize == 256);

constexpr char kPacked8KernelPtx[] =
#include "packed8_kernel_ptx.inc"
    ;

enum class InitializationStatus : std::int32_t {
  kSuccess = 0,
  kCudaDriverUnavailable = 1,
  kNvcompLibraryUnavailable = 2,
  kRequiredSymbolUnavailable = 3,
  kNvcompVersionUnsupported = 4,
  kCudaDriverFailure = 5,
  kCudaDriverTooOld = 6,
  kCudaDeviceUnavailable = 7,
  kComputeCapabilityUnsupported = 8,
  kNvcompInputAlignmentUnsupported = 9,
  kInternal = 10,
};

class InitializationFailure final : public std::runtime_error {
 public:
  InitializationFailure(InitializationStatus status, const std::string& detail)
      : std::runtime_error(detail), status_(status) {}

  [[nodiscard]] InitializationStatus status() const noexcept { return status_; }

 private:
  InitializationStatus status_;
};

class HandlerFailure final : public std::runtime_error {
 public:
  HandlerFailure(ErrorCode code, const std::string& detail) : std::runtime_error(detail), code_(code) {}

  [[nodiscard]] ErrorCode code() const noexcept { return code_; }

 private:
  ErrorCode code_;
};

[[noreturn]] void fail_initialization(InitializationStatus status, const std::string& detail) {
  throw InitializationFailure(status, detail);
}

[[noreturn]] void fail_handler(ErrorCode code, const std::string& detail) { throw HandlerFailure(code, detail); }

[[noreturn]] void fail_runtime(const std::string& detail) { fail_handler(ErrorCode::kInternal, detail); }

struct CudaErrorFactory {
  static InitializationFailure initialization_failure(InitializationStatus status, const std::string& detail) {
    return InitializationFailure(status, detail);
  }

  static InitializationFailure initialization_failure(g::cuda_native::CudaInitializationFailureKind failure_kind,
                                                      const std::string& detail) {
    InitializationStatus status = InitializationStatus::kInternal;
    switch (failure_kind) {
      case g::cuda_native::CudaInitializationFailureKind::kCudaDriverUnavailable:
        status = InitializationStatus::kCudaDriverUnavailable;
        break;
      case g::cuda_native::CudaInitializationFailureKind::kRequiredSymbolUnavailable:
        status = InitializationStatus::kRequiredSymbolUnavailable;
        break;
      case g::cuda_native::CudaInitializationFailureKind::kCudaDriverFailure:
        status = InitializationStatus::kCudaDriverFailure;
        break;
      case g::cuda_native::CudaInitializationFailureKind::kCudaDeviceUnavailable:
        status = InitializationStatus::kCudaDeviceUnavailable;
        break;
    }
    return InitializationFailure(status, detail);
  }

  static HandlerFailure runtime_failure(g::cuda_native::CudaRuntimeFailureKind failure_kind,
                                        const std::string& detail) {
    const ErrorCode code = failure_kind == g::cuda_native::CudaRuntimeFailureKind::kFailedPrecondition
                               ? ErrorCode::kFailedPrecondition
                               : ErrorCode::kInternal;
    return HandlerFailure(code, detail);
  }
};

using CudaDriverApi = g::cuda_native::CudaDriverApi<CudaErrorFactory>;
using CudaDevice = g::cuda_native::CudaDevice;
using CudaModuleOwner = g::cuda_native::CudaModuleOwner<CudaDriverApi>;

[[nodiscard]] constexpr bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

class NvcompApi {
 public:
  NvcompApi()
      : library_(
            g::cuda_native::open_dynamic_library<CudaErrorFactory>("libnvcomp.so.5",
                                                                   InitializationStatus::kNvcompLibraryUnavailable)),
        get_properties_(g::cuda_native::load_dynamic_library_symbol<NvcompGetProperties, CudaErrorFactory>(
            library_.get(),
            "nvCOMP",
            "nvcompGetProperties",
            InitializationStatus::kRequiredSymbolUnavailable)),
        get_status_string_(g::cuda_native::load_dynamic_library_symbol<NvcompGetStatusString, CudaErrorFactory>(
            library_.get(),
            "nvCOMP",
            "nvcompGetStatusString",
            InitializationStatus::kRequiredSymbolUnavailable)),
        get_required_alignments_(
            g::cuda_native::load_dynamic_library_symbol<NvcompDeflateGetRequiredAlignments, CudaErrorFactory>(
                library_.get(),
                "nvCOMP",
                "nvcompBatchedDeflateDecompressGetRequiredAlignments",
                InitializationStatus::kRequiredSymbolUnavailable)),
        get_temporary_size_(
            g::cuda_native::load_dynamic_library_symbol<NvcompDeflateGetTemporarySize, CudaErrorFactory>(
                library_.get(),
                "nvCOMP",
                "nvcompBatchedDeflateDecompressGetTempSizeAsync",
                InitializationStatus::kRequiredSymbolUnavailable)),
        decompress_(g::cuda_native::load_dynamic_library_symbol<NvcompDeflateDecompress, CudaErrorFactory>(
            library_.get(),
            "nvCOMP",
            "nvcompBatchedDeflateDecompressAsync",
            InitializationStatus::kRequiredSymbolUnavailable)) {
    const NvcompStatus properties_status = get_properties_(&properties_);
    if (properties_status != g::genotype_cuda::abi::kNvcompSuccess) {
      fail_initialization(InitializationStatus::kInternal,
                          "query nvCOMP properties failed with status " + std::to_string(properties_status));
    }
    options_.backend = g::genotype_cuda::abi::kNvcompCudaBackend;
    const NvcompStatus alignment_status = get_required_alignments_(options_, &alignments_);
    if (alignment_status != g::genotype_cuda::abi::kNvcompSuccess) {
      fail_initialization(InitializationStatus::kInternal,
                          "query nvCOMP DEFLATE alignments failed with status " + std::to_string(alignment_status));
    }
    if (!is_power_of_two(alignments_.input) || !is_power_of_two(alignments_.output) ||
        !is_power_of_two(alignments_.temporary)) {
      fail_initialization(InitializationStatus::kInternal,
                          "nvCOMP returned a buffer alignment that is not a nonzero power of two");
    }
  }

  NvcompApi(const NvcompApi&) = delete;
  NvcompApi& operator=(const NvcompApi&) = delete;

  [[nodiscard]] const NvcompProperties& properties() const noexcept { return properties_; }
  [[nodiscard]] const NvcompAlignmentRequirements& alignments() const noexcept { return alignments_; }

  [[nodiscard]] std::size_t temporary_size(std::size_t chunk_count,
                                           std::size_t output_stride,
                                           std::size_t total_output_bytes) const {
    std::size_t temporary_bytes = 0;
    check(get_temporary_size_(chunk_count, output_stride, options_, &temporary_bytes, total_output_bytes),
          "query nvCOMP DEFLATE temporary size");
    return temporary_bytes;
  }

  void decompress(const void* const* input_pointers,
                  const std::size_t* input_sizes,
                  const std::size_t* output_capacities,
                  std::size_t* actual_sizes,
                  std::size_t chunk_count,
                  void* temporary,
                  std::size_t temporary_bytes,
                  void* const* output_pointers,
                  NvcompStatus* statuses,
                  CudaStream stream) const {
    check(decompress_(input_pointers,
                      input_sizes,
                      output_capacities,
                      actual_sizes,
                      chunk_count,
                      temporary,
                      temporary_bytes,
                      output_pointers,
                      options_,
                      statuses,
                      stream),
          "launch nvCOMP DEFLATE decompression");
  }

 private:
  void check(NvcompStatus status, std::string_view operation) const {
    if (status == g::genotype_cuda::abi::kNvcompSuccess) {
      return;
    }
    const char* status_string = get_status_string_(status);
    const std::string detail = status_string == nullptr ? std::to_string(status) : status_string;
    fail_runtime(std::string(operation) + ": " + detail);
  }

  DynamicLibrary library_;
  NvcompGetProperties get_properties_;
  NvcompGetStatusString get_status_string_;
  NvcompDeflateGetRequiredAlignments get_required_alignments_;
  NvcompDeflateGetTemporarySize get_temporary_size_;
  NvcompDeflateDecompress decompress_;
  NvcompProperties properties_{};
  NvcompDeflateDecompressOptions options_{};
  NvcompAlignmentRequirements alignments_{};
};

class Packed8Kernels {
 public:
  explicit Packed8Kernels(const CudaDriverApi& driver)
      : driver_(driver), module_(driver, driver.load_module(kPacked8KernelPtx, "load packed8 compute_70 PTX")) {
    descriptor_function_ = driver_.get_function(module_.get(), kDescriptorKernelName);
    finalize_function_ = driver_.get_function(module_.get(), kFinalizeKernelName);
  }

  Packed8Kernels(const Packed8Kernels&) = delete;
  Packed8Kernels& operator=(const Packed8Kernels&) = delete;

  void launch_descriptors(const std::uint8_t* compressed_slab,
                          std::size_t compressed_slab_bytes,
                          const std::uint32_t* compressed_metadata,
                          std::size_t input_alignment,
                          std::uint8_t* fallback_input,
                          std::uint8_t* output_slab,
                          std::size_t output_stride,
                          const void** input_pointers,
                          std::size_t* input_sizes,
                          void** output_pointers,
                          std::size_t* output_capacities,
                          std::uint32_t* descriptor_statuses,
                          std::size_t chunk_count,
                          CudaStream stream) const {
    const std::size_t grid_size = (chunk_count + kKernelBlockSize - 1) / kKernelBlockSize;
    if (grid_size > std::numeric_limits<unsigned int>::max()) {
      fail_runtime("descriptor kernel grid exceeds uint32");
    }
    std::uint64_t compressed_slab_bytes_argument = compressed_slab_bytes;
    std::uint64_t input_alignment_argument = input_alignment;
    std::uint64_t output_stride_argument = output_stride;
    std::uint64_t chunk_count_argument = chunk_count;
    void* arguments[] = {
        &compressed_slab,
        &compressed_slab_bytes_argument,
        &compressed_metadata,
        &input_alignment_argument,
        &fallback_input,
        &output_slab,
        &output_stride_argument,
        &input_pointers,
        &input_sizes,
        &output_pointers,
        &output_capacities,
        &descriptor_statuses,
        &chunk_count_argument,
    };
    driver_.launch_kernel(descriptor_function_,
                          static_cast<unsigned int>(grid_size),
                          kKernelBlockSize,
                          stream,
                          arguments,
                          "launch descriptor kernel");
  }

  void launch_finalize(const std::uint8_t* decompressed_slab,
                       const std::size_t* actual_sizes,
                       const NvcompStatus* nvcomp_statuses,
                       const std::uint32_t* compressed_metadata,
                       const std::uint32_t* descriptor_statuses,
                       const std::uint32_t* selected_sample_indices,
                       std::int64_t selection_start,
                       std::size_t logical_variant_count,
                       std::size_t compute_variant_count,
                       std::size_t source_sample_count,
                       std::size_t selected_sample_count,
                       std::size_t output_stride,
                       std::uint8_t* probabilities,
                       std::uint64_t* raw_dosage_sums,
                       std::uint64_t* raw_dosage_square_sums,
                       std::uint32_t* zero_counts,
                       std::uint32_t* homozygous_alternate_counts,
                       std::uint32_t* statuses,
                       float* genotype_means,
                       CudaStream stream) const {
    if (compute_variant_count > std::numeric_limits<unsigned int>::max()) {
      fail_runtime("packed8 finalize kernel grid exceeds uint32");
    }
    std::uint64_t logical_variant_count_argument = logical_variant_count;
    std::uint64_t compute_variant_count_argument = compute_variant_count;
    std::uint64_t source_sample_count_argument = source_sample_count;
    std::uint64_t selected_sample_count_argument = selected_sample_count;
    std::uint64_t output_stride_argument = output_stride;
    void* arguments[] = {
        &decompressed_slab,
        &actual_sizes,
        &nvcomp_statuses,
        &compressed_metadata,
        &descriptor_statuses,
        &selected_sample_indices,
        &selection_start,
        &logical_variant_count_argument,
        &compute_variant_count_argument,
        &source_sample_count_argument,
        &selected_sample_count_argument,
        &output_stride_argument,
        &probabilities,
        &raw_dosage_sums,
        &raw_dosage_square_sums,
        &zero_counts,
        &homozygous_alternate_counts,
        &statuses,
        &genotype_means,
    };
    driver_.launch_kernel(finalize_function_,
                          static_cast<unsigned int>(compute_variant_count),
                          kKernelBlockSize,
                          stream,
                          arguments,
                          "launch packed8 finalize kernel");
  }

 private:
  const CudaDriverApi& driver_;
  CudaModuleOwner module_;
  CudaFunction descriptor_function_ = nullptr;
  CudaFunction finalize_function_ = nullptr;
};

class Packed8KernelCache {
 public:
  explicit Packed8KernelCache(const CudaDriverApi& driver) : driver_(driver) {}

  [[nodiscard]] const Packed8Kernels& for_current_context(CudaDevice qualified_device) {
    // JAX keeps CUDA contexts alive for the process lifetime. Holding modules by
    // context avoids repeated PTX loads; context destruction or handle reuse is
    // outside this handler's supported lifecycle.
    const CudaContext context = driver_.current_context();
    struct ThreadCache {
      const Packed8KernelCache* owner = nullptr;
      CudaContext context = nullptr;
      const Packed8Kernels* kernels = nullptr;
    };
    thread_local ThreadCache thread_cache;
    if (thread_cache.owner == this && thread_cache.context == context && thread_cache.kernels != nullptr) {
      return *thread_cache.kernels;
    }

    std::scoped_lock lock(mutex_);
    auto iterator = kernels_.find(context);
    if (iterator == kernels_.end()) {
      driver_.validate_current_context_device(qualified_device,
                                              kMinimumComputeCapabilityMajor,
                                              "packed8 nvCOMP FFI requires 7.0 or newer");
      iterator = kernels_.emplace(context, std::make_unique<Packed8Kernels>(driver_)).first;
    }
    thread_cache = ThreadCache{this, context, iterator->second.get()};
    return *thread_cache.kernels;
  }

 private:
  const CudaDriverApi& driver_;
  std::mutex mutex_;
  std::unordered_map<CudaContext, std::unique_ptr<Packed8Kernels>> kernels_;
};

class RuntimeState {
 public:
  RuntimeState() : kernels_(driver_) {}

  RuntimeState(const RuntimeState&) = delete;
  RuntimeState& operator=(const RuntimeState&) = delete;

  void validate(std::int32_t device_ordinal, std::size_t member_alignment, GGenotypeCudaCapability& capability) {
    const NvcompProperties& properties = nvcomp_.properties();
    const NvcompAlignmentRequirements& alignments = nvcomp_.alignments();
    capability.nvcomp_version = properties.version;
    capability.nvcomp_cuda_runtime_version = properties.cuda_runtime_version;
    capability.cuda_driver_version = driver_.driver_version();
    capability.device_ordinal = device_ordinal;
    capability.nvcomp_input_alignment = alignments.input;

    if (properties.version < g::genotype_cuda::abi::kMinimumNvcompVersion ||
        properties.version >= g::genotype_cuda::abi::kMaximumNvcompVersion) {
      fail_initialization(InitializationStatus::kNvcompVersionUnsupported,
                          "expected 5.2 <= nvCOMP < 6, observed encoded version " + std::to_string(properties.version));
    }
    if (capability.cuda_driver_version < kMinimumCudaDriverVersion) {
      fail_initialization(InitializationStatus::kCudaDriverTooOld,
                          "embedded PTX ISA 8.2 requires CUDA driver API version 12020 or newer");
    }
    if (!is_power_of_two(member_alignment)) {
      fail_initialization(InitializationStatus::kInternal,
                          "Rust DEFLATE member alignment must be a nonzero power of two");
    }
    if (member_alignment % alignments.input != 0) {
      fail_initialization(InitializationStatus::kNvcompInputAlignmentUnsupported,
                          "configured " + std::to_string(member_alignment) +
                              "-byte DEFLATE member alignment does not cover runtime nvCOMP input alignment " +
                              std::to_string(alignments.input));
    }

    const CudaDevice inspected_device = driver_.inspect_device(device_ordinal, capability);
    if (capability.compute_capability_major < kMinimumComputeCapabilityMajor) {
      fail_initialization(InitializationStatus::kComputeCapabilityUnsupported,
                          "nvCOMP DEFLATE requires compute capability 7.0 or newer");
    }
    CudaDevice unqualified_device = kUnqualifiedDevice;
    if (!qualified_device_.compare_exchange_strong(unqualified_device, inspected_device) &&
        unqualified_device != inspected_device) {
      fail_initialization(InitializationStatus::kInternal,
                          "packed8 nvCOMP runtime was already qualified for a different visible device");
    }
  }

  [[nodiscard]] CudaDevice require_qualified_device() const {
    const CudaDevice qualified_device = qualified_device_.load();
    if (qualified_device == kUnqualifiedDevice) {
      fail_handler(ErrorCode::kFailedPrecondition,
                   "packed8 nvCOMP handler was invoked before its execution device was qualified");
    }
    return qualified_device;
  }

  [[nodiscard]] const NvcompApi& nvcomp() const {
    static_cast<void>(require_qualified_device());
    return nvcomp_;
  }

  [[nodiscard]] const Packed8Kernels& kernels() { return kernels_.for_current_context(require_qualified_device()); }

 private:
  static constexpr CudaDevice kUnqualifiedDevice = -1;
  CudaDriverApi driver_;
  NvcompApi nvcomp_;
  std::atomic<CudaDevice> qualified_device_{kUnqualifiedDevice};
  Packed8KernelCache kernels_;
};

RuntimeState& runtime_state() {
  // JAX owns the CUDA contexts and can destroy them before C++ static
  // destructors run. Keep the driver/nvCOMP libraries and context-bound
  // modules alive until process exit instead of attempting unsafe teardown.
  static RuntimeState* const state = new RuntimeState();
  return *state;
}

class WorkspaceLayout {
 public:
  template <typename Value>
  std::size_t append_array(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(Value)) {
      fail_runtime("packed8 workspace array size overflows size_t");
    }
    return append_bytes(count * sizeof(Value), alignof(Value));
  }

  std::size_t append_bytes(std::size_t byte_count, std::size_t alignment) {
    if (!is_power_of_two(alignment)) {
      fail_runtime("packed8 workspace alignment must be a nonzero power of two");
    }
    const std::size_t remainder = byte_count_ % alignment;
    const std::size_t padding = remainder == 0 ? 0 : alignment - remainder;
    if (padding > std::numeric_limits<std::size_t>::max() - byte_count_) {
      fail_runtime("packed8 workspace alignment overflows size_t");
    }
    const std::size_t offset = byte_count_ + padding;
    if (byte_count > std::numeric_limits<std::size_t>::max() - offset) {
      fail_runtime("packed8 workspace size overflows size_t");
    }
    byte_count_ = offset + byte_count;
    alignment_ = std::max(alignment_, alignment);
    return offset;
  }

  [[nodiscard]] std::size_t byte_count() const noexcept { return byte_count_; }
  [[nodiscard]] std::size_t alignment() const noexcept { return alignment_; }

 private:
  std::size_t byte_count_ = 0;
  std::size_t alignment_ = 1;
};

template <DataType data_type>
bool is_result_vector(ResultBufferR1<data_type>& result, std::size_t expected_count) {
  const auto dimensions = result->dimensions();
  return dimensions.size() == 1 && dimensions[0] >= 0 && static_cast<std::uint64_t>(dimensions[0]) == expected_count;
}

Error decode_packed8(BufferR1<DataType::U8> compressed_slab,
                     BufferR2<DataType::U32> compressed_metadata,
                     BufferR1<DataType::U32> selected_sample_indices,
                     ResultBufferR3<DataType::U8> probabilities,
                     ResultBufferR1<DataType::U64> raw_dosage_sums,
                     ResultBufferR1<DataType::U64> raw_dosage_square_sums,
                     ResultBufferR1<DataType::U32> zero_counts,
                     ResultBufferR1<DataType::U32> homozygous_alternate_counts,
                     ResultBufferR1<DataType::U32> statuses,
                     ResultBufferR1<DataType::F32> genotype_means,
                     std::int64_t source_sample_count_attribute,
                     std::int64_t selection_start,
                     CudaStream stream,
                     ScratchAllocator scratch) {
  try {
    const auto metadata_dimensions = compressed_metadata.dimensions();
    if (metadata_dimensions.size() != 2 || metadata_dimensions[0] <= 0 || metadata_dimensions[1] != 3) {
      return Error::InvalidArgument("compressed metadata must have shape [logical_variants, 3]");
    }
    const std::size_t logical_variant_count = static_cast<std::size_t>(metadata_dimensions[0]);
    if (source_sample_count_attribute <= 0) {
      return Error::InvalidArgument("source_sample_count must be positive");
    }
    const std::size_t source_sample_count = static_cast<std::size_t>(source_sample_count_attribute);
    if (source_sample_count > kMaximumExactAdlerSourceSampleCount) {
      return Error::InvalidArgument(
          "source_sample_count exceeds the 126789562-sample exact CUDA Adler-32 accumulator limit");
    }
    if (source_sample_count > (std::numeric_limits<std::size_t>::max() - 10) / 3) {
      return Error::InvalidArgument("packed8 output stride overflows size_t");
    }
    const std::size_t output_stride = 3 * source_sample_count + 10;

    const auto probability_dimensions = probabilities->dimensions();
    if (probability_dimensions.size() != 3 || probability_dimensions[0] < 0 || probability_dimensions[1] <= 0 ||
        probability_dimensions[2] != 2) {
      return Error::InvalidArgument("packed8 probabilities must have shape [compute_variants, selected_samples, 2]");
    }
    const std::size_t compute_variant_count = static_cast<std::size_t>(probability_dimensions[0]);
    const std::size_t selected_sample_count = static_cast<std::size_t>(probability_dimensions[1]);
    if (compute_variant_count < logical_variant_count) {
      return Error::InvalidArgument("compute variant count must cover all logical variants");
    }

    const std::size_t selected_index_count = selected_sample_indices.element_count();
    if (selection_start >= 0) {
      if (selected_index_count != 0) {
        return Error::InvalidArgument("contiguous selection requires an empty selected-index operand");
      }
      const std::uint64_t selection_start_unsigned = static_cast<std::uint64_t>(selection_start);
      if (selection_start_unsigned > source_sample_count ||
          selected_sample_count > source_sample_count - selection_start_unsigned) {
        return Error::InvalidArgument("contiguous sample selection exceeds the source sample count");
      }
    } else if (selection_start == -1) {
      if (selected_index_count != selected_sample_count) {
        return Error::InvalidArgument("indexed selection must provide one source index per selected sample");
      }
    } else {
      return Error::InvalidArgument("selection_start must be -1 for indexed selection or nonnegative");
    }

    if (!is_result_vector(raw_dosage_sums, compute_variant_count) ||
        !is_result_vector(raw_dosage_square_sums, compute_variant_count) ||
        !is_result_vector(zero_counts, compute_variant_count) ||
        !is_result_vector(homozygous_alternate_counts, compute_variant_count) ||
        !is_result_vector(statuses, compute_variant_count) ||
        !is_result_vector(genotype_means, compute_variant_count)) {
      return Error::InvalidArgument("packed8 summary outputs must match the compute variant count");
    }
    if (logical_variant_count > std::numeric_limits<std::size_t>::max() / output_stride) {
      return Error::InvalidArgument("decompressed packed8 slab size overflows size_t");
    }

    RuntimeState& runtime = runtime_state();
    const Packed8Kernels& kernels = runtime.kernels();
    const NvcompApi& nvcomp = runtime.nvcomp();
    const NvcompAlignmentRequirements& alignments = nvcomp.alignments();
    const std::size_t compressed_slab_bytes = compressed_slab.element_count();
    if (compressed_slab_bytes == 0) {
      return Error::InvalidArgument("compressed slab must not be empty");
    }
    if (reinterpret_cast<std::uintptr_t>(compressed_slab.typed_data()) % alignments.input != 0) {
      return Error::InvalidArgument("compressed slab base violates nvCOMP DEFLATE input alignment");
    }
    if (alignments.output > 1 && output_stride % alignments.output != 0) {
      return Error::InvalidArgument("packed8 output stride violates nvCOMP alignment");
    }

    const std::size_t decompressed_byte_count = logical_variant_count * output_stride;
    const std::size_t temporary_bytes =
        nvcomp.temporary_size(logical_variant_count, output_stride, decompressed_byte_count);

    WorkspaceLayout workspace_layout;
    const std::size_t fallback_input_offset = workspace_layout.append_bytes(5, alignments.input);
    const std::size_t decompressed_slab_offset =
        workspace_layout.append_bytes(decompressed_byte_count, alignments.output);
    const std::size_t input_pointers_offset = workspace_layout.append_array<const void*>(logical_variant_count);
    const std::size_t input_sizes_offset = workspace_layout.append_array<std::size_t>(logical_variant_count);
    const std::size_t output_pointers_offset = workspace_layout.append_array<void*>(logical_variant_count);
    const std::size_t output_capacities_offset = workspace_layout.append_array<std::size_t>(logical_variant_count);
    const std::size_t actual_sizes_offset = workspace_layout.append_array<std::size_t>(logical_variant_count);
    const std::size_t nvcomp_statuses_offset = workspace_layout.append_array<NvcompStatus>(logical_variant_count);
    const std::size_t descriptor_statuses_offset = workspace_layout.append_array<std::uint32_t>(logical_variant_count);
    const std::size_t temporary_offset = workspace_layout.append_bytes(temporary_bytes, alignments.temporary);

    std::optional<void*> workspace_allocation =
        scratch.Allocate(workspace_layout.byte_count(), workspace_layout.alignment());
    if (!workspace_allocation.has_value()) {
      fail_runtime("XLA packed8 workspace allocation failed");
    }
    auto* workspace = static_cast<std::uint8_t*>(*workspace_allocation);
    std::uint8_t* fallback_input = workspace + fallback_input_offset;
    std::uint8_t* decompressed_slab = workspace + decompressed_slab_offset;
    const void** input_pointers = reinterpret_cast<const void**>(workspace + input_pointers_offset);
    std::size_t* input_sizes = reinterpret_cast<std::size_t*>(workspace + input_sizes_offset);
    void** output_pointers = reinterpret_cast<void**>(workspace + output_pointers_offset);
    std::size_t* output_capacities = reinterpret_cast<std::size_t*>(workspace + output_capacities_offset);
    std::size_t* actual_sizes = reinterpret_cast<std::size_t*>(workspace + actual_sizes_offset);
    NvcompStatus* nvcomp_statuses = reinterpret_cast<NvcompStatus*>(workspace + nvcomp_statuses_offset);
    std::uint32_t* descriptor_statuses = reinterpret_cast<std::uint32_t*>(workspace + descriptor_statuses_offset);
    void* temporary = temporary_bytes == 0 ? nullptr : workspace + temporary_offset;

    kernels.launch_descriptors(compressed_slab.typed_data(),
                               compressed_slab_bytes,
                               compressed_metadata.typed_data(),
                               alignments.input,
                               fallback_input,
                               decompressed_slab,
                               output_stride,
                               input_pointers,
                               input_sizes,
                               output_pointers,
                               output_capacities,
                               descriptor_statuses,
                               logical_variant_count,
                               stream);
    nvcomp.decompress(input_pointers,
                      input_sizes,
                      output_capacities,
                      actual_sizes,
                      logical_variant_count,
                      temporary,
                      temporary_bytes,
                      output_pointers,
                      nvcomp_statuses,
                      stream);

    kernels.launch_finalize(decompressed_slab,
                            actual_sizes,
                            nvcomp_statuses,
                            compressed_metadata.typed_data(),
                            descriptor_statuses,
                            selected_sample_indices.typed_data(),
                            selection_start,
                            logical_variant_count,
                            compute_variant_count,
                            source_sample_count,
                            selected_sample_count,
                            output_stride,
                            probabilities->typed_data(),
                            raw_dosage_sums->typed_data(),
                            raw_dosage_square_sums->typed_data(),
                            zero_counts->typed_data(),
                            homozygous_alternate_counts->typed_data(),
                            statuses->typed_data(),
                            genotype_means->typed_data(),
                            stream);
    return Error::Success();
  } catch (const HandlerFailure& failure) {
    return Error(failure.code(), failure.what());
  } catch (const std::exception& exception) {
    return Error(ErrorCode::kInternal, exception.what());
  } catch (...) {
    return Error(ErrorCode::kInternal, "unknown native packed8 handler failure");
  }
}

}  // namespace

extern "C" std::int32_t g_genotype_cuda_initialize_nvcomp_runtime(std::int32_t device_ordinal,
                                                                  std::size_t member_alignment,
                                                                  GGenotypeCudaCapability* capability,
                                                                  const char** detail) noexcept {
  thread_local std::string initialization_detail;
  if (detail != nullptr) {
    *detail = nullptr;
  }
  if (capability == nullptr || detail == nullptr) {
    return static_cast<std::int32_t>(InitializationStatus::kInternal);
  }
  *capability = GGenotypeCudaCapability{};
  try {
    runtime_state().validate(device_ordinal, member_alignment, *capability);
    return static_cast<std::int32_t>(InitializationStatus::kSuccess);
  } catch (const InitializationFailure& failure) {
    initialization_detail = failure.what();
    *detail = initialization_detail.c_str();
    return static_cast<std::int32_t>(failure.status());
  } catch (const std::exception& exception) {
    initialization_detail = exception.what();
    *detail = initialization_detail.c_str();
    return static_cast<std::int32_t>(InitializationStatus::kInternal);
  } catch (...) {
    initialization_detail = "unknown native CUDA initialization failure";
    *detail = initialization_detail.c_str();
    return static_cast<std::int32_t>(InitializationStatus::kInternal);
  }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(g_nvcomp_decode_packed8_ffi,
                              decode_packed8,
                              xla::ffi::Ffi::Bind()
                                  .Arg<BufferR1<DataType::U8>>()
                                  .Arg<BufferR2<DataType::U32>>()
                                  .Arg<BufferR1<DataType::U32>>()
                                  .Ret<xla::ffi::BufferR3<DataType::U8>>()
                                  .Ret<xla::ffi::BufferR1<DataType::U64>>()
                                  .Ret<xla::ffi::BufferR1<DataType::U64>>()
                                  .Ret<xla::ffi::BufferR1<DataType::U32>>()
                                  .Ret<xla::ffi::BufferR1<DataType::U32>>()
                                  .Ret<xla::ffi::BufferR1<DataType::U32>>()
                                  .Ret<xla::ffi::BufferR1<DataType::F32>>()
                                  .Attr<std::int64_t>("source_sample_count")
                                  .Attr<std::int64_t>("selection_start")
                                  .Ctx<xla::ffi::PlatformStream<CudaStream>>()
                                  .Ctx<ScratchAllocator>());
